"""Tiered retrieval router: question bank first, vector RAG second, else refuse.

The whole system hangs off this ordering. A pre-extracted Q/A pair is a far
stronger signal than a chunk that merely embeds nearby, so Tier 1 is tried on
its own terms and only a genuine miss falls through to Tier 2. Tier 3 refuses
rather than letting the model improvise from weak context.

Two sharp edges:

* A high embedding score is not the same as an answer. Tier 1 therefore runs an
  LLM *relevance gate* over its candidates when :data:`config.ENABLE_LLM_RERANK`
  is set. If the gate call itself fails we keep the embedding matches — a flaky
  model must never silently downgrade a route.
* Every decision is appended to ``notes`` in order. Those notes are the
  user-facing explanation of a refusal, so they carry concrete numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import config
from rag.ollama_client import OllamaClient, get_client
from rag.schemas import (
    ROUTE_HYBRID,
    ROUTE_QUESTION_BANK,
    ROUTE_REFUSED,
    ROUTE_VECTOR,
    ScoredChunk,
    ScoredQA,
)
from rag.text_normalize import tokenize

if TYPE_CHECKING:  # imported for typing only, so retrieval stays importable alone
    from rag.vectorstore import VectorStore


#: Reciprocal Rank Fusion constant. 60 is the value from the original TREC
#: work and is deliberately large: it flattens the head of each ranking so a
#: chunk needs agreement across phrasings, not one lucky first place.
RRF_K = 60

#: A sibling answer must be within QUESTION_MULTI_MARGIN of the best hit *and*
#: clear this fraction of the accept threshold. Without the floor, a weak best
#: score would drag in even weaker company.
MULTI_FLOOR_RATIO = 0.85

#: Word-overlap above which two answers are treated as the same fact extracted
#: from two overlapping chunks.
ANSWER_DUP_JACCARD = 0.9

#: How much the relevance gate's own confidence moves the final number. The
#: embedding score stays dominant; the gate is a veto, not a scorer.
GATE_CONFIDENCE_WEIGHT = 0.3

_MAX_EXPANSIONS = 3

_GATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keep": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number"},
    },
    "required": ["keep", "confidence"],
}

_EXPAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["queries"],
}

_DECOMPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "parts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["parts"],
}

_CHUNK_RERANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["order"],
}

@dataclass
class RetrievalOutcome:
    """What the router decided, plus the trail of how it got there."""

    query: str
    route: str
    qa_matches: list[ScoredQA] = field(default_factory=list)
    chunk_matches: list[ScoredChunk] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _words(text: str) -> set[str]:
    return set(tokenize(text))


def _near_identical(a: str, b: str) -> bool:
    """Cheap answer-level dedupe: exact after normalisation, or near-total overlap."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return wa == wb
    if wa == wb:
        return True
    overlap = len(wa & wb) / len(wa | wb)
    return overlap >= ANSWER_DUP_JACCARD


def _dedupe_answers(matches: list[ScoredQA]) -> tuple[list[ScoredQA], int]:
    """Keep the highest-scoring representative of each distinct answer."""
    kept: list[ScoredQA] = []
    dropped = 0
    for candidate in sorted(matches, key=lambda m: m.score, reverse=True):
        if any(_near_identical(candidate.qa.answer, k.qa.answer) for k in kept):
            dropped += 1
            continue
        kept.append(candidate)
    return kept, dropped


def expand_query(query: str, client: OllamaClient | None = None) -> list[str]:
    """Return the original query followed by up to three alternative phrasings.

    The original is always first and always present, so callers can use the
    result unconditionally: a failed or empty expansion simply degrades to a
    single-element list.
    """
    query = query.strip()
    if not query:
        return []

    client = client or get_client()
    prompt = (
        "Rewrite this search query into 2-3 alternative phrasings that would "
        "retrieve the same information from a technical document collection. "
        "Vary the terminology: use likely synonyms, domain jargon, and expanded "
        "acronyms. Keep each rewrite a standalone query.\n\n"
        f"Query: {query}\n\n"
        'Reply as {"queries": ["...", "..."]}.'
    )

    try:
        data = client.chat_json(prompt, schema=_EXPAND_SCHEMA, temperature=0.3)
    except Exception:  # expansion is an optimisation; never fail the search for it
        return [query]

    raw = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return [query]

    variants = [query]
    seen = {query.lower()}
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        variants.append(text)
        if len(variants) > _MAX_EXPANSIONS:
            break
    return variants


def _run_relevance_gate(
    query: str,
    candidates: list[ScoredQA],
    client: OllamaClient,
) -> tuple[list[ScoredQA] | None, float, str]:
    """Ask the LLM which candidates actually address the query.

    Returns ``(kept, confidence, note)``. ``kept`` is ``None`` when the call
    failed — distinct from an empty list, which is a real "none of these".
    """
    listing = "\n".join(
        f"{i}. {m.qa.question}" for i, m in enumerate(candidates)
    )
    prompt = (
        "You are a strict relevance gate for a document question-answering "
        "system. Below is a user's question and a numbered list of candidate "
        "questions that were pre-extracted from the documents.\n\n"
        "Keep only the candidates that genuinely ask for the same information "
        "as the user's question, such that their answer would answer the user. "
        "Being on the same topic is NOT enough. If none qualify, return an "
        "empty list.\n\n"
        f"User question: {query}\n\n"
        f"Candidates:\n{listing}\n\n"
        'Reply as {"keep": [indices], "confidence": 0.0-1.0} where confidence '
        "is how certain you are that the kept candidates answer the user."
    )

    try:
        data = client.chat_json(prompt, schema=_GATE_SCHEMA, temperature=0.0)
    except Exception as exc:  # a broken gate must not demote a good tier-1 hit
        return None, 0.0, f"relevance gate failed ({exc}); kept embedding matches"

    if not isinstance(data, dict):
        return None, 0.0, "relevance gate returned an unexpected shape; kept embedding matches"

    indices: list[int] = []
    for value in data.get("keep") or []:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(candidates) and idx not in indices:
            indices.append(idx)

    try:
        gate_confidence = _clamp(float(data.get("confidence", 0.0)))
    except (TypeError, ValueError):
        gate_confidence = 0.0

    kept = [candidates[i] for i in indices]
    note = (
        f"relevance gate kept {len(kept)}/{len(candidates)} candidates "
        f"(gate confidence {gate_confidence:.2f})"
    )
    return kept, gate_confidence, note


def _fuse(ranked_lists: list[list[ScoredChunk]]) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion across per-phrasing rankings.

    Each chunk keeps its *best raw similarity* so threshold comparisons stay in
    cosine space — RRF scores are ordinal only and must never be thresholded.
    """
    rrf: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}

    for ranking in ranked_lists:
        for rank, scored in enumerate(ranking):
            key = scored.chunk.chunk_id
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            if key not in best or scored.score > best[key].score:
                best[key] = scored

    return [best[k] for k in sorted(rrf, key=lambda k: (rrf[k], best[k].score), reverse=True)]


def decompose_query(query: str, client: OllamaClient | None = None) -> list[str]:
    """Split a multi-part question into standalone sub-queries (or [query])."""
    if not config.ENABLE_QUERY_DECOMPOSITION:
        return [query]
    query = query.strip()
    if not query:
        return [query]
    # Skip the LLM when the query is clearly a single short ask.
    multi_markers = (" and ", " و ", "؟", "?", ";", "؛")
    if len(query) < 80 and not any(m in query.lower() for m in multi_markers if m != "؟"):
        if "؟" not in query and "?" not in query:
            return [query]

    client = client or get_client()
    prompt = (
        "Split the user question into 1-3 atomic sub-questions that can be "
        "searched independently in a document collection. If it is already a "
        "single question, return it alone.\n\n"
        f"Question: {query}\n\n"
        'Reply as {"parts": ["...", "..."]}.'
    )
    try:
        data = client.chat_json(prompt, schema=_DECOMPOSE_SCHEMA, temperature=0.0)
    except Exception:
        return [query]
    raw = data.get("parts") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return [query]
    parts: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        text = item.strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        parts.append(text)
        if len(parts) >= 3:
            break
    return parts or [query]


def rerank_chunks(
    query: str,
    chunks: list[ScoredChunk],
    client: OllamaClient | None = None,
) -> list[ScoredChunk]:
    """LLM listwise rerank of chunk candidates (local, no cross-encoder dep)."""
    if not config.ENABLE_CHUNK_RERANK or len(chunks) <= 1:
        return chunks
    client = client or get_client()
    listing = "\n".join(
        f"{i}. {m.chunk.doc_name} {m.chunk.page_label}: {m.chunk.text[:400]}"
        for i, m in enumerate(chunks)
    )
    prompt = (
        "Reorder these passages by how well they answer the user question. "
        "Return the indices of the useful passages only, best first. Drop "
        "irrelevant ones.\n\n"
        f"Question: {query}\n\nPassages:\n{listing}\n\n"
        'Reply as {"order": [indices]}.'
    )
    try:
        data = client.chat_json(prompt, schema=_CHUNK_RERANK_SCHEMA, temperature=0.0)
    except Exception:
        return chunks
    order = data.get("order") if isinstance(data, dict) else None
    if not isinstance(order, list) or not order:
        return chunks
    kept: list[ScoredChunk] = []
    seen: set[int] = set()
    for value in order:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(chunks) and idx not in seen:
            seen.add(idx)
            kept.append(chunks[idx])
    return kept or chunks


def _adaptive_threshold(base: float, scores: list[float]) -> float:
    """Never raise the bar; optionally ease it slightly from the score head."""
    if not config.ENABLE_ADAPTIVE_THRESHOLDS or not scores:
        return base
    top = sorted(scores, reverse=True)[:5]
    if not top:
        return base
    soft = max(top) * 0.97
    floor = base * config.ADAPTIVE_THRESHOLD_FLOOR_RATIO
    # Only ease when the best score is near the bar (borderline corpus).
    if max(top) >= base:
        return base
    return max(floor, min(base, soft))


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------


def route_query(
    query: str,
    store: "VectorStore",
    client: OllamaClient | None = None,
) -> RetrievalOutcome:
    """Route one user query through the question bank, then vector RAG, then refusal."""
    notes: list[str] = []
    clean = (query or "").strip()
    if not clean:
        return RetrievalOutcome(
            query=query or "",
            route=ROUTE_REFUSED,
            qa_matches=[],
            chunk_matches=[],
            confidence=0.0,
            notes=["empty query — nothing to search"],
        )

    client = client or get_client()
    best_seen = 0.0

    # -- Tier 1: question bank --------------------------------------------
    try:
        qa_hits = store.search_questions(clean, top_k=config.QUESTION_TOP_K)
    except Exception as exc:  # store/model flakiness must not sink the request
        qa_hits = []
        notes.append(f"question-bank search failed ({exc}); skipping tier 1")

    if qa_hits:
        qa_hits = sorted(qa_hits, key=lambda m: m.score, reverse=True)
        best_qa = qa_hits[0].score
        best_seen = max(best_seen, best_qa)
        notes.append(
            f"question bank: {len(qa_hits)} candidates, best {best_qa:.2f} "
            f"(threshold {config.QUESTION_MATCH_THRESHOLD:.2f})"
        )
    else:
        best_qa = 0.0
        notes.append("question bank: no candidates")

    if qa_hits and best_qa >= config.QUESTION_MATCH_THRESHOLD:
        floor = max(
            best_qa - config.QUESTION_MULTI_MARGIN,
            config.QUESTION_MATCH_THRESHOLD * MULTI_FLOOR_RATIO,
        )
        accepted = [m for m in qa_hits if m.score >= floor]
        notes.append(
            f"kept {len(accepted)} hits within {config.QUESTION_MULTI_MARGIN:.2f} "
            f"of the best (score floor {floor:.2f})"
        )

        accepted, dropped = _dedupe_answers(accepted)
        if dropped:
            notes.append(f"dropped {dropped} near-duplicate answer(s)")

        gate_confidence: float | None = None
        if config.ENABLE_LLM_RERANK and accepted:
            gated, gate_confidence, gate_note = _run_relevance_gate(
                clean, accepted, client
            )
            notes.append(gate_note)
            if gated is None:
                gate_confidence = None  # degraded: no gate signal to blend
            elif gated:
                accepted = gated
            else:
                accepted = []
                notes.append("gate rejected every question-bank hit — falling through to vector RAG")

        if not accepted:
            tier1_accepted = []
            confidence = 0.0
        else:
            tier1_accepted = accepted
            best_accepted = max(m.score for m in accepted)
            confidence = best_accepted
            if gate_confidence is not None:
                confidence = (
                    (1.0 - GATE_CONFIDENCE_WEIGHT) * best_accepted
                    + GATE_CONFIDENCE_WEIGHT * gate_confidence
                )
            confidence = _clamp(confidence)

            # Strong Tier-1 hit → answer from the bank alone.
            # Weak-but-accepted → optionally merge Tier-2 chunks (hybrid).
            want_hybrid = (
                config.ENABLE_HYBRID_ROUTE
                and confidence < config.HYBRID_TIER1_CEILING
            )
            if not want_hybrid:
                notes.append(f"route: question bank ({len(accepted)} answer(s))")
                return RetrievalOutcome(
                    query=clean,
                    route=ROUTE_QUESTION_BANK,
                    qa_matches=accepted,
                    chunk_matches=[],
                    confidence=confidence,
                    notes=notes,
                )
            notes.append(
                f"question bank accepted ({len(accepted)} answer(s), "
                f"confidence {confidence:.2f} < hybrid ceiling "
                f"{config.HYBRID_TIER1_CEILING:.2f}) — also searching chunks"
            )
    else:
        tier1_accepted = []
        confidence = 0.0
        if qa_hits:
            notes.append(
                f"question bank missed by {config.QUESTION_MATCH_THRESHOLD - best_qa:.2f}"
            )

    # -- Tier 2: classical vector RAG --------------------------------------
    parts = decompose_query(clean, client)
    if len(parts) > 1:
        notes.append(f"decomposed query into {len(parts)} sub-question(s)")

    phrasings: list[str] = []
    for part in parts:
        phrasings.extend(expand_query(part, client))
    # Deduplicate phrasings while preserving order.
    deduped: list[str] = []
    seen_p: set[str] = set()
    for p in phrasings:
        key = p.casefold()
        if key in seen_p:
            continue
        seen_p.add(key)
        deduped.append(p)
    phrasings = deduped or [clean]

    if len(phrasings) > 1:
        notes.append(f"expanded into {len(phrasings)} search phrasing(s)")
    else:
        notes.append("query expansion unavailable; searching the original query only")

    rankings: list[list[ScoredChunk]] = []
    for phrasing in phrasings:
        try:
            rankings.append(store.search_chunks(phrasing, top_k=config.VECTOR_TOP_K))
        except Exception as exc:
            notes.append(f"chunk search failed for a phrasing ({exc})")

    fused = _fuse(rankings)
    if fused:
        before = len(fused)
        fused = rerank_chunks(clean, fused[: max(config.VECTOR_TOP_K, 8)], client)
        if len(fused) != before:
            notes.append(f"chunk rerank kept {len(fused)}/{before} passages")
        best_chunk = max(m.score for m in fused)
        best_seen = max(best_seen, best_chunk)
        vec_threshold = _adaptive_threshold(
            config.VECTOR_MATCH_THRESHOLD, [m.score for m in fused]
        )
        if vec_threshold < config.VECTOR_MATCH_THRESHOLD - 1e-9:
            notes.append(
                f"adaptive vector threshold {vec_threshold:.2f} "
                f"(base {config.VECTOR_MATCH_THRESHOLD:.2f})"
            )
        notes.append(
            f"vector RAG: {len(fused)} fused chunks, best similarity "
            f"{best_chunk:.2f} (threshold {vec_threshold:.2f})"
        )
    else:
        best_chunk = 0.0
        vec_threshold = config.VECTOR_MATCH_THRESHOLD
        notes.append("vector RAG: no chunks retrieved")

    accepted_chunks = (
        [m for m in fused if m.score >= vec_threshold]
        if fused and best_chunk >= vec_threshold
        else []
    )

    if tier1_accepted and accepted_chunks:
        notes.append(
            f"route: hybrid ({len(tier1_accepted)} QA + {len(accepted_chunks)} chunk(s))"
        )
        return RetrievalOutcome(
            query=clean,
            route=ROUTE_HYBRID,
            qa_matches=tier1_accepted,
            chunk_matches=accepted_chunks,
            confidence=max(confidence, _clamp(best_chunk)),
            notes=notes,
        )

    if tier1_accepted:
        notes.append(
            f"route: question bank ({len(tier1_accepted)} answer(s); "
            "no usable Tier-2 chunks to merge)"
        )
        return RetrievalOutcome(
            query=clean,
            route=ROUTE_QUESTION_BANK,
            qa_matches=tier1_accepted,
            chunk_matches=[],
            confidence=confidence,
            notes=notes,
        )

    if accepted_chunks:
        notes.append(f"route: vector RAG ({len(accepted_chunks)} chunk(s))")
        return RetrievalOutcome(
            query=clean,
            route=ROUTE_VECTOR,
            qa_matches=[],
            chunk_matches=accepted_chunks,
            confidence=_clamp(best_chunk),
            notes=notes,
        )

    # -- Tier 3: refuse (keep near-misses for the UI) ---------------------
    near_qa = (qa_hits or [])[:3]
    near_chunks = (fused or [])[:3]
    notes.append(
        f"best question-bank match scored {best_qa:.2f} "
        f"(threshold {config.QUESTION_MATCH_THRESHOLD:.2f}); "
        f"best chunk match scored {best_chunk:.2f} "
        f"(threshold {vec_threshold:.2f}) — refusing"
    )
    for m in near_qa:
        notes.append(
            f"near-miss question ({m.score:.2f}): {m.qa.question[:160]}"
        )
    for m in near_chunks:
        notes.append(
            f"near-miss passage ({m.score:.2f}): "
            f"{m.chunk.doc_name} {m.chunk.page_label}"
        )
    return RetrievalOutcome(
        query=clean,
        route=ROUTE_REFUSED,
        qa_matches=near_qa,
        chunk_matches=near_chunks,
        confidence=_clamp(best_seen),
        notes=notes,
    )


__all__ = [
    "RetrievalOutcome",
    "route_query",
    "expand_query",
    "decompose_query",
    "rerank_chunks",
]
