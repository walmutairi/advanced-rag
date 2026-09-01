"""Grounded answer synthesis, and the guard that keeps it honest.

The answerer turns a :class:`RetrievalOutcome` into an :class:`AnswerResult`.
Three things here are load-bearing rather than cosmetic:

* **The refusal path never calls the model.** If retrieval refused, so do we.
  There is no prompt in this module that a "helpful" model could answer from
  memory when the documents are silent — the strict system prompt makes the
  sentinel ``INSUFFICIENT_CONTEXT`` the only legal escape hatch.
* **Citations are re-derived from the generated text, not from what we sent.**
  The model is handed sources ``[1..n]`` but typically cites a subset, and
  occasionally invents an ``[n]`` that does not exist. After generation the
  markers are parsed, unknown ones stripped, and the survivors renumbered
  contiguously with the answer body rewritten to match, so the numbers under
  the answer always line up with the numbers inside it.
* **The groundedness check fails CLOSED.** A partly fabricated answer is worse
  than no answer, so a draft scoring below the threshold is discarded — and so
  is one whose check could not be run at all. Treating a failed check as a mere
  infrastructure hiccup and publishing anyway was the most reachable way to
  break the system's core promise: the auditor is a 31B call under the same
  timeout as everything else, so an ordinary outage silently removed the last
  guard. Refusing is recoverable; a confident fabrication is not.

The audit is also non-circular by construction. On the question-bank route the
stored ``answer`` was itself generated at extraction time, so auditing against
it would only confirm the model agrees with the model. Only the verbatim
evidence quotes are treated as ground truth — see ``_Source.audit_body``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import config
from rag.ollama_client import OllamaClient, OllamaError, get_client
from rag.retrieval import RetrievalOutcome

if TYPE_CHECKING:  # avoid a circular import at runtime
    from rag.vectorstore import VectorStore
from rag.schemas import (
    ROUTE_HYBRID,
    ROUTE_QUESTION_BANK,
    ROUTE_REFUSED,
    ROUTE_VECTOR,
    AnswerResult,
    Citation,
    QAPair,
    ScoredChunk,
    ScoredQA,
)


def _should_skip_groundedness(outcome: RetrievalOutcome) -> bool:
    """Strong question-bank-only hits already carry validated evidence quotes.

    Hybrid answers still run the audit — they blend free-form passages that
    were never evidence-checked at extract time.
    """
    if not config.SKIP_GROUNDEDNESS_ON_QUESTION_BANK:
        return False
    if outcome.route != ROUTE_QUESTION_BANK:
        return False
    if not outcome.qa_matches:
        return False
    return outcome.confidence >= config.QUESTION_BANK_SKIP_GROUNDEDNESS_MIN

#: Sentinel the model must emit when the sources do not answer the question.
#: Kept unmistakable and unlikely to occur inside a real answer.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"

#: Length of the excerpt stored on a citation for display.
QUOTE_CHARS = 400

#: gemma4 has a 262k window, so widening past the configured default is free
#: insurance. Silent truncation would drop sources mid-prompt, which is the
#: exact condition that makes a model start inventing.
MAX_CTX = 65536


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""You are a careful research assistant. You answer only from a numbered set of source excerpts supplied with each question.

Rules, in priority order:

1. Answer ONLY from the numbered sources. Outside knowledge is forbidden, even when you are certain it is correct, and even when the sources look incomplete or outdated. What you know is irrelevant here; only what the sources say counts.
2. If the sources do not contain the answer, reply with exactly {INSUFFICIENT} and nothing else — no apology, no explanation, no partial answer.
3. Never speculate, never extrapolate beyond what is stated, and never fill a gap with plausible detail. A missing fact is a refusal, not a guess.
4. Every factual sentence carries at least one citation marker of the form [n] identifying the source it came from. Cite only numbers that appear in the source list. Never invent a number.
5. Write in the same language as the question.
6. Answer the question directly. Do not review, rate or comment on the sources themselves, and do not mention that you were given excerpts."""

_INTEGRATE_INSTRUCTION = """Several sources matched this question. Integrate them into one coherent answer instead of answering from a single one.

Where they disagree, give different values, or describe different cases, say so explicitly and attribute each side to its source — for example: "the documents give two distinct figures: X [1] and Y [2]". Never silently pick one and drop the other."""

_GROUNDEDNESS_SYSTEM = """You are a strict factuality auditor. You are given source material and an answer that claims to be derived from it.

Break the answer into atomic factual claims — one verifiable assertion each. Ignore greetings, transitions, and pure restatements of the question; do not list those as claims.

Mark a claim SUPPORTED only if the source material states it directly or entails it unambiguously. Mark it UNSUPPORTED if it goes beyond the sources, adds detail they do not contain, or is merely consistent with general knowledge. Being true in the real world is not support. Judge only what the sources say."""

_GROUNDEDNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED"]},
                },
                "required": ["claim", "verdict"],
            },
        }
    },
    "required": ["claims"],
}


# --------------------------------------------------------------------------
# Internal source representation
# --------------------------------------------------------------------------


@dataclass
class _Source:
    """One numbered block as shown to the model, plus what a citation needs."""

    index: int
    doc_name: str
    page_label: str
    section: str
    body: str  # model-facing text
    quote: str  # human-facing excerpt
    origin: str
    #: Ground truth the groundedness audit is allowed to check against. It must
    #: never include a previously GENERATED answer, only text copied from the
    #: document — otherwise the audit is circular and a fabrication can
    #: self-certify at 1.0 by citing itself. ``None`` means "body is already
    #: document truth" (the vector route, where body is the raw chunk); an
    #: empty string means "this source contributes NO verifiable ground truth"
    #: and must stay empty rather than falling back to body.
    audit_body: str | None = None

    def evidence_for_audit(self) -> str:
        return self.body if self.audit_body is None else self.audit_body

    def render(self) -> str:
        header = f"[{self.index}] {self.doc_name or 'document'}, {self.page_label}"
        if self.section:
            header += f" — {self.section}"
        return f"{header}\n{self.body}"


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _excerpt(text: str, limit: int = QUOTE_CHARS) -> str:
    """Trim to ``limit`` chars on a word boundary."""
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.") + "…"


def _best_evidence(qa: QAPair) -> str:
    """The evidence quote that most overlaps the answer it supports."""
    quotes = [q.strip() for q in qa.evidence if q and q.strip()]
    if not quotes:
        return qa.answer
    answer_words = _words(qa.answer)
    return max(quotes, key=lambda q: (len(answer_words & _words(q)), len(q)))


def _qa_sources(
    matches: list[ScoredQA], chunk_texts: dict[str, str] | None = None
) -> list[_Source]:
    chunk_texts = chunk_texts or {}
    sources: list[_Source] = []
    seen: set[str] = set()

    for match in matches:
        qa = match.qa
        # A canonical question and one of its own paraphrases can both hit, so
        # the same pair may arrive twice; the first copy carries the top score.
        key = qa.qa_id or f"{qa.question}|{qa.answer}"
        if key in seen:
            continue
        seen.add(key)

        evidence = [q.strip() for q in qa.evidence if q and q.strip()]
        body = f"Question: {qa.question}\nAnswer: {qa.answer}"
        if evidence:
            quoted = "\n".join(f'  - "{q}"' for q in evidence)
            body += f"\nVerbatim evidence from the document:\n{quoted}"

        # A pair with no source chunk was authored by hand, not mined from a
        # document (every extracted pair carries the chunk_id it came from).
        # Its answer is therefore human ground truth, not a model generation,
        # so the usual bar on auditing against qa.answer — which exists only to
        # stop a generated answer self-certifying — does not apply. Without
        # this, a manual pair has NO checkable ground truth, every claim reads
        # as unsupported, and a perfect question-bank match is refused anyway.
        if qa.chunk_id:
            # Extracted: document truth only (quotes + the source chunk). A
            # quote is a fragment and can omit the context that makes a claim
            # checkable, so the surrounding chunk is included alongside it.
            audit_body = "\n".join(
                part for part in (*evidence, chunk_texts.get(qa.chunk_id, "")) if part
            )
        else:
            # Hand-authored: trust the human's own answer and any evidence they
            # attached as the ground truth to audit against.
            audit_body = "\n".join(part for part in (qa.answer, *evidence) if part)

        sources.append(
            _Source(
                index=len(sources) + 1,
                doc_name=qa.doc_name,
                page_label=qa.page_label,
                section=qa.section,
                body=body,
                quote=_excerpt(_best_evidence(qa)),
                origin="question_bank",
                audit_body=audit_body,
            )
        )
    return sources


def _chunk_sources(matches: list[ScoredChunk]) -> list[_Source]:
    sources: list[_Source] = []
    seen: set[str] = set()

    for match in matches:
        chunk = match.chunk
        key = chunk.chunk_id or chunk.text[:200]
        if key in seen:
            continue
        seen.add(key)

        sources.append(
            _Source(
                index=len(sources) + 1,
                doc_name=chunk.doc_name,
                page_label=chunk.page_label,
                section=chunk.section,
                body=chunk.text.strip(),
                quote=_excerpt(chunk.text),
                origin="vector",
            )
        )
    return sources


# --------------------------------------------------------------------------
# Citation parsing and renumbering
# --------------------------------------------------------------------------

#: Separators include the Arabic comma (U+060C) and semicolon (U+061B):
#: the answering prompt writes in the document's language, so a
#: multi-source marker in an Arabic answer arrives as "[1\u060c 2]". An
#: ASCII-only class left those markers unmatched, so renumbering rewrote
#: everything around them and they ended up pointing at the wrong source.
_SEPARATORS = r",;\u060C\u061B"
_MARKER_RE = re.compile(rf"\[\s*(\d+(?:\s*[{_SEPARATORS}]\s*\d+)*)\s*\]")

#: Any bracketed-digit token still present after remapping is a form we do
#: not understand (a range like "[1-3]", say). Leaving it would dangle or
#: mis-point, so it is deleted rather than shown.
_STRAY_MARKER_RE = re.compile(r"\[\s*\d+(?:\s*[^\]\d]{0,3}\s*\d+)*\s*\]")
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")
_SPACE_PUNCT_RE = re.compile(r"[ \t]+([.,;:!?)\]])")


def _remap_citations(
    answer: str, sources: list[_Source]
) -> tuple[str, list[Citation]]:
    """Keep only the sources the answer actually cites, renumbered from 1.

    Markers pointing at a source that was never supplied are deleted outright
    rather than left dangling — a number with nothing behind it reads as
    evidence to a user skimming the answer.
    """
    valid = {s.index for s in sources}

    # Drop bracketed-digit tokens we cannot parse (ranges like "[1-3]", exotic
    # separators) BEFORE renumbering. Left in place they would survive the
    # rewrite still pointing at the OLD numbering and mis-attribute a claim.
    # This must run first: afterwards the text contains freshly written "[n]"
    # markers that are indistinguishable from unparsed ones by shape alone.
    answer = _STRAY_MARKER_RE.sub(
        lambda m: m.group(0) if _MARKER_RE.fullmatch(m.group(0)) else "", answer
    )

    order: list[int] = []
    for group in _MARKER_RE.findall(answer):
        for raw in re.split(rf"[{_SEPARATORS}]", group):
            number = int(raw)
            if number in valid and number not in order:
                order.append(number)

    remap = {old: new for new, old in enumerate(order, start=1)}

    def _rewrite(match: re.Match[str]) -> str:
        kept = []
        for raw in re.split(rf"[{_SEPARATORS}]", match.group(1)):
            number = remap.get(int(raw))
            if number is not None and number not in kept:
                kept.append(number)
        return "".join(f"[{n}]" for n in kept)

    text = _MARKER_RE.sub(_rewrite, answer)
    text = _SPACE_RUN_RE.sub(" ", text)
    text = _SPACE_PUNCT_RE.sub(r"\1", text).strip()

    by_index = {s.index: s for s in sources}
    citations = [
        Citation(
            index=new,
            doc_name=by_index[old].doc_name,
            page_label=by_index[old].page_label,
            quote=by_index[old].quote,
            section=by_index[old].section,
            origin=by_index[old].origin,
        )
        for old, new in remap.items()
    ]
    citations.sort(key=lambda c: c.index)
    return text, citations


def _is_insufficient(answer: str) -> bool:
    """True when the reply is the sentinel, allowing for light decoration."""
    if not answer or not answer.strip():
        return True
    stripped = answer.strip().strip("`*_\"'“”. \n")
    if stripped.upper() == INSUFFICIENT:
        return True
    if INSUFFICIENT not in answer.upper():
        return False
    # The model sometimes wraps the sentinel in a short apology. Treat it as a
    # refusal unless there is enough surrounding text to be a real answer.
    remainder = re.sub(INSUFFICIENT, "", answer, flags=re.IGNORECASE)
    return len(re.sub(r"[\W_]+", "", remainder)) <= 60


def _strip_sentinel(answer: str) -> tuple[str, bool]:
    """Remove a surviving sentinel from a partially-answerable reply.

    When a question has several parts and the sources cover only some, the
    model answers what it can and appends INSUFFICIENT_CONTEXT for the rest.
    That reply is a genuine answer, so ``_is_insufficient`` correctly lets it
    through — but the raw token is an internal protocol string and must never
    reach the UI. Returns the cleaned text and whether anything was removed.
    """
    if INSUFFICIENT not in answer.upper():
        return answer, False

    # Take the whole line the sentinel sits on: it usually carries a
    # parenthetical naming the unanswered part, which is meaningless alone.
    cleaned = re.sub(
        rf"^.*{INSUFFICIENT}.*$", "", answer, flags=re.IGNORECASE | re.MULTILINE
    )
    cleaned = re.sub(INSUFFICIENT, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, True


def _context_window(*parts: str) -> int:
    chars = sum(len(p) for p in parts)
    needed = int(chars / 3.0) + 1536  # ~3 chars/token is a safe floor for mixed scripts
    window = config.LLM_NUM_CTX
    while window < needed and window < MAX_CTX:
        window *= 2
    return min(window, MAX_CTX)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def build_refusal(outcome: RetrievalOutcome) -> AnswerResult:
    """The one and only way this system declines. Makes no model call.

    Near-miss scores ride along in the notes so the UI can tell the user how
    close their query came instead of showing a bare dead end.
    """
    notes = list(outcome.notes)
    near_misses: list[str] = []

    if outcome.qa_matches:
        best = max(m.score for m in outcome.qa_matches)
        notes.append(
            f"Closest question-bank match scored {best:.2f} "
            f"(accepted at ≥ {config.QUESTION_MATCH_THRESHOLD:.2f})."
        )
        for m in outcome.qa_matches[:3]:
            near_misses.append(
                f"Q ({m.score:.2f}): {m.qa.question} — {m.qa.citation()}"
            )
    if outcome.chunk_matches:
        best = max(m.score for m in outcome.chunk_matches)
        notes.append(
            f"Closest passage scored {best:.2f} "
            f"(accepted at ≥ {config.VECTOR_MATCH_THRESHOLD:.2f})."
        )
        for m in outcome.chunk_matches[:3]:
            near_misses.append(
                f"Passage ({m.score:.2f}): {m.chunk.doc_name}, "
                f"{m.chunk.page_label}"
            )
    if not outcome.qa_matches and not outcome.chunk_matches:
        notes.append("Nothing in the indexed documents came close to this query.")

    return AnswerResult(
        query=outcome.query,
        answer=config.REFUSAL_MESSAGE,
        route=ROUTE_REFUSED,
        answered=False,
        citations=[],
        matched_questions=list(outcome.qa_matches),
        matched_chunks=list(outcome.chunk_matches),
        confidence=outcome.confidence,
        groundedness=None,
        uncertainty=1.0,
        near_misses=near_misses,
        notes=notes,
        elapsed_seconds=0.0,
    )


def _uncertainty(confidence: float, groundedness: float | None, answered: bool) -> float:
    if not answered:
        return 1.0
    g = 1.0 if groundedness is None else float(groundedness)
    return _clamp(1.0 - (0.6 * confidence + 0.4 * g))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _flag_uncited_sentences(answer: str) -> list[str]:
    """Return sentences that look like claims but lack [n] markers."""
    if not config.ENABLE_CLAIM_CITATIONS or not answer.strip():
        return []
    parts = re.split(r"(?<=[.!?۔؟])\s+", answer.strip())
    missing: list[str] = []
    for part in parts:
        text = part.strip()
        if len(text) < 40:
            continue
        if re.search(r"\[\d+\]", text):
            continue
        missing.append(text[:160])
    return missing


def check_groundedness(
    answer: str,
    contexts: list[str],
    client: OllamaClient | None = None,
) -> tuple[float, list[str]]:
    """Score how much of ``answer`` the ``contexts`` actually support.

    Returns ``(supported / total, unsupported_claims)``. An answer with no
    checkable claims scores 1.0 — there is nothing to fabricate. Raises
    :class:`OllamaError` if the audit call itself is unusable; callers must
    treat that as "unverified", never as "ungrounded".
    """
    if not answer or not answer.strip():
        return 1.0, []

    client = client or get_client()
    joined = "\n\n".join(
        f"--- source {i} ---\n{c.strip()}" for i, c in enumerate(contexts, start=1)
    )
    prompt = (
        f"SOURCE MATERIAL:\n{joined}\n\n"
        f"ANSWER TO AUDIT:\n{answer.strip()}\n\n"
        "List each atomic factual claim in the answer with its verdict. "
        "Ignore the [n] citation markers themselves when extracting claims."
    )

    payload = client.chat_json(
        prompt,
        schema=_GROUNDEDNESS_SCHEMA,
        system=_GROUNDEDNESS_SYSTEM,
        temperature=0.0,
        num_ctx=_context_window(joined, answer, _GROUNDEDNESS_SYSTEM),
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        raise OllamaError(f"groundedness audit returned an unusable payload: {payload!r}")

    supported = 0
    total = 0
    unsupported: list[str] = []
    for entry in payload["claims"]:
        if not isinstance(entry, dict):
            continue
        claim = str(entry.get("claim", "")).strip()
        if not claim:
            continue
        total += 1
        if str(entry.get("verdict", "")).strip().upper() == "SUPPORTED":
            supported += 1
        else:
            unsupported.append(claim)

    if total == 0:
        return 1.0, []
    return supported / total, unsupported


def synthesize(
    outcome: RetrievalOutcome,
    client: OllamaClient | None = None,
    store: "VectorStore | None" = None,
) -> AnswerResult:
    """Turn a retrieval outcome into a cited, verified answer — or a refusal.

    ``store`` is optional and used only to fetch the chunks behind matched QA
    pairs, giving the groundedness audit real document context. Without it the
    audit falls back to evidence quotes alone, which is stricter and can
    over-refuse on fragments.
    """
    start = time.perf_counter()
    result = _dispatch(outcome, client, store)
    result.elapsed_seconds = time.perf_counter() - start
    return result


def _dispatch(
    outcome: RetrievalOutcome,
    client: OllamaClient | None,
    store: "VectorStore | None" = None,
) -> AnswerResult:
    if outcome.route == ROUTE_REFUSED:
        return build_refusal(outcome)

    if outcome.route == ROUTE_QUESTION_BANK:
        chunk_texts: dict[str, str] = {}
        if store is not None:
            try:
                chunk_texts = store.chunk_texts(
                    [m.qa.chunk_id for m in outcome.qa_matches]
                )
            except Exception:  # audit context is a bonus, never a hard need
                chunk_texts = {}
        sources = _qa_sources(outcome.qa_matches, chunk_texts)
        header = (
            "The numbered sources below are question/answer pairs previously "
            "extracted from the documents, each with verbatim evidence."
        )
        extra = _INTEGRATE_INSTRUCTION if len(sources) > 1 else ""
    elif outcome.route == ROUTE_VECTOR:
        sources = _chunk_sources(outcome.chunk_matches)
        header = (
            "The numbered sources below are passages retrieved verbatim from "
            "the documents."
        )
        extra = ""
    elif outcome.route == ROUTE_HYBRID:
        chunk_texts = {}
        if store is not None:
            try:
                chunk_texts = store.chunk_texts(
                    [m.qa.chunk_id for m in outcome.qa_matches]
                )
            except Exception:
                chunk_texts = {}
        sources = _qa_sources(outcome.qa_matches, chunk_texts)
        sources.extend(_chunk_sources(outcome.chunk_matches))
        # Renumber after concat so citation markers stay contiguous.
        for i, source in enumerate(sources, start=1):
            source.index = i
        header = (
            "The numbered sources mix pre-extracted Q/A pairs and retrieved "
            "passages. Prefer the Q/A evidence when they agree; use passages "
            "to fill gaps. Cite every claim."
        )
        extra = _INTEGRATE_INSTRUCTION if len(sources) > 1 else ""
    else:
        raise ValueError(f"unknown retrieval route: {outcome.route!r}")

    if not sources:
        # Retrieval routed us here with nothing to read; refuse rather than
        # hand the model an empty source list it will happily fill in.
        refusal = build_refusal(outcome)
        refusal.notes.append(
            f"Route '{outcome.route}' was selected but carried no usable sources."
        )
        return refusal

    return _generate(outcome, sources, header, extra, client)


def synthesize_stream(
    outcome: RetrievalOutcome,
    client: OllamaClient | None = None,
    store: "VectorStore | None" = None,
):
    """Like :func:`synthesize`, but yields answer tokens before the final result.

    Yields ``str`` tokens during generation, then a final :class:`AnswerResult`.
    Refusals and failures yield only the :class:`AnswerResult`.
    """
    start = time.perf_counter()
    client = client or get_client()

    if outcome.route == ROUTE_REFUSED:
        result = build_refusal(outcome)
        result.elapsed_seconds = time.perf_counter() - start
        yield result
        return

    if outcome.route == ROUTE_QUESTION_BANK:
        chunk_texts: dict[str, str] = {}
        if store is not None:
            try:
                chunk_texts = store.chunk_texts(
                    [m.qa.chunk_id for m in outcome.qa_matches]
                )
            except Exception:
                chunk_texts = {}
        sources = _qa_sources(outcome.qa_matches, chunk_texts)
        header = (
            "The numbered sources below are question/answer pairs previously "
            "extracted from the documents, each with verbatim evidence."
        )
        extra = _INTEGRATE_INSTRUCTION if len(sources) > 1 else ""
    elif outcome.route == ROUTE_VECTOR:
        sources = _chunk_sources(outcome.chunk_matches)
        header = (
            "The numbered sources below are passages retrieved verbatim from "
            "the documents."
        )
        extra = ""
    elif outcome.route == ROUTE_HYBRID:
        chunk_texts = {}
        if store is not None:
            try:
                chunk_texts = store.chunk_texts(
                    [m.qa.chunk_id for m in outcome.qa_matches]
                )
            except Exception:
                chunk_texts = {}
        sources = _qa_sources(outcome.qa_matches, chunk_texts)
        sources.extend(_chunk_sources(outcome.chunk_matches))
        for i, source in enumerate(sources, start=1):
            source.index = i
        header = (
            "The numbered sources mix pre-extracted Q/A pairs and retrieved "
            "passages. Prefer the Q/A evidence when they agree; use passages "
            "to fill gaps. Cite every claim."
        )
        extra = _INTEGRATE_INSTRUCTION if len(sources) > 1 else ""
    else:
        raise ValueError(f"unknown retrieval route: {outcome.route!r}")

    if not sources:
        refusal = build_refusal(outcome)
        refusal.notes.append(
            f"Route '{outcome.route}' was selected but carried no usable sources."
        )
        refusal.elapsed_seconds = time.perf_counter() - start
        yield refusal
        return

    rendered = "\n\n".join(s.render() for s in sources)
    sections = [f"QUESTION:\n{outcome.query}", f"{header}\n\nSOURCES:\n{rendered}"]
    if extra:
        sections.append(extra)
    sections.append(
        f"Answer the question now, citing with [n] markers. "
        f"If the sources do not answer it, reply with exactly {INSUFFICIENT}."
    )
    prompt = "\n\n".join(sections)

    try:
        raw_parts: list[str] = []
        for token in client.chat_stream(
            prompt,
            system=_SYSTEM_PROMPT,
            num_ctx=_context_window(prompt, _SYSTEM_PROMPT),
        ):
            raw_parts.append(token)
            yield token
        raw = "".join(raw_parts).strip()
    except OllamaError as exc:
        refusal = build_refusal(outcome)
        refusal.notes.append(f"Answer generation failed: {exc}")
        refusal.elapsed_seconds = time.perf_counter() - start
        yield refusal
        return

    if _is_insufficient(raw):
        refusal = build_refusal(outcome)
        refusal.notes.append(
            "The model reported that the retrieved sources do not contain the answer."
        )
        refusal.elapsed_seconds = time.perf_counter() - start
        yield refusal
        return

    raw, partial = _strip_sentinel(raw)
    answer, citations = _remap_citations(raw, sources)
    notes = list(outcome.notes)
    if partial:
        notes.append(
            "The sources answered only part of this question; the unsupported "
            "part was omitted rather than guessed at."
        )
    if not answer.strip():
        refusal = build_refusal(outcome)
        refusal.notes.append(
            "The model reported that the retrieved sources do not contain the answer."
        )
        refusal.elapsed_seconds = time.perf_counter() - start
        yield refusal
        return
    if not citations:
        notes.append("The drafted answer carried no usable citation markers.")

    groundedness: float | None = None
    if config.ENABLE_GROUNDEDNESS_CHECK and not _should_skip_groundedness(outcome):
        try:
            score, unsupported = check_groundedness(
                answer, [s.evidence_for_audit() for s in sources], client
            )
        except OllamaError as exc:
            refusal = build_refusal(outcome)
            refusal.notes.append(
                f"An answer was drafted but could not be verified against the "
                f"sources, so it was withheld: {exc}"
            )
            refusal.elapsed_seconds = time.perf_counter() - start
            yield refusal
            return
        groundedness = score
        if score < config.GROUNDEDNESS_MIN_SCORE:
            refusal = build_refusal(outcome)
            refusal.groundedness = score
            refusal.notes.append(
                f"A draft answer was discarded: only {score:.0%} of its claims "
                f"were supported by the sources (minimum "
                f"{config.GROUNDEDNESS_MIN_SCORE:.0%})."
            )
            refusal.notes.extend(
                f"Unsupported claim: {c}" for c in unsupported[:5]
            )
            refusal.elapsed_seconds = time.perf_counter() - start
            yield refusal
            return
    elif _should_skip_groundedness(outcome):
        notes.append(
            "Groundedness audit skipped: strong question-bank hit with "
            "pre-validated evidence."
        )

    for missing in _flag_uncited_sentences(answer)[:3]:
        notes.append(f"Uncited claim sentence: {missing}")

    uncertainty = _uncertainty(outcome.confidence, groundedness, True)
    result = AnswerResult(
        query=outcome.query,
        answer=answer,
        route=outcome.route,
        answered=True,
        citations=citations,
        matched_questions=list(outcome.qa_matches),
        matched_chunks=list(outcome.chunk_matches),
        confidence=outcome.confidence,
        groundedness=groundedness,
        uncertainty=uncertainty,
        notes=notes,
        elapsed_seconds=time.perf_counter() - start,
    )
    yield result


def _generate(
    outcome: RetrievalOutcome,
    sources: list[_Source],
    header: str,
    extra: str,
    client: OllamaClient | None,
) -> AnswerResult:
    client = client or get_client()
    rendered = "\n\n".join(s.render() for s in sources)

    sections = [f"QUESTION:\n{outcome.query}", f"{header}\n\nSOURCES:\n{rendered}"]
    if extra:
        sections.append(extra)
    sections.append(
        f"Answer the question now, citing with [n] markers. "
        f"If the sources do not answer it, reply with exactly {INSUFFICIENT}."
    )
    prompt = "\n\n".join(sections)

    try:
        raw = client.chat(
            prompt,
            system=_SYSTEM_PROMPT,
            num_ctx=_context_window(prompt, _SYSTEM_PROMPT),
        )
    except OllamaError as exc:
        refusal = build_refusal(outcome)
        refusal.notes.append(f"Answer generation failed: {exc}")
        return refusal

    if _is_insufficient(raw):
        refusal = build_refusal(outcome)
        refusal.notes.append(
            "The model reported that the retrieved sources do not contain the answer."
        )
        return refusal

    raw, partial = _strip_sentinel(raw)
    answer, citations = _remap_citations(raw, sources)
    notes = list(outcome.notes)
    if partial:
        notes.append(
            "The sources answered only part of this question; the unsupported "
            "part was omitted rather than guessed at."
        )
    if not answer.strip():
        # Stripping the sentinel emptied the reply, so there was no real
        # answer under it after all.
        refusal = build_refusal(outcome)
        refusal.notes.append(
            "The model reported that the retrieved sources do not contain the answer."
        )
        return refusal
    if not citations:
        notes.append("The drafted answer carried no usable citation markers.")

    groundedness: float | None = None
    if config.ENABLE_GROUNDEDNESS_CHECK and not _should_skip_groundedness(outcome):
        try:
            score, unsupported = check_groundedness(
                answer, [s.evidence_for_audit() for s in sources], client
            )
        except OllamaError as exc:
            # FAIL CLOSED. Publishing an unverified draft was the single most
            # reachable way to violate the system's core promise: the audit is
            # a 31B call under the same timeout as everything else, so an
            # ordinary Ollama hiccup silently demoted the last guard to
            # nothing and a wholly fabricated answer went out with
            # groundedness=None as the only hint. A refusal is recoverable;
            # a confident fabrication is not.
            refusal = build_refusal(outcome)
            refusal.notes.append(
                f"An answer was drafted but could not be verified against the "
                f"sources, so it was withheld: {exc}"
            )
            return refusal
        else:
            groundedness = score
            if score < config.GROUNDEDNESS_MIN_SCORE:
                refusal = build_refusal(outcome)
                refusal.groundedness = score
                refusal.notes.append(
                    f"A draft answer was discarded: only {score:.0%} of its claims "
                    f"were supported by the sources (minimum "
                    f"{config.GROUNDEDNESS_MIN_SCORE:.0%})."
                )
                refusal.notes.extend(
                    f"Unsupported claim: {c}" for c in unsupported[:5]
                )
                return refusal
    elif _should_skip_groundedness(outcome):
        notes.append(
            "Groundedness audit skipped: strong question-bank hit with "
            "pre-validated evidence."
        )

    for missing in _flag_uncited_sentences(answer)[:3]:
        notes.append(f"Uncited claim sentence: {missing}")

    return AnswerResult(
        query=outcome.query,
        answer=answer,
        route=outcome.route,
        answered=True,
        citations=citations,
        matched_questions=list(outcome.qa_matches),
        matched_chunks=list(outcome.chunk_matches),
        confidence=outcome.confidence,
        groundedness=groundedness,
        uncertainty=_uncertainty(outcome.confidence, groundedness, True),
        notes=notes,
        elapsed_seconds=0.0,
    )


__all__ = [
    "synthesize",
    "synthesize_stream",
    "check_groundedness",
    "build_refusal",
    "INSUFFICIENT",
]
