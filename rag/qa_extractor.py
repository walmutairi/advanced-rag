"""Mine an examination-grade question bank out of each chunk.

This is where the system's ceiling is set. Retrieval can only ever surface a
question that was extracted here, so the prompt pushes hard for cognitive
spread (``config.QUESTION_TYPES`` x ``config.DIFFICULTY_LEVELS``) instead of
the definition list an unguided model defaults to.

The sharp edge is evidence grounding. A model asked for "verbatim quotes"
returns quotes that are *almost* verbatim: reflowed whitespace, smart quotes
normalised, an ellipsis in the middle. So every quote is matched against the
chunk on a punctuation-free, casefolded token stream, and the span that is
actually recovered from ``chunk.text`` is what gets stored — never the model's
copy. A pair whose quotes cannot be located at all is destroyed rather than
returned: an uncited pair is precisely the hallucination this system exists to
prevent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import config
from rag.ollama_client import OllamaClient, OllamaError, get_extract_client
from rag.schemas import Chunk, QAPair, _stable_id
from rag.text_normalize import normalize_for_match

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tunables that are implementation detail rather than user-facing policy
# --------------------------------------------------------------------------

#: Below this a chunk is a page header, a caption or a stray footnote. Calling
#: a 31B model on it costs ~30s to be told, correctly, that there is nothing
#: here — so short-circuit instead. Lives in config because the right value is
#: script-dependent: a hardcoded 200 silently emptied the question bank for a
#: 97-character Arabic page that in fact yielded three good pairs.

_MIN_ANSWER_CHARS = 40

#: A one- or two-word "quote" is trivially present in any passage and cites
#: nothing useful, so it does not count towards a pair's grounding.
_MIN_EVIDENCE_TOKENS = 3
_MAX_EVIDENCE = 3

#: Token-set overlap required before a near-miss quote is repaired to a real
#: span. Below this the model is paraphrasing, not quoting.
_EVIDENCE_JACCARD_MIN = 0.75

#: Extraction reasons over the whole chunk plus a long instruction block and
#: emits several hundred tokens of JSON; the global default is sized for the
#: cheaper calls elsewhere in the pipeline.
_NUM_CTX = max(config.LLM_NUM_CTX, 16384)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Answers sometimes open with a hedge that breaks the "reads as a direct
#: domain assertion" rule even when the content is fine. Cheaper to shave the
#: prefix than to discard a good answer.
_META_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"according to (?:the )?(?:passage|text|document|section|article|author|source)s?"
    r"|based on (?:the )?(?:passage|text|document|section)s?"
    r"|as (?:stated|described|noted|explained|mentioned)(?: in the (?:passage|text|document|section))?"
    r"|the (?:passage|text|document|section|article|author|source) "
    r"(?:states|notes|says|explains|describes|indicates|mentions|reports|defines)(?: that)?"
    r"|this (?:passage|text|document|section|article) "
    r"(?:states|describes|explains|covers|discusses|outlines)(?: that)?"
    r")\s*[,:]?\s+",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Structured-output schema
# --------------------------------------------------------------------------


def _build_schema() -> dict[str, Any]:
    """Strict schema handed to Ollama's ``format`` parameter.

    Built from config at import time so widening ``QUESTION_TYPES`` widens the
    enum the decoder is constrained to, with no second place to edit.
    """
    return {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "minItems": 0,
                "maxItems": config.QA_PER_CHUNK_MAX,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "question_type": {
                            "type": "string",
                            "enum": list(config.QUESTION_TYPES),
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": list(config.DIFFICULTY_LEVELS),
                        },
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": _MAX_EVIDENCE,
                        },
                        "paraphrases": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": config.QA_PARAPHRASES,
                            "maxItems": config.QA_PARAPHRASES,
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 8,
                        },
                    },
                    "required": [
                        "question",
                        "answer",
                        "question_type",
                        "difficulty",
                        "evidence",
                        "paraphrases",
                        "keywords",
                    ],
                },
            }
        },
        "required": ["pairs"],
    }


QA_EXTRACTION_SCHEMA: dict[str, Any] = _build_schema()


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

#: Glosses for the config vocabulary. A type absent from this map still lands
#: in the prompt, just without an explanation.
_TYPE_GLOSS = {
    "factual": "a specific fact stated in the passage",
    "definitional": "the meaning of a term the passage defines",
    "conceptual": "why or how a mechanism works",
    "causal": "what causes what, and through which mechanism",
    "comparative": "X versus Y, trade-offs, when one beats the other",
    "quantitative": "figures, thresholds, rates, limits, measurements",
    "procedural": "the steps, order or method for doing something",
    "multi_hop": "requires joining two or more separate statements in the passage",
    "application": "apply the material to a concrete scenario not spelled out",
    "critical": "limitations, assumptions, failure modes, implications",
}

_SYSTEM_PROMPT = """You are a domain expert who writes examination papers. \
Given one passage from a technical document, you build the question bank a \
graduate-level examiner would build from it: questions that test whether \
someone has genuinely understood the material, each with a complete, \
defensible model answer.

You are strictly bounded by the passage. You never contribute knowledge from \
outside it, never guess at what a truncated sentence was going to say, and \
never assert something the passage merely hints at. If the passage does not \
support a question, you do not write that question.

You always write the bank in the language of the passage you were given, never
in the language of the instructions you were given.

You reply with JSON only."""


def _hard_types() -> list[str]:
    """The genuinely demanding types, if the configured vocabulary has them."""
    return [t for t in ("multi_hop", "critical", "application") if t in config.QUESTION_TYPES]


def _format_type_menu() -> str:
    return "\n".join(
        f"  - {t}: {_TYPE_GLOSS[t]}" if t in _TYPE_GLOSS else f"  - {t}"
        for t in config.QUESTION_TYPES
    )


_WORKED_EXAMPLE = """\
WORKED EXAMPLE — the difference between a weak pair and a strong one.

Suppose the passage read:
  "Write-ahead logging appends every mutation to a sequential log before the
  page cache is modified. Because the log is fsynced on commit while data
  pages are flushed lazily, a crash leaves the pages stale but recoverable by
  replaying the log. The fsync on commit dominates write latency, which is why
  group commit batches concurrent transactions into a single flush."

WEAK (do not produce pairs like this):
  question: "What is write-ahead logging?"
  answer:   "Write-ahead logging is a technique where mutations are appended
             to a log."
  Why it is weak: it tests recall of one clause, the answer restates the
  passage's first sentence, and nothing about it requires understanding.

STRONG (produce pairs like this):
  question_type: "multi_hop", difficulty: "advanced"
  question: "Why does group commit improve throughput in a write-ahead
             logging system, and what property of the log makes it safe?"
  answer:   "Group commit works because the fsync issued at commit time is the
             dominant cost of a write, so batching several concurrent
             transactions into one flush amortises that cost across all of
             them. It remains safe because the log is sequential and is
             durable before any page cache modification is applied: data pages
             may be stale after a crash, but replaying the log reconstructs
             every committed mutation. The correctness argument therefore
             rests on log ordering rather than on when pages are flushed."
  Why it is strong: it joins the latency claim to the recovery claim — two
  separate statements — and the answer stands alone as a piece of domain
  writing.
"""


def build_extraction_prompt(
    chunk: Chunk,
    *,
    prev_chunk: Chunk | None = None,
    next_chunk: Chunk | None = None,
) -> str:
    """The user-side prompt. Exposed so the UI can show what was asked."""
    hard = _hard_types()
    hard_clause = (
        "At least ONE pair must be "
        + " or ".join(f"`{t}`" for t in hard)
        + " where the material can support it. These are the pairs that make "
        "the bank worth building.\n"
        if hard
        else ""
    )
    location = f"{chunk.doc_name}, {chunk.page_label}"
    if chunk.section:
        location += f" — section: {chunk.section}"

    neighbor_block = ""
    if prev_chunk is not None or next_chunk is not None:
        parts: list[str] = []
        if prev_chunk is not None and prev_chunk.text.strip():
            parts.append(
                f"PREVIOUS PASSAGE ({prev_chunk.page_label})\n"
                f"<<<PREV\n{prev_chunk.text}\nPREV>>>"
            )
        if next_chunk is not None and next_chunk.text.strip():
            parts.append(
                f"NEXT PASSAGE ({next_chunk.page_label})\n"
                f"<<<NEXT\n{next_chunk.text}\nNEXT>>>"
            )
        if parts:
            neighbor_block = (
                "\n\nNEIGHBOURING CONTEXT (for multi_hop only)\n"
                + "\n\n".join(parts)
                + "\nEvidence quotes for every pair MUST still come from the "
                "SOURCE PASSAGE above, character for character. Use neighbours "
                "only to form questions that join facts across the boundary.\n"
            )

    return f"""SOURCE PASSAGE ({location})
<<<PASSAGE
{chunk.text}
PASSAGE>>>{neighbor_block}

TASK
Build a question bank from the passage above.

LANGUAGE
Write every question, answer, paraphrase and keyword in THE SAME LANGUAGE as
the source passage. If the passage is Arabic, the entire bank is Arabic; if it
is English, the bank is English. Never translate the material into English
because the instructions below are written in English — these instructions
describe the shape of the output, not its language. A reader of the source
document must be able to read the bank without translating it. Evidence quotes
are copied verbatim and are therefore always in the source language.

QUANTITY
Produce between {config.QA_PER_CHUNK_MIN} and {config.QA_PER_CHUNK_MAX} pairs. \
Produce FEWER if the passage is thin — a table of contents, a reference list, \
a copyright notice, a page of headings, a fragment of a table. Produce an \
EMPTY list if there is no substantive content to examine. A short honest bank \
beats a padded one; never invent material to hit a count.

COVERAGE
Spread the pairs across these question types:
{_format_type_menu()}
The passage decides which types genuinely apply — a narrative passage with no
figures cannot support a `quantitative` question, and forcing one produces
garbage. Cover the types the content actually supports, and do not write three
pairs of the same type when a different angle is available.
{hard_clause}
Spread difficulty across {", ".join(f"`{d}`" for d in config.DIFFICULTY_LEVELS)}. \
Do not label everything `advanced`; the label must describe the reasoning the \
question actually demands.

QUESTIONS
- Self-contained. A reader who has never seen this passage must still know
  what is being asked. Name the actual subject.
- Therefore no deictic references: never "this method", "the above", "the
  author", "the diagram", "as described here". Say what the method IS.
- One question, one thing. Do not staple two unrelated questions together.

ANSWERS
- Substantive and self-contained: typically 2-6 sentences, and they must read
  correctly even if the reader never sees the question.
- Written as direct domain assertions. FORBIDDEN phrasings: "the passage
  states", "according to the text", "this section describes", "the document
  explains", "as mentioned above". Assert the content instead.
- Derived STRICTLY from the passage. No outside knowledge, no background you
  happen to know, no filling in of gaps.
- Complete the reasoning. If the question asks why, the answer explains the
  mechanism, not just the conclusion.

EVIDENCE (1-{_MAX_EVIDENCE} per pair)
Quotes copied CHARACTER FOR CHARACTER out of the passage, supporting the
answer. Do not fix the grammar, do not shorten with "...", do not merge two
sentences that were not adjacent, do not translate. These quotes are checked
programmatically against the passage: a pair whose quotes are not found is
discarded entirely. Copy, do not compose.

PARAPHRASES (exactly {config.QA_PARAPHRASES} per pair)
Genuinely different ways a real user would ask the same thing — not word-order
shuffles of the question, not synonym swaps.
- At least one terse keyword form, the way someone types into a search box
  (e.g. "group commit fsync throughput").
- At least one natural full sentence, the way someone types into a chat box.
- The rest may be alternative framings, an inverted phrasing, or a
  practitioner's wording of the same need.

KEYWORDS (3-8 per pair)
The domain terms someone would actually search on. Real terminology from the
passage, not generic words like "system" or "process".

{_WORKED_EXAMPLE}
Return JSON: {{"pairs": [...]}} — nothing else."""


# --------------------------------------------------------------------------
# Normalisation and evidence grounding
# --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Casefolded, Arabic-folded, punctuation-free token stream for equality.

    Used for question deduplication. Evidence span recovery still tokenises the
    raw chunk so character offsets stay aligned with ``chunk.text``.
    """
    return normalize_for_match(text)


class _ChunkIndex:
    """Token view of a chunk that can map a match back to a raw span.

    Built once per chunk because every evidence quote scans it.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens: list[str] = []
        self.spans: list[tuple[int, int]] = []
        for match in _TOKEN_RE.finditer(text):
            self.tokens.append(match.group(0).casefold())
            self.spans.append(match.span())

        # Padded join lets a substring search behave like a token-sequence
        # search, so "cat" cannot match inside "catalogue".
        self.joined = " " + " ".join(self.tokens) + " "
        self._char_to_index: dict[int, int] = {}
        cursor = 1
        for index, token in enumerate(self.tokens):
            self._char_to_index[cursor] = index
            cursor += len(token) + 1

    def raw_span(self, start_token: int, end_token: int) -> str:
        """Verbatim chunk text covering tokens [start_token, end_token]."""
        return self.text[self.spans[start_token][0] : self.spans[end_token][1]]

    def locate(self, quote: str) -> str | None:
        """Recover the real passage span a quote refers to, or None.

        Exact token-sequence match first; failing that a sliding-window scan
        that tolerates a dropped word, an inserted gloss or a silent ellipsis.
        """
        q_tokens = [m.group(0).casefold() for m in _TOKEN_RE.finditer(quote)]
        if len(q_tokens) < _MIN_EVIDENCE_TOKENS or not self.tokens:
            return None

        needle = " " + " ".join(q_tokens) + " "
        position = self.joined.find(needle)
        if position != -1:
            start = self._char_to_index.get(position + 1)
            if start is not None:
                return self.raw_span(start, start + len(q_tokens) - 1)

        q_set = set(q_tokens)
        best_score = 0.0
        best: tuple[int, int] | None = None

        # Window lengths bracket the quote so both a truncated and a padded
        # rendering of the same sentence can still be recovered.
        base = len(q_tokens)
        lengths = {base, max(_MIN_EVIDENCE_TOKENS, int(base * 0.8)), int(base * 1.2) + 1}
        for length in sorted(lengths):
            if length <= 0 or length > len(self.tokens):
                continue
            for start in range(0, len(self.tokens) - length + 1):
                window = set(self.tokens[start : start + length])
                overlap = len(q_set & window)
                if not overlap:
                    continue
                score = overlap / len(q_set | window)
                if score > best_score:
                    best_score = score
                    best = (start, start + length - 1)

        if best is not None and best_score >= _EVIDENCE_JACCARD_MIN:
            return self.raw_span(*best)
        return None


def _ground_evidence(quotes: list[str], index: _ChunkIndex) -> list[str]:
    """Replace each quote with the real span it matches; drop the unmatched."""
    grounded: list[str] = []
    seen: set[str] = set()
    for quote in quotes:
        if not isinstance(quote, str) or not quote.strip():
            continue
        span = index.locate(quote)
        if span is None:
            continue
        key = _normalise(span)
        if key and key not in seen:
            seen.add(key)
            grounded.append(span.strip())
        if len(grounded) >= _MAX_EVIDENCE:
            break
    return grounded


# --------------------------------------------------------------------------
# Field coercion
# --------------------------------------------------------------------------

_TYPE_FALLBACK = "factual" if "factual" in config.QUESTION_TYPES else config.QUESTION_TYPES[0]
_DIFFICULTY_FALLBACK = (
    "intermediate" if "intermediate" in config.DIFFICULTY_LEVELS else config.DIFFICULTY_LEVELS[0]
)


def _coerce_vocab(value: Any, allowed: list[str], fallback: str) -> str:
    """Snap a model-supplied label into the allowed vocabulary.

    Never a reason to discard a pair — a good question mislabelled
    "multi-hop" instead of "multi_hop" is still a good question.
    """
    if not isinstance(value, str):
        return fallback
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
    if not key:
        return fallback
    lookup = {re.sub(r"[^a-z0-9]+", "_", a.casefold()).strip("_"): a for a in allowed}
    if key in lookup:
        return lookup[key]
    for normalised, original in lookup.items():
        if key.startswith(normalised) or normalised.startswith(key):
            return original
    return fallback


def _clean_answer(answer: str) -> str:
    text = answer.strip()
    for _ in range(2):  # models occasionally stack two hedges
        stripped = _META_PREFIX_RE.sub("", text)
        if stripped == text:
            break
        text = stripped
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _clean_question(question: str) -> str:
    text = " ".join(question.split()).strip()
    if not text:
        return ""
    text = text.rstrip(".:;,")
    # "?" is not the only question mark in use: Arabic/Persian end with "؟",
    # Greek with ";" and full-width CJK with "？". Testing only for the ASCII
    # form appends a second, wrong-script mark to every non-Latin question.
    if not text.endswith(("?", "؟", "？", "⁇")):
        text += "?"
    return text


def _clean_list(values: Any, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if not isinstance(values, list):
        return out
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.split()).strip()
        key = _normalise(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_pair(raw: Any, chunk: Chunk, index: _ChunkIndex) -> QAPair | None:
    """Turn one raw model object into a QAPair, or None if it cannot survive."""
    if not isinstance(raw, dict):
        return None

    question = _clean_question(str(raw.get("question") or ""))
    answer = _clean_answer(str(raw.get("answer") or ""))
    if len(question) < 2 or len(answer) < _MIN_ANSWER_CHARS:
        return None

    evidence = _ground_evidence(
        raw.get("evidence") if isinstance(raw.get("evidence"), list) else [], index
    )
    if not evidence:
        return None

    question_key = _normalise(question)
    if not question_key:
        return None

    paraphrases = [
        p for p in _clean_list(raw.get("paraphrases"), config.QA_PARAPHRASES * 2)
        if _normalise(p) != question_key
    ][: config.QA_PARAPHRASES]

    return QAPair(
        qa_id=_stable_id(chunk.chunk_id, question_key),
        question=question,
        answer=answer,
        question_type=_coerce_vocab(
            raw.get("question_type"), config.QUESTION_TYPES, _TYPE_FALLBACK
        ),
        difficulty=_coerce_vocab(
            raw.get("difficulty"), config.DIFFICULTY_LEVELS, _DIFFICULTY_FALLBACK
        ),
        evidence=evidence,
        paraphrases=paraphrases,
        keywords=_clean_list(raw.get("keywords"), 8),
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        doc_name=chunk.doc_name,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section=chunk.section,
    )


def _iter_raw_pairs(payload: Any) -> list[Any]:
    """Accept {"pairs": [...]}, a bare list, or a single object."""
    if isinstance(payload, dict):
        for key in ("pairs", "questions", "qa_pairs", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload] if "question" in payload else []
    if isinstance(payload, list):
        return payload
    return []


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_qa(
    chunk: Chunk,
    client: OllamaClient | None = None,
    *,
    prev_chunk: Chunk | None = None,
    next_chunk: Chunk | None = None,
) -> list[QAPair]:
    """Extract validated, evidence-grounded Q/A pairs from one chunk.

    Returns an empty list for a chunk with nothing to examine. Raises
    ``OllamaError`` if the model call itself fails — batch callers catch it.
    """
    if not isinstance(chunk, Chunk):
        raise TypeError(f"extract_qa expects a Chunk, got {type(chunk).__name__}")

    text = chunk.text or ""
    if len(text.strip()) < config.QA_MIN_CHUNK_CHARS:
        # Never skip silently: an empty question bank is otherwise
        # indistinguishable from a document that had nothing to ask about.
        log.warning(
            "chunk %s (%s %s) skipped: %d chars < QA_MIN_CHUNK_CHARS=%d",
            chunk.ordinal,
            chunk.doc_name,
            chunk.page_label,
            len(text.strip()),
            config.QA_MIN_CHUNK_CHARS,
        )
        return []

    client = client or get_extract_client()
    payload = client.chat_json(
        build_extraction_prompt(
            chunk, prev_chunk=prev_chunk, next_chunk=next_chunk
        ),
        schema=QA_EXTRACTION_SCHEMA,
        system=_SYSTEM_PROMPT,
        num_ctx=_NUM_CTX,
        timeout=config.EXTRACT_TIMEOUT,
        max_retries=1,
        retries=1,
    )

    index = _ChunkIndex(text)
    pairs: list[QAPair] = []
    seen: set[str] = set()

    for raw in _iter_raw_pairs(payload):
        pair = _validate_pair(raw, chunk, index)
        if pair is None:
            continue
        key = _normalise(pair.question)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(pair)
        if len(pairs) >= config.QA_PER_CHUNK_MAX:
            break

    return pairs


@dataclass
class BatchStats:
    """Outcome of the last ``extract_qa_batch`` run."""

    chunks_total: int = 0
    chunks_succeeded: int = 0
    chunks_failed: int = 0
    chunks_empty: int = 0
    pairs_kept: int = 0
    pairs_duplicate: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)  # (chunk_id, error)


_last_batch_stats = BatchStats()


def last_batch_stats() -> BatchStats:
    """Stats from the most recent batch, including per-chunk failures."""
    return _last_batch_stats


_thread_local = threading.local()


def _worker_client() -> OllamaClient:
    """One HTTP session per worker thread — ``requests.Session`` is not thread-safe."""
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = get_extract_client()
        _thread_local.client = client
    return client


def _merge_pairs(
    pairs: list[QAPair],
    by_question: dict[str, QAPair],
    seen_ids: set[str],
    stats: BatchStats,
) -> None:
    for pair in pairs:
        key = _normalise(pair.question)
        existing = by_question.get(key)
        if existing is None and pair.qa_id not in seen_ids:
            by_question[key] = pair
            seen_ids.add(pair.qa_id)
            continue
        stats.pairs_duplicate += 1
        if existing is not None and len(pair.answer) > len(existing.answer):
            by_question[key] = pair
            seen_ids.add(pair.qa_id)


def _chunk_text_digest(chunk: Chunk) -> str:
    payload = (
        f"{chunk.doc_id}\0{chunk.ordinal}\0{chunk.text}\0"
        f"neighbors={int(config.EXTRACT_NEIGHBOR_CHUNKS)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_cache_path(digest: str) -> Path:
    return config.CACHE_DIR / "extract" / f"{digest}.json"


def _load_extract_cache(chunk: Chunk) -> list[QAPair] | None:
    if not config.ENABLE_EXTRACT_CACHE:
        return None
    path = _extract_cache_path(_chunk_text_digest(chunk))
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [QAPair(**item) for item in raw]
    except Exception:
        log.debug("extract cache miss/corrupt for %s", chunk.chunk_id, exc_info=True)
        return None


def _store_extract_cache(chunk: Chunk, pairs: list[QAPair]) -> None:
    if not config.ENABLE_EXTRACT_CACHE:
        return
    path = _extract_cache_path(_chunk_text_digest(chunk))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(p) for p in pairs], ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        log.debug("extract cache write failed for %s", path, exc_info=True)


def _neighbors(
    chunks: list[Chunk], index: int
) -> tuple[Chunk | None, Chunk | None]:
    if not config.EXTRACT_NEIGHBOR_CHUNKS:
        return None, None
    chunk = chunks[index]
    prev_chunk = chunks[index - 1] if index > 0 else None
    next_chunk = chunks[index + 1] if index + 1 < len(chunks) else None
    if prev_chunk is not None and prev_chunk.doc_id != chunk.doc_id:
        prev_chunk = None
    if next_chunk is not None and next_chunk.doc_id != chunk.doc_id:
        next_chunk = None
    return prev_chunk, next_chunk


def extract_qa_batch(
    chunks: list[Chunk],
    client: OllamaClient | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[QAPair]:
    """Extract across many chunks, surviving individual failures.

    Ingestion of a large PDF is measured in tens of minutes, so one bad chunk
    must never cost the whole run: failures are logged, counted into
    ``last_batch_stats()`` and stepped over.

    Overlapping chunks legitimately rediscover the same question, so results
    are deduplicated globally on ``qa_id`` and on normalised question text,
    keeping whichever copy has the longer (more complete) answer.

    When ``EXTRACT_NEIGHBOR_CHUNKS`` is on, each call also sees the previous and
    next chunk from the same document (for multi_hop). Unchanged chunk text is
    served from ``data/cache/extract`` when ``ENABLE_EXTRACT_CACHE`` is on.
    """
    global _last_batch_stats

    client = client or get_extract_client()
    # Parents are context windows only — mining them would duplicate the bank.
    extractable = [
        c for c in chunks if (getattr(c, "chunk_kind", "child") or "child") != "parent"
    ]
    total = len(extractable)
    stats = BatchStats(chunks_total=total)
    _last_batch_stats = stats

    by_question: dict[str, QAPair] = {}
    seen_ids: set[str] = set()
    merge_lock = threading.Lock()
    workers = min(config.QA_EXTRACT_WORKERS, total) if total else 1

    def _abort_if_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            from rag.pipeline import IngestCancelled

            raise IngestCancelled("Ingest cancelled.")

    def _process_index(
        index: int,
    ) -> tuple[Chunk, list[QAPair], str, str | None]:
        chunk = extractable[index]
        cached = _load_extract_cache(chunk)
        if cached is not None:
            message = (
                f"{chunk.doc_name} {chunk.page_label}: "
                f"{len(cached)} questions (cache)"
            )
            return chunk, cached, message, None
        prev_chunk, next_chunk = _neighbors(extractable, index)
        try:
            pairs = extract_qa(
                chunk,
                client=_worker_client() if workers > 1 else client,
                prev_chunk=prev_chunk,
                next_chunk=next_chunk,
            )
            _store_extract_cache(chunk, pairs)
            message = f"{chunk.doc_name} {chunk.page_label}: {len(pairs)} questions"
            return chunk, pairs, message, None
        except (OllamaError, ValueError, KeyError, TypeError) as exc:
            log.warning("QA extraction failed for chunk %s: %s", chunk.chunk_id, exc)
            message = f"{chunk.doc_name} {chunk.page_label}: extraction failed ({exc})"
            return chunk, [], message, str(exc)

    done = 0
    if workers <= 1:
        for index in range(total):
            _abort_if_cancelled()
            chunk, pairs, message, error = _process_index(index)
            done += 1
            if error:
                stats.chunks_failed += 1
                stats.failures.append((chunk.chunk_id, error))
            else:
                stats.chunks_succeeded += 1
                if not pairs:
                    stats.chunks_empty += 1
            with merge_lock:
                _merge_pairs(pairs, by_question, seen_ids, stats)
            if progress_cb is not None:
                progress_cb(done, total, message)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_index, index): index for index in range(total)
            }
            for future in as_completed(futures):
                _abort_if_cancelled()
                chunk, pairs, message, error = future.result()
                done += 1
                if error:
                    stats.chunks_failed += 1
                    stats.failures.append((chunk.chunk_id, error))
                else:
                    stats.chunks_succeeded += 1
                    if not pairs:
                        stats.chunks_empty += 1
                with merge_lock:
                    _merge_pairs(pairs, by_question, seen_ids, stats)
                if progress_cb is not None:
                    progress_cb(done, total, message)

    if config.ENABLE_CROSS_CHUNK_SYNTHESIS and len(extractable) >= 2:
        _abort_if_cancelled()
        if progress_cb is not None:
            cap = min(config.CROSS_CHUNK_MAX_PAIRS, max(0, len(extractable) - 1))
            progress_cb(
                total,
                total + max(1, cap),
                f"{extractable[0].doc_name}: starting cross-chunk "
                f"(up to {cap} pairs) …",
            )
        cross_pairs = _cross_chunk_synthesis(
            extractable,
            client=_worker_client() if workers > 1 else client,
            cancel_event=cancel_event,
            progress_cb=progress_cb,
        )
        with merge_lock:
            _merge_pairs(cross_pairs, by_question, seen_ids, stats)

    kept = list(by_question.values())
    stats.pairs_kept = len(kept)
    log.info(
        "extracted %d pairs from %d/%d chunks (%d failed, %d empty, %d duplicates)",
        stats.pairs_kept,
        stats.chunks_succeeded,
        total,
        stats.chunks_failed,
        stats.chunks_empty,
        stats.pairs_duplicate,
    )
    return kept


def _cross_chunk_synthesis(
    chunks: list[Chunk],
    client: OllamaClient,
    cancel_event: threading.Event | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[QAPair]:
    """Ask for multi_hop pairs that truly need two adjacent passages."""
    from rag.pipeline import IngestCancelled

    pairs: list[QAPair] = []
    # Prefer later boundaries (often denser mid/end of docs) but hard-cap cost.
    candidate_idxs = list(range(len(chunks) - 1))
    if len(candidate_idxs) > config.CROSS_CHUNK_MAX_PAIRS:
        step = len(candidate_idxs) / config.CROSS_CHUNK_MAX_PAIRS
        candidate_idxs = sorted(
            {
                min(len(chunks) - 2, int(i * step))
                for i in range(config.CROSS_CHUNK_MAX_PAIRS)
            }
        )
    boundaries = max(1, len(candidate_idxs))
    for n, i in enumerate(candidate_idxs, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise IngestCancelled("Ingest cancelled.")
        if progress_cb is not None:
            # Keep the bar moving past the flat "done/total" peak at 95%.
            progress_cb(
                len(chunks) + n,
                len(chunks) + boundaries,
                f"{chunks[i].doc_name}: cross-chunk {n}/{boundaries} …",
            )
        left, right = chunks[i], chunks[i + 1]
        if left.doc_id != right.doc_id:
            continue
        if len(left.text.strip()) < config.QA_MIN_CHUNK_CHARS:
            continue
        if len(right.text.strip()) < config.QA_MIN_CHUNK_CHARS:
            continue
        prompt = f"""Two adjacent SOURCE PASSAGES from the same document.

PASSAGE A ({left.doc_name}, {left.page_label})
<<<A
{left.text}
A>>>

PASSAGE B ({right.doc_name}, {right.page_label})
<<<B
{right.text}
B>>>

TASK
Produce 0-2 `multi_hop` question/answer pairs that REQUIRE joining a fact from
A with a fact from B. If nothing genuinely joins, return {{"pairs": []}}.

LANGUAGE: same language as the passages.
EVIDENCE: quotes must be copied CHARACTER FOR CHARACTER from A or B.
Answers must be self-contained domain assertions (no "the passage says").

Return JSON: {{"pairs": [...]}}"""
        try:
            payload = client.chat_json(
                prompt,
                schema=QA_EXTRACTION_SCHEMA,
                system=_SYSTEM_PROMPT,
                num_ctx=_NUM_CTX,
                timeout=config.EXTRACT_TIMEOUT,
                max_retries=1,
                retries=0,
            )
        except OllamaError as exc:
            log.warning("cross-chunk synthesis failed at %s: %s", left.chunk_id, exc)
            continue
        # Validate evidence against the concatenated text of both chunks.
        combined = Chunk(
            chunk_id=left.chunk_id,
            doc_id=left.doc_id,
            doc_name=left.doc_name,
            text=f"{left.text}\n\n{right.text}",
            page_start=left.page_start,
            page_end=right.page_end,
            section=left.section,
            ordinal=left.ordinal,
        )
        index = _ChunkIndex(combined.text)
        for raw in _iter_raw_pairs(payload):
            if (raw.get("question_type") or "").strip().lower() not in {
                "multi_hop",
                "comparative",
                "critical",
            }:
                raw = dict(raw)
                raw["question_type"] = "multi_hop"
            pair = _validate_pair(raw, combined, index)
            if pair is not None:
                pairs.append(pair)
    return pairs


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def extraction_report(pairs: list[QAPair]) -> dict[str, Any]:
    """Shape of a question bank: counts by type, difficulty and document."""
    by_type = {t: 0 for t in config.QUESTION_TYPES}
    by_difficulty = {d: 0 for d in config.DIFFICULTY_LEVELS}
    by_document: dict[str, int] = {}

    total_evidence = 0
    for pair in pairs:
        by_type[pair.question_type] = by_type.get(pair.question_type, 0) + 1
        by_difficulty[pair.difficulty] = by_difficulty.get(pair.difficulty, 0) + 1
        by_document[pair.doc_name] = by_document.get(pair.doc_name, 0) + 1
        total_evidence += len(pair.evidence)

    hard = _hard_types()
    return {
        "total": len(pairs),
        "by_type": by_type,
        "by_difficulty": by_difficulty,
        "by_document": by_document,
        "chunks_covered": len({p.chunk_id for p in pairs}),
        "hard_question_share": (
            round(sum(by_type.get(t, 0) for t in hard) / len(pairs), 3) if pairs else 0.0
        ),
        "avg_evidence_per_pair": round(total_evidence / len(pairs), 2) if pairs else 0.0,
        "avg_answer_chars": (
            round(sum(len(p.answer) for p in pairs) / len(pairs), 1) if pairs else 0.0
        ),
    }


__all__ = [
    "QA_EXTRACTION_SCHEMA",
    "BatchStats",
    "build_extraction_prompt",
    "extract_qa",
    "extract_qa_batch",
    "extraction_report",
    "last_batch_stats",
]
