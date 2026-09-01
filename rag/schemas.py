"""Data contracts shared by every stage of the pipeline.

These are plain dataclasses rather than pydantic models: they cross module
boundaries constantly and need cheap construction plus trivial round-tripping
into Chroma metadata (which only accepts str/int/float/bool scalars).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


def _stable_id(*parts: str) -> str:
    """Deterministic short id, so re-ingesting a document is idempotent."""
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return digest[:20]


# --------------------------------------------------------------------------
# Documents and chunks
# --------------------------------------------------------------------------


@dataclass
class PageText:
    """One page lifted out of a PDF."""

    page_number: int  # 1-indexed, as a human would cite it
    text: str


@dataclass
class Chunk:
    """A contiguous, page-aware slice of a document."""

    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    page_start: int
    page_end: int
    section: str = ""  # nearest preceding heading, best effort
    ordinal: int = 0  # position of this chunk within the document
    parent_id: str = ""  # parent window id when using parent-child chunking
    chunk_kind: str = "child"  # "child" | "parent" | "leaf" (legacy = child)

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
            "ordinal": self.ordinal,
            "parent_id": self.parent_id,
            "chunk_kind": self.chunk_kind,
        }


# --------------------------------------------------------------------------
# Extracted question/answer pairs
# --------------------------------------------------------------------------


@dataclass
class QAPair:
    """A question mined from the source text, with its grounded answer.

    ``evidence`` holds verbatim quotes copied out of the chunk. They are what
    makes a citation checkable: the UI shows them, and the groundedness pass
    scores the final answer against them.
    """

    qa_id: str
    question: str
    answer: str
    question_type: str
    difficulty: str
    evidence: list[str] = field(default_factory=list)
    paraphrases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    # provenance
    chunk_id: str = ""
    doc_id: str = ""
    doc_name: str = ""
    page_start: int = 0
    page_end: int = 0
    section: str = ""

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    def citation(self) -> str:
        return f"{self.doc_name}, {self.page_label}"

    def to_metadata(self) -> dict[str, Any]:
        """Flatten to Chroma-safe scalars (lists become JSON strings)."""
        return {
            "qa_id": self.qa_id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "evidence_json": json.dumps(self.evidence, ensure_ascii=False),
            "paraphrases_json": json.dumps(self.paraphrases, ensure_ascii=False),
            "keywords_json": json.dumps(self.keywords, ensure_ascii=False),
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
        }

    @classmethod
    def from_metadata(cls, meta: dict[str, Any]) -> "QAPair":
        def _load(key: str) -> list[str]:
            raw = meta.get(key) or "[]"
            try:
                value = json.loads(raw)
                return value if isinstance(value, list) else []
            except (json.JSONDecodeError, TypeError):
                return []

        return cls(
            qa_id=meta.get("qa_id", ""),
            question=meta.get("question", ""),
            answer=meta.get("answer", ""),
            question_type=meta.get("question_type", ""),
            difficulty=meta.get("difficulty", ""),
            evidence=_load("evidence_json"),
            paraphrases=_load("paraphrases_json"),
            keywords=_load("keywords_json"),
            chunk_id=meta.get("chunk_id", ""),
            doc_id=meta.get("doc_id", ""),
            doc_name=meta.get("doc_name", ""),
            page_start=int(meta.get("page_start", 0) or 0),
            page_end=int(meta.get("page_end", 0) or 0),
            section=meta.get("section", ""),
        )


# --------------------------------------------------------------------------
# Retrieval results
# --------------------------------------------------------------------------


@dataclass
class ScoredQA:
    qa: QAPair
    score: float
    matched_on: str = "question"  # "question" | "paraphrase"


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    """A single numbered source shown beneath an answer."""

    index: int  # the [n] marker used in the answer body
    doc_name: str
    page_label: str
    quote: str
    section: str = ""
    origin: str = "question_bank"  # "question_bank" | "vector"

    def label(self) -> str:
        base = f"[{self.index}] {self.doc_name}, {self.page_label}"
        return f"{base} — {self.section}" if self.section else base


# Route names, kept as constants so UI and pipeline agree.
ROUTE_QUESTION_BANK = "question_bank"
ROUTE_VECTOR = "vector_rag"
ROUTE_HYBRID = "hybrid"
ROUTE_REFUSED = "refused"

# Synthetic document a hand-authored question is filed under when it is not
# tied to an ingested PDF. Giving manual entries a real doc_id keeps the corpus
# counts honest (list_documents counts questions per doc_id) and lets them be
# managed — browsed, edited, bulk-deleted — exactly like extracted ones.
MANUAL_DOC_ID = "manual"
MANUAL_DOC_NAME = "Manual entries"


@dataclass
class AnswerResult:
    """Everything the UI needs to render one search."""

    query: str
    answer: str
    route: str
    answered: bool
    citations: list[Citation] = field(default_factory=list)
    matched_questions: list[ScoredQA] = field(default_factory=list)
    matched_chunks: list[ScoredChunk] = field(default_factory=list)
    confidence: float = 0.0
    groundedness: float | None = None
    uncertainty: float = 0.0  # 0 = certain, 1 = refuse / unknown
    near_misses: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "route": self.route,
            "answered": self.answered,
            "confidence": round(self.confidence, 4),
            "groundedness": self.groundedness,
            "uncertainty": round(self.uncertainty, 4),
            "near_misses": self.near_misses,
            "citations": [asdict(c) for c in self.citations],
            "notes": self.notes,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


__all__ = [
    "PageText",
    "Chunk",
    "QAPair",
    "ScoredQA",
    "ScoredChunk",
    "Citation",
    "AnswerResult",
    "ROUTE_QUESTION_BANK",
    "ROUTE_VECTOR",
    "ROUTE_HYBRID",
    "ROUTE_REFUSED",
    "MANUAL_DOC_ID",
    "MANUAL_DOC_NAME",
    "_stable_id",
]
