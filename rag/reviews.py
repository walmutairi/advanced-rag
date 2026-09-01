"""Pending question-bank reviews from thumbs-down feedback.

A thumbs-down on a live answer parks that Q/A here so a human can fix it,
then approve it into the real question bank. Thumbs-up is a no-op: nothing
is stored.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from typing import Any

import config
from rag.schemas import AnswerResult

_REVIEWS_PATH = config.DATA_DIR / "pending_reviews.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _review_id(query: str) -> str:
    digest = hashlib.sha256(query.strip().casefold().encode("utf-8")).hexdigest()
    return f"review_{digest[:16]}"


def _load() -> list[dict[str, Any]]:
    if not _REVIEWS_PATH.exists():
        return []
    try:
        data = json.loads(_REVIEWS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _save(items: list[dict[str, Any]]) -> None:
    _REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REVIEWS_PATH.write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_reviews() -> list[dict[str, Any]]:
    """Oldest first — the ones waiting longest sit at the top."""
    return list(_load())


def review_count() -> int:
    return len(_load())


def get_review(review_id: str) -> dict[str, Any] | None:
    for item in _load():
        if item.get("review_id") == review_id:
            return item
    return None


def find_review_for_query(query: str) -> dict[str, Any] | None:
    rid = _review_id(query)
    return get_review(rid)


def discard_review(review_id: str) -> None:
    _save([item for item in _load() if item.get("review_id") != review_id])


def update_review(review_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    items = _load()
    updated: dict[str, Any] | None = None
    for item in items:
        if item.get("review_id") != review_id:
            continue
        item.update(fields)
        item["updated_at"] = _now()
        updated = item
        break
    if updated is not None:
        _save(items)
    return updated


def _provenance(result: AnswerResult) -> dict[str, Any]:
    if result.matched_questions:
        qa = result.matched_questions[0].qa
        return {
            "source_qa_id": qa.qa_id,
            "chunk_id": qa.chunk_id,
            "doc_id": qa.doc_id,
            "doc_name": qa.doc_name,
            "page_start": qa.page_start,
            "page_end": qa.page_end,
            "section": qa.section,
            "question_type": qa.question_type,
            "difficulty": qa.difficulty,
            "paraphrases": list(qa.paraphrases),
            "keywords": list(qa.keywords),
        }
    if result.matched_chunks:
        chunk = result.matched_chunks[0].chunk
        return {
            "source_qa_id": "",
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "doc_name": chunk.doc_name,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section": chunk.section or "",
            "question_type": "factual",
            "difficulty": "basic",
            "paraphrases": [],
            "keywords": [],
        }
    if result.citations:
        citation = result.citations[0]
        return {
            "source_qa_id": "",
            "chunk_id": "",
            "doc_id": "",
            "doc_name": citation.doc_name,
            "page_start": 0,
            "page_end": 0,
            "section": citation.section,
            "question_type": "factual",
            "difficulty": "basic",
            "paraphrases": [],
            "keywords": [],
        }
    return {
        "source_qa_id": "",
        "chunk_id": "",
        "doc_id": "",
        "doc_name": "",
        "page_start": 0,
        "page_end": 0,
        "section": "",
        "question_type": "factual",
        "difficulty": "basic",
        "paraphrases": [],
        "keywords": [],
    }


def _merge_existing_review(existing: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Re-flagging updates the live answer but keeps saved draft edits."""
    merged = dict(fresh)
    merged["created_at"] = existing.get("created_at", fresh["created_at"])
    draft_saved = existing.get("updated_at") and (
        existing.get("updated_at") != existing.get("created_at")
    )
    if draft_saved:
        for key in (
            "question",
            "answer",
            "question_type",
            "difficulty",
            "keywords",
            "paraphrases",
            "evidence",
        ):
            if existing.get(key):
                merged[key] = existing[key]
    else:
        for key in ("question_type", "difficulty", "keywords", "paraphrases"):
            if existing.get(key) and not fresh.get(key):
                merged[key] = existing[key]
    if existing.get("source_qa_id") and not fresh.get("source_qa_id"):
        merged["source_qa_id"] = existing["source_qa_id"]
    return merged


def flag_answer(result: AnswerResult) -> dict[str, Any]:
    """Park a thumbs-downed answer for review. Re-flagging the same query updates it."""
    query = (result.query or "").strip()
    if not query:
        raise ValueError("Cannot flag an empty question.")

    evidence = [c.quote.strip() for c in result.citations if c.quote and c.quote.strip()]
    provenance = _provenance(result)
    item = {
        "review_id": _review_id(query),
        "query": query,
        "question": query,
        "answer": (result.answer or "").strip(),
        "route": result.route,
        "evidence": evidence,
        "created_at": _now(),
        "updated_at": _now(),
        **provenance,
    }

    existing = find_review_for_query(query)
    if existing is not None:
        item = _merge_existing_review(existing, item)

    items = [row for row in _load() if row.get("review_id") != item["review_id"]]
    items.append(item)
    _save(items)
    return item


def clear_all_reviews() -> None:
    """Drop every pending review — used when the index is reset."""
    if _REVIEWS_PATH.exists():
        _REVIEWS_PATH.unlink()


__all__ = [
    "clear_all_reviews",
    "discard_review",
    "find_review_for_query",
    "flag_answer",
    "get_review",
    "list_reviews",
    "review_count",
    "update_review",
]
