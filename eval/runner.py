"""Shared eval logic for CLI, post-ingest hooks, and naive-RAG baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import config
from rag.pipeline import RAGPipeline
from rag.retrieval import route_query
from rag.schemas import ROUTE_REFUSED, ROUTE_VECTOR, AnswerResult


def load_questions(path: Path | str | None = None) -> list[dict]:
    target = Path(path or config.EVAL_QUESTIONS_PATH)
    if not target.is_file():
        return []
    data = json.loads(target.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _route_ok(actual: str, expected: Any) -> bool:
    if expected is None:
        return True
    if isinstance(expected, str):
        return actual == expected
    return actual in list(expected)


def run_eval(
    pipeline: RAGPipeline,
    questions: list[dict] | None = None,
    *,
    route_only: bool = False,
    search_fn: Callable[[str], AnswerResult] | None = None,
) -> dict[str, Any]:
    """Evaluate question-bank-first pipeline (or a custom search_fn baseline)."""
    items = questions if questions is not None else load_questions()
    stats = {
        "total": 0,
        "route_ok": 0,
        "answered": 0,
        "refused": 0,
        "contain_ok": 0,
        "contain_checked": 0,
        "avg_confidence": 0.0,
        "avg_groundedness": 0.0,
        "rows": [],
    }
    confidences: list[float] = []
    grounded: list[float] = []

    for item in items:
        query = (item.get("query") or "").strip()
        if not query:
            continue
        stats["total"] += 1
        qid = item.get("id") or f"q{stats['total']}"

        if search_fn is not None:
            result = search_fn(query)
            route = result.route
            answered = result.answered
            answer = result.answer or ""
            confidences.append(result.confidence)
            if result.groundedness is not None:
                grounded.append(result.groundedness)
            notes = "; ".join(result.notes[:2])
        elif route_only:
            outcome = route_query(query, pipeline.store, client=pipeline.client)
            route = outcome.route
            answered = route != ROUTE_REFUSED
            answer = ""
            confidences.append(outcome.confidence)
            notes = "; ".join(outcome.notes[:2])
        else:
            result = pipeline.search(query)
            route = result.route
            answered = result.answered
            answer = result.answer or ""
            confidences.append(result.confidence)
            if result.groundedness is not None:
                grounded.append(result.groundedness)
            notes = "; ".join(result.notes[:2])

        if answered:
            stats["answered"] += 1
        else:
            stats["refused"] += 1

        route_match = _route_ok(route, item.get("expect_route"))
        if item.get("expect_answered") is not None:
            route_match = route_match and bool(item["expect_answered"]) == answered
        if route_match:
            stats["route_ok"] += 1

        contain_ok = True
        must = item.get("must_contain") or []
        if must and answered:
            stats["contain_checked"] += 1
            contain_ok = all(p.casefold() in answer.casefold() for p in must)
            if contain_ok:
                stats["contain_ok"] += 1

        stats["rows"].append(
            {
                "id": qid,
                "ok": route_match and contain_ok,
                "route": route,
                "answered": answered,
                "query": query,
                "notes": notes,
            }
        )

    total = max(stats["total"], 1)
    stats["route_accuracy"] = stats["route_ok"] / total
    stats["refuse_rate"] = stats["refused"] / total
    stats["avg_confidence"] = (
        round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    )
    stats["avg_groundedness"] = (
        round(sum(grounded) / len(grounded), 3) if grounded else None
    )
    if stats["contain_checked"]:
        stats["contain_accuracy"] = stats["contain_ok"] / stats["contain_checked"]
    return stats


def naive_vector_search(pipeline: RAGPipeline, query: str) -> AnswerResult:
    """Baseline: force Tier-2-style chunk RAG only (no question bank)."""
    from rag.answerer import synthesize
    from rag.retrieval import RetrievalOutcome, expand_query, _fuse
    from rag.schemas import ROUTE_REFUSED, ROUTE_VECTOR

    notes = ["baseline: naive vector RAG (question bank disabled)"]
    phrasings = expand_query(query, pipeline.client)
    rankings = []
    for phrasing in phrasings:
        try:
            rankings.append(pipeline.store.search_chunks(phrasing))
        except Exception as exc:
            notes.append(f"chunk search failed ({exc})")
    fused = _fuse(rankings) if rankings else []
    threshold = config.VECTOR_MATCH_THRESHOLD
    accepted = [m for m in fused if m.score >= threshold]
    if not accepted:
        return synthesize(
            RetrievalOutcome(
                query=query,
                route=ROUTE_REFUSED,
                confidence=max((m.score for m in fused), default=0.0),
                notes=notes + ["baseline refused: no chunk above threshold"],
            ),
            client=pipeline.client,
            store=pipeline.store,
        )
    return synthesize(
        RetrievalOutcome(
            query=query,
            route=ROUTE_VECTOR,
            chunk_matches=accepted,
            confidence=max(m.score for m in accepted),
            notes=notes,
        ),
        client=pipeline.client,
        store=pipeline.store,
    )


def compare_to_baseline(pipeline: RAGPipeline, questions: list[dict] | None = None) -> dict:
    """Run full system vs naive vector RAG on the same golden set."""
    items = questions if questions is not None else load_questions()
    if not items:
        return {"error": "no questions", "system": None, "baseline": None}
    system = run_eval(pipeline, items, route_only=False)
    baseline = run_eval(
        pipeline,
        items,
        search_fn=lambda q: naive_vector_search(pipeline, q),
    )
    return {
        "system": {
            "route_accuracy": system["route_accuracy"],
            "refuse_rate": system["refuse_rate"],
            "answered": system["answered"],
            "avg_confidence": system["avg_confidence"],
            "avg_groundedness": system["avg_groundedness"],
        },
        "baseline": {
            "route_accuracy": baseline["route_accuracy"],
            "refuse_rate": baseline["refuse_rate"],
            "answered": baseline["answered"],
            "avg_confidence": baseline["avg_confidence"],
            "avg_groundedness": baseline["avg_groundedness"],
        },
        "delta_route_accuracy": system["route_accuracy"] - baseline["route_accuracy"],
        "delta_refuse_rate": system["refuse_rate"] - baseline["refuse_rate"],
    }


__all__ = [
    "load_questions",
    "run_eval",
    "naive_vector_search",
    "compare_to_baseline",
]
