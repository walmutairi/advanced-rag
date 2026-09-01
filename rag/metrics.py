"""Lightweight local observability for refuse rate, latency, OCR, etc."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import config

_LOCK = threading.Lock()
_PATH = config.CACHE_DIR / "metrics.json"
_MAX_EVENTS = 500


def _load() -> dict[str, Any]:
    if not _PATH.is_file():
        return {"events": [], "totals": {}}
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"events": [], "totals": {}}


def _save(data: dict[str, Any]) -> None:
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def record_search(
    *,
    route: str,
    answered: bool,
    confidence: float,
    groundedness: float | None,
    elapsed: float,
) -> None:
    with _LOCK:
        data = _load()
        events = data.setdefault("events", [])
        events.append(
            {
                "ts": time.time(),
                "kind": "search",
                "route": route,
                "answered": answered,
                "confidence": confidence,
                "groundedness": groundedness,
                "elapsed": elapsed,
            }
        )
        data["events"] = events[-_MAX_EVENTS:]
        totals = data.setdefault("totals", {})
        totals["searches"] = int(totals.get("searches", 0)) + 1
        totals["answered"] = int(totals.get("answered", 0)) + int(answered)
        totals["refused"] = int(totals.get("refused", 0)) + int(not answered)
        _save(data)


def record_ingest(*, ocr_pages: int, questions: int, chunks: int, elapsed: float) -> None:
    with _LOCK:
        data = _load()
        events = data.setdefault("events", [])
        events.append(
            {
                "ts": time.time(),
                "kind": "ingest",
                "ocr_pages": ocr_pages,
                "questions": questions,
                "chunks": chunks,
                "elapsed": elapsed,
            }
        )
        data["events"] = events[-_MAX_EVENTS:]
        totals = data.setdefault("totals", {})
        totals["ingests"] = int(totals.get("ingests", 0)) + 1
        totals["ocr_pages"] = int(totals.get("ocr_pages", 0)) + int(ocr_pages)
        _save(data)


def summary() -> dict[str, Any]:
    with _LOCK:
        data = _load()
    events = data.get("events") or []
    searches = [e for e in events if e.get("kind") == "search"]
    answered = sum(1 for e in searches if e.get("answered"))
    refused = sum(1 for e in searches if not e.get("answered"))
    latencies = [float(e.get("elapsed") or 0) for e in searches if e.get("elapsed")]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    totals = data.get("totals") or {}
    return {
        "searches": len(searches),
        "answered": answered,
        "refused": refused,
        "refuse_rate": (refused / len(searches)) if searches else 0.0,
        "avg_latency_s": round(avg_latency, 2),
        "lifetime_ingests": int(totals.get("ingests", 0)),
        "lifetime_ocr_pages": int(totals.get("ocr_pages", 0)),
    }


__all__ = ["record_search", "record_ingest", "summary"]
