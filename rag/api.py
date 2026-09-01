"""Minimal local HTTP API for ingest/ask/stats (stdlib only).

Run:
    ./.venv/bin/python -m rag.api
    curl -s http://127.0.0.1:8765/health
    curl -s -X POST http://127.0.0.1:8765/ask -H 'content-type: application/json' \\
      -d '{"query":"What are the limitations?"}'
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rag.pipeline import RAGPipeline

log = logging.getLogger(__name__)
_PIPELINE: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RAGPipeline()
    return _PIPELINE


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        pipeline = get_pipeline()
        if path == "/health":
            ok, message = pipeline.health()
            self._json(200 if ok else 503, {"ok": ok, "message": message})
            return
        if path == "/stats":
            self._json(200, pipeline.stats())
            return
        if path == "/documents":
            self._json(200, {"documents": pipeline.documents()})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        pipeline = get_pipeline()
        try:
            data = self._read_json()
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return

        if path == "/ask":
            query = (data.get("query") or "").strip()
            if not query:
                self._json(400, {"error": "query is required"})
                return
            result = pipeline.search(query)
            self._json(200, result.as_dict())
            return

        if path == "/ingest":
            paths = data.get("paths") or []
            if not paths:
                self._json(400, {"error": "paths is required"})
                return
            force = bool(data.get("force", False))
            force_ocr = bool(data.get("force_ocr", False))
            try:
                result = pipeline.ingest_pdfs(
                    [Path(p) for p in paths],
                    force=force,
                    force_ocr=force_ocr,
                )
            except Exception as exc:  # noqa: BLE001
                self._json(500, {"error": str(exc)})
                return
            self._json(200, result)
            return

        self._json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)


def main(host: str = "127.0.0.1", port: int = 8765) -> None:
    logging.basicConfig(level=logging.INFO)
    server = ThreadingHTTPServer((host, port), Handler)
    log.info("RAG API listening on http://%s:%d", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main()
