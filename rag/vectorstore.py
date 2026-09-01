"""Persistent hybrid store over two Chroma collections.

The question bank and the raw chunks live side by side because the retriever
routes between them: a hit in the question bank is a pre-answered question, a
hit in the chunks is classical RAG. Both are searched the same way — dense
embeddings from Ollama blended with an in-memory BM25 index.

Two sharp edges:

* Chroma never embeds anything here. Vectors always come from
  ``OllamaClient.embed`` (L2-normalised), so every collection is created with
  ``embedding_function=None``. Letting Chroma fall back to its bundled
  MiniLM would silently mix two incompatible vector spaces in one index.
* A QAPair occupies *several* rows in the question collection — one per
  paraphrase plus the canonical question — so a raw top-k is easily filled by
  the variants of a single question. Dense search therefore over-fetches and
  collapses by ``qa_id`` before truncating.
"""

from __future__ import annotations

import fcntl
import json
import logging
import pickle
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

import config

from .ollama_client import OllamaClient, OllamaError, get_client
from .schemas import Chunk, QAPair, ScoredChunk, ScoredQA
from .text_normalize import tokenize as _tokenise

#: chromadb 0.5.23 logs a scary "Failed to send telemetry event" error on every
#: call even with telemetry disabled. It is harmless and it is the first thing
#: a user sees on a cold start, so it gets muted at the source.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

#: Chroma rejects oversized writes; stay well under the per-call ceiling.
_BATCH = 128

#: Dense search over-fetch factor, see the module docstring.
_OVERFETCH = 4

#: Guard the blend so the final score cannot leave [0, 1] — the routing
#: thresholds in config are expressed as similarities and would stop meaning
#: anything if a misconfigured weight let scores drift outside that range.
_DENSE_W = min(max(config.HYBRID_DENSE_WEIGHT, 0.0), 1.0)
_SPARSE_W = 1.0 - _DENSE_W

def _similarity(distance: Any) -> float:
    """Cosine distance -> similarity, given L2-normalised vectors."""
    return min(1.0, max(0.0, 1.0 - float(distance)))


def _batched(items: Sequence[Any], size: int = _BATCH) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class _Sparse:
    """A BM25 index plus the payloads needed to answer from it alone."""

    def __init__(self, keys: list[str], corpus: list[list[str]], payloads: dict[str, Any]):
        self.keys = keys
        self.corpus = corpus
        self.payloads = payloads
        self._key_index = {key: index for index, key in enumerate(keys)}
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def upsert(self, key: str, tokens: list[str], payload: Any) -> None:
        if key in self._key_index:
            index = self._key_index[key]
            self.corpus[index] = tokens
            self.payloads[key] = payload
        else:
            self._key_index[key] = len(self.keys)
            self.keys.append(key)
            self.corpus.append(tokens)
            self.payloads[key] = payload
        self._rebuild_bm25()

    def remove(self, key: str) -> None:
        if key not in self._key_index:
            return
        index = self._key_index.pop(key)
        del self.payloads[key]
        self.keys.pop(index)
        self.corpus.pop(index)
        self._key_index = {key: i for i, key in enumerate(self.keys)}
        self._rebuild_bm25()

    def remove_many(self, keys: Iterable[str]) -> None:
        drop = {key for key in keys if key in self._key_index}
        if not drop:
            return
        kept_keys: list[str] = []
        kept_corpus: list[list[str]] = []
        kept_payloads: dict[str, Any] = {}
        for key in self.keys:
            if key in drop:
                continue
            kept_keys.append(key)
            kept_corpus.append(self.corpus[self._key_index[key]])
            kept_payloads[key] = self.payloads[key]
        self.keys = kept_keys
        self.corpus = kept_corpus
        self.payloads = kept_payloads
        self._key_index = {key: index for index, key in enumerate(self.keys)}
        self._rebuild_bm25()

    def scores(self, query: str, limit: int) -> dict[str, float]:
        tokens = _tokenise(query)
        if self.bm25 is None or not tokens:
            return {}
        raw = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(raw)), key=lambda i: raw[i], reverse=True)[:limit]
        return {self.keys[i]: float(raw[i]) for i in ranked if raw[i] > 0.0}


def _fuse(dense: dict[str, float], sparse: dict[str, float]) -> list[tuple[str, float]]:
    """Blend dense and sparse scores over the union of both candidate sets.

    BM25 is unbounded, so it is min-maxed across the candidates actually under
    consideration before blending. A key missing from one side scores 0 there
    rather than being dropped.
    """
    top = max(sparse.values(), default=0.0)

    # A degenerate sparse side must NOT be blended in as a constant zero.
    # BM25 collapses to all-zeros whenever every candidate shares the query's
    # terms — routine on a small or homogeneous corpus, where IDF approaches
    # zero for exactly the words that matter. Blending then multiplies every
    # dense score by _DENSE_W, so a verbatim question match scoring 1.0 dense
    # arrives as 0.65 and lands *under* QUESTION_MATCH_THRESHOLD. That makes
    # the question-bank tier unreachable rather than merely worse-ranked, so
    # when sparse carries no signal we defer to dense alone.
    if top <= 0.0:
        return sorted(dense.items(), key=lambda kv: kv[1], reverse=True)

    # Mirror image: when the embedding endpoint is down, _dense_rows returns {}
    # so the search is meant to degrade to BM25 alone. Blending against an
    # absent dense side caps every score at _SPARSE_W (0.35), which is below
    # both routing thresholds — so the "degradation" would refuse every query,
    # including exact keyword matches. Rank on sparse alone instead.
    if not dense:
        return sorted(
            ((k, v / top) for k, v in sparse.items()),
            key=lambda kv: kv[1],
            reverse=True,
        )

    normalised = {k: v / top for k, v in sparse.items()}
    fused = {
        key: _DENSE_W * dense.get(key, 0.0) + _SPARSE_W * normalised.get(key, 0.0)
        for key in set(dense) | set(normalised)
    }

    # Hybrid ranking may REORDER candidates, but it must never score one lower
    # than dense retrieval alone would have. Without this floor a strong dense
    # hit that BM25 merely ranked poorly is scaled toward _DENSE_W * dense,
    # which can push a verbatim match under QUESTION_MATCH_THRESHOLD.
    #
    # The floor must cover EVERY dense candidate, not just those absent from
    # sparse: gating it on absence protects the candidates BM25 ignored while
    # punishing the ones BM25 weakly agreed with, so adding sparse evidence for
    # a document would lower its score.
    for key, score in dense.items():
        if score > fused.get(key, 0.0):
            fused[key] = score

    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class _FileLock:
    """Cross-process exclusive lock for Chroma and BM25 persistence."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "a+", encoding="utf-8")

    def acquire(self) -> None:
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)

    def release(self) -> None:
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)

    def close(self) -> None:
        self.release()
        self._handle.close()


def _sparse_path(kind: str, persist_dir: Path) -> Path:
    return persist_dir / f"bm25_{kind}.pkl"


def _sparse_meta_path(persist_dir: Path) -> Path:
    return persist_dir / "bm25_meta.json"


def _load_sparse_file(path: Path) -> _Sparse | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return _Sparse(payload["keys"], payload["corpus"], payload["payloads"])
    except Exception:  # noqa: BLE001 - corrupt cache falls back to a rebuild
        log.warning("could not load BM25 cache %s; will rebuild", path, exc_info=True)
        return None


def _save_sparse_file(path: Path, sparse: _Sparse) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(
            {"keys": sparse.keys, "corpus": sparse.corpus, "payloads": sparse.payloads},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp.replace(path)


def _read_sparse_meta(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {"questions": 0, "chunks": 0}
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            "questions": int(data.get("questions", 0)),
            "chunks": int(data.get("chunks", 0)),
        }
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return {"questions": 0, "chunks": 0}


def _write_sparse_meta(path: Path, questions: int, chunks: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump({"questions": questions, "chunks": chunks}, handle)
    tmp.replace(path)


class VectorStore:
    """Hybrid persistent store for chunks and extracted question/answer pairs."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        client: OllamaClient | None = None,
    ) -> None:
        self._persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self._ollama = client or get_client()
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._questions = self._collection(config.QUESTION_COLLECTION)
        self._chunks = self._collection(config.CHUNK_COLLECTION)

        self._q_sparse: _Sparse | None = None
        self._c_sparse: _Sparse | None = None
        self._q_sparse_mtime: float = 0.0
        self._c_sparse_mtime: float = 0.0
        self._sparse_rev = _read_sparse_meta(_sparse_meta_path(config.CACHE_DIR))
        self._thread_lock = threading.RLock()
        self._file_lock = _FileLock(config.CACHE_DIR / ".store.lock")

        self._load_sparse_cache("questions")
        self._load_sparse_cache("chunks")

    def _collection(self, name: str):
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    @contextmanager
    def _store_lock(self):
        """Serialise Chroma writes and BM25 persistence across threads/processes."""
        with self._thread_lock:
            self._file_lock.acquire()
            try:
                self._reload_sparse_if_stale()
                yield
            finally:
                self._file_lock.release()

    def _sparse_cache_path(self, kind: str) -> Path:
        return _sparse_path(kind, config.CACHE_DIR)

    def _load_sparse_cache(self, kind: str) -> None:
        path = self._sparse_cache_path(kind)
        sparse = _load_sparse_file(path)
        if sparse is None:
            return
        mtime = path.stat().st_mtime
        if kind == "questions":
            self._q_sparse = sparse
            self._q_sparse_mtime = mtime
        else:
            self._c_sparse = sparse
            self._c_sparse_mtime = mtime

    def _reload_sparse_if_stale(self) -> None:
        for kind, mtime_attr, sparse_attr in (
            ("questions", "_q_sparse_mtime", "_q_sparse"),
            ("chunks", "_c_sparse_mtime", "_c_sparse"),
        ):
            path = self._sparse_cache_path(kind)
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
            if mtime <= getattr(self, mtime_attr):
                continue
            loaded = _load_sparse_file(path)
            if loaded is not None:
                setattr(self, sparse_attr, loaded)
                setattr(self, mtime_attr, mtime)
        self._sparse_rev = _read_sparse_meta(_sparse_meta_path(config.CACHE_DIR))

    def _persist_sparse(self, kind: str) -> None:
        sparse = self._q_sparse if kind == "questions" else self._c_sparse
        path = self._sparse_cache_path(kind)
        if sparse is None:
            if path.is_file():
                path.unlink(missing_ok=True)
            if kind == "questions":
                self._q_sparse_mtime = 0.0
            else:
                self._c_sparse_mtime = 0.0
        else:
            _save_sparse_file(path, sparse)
            mtime = path.stat().st_mtime
            if kind == "questions":
                self._q_sparse_mtime = mtime
            else:
                self._c_sparse_mtime = mtime
        rev = self._sparse_rev.get(kind, 0) + 1
        self._sparse_rev[kind] = rev
        _write_sparse_meta(
            _sparse_meta_path(config.CACHE_DIR),
            self._sparse_rev.get("questions", 0),
            self._sparse_rev.get("chunks", 0),
        )

    def _touch_questions_sparse(self, pairs: list[QAPair]) -> None:
        if self._q_sparse is None:
            self._q_sparse = _Sparse([], [], {})
        for qa in pairs:
            meta = qa.to_metadata()
            self._q_sparse.upsert(
                qa.qa_id,
                _tokenise(" ".join([qa.question, *qa.paraphrases, *qa.keywords])),
                meta,
            )

    def _touch_chunks_sparse(self, chunks: list[Chunk]) -> None:
        if self._c_sparse is None:
            self._c_sparse = _Sparse([], [], {})
        for chunk in chunks:
            meta = chunk.to_metadata()
            self._c_sparse.upsert(
                chunk.chunk_id,
                _tokenise(chunk.text),
                (meta, chunk.text),
            )

    def _drop_document_sparse(self, doc_id: str) -> None:
        if self._c_sparse is not None:
            drop = [
                key
                for key, (meta, _text) in self._c_sparse.payloads.items()
                if (meta or {}).get("doc_id") == doc_id
            ]
            self._c_sparse.remove_many(drop)
        if self._q_sparse is not None:
            drop = [
                key
                for key, meta in self._q_sparse.payloads.items()
                if (meta or {}).get("doc_id") == doc_id
            ]
            self._q_sparse.remove_many(drop)

    def _clear_sparse(self) -> None:
        self._q_sparse = None
        self._c_sparse = None
        self._q_sparse_mtime = 0.0
        self._c_sparse_mtime = 0.0
        self._sparse_rev = {"questions": 0, "chunks": 0}
        for kind in ("questions", "chunks"):
            path = self._sparse_cache_path(kind)
            path.unlink(missing_ok=True)
        _write_sparse_meta(_sparse_meta_path(config.CACHE_DIR), 0, 0)

    # -- ingestion ---------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        vectors = self._ollama.embed(
            [c.text for c in chunks], batch_size=config.EMBED_BATCH_SIZE
        )
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.to_metadata() for c in chunks]

        with self._store_lock():
            for lo in range(0, len(ids), _BATCH):
                hi = lo + _BATCH
                self._chunks.upsert(
                    ids=ids[lo:hi],
                    embeddings=vectors[lo:hi],
                    documents=docs[lo:hi],
                    metadatas=metas[lo:hi],
                )
            self._touch_chunks_sparse(chunks)
            self._persist_sparse("chunks")

        return len(chunks)

    def add_qa_pairs(self, pairs: list[QAPair]) -> int:
        if not pairs:
            return 0

        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict[str, Any]] = []

        for qa in pairs:
            base = qa.to_metadata()
            ids.append(f"{qa.qa_id}::q")
            texts.append(qa.question)
            metas.append({**base, "row_kind": "question"})

            for i, paraphrase in enumerate(qa.paraphrases):
                if not paraphrase or not paraphrase.strip():
                    continue
                ids.append(f"{qa.qa_id}::p{i}")
                texts.append(paraphrase)
                metas.append({**base, "row_kind": "paraphrase"})

        vectors = self._ollama.embed(texts, batch_size=config.EMBED_BATCH_SIZE)

        with self._store_lock():
            qa_ids = [qa.qa_id for qa in pairs]
            for batch in _batched(qa_ids):
                self._questions.delete(where={"qa_id": {"$in": list(batch)}})

            for lo in range(0, len(ids), _BATCH):
                hi = lo + _BATCH
                self._questions.upsert(
                    ids=ids[lo:hi],
                    embeddings=vectors[lo:hi],
                    documents=texts[lo:hi],
                    metadatas=metas[lo:hi],
                )
            self._touch_questions_sparse(pairs)
            self._persist_sparse("questions")

        return len(pairs)

    # -- sparse indexes ----------------------------------------------------

    def _question_sparse(self) -> _Sparse:
        self._reload_sparse_if_stale()
        if self._q_sparse is not None:
            return self._q_sparse

        rows = self._questions.get(include=["metadatas"])
        keys: list[str] = []
        corpus: list[list[str]] = []
        payloads: dict[str, dict[str, Any]] = {}

        for meta in rows.get("metadatas") or []:
            qa_id = (meta or {}).get("qa_id")
            if not qa_id or qa_id in payloads:
                continue
            qa = QAPair.from_metadata(meta)
            payloads[qa_id] = meta
            keys.append(qa_id)
            corpus.append(
                _tokenise(" ".join([qa.question, *qa.paraphrases, *qa.keywords]))
            )

        self._q_sparse = _Sparse(keys, corpus, payloads)
        self._persist_sparse("questions")
        return self._q_sparse

    def _chunk_sparse(self) -> _Sparse:
        self._reload_sparse_if_stale()
        if self._c_sparse is not None:
            return self._c_sparse

        rows = self._chunks.get(include=["metadatas", "documents"])
        ids = rows.get("ids") or []
        metas = rows.get("metadatas") or []
        docs = rows.get("documents") or []

        keys: list[str] = []
        corpus: list[list[str]] = []
        payloads: dict[str, tuple[dict[str, Any], str]] = {}

        for chunk_id, meta, text in zip(ids, metas, docs):
            payloads[chunk_id] = (meta or {}, text or "")
            keys.append(chunk_id)
            corpus.append(_tokenise(text or ""))

        self._c_sparse = _Sparse(keys, corpus, payloads)
        self._persist_sparse("chunks")
        return self._c_sparse

    # -- search ------------------------------------------------------------

    def search_questions(self, query: str, top_k: int | None = None) -> list[ScoredQA]:
        k = top_k or config.QUESTION_TOP_K
        if k <= 0 or self._questions.count() == 0 or not query.strip():
            return []

        dense: dict[str, float] = {}
        matched_on: dict[str, str] = {}
        metas: dict[str, dict[str, Any]] = {}

        for meta, score in self._dense_rows(self._questions, query, k * _OVERFETCH):
            qa_id = meta.get("qa_id")
            if not qa_id:
                continue
            # Collapse the canonical row and its paraphrases to their best hit.
            if score > dense.get(qa_id, -1.0):
                dense[qa_id] = score
                matched_on[qa_id] = meta.get("row_kind", "question")
                metas[qa_id] = meta

        sparse_index = self._question_sparse()
        sparse = sparse_index.scores(query, k * _OVERFETCH)
        metas.update(
            {qa_id: sparse_index.payloads[qa_id] for qa_id in sparse if qa_id not in metas}
        )

        results: list[ScoredQA] = []
        for qa_id, score in _fuse(dense, sparse)[:k]:
            meta = metas.get(qa_id)
            if meta is None:
                continue
            results.append(
                ScoredQA(
                    qa=QAPair.from_metadata(meta),
                    score=round(score, 6),
                    # A BM25-only hit has no winning row; credit the canonical.
                    matched_on=matched_on.get(qa_id, "question"),
                )
            )
        return results

    def search_chunks(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or config.VECTOR_TOP_K
        if k <= 0 or self._chunks.count() == 0 or not query.strip():
            return []

        dense: dict[str, float] = {}
        payloads: dict[str, tuple[dict[str, Any], str]] = {}

        for meta, score, text in self._dense_rows(
            self._chunks, query, k * _OVERFETCH, with_text=True
        ):
            chunk_id = meta.get("chunk_id")
            if not chunk_id:
                continue
            # Parents are context-only; retrieve on children / legacy leaves.
            if (meta.get("chunk_kind") or "child") == "parent":
                continue
            dense[chunk_id] = score
            payloads[chunk_id] = (meta, text)

        sparse_index = self._chunk_sparse()
        sparse = sparse_index.scores(query, k * _OVERFETCH)
        for cid in sparse:
            if cid in payloads:
                continue
            entry = sparse_index.payloads.get(cid)
            if entry is None:
                continue
            meta, _text = entry
            if (meta.get("chunk_kind") or "child") == "parent":
                continue
            payloads[cid] = entry

        results: list[ScoredChunk] = []
        for chunk_id, score in _fuse(dense, sparse)[:k]:
            entry = payloads.get(chunk_id)
            if entry is None:
                continue
            meta, text = entry
            chunk = _chunk_from(meta, text)
            # Expand child hit to parent window text for richer answering.
            if chunk.parent_id:
                parent_text = self.chunk_texts([chunk.parent_id]).get(chunk.parent_id)
                if parent_text:
                    chunk = Chunk(
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        doc_name=chunk.doc_name,
                        text=parent_text,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        section=chunk.section,
                        ordinal=chunk.ordinal,
                        parent_id=chunk.parent_id,
                        chunk_kind=chunk.chunk_kind,
                    )
            results.append(ScoredChunk(chunk=chunk, score=round(score, 6)))
        return results

    def _dense_rows(
        self, collection, query: str, n: int, with_text: bool = False
    ) -> list[tuple[dict[str, Any], float, str]] | list[tuple[dict[str, Any], float]]:
        """Embed the query and pull raw rows; empty on embedding failure.

        A dead or flaky embedding endpoint degrades the search to BM25-only
        rather than taking down the whole UI.
        """
        try:
            vector = self._ollama.embed_one(query)
        except OllamaError:
            return []

        # Asking for more rows than exist makes Chroma log a warning on every
        # query against a small store; clamp instead.
        response = collection.query(
            query_embeddings=[vector],
            n_results=max(1, min(n, collection.count())),
            include=["metadatas", "documents", "distances"],
        )
        metas = (response.get("metadatas") or [[]])[0]
        dists = (response.get("distances") or [[]])[0]
        docs = (response.get("documents") or [[]])[0]

        if with_text:
            return [
                (m or {}, _similarity(d), t or "")
                for m, d, t in zip(metas, dists, docs)
            ]
        return [(m or {}, _similarity(d)) for m, d in zip(metas, dists)]

    # -- inspection --------------------------------------------------------

    def chunk_texts(self, chunk_ids: list[str]) -> dict[str, str]:
        """Fetch raw chunk text by id.

        The groundedness audit needs it: a QA pair's evidence quotes are
        fragments, and a fragment can omit the very word that makes a claim
        checkable ("entitled to 30 days per year" without "of leave"). The
        chunk is the surrounding document truth that restores that context —
        and unlike the pair's stored answer, it was never model-generated.
        """
        wanted = [c for c in dict.fromkeys(chunk_ids) if c]
        if not wanted:
            return {}

        found: dict[str, str] = {}
        for batch in _batched(wanted):
            rows = self._chunks.get(
                where={"chunk_id": {"$in": list(batch)}},
                include=["documents", "metadatas"],
            )
            for text, meta in zip(
                rows.get("documents") or [], rows.get("metadatas") or []
            ):
                if meta and meta.get("chunk_id"):
                    found[meta["chunk_id"]] = text or ""
        return found

    def all_qa_pairs(self) -> list[QAPair]:
        rows = self._questions.get(include=["metadatas"])
        seen: dict[str, QAPair] = {}
        for meta in rows.get("metadatas") or []:
            qa_id = (meta or {}).get("qa_id")
            if qa_id and qa_id not in seen:
                seen[qa_id] = QAPair.from_metadata(meta)
        return list(seen.values())

    def qa_bank_summary(self, doc_id: str | None = None) -> dict[str, Any]:
        """Lightweight counts for the UI without loading full answer bodies."""
        rows = self._questions.get(include=["metadatas"])
        by_type: dict[str, int] = {}
        by_difficulty: dict[str, int] = {}
        seen: set[str] = set()
        total = 0
        hard = 0
        for meta in rows.get("metadatas") or []:
            meta = meta or {}
            qa_id = meta.get("qa_id")
            if not qa_id or qa_id in seen:
                continue
            if doc_id and meta.get("doc_id") != doc_id:
                continue
            seen.add(qa_id)
            total += 1
            qtype = meta.get("question_type", "")
            level = meta.get("difficulty", "")
            by_type[qtype] = by_type.get(qtype, 0) + 1
            by_difficulty[level] = by_difficulty.get(level, 0) + 1
            if qtype in ("multi_hop", "critical", "application"):
                hard += 1
        return {
            "total": total,
            "by_type": by_type,
            "by_difficulty": by_difficulty,
            "hard": hard,
            "type_count": len(by_type),
        }

    def query_qa_pairs(
        self,
        *,
        doc_id: str | None = None,
        search: str = "",
        types: Sequence[str] | None = None,
        difficulties: Sequence[str] | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[QAPair], int]:
        """Filter and paginate question-bank entries server-side."""
        pairs = self.all_qa_pairs()
        if doc_id:
            pairs = [pair for pair in pairs if pair.doc_id == doc_id]

        type_set = set(types or [])
        level_set = set(difficulties or [])
        needle = search.strip().casefold()

        filtered: list[QAPair] = []
        for pair in pairs:
            if type_set and pair.question_type not in type_set:
                continue
            if level_set and pair.difficulty not in level_set:
                continue
            if needle and not (
                needle in pair.question.casefold()
                or needle in pair.answer.casefold()
                or any(needle in keyword.casefold() for keyword in pair.keywords)
            ):
                continue
            filtered.append(pair)

        filtered.sort(
            key=lambda pair: (
                pair.doc_name.casefold(),
                pair.page_start,
                pair.question.casefold(),
            )
        )
        start = max(0, offset)
        end = start + max(1, limit)
        return filtered[start:end], len(filtered)

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict[str, Any]] = {}

        def _slot(meta: dict[str, Any]) -> dict[str, Any] | None:
            doc_id = meta.get("doc_id")
            if not doc_id:
                return None
            return docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "doc_name": meta.get("doc_name", ""),
                    "chunk_count": 0,
                    "qa_count": 0,
                },
            )

        for meta in self._chunks.get(include=["metadatas"]).get("metadatas") or []:
            slot = _slot(meta or {})
            if slot is not None:
                slot["chunk_count"] += 1

        # Question rows are per-paraphrase, so count distinct qa_ids.
        counted: set[str] = set()
        for meta in self._questions.get(include=["metadatas"]).get("metadatas") or []:
            meta = meta or {}
            qa_id = meta.get("qa_id")
            if not qa_id or qa_id in counted:
                continue
            counted.add(qa_id)
            slot = _slot(meta)
            if slot is not None:
                slot["qa_count"] += 1

        return sorted(docs.values(), key=lambda d: d["doc_name"].casefold())

    def has_document(self, doc_id: str) -> bool:
        if not doc_id:
            return False
        for collection in (self._chunks, self._questions):
            if collection.get(where={"doc_id": doc_id}, limit=1).get("ids"):
                return True
        return False

    def stats(self) -> dict:
        documents = self.list_documents()
        return {
            "n_documents": len(documents),
            "n_chunks": sum(d["chunk_count"] for d in documents),
            "n_questions": sum(d["qa_count"] for d in documents),
        }

    # -- mutation ----------------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        if not doc_id:
            raise ValueError("delete_document requires a doc_id")
        with self._store_lock():
            for collection in (self._chunks, self._questions):
                collection.delete(where={"doc_id": doc_id})
            self._drop_document_sparse(doc_id)
            self._persist_sparse("questions")
            self._persist_sparse("chunks")

    def delete_qa_pair(self, qa_id: str) -> None:
        """Drop one question-bank entry and all of its paraphrase rows."""
        if not qa_id:
            raise ValueError("delete_qa_pair requires a qa_id")
        with self._store_lock():
            self._questions.delete(where={"qa_id": qa_id})
            if self._q_sparse is not None:
                self._q_sparse.remove(qa_id)
            self._persist_sparse("questions")

    def reset(self) -> None:
        with self._store_lock():
            for name in (config.QUESTION_COLLECTION, config.CHUNK_COLLECTION):
                try:
                    self._client.delete_collection(name)
                except Exception:  # noqa: BLE001 - absent collection is not an error
                    pass
            self._questions = self._collection(config.QUESTION_COLLECTION)
            self._chunks = self._collection(config.CHUNK_COLLECTION)
            self._clear_sparse()


def _chunk_from(meta: dict[str, Any], text: str) -> Chunk:
    kind = meta.get("chunk_kind") or "child"
    return Chunk(
        chunk_id=meta.get("chunk_id", ""),
        doc_id=meta.get("doc_id", ""),
        doc_name=meta.get("doc_name", ""),
        text=text,
        page_start=int(meta.get("page_start", 0) or 0),
        page_end=int(meta.get("page_end", 0) or 0),
        section=meta.get("section", ""),
        ordinal=int(meta.get("ordinal", 0) or 0),
        parent_id=meta.get("parent_id", "") or "",
        chunk_kind=kind,
    )


__all__ = ["VectorStore"]
