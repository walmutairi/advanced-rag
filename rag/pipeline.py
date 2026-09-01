"""The single façade the UI talks to.

Everything the Streamlit app needs is one method on :class:`RAGPipeline`, so
the UI never has to know that ingestion is five stages or that answering is a
router plus a synthesiser. Two behaviours are worth stating outright:

* **Ingestion is idempotent by content.** ``doc_id`` comes from the file's
  content hash, so re-uploading the same PDF is skipped rather than re-run.
  That is the whole point of the content hash: question extraction on a 31B
  model costs tens of minutes, and it must never be paid twice by accident.
* **Search never raises.** A failure anywhere in routing or synthesis comes
  back as an unanswered :class:`AnswerResult` carrying the error in ``notes``.
  A traceback in a search box is a broken app; a refusal with an explanation
  is a working one.

The progress budget is deliberately lopsided. Question extraction is one LLM
call per chunk and dominates wall-clock by an order of magnitude, so it owns
the middle 75% of the bar — a bar that races to 90% and then sits still for
half an hour is worse than no bar at all.

OCR bends that budget. It costs seconds per page and runs inside ``load_pdf``,
before chunking, so a scanned document would otherwise sit at 0% for minutes.
The budget is therefore chosen *after* the load, once the load report says
whether OCR actually ran — safe only because the wide OCR slice is drawn on
exclusively when OCR fires, so the plain budget never has to reserve room for
work that never happens.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, Iterator, Sequence

import config
from rag.answerer import synthesize, synthesize_stream
from rag.chunker import chunk_pages
from rag.ollama_client import OllamaClient, OllamaError, get_client, get_extract_client
from rag.pdf_loader import analyze_pdf, last_load_report, load_pdf, page_count
from rag.qa_extractor import extract_qa_batch, extraction_report, last_batch_stats
from rag.retrieval import route_query
from rag.schemas import (
    MANUAL_DOC_ID,
    MANUAL_DOC_NAME,
    ROUTE_REFUSED,
    AnswerResult,
    QAPair,
    _stable_id,
)
from rag.vectorstore import VectorStore

log = logging.getLogger(__name__)


class IngestCancelled(Exception):
    """Raised when the user aborts a long-running ingest."""


def _check_cancel(cancel_event: Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise IngestCancelled("Ingest cancelled.")

#: ``progress_cb(fraction, message)`` — fraction is monotonic in [0, 1].
ProgressCallback = Callable[[float, str], None]

#: Where an OCR pass draws, when there is one. Nothing else may use this
#: slice: it stays blank on documents that never touch OCR.
_OCR_LOW = 0.02
_OCR_HIGH = 0.30

#: Stage boundaries within one document's share of the bar, as
#: ``(loaded, chunked, embedded, extracted)``. Chosen so the cheap stages
#: cannot visually dominate the expensive one.
#: Extraction ends at 0.98 so a final indexing tick is still visible after
#: the last chunk / optional cross-chunk pass (previously the bar sat at 95%).
_BUDGET_PLAIN = (0.06, 0.10, 0.20, 0.98)

#: The OCR variant: loading ends where the OCR slice does. Extraction still
#: owns the clear majority, because it is still the slower half.
_BUDGET_OCR = (_OCR_HIGH, 0.33, 0.40, 0.98)


class RAGPipeline:
    """Ingestion, retrieval and answering behind one object."""

    def __init__(self, persist_dir: Path | str | None = None) -> None:
        self.client: OllamaClient = get_client()
        self.extract_client: OllamaClient = get_extract_client()
        self.store = VectorStore(persist_dir=persist_dir, client=self.client)

    # -- diagnostics -------------------------------------------------------

    def health(self) -> tuple[bool, str]:
        """Is the system usable right now, and if not, what is wrong?

        The store is checked as well as Ollama: a corrupt Chroma directory
        fails at query time, long after the health badge said everything was
        fine.
        """
        ok, message = self.client.health()
        if not ok:
            return False, message

        if self.extract_client is not self.client:
            ok, extract_msg = self.extract_client.health()
            if not ok:
                return False, f"Extract model unavailable: {extract_msg}"

        try:
            counts = self.store.stats()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            return False, f"Vector store unavailable: {exc}"

        return True, (
            f"{message} — {counts['n_documents']} document(s), "
            f"{counts['n_chunks']} chunks, {counts['n_questions']} questions"
        )

    def ocr_status(self) -> tuple[bool, str]:
        """Whether OCR is available and a one-line reason either way."""
        from rag import ocr

        return ocr.is_available(), ocr.ocr_available_reason()

    # -- ingestion ---------------------------------------------------------

    def ingest_pdf(
        self,
        path: Path | str,
        progress_cb: ProgressCallback | None = None,
        force: bool = False,
        force_ocr: bool = False,
        cancel_event: Event | None = None,
    ) -> dict:
        """Ingest one PDF end to end and return its stats.

        ``force_ocr`` sends every page to OCR regardless of how good its text
        layer looks. It is ignored, with a note in the load report, when OCR is
        unavailable — it can never turn a working ingest into a failed one.

        Raises ``ValueError`` for a file that cannot be read as a PDF — that is
        a fact about the input the caller has to hear. Model flakiness during
        question extraction is not fatal: failed chunks are counted into
        ``failures`` and the rest of the document still lands.
        """
        return self._ingest_one(
            path, _Progress(progress_cb, 0.0, 1.0), force, force_ocr, cancel_event
        )

    def ingest_pdfs(
        self,
        paths: Sequence[Path | str] | Iterable[Path | str],
        progress_cb: ProgressCallback | None = None,
        force: bool = False,
        force_ocr: bool = False,
        cancel_event: Event | None = None,
    ) -> dict:
        """Ingest several PDFs, giving each an equal share of the bar.

        Unlike :meth:`ingest_pdf` this never raises for a bad file: one
        unreadable PDF in a ten-file upload records an error and the batch
        carries on.

        Note the shape change across the two levels: each document's
        ``ocr_pages`` is the *list* of page numbers OCR'd, while the batch's
        ``ocr_pages`` is the *total count* across documents.
        """
        items = list(paths)
        if not items:
            return _empty_batch()

        documents: list[dict] = []
        errors: list[dict] = []
        started = time.perf_counter()
        share = 1.0 / len(items)

        for i, path in enumerate(items):
            _check_cancel(cancel_event)
            window = _Progress(progress_cb, i * share, (i + 1) * share)
            try:
                documents.append(
                    self._ingest_one(path, window, force, force_ocr, cancel_event)
                )
            except IngestCancelled:
                raise
            except (ValueError, OllamaError) as exc:
                name = Path(path).name
                log.warning("ingest failed for %s: %s", name, exc)
                errors.append({"path": str(path), "doc_name": name, "error": str(exc)})
                window.emit(1.0, f"{name}: failed — {exc}")

        batch = {
            "documents": documents,
            "errors": errors,
            "ingested": sum(1 for d in documents if not d["skipped"]),
            "skipped": sum(1 for d in documents if d["skipped"]),
            "failed": len(errors),
            "pages": sum(d["pages"] for d in documents),
            "chunks": sum(d["chunks"] for d in documents),
            "questions": sum(d["questions"] for d in documents),
            "failures": sum(d["failures"] for d in documents),
            # A count, not a list — per-document ocr_pages are page numbers and
            # would collide across documents if merged.
            "ocr_pages": sum(len(d["ocr_pages"]) for d in documents),
            "elapsed": round(time.perf_counter() - started, 2),
        }
        try:
            from rag.metrics import record_ingest

            record_ingest(
                ocr_pages=batch["ocr_pages"],
                questions=batch["questions"],
                chunks=batch["chunks"],
                elapsed=batch["elapsed"],
            )
        except Exception:
            log.debug("metrics record_ingest failed", exc_info=True)

        if config.ENABLE_POST_INGEST_EVAL and batch["ingested"]:
            try:
                from eval.runner import run_eval

                batch["eval"] = run_eval(self, route_only=True)
            except Exception as exc:
                log.warning("post-ingest eval failed: %s", exc)
                batch["eval"] = {"error": str(exc)}
        return batch

    def _ingest_one(
        self,
        path: Path | str,
        progress: _Progress,
        force: bool,
        force_ocr: bool,
        cancel_event: Event | None = None,
    ) -> dict:
        started = time.perf_counter()
        source = Path(path)

        progress.emit(0.0, f"Reading {source.name} ...")
        _check_cancel(cancel_event)

        try:
            pdf_bytes = source.read_bytes()
        except OSError as exc:
            raise ValueError(f"Could not read {source.name}: {exc}") from exc

        known_id = _stable_id(hashlib.sha256(pdf_bytes).hexdigest())
        if known_id and not force and self.store.has_document(known_id):
            progress.emit(1.0, f"{source.name}: already indexed — skipped")
            return _doc_stats(
                known_id, source.name, source, page_count(source, pdf_bytes=pdf_bytes),
                0, 0, 0, started, ocr_pages=[], skipped=True,
            )

        doc_id, doc_name, pages = load_pdf(
            source,
            pdf_bytes=pdf_bytes,
            force_ocr=force_ocr,
            progress_cb=progress.ocr_mapper(_OCR_LOW, _OCR_HIGH, source.name),
        )

        # What load_pdf actually did, rather than what config suggested it
        # might: OCR silently no-ops when surya is missing or disabled.
        ocr_pages = list(last_load_report(doc_id).get("ocr_pages") or [])
        stage_loaded, stage_chunked, stage_embedded, stage_extracted = (
            _BUDGET_OCR if ocr_pages else _BUDGET_PLAIN
        )

        # The content hash is only worth carrying if it actually saves the
        # extraction run, so the skip check happens before any model call.
        if self.store.has_document(doc_id):
            if not force:
                progress.emit(1.0, f"{doc_name}: already indexed — skipped")
                return _doc_stats(
                    doc_id, doc_name, source, len(pages), 0, 0, 0,
                    started, ocr_pages=ocr_pages, skipped=True,
                )
            # Upserting alone cannot clear a forced re-ingest: chunk ids are
            # derived from the text, so any shift leaves the previous run's
            # chunks and their QA pairs behind as unreachable orphans.
            self.store.delete_document(doc_id)

        ocr_note = f" ({len(ocr_pages)} OCR'd)" if ocr_pages else ""
        progress.emit(
            stage_loaded, f"{doc_name}: {len(pages)} pages{ocr_note} — chunking ..."
        )
        _check_cancel(cancel_event)
        chunks = chunk_pages(doc_id, doc_name, pages)
        if not chunks:
            raise ValueError(
                f"{doc_name} produced no chunks: the extracted text was empty "
                "after cleaning. It may be a scan or a form with no body text."
            )

        progress.emit(
            stage_chunked,
            f"{doc_name}: embedding {len(chunks)} chunks ...",
        )
        self.store.add_chunks(chunks)

        progress.emit(
            stage_embedded,
            f"{doc_name}: extracting questions from {len(chunks)} chunks ...",
        )
        pairs = extract_qa_batch(
            chunks,
            client=self.extract_client,
            progress_cb=progress.stage_mapper(
                stage_embedded, stage_extracted, doc_name
            ),
            cancel_event=cancel_event,
        )
        _check_cancel(cancel_event)
        batch = last_batch_stats()
        failures = len(batch.failures)
        load_report = last_load_report(doc_id)

        progress.emit(
            stage_extracted,
            f"{doc_name}: indexing {len(pairs)} questions ...",
        )
        self.store.add_qa_pairs(pairs)

        progress.emit(
            1.0,
            f"{doc_name}: done — {len(chunks)} chunks, {len(pairs)} questions",
        )
        coverage = {
            "chunks_total": batch.chunks_total,
            "chunks_succeeded": batch.chunks_succeeded,
            "chunks_failed": batch.chunks_failed,
            "chunks_empty": batch.chunks_empty,
            "pairs_kept": batch.pairs_kept,
            "pairs_duplicate": batch.pairs_duplicate,
            "empty_pages": list(load_report.get("empty_pages") or []),
            "degraded_pages": list(load_report.get("degraded_pages") or []),
            "text_previews": list(load_report.get("text_previews") or []),
            "failure_samples": [
                {"chunk_id": cid, "error": err} for cid, err in batch.failures[:8]
            ],
            "questions_per_chunk": round(
                (batch.pairs_kept / batch.chunks_succeeded)
                if batch.chunks_succeeded
                else 0.0,
                2,
            ),
        }
        return _doc_stats(
            doc_id, doc_name, source, len(pages), len(chunks), len(pairs),
            failures, started, ocr_pages=ocr_pages, skipped=False,
            coverage=coverage,
        )

    # -- search ------------------------------------------------------------

    def search(self, query: str) -> AnswerResult:
        """Answer one query. Never raises — failures come back as a refusal."""
        started = time.perf_counter()
        try:
            outcome = route_query(query, self.store, client=self.client)
            result = synthesize(outcome, client=self.client, store=self.store)
        except Exception as exc:  # noqa: BLE001 - a search box must not traceback
            log.exception("search failed for %r", query)
            result = AnswerResult(
                query=query,
                answer=config.REFUSAL_MESSAGE,
                route=ROUTE_REFUSED,
                answered=False,
                notes=[f"The search failed: {type(exc).__name__}: {exc}"],
            )

        # synthesize() times only generation; the user waited for routing too.
        result.elapsed_seconds = time.perf_counter() - started
        try:
            from rag.metrics import record_search

            record_search(
                route=result.route,
                answered=result.answered,
                confidence=result.confidence,
                groundedness=result.groundedness,
                elapsed=result.elapsed_seconds,
            )
        except Exception:
            log.debug("metrics record_search failed", exc_info=True)
        return result

    def search_stream(self, query: str) -> Iterator[str | AnswerResult]:
        """Answer with streamed tokens. The final yield is always an AnswerResult."""
        started = time.perf_counter()
        try:
            outcome = route_query(query, self.store, client=self.client)
            for piece in synthesize_stream(outcome, client=self.client, store=self.store):
                if isinstance(piece, AnswerResult):
                    piece.elapsed_seconds = time.perf_counter() - started
                    try:
                        from rag.metrics import record_search

                        record_search(
                            route=piece.route,
                            answered=piece.answered,
                            confidence=piece.confidence,
                            groundedness=piece.groundedness,
                            elapsed=piece.elapsed_seconds,
                        )
                    except Exception:
                        log.debug("metrics record_search failed", exc_info=True)
                    yield piece
                else:
                    yield piece
        except Exception as exc:  # noqa: BLE001
            log.exception("search_stream failed for %r", query)
            yield AnswerResult(
                query=query,
                answer=config.REFUSAL_MESSAGE,
                route=ROUTE_REFUSED,
                answered=False,
                notes=[f"The search failed: {type(exc).__name__}: {exc}"],
                elapsed_seconds=time.perf_counter() - started,
            )

    # -- inspection --------------------------------------------------------

    def analyze(
        self,
        path: Path | str | None = None,
        *,
        pdf_bytes: bytes | None = None,
        force_ocr: bool = False,
    ) -> dict:
        """Report what an ingest would do to this PDF, without doing it."""
        return analyze_pdf(path, pdf_bytes=pdf_bytes, force_ocr=force_ocr)

    def documents(self) -> list[dict]:
        return self.store.list_documents()

    def question_bank(self, doc_id: str | None = None) -> list[QAPair]:
        """Every extracted pair, optionally narrowed to one document."""
        pairs = self.store.all_qa_pairs()
        if doc_id:
            pairs = [p for p in pairs if p.doc_id == doc_id]
        return sorted(
            pairs,
            key=lambda p: (p.doc_name.casefold(), p.page_start, p.question.casefold()),
        )

    def question_bank_summary(self, doc_id: str | None = None) -> dict:
        """Counts and distribution without loading full answer bodies."""
        return self.store.qa_bank_summary(doc_id)

    def question_bank_page(
        self,
        *,
        doc_id: str | None = None,
        search: str = "",
        types: Sequence[str] | None = None,
        difficulties: Sequence[str] | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> tuple[list[QAPair], int]:
        """One page of filtered question-bank entries plus the filtered total."""
        page_size = limit or config.QUESTION_BANK_PAGE_SIZE
        return self.store.query_qa_pairs(
            doc_id=doc_id,
            search=search,
            types=types,
            difficulties=difficulties,
            offset=offset,
            limit=page_size,
        )

    def export_question_bank(
        self,
        *,
        doc_id: str | None = None,
        search: str = "",
        types: Sequence[str] | None = None,
        difficulties: Sequence[str] | None = None,
    ) -> list[QAPair]:
        """All pairs matching the filters — for JSON export."""
        pairs, _total = self.store.query_qa_pairs(
            doc_id=doc_id,
            search=search,
            types=types,
            difficulties=difficulties,
            offset=0,
            limit=10_000_000,
        )
        return pairs

    def stats(self) -> dict:
        """Store counts plus the shape of the question bank, for the UI."""
        counts = self.store.stats()
        try:
            from rag.metrics import summary as metrics_summary

            observability = metrics_summary()
        except Exception:
            observability = {}
        return {
            **counts,
            "documents": self.store.list_documents(),
            "question_bank": extraction_report(self.store.all_qa_pairs()),
            "extract_model": self.extract_client.model,
            "answer_model": self.client.model,
            "llm_model": self.client.model,
            "embed_model": self.client.embed_model,
            "persist_dir": str(config.CHROMA_DIR),
            "observability": observability,
        }

    # -- mutation ----------------------------------------------------------

    def delete_document(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)

    def upsert_qa_pair(
        self,
        *,
        question: str,
        answer: str,
        question_type: str,
        difficulty: str,
        qa_id: str | None = None,
        evidence: Iterable[str] | None = None,
        paraphrases: Iterable[str] | None = None,
        keywords: Iterable[str] | None = None,
        doc_id: str = "",
        doc_name: str = "",
        page_start: int = 0,
        page_end: int = 0,
        section: str = "",
    ) -> QAPair:
        """Create or replace one question-bank entry.

        The store already upserts by ``qa_id`` (purging stale paraphrase rows
        first), so this one method covers both adding a brand-new question and
        editing an existing one: pass no ``qa_id`` to mint a new entry, or the
        existing one to edit in place. A question with no owning document is
        filed under the synthetic ``MANUAL_DOC_ID`` so it still counts and can
        be managed like any extracted pair.
        """
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question:
            raise ValueError("A question is required.")
        if not answer:
            raise ValueError("An answer is required.")

        if not doc_id:
            doc_id = MANUAL_DOC_ID
            doc_name = doc_name or MANUAL_DOC_NAME

        def _clean(values: Iterable[str] | None) -> list[str]:
            return [v.strip() for v in (values or []) if v and v.strip()]

        qa = QAPair(
            qa_id=qa_id or _stable_id("manual", doc_id, question),
            question=question,
            answer=answer,
            question_type=question_type or "factual",
            difficulty=difficulty or "basic",
            evidence=_clean(evidence),
            paraphrases=_clean(paraphrases),
            keywords=_clean(keywords),
            chunk_id="",
            doc_id=doc_id,
            doc_name=doc_name,
            page_start=int(page_start or 0),
            page_end=int(page_end or 0),
            section=section or "",
        )
        self.store.add_qa_pairs([qa])
        return qa

    def delete_qa_pair(self, qa_id: str) -> None:
        self.store.delete_qa_pair(qa_id)

    def reset(self) -> None:
        """Drop every document, chunk and question. Not undoable."""
        self.store.reset()


# --------------------------------------------------------------------------
# Progress plumbing
# --------------------------------------------------------------------------


class _Progress:
    """Maps a stage's local 0-1 progress onto a slice of the overall bar.

    Callback errors are logged and swallowed rather than raised: a Streamlit
    widget that went stale mid-run must not destroy a forty-minute ingest.
    """

    def __init__(
        self, callback: ProgressCallback | None, low: float, high: float
    ) -> None:
        self._callback = callback
        self._low = low
        self._span = max(0.0, high - low)
        # High-water mark. The load budget is only known once load_pdf has
        # returned, so a bar already carried to _OCR_HIGH by an OCR pass that
        # yielded no text must not fall back to the plain budget.
        self._peak = 0.0

    def emit(self, local: float, message: str) -> None:
        if self._callback is None:
            return
        fraction = min(1.0, max(0.0, self._low + self._span * min(1.0, max(0.0, local))))
        fraction = self._peak = max(self._peak, fraction)
        try:
            self._callback(fraction, message)
        except Exception:  # noqa: BLE001 - see class docstring
            log.warning("progress callback raised; continuing", exc_info=True)

    def stage_mapper(
        self, low: float, high: float, doc_name: str
    ) -> Callable[[int, int, str], None] | None:
        """Adapt ``extract_qa_batch``'s (done, total, msg) to this bar."""
        if self._callback is None:
            return None

        def _on_chunk(done: int, total: int, message: str) -> None:
            local = low + (high - low) * (done / total if total else 1.0)
            # Cross-chunk messages already include their own prefix.
            if "cross-chunk" in (message or "").lower():
                label = f"{doc_name}: {message}"
            else:
                label = (
                    f"{doc_name}: extracting questions from chunk "
                    f"{done}/{total} — {message}"
                )
            self.emit(local, label)

        return _on_chunk

    def ocr_mapper(
        self, low: float, high: float, doc_name: str
    ) -> Callable[[int, int, str], None] | None:
        """Adapt ``ocr_pdf_pages``'s (done, total, msg) to this bar.

        ``total`` counts the pages being OCR'd, not the pages in the document,
        so "OCR page 3/12" means three of the twelve pages that needed it.

        The ETA is measured from this pass rather than assumed from a per-page
        constant: the real cost swings with page size, DPI and whether the
        model is loading cold, so a hardcoded figure would be wrong on every
        machine but the one it was measured on.
        """
        if self._callback is None:
            return None

        started = time.perf_counter()

        def _on_pages(done: int, total: int, message: str) -> None:
            # ocr's own message only repeats the filename we already show.
            local = low + (high - low) * (done / total if total else 1.0)
            self.emit(
                local,
                f"{doc_name}: OCR page {done}/{total}{_eta(started, done, total)}",
            )

        return _on_pages


def _eta(started: float, done: int, total: int) -> str:
    """A " — about 40s left" suffix, or "" when there is nothing to say."""
    remaining = total - done
    if done <= 0 or remaining <= 0:
        return ""

    seconds = (time.perf_counter() - started) / done * remaining
    if seconds < 1:
        return ""
    if seconds < 90:
        return f" — about {seconds:.0f}s left"
    return f" — about {seconds / 60:.0f} min left"


def _doc_stats(
    doc_id: str,
    doc_name: str,
    path: Path,
    pages: int,
    chunks: int,
    questions: int,
    failures: int,
    started: float,
    *,
    ocr_pages: list[int],
    skipped: bool,
    coverage: dict | None = None,
) -> dict:
    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "path": str(path),
        "pages": pages,
        "chunks": chunks,
        "questions": questions,
        "failures": failures,
        # Page numbers, 1-indexed to match citations. Populated even on a
        # skipped document, because the OCR was still paid for.
        "ocr_pages": list(ocr_pages),
        "ocr_used": bool(ocr_pages),
        "elapsed": round(time.perf_counter() - started, 2),
        "skipped": skipped,
        "coverage": coverage or {},
    }


def _empty_batch() -> dict:
    return {
        "documents": [],
        "errors": [],
        "ingested": 0,
        "skipped": 0,
        "failed": 0,
        "pages": 0,
        "chunks": 0,
        "questions": 0,
        "failures": 0,
        "ocr_pages": 0,
        "elapsed": 0.0,
    }


# --------------------------------------------------------------------------
# Headless CLI
# --------------------------------------------------------------------------


def _print_result(result: AnswerResult) -> None:
    print(f"\nQ: {result.query}")
    print(f"route: {result.route}  answered: {result.answered}  "
          f"confidence: {result.confidence:.2f}  "
          f"groundedness: {result.groundedness if result.groundedness is None else f'{result.groundedness:.2f}'}  "
          f"({result.elapsed_seconds:.1f}s)")
    print(f"\n{result.answer}\n")

    if result.citations:
        print("Citations:")
        for citation in result.citations:
            print(f"  {citation.label()}")
            print(f"      \"{citation.quote}\"")
    if result.notes:
        print("\nNotes:")
        for note in result.notes:
            print(f"  - {note}")


def _cli(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    usage = (
        "usage:\n"
        "  python -m rag.pipeline ingest [--force] [--force-ocr] <pdf> [<pdf> ...]\n"
        "  python -m rag.pipeline analyze <pdf>\n"
        '  python -m rag.pipeline ask "question"\n'
        "  python -m rag.pipeline stats\n"
        "\n"
        "  --force      re-ingest a document that is already indexed\n"
        "  --force-ocr  OCR every page, ignoring the text layer (slow)"
    )
    if not argv:
        print(usage)
        return 2

    command, args = argv[0], argv[1:]
    pipeline = RAGPipeline()

    # analyze only reads a PDF; refusing it because Ollama is down would be
    # gratuitous, and it is exactly what you want when diagnosing a bad ingest.
    ok, message = pipeline.health()
    print(f"health: {message}")
    if not ok and command not in ("stats", "analyze"):
        return 1

    if command == "ingest":
        flags = {a for a in args if a.startswith("-")}
        files = [a for a in args if not a.startswith("-")]
        unknown = flags - {"--force", "--force-ocr"}
        if unknown:
            print(f"unknown option(s): {', '.join(sorted(unknown))}\n\n{usage}")
            return 2
        if not files:
            print(usage)
            return 2

        report = pipeline.ingest_pdfs(
            files,
            progress_cb=lambda f, m: print(f"[{f:6.1%}] {m}"),
            force="--force" in flags,
            force_ocr="--force-ocr" in flags,
        )
        print(
            f"\ningested {report['ingested']}, skipped {report['skipped']}, "
            f"failed {report['failed']}, OCR'd {report['ocr_pages']} page(s) "
            f"in {report['elapsed']}s"
        )
        for doc in report["documents"]:
            state = "skipped" if doc["skipped"] else "ok"
            ocr_note = f", OCR on page(s) {doc['ocr_pages']}" if doc["ocr_used"] else ""
            print(
                f"  {doc['doc_name']}: {state} — {doc['pages']} pages, "
                f"{doc['chunks']} chunks, {doc['questions']} questions, "
                f"{doc['failures']} chunk failure(s), {doc['elapsed']}s{ocr_note}"
            )
        for error in report["errors"]:
            print(f"  {error['doc_name']}: FAILED — {error['error']}")
        return 1 if report["failed"] else 0

    if command == "analyze":
        if len(args) != 1:
            print(usage)
            return 2
        try:
            info = pipeline.analyze(args[0])
        except ValueError as exc:  # an unreadable file is a message, not a trace
            print(f"cannot analyze: {exc}")
            return 1

        summary = info["summary"]
        print(
            f"\n{summary['doc_name']}: {summary['page_count']} page(s), "
            f"{summary['total_chars']} extractable char(s)"
        )
        print(f"ocr mode: {summary['ocr_mode']}")
        print(f"  {summary['ocr_reason']}")
        print(
            f"text layers: {summary['layer_ok']} ok, "
            f"{summary['layer_unreliable']} unreliable, {summary['layer_empty']} empty"
        )

        print(f"\n{'page':>5} {'chars':>7} {'imgs':>5} {'rtl':>5}  {'layer':<11} ocr")
        for row in info["pages"]:
            print(
                f"{row['page']:>5} {row['chars']:>7} {row['images']:>5} "
                f"{row['rtl_ratio']:>5.2f}  {row['layer']:<11} "
                f"{'yes' if row['will_ocr'] else '-'}"
            )

        if summary["pages_to_ocr"]:
            print(
                f"\nwould OCR {len(summary['pages_to_ocr'])} page(s): "
                f"{summary['pages_to_ocr']}"
            )
        if summary["degraded_pages"]:
            print(
                f"\nDEGRADED: {len(summary['degraded_pages'])} page(s) need OCR and "
                f"cannot get it: {summary['degraded_pages']}\n"
                "  Their text will be missing or corrupted (RTL ligatures)."
            )
        return 0

    if command == "ask":
        if not args:
            print(usage)
            return 2
        _print_result(pipeline.search(" ".join(args)))
        return 0

    if command == "stats":
        info = pipeline.stats()
        print(
            f"\n{info['n_documents']} document(s), {info['n_chunks']} chunks, "
            f"{info['n_questions']} questions"
        )
        print(
            f"models: extract={info['extract_model']}, "
            f"answer={info['answer_model']} + {info['embed_model']}"
        )
        print(f"store:  {info['persist_dir']}")
        for doc in info["documents"]:
            print(
                f"  {doc['doc_name']}  ({doc['doc_id']})  "
                f"{doc['chunk_count']} chunks, {doc['qa_count']} questions"
            )
        bank = info["question_bank"]
        if bank["total"]:
            print(f"\nquestion bank: {bank['total']} pairs, "
                  f"{bank['hard_question_share']:.0%} hard, "
                  f"{bank['avg_evidence_per_pair']} evidence quotes/pair")
            for label, counts in (("type", bank["by_type"]), ("difficulty", bank["by_difficulty"])):
                shown = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
                print(f"  by {label}: {shown or 'none'}")
        return 0

    print(f"unknown command {command!r}\n\n{usage}")
    return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))


__all__ = ["RAGPipeline", "ProgressCallback", "IngestCancelled"]
