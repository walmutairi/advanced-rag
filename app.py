"""Streamlit front end for the Advanced RAG system.

Thin by design: every decision lives in the pipeline, and this module only
uploads, renders and explains. The one thing it does take a strong position on
is making the RETRIEVAL ROUTE visible — a user needs to see at a glance whether
an answer came from the question bank, from fallback vector search, or whether
the system declined to answer at all.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import streamlit as st

import config
from rag import reviews
from rag.pipeline import IngestCancelled, RAGPipeline
from rag.schemas import (
    MANUAL_DOC_ID,
    ROUTE_HYBRID,
    ROUTE_QUESTION_BANK,
    ROUTE_REFUSED,
    ROUTE_VECTOR,
    AnswerResult,
    QAPair,
)

st.set_page_config(page_title="Advanced RAG", page_icon="📚", layout="wide")

MAX_HISTORY = 8
QUESTION_BANK_PAGE_SIZE = config.QUESTION_BANK_PAGE_SIZE
STREAMLIT_CONFIG = Path(__file__).resolve().parent / ".streamlit" / "config.toml"

# Colours are given as rgba over the theme's own background so the badges stay
# legible in both light and dark Streamlit themes.
ROUTE_STYLE = {
    ROUTE_QUESTION_BANK: ("🟢", "Question Bank", "34, 160, 90"),
    ROUTE_HYBRID: ("🟣", "Hybrid", "120, 90, 180"),
    ROUTE_VECTOR: ("🔵", "Vector RAG", "45, 125, 210"),
    ROUTE_REFUSED: ("🔴", "Not Found", "205, 70, 70"),
}

CSS = """
<style>
  .rag-badges { display: flex; flex-wrap: wrap; gap: .5rem; margin: .25rem 0 1rem; }
  .rag-badge {
    display: inline-flex; align-items: center; gap: .4rem;
    padding: .3rem .75rem; border-radius: 999px;
    font-size: .82rem; font-weight: 600; line-height: 1.2;
    border: 1px solid rgba(var(--rag-accent), .45);
    background: rgba(var(--rag-accent), .12);
    color: inherit;
  }
  .rag-badge small { font-weight: 400; opacity: .75; }
  .rag-answer {
    max-width: 78ch; font-size: 1.02rem; line-height: 1.65;
    padding: 1.1rem 1.3rem; border-radius: .6rem;
    border: 1px solid rgba(128, 128, 128, .25);
    background: rgba(128, 128, 128, .06);
    white-space: pre-wrap;
  }
  .rag-refusal {
    max-width: 78ch; font-size: 1.02rem; line-height: 1.6;
    padding: 1.1rem 1.3rem; border-radius: .6rem;
    border: 1px solid rgba(205, 70, 70, .4);
    background: rgba(205, 70, 70, .08);
    white-space: pre-wrap;
  }
  .rag-quote {
    border-left: 3px solid rgba(128, 128, 128, .45);
    padding: .2rem 0 .2rem .9rem; margin: .4rem 0;
    font-style: italic; opacity: .9;
    white-space: pre-wrap;
  }
</style>
"""


def _esc(text: str | None) -> str:
    """Escape text before embedding in HTML we render with ``unsafe_allow_html``."""
    return html.escape(text or "", quote=True)


def _html_div(class_name: str, text: str) -> str:
    """Wrap escaped prose in a styled div. ``class_name`` must be a fixed literal."""
    return f"<div class='{class_name}'>{_esc(text)}</div>"


# --------------------------------------------------------------------------
# Pipeline lifecycle
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Connecting to Ollama and opening the index…")
def get_pipeline() -> RAGPipeline:
    """One pipeline per session-server, so Chroma and the HTTP session persist."""
    pipeline = RAGPipeline()
    _warmup_ocr()
    return pipeline


@st.cache_resource(show_spinner=False)
def _warmup_ocr() -> bool:
    """Load Surya predictors once so the first OCR page is not a cold start."""
    try:
        from rag import ocr
    except Exception:  # noqa: BLE001 - optional dependency
        return False
    if ocr.is_available():
        ocr.warmup()
        return True
    return False


def _refresh() -> None:
    """Invalidate cached corpus views after a mutation."""
    load_documents.clear()
    load_stats.clear()
    load_question_bank_summary.clear()
    load_question_bank_page.clear()
    load_upload_preview.clear()


def _unique_upload_path(name: str) -> Path:
    """Avoid silently overwriting an earlier upload with the same filename."""
    safe = Path(name).name
    target = config.UPLOAD_DIR / safe
    if not target.exists():
        return target
    stem = Path(safe).stem
    suffix = Path(safe).suffix
    counter = 2
    while True:
        candidate = config.UPLOAD_DIR / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _save_uploads(uploads: list) -> list[Path]:
    paths: list[Path] = []
    for upload in uploads:
        target = _unique_upload_path(upload.name)
        target.write_bytes(upload.getbuffer())
        paths.append(target)
    return paths


@st.cache_data(show_spinner=False)
def load_upload_preview(
    token: int,
    upload_key: tuple[tuple[str, bytes], ...],
    force_ocr: bool,
) -> list[dict]:
    """Per-file ingest preview keyed on upload name + bytes."""
    pipeline = get_pipeline()
    previews: list[dict] = []
    for name, data in upload_key:
        try:
            info = pipeline.analyze(name, pdf_bytes=data, force_ocr=force_ocr)
        except ValueError as exc:
            previews.append({"doc_name": name, "error": str(exc)})
            continue
        summary = info["summary"]
        ocr_count = (
            summary["page_count"]
            if force_ocr and summary.get("ocr_available")
            else len(summary.get("pages_to_ocr") or [])
        )
        previews.append(
            {
                "doc_name": summary["doc_name"],
                "page_count": summary["page_count"],
                "ocr_count": ocr_count,
                "degraded_count": len(summary.get("degraded_pages") or []),
                "total_chars": summary.get("total_chars", 0),
                "pages_to_ocr": summary.get("pages_to_ocr") or [],
                "degraded_pages": summary.get("degraded_pages") or [],
                "ocr_available": summary.get("ocr_available", False),
            }
        )
    return previews


def _upload_cache_key(uploads: list) -> tuple[tuple[str, bytes], ...]:
    return tuple((upload.name, bytes(upload.getbuffer())) for upload in uploads)


def _read_streamlit_watcher() -> str:
    if not STREAMLIT_CONFIG.is_file():
        return "none"
    match = re.search(
        r'fileWatcherType\s*=\s*"(none|poll|watchdog|auto)"',
        STREAMLIT_CONFIG.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else "none"


def _write_streamlit_watcher(mode: str) -> None:
    if mode not in {"none", "poll", "watchdog", "auto"}:
        raise ValueError(f"unsupported watcher mode: {mode}")
    text = STREAMLIT_CONFIG.read_text(encoding="utf-8")
    if re.search(r"fileWatcherType\s*=", text):
        text = re.sub(
            r'fileWatcherType\s*=\s*"(none|poll|watchdog|auto)"',
            f'fileWatcherType = "{mode}"',
            text,
            count=1,
        )
    else:
        text = text.replace("[server]\n", f'[server]\nfileWatcherType = "{mode}"\n', 1)
    STREAMLIT_CONFIG.write_text(text, encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_documents(token: int) -> list[dict]:
    return get_pipeline().documents()


@st.cache_data(show_spinner=False)
def load_stats(token: int) -> dict:
    return get_pipeline().stats()


@st.cache_data(show_spinner=False)
def load_ocr_status(_pipeline: RAGPipeline) -> tuple[bool, str]:
    """Is OCR usable, and what should the user be told either way?

    Cached because the availability probe imports torch to resolve the device,
    and a Streamlit rerun happens on every keystroke. Asked of the pipeline
    first so the UI stays on the façade; a build whose pipeline predates OCR
    falls back to the module, and a module that will not import means
    "unavailable" rather than a sidebar that refuses to render.
    """
    status = getattr(_pipeline, "ocr_status", None)
    if callable(status):
        try:
            available, reason = status()
            return bool(available), str(reason)
        except Exception as exc:  # noqa: BLE001 - a status probe must not break ingest
            return False, f"OCR status could not be determined: {exc}"

    try:
        from rag import ocr
    except Exception as exc:  # noqa: BLE001 - see above
        return False, f"OCR is unavailable: {exc}"
    return ocr.is_available(), ocr.ocr_available_reason()


def _qa_pair_to_dict(pair: QAPair) -> dict:
    return {
        "qa_id": pair.qa_id,
        "question": pair.question,
        "answer": pair.answer,
        "type": pair.question_type,
        "difficulty": pair.difficulty,
        "citation": pair.citation(),
        "evidence": pair.evidence,
        "paraphrases": pair.paraphrases,
        "keywords": pair.keywords,
        "doc_id": pair.doc_id,
        "doc_name": pair.doc_name,
        "page_start": pair.page_start,
        "page_end": pair.page_end,
        "section": pair.section,
    }


@st.cache_data(show_spinner=False)
def load_question_bank_summary(token: int, doc_id: str | None) -> dict:
    return get_pipeline().question_bank_summary(doc_id)


@st.cache_data(show_spinner=False)
def load_question_bank_page(
    token: int,
    doc_id: str | None,
    search: str,
    types: tuple[str, ...],
    difficulties: tuple[str, ...],
    offset: int,
    limit: int,
) -> dict:
    pairs, total = get_pipeline().question_bank_page(
        doc_id=doc_id,
        search=search,
        types=list(types),
        difficulties=list(difficulties),
        offset=offset,
        limit=limit,
    )
    return {
        "pairs": [_qa_pair_to_dict(pair) for pair in pairs],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def render_sidebar(pipeline: RAGPipeline) -> None:
    with st.sidebar:
        st.header("Corpus")

        if st.session_state.get("ingest_report"):
            _show_ingest_report(st.session_state.pop("ingest_report"))

        uploads = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Questions are extracted at upload time, not at search time.",
        )

        force_ocr = _render_ocr_controls(pipeline)
        force_reingest = st.checkbox(
            "Re-ingest already indexed documents",
            value=False,
            help=(
                "Replace chunks and extracted questions for PDFs that are already "
                "in the index. Use this to re-run question extraction without OCR."
            ),
        )

        upload_key = _upload_cache_key(uploads) if uploads else ()
        if uploads:
            previews = load_upload_preview(
                st.session_state.corpus_token, upload_key, force_ocr
            )
            _render_upload_preview(previews, force_ocr)

        if uploads and st.button("Ingest", type="primary", use_container_width=True):
            st.session_state.ingest_job = {
                "paths": [str(p) for p in _save_uploads(uploads)],
                "force": force_reingest or force_ocr,
                "force_ocr": force_ocr,
            }
            st.rerun()

        _render_dev_options()

        stats = load_stats(st.session_state.corpus_token)
        cols = st.columns(3)
        cols[0].metric("Docs", stats.get("n_documents", 0))
        cols[1].metric("Chunks", stats.get("n_chunks", 0))
        cols[2].metric("Questions", stats.get("n_questions", 0))

        obs = stats.get("observability") or {}
        if obs.get("searches"):
            st.caption(
                f"Observability: refuse rate {obs.get('refuse_rate', 0):.0%} · "
                f"avg latency {obs.get('avg_latency_s', 0)}s · "
                f"{obs.get('searches', 0)} searches"
            )

        _render_document_list(pipeline)
        _render_question_bank()
        _render_danger_zone(pipeline)


def _render_ocr_controls(pipeline: RAGPipeline) -> bool:
    """OCR toggle plus an honest statement of what OCR is doing right now.

    The cost warning is not decoration: recognition runs at roughly seven
    seconds per page, so a user who ticks the box on a 200-page upload has
    committed to twenty minutes before question extraction even starts.
    """
    available, reason = load_ocr_status(pipeline)
    mode = (config.OCR_MODE or "auto").strip().lower()

    force_ocr = st.checkbox(
        "Force OCR on all pages",
        value=mode == "always",
        disabled=not available,
        help=(
            "Ignore the embedded text layer and read every page from a rendered "
            "image. Slow, but the only reliable path for scans and RTL text."
        ),
    )

    if not available:
        st.warning(
            f"{reason}\n\nScanned pages will be skipped entirely, and "
            "Arabic/Hebrew pages will fall back to the PDF text layer, which "
            "reverses word order and corrupts lam-alef ligatures.",
            icon="⚠️",
        )
        return False

    scope = (
        "every page"
        if force_ocr or mode == "always"
        else "pages with a missing or unreliable text layer"
    )
    st.caption(
        f"OCR mode `{mode}` — {reason} Will recognise {scope}, at roughly "
        "**7 seconds per page** on this machine."
    )
    return force_ocr


def _render_upload_preview(previews: list[dict], force_ocr: bool) -> None:
    """Show what ingest would do before the user commits."""
    with st.expander("Pre-ingest analysis", expanded=True):
        for item in previews:
            if item.get("error"):
                st.error(f"**{item['doc_name']}**: {item['error']}")
                continue

            name = item["doc_name"]
            pages = item["page_count"]
            ocr_count = item["ocr_count"]
            st.write(f"{name} — {pages} page(s), {item['total_chars']} chars")

            if force_ocr and item.get("ocr_available"):
                st.warning(
                    f"Forced OCR on **all {pages}** page(s) (~{pages * 7 // 60} min OCR "
                    "at 7 s/page, before question extraction).",
                    icon="🔍",
                )
            elif ocr_count:
                eta_min = max(1, ocr_count * 7 // 60)
                page_list = item["pages_to_ocr"]
                shown = ", ".join(str(p) for p in page_list[:20])
                if len(page_list) > 20:
                    shown += f", … (+{len(page_list) - 20} more)"
                st.info(
                    f"Would OCR **{ocr_count}** page(s): {shown}. "
                    f"Roughly **{eta_min} min** of OCR before extraction.",
                    icon="📄",
                )
            else:
                st.caption("No OCR needed — text layer looks usable.")

            if item.get("degraded_count"):
                st.warning(
                    f"**{item['degraded_count']}** page(s) need OCR but cannot get it "
                    f"(pages {item['degraded_pages'][:12]}). Text may be missing or corrupt.",
                    icon="⚠️",
                )


def _render_dev_options() -> None:
    with st.expander("Developer options"):
        current = _read_streamlit_watcher()
        hot_reload = st.checkbox(
            "Enable hot reload on save",
            value=current == "poll",
            help=(
                "Switches Streamlit's file watcher to poll mode. May print torch "
                "__path__ warnings when modules reload."
            ),
        )
        desired = "poll" if hot_reload else "none"
        st.caption(f"Current watcher: `{current}`")
        if desired != current:
            if st.button("Apply and restart later", use_container_width=True):
                _write_streamlit_watcher(desired)
                st.warning(
                    f"Saved `fileWatcherType = \"{desired}\"`. "
                    "**Restart Streamlit** for it to take effect."
                )
        elif hot_reload:
            st.caption("Hot reload is active after the next app restart.")


def _run_ingest_with_progress(
    pipeline: RAGPipeline,
    paths: list[Path],
    *,
    force: bool,
    force_ocr: bool,
) -> None:
    """Run ingest on the main thread so Streamlit progress widgets update live."""
    st.header("⏳ Ingesting PDFs")
    st.warning(
        f"Using extract model `{config.EXTRACT_MODEL}`. "
        "Keep this tab open — the bar below updates as each stage finishes.",
        icon="📄",
    )
    progress_bar = st.progress(0, text="0% — starting…")
    status = st.status("Starting ingest…", expanded=True)
    detail = st.empty()

    def on_progress(fraction: float, message: str) -> None:
        pct = min(max(float(fraction), 0.0), 1.0)
        label = f"{pct * 100:.0f}% — {message}" if message else f"{pct * 100:.0f}%"
        progress_bar.progress(pct, text=label)
        detail.markdown(f"### {message or 'Working…'}")
        status.update(label=message or "Working…", state="running")

    try:
        result = pipeline.ingest_pdfs(
            paths,
            progress_cb=on_progress,
            force=force,
            force_ocr=force_ocr,
        )
        progress_bar.progress(1.0, text="100% — done")
        status.update(label="Ingest finished", state="complete")
        detail.success("Ingest finished.")
        st.session_state.ingest_report = {
            "result": result,
            "error": None,
            "cancelled": False,
        }
    except IngestCancelled:
        status.update(label="Ingest cancelled", state="error")
        st.session_state.ingest_report = {
            "result": None,
            "error": None,
            "cancelled": True,
        }
    except Exception as exc:  # noqa: BLE001
        status.update(label=f"Ingest failed: {exc}", state="error")
        detail.error(f"Ingest failed: {exc}")
        st.session_state.ingest_report = {
            "result": None,
            "error": str(exc),
            "cancelled": False,
        }

    st.session_state.corpus_token += 1
    _refresh()
    st.rerun()


def _show_ingest_report(report: dict) -> None:
    if report.get("cancelled"):
        st.sidebar.warning(
            "Ingest cancelled. Any documents finished before cancel are indexed."
        )
    elif report.get("error"):
        st.sidebar.error(f"Ingest failed: {report['error']}")
    else:
        result = report.get("result") or {}
        if result.get("ingested"):
            ocr_pages = int(result.get("ocr_pages") or 0)
            via_ocr = f" ({ocr_pages} page(s) via OCR)" if ocr_pages else ""
            st.sidebar.success(
                f"Indexed {result['ingested']} document(s): {result['chunks']} chunks, "
                f"{result['questions']} questions{via_ocr} in {result['elapsed']}s."
            )
        if result.get("skipped"):
            st.sidebar.info(f"{result['skipped']} document(s) already indexed — skipped.")
        if result.get("failures"):
            st.sidebar.warning(
                f"{result['failures']} chunk(s) failed extraction and were skipped."
            )
        for err in result.get("errors") or []:
            if isinstance(err, dict):
                st.sidebar.error(
                    f"{err.get('doc_name', 'Document')}: {err.get('error', err)}"
                )
            else:
                st.sidebar.error(str(err))
        _render_coverage_report(result)
        eval_stats = result.get("eval")
        if eval_stats and not eval_stats.get("error"):
            st.sidebar.info(
                f"Post-ingest eval: route accuracy "
                f"{eval_stats.get('route_accuracy', 0):.0%} · "
                f"refuse rate {eval_stats.get('refuse_rate', 0):.0%} "
                f"({eval_stats.get('total', 0)} golden questions)"
            )
        elif eval_stats and eval_stats.get("error"):
            st.sidebar.caption(f"Post-ingest eval skipped: {eval_stats['error']}")


def _render_coverage_report(result: dict) -> None:
    """Per-document coverage: empty chunks, OCR pages, extract failures, OCR text."""
    documents = result.get("documents") or []
    if not documents:
        return
    with st.sidebar.expander("Coverage report", expanded=True):
        for doc in documents:
            if doc.get("skipped"):
                st.caption(f"**{_esc(doc['doc_name'])}** — skipped (already indexed)")
                continue
            coverage = doc.get("coverage") or {}
            st.markdown(f"**{_esc(doc['doc_name'])}**")
            st.caption(
                f"{doc.get('pages', 0)} pages · {doc.get('chunks', 0)} chunks · "
                f"{doc.get('questions', 0)} questions · "
                f"{len(doc.get('ocr_pages') or [])} OCR'd"
            )
            if coverage:
                st.caption(
                    f"Extract: {coverage.get('chunks_succeeded', 0)} ok / "
                    f"{coverage.get('chunks_empty', 0)} empty / "
                    f"{coverage.get('chunks_failed', 0)} failed · "
                    f"{coverage.get('questions_per_chunk', 0)} Qs/chunk"
                )
                if coverage.get("degraded_pages"):
                    st.warning(
                        f"Degraded pages: {coverage['degraded_pages'][:16]}",
                        icon="⚠️",
                    )
                if coverage.get("empty_pages"):
                    st.caption(f"Empty pages: {coverage['empty_pages'][:16]}")
                for failure in coverage.get("failure_samples") or []:
                    st.caption(
                        f"Failed `{failure.get('chunk_id', '?')}`: "
                        f"{failure.get('error', '')[:120]}"
                    )
                previews = coverage.get("text_previews") or []
                if previews:
                    # No nested expanders — Streamlit forbids expander-in-expander.
                    st.markdown("**OCR text preview**")
                    for preview in previews[:6]:
                        st.caption(f"Page {preview.get('page')}")
                        st.text((preview.get("preview") or "")[:600] or "(empty)")


def _render_document_list(pipeline: RAGPipeline) -> None:
    documents = load_documents(st.session_state.corpus_token)
    if not documents:
        st.caption("No documents indexed yet. Upload a PDF to begin.")
        return

    st.subheader("Indexed documents")
    for doc in documents:
        row, action = st.columns([5, 1])
        row.markdown(
            f"**{_esc(doc['doc_name'])}**  \n"
            f"<small>{doc['chunk_count']} chunks · {doc['qa_count']} questions"
            "</small>",
            unsafe_allow_html=True,
        )
        if action.button("🗑", key=f"del_{doc['doc_id']}", help="Remove from index"):
            pipeline.delete_document(doc["doc_id"])
            st.session_state.corpus_token += 1
            _refresh()
            st.rerun()


def _render_question_bank() -> None:
    """Compact sidebar summary that hands off to the full-page browser.

    Deliberately NOT an expander full of per-question expanders: Streamlit
    forbids nesting them, and the sidebar is too narrow to read a question and
    its evidence side by side anyway. The real browser lives in the main panel.
    """
    documents = load_documents(st.session_state.corpus_token)
    pending = reviews.review_count()
    if not documents:
        if pending:
            st.caption(f"**{pending}** answer(s) waiting for review.")
            if st.button(
                f"Browse question bank ({pending} to review)",
                use_container_width=True,
            ):
                st.session_state.show_questions = True
                st.rerun()
        return

    summary = load_question_bank_summary(st.session_state.corpus_token, None)
    total = summary.get("total", 0)
    if total == 0 and pending == 0:
        st.caption("No questions extracted yet.")
        return
    if total == 0:
        st.caption(f"**{pending}** answer(s) waiting for review.")
    else:
        pending_note = f" · **{pending}** need review" if pending else ""
        st.caption(
            f"**{total}** questions · "
            f"**{summary.get('type_count', 0)}** types · "
            f"**{summary.get('hard', 0)}** hard{pending_note}"
        )
    browse_label = "Browse question bank"
    if pending:
        browse_label = f"Browse question bank ({pending} to review)"
    if st.button(browse_label, use_container_width=True):
        st.session_state.show_questions = True
        st.rerun()


def _bar_list(container, counts: dict[str, int]) -> None:
    """Render a horizontal bar chart without going through Arrow.

    Deliberately not ``st.bar_chart``: that serialises through pyarrow, whose
    wheels are built against the numpy 2.x C ABI while this project is pinned
    to numpy 1.26 (surya and chromadb both require it). The mismatch SIGSEGVs
    the interpreter — not an exception, so nothing catches it and the whole
    Streamlit server dies, taking every session with it. These two charts were
    the app's only Arrow consumers, so drawing them by hand removes the
    dependency and the entire class of crash along with it.
    """
    if not counts:
        container.caption("Nothing to show.")
        return

    widest = max(counts.values()) or 1
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        filled = round((count / widest) * 18)
        container.markdown(
            f"<div style='display:flex;gap:.6rem;align-items:center;"
            f"font-size:.85rem;line-height:1.7'>"
            f"<span style='flex:0 0 8.5rem;opacity:.85'>{_esc(label)}</span>"
            f"<span style='font-family:monospace;letter-spacing:-1px;"
            f"opacity:.55'>{'█' * filled}</span>"
            f"<span style='opacity:.7'>{count}</span></div>",
            unsafe_allow_html=True,
        )


def _option_index(options: list[str], value: str | None, default: int = 0) -> int:
    try:
        return options.index(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return default


def _qa_editor_form(
    key_prefix: str, prefill: dict | None = None, *, submit_label: str = "Save"
) -> dict | str | None:
    """Render the question editor inside a form, shared by add and edit.

    Returns the collected field values on submit, the string ``"cancel"`` if
    the user backed out, or ``None`` while the form is merely sitting there.
    Lists are entered as free text — commas for keywords, one-per-line for
    paraphrases and evidence — which is how the rest of the app renders them.
    """
    p = prefill or {}
    with st.form(key=f"{key_prefix}_form", clear_on_submit=prefill is None):
        question = st.text_area(
            "Question", value=p.get("question", ""), key=f"{key_prefix}_q"
        )
        answer = st.text_area(
            "Answer", value=p.get("answer", ""), key=f"{key_prefix}_a"
        )
        left, right = st.columns(2)
        qtype = left.selectbox(
            "Type",
            config.QUESTION_TYPES,
            index=_option_index(config.QUESTION_TYPES, p.get("type")),
            key=f"{key_prefix}_t",
        )
        level = right.selectbox(
            "Difficulty",
            config.DIFFICULTY_LEVELS,
            index=_option_index(config.DIFFICULTY_LEVELS, p.get("difficulty")),
            key=f"{key_prefix}_d",
        )
        keywords = st.text_input(
            "Keywords (comma-separated)",
            value=", ".join(p.get("keywords", [])),
            key=f"{key_prefix}_k",
        )
        paraphrases = st.text_area(
            "Paraphrases (one per line)",
            value="\n".join(p.get("paraphrases", [])),
            help="Alternate phrasings, each indexed so a differently worded "
            "search still lands on this answer.",
            key=f"{key_prefix}_p",
        )
        evidence = st.text_area(
            "Evidence quotes (one per line)",
            value="\n".join(p.get("evidence", [])),
            help="Verbatim source snippets shown as the citation and scored by "
            "the groundedness check.",
            key=f"{key_prefix}_e",
        )
        save_col, cancel_col = st.columns(2)
        submitted = save_col.form_submit_button(
            submit_label, type="primary", use_container_width=True
        )
        cancelled = cancel_col.form_submit_button(
            "Cancel", use_container_width=True
        )

    if cancelled:
        return "cancel"
    if not submitted:
        return None
    return {
        "question": question,
        "answer": answer,
        "question_type": qtype,
        "difficulty": level,
        "keywords": [k.strip() for k in keywords.split(",") if k.strip()],
        "paraphrases": [ln.strip() for ln in paraphrases.splitlines() if ln.strip()],
        "evidence": [ln.strip() for ln in evidence.splitlines() if ln.strip()],
    }


def _after_qa_mutation(message: str, icon: str) -> None:
    """Shared post-write bookkeeping: bust caches, bump the token, rerun."""
    st.session_state.corpus_token += 1
    _refresh()
    st.toast(message, icon=icon)
    st.rerun()


def _render_add_question(pipeline: RAGPipeline, by_name: dict[str, str]) -> None:
    with st.expander("➕ Add a question", expanded=False):
        st.caption(
            "Human-in-the-loop: add or edit a pair and it is re-embedded into the "
            "question bank immediately — no full re-ingest required."
        )
        assoc = st.selectbox(
            "Attach to",
            ["Manual entries", *by_name],
            key="qb_add_assoc",
            help="Manual entries are grouped under their own document; attaching "
            "to an ingested PDF lets the answer cite that document.",
        )
        values = _qa_editor_form("qb_add", submit_label="Add question")
        if values == "cancel" or values is None:
            return
        doc_id = by_name.get(assoc, "") if assoc != "Manual entries" else ""
        doc_name = assoc if assoc != "Manual entries" else ""
        try:
            with st.spinner("Embedding and saving…"):
                pipeline.upsert_qa_pair(doc_id=doc_id, doc_name=doc_name, **values)
        except ValueError as exc:
            st.error(str(exc))
            return
        _after_qa_mutation("Question added", "✅")


def _render_qa_entry(pipeline: RAGPipeline, pair: dict) -> None:
    """One question-bank row, in view / edit / confirm-delete states."""
    qa_id = pair["qa_id"]

    if st.session_state.get("editing_qa") == qa_id:
        values = _qa_editor_form(
            f"qb_edit_{qa_id}", prefill=pair, submit_label="Save changes"
        )
        if values == "cancel":
            st.session_state.editing_qa = None
            st.rerun()
        elif values:
            try:
                with st.spinner("Embedding and saving…"):
                    pipeline.upsert_qa_pair(
                        qa_id=qa_id,
                        doc_id=pair.get("doc_id", ""),
                        doc_name=pair.get("doc_name", ""),
                        page_start=pair.get("page_start", 0),
                        page_end=pair.get("page_end", 0),
                        section=pair.get("section", ""),
                        **values,
                    )
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state.editing_qa = None
            _after_qa_mutation("Question updated", "✅")
        return

    st.caption(
        f"{pair['type']} · {pair['difficulty']} · {pair['citation']}"
    )
    st.write(pair["answer"])
    if pair["evidence"]:
        st.caption("Evidence quoted from the source")
        for quote in pair["evidence"]:
            st.markdown(_html_div("rag-quote", quote), unsafe_allow_html=True)
    if pair["paraphrases"]:
        st.caption("Also indexed as: " + " · ".join(pair["paraphrases"]))
    if pair["keywords"]:
        st.caption("Keywords: " + ", ".join(pair["keywords"]))

    edit_col, del_col = st.columns(2)
    if edit_col.button("✏️ Edit", key=f"edit_{qa_id}", use_container_width=True):
        st.session_state.editing_qa = qa_id
        st.session_state.pending_delete = None
        st.rerun()
    if del_col.button("🗑️ Delete", key=f"delqa_{qa_id}", use_container_width=True):
        st.session_state.pending_delete = qa_id
        st.rerun()

    if st.session_state.get("pending_delete") == qa_id:
        st.warning("Delete this question permanently? This cannot be undone.")
        yes_col, no_col = st.columns(2)
        if yes_col.button(
            "Confirm delete", key=f"cdel_{qa_id}", type="primary",
            use_container_width=True,
        ):
            pipeline.delete_qa_pair(qa_id)
            st.session_state.pending_delete = None
            _after_qa_mutation("Question deleted", "🗑️")
        if no_col.button("Cancel", key=f"canc_{qa_id}", use_container_width=True):
            st.session_state.pending_delete = None
            st.rerun()


def render_question_bank_page(pipeline: RAGPipeline) -> None:
    """Full-width browser over everything mined from the corpus.

    The sidebar expander is fine for a glance, but the question bank IS the
    primary index here — if it is thin or lopsided, retrieval quality is
    already decided before a user ever searches. This view is what makes that
    visible, so it leads with the distribution rather than the list.
    """
    documents = load_documents(st.session_state.corpus_token)
    by_name = {d["doc_name"]: d["doc_id"] for d in documents}

    header, close = st.columns([4, 1])
    header.subheader("Extracted question bank")
    if close.button("← Back to search", use_container_width=True):
        st.session_state.show_questions = False
        st.rerun()

    _render_add_question(pipeline, by_name)
    _render_review_queue(pipeline)

    choice = st.selectbox("Document", ["All documents", *by_name], key="qb_doc")
    doc_id = by_name.get(choice)
    summary = load_question_bank_summary(st.session_state.corpus_token, doc_id)

    if summary.get("total", 0) == 0:
        st.warning(
            "No extracted questions in this scope yet. Add one by hand above, "
            "approve a flagged answer, or ingest a PDF — without a question bank "
            "every search falls through to plain vector retrieval.",
            icon="⚠️",
        )
    else:
        by_type = summary.get("by_type", {})
        by_level = summary.get("by_difficulty", {})
        hard = int(summary.get("hard", 0))

        cols = st.columns(4)
        cols[0].metric("Questions", summary["total"])
        cols[1].metric(
            "Types covered", f"{summary.get('type_count', 0)}/{len(config.QUESTION_TYPES)}"
        )
        cols[2].metric("Advanced", by_level.get("advanced", 0))
        cols[3].metric(
            "Hard types", hard, help="multi_hop + critical + application"
        )

        if hard == 0:
            st.info(
                "No multi-hop, critical or application questions were extracted. "
                "The bank will handle lookups but not reasoning-style queries — "
                "raising `RAG_QA_PER_CHUNK_MAX` and re-ingesting usually helps.",
                icon="💡",
            )

        with st.expander("Distribution", expanded=False):
            left, right = st.columns(2)
            left.caption("By type")
            _bar_list(left, by_type)
            right.caption("By difficulty")
            _bar_list(
                right, {d: by_level.get(d, 0) for d in config.DIFFICULTY_LEVELS}
            )

        st.divider()

        search = st.text_input(
            "Filter", placeholder="Substring match on question, answer or keyword…"
        )
        filters = st.columns(2)
        picked_types = filters[0].multiselect("Type", sorted(by_type), default=[])
        picked_levels = filters[1].multiselect(
            "Difficulty",
            [d for d in config.DIFFICULTY_LEVELS if by_level.get(d)],
            default=[],
        )

        st.session_state.setdefault("qb_page", 0)
        filter_key = (
            choice,
            search.strip().casefold(),
            tuple(sorted(picked_types)),
            tuple(sorted(picked_levels)),
        )
        if st.session_state.get("qb_filter_key") != filter_key:
            st.session_state.qb_filter_key = filter_key
            st.session_state.qb_page = 0

        offset = st.session_state.qb_page * QUESTION_BANK_PAGE_SIZE
        page = load_question_bank_page(
            st.session_state.corpus_token,
            doc_id,
            search.strip(),
            tuple(picked_types),
            tuple(picked_levels),
            offset,
            QUESTION_BANK_PAGE_SIZE,
        )
        shown = page["pairs"]
        filtered_total = page["total"]

        top, download = st.columns([3, 1])
        top.caption(
            f"Showing {offset + 1}-{offset + len(shown)} of {filtered_total} "
            f"matching questions ({summary['total']} total in scope)"
        )
        if filtered_total:
            export_pairs = pipeline.export_question_bank(
                doc_id=doc_id,
                search=search.strip(),
                types=picked_types,
                difficulties=picked_levels,
            )
            download.download_button(
                "Download JSON",
                data=json.dumps(
                    [_qa_pair_to_dict(pair) for pair in export_pairs],
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name="question_bank.json",
                mime="application/json",
                use_container_width=True,
                help="Exports the filtered set, so it doubles as an evaluation dataset.",
            )

        if filtered_total > QUESTION_BANK_PAGE_SIZE:
            pages = (filtered_total + QUESTION_BANK_PAGE_SIZE - 1) // QUESTION_BANK_PAGE_SIZE
            nav = st.columns([1, 2, 1])
            if nav[0].button(
                "← Previous",
                disabled=st.session_state.qb_page <= 0,
                use_container_width=True,
            ):
                st.session_state.qb_page -= 1
                st.rerun()
            nav[1].caption(
                f"Page {st.session_state.qb_page + 1} of {pages}",
            )
            if nav[2].button(
                "Next →",
                disabled=st.session_state.qb_page >= pages - 1,
                use_container_width=True,
            ):
                st.session_state.qb_page += 1
                st.rerun()

        for pair in shown:
            active = st.session_state.get("editing_qa") == pair["qa_id"] or (
                st.session_state.get("pending_delete") == pair["qa_id"]
            )
            with st.expander(pair["question"], expanded=active):
                _render_qa_entry(pipeline, pair)
        if filtered_total == 0:
            st.caption("No questions match the current filter.")


def _render_danger_zone(pipeline: RAGPipeline) -> None:
    with st.expander("Reset index"):
        st.caption(
            "Deletes every chunk and extracted question. Re-ingesting means "
            "paying the extraction cost again."
        )
        confirmed = st.checkbox("I understand", key="reset_ok")
        if st.button("Reset", disabled=not confirmed, use_container_width=True):
            pipeline.reset()
            reviews.clear_all_reviews()
            st.session_state.corpus_token += 1
            st.session_state.search_history = []
            st.session_state.answer_feedback = {}
            # Disarm the confirmation, or the checkbox stays ticked after the
            # reset and one stray click wipes the index again — expensive,
            # since re-ingesting means paying the extraction cost over.
            st.session_state.reset_ok = False
            _refresh()
            st.rerun()


# --------------------------------------------------------------------------
# Answer rendering
# --------------------------------------------------------------------------


def _known_doc_ids() -> set[str]:
    return {doc["doc_id"] for doc in load_documents(st.session_state.corpus_token)}


def _resolve_review_provenance(item: dict) -> tuple[str, str, int, int, str]:
    """Use manual filing when the source document was removed from the index."""
    doc_id = (item.get("doc_id") or "").strip()
    doc_name = (item.get("doc_name") or "").strip()
    page_start = int(item.get("page_start") or 0)
    page_end = int(item.get("page_end") or 0)
    section = (item.get("section") or "").strip()
    if doc_id and doc_id != MANUAL_DOC_ID and doc_id not in _known_doc_ids():
        doc_id = ""
        doc_name = ""
        page_start = 0
        page_end = 0
        section = ""
    return doc_id, doc_name, page_start, page_end, section


def _render_review_queue(pipeline: RAGPipeline) -> None:
    """Thumbs-downed answers waiting to be fixed and approved into the bank."""
    items = reviews.list_reviews()
    heading = (
        f"Needs review ({len(items)})"
        if items
        else "Needs review"
    )
    with st.expander(heading, expanded=bool(items)):
        if not items:
            st.caption(
                "Thumbs-down an answer on the search page to park it here. "
                "Thumbs-up does not save anything."
            )
            return
        st.caption(
            "These answers were flagged as wrong or incomplete. Edit if needed, "
            "then approve to add them to the question bank as guardrails."
        )
        for item in items:
            _render_review_item(pipeline, item)


def _render_review_item(pipeline: RAGPipeline, item: dict) -> None:
    review_id = item["review_id"]
    title = item.get("question") or item.get("query") or review_id
    with st.container(border=True):
        st.markdown(f"**{_esc(title)}**")
        route = (item.get("route") or "").replace("_", " ")
        if route:
            st.caption(f"Flagged from {route}")
        doc_id = (item.get("doc_id") or "").strip()
        if (
            doc_id
            and doc_id != MANUAL_DOC_ID
            and doc_id not in _known_doc_ids()
        ):
            st.warning(
                "The source document is no longer in the index. Approving will "
                "file this as a manual question-bank entry.",
                icon="⚠️",
            )
        if item.get("source_qa_id"):
            st.caption(
                f"Will update existing bank entry `{item['source_qa_id']}` on approve."
            )
        prefill = {
            "question": item.get("question") or item.get("query", ""),
            "answer": item.get("answer", ""),
            "type": item.get("question_type"),
            "difficulty": item.get("difficulty"),
            "keywords": item.get("keywords") or [],
            "paraphrases": item.get("paraphrases") or [],
            "evidence": item.get("evidence") or [],
        }
        values = _qa_review_form(f"rev_{review_id}", prefill)
        if values is None:
            return
        action = values.pop("_action")
        if action == "discard":
            reviews.discard_review(review_id)
            st.toast("Review discarded", icon="🗑️")
            st.rerun()
        if action == "draft":
            reviews.update_review(review_id, values)
            st.toast("Draft saved — still waiting for approval", icon="📝")
            st.rerun()
        if action == "approve":
            doc_id, doc_name, page_start, page_end, section = _resolve_review_provenance(
                item
            )
            source_qa_id = (item.get("source_qa_id") or "").strip() or None
            try:
                with st.spinner("Approving into the question bank…"):
                    pipeline.upsert_qa_pair(
                        qa_id=source_qa_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        page_start=page_start,
                        page_end=page_end,
                        section=section,
                        **values,
                    )
            except ValueError as exc:
                st.error(str(exc))
                return
            reviews.discard_review(review_id)
            _after_qa_mutation("Approved into the question bank", "✅")


def _qa_review_form(key_prefix: str, prefill: dict) -> dict | None:
    """Same fields as the bank editor, plus draft / approve / discard."""
    p = prefill
    with st.form(key=f"{key_prefix}_form"):
        question = st.text_area(
            "Question", value=p.get("question", ""), key=f"{key_prefix}_q"
        )
        answer = st.text_area(
            "Answer", value=p.get("answer", ""), key=f"{key_prefix}_a"
        )
        left, right = st.columns(2)
        qtype = left.selectbox(
            "Type",
            config.QUESTION_TYPES,
            index=_option_index(config.QUESTION_TYPES, p.get("type")),
            key=f"{key_prefix}_t",
        )
        level = right.selectbox(
            "Difficulty",
            config.DIFFICULTY_LEVELS,
            index=_option_index(config.DIFFICULTY_LEVELS, p.get("difficulty")),
            key=f"{key_prefix}_d",
        )
        keywords = st.text_input(
            "Keywords (comma-separated)",
            value=", ".join(p.get("keywords", [])),
            key=f"{key_prefix}_k",
        )
        paraphrases = st.text_area(
            "Paraphrases (one per line)",
            value="\n".join(p.get("paraphrases", [])),
            key=f"{key_prefix}_p",
        )
        evidence = st.text_area(
            "Evidence quotes (one per line)",
            value="\n".join(p.get("evidence", [])),
            key=f"{key_prefix}_e",
        )
        approve_col, draft_col, discard_col = st.columns(3)
        approved = approve_col.form_submit_button(
            "Approve into bank", type="primary", use_container_width=True
        )
        drafted = draft_col.form_submit_button(
            "Save draft", use_container_width=True
        )
        discarded = discard_col.form_submit_button(
            "Discard", use_container_width=True
        )

    if not (approved or drafted or discarded):
        return None
    if discarded:
        return {"_action": "discard"}
    payload = {
        "question": question,
        "answer": answer,
        "question_type": qtype,
        "difficulty": level,
        "keywords": [k.strip() for k in keywords.split(",") if k.strip()],
        "paraphrases": [ln.strip() for ln in paraphrases.splitlines() if ln.strip()],
        "evidence": [ln.strip() for ln in evidence.splitlines() if ln.strip()],
        "_action": "approve" if approved else "draft",
    }
    return payload


def _render_answer_feedback(result: AnswerResult, feedback_key: str) -> None:
    """Thumbs-down parks the answer for review; thumbs-up stores nothing."""
    st.session_state.setdefault("answer_feedback", {})
    voted = st.session_state.answer_feedback.get(result.query)
    already = reviews.find_review_for_query(result.query)
    refuse_confirm_key = f"refuse_down_{feedback_key}"

    if already is not None:
        st.info(
            "This answer is in the question bank **Needs review** queue. "
            "Open the question bank to fix it and approve it.",
            icon="📝",
        )
        return
    if voted == "up":
        st.caption("Marked as good — nothing was saved.")
        return

    if (
        not result.answered
        and st.session_state.get(refuse_confirm_key) == result.query
    ):
        st.warning(
            "This answer was **refused** — there is no grounded response to save. "
            "Queue it only if you want to add a corrected question and answer manually.",
            icon="⚠️",
        )
        yes_col, no_col = st.columns(2)
        if yes_col.button(
            "Yes, queue for review",
            key=f"confirm_down_{feedback_key}",
            use_container_width=True,
        ):
            try:
                reviews.flag_answer(result)
            except ValueError as exc:
                st.error(str(exc))
                return
            st.session_state.answer_feedback[result.query] = "down"
            st.session_state.pop(refuse_confirm_key, None)
            st.toast("Sent to the question bank for review", icon="👎")
            st.rerun()
        if no_col.button(
            "Cancel",
            key=f"cancel_down_{feedback_key}",
            use_container_width=True,
        ):
            st.session_state.pop(refuse_confirm_key, None)
            st.rerun()
        return

    up_col, down_col = st.columns(2)
    if up_col.button(
        "👍 Looks good",
        key=f"up_{feedback_key}",
        use_container_width=True,
        help="No result is stored. The question bank is unchanged.",
    ):
        st.session_state.answer_feedback[result.query] = "up"
        st.toast("Thanks — nothing saved", icon="👍")
        st.rerun()
    if down_col.button(
        "👎 Needs review",
        key=f"down_{feedback_key}",
        use_container_width=True,
        help="Send this Q/A to the question bank so you can fix and approve it.",
    ):
        if not result.answered:
            st.session_state[refuse_confirm_key] = result.query
            st.rerun()
        try:
            reviews.flag_answer(result)
        except ValueError as exc:
            st.error(str(exc))
            return
        st.session_state.answer_feedback[result.query] = "down"
        st.toast("Sent to the question bank for review", icon="👎")
        st.rerun()


def render_answer(
    result: AnswerResult,
    *,
    show_diagnostics: bool = True,
    feedback_key: str = "answer",
) -> None:
    icon, label, accent = ROUTE_STYLE.get(result.route, ("⚪", result.route, "128,128,128"))

    badges = [
        f"<span class='rag-badge' style='--rag-accent:{accent}'>"
        f"{icon} {_esc(label)}</span>",
        f"<span class='rag-badge' style='--rag-accent:128,128,128'>"
        f"confidence <small>{result.confidence:.2f}</small></span>",
    ]
    if result.groundedness is not None:
        badges.append(
            f"<span class='rag-badge' style='--rag-accent:128,128,128'>"
            f"groundedness <small>{result.groundedness:.2f}</small></span>"
        )
    badges.append(
        f"<span class='rag-badge' style='--rag-accent:128,128,128'>"
        f"uncertainty <small>{getattr(result, 'uncertainty', 0.0):.2f}</small></span>"
    )
    badges.append(
        f"<span class='rag-badge' style='--rag-accent:128,128,128'>"
        f"<small>{result.elapsed_seconds:.1f}s</small></span>"
    )
    st.markdown(
        f"<div class='rag-badges'>{''.join(badges)}</div>", unsafe_allow_html=True
    )

    body_class = "rag-refusal" if not result.answered else "rag-answer"
    st.markdown(_html_div(body_class, result.answer), unsafe_allow_html=True)
    _render_answer_feedback(result, feedback_key)

    near_misses = getattr(result, "near_misses", None) or []
    if not result.answered and near_misses:
        with st.container(border=True):
            st.markdown("**Why refused — closest near-misses**")
            for miss in near_misses[:5]:
                st.write(f"- {miss}")

    if result.citations:
        st.markdown("#### Sources")
        for citation in result.citations:
            with st.container(border=True):
                st.markdown(f"**{_esc(citation.label())}**")
                st.markdown(_html_div("rag-quote", citation.quote), unsafe_allow_html=True)
                st.caption(f"Retrieved via {citation.origin.replace('_', ' ')}")

    if result.matched_questions:
        with st.container(border=True):
            st.markdown(f"**Matched questions ({len(result.matched_questions)})**")
            for scored in result.matched_questions:
                qa = scored.qa
                st.markdown(f"**{_esc(qa.question)}**")
                st.caption(
                    f"score {scored.score:.3f} · matched on {scored.matched_on} · "
                    f"{qa.question_type} · {qa.difficulty} · {qa.citation()}"
                )
                st.write(qa.answer)
                st.divider()

    if result.matched_chunks:
        with st.container(border=True):
            st.markdown(f"**Retrieved passages ({len(result.matched_chunks)})**")
            for scored in result.matched_chunks:
                chunk = scored.chunk
                st.markdown(f"**{_esc(chunk.doc_name)}, {_esc(chunk.page_label)}**")
                st.caption(f"score {scored.score:.3f}")
                st.text(chunk.text[:1200] + ("…" if len(chunk.text) > 1200 else ""))
                st.divider()

    if show_diagnostics:
        with st.container(border=True):
            st.markdown("**Retrieval diagnostics**")
            if result.notes:
                for note in result.notes:
                    st.write(f"- {note}")
            else:
                st.caption("No diagnostics recorded.")


def _run_search(pipeline: RAGPipeline, query: str) -> AnswerResult:
    """Route, stream answer tokens, then return the final verified result."""
    stream_slot = st.empty()
    accumulated = ""
    result: AnswerResult | None = None

    status = st.status("Searching and retrieving…", expanded=False)
    try:
        for piece in pipeline.search_stream(query):
            if isinstance(piece, AnswerResult):
                result = piece
            else:
                accumulated += piece
                stream_slot.markdown(
                    _html_div("rag-answer", accumulated),
                    unsafe_allow_html=True,
                )
    finally:
        status.update(label="Answer ready", state="complete")

    stream_slot.empty()
    if result is None:
        result = AnswerResult(
            query=query,
            answer=config.REFUSAL_MESSAGE,
            route=ROUTE_REFUSED,
            answered=False,
            notes=["No result was returned from the pipeline."],
        )
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("corpus_token", 0)
    st.session_state.setdefault("search_history", [])
    st.session_state.setdefault("pending_query", "")
    st.session_state.setdefault("show_questions", False)
    st.session_state.setdefault("editing_qa", None)
    st.session_state.setdefault("pending_delete", None)
    st.session_state.setdefault("qb_page", 0)
    st.session_state.setdefault("answer_feedback", {})

    pipeline = get_pipeline()

    healthy, message = pipeline.health()
    if not healthy:
        st.error(message, icon="🚫")
        st.caption(
            "Start the server with `ollama serve`, then pull the models named above."
        )
        st.stop()

    st.title("📚 Advanced RAG")

    # Ingest takes over the whole page so the progress bar cannot be missed.
    ingest_job = st.session_state.pop("ingest_job", None)
    if ingest_job is not None:
        _run_ingest_with_progress(
            pipeline,
            [Path(p) for p in ingest_job["paths"]],
            force=bool(ingest_job["force"]),
            force_ocr=bool(ingest_job["force_ocr"]),
        )
        return

    st.caption(
        "Questions are mined from your PDFs up front. Search matches those "
        "questions first, falls back to vector retrieval, and declines to answer "
        "when neither has it."
    )
    st.caption(
        "Local-only: no authentication, no network egress, one shared index per "
        "server process. Do not expose this UI on a shared or public host without "
        "adding your own access controls."
    )

    render_sidebar(pipeline)

    if st.session_state.show_questions:
        render_question_bank_page(pipeline)
        return

    if not load_documents(st.session_state.corpus_token):
        st.info("Upload a PDF in the sidebar to get started.", icon="👈")
        return

    n_questions = load_stats(st.session_state.corpus_token).get("n_questions", 0)
    n_reviews = reviews.review_count()
    review_suffix = f" · {n_reviews} to review" if n_reviews else ""
    if st.button(
        f"📋 Show extracted questions ({n_questions}{review_suffix})",
        use_container_width=True,
        help="Browse everything mined from your PDFs, with evidence and citations. "
        "Thumbs-downed answers wait here until you fix and approve them.",
    ):
        st.session_state.show_questions = True
        st.rerun()

    with st.form("search", clear_on_submit=False):
        query = st.text_input(
            "Ask a question",
            value=st.session_state.pending_query,
            placeholder="What does the document say about…?",
        )
        submitted = st.form_submit_button("Search", type="primary")

    st.session_state.pending_query = ""

    # No `query.strip()` guard: the pipeline already answers a blank query with
    # a proper refusal, and swallowing the submit made the button look broken.
    if submitted:
        result = _run_search(pipeline, query)
        history = [
            entry for entry in st.session_state.search_history if entry.query != query
        ]
        st.session_state.search_history = [result, *history][:MAX_HISTORY]

    if st.session_state.search_history:
        st.divider()
        st.caption("Recent searches")
        for i, past in enumerate(st.session_state.search_history):
            with st.expander(past.query, expanded=(i == 0)):
                render_answer(
                    past,
                    show_diagnostics=(i == 0),
                    feedback_key=f"hist_{i}",
                )
                if i > 0 and st.button("Search again", key=f"requery_{i}"):
                    st.session_state.pending_query = past.query
                    st.rerun()


if __name__ == "__main__":
    main()
