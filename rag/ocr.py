"""Surya OCR, wrapped so nothing else in the codebase imports surya directly.

PyMuPDF's text layer is untrustworthy for RTL scripts: it returns spans in
visual order and mis-decomposes lam-alef ligatures (الاجازات -> االجازات), which
corrupts embeddings and the question bank silently. Surya re-reads the rendered
page and emits correct logical order, so it is the fallback authority.

Three sharp edges live here:

* The predictors are a lazy, process-wide singleton behind a lock. They are
  expensive to build and large in memory; Streamlit reruns can race two loads
  into existence and double the footprint.
* Surya's line order is already correct reading order. Do NOT re-sort by bbox
  and do NOT apply any bidi/reversal transform — the text comes out logical,
  and every "fix" applied to it re-breaks it.
* surya is imported lazily inside functions. A machine without it installed
  still runs the whole app; ``is_available()`` just reports False and the
  loader falls back to the embedded text layer.

This module is pinned to the surya 0.17.x v1 predictor API. 0.20+ replaced it
with a client/server VLM that needs an external llama-server binary.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import io
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

import fitz

import config

logger = logging.getLogger(__name__)

#: Pages rendered and recognised per batch. Surya is happiest with a few
#: images at once, but a full-resolution bitmap is tens of megabytes — a
#: 400-page document must never materialise 400 of them, so windows are
#: released before the next one is rendered.
_PAGE_WINDOW = 4

#: Inline markup tags Surya may emit inside ``line.text``. This project
#: extracts prose, so the tags are noise: they would be embedded, BM25
#: tokenised and shown to the reader inside citation snippets. Surya's own
#: default list covers block tags only, so the inline ones are added here.
_FILTER_TAGS = [
    "p", "li", "ul", "ol", "table", "td", "tr", "th", "tbody", "pre",
    "br", "b", "i", "u", "sup", "sub", "del", "mark", "math",
]

ProgressCallback = Callable[[int, int, str], None]

# --------------------------------------------------------------------------
# Predictor singleton
# --------------------------------------------------------------------------

_lock = threading.Lock()
_predictors: tuple[Any, Any, Any] | None = None  # (foundation, recognition, detection)

#: Held across every inference call. Surya's FoundationPredictor keeps its
#: prompt queue and KV cache on the instance and rebuilds them per call, so two
#: threads sharing the singleton interleave into one decode loop and silently
#: return merged or truncated text. Streamlit runs each browser session in its
#: own thread, so two tabs ingesting at once is ordinary usage. Separate from
#: ``_lock`` so a caller waiting on inference never blocks construction.
_infer_lock = threading.Lock()


def _surya_installed() -> bool:
    """True when the surya package can be imported without importing it."""
    try:
        return importlib.util.find_spec("surya") is not None
    except (ImportError, ValueError):
        return False


def is_available() -> bool:
    """OCR can actually be attempted: surya is installed and OCR is enabled."""
    return bool(config.ENABLE_OCR) and _surya_installed()


def ocr_available_reason() -> str:
    """One sentence for the UI explaining the current OCR status.

    Non-empty in both directions: callers that only care about the failure
    case should gate on ``is_available()`` and use this purely as the message.
    """
    if not config.ENABLE_OCR:
        return "OCR is disabled in configuration (set RAG_ENABLE_OCR=1 to enable it)."
    if not _surya_installed():
        return (
            "OCR is unavailable: the 'surya-ocr' package is not installed. "
            "Documents will fall back to the embedded PDF text layer, which "
            "mis-orders and corrupts Arabic and other RTL text."
        )
    return f"OCR is available (Surya on device '{_resolve_device()}')."


def _resolve_device() -> str:
    """Map config.OCR_DEVICE onto a concrete torch device string."""
    requested = (config.OCR_DEVICE or "auto").strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # torch missing or a broken backend probe
        logger.debug("Could not probe torch devices; defaulting to cpu", exc_info=True)
    return "cpu"


def _get_predictors() -> tuple[Any, Any, Any]:
    """Build (or return) the process-wide predictor triple.

    Double-checked locking: the fast path is a bare read, and only the first
    caller through pays for construction while the rest block.
    """
    global _predictors

    if _predictors is not None:
        return _predictors

    with _lock:
        if _predictors is not None:  # another thread won the race while we waited
            return _predictors

        device = _resolve_device()

        # Surya bakes settings.TORCH_DEVICE_MODEL into predictor default
        # arguments at *import* time, so TORCH_DEVICE has to be in the
        # environment before the first surya import to have any effect. We
        # also pass device= explicitly, which covers the case where some
        # other module imported surya before us. Assign rather than
        # setdefault: surya derives dtype, batch size, encoder chunk size and
        # KV-cache length from settings.TORCH_DEVICE_MODEL and never from the
        # device= argument, so an inherited TORCH_DEVICE that disagrees with
        # the device we actually load on picks the wrong heuristics.
        os.environ["TORCH_DEVICE"] = device

        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        logger.info("Loading Surya predictors on device '%s'", device)
        foundation = FoundationPredictor(device=device)
        recognition = RecognitionPredictor(foundation)
        detection = DetectionPredictor(device=device)

        _predictors = (foundation, recognition, detection)
        return _predictors


def warmup() -> None:
    """Force the model load now, outside a request path.

    Silently does nothing when OCR is unavailable — a warmup is an
    optimisation, never a reason to fail a caller.
    """
    if not is_available():
        return
    try:
        _get_predictors()
    except Exception:
        logger.warning("Surya warmup failed; OCR will retry on first use", exc_info=True)


def unload() -> None:
    """Drop the predictors and hand accelerator memory back."""
    global _predictors

    with _lock:
        if _predictors is None:
            return
        _predictors = None

    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Could not empty the torch device cache", exc_info=True)


# --------------------------------------------------------------------------
# Page rendering and recognition
# --------------------------------------------------------------------------


def _render_page(doc: fitz.Document, page_index: int) -> Any:
    """Render one 0-indexed page to a PIL Image at config.OCR_DPI.

    Goes through PNG bytes rather than raw samples so we never have to reason
    about pixmap colourspace or alpha layout, and so no pdf2image/poppler
    dependency is needed.
    """
    from PIL import Image

    pixmap = doc[page_index].get_pixmap(dpi=config.OCR_DPI)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    image.load()  # decode now; the BytesIO goes out of scope immediately
    return image


def _text_from_result(result: Any) -> str:
    """Join one page's recognised lines into text.

    Order is Surya's own, which is already correct reading order. Re-sorting
    by bbox or applying a bidi pass would re-break exactly the RTL ordering
    this module exists to get right.
    """
    lines: list[str] = []
    for line in getattr(result, "text_lines", []) or []:
        confidence = getattr(line, "confidence", None)
        if confidence is not None and confidence < config.OCR_MIN_CONFIDENCE:
            continue
        text = (getattr(line, "text", "") or "").strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _recognise(images: list[Any]) -> list[str]:
    """Recognise a window of images, falling back to per-image on failure.

    A batch failure must not cost the whole window: retry each image alone so
    one unrenderable or pathological page loses only itself.

    Predictor construction is guarded too: ``is_available()`` only checks that
    surya is importable, so a broken install, an incompatible version or a
    failed weight download surfaces here and must degrade to empty text rather
    than escape to the caller.
    """
    try:
        _, recognition, detection = _get_predictors()
    except Exception:
        logger.exception("Surya predictors unavailable; recording %d page(s) as empty", len(images))
        return ["" for _ in images]

    try:
        with _infer_lock:
            results = recognition(
                images, det_predictor=detection, filter_tag_list=_FILTER_TAGS
            )
        return [_text_from_result(r) for r in results]
    except Exception:
        logger.warning(
            "Surya batch of %d page(s) failed; retrying individually",
            len(images),
            exc_info=True,
        )

    texts: list[str] = []
    for image in images:
        try:
            with _infer_lock:
                results = recognition(
                    [image], det_predictor=detection, filter_tag_list=_FILTER_TAGS
                )
            texts.append(_text_from_result(results[0]))
        except Exception:
            logger.exception("Surya failed on a page; recording it as empty")
            texts.append("")
    return texts


def _pdf_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _ocr_cache_path(digest: str, page_number: int) -> Path:
    key = (
        f"{digest}_p{page_number}_dpi{config.OCR_DPI}"
        f"_c{config.OCR_MIN_CONFIDENCE:.2f}"
    )
    return config.CACHE_DIR / "ocr" / f"{key}.txt"


def _read_ocr_cache(digest: str, page_number: int) -> str | None:
    if not config.ENABLE_OCR_CACHE:
        return None
    path = _ocr_cache_path(digest, page_number)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        logger.debug("OCR cache read failed for %s", path, exc_info=True)
        return None


def _write_ocr_cache(digest: str, page_number: int, text: str) -> None:
    if not config.ENABLE_OCR_CACHE:
        return
    path = _ocr_cache_path(digest, page_number)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        logger.debug("OCR cache write failed for %s", path, exc_info=True)


def ocr_pdf_pages(
    path: Path | str,
    page_numbers: list[int] | None = None,
    progress_cb: ProgressCallback | None = None,
) -> dict[int, str]:
    """OCR selected pages of a PDF.

    ``page_numbers`` is 1-indexed to match ``PageText.page_number`` and what a
    reader sees in their viewer; ``None`` means every page. Returns a mapping
    of page number to recognised text. Pages that fail map to "" rather than
    aborting the document; page numbers outside the document are dropped with
    a warning. Returns {} when OCR is unavailable.

    When ``ENABLE_OCR_CACHE`` is on, recognised pages are persisted under
    ``data/cache/ocr`` keyed by file content hash, page number and DPI.
    """
    if not is_available():
        logger.info("OCR requested but unavailable: %s", ocr_available_reason())
        return {}

    path = Path(path)
    results: dict[int, str] = {}
    digest = _pdf_digest(path)

    with fitz.open(path) as doc:
        page_count = doc.page_count

        if page_numbers is None:
            wanted = list(range(1, page_count + 1))
        else:
            wanted = sorted({int(n) for n in page_numbers})
            out_of_range = [n for n in wanted if not 1 <= n <= page_count]
            if out_of_range:
                logger.warning(
                    "Ignoring page numbers outside %s (1-%d): %s",
                    path.name,
                    page_count,
                    out_of_range,
                )
            wanted = [n for n in wanted if 1 <= n <= page_count]

        total = len(wanted)
        if total == 0:
            return {}

        # Serve cached pages first so progress still advances for cache hits.
        pending: list[int] = []
        for page_number in wanted:
            cached = _read_ocr_cache(digest, page_number)
            if cached is not None:
                results[page_number] = cached
            else:
                pending.append(page_number)

        done = total - len(pending)
        if progress_cb is not None and done:
            progress_cb(done, total, f"OCR cache hit {done}/{total} pages of {path.name}")

        for start in range(0, len(pending), _PAGE_WINDOW):
            window = pending[start : start + _PAGE_WINDOW]

            images: list[Any] = []
            rendered: list[int] = []
            for page_number in window:
                try:
                    images.append(_render_page(doc, page_number - 1))
                    rendered.append(page_number)
                except Exception:
                    logger.exception("Could not render page %d of %s", page_number, path.name)
                    results[page_number] = ""
                    _write_ocr_cache(digest, page_number, "")

            if images:
                texts = _recognise(images)
                for page_number, text in zip(rendered, texts):
                    results[page_number] = text
                    _write_ocr_cache(digest, page_number, text)

                for image in images:
                    try:
                        image.close()
                    except Exception:
                        pass
                images.clear()

            done += len(window)
            if progress_cb is not None:
                progress_cb(done, total, f"OCR {done}/{total} pages of {path.name}")

    return results


__all__ = [
    "is_available",
    "ocr_available_reason",
    "ocr_pdf_pages",
    "warmup",
    "unload",
]
