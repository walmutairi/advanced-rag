"""Page-attributed text extraction from PDFs, with per-page OCR routing.

Everything downstream — chunk boundaries, page citations, the evidence quotes
the groundedness pass checks against — inherits whatever this module produces,
so the cleaning here is deliberately conservative and order-sensitive.

Sharp edges worth knowing about:

* Blank lines are load-bearing. The chunker splits on paragraph breaks, so
  single newlines inside a paragraph are collapsed to spaces while blank-line
  separators survive untouched.
* Pages are never dropped. A page that cleans down to nothing stays in the
  list as an empty string, because ``PageText.page_number`` is a citation and
  must keep matching what the reader sees in their PDF viewer.
* "Is this a scan?" is decided PER PAGE, never per document. The old
  document-level character-count test declared any legitimately short PDF to
  be image-only and refused it outright. A document may also genuinely mix
  scanned and digital pages.
* PyMuPDF cannot be trusted with RTL text at all. It returns spans in visual
  order (``[' : ', 'االجازات', ' ', 'انواع']`` for one line) and it
  mis-decomposes lam-alef ligatures (الاجازات -> االجازات). Reversing span
  order repairs the first problem; nothing repairs the second short of OCR,
  so a substantially-Arabic/Hebrew page is routed to Surya whenever possible
  and the reversal path is strictly a degraded fallback.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

import fitz

import config
from rag import ocr
from rag.schemas import PageText, _stable_id

ProgressCallback = Callable[[int, int, str], None]

# --------------------------------------------------------------------------
# Unicode normalisation
# --------------------------------------------------------------------------
# Done with an explicit table rather than unicodedata.normalize("NFKC"),
# which would also rewrite superscripts, fractions and maths symbols and
# quietly corrupt technical source material.

_CHAR_MAP = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    # All dash variants fold to ASCII hyphen so the BM25 tokeniser and the
    # embedding model see one spelling of "state-of-the-art".
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", "　": " ",
    "…": "...",
    # Soft hyphens and zero-width joiners are invisible but break tokenisation.
    "­": "", "​": "", "‌": "", "‍": "", "﻿": "",
}

_TRANSLATION = str.maketrans(_CHAR_MAP)

# --------------------------------------------------------------------------
# Line / paragraph shaping
# --------------------------------------------------------------------------

#: A hyphen immediately before a line break, with the whole preceding token
#: captured so the join can be judged in context.
_HYPHEN_BREAK_RE = re.compile(r"(\S+)-\n[ \t]*(\w)")

#: Prefixes that stay hyphenated in technical prose far more often than they
#: get broken mid-word. Without this, "self-\nreport" merges to "selfreport",
#: which no query will ever match. The cost is the rarer inverse ("co-\nsine"
#: surviving as "co-sine"), which at least still tokenises into real words.
_COMPOUND_PREFIXES = frozenset(
    {"self", "non", "co", "anti", "quasi", "pseudo", "cross", "ex", "well", "half", "all"}
)

#: A newline with no blank line on either side is intra-paragraph wrapping.
_SOFT_NEWLINE_RE = re.compile(r"(?<!\n)\n(?!\n)")

_BLANK_RUN_RE = re.compile(r"\n{3,}")
_SPACE_RUN_RE = re.compile(r"[ \t]{2,}")

#: Explicit folio markers: "Page 12", "12 of 340", "- 12 -", "[12]".
_FOLIO_RE = re.compile(
    r"^[\s\-\[\(\|]*(?:page|pg\.?|p\.?)?\s*\d{1,4}\s*(?:of\s*\d{1,4})?[\s\-\]\)\|]*$",
    re.IGNORECASE,
)

#: Bare numbers and roman numerals. Only trusted at a page edge, since a lone
#: "7" in the middle of a page is far more likely to be a table cell.
_BARE_FOLIO_RE = re.compile(r"^[\s\-\[\(\|]*(?:\d{1,4}|[ivxlcdm]{1,7})[\s\-\]\)\|]*$",
                            re.IGNORECASE)

_DIGIT_RE = re.compile(r"\d+")
_PUNCT_RE = re.compile(r"[^\w\s]+")
_WS_RE = re.compile(r"\s+")

#: How many non-empty lines at each page edge are eligible for header/footer
#: removal. Two, because a running head is often "Title" over "Section".
_EDGE_DEPTH = 2

#: A line must appear at the edge of at least this share of pages to count as
#: running furniture rather than body text that happened to land there.
_REPEAT_RATIO = 0.60
_MIN_PAGES_FOR_REPEAT = 4

# --------------------------------------------------------------------------
# RTL detection
# --------------------------------------------------------------------------

#: Arabic (base, supplement, presentation forms A and B) plus Hebrew. These
#: are the scripts whose visual-order span output PyMuPDF gets wrong.
_RTL_RE = re.compile(
    "[֐-׿؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]"
)

#: Page text-layer classifications.
LAYER_OK = "ok"
LAYER_EMPTY = "empty"
LAYER_UNRELIABLE = "unreliable"

#: How a page's final text was obtained, recorded in the load report.
SOURCE_TEXT = "text"
SOURCE_TEXT_RTL = "text_rtl_reordered"
SOURCE_OCR = "ocr"
SOURCE_EMPTY = "empty"

#: Load reports are kept in memory so the pipeline and UI can ask what
#: happened after the fact. Bounded, because a long-lived Streamlit process
#: would otherwise accumulate one per upload forever.
_REPORT_LIMIT = 64
_reports: dict[str, dict[str, Any]] = {}
_reports_lock = threading.Lock()


# --------------------------------------------------------------------------
# Opening
# --------------------------------------------------------------------------


def _read_pdf_bytes(path: Path) -> bytes:
    if not path.exists():
        raise ValueError(f"PDF not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    data = path.read_bytes()
    # Sniff the header rather than trusting the extension; some producers
    # prepend junk, so allow the marker anywhere in the first block.
    if data[:1024].find(b"%PDF") == -1:
        raise ValueError(f"Not a PDF (missing %PDF header): {path}")
    return data


def _open(path: Path, data: bytes | None = None) -> tuple[bytes, fitz.Document]:
    """Return the file bytes and an open, decrypted document."""
    if data is None:
        data = _read_pdf_bytes(path)

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # pymupdf raises a grab-bag of error types
        raise ValueError(f"Could not open PDF {path.name}: {exc}") from exc

    if doc.needs_pass:
        # Many "encrypted" PDFs are only owner-locked and open with no
        # password at all; that is worth trying before giving up.
        if not doc.authenticate(""):
            doc.close()
            raise ValueError(
                f"PDF {path.name} is password-protected and cannot be opened. "
                "Remove the password and re-upload it."
            )

    if doc.page_count == 0:
        doc.close()
        raise ValueError(f"PDF {path.name} contains no pages.")

    return data, doc


# --------------------------------------------------------------------------
# Text-layer extraction
# --------------------------------------------------------------------------


def _table_text(page: fitz.Page) -> str:
    """Append structured table rows when PyMuPDF can find them.

    Many PDFs bury answers in tables that ``get_text`` flattens poorly. A
    best-effort table dump gives the chunker and extractor a clearer view
    without a separate table model.
    """
    try:
        finder = page.find_tables()
    except Exception:
        return ""
    tables = getattr(finder, "tables", None) or []
    if not tables:
        return ""
    blocks: list[str] = []
    for table in tables:
        try:
            rows = table.extract()
        except Exception:
            continue
        if not rows:
            continue
        lines = []
        for row in rows:
            cells = [("" if cell is None else str(cell).strip()) for cell in row]
            if any(cells):
                lines.append(" | ".join(cells))
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _figure_captions(page: fitz.Page) -> str:
    """Pull likely figure/table captions that sit near images."""
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return ""
    captions: list[str] = []
    for block in blocks:
        if len(block) < 5:
            continue
        text = (block[4] or "").strip()
        if not text:
            continue
        head = text[:40].lower()
        if head.startswith(("fig.", "figure", "table", "tab.", "الشكل", "جدول")):
            captions.append(text)
    return "\n\n".join(captions)


def _plain_text(page: fitz.Page) -> str:
    try:
        body = page.get_text("text") or ""
    except Exception:
        # One malformed page must not lose the other 300; the empty string
        # keeps page numbering aligned.
        body = ""
    extras: list[str] = []
    tables = _table_text(page)
    if tables and tables not in body:
        extras.append(tables)
    captions = _figure_captions(page)
    if captions:
        for caption in captions.split("\n\n"):
            if caption and caption not in body:
                extras.append(caption)
    if not extras:
        return body
    return f"{body.rstrip()}\n\n" + "\n\n".join(extras)


def _rtl_ordered_text(page: fitz.Page) -> str:
    """Re-extract a page with the span order of RTL lines reversed.

    PyMuPDF hands back spans in visual (left-to-right) order, so an Arabic
    line arrives as [' : ', 'الاجازات', ' ', 'انواع'] and joins into gibberish.
    Reversing the spans of any line containing RTL characters restores logical
    word order — verified against a known page.

    DEGRADED PATH. This fixes ordering only. Lam-alef ligatures still come out
    mis-decomposed (الاجازات -> االجازات, الموظف -> املوظف), which silently
    corrupts embeddings and the question bank. Do NOT try to repair that by
    string substitution: "اال" is a legitimate sequence in real words and a
    blind swap would corrupt text that is currently correct. OCR is the only
    real fix, so every page that lands here is recorded as degraded.
    """
    try:
        data = page.get_text("dict")
    except Exception:
        return ""

    blocks: list[str] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 1 == image block, no text to recover
            continue

        lines: list[str] = []
        for line in block.get("lines", []):
            spans = [span.get("text", "") for span in line.get("spans", [])]
            if not spans:
                continue
            joined = "".join(spans)
            if _RTL_RE.search(joined):
                # Spans carry their own leading/trailing spaces, so an empty
                # join keeps the original spacing once the order is fixed.
                joined = "".join(reversed(spans))
            lines.append(joined)

        if lines:
            blocks.append("\n".join(lines))

    # get_text("text") separates blocks with a single newline; matching that
    # keeps paragraph shaping identical between the two extraction paths.
    return "\n".join(blocks)


# --------------------------------------------------------------------------
# Per-page classification
# --------------------------------------------------------------------------


def _rtl_ratio(text: str) -> float:
    """Share of the page's letters that belong to an RTL script.

    Digits and punctuation are excluded from BOTH sides: a mostly-numeric
    Arabic table has few letters, and they should still be enough to condemn
    the layer. Counting digits would dilute the ratio below the trigger and
    let a corrupted table through.
    """
    letters = 0
    rtl = 0
    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        if _RTL_RE.match(char):
            rtl += 1
    return (rtl / letters) if letters else 0.0


def _char_count(text: str) -> int:
    """Extractable characters, ignoring whitespace-only padding."""
    return len(_WS_RE.sub(" ", text).strip())


def _classify(text: str) -> tuple[str, int, float]:
    """Judge one page's text layer. Returns (layer, chars, rtl_ratio)."""
    chars = _char_count(text)
    ratio = _rtl_ratio(text)

    if chars < config.OCR_MIN_CHARS_PER_PAGE:
        return LAYER_EMPTY, chars, ratio
    if ratio > config.OCR_RTL_TRIGGER_RATIO:
        return LAYER_UNRELIABLE, chars, ratio
    return LAYER_OK, chars, ratio


def _image_count(page: fitz.Page) -> int:
    try:
        return len(page.get_images(full=True))
    except Exception:
        return 0


def _wants_ocr(layer: str, mode: str, force_ocr: bool) -> bool:
    """Would this page be sent to OCR, ignoring whether OCR actually works?"""
    if force_ocr:
        return True
    if mode == "always":
        return True
    if mode == "never":
        return False
    # "auto" (and any unrecognised value, which fails safe towards accuracy):
    # only pages we cannot read or cannot trust.
    return layer in (LAYER_EMPTY, LAYER_UNRELIABLE)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def _normalise_unicode(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").translate(_TRANSLATION)


def _page_lines(text: str) -> list[str]:
    return [line.strip() for line in _normalise_unicode(text).split("\n")]


def _repeat_key(line: str) -> str:
    """Normalised form used to match a header across pages.

    Digits are dropped so "Page 12" and "Page 13" — or a chapter number that
    ticks up — collapse onto the same key.
    """
    stripped = _DIGIT_RE.sub("", line)
    stripped = _PUNCT_RE.sub(" ", stripped)
    return _WS_RE.sub(" ", stripped).strip().lower()


def _edge_keys(lines: list[str]) -> set[str]:
    """Repeat-keys of the outermost non-empty lines of one page."""
    filled = [line for line in lines if line]
    if not filled:
        return set()

    edges = filled[:_EDGE_DEPTH] + filled[-_EDGE_DEPTH:]
    # Short keys ("ch", "") are too weak to be evidence of a running head;
    # standalone folios are handled separately by the folio matchers.
    return {key for key in (_repeat_key(line) for line in edges) if len(key) >= 3}


def _find_running_lines(pages: list[list[str]]) -> set[str]:
    n_pages = len(pages)
    if n_pages < _MIN_PAGES_FOR_REPEAT:
        return set()

    counts: dict[str, int] = {}
    for lines in pages:
        for key in _edge_keys(lines):
            counts[key] = counts.get(key, 0) + 1

    threshold = max(2, math.ceil(_REPEAT_RATIO * n_pages))
    return {key for key, count in counts.items() if count >= threshold}


def _is_folio(line: str) -> bool:
    return bool(_FOLIO_RE.match(line) or _BARE_FOLIO_RE.match(line))


def _strip_furniture(lines: list[str], running: set[str]) -> list[str]:
    """Blank out running headers/footers and folios at both page edges.

    Lines are blanked rather than deleted so the surrounding blank-line
    structure — and therefore the paragraph boundaries — stays intact.
    """
    out = list(lines)

    def _peel(order: list[int]) -> None:
        for i in order:
            line = out[i]
            if _is_folio(line) or _repeat_key(line) in running:
                out[i] = ""
            else:
                # Stop at the first real line: anything further in is body text.
                break

    filled = [i for i, line in enumerate(out) if line]
    _peel(filled[:_EDGE_DEPTH])
    # Recompute, so the tail pass never trips over a line the head pass blanked.
    filled = [i for i, line in enumerate(out) if line]
    _peel(list(reversed(filled[-_EDGE_DEPTH:])))

    return out


def _mend_hyphen(match: re.Match[str]) -> str:
    """Decide whether a hyphen at a line end split a word or joined a compound.

    Only a genuine mid-word break gets the hyphen removed. Everything else is
    closed up but keeps its hyphen, because "state-of-\\nthe-art" and
    "Fourier-\\nTransform" are one token that happened to wrap.
    """
    head, nxt = match.group(1), match.group(2)

    compound = (
        "-" in head  # already hyphenated: "state-of-" continues a compound
        or not nxt.islower()  # "Fourier-Transform", "Type-2"
        or len(head) < 2  # initialisms: "e-mail", "x-axis"
        or not head.isalpha()
        or head.lower() in _COMPOUND_PREFIXES
    )
    return f"{head}-{nxt}" if compound else f"{head}{nxt}"


def _reflow(lines: list[str]) -> str:
    """Turn cleaned lines into paragraphs with wrapping newlines removed."""
    text = _BLANK_RUN_RE.sub("\n\n", "\n".join(lines).strip("\n"))
    text = _HYPHEN_BREAK_RE.sub(_mend_hyphen, text)
    text = _SOFT_NEWLINE_RE.sub(" ", text)
    text = _SPACE_RUN_RE.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _clean_pages(raw: list[str]) -> list[PageText]:
    """Clean every page together, since headers are found across pages.

    OCR output goes through exactly the same shaping as text-layer output, so
    downstream stages never have to ask where a page came from.
    """
    per_page_lines = [_page_lines(text) for text in raw]
    running = _find_running_lines(per_page_lines)
    return [
        PageText(page_number=i + 1, text=_reflow(_strip_furniture(lines, running)))
        for i, lines in enumerate(per_page_lines)
    ]


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------


def _scan(
    doc: fitz.Document,
    mode: str,
    force_ocr: bool,
    ocr_ok: bool,
    with_fallback: bool,
) -> list[dict[str, Any]]:
    """Classify every page and decide its route.

    ``with_fallback`` also extracts the span-reordered variant of unreliable
    pages. Only ``load_pdf`` needs it; ``analyze_pdf`` would be paying for a
    second full extraction of every Arabic page to report a number.
    """
    rows: list[dict[str, Any]] = []
    for index in range(doc.page_count):
        # A page that will not even load is, for routing purposes, a scan:
        # there is nothing to read and OCR is the only hope.
        row: dict[str, Any] = {
            "page": index + 1,
            "chars": 0,
            "images": 0,
            "rtl_ratio": 0.0,
            "layer": LAYER_EMPTY,
            "will_ocr": ocr_ok,
            # Carried privately so load_pdf need not extract twice.
            "text": "",
            "_wants_ocr": True,
            "_rtl_text": "",
        }

        try:
            page = doc.load_page(index)
        except Exception:
            rows.append(row)
            continue

        text = _plain_text(page)
        layer, chars, ratio = _classify(text)
        wants = _wants_ocr(layer, mode, force_ocr)

        row.update(
            {
                "chars": chars,
                "images": _image_count(page),
                "rtl_ratio": round(ratio, 4),
                "layer": layer,
                "will_ocr": wants and ocr_ok,
                "text": text,
                "_wants_ocr": wants,
                "_rtl_text": (
                    _rtl_ordered_text(page)
                    if with_fallback and layer == LAYER_UNRELIABLE
                    else ""
                ),
            }
        )
        rows.append(row)
    return rows


def analyze_pdf(
    path: Path | str | None = None,
    *,
    pdf_bytes: bytes | None = None,
    force_ocr: bool = False,
) -> dict[str, Any]:
    """Report what ``load_pdf`` would do with this file, without doing it.

    Returns ``{"pages": [{"page", "chars", "images", "rtl_ratio", "layer",
    "will_ocr"}], "summary": {...}}``. Pass ``force_ocr=True`` to preview
    what a forced OCR ingest would do.
    """
    if pdf_bytes is None:
        if path is None:
            raise ValueError("analyze_pdf requires path or pdf_bytes")
        p = Path(path)
        _, doc = _open(p)
        doc_name = p.name
    else:
        label = Path(path).name if path else "upload.pdf"
        _, doc = _open(Path(label), pdf_bytes)
        doc_name = label

    mode = (config.OCR_MODE or "auto").strip().lower()
    ocr_ok = ocr.is_available()

    try:
        rows = _scan(doc, mode, force_ocr, ocr_ok, with_fallback=False)
    finally:
        doc.close()

    pages = [
        {k: row[k] for k in ("page", "chars", "images", "rtl_ratio", "layer", "will_ocr")}
        for row in rows
    ]
    counts = {LAYER_OK: 0, LAYER_EMPTY: 0, LAYER_UNRELIABLE: 0}
    for row in rows:
        counts[row["layer"]] += 1

    # Pages that need OCR and will not get it are the ones whose text is
    # either missing or known-corrupt — the number worth surfacing in the UI.
    degraded = [row["page"] for row in rows if row["_wants_ocr"] and not ocr_ok]

    return {
        "pages": pages,
        "summary": {
            "doc_name": doc_name,
            "page_count": len(rows),
            "ocr_mode": mode,
            "force_ocr": force_ocr,
            "ocr_available": ocr_ok,
            "ocr_reason": ocr.ocr_available_reason(),
            "layer_ok": counts[LAYER_OK],
            "layer_empty": counts[LAYER_EMPTY],
            "layer_unreliable": counts[LAYER_UNRELIABLE],
            "pages_to_ocr": [row["page"] for row in rows if row["will_ocr"]],
            "degraded_pages": degraded,
            "total_chars": sum(row["chars"] for row in rows),
        },
    }


# --------------------------------------------------------------------------
# Load reports
# --------------------------------------------------------------------------


def _store_report(doc_id: str, report: dict[str, Any]) -> None:
    with _reports_lock:
        _reports[doc_id] = report
        while len(_reports) > _REPORT_LIMIT:
            _reports.pop(next(iter(_reports)))


def last_load_report(doc_id: str) -> dict[str, Any]:
    """What ``load_pdf`` actually did for ``doc_id``, or ``{}`` if unknown.

    In-memory only, and bounded: a report survives the process, not a restart.
    Treat a missing entry as "no information", never as "nothing happened".
    """
    with _reports_lock:
        report = _reports.get(doc_id)
    return dict(report) if report else {}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def load_pdf(
    path: Path | str,
    *,
    pdf_bytes: bytes | None = None,
    force_ocr: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> tuple[str, str, list[PageText]]:
    """Extract one PDF into ``(doc_id, doc_name, pages)``.

    ``doc_id`` is derived from the file's content hash, so re-uploading the
    same PDF re-ingests over the existing entries instead of duplicating them.

    Each page is routed independently: pages with a trustworthy text layer are
    read straight out of the PDF, pages that are empty or substantially RTL go
    to OCR (subject to ``config.OCR_MODE`` and ``force_ocr``). A scanned page
    is no longer an error — when OCR cannot run, the page degrades rather than
    failing the whole document, and the reason lands in the load report.

    Raises ``ValueError`` only for a missing file, a non-PDF, a PDF that will
    not decrypt with an empty password, and the one unrecoverable case: every
    page came out empty and OCR was unavailable.
    """
    p = Path(path)
    data, doc = _open(p, pdf_bytes)

    mode = (config.OCR_MODE or "auto").strip().lower()
    ocr_ok = ocr.is_available()

    try:
        rows = _scan(doc, mode, force_ocr, ocr_ok, with_fallback=True)
    finally:
        doc.close()

    notes: list[str] = []
    if force_ocr and not ocr_ok:
        notes.append(f"force_ocr was requested but ignored: {ocr.ocr_available_reason()}")

    ocr_targets = [row["page"] for row in rows if row["will_ocr"]]
    ocr_text: dict[int, str] = {}
    if ocr_targets:
        # ocr_pdf_pages re-opens the file itself and never raises: a page it
        # cannot read comes back as "" and falls through to the text layer.
        ocr_text = ocr.ocr_pdf_pages(p, ocr_targets, progress_cb=progress_cb)

    raw: list[str] = []
    for row in rows:
        page_number = row["page"]
        recognised = (ocr_text.get(page_number) or "").strip()

        if recognised:
            row["source"] = SOURCE_OCR
            raw.append(recognised)
            continue

        if row["will_ocr"]:
            notes.append(f"page {page_number}: OCR returned nothing; used the text layer")

        if row["layer"] == LAYER_UNRELIABLE:
            # Span-order repair only — the ligature corruption survives, so
            # this page's text is usable but not trustworthy.
            row["source"] = SOURCE_TEXT_RTL
            raw.append(row["_rtl_text"] or row["text"])
            notes.append(
                f"page {page_number}: RTL text layer used with spans reordered; "
                "lam-alef ligatures remain corrupted — OCR is recommended"
            )
        elif row["layer"] == LAYER_EMPTY and not row["text"].strip():
            row["source"] = SOURCE_EMPTY
            raw.append("")
        else:
            row["source"] = SOURCE_TEXT
            raw.append(row["text"])

    pages = _clean_pages(raw)

    empty_pages = [page.page_number for page in pages if not page.text]
    if len(empty_pages) == len(pages) and not ocr_ok:
        raise ValueError(
            f"No text could be extracted from {p.name}: every one of its "
            f"{len(pages)} page(s) is empty and OCR is unavailable. "
            f"{ocr.ocr_available_reason()} Enable OCR (RAG_ENABLE_OCR=1) and "
            "install surya-ocr, then re-upload the file."
        )

    doc_id = _stable_id(hashlib.sha256(data).hexdigest())

    ocred = [row["page"] for row in rows if row["source"] == SOURCE_OCR]
    degraded = [row["page"] for row in rows if row["source"] in (SOURCE_TEXT_RTL, SOURCE_EMPTY)]
    if degraded and not ocr_ok:
        notes.append(
            f"{len(degraded)} page(s) fell back to the raw text layer: "
            f"{ocr.ocr_available_reason()}"
        )

    page_by_number = {page.page_number: page.text for page in pages}
    text_previews = [
        {
            "page": page_number,
            "source": SOURCE_OCR,
            "preview": (page_by_number.get(page_number) or "")[:800],
        }
        for page_number in ocred[:12]
    ]

    _store_report(
        doc_id,
        {
            "doc_id": doc_id,
            "doc_name": p.name,
            "path": str(p),
            "page_count": len(pages),
            "ocr_mode": mode,
            "force_ocr": force_ocr,
            "ocr_available": ocr_ok,
            "ocr_reason": ocr.ocr_available_reason(),
            "ocr_pages": ocred,
            "degraded_pages": degraded,
            "empty_pages": empty_pages,
            "text_previews": text_previews,
            "pages": [
                {
                    "page": row["page"],
                    "chars": row["chars"],
                    "images": row["images"],
                    "rtl_ratio": row["rtl_ratio"],
                    "layer": row["layer"],
                    "will_ocr": row["will_ocr"],
                    "source": row["source"],
                }
                for row in rows
            ],
            "notes": notes,
        },
    )

    return doc_id, p.name, pages


def load_pdfs(
    paths: Iterable[Path | str],
    *,
    force_ocr: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> list[tuple[str, str, list[PageText]]]:
    """Load several PDFs in order. Any unreadable file raises immediately."""
    return [load_pdf(path, force_ocr=force_ocr, progress_cb=progress_cb) for path in paths]


def page_count(path: Path | str, pdf_bytes: bytes | None = None) -> int:
    """Page count without paying for text extraction."""
    if pdf_bytes is not None:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Could not open PDF {Path(path).name}: {exc}") from exc
        try:
            return doc.page_count
        finally:
            doc.close()
    _, doc = _open(Path(path))
    try:
        return doc.page_count
    finally:
        doc.close()


__all__ = [
    "load_pdf",
    "load_pdfs",
    "page_count",
    "analyze_pdf",
    "last_load_report",
    "LAYER_OK",
    "LAYER_EMPTY",
    "LAYER_UNRELIABLE",
]
