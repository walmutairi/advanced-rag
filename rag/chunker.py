"""Turn page-attributed text into page-aware, structure-respecting chunks.

The unit of retrieval in this system is the chunk, so a chunk has to be
self-contained: it carries the section heading it lives under, it never ends
mid-sentence, and it knows exactly which pages it came from (citations are
worthless otherwise).

The sharp edge is PDF text. Extractors emit hard line breaks inside
paragraphs, hyphenate across those breaks, and give no markup at all — so
headings are recovered heuristically and paragraph boundaries are recovered
from blank lines. Every heuristic here degrades to "treat it as body text",
which costs a little retrieval quality but never loses content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from rag.schemas import Chunk, PageText, _stable_id

__all__ = ["chunk_pages"]


# --------------------------------------------------------------------------
# Sentence segmentation
# --------------------------------------------------------------------------

#: Trailing tokens that end in a period without ending a sentence. Kept
#: without their final dot so "e.g." and "Fig." are matched the same way.
_ABBREVIATIONS = {
    "e.g", "i.e", "et", "al", "etc", "cf", "vs", "viz", "approx", "ca",
    "fig", "figs", "eq", "eqs", "ref", "refs", "tab", "sec", "ch", "chap",
    "vol", "no", "nos", "pp", "p", "para", "ed", "eds", "est", "min", "max",
    "avg", "dept", "univ", "inc", "ltd", "co", "corp", "st", "mt",
    "dr", "prof", "mr", "mrs", "ms", "jr", "sr", "u.s", "u.k",
}

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")

#: A boundary is terminal punctuation (plus any closing quote/bracket)
#: followed by whitespace and something that can open a sentence. Decimals
#: like "3.14" are excluded for free by the required whitespace.
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?P<end>[.!?]+[\"'’”)\]]*)\s+(?=[\"'‘“(\[]*[A-Z0-9])"
)

#: The word immediately before a candidate boundary's period.
_TRAILING_WORD_RE = re.compile(r"([A-Za-z][A-Za-z.]*)\.[\"'’”)\]]*$")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "these", "this", "to", "was", "were", "which", "with", "its",
}

_NUMBERED_HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVXLCDM]{1,7}\.)\s+\S")
_WORD_RE = re.compile(r"[A-Za-z][\w'’-]*")


def _ends_with_abbreviation(head: str) -> bool:
    match = _TRAILING_WORD_RE.search(head)
    if not match:
        return False
    word = match.group(1).rstrip(".").lower()
    # A lone letter is an initial ("J. Smith"), not a sentence end.
    return word in _ABBREVIATIONS or len(word) == 1


def _split_paragraph_sentences(para: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(para):
        head = para[start : match.end("end")]
        if _ends_with_abbreviation(head):
            continue
        piece = head.strip()
        if piece:
            sentences.append(piece)
            start = match.end()
    tail = para[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or ([para.strip()] if para.strip() else [])


def _split_sentences(text: str) -> list[str]:
    """Sentence units, with paragraph breaks treated as hard boundaries."""
    units: list[str] = []
    for para in _BLANK_LINE_RE.split(text):
        para = para.strip()
        if para:
            units.extend(_split_paragraph_sentences(para))
    return units


# --------------------------------------------------------------------------
# Heading detection
# --------------------------------------------------------------------------


def _is_heading(raw: str) -> bool:
    """Best-effort structural heading test on an un-normalised paragraph."""
    text = raw.strip()
    if not text or "\n" in text or len(text) >= 120:
        return False
    if text.endswith((".", ",", ";", "!", "?")):
        return False
    words = _WORD_RE.findall(text)
    if not words or len(words) > 14:
        return False

    if _NUMBERED_HEADING_RE.match(text):
        return True
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 3 and text == text.upper():
        return True
    return _is_title_case(words)


def _is_title_case(words: list[str]) -> bool:
    # Every lowercase word must be a stopword, and stopwords must stay a
    # minority — that is what separates "Model Architecture and Training"
    # from a fragment of a sentence that happens to lack a full stop.
    lowered = [w for w in words if w[0].islower()]
    if any(w.lower() not in _STOPWORDS for w in lowered):
        return False
    stops = sum(1 for w in words if w.lower() in _STOPWORDS)
    return stops <= max(1, len(words) // 3)


# --------------------------------------------------------------------------
# Block stream
# --------------------------------------------------------------------------

_HYPHEN_BREAK_RE = re.compile(r"(?<=[a-zà-ÿ])-\n(?=[a-zà-ÿ])")
_WHITESPACE_RE = re.compile(r"[ \t ]+")


@dataclass
class _Block:
    text: str
    page: int
    is_heading: bool


def _normalise_paragraph(raw: str) -> str:
    # PDF line wrapping is layout, not content: join it away, and repair the
    # words it split with a hyphen.
    text = _HYPHEN_BREAK_RE.sub("", raw)
    text = text.replace("\r", "\n").replace("\n", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _hard_wrap(text: str, limit: int) -> list[str]:
    """Last resort for a single sentence larger than the target.

    Splitting mid-sentence is the lesser evil here: an unbroken multi-page
    table dumped as one "sentence" would otherwise produce a chunk too big to
    embed.
    """
    pieces: list[str] = []
    words = text.split(" ")
    current: list[str] = []
    length = 0
    for word in words:
        # A "word" longer than the limit is not prose (a base64 blob, a
        # de-spaced table row); slice it rather than emit an oversized chunk.
        while len(word) > limit:
            if current:
                pieces.append(" ".join(current))
                current, length = [], 0
            pieces.append(word[:limit])
            word = word[limit:]
        if current and length + len(word) + 1 > limit:
            pieces.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        pieces.append(" ".join(current))
    return [p for p in pieces if p]


def _split_oversized(text: str, limit: int) -> list[str]:
    """Break a too-long paragraph into ≤ limit pieces on sentence boundaries."""
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in _split_paragraph_sentences(text):
        if len(sentence) > limit:
            if current:
                pieces.append(" ".join(current))
                current, length = [], 0
            pieces.extend(_hard_wrap(sentence, limit))
            continue
        if current and length + len(sentence) + 1 > limit:
            pieces.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def _flatten(pages: list[PageText]) -> list[_Block]:
    limit = config.CHUNK_TARGET_CHARS
    blocks: list[_Block] = []
    for page in pages:
        if page.page_number < 1:
            raise ValueError(
                f"PageText.page_number must be 1-indexed, got {page.page_number}"
            )
        for raw in _BLANK_LINE_RE.split(page.text or ""):
            if not raw.strip():
                continue
            heading = _is_heading(raw)
            text = _normalise_paragraph(raw)
            if not text:
                continue
            if heading or len(text) <= limit:
                blocks.append(_Block(text, page.page_number, heading))
            else:
                blocks.extend(
                    _Block(piece, page.page_number, False)
                    for piece in _split_oversized(text, limit)
                )
    return blocks


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


@dataclass
class _Draft:
    blocks: list[_Block] = field(default_factory=list)
    section: str = ""
    overlap: str = ""
    overlap_page: int = 0


def _tail_overlap(body: str, budget: int) -> str:
    """Trailing sentences of ``body`` totalling roughly ``budget`` chars.

    The first sentence is always withheld, which is what guarantees an
    overlap can never reproduce an entire preceding chunk.
    """
    sentences = _split_sentences(body)
    if len(sentences) <= 1:
        return ""
    picked: list[str] = []
    total = 0
    for sentence in reversed(sentences[1:]):
        if picked and total + len(sentence) > budget:
            break
        picked.append(sentence)
        total += len(sentence) + 1
    picked.reverse()
    return " ".join(picked)


def _body_text(draft: _Draft) -> str:
    return "\n\n".join(b.text for b in draft.blocks)


def _render(draft: _Draft) -> str:
    parts = [p for p in (draft.overlap, _body_text(draft)) if p]
    text = "\n\n".join(parts).strip()
    # Retrieval sees the chunk alone, so it has to carry its own context.
    if draft.section and not text.startswith(draft.section):
        text = f"Section: {draft.section}\n\n{text}"
    return text


def _pack(blocks: list[_Block]) -> list[_Draft]:
    target = config.CHUNK_TARGET_CHARS
    drafts: list[_Draft] = []

    current = _Draft()
    current_len = 0
    section = ""
    pending_overlap = ""
    pending_page = 0

    def flush() -> None:
        nonlocal current, current_len, pending_overlap, pending_page
        if not current.blocks:
            return
        drafts.append(current)
        pending_overlap = _tail_overlap(_body_text(current), config.CHUNK_OVERLAP_CHARS)
        pending_page = current.blocks[-1].page
        current = _Draft()
        current_len = 0

    for block in blocks:
        if block.is_heading:
            # A heading opens a new section; start a fresh chunk for it unless
            # the one in progress is still too small to stand alone.
            if current.blocks and current_len >= config.CHUNK_MIN_CHARS:
                flush()
            section = block.text
        elif current.blocks and current_len + len(block.text) + 2 > target:
            flush()

        if not current.blocks:
            current.section = section
            current.overlap = pending_overlap
            current.overlap_page = pending_page if pending_overlap else 0
            current_len = len(current.overlap)
            if section and not (block.is_heading and block.text == section):
                current_len += len(section) + len("Section: \n\n")

        current.blocks.append(block)
        current_len += len(block.text) + 2

    flush()
    return drafts


def _merge_trailing_runt(drafts: list[_Draft]) -> list[_Draft]:
    # A one-paragraph document is a legitimate single short chunk; only a
    # *trailing* fragment of a longer document is a runt worth absorbing.
    while len(drafts) > 1 and len(_render(drafts[-1])) < config.CHUNK_MIN_CHARS:
        runt = drafts.pop()
        drafts[-1].blocks.extend(runt.blocks)
    return drafts


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def _with_parent_windows(children: list[Chunk]) -> list[Chunk]:
    """Attach sliding parent windows so retrieval can expand child hits.

    Children stay the retrieval unit. Parents are stored for context expansion
    only (``chunk_kind='parent'``) and are skipped by default search filters.
    """
    if not config.ENABLE_PARENT_CHILD_CHUNKS or len(children) < 2:
        return children

    window = config.PARENT_CHILD_WINDOW
    parents: list[Chunk] = []
    child_out: list[Chunk] = []

    for start in range(0, len(children), window):
        group = children[start : start + window]
        if not group:
            continue
        parent_text = "\n\n".join(c.text for c in group)
        parent_id = _stable_id(
            group[0].doc_id, "parent", str(start), parent_text[:200]
        )
        parents.append(
            Chunk(
                chunk_id=parent_id,
                doc_id=group[0].doc_id,
                doc_name=group[0].doc_name,
                text=parent_text,
                page_start=group[0].page_start,
                page_end=group[-1].page_end,
                section=group[0].section,
                ordinal=start,
                parent_id="",
                chunk_kind="parent",
            )
        )
        for child in group:
            child_out.append(
                Chunk(
                    chunk_id=child.chunk_id,
                    doc_id=child.doc_id,
                    doc_name=child.doc_name,
                    text=child.text,
                    page_start=child.page_start,
                    page_end=child.page_end,
                    section=child.section,
                    ordinal=child.ordinal,
                    parent_id=parent_id,
                    chunk_kind="child",
                )
            )

    return child_out + parents


def chunk_pages(doc_id: str, doc_name: str, pages: list[PageText]) -> list[Chunk]:
    """Chunk one document's pages into page-aware, overlapping chunks."""
    if not doc_id or not doc_name:
        raise ValueError("chunk_pages requires a non-empty doc_id and doc_name")

    blocks = _flatten(pages)
    if not blocks:
        return []

    doc_first = min(p.page_number for p in pages)
    doc_last = max(p.page_number for p in pages)

    chunks: list[Chunk] = []
    for ordinal, draft in enumerate(_merge_trailing_runt(_pack(blocks))):
        text = _render(draft)
        if not text:
            raise AssertionError(f"chunk {ordinal} of {doc_name} rendered empty")

        seen = {b.page for b in draft.blocks}
        if draft.overlap_page:
            seen.add(draft.overlap_page)
        page_start, page_end = min(seen), max(seen)
        if not doc_first <= page_start <= page_end <= doc_last:
            raise AssertionError(
                f"chunk {ordinal} of {doc_name} spans pages "
                f"{page_start}-{page_end}, outside {doc_first}-{doc_last}"
            )

        chunks.append(
            Chunk(
                chunk_id=_stable_id(doc_id, str(ordinal), text[:200]),
                doc_id=doc_id,
                doc_name=doc_name,
                text=text,
                page_start=page_start,
                page_end=page_end,
                section=draft.section,
                ordinal=ordinal,
                parent_id="",
                chunk_kind="child",
            )
        )

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    return _with_parent_windows(chunks)
