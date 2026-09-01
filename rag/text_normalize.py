"""Shared text normalisation for multilingual (esp. Arabic) retrieval.

BM25 and question dedupe both need a stable token form. Arabic has several
equivalent letter shapes and optional diacritics that otherwise fragment the
same word into different tokens.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Alef variants → bare alef; yeh/alef maksura; teh marbuta → heh.
_ARABIC_LETTER_MAP = str.maketrans(
    {
        "\u0622": "\u0627",  # ALEF WITH MADDA ABOVE
        "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE
        "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW
        "\u0671": "\u0627",  # ALEF WASLA
        "\u0649": "\u064a",  # ALEF MAKSURA → YEH
        "\u06cc": "\u064a",  # FARSI YEH → YEH
        "\u0629": "\u0647",  # TEH MARBUTA → HEH
        "\u06c1": "\u0647",  # HEH GOAL → HEH
    }
)

#: Arabic / Persian / Urdu combining marks (tashkeel) and tatweel.
_ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]"
)

#: Eastern Arabic / Persian digits → ASCII.
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def fold_arabic(text: str) -> str:
    """NFC + alef/ya/ta-marbuta folding, diacritic and tatweel strip, digit fold."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ARABIC_LETTER_MAP)
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.translate(_DIGIT_MAP)
    return text


def normalize_for_match(text: str) -> str:
    """Casefolded, Arabic-folded, punctuation-free token stream for equality."""
    folded = fold_arabic(text or "")
    return " ".join(m.group(0).casefold() for m in _TOKEN_RE.finditer(folded))


def tokenize(text: str) -> list[str]:
    """Tokens for BM25 / overlap — Arabic-aware, casefolded."""
    folded = fold_arabic(text or "")
    return _TOKEN_RE.findall(folded.casefold())


__all__ = ["fold_arabic", "normalize_for_match", "tokenize"]
