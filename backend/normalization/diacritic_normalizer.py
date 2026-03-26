"""Normalization utilities for search-friendly text processing."""

import re
import unicodedata


class DiacriticNormalizer:
    """Normalizes Indic diacritics for search while preserving source text elsewhere."""

    # Safe cleanup for common OCR/typography artifacts seen in scanned PDFs.
    _SAFE_CLEANUP_MAP = {
        "\u00AD": "",   # soft hyphen
        "\u200B": "",   # zero-width space
        "\u200C": "",   # zero-width non-joiner
        "\u200D": "",   # zero-width joiner
        "\uFEFF": "",   # BOM
        "\u00A0": " ",  # non-breaking space
        "\u2019": "'",  # right single quote
        "\u2018": "'",  # left single quote
        "\u201C": '"',   # left double quote
        "\u201D": '"',   # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # ellipsis
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",
    }

    # Multi-character transliteration patterns must run before single-char mapping.
    _MULTI_CHAR_MAP = {
        "ṭh": "th",
        "Ṭh": "Th",
        "ṭH": "tH",
        "ṬH": "TH",
        "ḍh": "dh",
        "Ḍh": "Dh",
        "ḍH": "dH",
        "ḌH": "DH",
    }

    _BASE_DIACRITIC_MAP = {
        "ā": "a",
        "ī": "i",
        "ū": "u",
        "ē": "e",
        "ō": "o",
        "ṣ": "s",
        "ś": "s",
        "ḥ": "h",
        "ṃ": "m",
        "ṭ": "t",
        "ḍ": "d",
        "ṇ": "n",
        "ñ": "n",
        "ṅ": "n",
        "ṉ": "n",
        "ṛ": "r",
        "ṟ": "r",
        "ḷ": "l",
        "ḻ": "l",
        "ẓ": "z",
    }

    _UPPERCASE_DIACRITIC_MAP = {
        k.upper(): v.upper()
        for k, v in _BASE_DIACRITIC_MAP.items()
    }

    _DIACRITIC_MAP = {
        **_BASE_DIACRITIC_MAP,
        **_UPPERCASE_DIACRITIC_MAP,
    }

    _OPTIONAL_LATIN_FOLD_MAP = {
        "à": "a", "á": "a", "ä": "a", "ã": "a", "å": "a",
        "è": "e", "é": "e", "ë": "e",
        "ì": "i", "í": "i", "ï": "i",
        "ò": "o", "ó": "o", "ö": "o", "õ": "o", "ø": "o",
        "ù": "u", "ú": "u", "ü": "u",
        "ý": "y", "ÿ": "y",
        "ç": "c",
        "ś": "s", "š": "s",
        "ź": "z", "ž": "z", "ż": "z",
        "ł": "l",
        "À": "A", "Á": "A", "Ä": "A", "Ã": "A", "Å": "A",
        "È": "E", "É": "E", "Ë": "E",
        "Ì": "I", "Í": "I", "Ï": "I",
        "Ò": "O", "Ó": "O", "Ö": "O", "Õ": "O", "Ø": "O",
        "Ù": "U", "Ú": "U", "Ü": "U",
        "Ç": "C",
    }

    _SAFE_CLEANUP_TABLE = str.maketrans(_SAFE_CLEANUP_MAP)
    _TRANSLATION_TABLE = str.maketrans(_DIACRITIC_MAP)
    _OPTIONAL_LATIN_FOLD_TABLE = str.maketrans(_OPTIONAL_LATIN_FOLD_MAP)

    _MULTI_CHAR_REGEX = re.compile(
        "|".join(sorted((re.escape(k) for k in _MULTI_CHAR_MAP.keys()), key=len, reverse=True))
    )

    def normalize(self, text: str, aggressive_latin_fold: bool = False) -> str:
        """Return NFC-normalized text with search-friendly cleanup and diacritic replacements."""
        if not text:
            return ""

        nfc_text = unicodedata.normalize("NFC", text)
        normalized = nfc_text.translate(self._SAFE_CLEANUP_TABLE)

        normalized = self._MULTI_CHAR_REGEX.sub(
            lambda m: self._MULTI_CHAR_MAP.get(m.group(0), m.group(0)),
            normalized,
        )

        normalized = normalized.translate(self._TRANSLATION_TABLE)

        if aggressive_latin_fold:
            normalized = normalized.translate(self._OPTIONAL_LATIN_FOLD_TABLE)

        # Remove OCR artifacts like "A \ rilal" while preserving regular punctuation.
        normalized = re.sub(r"\s+\\\s+", " ", normalized)
        return normalized

    def detect_script(self, text: str) -> str:
        """
        Return the dominant script found in text.

        Checks character by character and returns on the first match, so the
        dominant script is whichever appears earliest in the string.  For
        purely Latin / romanized transliteration (IAST, ISO 15919, Unani) it
        returns "latin".

        Return values: "devanagari" | "tamil" | "telugu" | "bengali" |
                       "arabic" | "latin"
        """
        if not text:
            return "latin"

        nfc_text = unicodedata.normalize("NFC", text)
        for ch in nfc_text:
            cp = ord(ch)
            # Devanagari (Sanskrit, Hindi, Marathi)
            if 0x0900 <= cp <= 0x097F:
                return "devanagari"
            # Devanagari Extended
            if 0xA8E0 <= cp <= 0xA8FF:
                return "devanagari"
            # Tamil
            if 0x0B80 <= cp <= 0x0BFF:
                return "tamil"
            # Telugu
            if 0x0C00 <= cp <= 0x0C7F:
                return "telugu"
            # Bengali / Assamese
            if 0x0980 <= cp <= 0x09FF:
                return "bengali"
            # Arabic / Urdu / Persian
            if 0x0600 <= cp <= 0x06FF:
                return "arabic"

        return "latin"

    def is_devanagari(self, text: str) -> bool:
        """
        Convenience wrapper kept for backward compatibility.
        Prefer detect_script() for new code.
        """
        return self.detect_script(text) == "devanagari"