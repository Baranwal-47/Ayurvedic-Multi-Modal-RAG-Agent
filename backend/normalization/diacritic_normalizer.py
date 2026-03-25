"""Normalization utilities for search-friendly text processing."""

import unicodedata


class DiacriticNormalizer:
    """Normalizes Indic diacritics for search while preserving source text elsewhere."""

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

    _TRANSLATION_TABLE = str.maketrans(_DIACRITIC_MAP)

    def normalize(self, text: str) -> str:
        """Return NFC-normalized text with configured diacritic replacements."""
        if not text:
            return ""
        nfc_text = unicodedata.normalize("NFC", text)
        return nfc_text.translate(self._TRANSLATION_TABLE)

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