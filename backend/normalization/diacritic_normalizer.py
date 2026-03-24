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

	def is_devanagari(self, text: str) -> bool:
		"""Detect whether text contains Devanagari characters."""
		if not text:
			return False

		nfc_text = unicodedata.normalize("NFC", text)
		return any(
			("\u0900" <= ch <= "\u097F") or ("\uA8E0" <= ch <= "\uA8FF")
			for ch in nfc_text
		)
