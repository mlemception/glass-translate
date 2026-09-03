"""Source-language detection for OCR'd text.

Two stages: cheap Unicode-script heuristics decide unambiguous scripts
(Japanese, Korean, Chinese, Cyrillic, Arabic, Greek, Thai, Hebrew); anything
written in Latin script goes to ``py3langid`` restricted to a configurable set
of languages, so that a screenshot never gets classified as e.g. Volapük.
"""
from __future__ import annotations

import logging
import threading
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from glasstranslate.core.interfaces import LanguageDetector

log = logging.getLogger(__name__)

# Latin-script languages considered by py3langid unless overridden.  All are
# ISO-639-1 codes (py3langid also knows many 3-letter codes we never want).
DEFAULT_LATIN_LANGUAGES: Tuple[str, ...] = (
    "en", "de", "fr", "es", "it", "pt", "nl", "pl", "tr", "sv", "da", "no",
    "fi", "cs", "hu", "ro", "id", "vi",
)

# (first code point, last code point, language) for scripts that map to a
# single language for our purposes.  CJK ideographs are handled separately
# because they are shared between Japanese and Chinese.
_SCRIPT_RANGES: Tuple[Tuple[int, int, str], ...] = (
    (0x3040, 0x30FF, "ja"),  # Hiragana + Katakana
    (0x31F0, 0x31FF, "ja"),  # Katakana phonetic extensions
    (0xFF66, 0xFF9F, "ja"),  # half-width Katakana
    (0x1100, 0x11FF, "ko"),  # Hangul Jamo
    (0x3130, 0x318F, "ko"),  # Hangul compatibility Jamo
    (0xAC00, 0xD7AF, "ko"),  # Hangul syllables
    (0x0400, 0x04FF, "ru"),  # Cyrillic
    (0x0500, 0x052F, "ru"),  # Cyrillic supplement
    (0x0600, 0x06FF, "ar"),  # Arabic
    (0x0750, 0x077F, "ar"),  # Arabic supplement
    (0x0370, 0x03FF, "el"),  # Greek and Coptic
    (0x1F00, 0x1FFF, "el"),  # Greek extended
    (0x0E00, 0x0E7F, "th"),  # Thai
    (0x0590, 0x05FF, "he"),  # Hebrew
)
_CJK_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0x3400, 0x4DBF),  # extension A
    (0x20000, 0x2A6DF),  # extension B
    (0xF900, 0xFAFF),  # compatibility ideographs
)
_CJK = "cjk"
_LATIN = "latin"
_OTHER = "other"


def _script_of(ch: str) -> str:
    """Bucket one alphabetic character into a language code or script tag."""
    cp = ord(ch)
    for lo, hi, lang in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return lang
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return _CJK
    if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF5A:
        return _LATIN  # basic/extended Latin, Latin extended additional, full-width
    return _OTHER


def _letter_scripts(text: str) -> Counter:
    """Count alphabetic characters per script bucket; digits and symbols are
    ignored so "42%" contributes nothing."""
    return Counter(_script_of(ch) for ch in unicodedata.normalize("NFC", text) if ch.isalpha())


class ScriptLanguageDetector(LanguageDetector):
    """Unicode-script heuristics plus ``py3langid`` for Latin-script text.

    Args:
        languages: Latin-script ISO-639-1 codes ``py3langid`` may choose from.
            Codes the installed model does not know are ignored.
        min_confidence: minimum normalised ``py3langid`` probability for a
            Latin-script verdict; below it :meth:`detect` returns None.
    """

    def __init__(
        self,
        languages: Optional[Iterable[str]] = None,
        min_confidence: float = 0.3,
    ) -> None:
        self.languages: Tuple[str, ...] = tuple(dict.fromkeys(languages or DEFAULT_LATIN_LANGUAGES))
        self.min_confidence = float(min_confidence)
        self._identifier: Any = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ py3langid
    def _langid(self) -> Any:
        """Lazily build a private ``LanguageIdentifier`` (the module-level
        ``py3langid.classify`` mutates global state, so we avoid it)."""
        with self._lock:
            if self._identifier is None:
                from py3langid.langid import MODEL_FILE, LanguageIdentifier

                identifier = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
                known = set(identifier.labels)
                langs = [code for code in self.languages if code in known]
                dropped = set(self.languages) - set(langs)
                if dropped:
                    log.warning("py3langid does not know %s; ignoring", sorted(dropped))
                if langs:
                    identifier.set_languages(langs)
                self._identifier = identifier
            return self._identifier

    def _detect_latin(self, text: str) -> Optional[str]:
        lang, score = self._langid().classify(text)
        if score < self.min_confidence:
            return None
        return str(lang)

    # ------------------------------------------------------------ public API
    def detect(self, text: str) -> Optional[str]:
        """ISO-639-1 code for ``text`` or None if it has fewer than three
        letters or the classifier is not confident enough."""
        scripts = _letter_scripts(text)
        total = sum(scripts.values())
        if total < 3:
            return None

        latin = scripts.get(_LATIN, 0)
        cjk = scripts.get(_CJK, 0)
        kana = scripts.get("ja", 0)
        # Kana is unmistakably Japanese; ideographs alone default to Chinese.
        if kana:
            scripts["ja"] = kana + cjk
        elif cjk:
            scripts["zh"] = cjk
        scripts.pop(_CJK, None)
        scripts.pop(_OTHER, None)
        scripts.pop(_LATIN, None)

        if scripts:
            lang, count = scripts.most_common(1)[0]
            if count >= latin:  # ties go to the non-Latin script
                return lang
        if latin >= 3:
            return self._detect_latin(text)
        return None

    def detect_dominant(self, texts: Sequence[str]) -> Optional[str]:
        """Majority vote over ``texts`` weighted by letter count, so a page of
        German with one English button label is German.  None if nothing was
        detectable."""
        votes: Dict[str, int] = {}
        for text in texts:
            lang = self.detect(text)
            if lang is None:
                continue
            weight = sum(1 for ch in text if ch.isalpha())
            votes[lang] = votes.get(lang, 0) + weight
        if not votes:
            return None
        return max(votes.items(), key=lambda kv: kv[1])[0]
