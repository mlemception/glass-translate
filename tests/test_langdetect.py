"""Tests for glasstranslate.ocr.langdetect (no GPU/models/network needed;
py3langid ships its model inside the wheel)."""
from __future__ import annotations

import pytest

from glasstranslate.ocr.langdetect import DEFAULT_LATIN_LANGUAGES, ScriptLanguageDetector


@pytest.fixture(scope="module")
def det() -> ScriptLanguageDetector:
    return ScriptLanguageDetector()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("こんにちは世界", "ja"),  # kana + kanji
        ("設定を開く", "ja"),  # kanji with a single hiragana
        ("カタカナ", "ja"),
        ("안녕하세요 세계", "ko"),
        ("你好世界", "zh"),  # ideographs only
        ("Привет, как дела?", "ru"),
        ("مرحبا بالعالم", "ar"),
        ("Γειά σου Κόσμε", "el"),
        ("สวัสดีชาวโลก", "th"),
        ("שלום עולם", "he"),
    ],
)
def test_script_heuristics(det: ScriptLanguageDetector, text: str, expected: str) -> None:
    assert det.detect(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, how are you doing today?", "en"),
        ("Guten Morgen, wie geht es dir heute?", "de"),
        ("Bonjour, comment allez-vous aujourd'hui?", "fr"),
        ("Hola, ¿cómo estás hoy, amigo mío?", "es"),
    ],
)
def test_latin_via_langid(det: ScriptLanguageDetector, text: str, expected: str) -> None:
    assert det.detect(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "42", "OK", "->", "12:30 %", "a1"])
def test_too_short_returns_none(det: ScriptLanguageDetector, text: str) -> None:
    assert det.detect(text) is None


def test_mixed_script_prefers_non_latin(det: ScriptLanguageDetector) -> None:
    # A Japanese UI string with an embedded product name is still Japanese.
    assert det.detect("Windowsを再起動してください") == "ja"
    # Mostly Latin with one stray Cyrillic letter stays Latin.
    assert det.detect("The quick brown fox jumps over the lazy dog д") == "en"


def test_latin_restricted_to_configured_set() -> None:
    det = ScriptLanguageDetector(languages=["en", "de"])
    assert det.languages == ("en", "de")
    assert det.detect("Bonjour, comment allez-vous aujourd'hui?") in {"en", "de"}
    # Unknown codes are dropped rather than raising.
    lenient = ScriptLanguageDetector(languages=["en", "xx-not-a-language"])
    assert lenient.detect("Hello, how are you doing today?") == "en"


def test_low_confidence_returns_none() -> None:
    strict = ScriptLanguageDetector(min_confidence=1.01)
    assert strict.detect("Hello, how are you doing today?") is None
    # Script heuristics are not subject to the langid threshold.
    assert strict.detect("こんにちは世界") == "ja"


def test_default_languages_are_iso639_1() -> None:
    assert all(len(code) == 2 for code in DEFAULT_LATIN_LANGUAGES)
    assert "en" in DEFAULT_LATIN_LANGUAGES


def test_detect_dominant_weighted_by_length(det: ScriptLanguageDetector) -> None:
    texts = [
        "Die Einstellungen wurden erfolgreich gespeichert und übernommen.",
        "Bitte starten Sie die Anwendung neu, damit alle Änderungen wirksam werden.",
        "OK",  # undecidable, ignored
        "Cancel",  # short English label, outweighed
    ]
    assert det.detect_dominant(texts) == "de"


def test_detect_dominant_by_script(det: ScriptLanguageDetector) -> None:
    assert det.detect_dominant(["こんにちは世界", "設定を開く", "File"]) == "ja"
    assert det.detect_dominant(["Привет, как дела?", "Хорошо, спасибо!", "OK"]) == "ru"


def test_detect_dominant_none_when_nothing_detectable(det: ScriptLanguageDetector) -> None:
    assert det.detect_dominant([]) is None
    assert det.detect_dominant(["42", "%", "OK"]) is None
