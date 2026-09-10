"""F4/F5 wiring: Gemini factory registration, the series-context push into the translator,
context-scoped caching and provider error reporting.  Fakes only (no network, models or GUI)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pytest

from glasstranslate.config import secrets as S
from glasstranslate.config.settings import AppConfig
from glasstranslate.core.interfaces import LanguageDetector, OCREngine, ScreenCapture, Translator
from glasstranslate.core.pipeline import Pipeline
from glasstranslate.core.types import Frame, Rect, Segment
from glasstranslate.translate import create_translator
from glasstranslate.translate.context import DEFAULT_TEMPLATE, resolve_prompt
from glasstranslate.translate.gemini import GeminiTranslator

W, H = 200, 120


# ----------------------------------------------------------------- fakes
class _Capture(ScreenCapture):
    name = "fake"

    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def grab(self, region: Rect) -> Optional[Frame]:
        return Frame(self.image.copy(), region.x, region.y, 0.0)


class _OCR(OCREngine):
    name = "fake"
    device = "cpu"

    def __init__(self, segments: Sequence[Segment]) -> None:
        self.segments = list(segments)

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        return [Segment(s.text, s.quad.copy(), s.confidence) for s in self.segments]

    def warmup(self) -> None:
        pass


class _Detector(LanguageDetector):
    def detect(self, text: str) -> Optional[str]:
        return "ja"


class _ContextTranslator(Translator):
    """Records ``set_context`` like Gemini does and translates deterministically."""

    name = "ctx"
    device = "remote"
    offline = False

    def __init__(self) -> None:
        self.prompts: List[str] = []
        self.calls: List[List[str]] = []
        self.last_error: Optional[str] = None
        self.fail_with: Optional[str] = None

    def supported_pairs(self):
        return ()

    def supports(self, src: str, tgt: str) -> bool:
        return True

    def set_context(self, system_prompt: str) -> None:
        self.prompts.append(system_prompt)

    def context_key(self) -> str:
        return f"k{len(self.prompts)}:{hash(self.prompts[-1]) if self.prompts else ''}"

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        self.calls.append(list(texts))
        if self.fail_with:
            self.last_error = self.fail_with
            return list(texts)
        self.last_error = None
        return [f"[{len(self.prompts)}] {t}" for t in texts]


class _OfflineTranslator(Translator):
    name = "offline"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def supported_pairs(self):
        return [("ja", "en")]

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        self.calls.append(list(texts))
        return [f"EN:{t}" for t in texts]


def _page() -> tuple[np.ndarray, List[Segment]]:
    img = np.full((H, W, 3), 255, np.uint8)
    quad = np.array([[10, 10], [110, 10], [110, 40], [10, 40]], dtype=np.float32)
    return img, [Segment("先生", quad, 0.9)]


def _pipeline(translator: Translator, cfg: AppConfig) -> tuple[Pipeline, List[str]]:
    img, segs = _page()
    status: List[str] = []
    pipe = Pipeline(
        cfg,
        region_provider=lambda: Rect(0, 0, W, H),
        on_result=lambda segments, stats: None,
        on_status=status.append,
        capture_factory=lambda c: _Capture(img),
        ocr_factory=lambda c: _OCR(segs),
        translator_factory=lambda c: translator,
        detector_factory=_Detector,
    )
    return pipe, status


# ------------------------------------------------------------ factory (F4-3)
def test_factory_builds_gemini_from_the_secret_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(S.SECRETS_FILE_ENV, str(tmp_path / "secrets.json"))
    monkeypatch.delenv(S.GEMINI_API_KEY_ENV, raising=False)
    S.set_gemini_api_key("AIzaFAKE-SECRET-1234567890")
    cfg = AppConfig(
        translation_backend="gemini", gemini_model="gemini-2.5-flash", gemini_api_format="openai",
        gemini_base_url="https://gw.example/", gemini_timeout_s=7.0, gemini_max_retries=1,
    )
    tr = create_translator(cfg)
    assert isinstance(tr, GeminiTranslator)
    assert tr.model == "gemini-2.5-flash" and tr.api_format == "openai" and tr.base_url == "https://gw.example"
    assert tr.name == "gemini" and tr.offline is False
    assert "AIzaFAKE" not in repr(tr)
    assert "gemini_api_key" not in cfg.to_dict() and "AIzaFAKE" not in str(cfg.to_dict())


def test_factory_gemini_without_key_reports_instead_of_raising(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(S.SECRETS_FILE_ENV, str(tmp_path / "none.json"))
    monkeypatch.delenv(S.GEMINI_API_KEY_ENV, raising=False)
    tr = create_translator(AppConfig(translation_backend="gemini"))
    assert tr.translate_batch(["こんにちは"], "ja", "en") == ["こんにちは"]
    assert tr.last_error and "key" in tr.last_error.lower()


# ------------------------------------------------------ pipeline context (F5-2)
def test_pipeline_pushes_resolved_series_prompt_to_translator() -> None:
    tr = _ContextTranslator()
    pipe, _ = _pipeline(tr, AppConfig(source_lang="ja", target_lang="en", series_name="Jujutsu Kaisen"))
    pipe.step()
    assert tr.prompts == [resolve_prompt(DEFAULT_TEMPLATE, "Jujutsu Kaisen").prompt]
    assert "Jujutsu Kaisen" in tr.prompts[0] and "[Series Name]" not in tr.prompts[0]


def test_empty_series_name_strips_placeholder_from_default_prompt() -> None:
    tr = _ContextTranslator()
    pipe, status = _pipeline(tr, AppConfig(source_lang="ja", target_lang="en"))
    pipe.step()
    assert "[Series Name]" not in tr.prompts[0] and "series;" in tr.prompts[0]
    assert not any("invalid" in s.lower() for s in status)  # an empty template means "built-in"


def test_context_change_pushes_new_prompt_without_rebuilding_translator_and_busts_cache() -> None:
    tr = _ContextTranslator()
    cfg = AppConfig(source_lang="ja", target_lang="en", series_name="A")
    pipe, _ = _pipeline(tr, cfg)
    pipe.step()
    pipe.step()  # unchanged frame: cached / skipped, no new translate call
    calls_before = len(tr.calls)
    cfg.series_name = "B"
    pipe.set_config(cfg)
    pipe.step()
    assert len(tr.prompts) == 2 and "B" in tr.prompts[-1]
    assert len(tr.calls) == calls_before + 1  # same text translated again under the new context
    assert pipe._translator is tr  # not rebuilt


def test_malformed_template_falls_back_and_reports_once_for_online_backend() -> None:
    tr = _ContextTranslator()
    cfg = AppConfig(source_lang="ja", target_lang="en", series_prompt_template="no placeholder here")
    pipe, status = _pipeline(tr, cfg)
    pipe.step()
    pipe.step()
    warnings = [s for s in status if "Prompt template invalid" in s]
    assert len(warnings) == 1 and "placeholder" in warnings[0]
    assert tr.prompts[0] == resolve_prompt(DEFAULT_TEMPLATE, "").prompt


def test_offline_backends_ignore_the_context_and_keep_an_empty_cache_key() -> None:
    tr = _OfflineTranslator()
    cfg = AppConfig(source_lang="ja", target_lang="en", series_name="Jujutsu Kaisen",
                    series_prompt_template="broken")
    pipe, status = _pipeline(tr, cfg)
    pipe.step()
    assert tr.context_key() == ""
    assert pipe.cache.get("先生", "ja", "en") == "EN:先生"
    assert not any("Prompt template invalid" in s for s in status)  # offline: no prompt, no warning


def test_translator_error_is_reported_once_per_distinct_error() -> None:
    tr = _ContextTranslator()
    tr.fail_with = "Gemini HTTP 429 (rate limited)"
    cfg = AppConfig(source_lang="ja", target_lang="en")
    pipe, status = _pipeline(tr, cfg)
    pipe.step()
    pipe.invalidate()
    pipe.step()  # failed translations are never cached, so this calls the translator again
    assert len(tr.calls) == 2
    assert status.count("Gemini HTTP 429 (rate limited)") == 1
    tr.fail_with = "Gemini request timed out"
    pipe.invalidate()
    pipe.step()
    assert status.count("Gemini request timed out") == 1
    tr.fail_with = None
    pipe.invalidate()
    pipe.step()  # recovery: no new status line, the result is cached
    assert len([s for s in status if s.startswith("Gemini")]) == 2
