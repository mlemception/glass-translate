"""Engine factories and the config-field -> engine map (split out of ``core/pipeline.py``).

``Pipeline`` re-exports every name here, so ``from ..core import pipeline as P; P._accepts_status``
keeps working for the tests.
"""
from __future__ import annotations

import inspect
from typing import Callable, Optional

from ..config.settings import AppConfig
from .interfaces import LanguageDetector, OCREngine, ScreenCapture, Translator

StatusCallback = Callable[[str], None]

# Config fields whose change requires rebuilding an engine.
# ``models_dir`` is where the manga-ocr models live, so moving it rebuilds the
# OCR chain as well as the translator.
_OCR_FIELDS = ("ocr_engine", "ocr_device", "min_confidence", "models_dir")
_TRANSLATOR_FIELDS = (
    "translation_backend",
    "translation_api_key",
    "translation_api_url",
    "translate_device",
    "models_dir",
    "gemini_model",
    "gemini_api_format",
    "gemini_base_url",
    "gemini_timeout_s",
    "gemini_max_retries",
)
_DETECTOR_FIELDS = ("tile_size", "change_threshold")
_LANGUAGE_FIELDS = ("source_lang", "target_lang")
_LAYOUT_FIELDS = ("manga_mode",)
# Series context: pushed to the translator without rebuilding it; the cache key carries the
# translator's context digest, so a change only invalidates the live segments.
_CONTEXT_FIELDS = ("series_name", "series_prompt_template")


def _default_capture_factory(cfg: AppConfig) -> ScreenCapture:
    from ..capture import create_capture

    return create_capture()


def _default_ocr_factory(cfg: AppConfig, status: Optional[StatusCallback] = None) -> OCREngine:
    """Build the configured OCR engine; ``status`` receives the fallback
    chain's transition lines ("OCR: mangaocr failed ...; using paddleocr")."""
    from ..ocr import create_ocr

    return create_ocr(
        cfg.ocr_engine, cfg.ocr_device, cfg.min_confidence, models_dir=cfg.models_dir, status=status
    )


def _accepts_status(factory: Callable[..., object]) -> bool:
    """True when ``factory(cfg, status=...)`` is a valid call (see
    ``_default_ocr_factory``); injected test factories usually take ``cfg`` only."""
    try:
        inspect.signature(factory).bind(None, status=None)
    except (TypeError, ValueError):
        return False
    return True


def _default_translator_factory(cfg: AppConfig) -> Translator:
    from ..translate import create_translator

    return create_translator(cfg)


def _default_detector_factory() -> LanguageDetector:
    from ..ocr import ScriptLanguageDetector

    return ScriptLanguageDetector()
