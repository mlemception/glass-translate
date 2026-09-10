"""OCR engines and source-language detection."""
from .chain import FallbackOCREngine
from .factory import (
    DEFAULT_ENGINE,
    ENGINE_LABELS,
    available_engines,
    create_ocr,
    engine_label,
    normalize_engine_name,
)
from .furigana import furigana_indices, is_kana_text, strip_furigana
from .langdetect import ScriptLanguageDetector
from .rapid import RapidOCREngine

__all__ = [
    "DEFAULT_ENGINE",
    "ENGINE_LABELS",
    "FallbackOCREngine",
    "RapidOCREngine",
    "ScriptLanguageDetector",
    "available_engines",
    "create_ocr",
    "engine_label",
    "furigana_indices",
    "is_kana_text",
    "normalize_engine_name",
    "strip_furigana",
]
