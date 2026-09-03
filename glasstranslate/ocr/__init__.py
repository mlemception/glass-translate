"""OCR engines and source-language detection."""
from .factory import available_engines, create_ocr
from .langdetect import ScriptLanguageDetector
from .rapid import RapidOCREngine

__all__ = ["RapidOCREngine", "ScriptLanguageDetector", "available_engines", "create_ocr"]
