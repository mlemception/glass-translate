"""Construct OCR engines by name.

Registry
--------
``"mangaocr"`` (default)
    manga-ocr (ONNX) recogniser over the PaddleOCR detector, wrapped in a
    :class:`~glasstranslate.ocr.chain.FallbackOCREngine` that falls back to
    the PaddleOCR engine when the models are missing or the recogniser
    fails.  The ``mangaocr`` module is imported lazily, inside the chain's
    factory, so a broken or absent module is just another reason to fall
    back.
``"paddleocr"``
    ``rapidocr`` running PP-OCRv5 through onnxruntime (PaddleOCR's own
    models), labelled "PaddleOCR (PP-OCRv5 via ONNX)".  ``"rapidocr"`` is
    accepted as an alias for existing config files but not listed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from glasstranslate.config.settings import default_models_dir
from glasstranslate.core.interfaces import OCREngine

from .chain import FallbackOCREngine, StatusCallback
from .rapid import RapidOCREngine

MANGAOCR = "mangaocr"
PADDLEOCR = "paddleocr"
DEFAULT_ENGINE = MANGAOCR

# Registered config names, in the order the UI lists them.
_ENGINES = (MANGAOCR, PADDLEOCR)
# Accepted spellings that map onto a registered name (not listed).
_ALIASES: Dict[str, str] = {"rapidocr": PADDLEOCR}
ENGINE_LABELS: Dict[str, str] = {
    MANGAOCR: "manga-ocr (ONNX)",
    PADDLEOCR: "PaddleOCR (PP-OCRv5 via ONNX)",
}


def available_engines() -> List[str]:
    """Names accepted by :func:`create_ocr` (a fresh list; aliases excluded)."""
    return list(_ENGINES)


def normalize_engine_name(name: str) -> str:
    """Map ``name`` (any case, surrounding space, alias) onto a registered
    engine name.  Raises ``ValueError`` listing the available engines."""
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _ENGINES:
        raise ValueError(f"unknown OCR engine {name!r}; available: {', '.join(_ENGINES)}")
    return key


def engine_label(name: str) -> str:
    """Human-readable label for the UI; unknown names are returned as-is so a
    stale config value still shows up in the engine list."""
    try:
        return ENGINE_LABELS[normalize_engine_name(name)]
    except ValueError:
        return str(name)


def _mangaocr_factory(
    models_dir: Optional[str], detector: OCREngine, device: str, min_confidence: float
) -> Callable[[], OCREngine]:
    """Deferred constructor for the manga-ocr primary (see the module docstring)."""
    resolved = str(default_models_dir()) if models_dir is None else str(models_dir)

    def build() -> OCREngine:
        from .mangaocr import MangaOcrEngine  # lazy: missing module == unavailable primary

        return MangaOcrEngine(resolved, detector, device=device, min_confidence=min_confidence)

    return build


def create_ocr(
    name: str,
    device: str,
    min_confidence: float,
    *,
    models_dir: Optional[str] = None,
    status: Optional[StatusCallback] = None,
) -> OCREngine:
    """Build the OCR engine ``name`` (see :func:`available_engines`).

    ``device`` is ``"auto"``, ``"gpu"`` or ``"cpu"``; the engine reports the
    device it actually ended up on through its ``device`` attribute.
    ``models_dir`` (default :func:`default_models_dir`) is where the manga-ocr
    models live; ``status`` receives the fallback chain's transition lines.
    Raises ``ValueError`` for an unknown name.
    """
    key = normalize_engine_name(name)
    detector = RapidOCREngine(device=device, min_confidence=min_confidence)
    if key == PADDLEOCR:
        return detector
    return FallbackOCREngine(
        primary_factory=_mangaocr_factory(models_dir, detector, device, min_confidence),
        fallback=detector,
        primary_name=MANGAOCR,
        status=status,
    )
