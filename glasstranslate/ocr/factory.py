"""Construct OCR engines by name."""
from __future__ import annotations

from typing import List

from glasstranslate.core.interfaces import OCREngine

_ENGINES = ("rapidocr",)


def available_engines() -> List[str]:
    """Names accepted by :func:`create_ocr`."""
    return list(_ENGINES)


def create_ocr(name: str, device: str, min_confidence: float) -> OCREngine:
    """Build the OCR engine ``name`` (see :func:`available_engines`).

    ``device`` is ``"auto"``, ``"gpu"`` or ``"cpu"``; the engine reports the
    device it actually ended up on through its ``device`` attribute.
    """
    if name == "rapidocr":
        from .rapid import RapidOCREngine

        return RapidOCREngine(device=device, min_confidence=min_confidence)
    raise ValueError(f"unknown OCR engine {name!r}; available: {', '.join(_ENGINES)}")
