"""Abstract interfaces implemented by the pluggable engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence

import numpy as np

from .types import Frame, Rect, Segment


class ScreenCapture(ABC):
    """Grabs the pixels of a screen region.  Must never include the overlay
    window (the overlay excludes itself via SetWindowDisplayAffinity on Windows;
    other platforms hide/skip it, see capture/README in module docstring)."""

    name: str = "base"

    @abstractmethod
    def grab(self, region: Rect) -> Optional[Frame]:
        """Return a BGR frame for ``region`` (virtual-screen coords) or None if
        nothing could be captured this tick (e.g. desktop duplication timeout)."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class OCREngine(ABC):
    name: str = "base"
    device: str = "cpu"

    @abstractmethod
    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        """Return text segments with rotated quads and confidences.  Quads are
        in the coordinate system of ``image_bgr``."""

    def warmup(self) -> None:
        self.recognize(np.full((64, 256, 3), 255, np.uint8))


class Translator(ABC):
    """Translation backend.  Implementations must be thread-safe for
    ``translate_batch`` because the pipeline calls it from a worker thread."""

    name: str = "base"
    device: str = "cpu"
    offline: bool = True

    @abstractmethod
    def supported_pairs(self) -> Iterable[tuple[str, str]]:
        """(src, tgt) ISO-639-1 pairs this backend can serve right now."""

    @abstractmethod
    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        """Translate ``texts`` from ``src`` to ``tgt``.  Must return a list the
        same length as ``texts``; return the input unchanged for anything it
        cannot translate rather than raising."""

    def supports(self, src: str, tgt: str) -> bool:
        return (src, tgt) in set(self.supported_pairs())

    def set_context(self, system_prompt: str) -> None:
        """Per-session context (series name / prompt template) for backends that can use it.

        Model backends ignore it; only prompt-driven providers such as Gemini override this.
        """
        return None

    def context_key(self) -> str:
        """Short stable digest of the active context, mixed into the translation cache key so
        results never leak between contexts.  ``""`` when the backend has none."""
        return ""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class LanguageDetector(ABC):
    @abstractmethod
    def detect(self, text: str) -> Optional[str]:
        """ISO-639-1 code or None when undecidable (too short / symbols)."""
