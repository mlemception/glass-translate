"""Shared data contracts for the GlassTranslate pipeline.

Every stage of  capture -> diff -> ocr -> detect -> translate -> render
communicates through the dataclasses below.  Keep this module dependency-free
(numpy only) so every package can import it without pulling in Qt / ONNX.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class Rect:
    """Axis-aligned integer rectangle in *capture-frame* pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def intersects(self, other: "Rect") -> bool:
        return not (
            self.x2 <= other.x or other.x2 <= self.x or self.y2 <= other.y or other.y2 <= self.y
        )

    def clamp(self, width: int, height: int) -> "Rect":
        x = max(0, min(self.x, width))
        y = max(0, min(self.y, height))
        x2 = max(0, min(self.x2, width))
        y2 = max(0, min(self.y2, height))
        return Rect(x, y, max(0, x2 - x), max(0, y2 - y))

    def union(self, other: "Rect") -> "Rect":
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        return Rect(x, y, max(self.x2, other.x2) - x, max(self.y2, other.y2) - y)

    @staticmethod
    def from_quad(quad: np.ndarray) -> "Rect":
        xs = quad[:, 0]
        ys = quad[:, 1]
        x, y = int(np.floor(xs.min())), int(np.floor(ys.min()))
        return Rect(x, y, int(np.ceil(xs.max())) - x, int(np.ceil(ys.max())) - y)


@dataclass
class Segment:
    """One text segment as returned by an OCR engine.

    ``quad`` is a (4, 2) float32 array of corner points in frame pixel
    coordinates, ordered top-left, top-right, bottom-right, bottom-left
    *in the text's own reading frame* (i.e. for rotated text the "top-left" is
    the start of the first character's top edge).
    """

    text: str
    quad: np.ndarray
    confidence: float
    lang_hint: Optional[str] = None  # engine-provided script/language hint, if any

    @property
    def bbox(self) -> Rect:
        return Rect.from_quad(self.quad)


@dataclass
class SegmentStyle:
    """Visual style measured from the pixels under a segment."""

    fg: RGB
    bg: RGB
    angle_deg: float  # counter-clockwise positive, in [-90, 90)
    text_height_px: float  # height of the quad along its short axis
    vertical: bool = False  # True for top-to-bottom script layout


@dataclass
class StyledSegment:
    segment: Segment
    style: SegmentStyle
    src_lang: str  # ISO-639-1 after detection ("en", "ja", ...)


@dataclass
class TranslatedSegment:
    styled: StyledSegment
    translation: str
    tgt_lang: str
    from_cache: bool = False

    @property
    def quad(self) -> np.ndarray:
        return self.styled.segment.quad

    @property
    def style(self) -> SegmentStyle:
        return self.styled.style

    @property
    def source_text(self) -> str:
        return self.styled.segment.text


@dataclass
class Frame:
    """A captured BGR frame plus where it came from on the virtual screen."""

    image: np.ndarray  # HxWx3 uint8, BGR (OpenCV convention)
    origin_x: int  # virtual-screen x of image[0,0]
    origin_y: int
    timestamp: float  # time.perf_counter() at capture

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class PipelineStats:
    """Latency breakdown for one pipeline pass; surfaced in the control window."""

    capture_ms: float = 0.0
    diff_ms: float = 0.0
    ocr_ms: float = 0.0
    style_ms: float = 0.0
    translate_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0
    fps: float = 0.0
    segments: int = 0
    dirty_regions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    skipped_unchanged: bool = False
    ocr_device: str = ""
    translate_device: str = ""
    extra: dict = field(default_factory=dict)
