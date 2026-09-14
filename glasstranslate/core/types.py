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
    """Visual style measured from the pixels under a segment.

    The first five fields describe the source text.  The remaining fields are
    filled in by the typesetting layer (``render/layout.py``) when a segment
    is a grouped *text block* rather than a raw OCR line; renderers fall back
    to the plain per-quad behaviour when they are left at their defaults.
    """

    fg: RGB
    bg: RGB
    angle_deg: float  # counter-clockwise positive, in [-90, 90)
    text_height_px: float  # height of the quad along its short axis
    vertical: bool = False  # True for top-to-bottom script layout
    # --- typesetting (optional) -------------------------------------------
    # Axis-aligned area (frame pixels) the translation may occupy.  For text
    # inside a speech bubble this is the bubble's inscribed rectangle, which
    # is usually much wider than the original columns; None = use the quad.
    layout_box: Optional["Rect"] = None
    # Boolean mask of shape (layout_box.h, layout_box.w): which pixels of the
    # layout box are inside the bubble.  None = the whole box.
    layout_mask: Optional[np.ndarray] = None
    # Lay the translation out upright and horizontal regardless of the
    # source orientation (manga: vertical Japanese -> horizontal English).
    upright: bool = False
    # BGR patch covering ``clean_rect`` that shows the frame with the
    # original glyphs erased (filled with bg inside bubbles, inpainted over
    # art).  Renderers paint it before the text instead of a solid box.
    clean_patch: Optional[np.ndarray] = None
    clean_rect: Optional["Rect"] = None
    # Draw the text with a contrasting stroke (bg-coloured halo) because it
    # sits on artwork rather than on a flat background.
    outline: bool = False
    # Font pixel size ceiling derived from the source glyph size so that all
    # blocks on a page share a consistent scale; None = no ceiling.
    max_font_px: Optional[float] = None
    # Quads of the OCR lines whose ink this block covers (N, 4, 2): its member
    # lines *and* its furigana, which is more than the block was assembled
    # from - ruby is erased with the block but never translated.  The
    # footprint they span is what ``render/place._source_rect`` anchors
    # free-text lettering on (``src``, its ``centre``, and the ``must_cover``
    # test for horizontal captions), and it has to be the ink that was erased
    # rather than the text that was read, or the lettering is placed off the
    # hole it is filling.  Including the ruby therefore moves a captioned
    # block's centre by about half the ruby's depth, deliberately.  None for
    # plain single-line segments.
    source_quads: Optional[np.ndarray] = None
    # Preferred footprint for text that is not inside a bubble: the renderer
    # lays the translation out here first and only grows the box (towards
    # ``layout_box``, which is then the growth limit) when the text would
    # otherwise have to shrink below a comfortable size.  None = use
    # ``layout_box`` as is (bubbles: the text is centred in the bubble).
    layout_seed: Optional["Rect"] = None
    # True when the block sits in a detected speech bubble / caption box.
    in_bubble: bool = False
    # --- free-text placement (optional, filled by ``render/place.prepare``) --
    # Frame rectangle the lettering of a free-text block may be placed in
    # (the source text's neighbourhood, bounded by its panel and the page).
    # ``ink_map`` and ``blocked_map`` are both of shape (search_box.h,
    # search_box.w) and aligned with it.  None = ``layout_box`` is the only
    # constraint (the renderer's fallback).
    search_box: Optional["Rect"] = None
    # uint8, 1 where the page carries artwork ("ink") that lettering would
    # hide; the block's own source glyphs count as paper (they are erased).
    ink_map: Optional[np.ndarray] = None
    # bool, True where lettering may not go at all: panel borders and other
    # panels, other blocks' text / bubbles (with a margin) and the half of the
    # gap between neighbouring blocks that belongs to the neighbour.
    blocked_map: Optional[np.ndarray] = None
    # --- quality renderer (optional, see ``render/quality.py``) --------------
    # Bool mask of shape (clean_rect.h, clean_rect.w): the pixels the eraser
    # classified as glyph and painted over.  It is the inpainting mask the
    # sidecar regenerates; None for plain segments and for blocks the eraser
    # never touched.
    erase_mask: Optional[np.ndarray] = None
    # Incremented every time ``clean_patch`` is replaced by an upgrade (a
    # sidecar result).  Renderers cache converted patches per (segment,
    # serial), so a bump - and nothing else - forces a re-conversion.
    clean_patch_serial: int = 0
    # The panel the block lies in (frame coordinates), as ``render/place.py``
    # labels the page; None when no panel could be identified.  One inpainting
    # job is planned per panel.
    panel_box: Optional["Rect"] = None

    @property
    def is_block(self) -> bool:
        """True when the typesetting layer laid this segment out as a block."""
        return self.layout_box is not None

    def shifted(self, dx: int, dy: int) -> "SegmentStyle":
        """Copy with every frame-coordinate field moved by ``(dx, dy)``."""
        if not dx and not dy:
            return self
        delta = np.array([dx, dy], dtype=np.float32)
        return SegmentStyle(
            fg=self.fg,
            bg=self.bg,
            angle_deg=self.angle_deg,
            text_height_px=self.text_height_px,
            vertical=self.vertical,
            layout_box=None if self.layout_box is None else Rect(
                self.layout_box.x + dx, self.layout_box.y + dy, self.layout_box.w, self.layout_box.h
            ),
            layout_mask=self.layout_mask,
            upright=self.upright,
            clean_patch=self.clean_patch,
            clean_rect=None if self.clean_rect is None else Rect(
                self.clean_rect.x + dx, self.clean_rect.y + dy, self.clean_rect.w, self.clean_rect.h
            ),
            outline=self.outline,
            max_font_px=self.max_font_px,
            source_quads=None if self.source_quads is None else self.source_quads + delta,
            layout_seed=None if self.layout_seed is None else Rect(
                self.layout_seed.x + dx, self.layout_seed.y + dy, self.layout_seed.w, self.layout_seed.h
            ),
            in_bubble=self.in_bubble,
            search_box=None if self.search_box is None else Rect(
                self.search_box.x + dx, self.search_box.y + dy, self.search_box.w, self.search_box.h
            ),
            ink_map=self.ink_map,
            blocked_map=self.blocked_map,
            erase_mask=self.erase_mask,
            clean_patch_serial=self.clean_patch_serial,
            panel_box=None if self.panel_box is None else Rect(
                self.panel_box.x + dx, self.panel_box.y + dy, self.panel_box.w, self.panel_box.h
            ),
        )


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

    @property
    def untranslated(self) -> bool:
        """True when the translation is just the source text (no model for
        the pair, an all-``<unk>`` result, or source language == target):
        renderers leave the original pixels alone instead of re-lettering
        the same text."""
        return self.translation.strip() == self.source_text.strip()


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
