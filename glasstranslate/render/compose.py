"""PIL-based renderer used by the demo and tests (no Qt dependency).

Draws translated segments onto a BGR image: optionally paints the quad with
the measured background colour, then renders the translation with
``fit_text`` in the foreground colour, rotated by the segment angle about the
quad centre.  Vertical segments stack their characters top-to-bottom.
"""
from __future__ import annotations

import unicodedata

import os
from functools import lru_cache
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.types import Rect, TranslatedSegment
from .fit import FitResult, Measure, fit_text
from .style import quad_text_height, quad_text_width
from .typeset import Typeset, mask_spans, rect_spans, typeset

# Comic-style lettering font bundled with the package (SIL Open Font License).
_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
MANGA_FONT_PATH = os.path.join(_FONT_DIR, "ComicNeue-Bold.ttf")

# Candidate default fonts, best CJK coverage first.
_DEFAULT_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/PingFang.ttc",
)
# Fraction of the font size used as gap between wrapped lines / stacked chars.
_LINE_GAP = 0.1
# Minimum PIL font size that still renders legibly.
_MIN_FONT_PX = 4


def default_font_path() -> Optional[str]:
    """First existing font from the built-in candidate list, else None."""
    for path in _DEFAULT_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def manga_font_path() -> Optional[str]:
    """The bundled comic lettering font used for typeset blocks, if present."""
    return MANGA_FONT_PATH if os.path.exists(MANGA_FONT_PATH) else None


@lru_cache(maxsize=256)
def _load_font(font_path: Optional[str], size_px: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size_px)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:  # very old Pillow: bitmap default font, no size argument
        return ImageFont.load_default()


def _font(font_path: Optional[str], size: float):
    return _load_font(font_path, max(_MIN_FONT_PX, int(round(size))))


def pil_measurer(font_path: Optional[str] = None) -> Measure:
    """Build a ``measure(text, size) -> (width, height)`` callable backed by
    PIL font metrics for ``font_path`` (default: :func:`default_font_path`).
    Height is the font's ascent+descent so line stacking is consistent
    regardless of which glyphs a line contains.
    """
    path = font_path if font_path is not None else default_font_path()

    def measure(text: str, size: float) -> tuple[float, float]:
        font = _font(path, size)
        width = float(font.getlength(text)) if text else 0.0
        ascent, descent = font.getmetrics()
        return width, float(ascent + descent)

    return measure


_VERTICAL_MAX_LINES = 6  # a tall column translated to Latin may need many short lines


def stacks_vertically(text: str) -> bool:
    """True when ``text`` should be drawn one glyph per row (CJK-style).
    Scripts written with spaces between words read badly when stacked, so
    they are laid out horizontally along the column instead."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    cjk = sum(1 for c in letters if unicodedata.east_asian_width(c) in ("W", "F"))
    return cjk >= len(letters) / 2


def _render_text_layer(
    lines: List[str],
    size: float,
    fg: tuple[int, int, int],
    font_path: Optional[str],
    measure: Measure,
    layer_w: int,
    layer_h: int,
) -> Image.Image:
    """Render ``lines`` centred on a transparent ``layer_w`` x ``layer_h`` RGBA
    image (horizontal layout, one line per row)."""
    layer = Image.new("RGBA", (max(1, layer_w), max(1, layer_h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(font_path, size)
    metrics = [measure(line, size) for line in lines]
    line_h = max((h for _, h in metrics), default=size)
    gap = size * _LINE_GAP
    total_h = line_h * len(lines) + gap * (len(lines) - 1)
    y = (layer_h - total_h) / 2.0
    for line, (w, _) in zip(lines, metrics):
        x = (layer_w - w) / 2.0
        draw.text((x, y), line, font=font, fill=(*fg, 255))
        y += line_h + gap
    return layer


def _render_vertical_layer(
    text: str,
    fg: tuple[int, int, int],
    font_path: Optional[str],
    measure: Measure,
    layer_w: int,
    layer_h: int,
) -> Image.Image:
    """Render ``text`` as a single column of stacked characters, sized so the
    column fits the layer both in width and in total height."""
    chars = [c for c in text if not c.isspace()] or [" "]
    n = len(chars)

    def column_fits(size: float) -> bool:
        widest = max(measure(c, size)[0] for c in chars)
        glyph_h = measure("M", size)[1]
        total = glyph_h * n + size * _LINE_GAP * (n - 1)
        return widest <= layer_w and total <= layer_h

    size = max(float(_MIN_FONT_PX), min(layer_w * 0.9, layer_h / n))
    while size > _MIN_FONT_PX and not column_fits(size):
        size = max(size * 0.9, float(_MIN_FONT_PX))

    layer = Image.new("RGBA", (max(1, layer_w), max(1, layer_h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(font_path, size)
    glyph_h = measure("M", size)[1]
    step = glyph_h + size * _LINE_GAP
    total_h = step * n - size * _LINE_GAP
    y = (layer_h - total_h) / 2.0
    for c in chars:
        w, _ = measure(c, size)
        draw.text(((layer_w - w) / 2.0, y), c, font=font, fill=(*fg, 255))
        y += step
    return layer


def block_spans(style, box: Rect) -> np.ndarray:
    """Row spans for a block's layout region (bubble mask or plain box)."""
    if style.layout_mask is not None and style.layout_mask.shape[:2] == (box.h, box.w):
        return mask_spans(style.layout_mask, box)
    return rect_spans(box)


def typeset_block(text: str, style, measure: Measure, *, min_size: float = 7.0) -> Typeset:
    """Typeset ``text`` into a block's layout region using its size ceiling."""
    box = style.layout_box
    assert box is not None
    max_size = style.max_font_px or max(min_size, style.text_height_px * 0.7)
    return typeset(text, block_spans(style, box), float(box.y), measure, max_size=max(max_size, min_size), min_size=min_size)


def _draw_block(
    canvas: Image.Image,
    seg: TranslatedSegment,
    font_path: Optional[str],
    measure: Measure,
    hide_original: bool,
    uppercase: bool,
) -> None:
    """Render a typeset block: erase the original with its clean patch, then
    flow the translation through the bubble/box outline, with a halo when the
    text sits on artwork."""
    style = seg.style
    if hide_original and style.clean_patch is not None and style.clean_rect is not None:
        r = style.clean_rect
        patch = Image.fromarray(np.ascontiguousarray(style.clean_patch[:, :, ::-1])).convert("RGBA")
        canvas.paste(patch, (r.x, r.y))
    text = seg.translation.strip()
    if not text:
        return
    if uppercase:
        text = text.upper()
    ts = typeset_block(text, style, measure)
    if not ts.lines:
        return
    font = _font(font_path, ts.size)
    stroke = max(1, int(round(ts.size * 0.12))) if style.outline else 0
    box = style.layout_box
    assert box is not None
    if abs(style.angle_deg) < 0.5:
        draw = ImageDraw.Draw(canvas)
        for line in ts.lines:
            x = line.cx - line.width / 2.0
            draw.text((x, line.top), line.text, font=font, fill=(*style.fg, 255),
                      stroke_width=stroke, stroke_fill=(*style.bg, 255) if stroke else None)
        return
    # Angled block: draw into a layer the size of the layout box, rotate about its centre.
    layer = Image.new("RGBA", (max(1, box.w), max(1, box.h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for line in ts.lines:
        x = line.cx - line.width / 2.0 - box.x
        draw.text((x, line.top - box.y), line.text, font=font, fill=(*style.fg, 255),
                  stroke_width=stroke, stroke_fill=(*style.bg, 255) if stroke else None)
    _paste_rotated(canvas, layer, style.angle_deg, box.x + box.w / 2.0, box.y + box.h / 2.0)


def _paste_rotated(canvas: Image.Image, layer: Image.Image, angle_deg: float, cx: float, cy: float) -> None:
    """Rotate ``layer`` CCW-on-screen by ``angle_deg`` and alpha-composite it
    onto ``canvas`` centred at ``(cx, cy)``.

    ``Image.rotate`` is counter-clockwise in image coordinates, which is also
    counter-clockwise on screen, so the angle is passed through unchanged.
    """
    rotated = layer.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    x = int(round(cx - rotated.width / 2.0))
    y = int(round(cy - rotated.height / 2.0))
    canvas.alpha_composite(rotated, dest=(max(x, 0), max(y, 0)), source=(max(-x, 0), max(-y, 0)))


def compose(
    img_bgr: np.ndarray,
    segments: List[TranslatedSegment],
    font_path: Optional[str] = None,
    hide_original: bool = True,
    *,
    block_font_path: Optional[str] = None,
    uppercase: bool = True,
) -> np.ndarray:
    """Return a copy of ``img_bgr`` with every segment's translation drawn.

    Typeset blocks (``style.is_block``, produced by ``render.layout``) are
    erased with their clean patch and flowed through their bubble outline in
    the comic lettering font (``block_font_path``, default the bundled Comic
    Neue), upper-cased when ``uppercase`` is set.

    Plain segments: the quad polygon is filled with ``style.bg`` (when
    ``hide_original``), the translation is fitted into the quad's unrotated
    ``width x text_height`` box with :func:`fit_text`, rendered in
    ``style.fg`` and rotated by ``style.angle_deg`` about the quad centre.
    Vertical segments render one character per row.  ``font_path`` None
    picks a system font with CJK coverage when available.
    """
    path = font_path if font_path is not None else default_font_path()
    measure = pil_measurer(path)
    block_path = block_font_path if block_font_path is not None else (manga_font_path() or path)
    block_measure = pil_measurer(block_path)

    canvas = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1])).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    # Blocks first erase, then all draw: a later block's clean patch must not
    # paint over an earlier block's lettering.
    blocks = [seg for seg in segments if seg.style.is_block]
    plain = [seg for seg in segments if not seg.style.is_block]
    if hide_original:
        for seg in blocks:
            st = seg.style
            if st.clean_patch is not None and st.clean_rect is not None:
                r = st.clean_rect
                patch = Image.fromarray(np.ascontiguousarray(st.clean_patch[:, :, ::-1])).convert("RGBA")
                canvas.paste(patch, (r.x, r.y))
    for seg in blocks:
        _draw_block(canvas, seg, block_path, block_measure, hide_original=False, uppercase=uppercase)

    for seg in plain:
        quad = np.asarray(seg.quad, dtype=np.float64).reshape(4, 2)
        style = seg.style
        if hide_original:
            draw.polygon([tuple(p) for p in quad.tolist()], fill=(*style.bg, 255))
        text = seg.translation.strip()
        if not text:
            continue
        cx, cy = (float(v) for v in quad.mean(axis=0))
        box_w = max(1, int(round(quad_text_width(quad))))
        box_h = max(1, int(round(quad_text_height(quad))))
        angle = style.angle_deg
        if style.vertical and stacks_vertically(text):
            layer = _render_vertical_layer(text, style.fg, path, measure, box_w, box_h)
        elif style.vertical:
            # Vertical CJK column translated into a horizontal script (the
            # manga case): lay the text out horizontally along the column,
            # wrapped to the column height, and rotate the layer 90 degrees.
            fit = fit_text(text, box_h, box_w, measure, max_lines=_VERTICAL_MAX_LINES)
            layer = _render_text_layer(fit.lines, fit.size, style.fg, path, measure, box_h, box_w)
            angle -= 90.0  # rotate CW: the text reads top-to-bottom like the column it replaces
        else:
            fit = fit_text(text, box_w, box_h, measure)
            layer = _render_text_layer(fit.lines, fit.size, style.fg, path, measure, box_w, box_h)
        _paste_rotated(canvas, layer, angle, cx, cy)

    rgb = np.asarray(canvas.convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])
