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

from ..core.types import Rect, SegmentStyle, TranslatedSegment
from .fit import Measure, fit_text
from .style import quad_text_height, quad_text_width
from . import place
from .typeset import Typeset, mask_spans, rect_spans

# Comic lettering font bundled with the package: Anime Ace 2.0 BB (Blambot,
# free for independent comic / non-profit use; see ``fonts/animeace/font
# info.txt``), the standard scanlation lettering face.  Dialogue is lettered
# in the italic (the convention for speech and thought in translated manga);
# horizontal captions and titles stay upright.
_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts", "animeace")
MANGA_FONT_PATH = os.path.join(_FONT_DIR, "animeace2_reg.ttf")
MANGA_FONT_ITALIC_PATH = os.path.join(_FONT_DIR, "animeace2_ital.ttf")
# Halo (stroke) around lettering placed on artwork, as a fraction of the font
# size: about 3 px at dialogue size, paper-coloured, rounded (measured on
# professionally lettered pages: 0.10-0.11 em).
HALO_RATIO = 0.11
# Foreground stroke added to every letter (fraction of the font size) to
# thicken the lettering.  Zero: Anime Ace is drawn at its designed weight.
WEIGHT_RATIO = 0.0

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


def manga_font_path(italic: bool = False) -> Optional[str]:
    """The bundled comic lettering font used for typeset blocks, if present."""
    path = MANGA_FONT_ITALIC_PATH if italic else MANGA_FONT_PATH
    if os.path.exists(path):
        return path
    return MANGA_FONT_PATH if os.path.exists(MANGA_FONT_PATH) else None


def has_cjk(text: str) -> bool:
    """True when ``text`` contains CJK ideographs or kana (which the bundled
    comic font cannot draw; such blocks use the CJK system font instead)."""
    return any(0x2E80 <= ord(c) <= 0x9FFF or 0xF900 <= ord(c) <= 0xFAFF or 0xFF00 <= ord(c) <= 0xFFEF for c in text)


def block_italic(style: SegmentStyle) -> bool:
    """Whether a block is lettered in the italic face.  Nothing is, today.

    This used to return ``style.vertical``, so every speech, thought and
    narration block was set oblique, on the stated convention that translated
    manga letters dialogue in italic.  The corpus does not bear that out.
    Measured as the de-shear angle that best aligns vertical stems, the
    reference edition's own English ink comes out at -2.0, +1.5, -2.5, +0.0
    and -5.0 degrees over five pages, while the two bundled faces measure
    ``animeace2_reg`` +0.0 and ``animeace2_ital`` **+14.0**.  Two blind reviews
    raised the slant unprompted, on different pages, before it was measured.

    The convention is real, but it is for *thought* and flashback, and nothing
    available here tells a thought balloon from a speech balloon.  Defaulting
    every balloon to italic is measurably further from the release than
    defaulting every balloon to roman, so it defaults to roman.  The italic
    face stays bundled and reachable through :func:`manga_font_path` for when
    a signal exists."""
    return False


def block_font_path_for(style: SegmentStyle, text: str, fallback: Optional[str]) -> Optional[str]:
    """Font file for one block's translation: the comic font (italic for
    dialogue) unless the text still contains CJK, then ``fallback``."""
    if has_cjk(text):
        return fallback
    return manga_font_path(block_italic(style)) or fallback


# Punctuation the comic face has no glyph for, and what a letterer sets instead.
# Anything not listed falls through to a compatibility fold, then to deletion.
_FOLD_TABLE = {
    "•": "-",     # BULLET
    "‣": "-",     # TRIANGULAR BULLET
    "●": "-",     # BLACK CIRCLE
    "·": "-",     # MIDDLE DOT (absent from the face too)
    "—": "--",    # EM DASH
    "–": "-",     # EN DASH
    "−": "-",     # MINUS SIGN
    # Latin letters with a stroke: no compatibility decomposition, so the fold below
    # cannot reach them and they would be dropped outright.
    "Ł": "L", "ł": "l",   # L WITH STROKE
    "Ø": "O", "ø": "o",   # O WITH STROKE
    "Đ": "D", "đ": "d",   # D WITH STROKE
}

_PROBE_PX = 32
# A private-use codepoint no real face carries: its bitmap IS ``.notdef``.
_NOTDEF_PROBE = ""


@lru_cache(maxsize=8)
def _notdef_signature(font_path: str) -> tuple:
    mask = _load_font(font_path, _PROBE_PX).getmask(_NOTDEF_PROBE)
    return (mask.size, bytes(mask))


@lru_cache(maxsize=4096)
def _renders(font_path: str, ch: str) -> bool:
    """Whether ``font_path`` has a real glyph for ``ch``.

    FreeType maps an absent codepoint to ``.notdef``, so a character whose bitmap
    matches the private-use probe's is missing.  Cached per face and character: the
    cost is one rasterised glyph the first time a character is ever seen.
    """
    if ch.isspace():
        return True
    try:
        mask = _load_font(font_path, _PROBE_PX).getmask(ch)
        return (mask.size, bytes(mask)) != _notdef_signature(font_path)
    except Exception:
        return False


def unrenderable_chars(text: str, font_path: Optional[str]) -> List[str]:
    """The characters of ``text`` that ``font_path`` would draw as a tofu box."""
    if not font_path:
        return []
    return [ch for ch in text if not _renders(font_path, ch)]


def fold_to_face(text: str, font_path: Optional[str]) -> str:
    """Rewrite ``text`` so every character has a glyph in ``font_path``.

    A ``.notdef`` box is strictly worse on the page than any reasonable substitute, so
    each missing character is, in order: looked up in :data:`_FOLD_TABLE`; decomposed
    and stripped of its combining marks, which turns accented Latin into its base
    letter; and failing both, dropped.

    Called before the text is measured, so line breaking and drawing agree.  Note this
    is a *substitution*, not per-character font fallback -- drawing one bullet from the
    system face would mean threading a second font through measurement and layout.
    """
    if not font_path or not text:
        return text
    if not unrenderable_chars(text, font_path):
        return text
    out = []
    for ch in text:
        if _renders(font_path, ch):
            out.append(ch)
            continue
        sub = _FOLD_TABLE.get(ch)
        if sub is None:
            decomposed = unicodedata.normalize("NFKD", ch)
            sub = "".join(c for c in decomposed if not unicodedata.combining(c))
        if sub and sub != ch and not unrenderable_chars(sub, font_path):
            out.append(sub)
    return "".join(out)


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


@lru_cache(maxsize=32768)
def _measure_cached(font_path: Optional[str], size_px: int, text: str) -> tuple[float, float]:
    """``(advance, ascent + descent)`` of ``text`` in the font at ``size_px``.
    Process-wide: the typesetter re-measures the same few hundred strings on
    every compose / frame, and ``FreeTypeFont.getlength`` costs ~150 us."""
    font = _load_font(font_path, size_px)
    width = float(font.getlength(text)) if text else 0.0
    ascent, descent = font.getmetrics()
    return width, float(ascent + descent)


def pil_measurer(font_path: Optional[str] = None) -> Measure:
    """Build a ``measure(text, size) -> (width, height)`` callable backed by
    PIL font metrics for ``font_path`` (default: :func:`default_font_path`).
    Height is the font's ascent+descent so line stacking is consistent
    regardless of which glyphs a line contains.  Measurements are cached
    process-wide by ``(font, rounded size, text)``.
    """
    path = font_path if font_path is not None else default_font_path()

    def measure(text: str, size: float) -> tuple[float, float]:
        return _measure_cached(path, max(_MIN_FONT_PX, int(round(size))), text)

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


def block_spans(style, box: Rect, cx: Optional[float] = None) -> np.ndarray:
    """Row spans for a block's layout region (bubble mask or plain box)."""
    if style.layout_mask is not None and style.layout_mask.shape[:2] == (box.h, box.w):
        return mask_spans(style.layout_mask, box, cx)
    return rect_spans(box)


def typeset_block(text: str, style, measure: Measure, *, min_size: float = 7.0, lang: str = "en") -> Typeset:
    """Typeset ``text`` for a block: the geometry shared by this PIL renderer
    and the Qt overlay.  Delegates to :func:`glasstranslate.render.place.typeset_block`:
    bubble text is flowed through the bubble mask and centred in it; free
    text is set as a compact block on the quietest nearby patch of the panel
    (see that module).  The size ceiling is :func:`place.size_ceiling`.
    """
    return place.typeset_block(text, style, measure, min_size=min_size, lang=lang)


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
    text sits on artwork.  Untranslated blocks are left untouched."""
    style = seg.style
    if seg.untranslated:
        return
    if hide_original and style.clean_patch is not None and style.clean_rect is not None:
        r = style.clean_rect
        patch = Image.fromarray(np.ascontiguousarray(style.clean_patch[:, :, ::-1])).convert("RGBA")
        canvas.paste(patch, (r.x, r.y))
    text = seg.translation.strip()
    if not text:
        return
    if uppercase and not has_cjk(text):
        text = text.upper()
    text = fold_to_face(text, font_path)
    if not text:
        return
    ts = typeset_block(text, style, measure, lang=seg.tgt_lang or "en")
    if not ts.lines:
        return
    font = _font(font_path, ts.size)
    box = style.layout_box
    assert box is not None
    if abs(style.angle_deg) < 0.5:
        draw_typeset(ImageDraw.Draw(canvas), ts, font, style.fg, style.bg, style.outline)
        return
    # Angled block: draw into a layer covering the layout box and the placed
    # lines, rotate it about the layout box centre.
    ext = box
    bb = ts.bbox
    if bb is not None:
        pad = halo_px(ts.size) + weight_px(ts.size) + 2
        ext = ext.union(Rect(bb.x - pad, bb.y - pad, bb.w + 2 * pad, bb.h + 2 * pad))
    layer = Image.new("RGBA", (max(1, ext.w), max(1, ext.h)), (0, 0, 0, 0))
    draw_typeset(ImageDraw.Draw(layer), ts, font, style.fg, style.bg, style.outline, dx=ext.x, dy=ext.y)
    cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
    # The layer is centred on ``ext``; rotate about the layout box centre.
    _paste_rotated_about(canvas, layer, style.angle_deg, ext.x + ext.w / 2.0, ext.y + ext.h / 2.0, cx, cy)


def halo_px(size: float) -> int:
    """Halo stroke width in pixels for lettering at ``size``."""
    return max(1, int(round(size * HALO_RATIO)))


def weight_px(size: float) -> int:
    """Foreground stroke width (weight) in pixels for lettering at ``size``."""
    return int(round(size * WEIGHT_RATIO))


def draw_typeset(
    draw: ImageDraw.ImageDraw,
    ts: Typeset,
    font,
    fg: tuple[int, int, int],
    bg: tuple[int, int, int],
    outline: bool,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> None:
    """Draw the placed lines of ``ts`` with PIL, offset by ``(-dx, -dy)``.
    Two passes: first the rounded paper-coloured halo of every line (only
    when ``outline``), then the letters with a thin foreground stroke that
    approximates the heavier weight of professional lettering.  Drawing all
    halos before any letters keeps a line's halo from cutting into the
    descenders of the line above."""
    weight = weight_px(ts.size)
    if outline:
        halo = halo_px(ts.size) + weight
        for line in ts.lines:
            x = line.cx - line.width / 2.0 - dx
            draw.text((x, line.top - dy), line.text, font=font, fill=(*bg, 255), stroke_width=halo, stroke_fill=(*bg, 255))
    for line in ts.lines:
        x = line.cx - line.width / 2.0 - dx
        draw.text((x, line.top - dy), line.text, font=font, fill=(*fg, 255),
                  stroke_width=weight, stroke_fill=(*fg, 255) if weight else None)


def _paste_rotated_about(
    canvas: Image.Image, layer: Image.Image, angle_deg: float, lcx: float, lcy: float, cx: float, cy: float
) -> None:
    """Rotate ``layer`` (whose centre sits at frame point ``(lcx, lcy)``) by
    ``angle_deg`` about the frame point ``(cx, cy)`` and composite it."""
    a = np.deg2rad(angle_deg)
    # Rotating CCW on screen (y down) about (cx, cy) maps the layer centre to:
    vx, vy = lcx - cx, lcy - cy
    rx = cx + vx * np.cos(a) + vy * np.sin(a)
    ry = cy - vx * np.sin(a) + vy * np.cos(a)
    _paste_rotated(canvas, layer, angle_deg, float(rx), float(ry))


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
    erased with their clean patch and lettered with :func:`typeset_block`
    (bubble text flowed through the bubble outline, free text placed by
    ``render.place``) in the comic lettering font (``block_font_path``,
    default the bundled Anime Ace), upper-cased when ``uppercase`` is set.

    Plain segments: the quad polygon is filled with ``style.bg`` (when
    ``hide_original``), the translation is fitted into the quad's unrotated
    ``width x text_height`` box with :func:`fit_text`, rendered in
    ``style.fg`` and rotated by ``style.angle_deg`` about the quad centre.
    Vertical segments render one character per row.  ``font_path`` None
    picks a system font with CJK coverage when available.
    """
    path = font_path if font_path is not None else default_font_path()
    measure = pil_measurer(path)

    canvas = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1])).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    # Blocks first erase, then all draw: a later block's clean patch must not
    # paint over an earlier block's lettering.  Blocks whose translation is
    # just the source text are left exactly as they were.
    blocks = [seg for seg in segments if seg.style.is_block and not seg.untranslated]
    plain = [seg for seg in segments if not seg.style.is_block]
    if hide_original:
        for seg in blocks:
            st = seg.style
            if st.clean_patch is not None and st.clean_rect is not None:
                r = st.clean_rect
                patch = Image.fromarray(np.ascontiguousarray(st.clean_patch[:, :, ::-1])).convert("RGBA")
                canvas.paste(patch, (r.x, r.y))
    for seg in blocks:
        text = seg.translation.strip()
        if block_font_path is not None and not has_cjk(text):
            bpath: Optional[str] = block_font_path
        else:
            bpath = block_font_path_for(seg.style, text, path)
        _draw_block(canvas, seg, bpath, pil_measurer(bpath), hide_original=False, uppercase=uppercase)

    for seg in plain:
        quad = np.asarray(seg.quad, dtype=np.float64).reshape(4, 2)
        style = seg.style
        if hide_original:
            draw.polygon([tuple(p) for p in quad.tolist()], fill=(*style.bg, 255))
        text = fold_to_face(seg.translation.strip(), path)
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
