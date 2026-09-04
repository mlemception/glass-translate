"""Manga-aware typesetting layer.

Turns raw OCR lines into *text blocks* ready for translation and rendering:

1. **Furigana removal** - small kana-only lines hugging a larger line are the
   pronunciation guides printed beside kanji; they are dropped from the text
   but their pixels are still erased.
2. **Grouping** - adjacent lines of the same orientation and glyph size that
   sit on the same patch of paper (no bubble outline between them) become one
   block.  Vertical columns are read right-to-left, horizontal lines
   top-to-bottom, so the block text is the whole utterance in order, which
   translates far better than one column at a time.
3. **Bubble detection** - the paper around the block is flood-filled (with the
   text itself painted over) to find the speech bubble.  When the enclosing
   region is bubble-sized the translation is typeset inside its outline;
   otherwise the text is treated as free text over artwork.
4. **Sizing** - the translation's font size ceiling is the source
   characters' em (their pitch along the column), shared by every dialogue
   block on the page so all speech is lettered at one size.
5. **Erasing** (:mod:`.erase`) - glyph strokes are separated from the
   artwork with connected components and removed; paper stays paper and
   art lines run on through the erased area.  Text left on artwork is
   flagged for a halo (``outline``).
6. **Placement data** (:mod:`.place`) - free text (no bubble) gets an ink
   map and a blocked map of its neighbourhood so the renderer can letter it
   where a human letterer would: a compact block on quiet paper near the
   source, never over another block or across a panel border.

The output is a :class:`~glasstranslate.core.types.SegmentStyle` per block
with its typesetting fields filled (``layout_box``, ``layout_mask``,
``clean_patch``, ``outline``, ``max_font_px``, ``search_box``...).
Renderers that do not understand blocks still work: the block's segment
quad is the union of its lines, and the plain style fields are valid.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.types import RGB, Rect, Segment, SegmentStyle
from .style import extract_colors, is_vertical, quad_angle_deg, quad_text_height, quad_text_width

# --- tunables --------------------------------------------------------------
_LIGHT_THRESHOLD = 200  # grey level above which a pixel is "paper" (light backgrounds)
_DARK_THRESHOLD = 60  # grey level below which a pixel is "paper" (dark backgrounds)
_FURIGANA_MAX_RATIO = 0.7  # glyph size relative to the neighbouring main text
_FURIGANA_REACH = 1.4  # how far (in main glyph sizes) furigana may sit from its text
_GROUP_REACH = 0.7  # gap (in glyph sizes) that still joins two lines into a block
_GROUP_SIZE_RATIO = 1.7  # max glyph-size ratio between the lines of one block
_BUBBLE_MAX_AREA_RATIO = 12.0  # paper area / block area beyond which it is not a bubble
# A bubble hugs its text: its width/height may exceed the text's by at most
# this factor plus a two-glyph margin.  Larger paper regions are the panel
# background, and the text on them is free text laid out where it was.
_BUBBLE_MAX_DIM_RATIO = 2.5
_BUBBLE_MIN_SOLIDITY = 0.75  # paper area / convex hull area: bubbles are convex-ish
# Inset of the layout region from the bubble outline, in ems of the block's
# source text (the lettering's size ceiling): the lettering keeps this much
# air from the outline on every side (reference pages: ~0.25 em from ink to
# line, of which the halo takes 0.1).  Measured in ems, not OCR box widths,
# which run 1.4 x the em and would waste a third of a caption box.
_BUBBLE_MARGIN_EM = 0.18
# Free text (no bubble): ``layout_seed`` is the source footprint slightly
# widened, ``layout_box`` the same grown to these limits.  The renderer
# places the lettering with the ``search_box`` / ink / blocked maps from
# :mod:`.place`; the layout box is its fallback region and what the pipeline
# treats as the block's footprint.
_OPEN_SEED_X = 0.3  # glyphs added on each side of the source footprint
_OPEN_SEED_Y = 0.1
_OPEN_LIMIT_X = 0.75  # fraction of the source width added on each side...
_OPEN_LIMIT_MIN_X = 1.5  # ...but at least this many glyphs
_OPEN_LIMIT_Y = 0.5  # glyphs added above and below
# Translation font size ceiling relative to the *source em* (the character
# pitch along a column, ``TextBlock.em_px``): professionally typeset pages
# letter the English with a cap height of ~0.69 x the Japanese pitch
# (reference: cap 19 px on a 27.8 px pitch, about half the OCR column
# width, which runs ~1.37 x the pitch).  Anime Ace's cap height is 0.84 em,
# so its em is 0.82 x the pitch.
_MAX_FONT_RATIO = 0.82
# Blocks whose em is within this band of the page median share the
# median-based size, so all dialogue on a page is lettered at one size.
_UNIFORM_BAND = (0.7, 1.4)
_FG_SNAP_LUMA = 90.0  # text this dark is lettered in pure black
_TEXT_PAD = 2  # pixels added around a line's bbox when masking


@dataclass
class TextBlock:
    """One typeset unit: the merged segment, its style and its source lines."""

    segment: Segment
    style: SegmentStyle
    members: List[Segment]
    furigana: List[Segment] = field(default_factory=list)
    bubble: Optional[Rect] = None  # bounding box of the detected bubble, if any
    # Em of the source characters (median character pitch of the member
    # lines, pixels); the translation's size ceiling derives from it.  0 =
    # unknown (callers fall back to ``style.text_height_px``).
    em_px: float = 0.0


@dataclass
class _Line:
    seg: Segment
    bbox: Rect
    glyph: float  # column width (vertical) / line height (horizontal), pixels
    vertical: Optional[bool]  # None: single glyph, orientation unknown
    kana: bool
    fg: RGB = (0, 0, 0)
    bg: RGB = (255, 255, 255)
    comp: Tuple[bool, int] = (False, 0)  # (dark paper?, component label)
    furigana_of: Optional[int] = None
    em: float = 0.0  # character em (pitch), pixels; see :func:`glyph_em`

    @property
    def area(self) -> int:
        return self.bbox.w * self.bbox.h


# --------------------------------------------------------------- helpers
def _is_kana(ch: str) -> bool:
    o = ord(ch)
    return 0x3041 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF or ch in "ー゛゜"


def is_kana_only(text: str) -> bool:
    """True when every letter of ``text`` is hiragana/katakana (furigana)."""
    letters = [c for c in text if not c.isspace() and c not in "。、！？…・「」"]
    return bool(letters) and all(_is_kana(c) for c in letters)


def has_cjk(text: str) -> bool:
    return any(0x2E80 <= ord(c) <= 0x9FFF or 0xF900 <= ord(c) <= 0xFAFF or 0xFF00 <= ord(c) <= 0xFFEF for c in text)


def _luma(rgb: RGB) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _grow(rect: Rect, dx: float, dy: float, width: int, height: int) -> Rect:
    dx_i, dy_i = int(round(dx)), int(round(dy))
    return Rect(rect.x - dx_i, rect.y - dy_i, rect.w + 2 * dx_i, rect.h + 2 * dy_i).clamp(width, height)


def _union(rects: Sequence[Rect]) -> Rect:
    out = rects[0]
    for r in rects[1:]:
        out = out.union(r)
    return out


def _rect_quad(rect: Rect) -> np.ndarray:
    return np.array(
        [[rect.x, rect.y], [rect.x2, rect.y], [rect.x2, rect.y2], [rect.x, rect.y2]], dtype=np.float32
    )


def glyph_size(seg: Segment) -> float:
    """Size of a line across its reading direction: the column width for
    vertical text, the line height for horizontal text, the smaller side for
    a lone glyph.  OCR boxes carry padding, so this runs about 1.4 x the
    character em (see :func:`glyph_em`)."""
    w = quad_text_width(seg.quad)
    h = quad_text_height(seg.quad)
    if len(seg.text.strip()) <= 1:
        return max(1.0, min(w, h))
    return max(1.0, w if is_vertical(seg.quad, seg.text) else h)


def _cell_count(text: str) -> float:
    """Number of character cells ``text`` occupies along its line: CJK and
    other full-width characters take one cell, everything else (Latin
    letters, digits, spaces) half a cell."""
    total = 0.0
    for c in text:
        if has_cjk(c) or 0x3000 <= ord(c) <= 0x30FF:
            total += 1.0
        else:
            total += 0.5
    return max(1.0, total)


def glyph_em(seg: Segment) -> float:
    """Em of a line's characters: the character pitch (extent along the
    reading direction over the number of character cells), capped by the
    extent across it.  Immune to boxes the detector inflated sideways over
    a neighbouring column (their pitch is still right) and to the padding
    every box carries across the text."""
    w = quad_text_width(seg.quad)
    h = quad_text_height(seg.quad)
    text = seg.text.strip()
    if len(text) <= 1:
        return max(1.0, min(w, h))
    along, across = (h, w) if is_vertical(seg.quad, text) else (w, h)
    pitch = along / _cell_count(text)
    return max(1.0, min(across, pitch))


def _line(seg: Segment, width: int, height: int) -> _Line:
    text = seg.text.strip()
    vertical: Optional[bool]
    if len(text) <= 1:
        vertical = None
    else:
        vertical = is_vertical(seg.quad, seg.text)
    return _Line(seg, seg.bbox.clamp(width, height), glyph_size(seg), vertical, is_kana_only(text), em=glyph_em(seg))


class _PaperMaps:
    """Connected components of the paper (background) with every text line
    painted over, computed lazily for light and for dark paper."""

    def __init__(self, gray: np.ndarray, text_boxes: Sequence[Rect]) -> None:
        self._gray = gray
        self._boxes = text_boxes
        self._maps: Dict[bool, Tuple[np.ndarray, np.ndarray]] = {}

    def get(self, dark: bool) -> Tuple[np.ndarray, np.ndarray]:
        """``(labels, stats)`` as returned by ``connectedComponentsWithStats``."""
        cached = self._maps.get(dark)
        if cached is not None:
            return cached
        mask = (self._gray < _DARK_THRESHOLD) if dark else (self._gray > _LIGHT_THRESHOLD)
        mask = mask.astype(np.uint8)
        h, w = mask.shape
        for box in self._boxes:
            g = _grow(box, _TEXT_PAD, _TEXT_PAD, w, h)
            mask[g.y : g.y2, g.x : g.x2] = 1
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        self._maps[dark] = (labels, stats)
        return labels, stats

    def label_at(self, dark: bool, rect: Rect) -> int:
        labels, _ = self.get(dark)
        cy = min(labels.shape[0] - 1, rect.y + rect.h // 2)
        cx = min(labels.shape[1] - 1, rect.x + rect.w // 2)
        return int(labels[cy, cx])


# ------------------------------------------------------------ grouping
def _mark_furigana(lines: List[_Line]) -> None:
    for i, s in enumerate(lines):
        if not s.kana:
            continue
        best: Optional[Tuple[float, int]] = None
        for j, m in enumerate(lines):
            if j == i or m.kana and m.glyph <= s.glyph:
                continue
            if s.glyph >= _FURIGANA_MAX_RATIO * m.glyph or s.comp != m.comp:
                continue
            reach = _FURIGANA_REACH * m.glyph
            near = Rect(m.bbox.x - int(reach), m.bbox.y - int(reach), m.bbox.w + 2 * int(reach), m.bbox.h + 2 * int(reach))
            if not near.intersects(s.bbox):
                continue
            dist = abs((s.bbox.x + s.bbox.w / 2) - (m.bbox.x + m.bbox.w / 2)) + abs(
                (s.bbox.y + s.bbox.h / 2) - (m.bbox.y + m.bbox.h / 2)
            )
            if best is None or dist < best[0]:
                best = (dist, j)
        if best is not None:
            s.furigana_of = best[1]


def _compatible(a: _Line, b: _Line) -> bool:
    if a.comp != b.comp:
        return False
    if a.vertical is not None and b.vertical is not None and a.vertical != b.vertical:
        return False
    big, small = max(a.glyph, b.glyph), min(a.glyph, b.glyph)
    if big > _GROUP_SIZE_RATIO * small:
        return False
    reach = _GROUP_REACH * big
    grown = Rect(a.bbox.x - int(reach), a.bbox.y - int(reach), a.bbox.w + 2 * int(reach), a.bbox.h + 2 * int(reach))
    return grown.intersects(b.bbox)


def _group(lines: List[_Line]) -> List[List[int]]:
    """Union-find over main (non-furigana) lines; returns index groups."""
    idx = [i for i, l in enumerate(lines) if l.furigana_of is None]
    parent = {i: i for i in idx}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_pos, a in enumerate(idx):
        for b in idx[a_pos + 1 :]:
            if _compatible(lines[a], lines[b]):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    groups: Dict[int, List[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _ordered_text(members: List[_Line], vertical: bool) -> str:
    if vertical:
        ordered = sorted(members, key=lambda l: -(l.bbox.x + l.bbox.w / 2.0))
    else:
        ordered = sorted(members, key=lambda l: (l.bbox.y + l.bbox.h / 2.0, l.bbox.x))
    texts = [l.seg.text.strip() for l in ordered]
    joiner = "" if any(has_cjk(t) for t in texts) else " "
    return joiner.join(t for t in texts if t)


# ------------------------------------------------------------ blocks
def _solidity(region: np.ndarray, area: int) -> float:
    """Area of the paper component over the area of its convex hull."""
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    hull = cv2.convexHull(np.concatenate(contours))
    hull_area = float(cv2.contourArea(hull))
    return area / hull_area if hull_area > 0 else 0.0


def _make_block(
    img: np.ndarray,
    lines: List[_Line],
    member_idx: List[int],
    furigana_idx: List[int],
    maps: _PaperMaps,
) -> TextBlock:
    height, width = img.shape[:2]
    members = [lines[i] for i in member_idx]
    furigana = [lines[i] for i in furigana_idx]
    votes = [l.vertical for l in members if l.vertical is not None]
    vertical = bool(votes) and sum(votes) * 2 >= len(votes)
    glyph = float(np.median([l.glyph for l in members]))
    text = _ordered_text(members, vertical)
    source_rect = _union([l.bbox for l in members + furigana])
    quad = _rect_quad(source_rect)
    # The block's colours are the medians of its lines' (measured once in
    # build_blocks); re-clustering the union quad gave the same answer for
    # single-colour blocks at 0.6 ms a block.
    fg = tuple(int(v) for v in np.median([l.fg for l in members], axis=0))
    bg = tuple(int(v) for v in np.median([l.bg for l in members], axis=0))
    if _luma(fg) < _FG_SNAP_LUMA:
        fg = (0, 0, 0)
    dark = _luma(bg) < 128.0
    angle = 0.0
    if not vertical:
        angles = [quad_angle_deg(l.seg.quad) for l in members if l.vertical is not None]
        median = float(np.median(angles)) if angles else 0.0
        angle = median if abs(median) > 4.0 else 0.0

    labels, stats = maps.get(dark)
    comp = maps.label_at(dark, members[0].bbox)
    cx, cy, cw, ch, carea = (int(v) for v in stats[comp][:5])
    comp_rect = Rect(cx, cy, cw, ch)
    if comp > 0 and cw <= source_rect.w + 2 * _TEXT_PAD + 2 and ch <= source_rect.h + 2 * _TEXT_PAD + 2:
        # The component is just the text boxes painted into the paper map
        # (a backdrop whose grey is neither light nor dark paper): there is
        # no paper around the text, so it is neither a bubble nor bounded.
        comp = 0
    region = (labels[cy : cy + ch, cx : cx + cw] == comp).astype(np.uint8) if comp > 0 else None
    contains = comp_rect.x <= source_rect.x + 2 and comp_rect.y <= source_rect.y + 2 and (
        comp_rect.x2 >= source_rect.x2 - 2 and comp_rect.y2 >= source_rect.y2 - 2
    )
    is_bubble = (
        comp > 0
        and contains
        and carea <= _BUBBLE_MAX_AREA_RATIO * max(1, source_rect.w * source_rect.h)
        and cw <= _BUBBLE_MAX_DIM_RATIO * source_rect.w + 2 * glyph
        and ch <= _BUBBLE_MAX_DIM_RATIO * source_rect.h + 2 * glyph
        and _solidity(region, carea) >= _BUBBLE_MIN_SOLIDITY
    )

    layout_box: Rect
    layout_mask: Optional[np.ndarray] = None
    layout_seed: Optional[Rect] = None
    bubble: Optional[Rect] = None
    em = float(np.median([l.em for l in members]))
    if is_bubble:
        margin = max(3, int(round(_BUBBLE_MARGIN_EM * em)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        # An explicit zero border: the component always touches its own
        # bounding box, and OpenCV's default border would leave those edges
        # (a caption box's outer sides) un-eroded, centring the text off.
        eroded = cv2.erode(region, kernel, borderValue=0).astype(bool)
        if eroded.any():
            layout_box = comp_rect
            layout_mask = eroded
            bubble = comp_rect
        else:
            is_bubble = False
    if not is_bubble:
        # Free text: the seed is where it was, the box how far it may reach
        # without placement data (see the module docstring, step 6).
        layout_seed = _grow(source_rect, _OPEN_SEED_X * glyph, _OPEN_SEED_Y * glyph, width, height)
        limit_x = max(_OPEN_LIMIT_X * source_rect.w, _OPEN_LIMIT_MIN_X * glyph)
        layout_box = _grow(source_rect, limit_x, _OPEN_LIMIT_Y * glyph, width, height)
        if comp > 0 and contains:
            # On paper bounded by an outline (a face, a panel edge): the
            # mask tells the placement which part of the box is that paper.
            layout_mask = labels[layout_box.y : layout_box.y2, layout_box.x : layout_box.x2] == comp
            # The text itself was painted into the paper map; make sure the
            # seed is entirely usable even if the map missed a glyph.
            sy, sx = layout_seed.y - layout_box.y, layout_seed.x - layout_box.x
            layout_mask[sy : sy + layout_seed.h, sx : sx + layout_seed.w] = True

    style = SegmentStyle(
        fg=fg,
        bg=bg,
        angle_deg=angle,
        text_height_px=glyph,
        vertical=vertical,
        layout_box=layout_box,
        layout_mask=layout_mask,
        upright=True,
        max_font_px=_MAX_FONT_RATIO * em,
        source_quads=np.stack([l.seg.quad for l in members]).astype(np.float32),
        layout_seed=layout_seed,
        in_bubble=is_bubble,
    )
    confidence = min(l.seg.confidence for l in members)
    segment = Segment(text=text, quad=quad, confidence=confidence, lang_hint=members[0].seg.lang_hint)
    return TextBlock(segment, style, [l.seg for l in members], [l.seg for l in furigana], bubble, em)


def _split_shared_bubbles(blocks: List[TextBlock]) -> None:
    """Two blocks in one paper component (joined bubbles) each keep the part
    of the region nearer to their own text, so their layouts never overlap."""
    by_bubble: Dict[Tuple[int, int, int, int], List[TextBlock]] = {}
    for b in blocks:
        if b.bubble is not None:
            by_bubble.setdefault((b.bubble.x, b.bubble.y, b.bubble.w, b.bubble.h), []).append(b)
    for group in by_bubble.values():
        if len(group) < 2:
            continue
        box = group[0].bubble
        assert box is not None
        dists = []
        for b in group:
            src = b.segment.bbox
            inv = np.full((box.h, box.w), 255, dtype=np.uint8)
            x1, y1 = max(0, src.x - box.x), max(0, src.y - box.y)
            x2, y2 = min(box.w, src.x2 - box.x), min(box.h, src.y2 - box.y)
            if x2 > x1 and y2 > y1:
                inv[y1:y2, x1:x2] = 0
            dists.append(cv2.distanceTransform(inv, cv2.DIST_L2, 3))
        owner = np.argmin(np.stack(dists), axis=0)
        for i, b in enumerate(group):
            assert b.style.layout_mask is not None
            b.style.layout_mask = b.style.layout_mask & (owner == i)


def page_em(blocks: Sequence[TextBlock]) -> float:
    """Median source em of the page's CJK blocks (Latin watermarks and
    logos do not vote); 0 when there is none."""
    ems = [b.em_px for b in blocks if b.em_px > 0 and has_cjk(b.segment.text)]
    if not ems:
        ems = [b.em_px for b in blocks if b.em_px > 0]
    return float(np.median(ems)) if ems else 0.0


def _uniform_font_sizes(blocks: List[TextBlock]) -> None:
    """Letter every dialogue block on the page at one size: blocks whose em
    is within ``_UNIFORM_BAND`` of the page median get the median's ceiling;
    outliers (titles, sound effects) keep their own."""
    if len(blocks) < 2:
        return
    median = page_em(blocks)
    if median <= 0:
        return
    lo, hi = _UNIFORM_BAND
    for b in blocks:
        if lo * median <= b.em_px <= hi * median:
            b.style.max_font_px = _MAX_FONT_RATIO * median


def build_blocks(
    img_bgr: np.ndarray,
    segments: Sequence[Segment],
    all_segments: Optional[Sequence[Segment]] = None,
    timings: Optional[Dict[str, float]] = None,
) -> List[TextBlock]:
    """Group ``segments`` (raw OCR lines in the coordinates of ``img_bgr``)
    into typeset-ready blocks; see the module docstring.

    ``all_segments`` may carry the OCR lines of *any* confidence (a superset
    of ``segments``): lines that belong to no block are used by the eraser as
    evidence of glyphs the confident boxes missed.  ``timings``, when given,
    receives the milliseconds spent per stage (``group``, ``erase``,
    ``place``).
    """
    if not segments:
        return []
    t0 = time.perf_counter()
    height, width = img_bgr.shape[:2]
    lines = [_line(s, width, height) for s in segments if s.text.strip()]
    lines = [l for l in lines if l.bbox.w > 0 and l.bbox.h > 0]
    if not lines:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    maps = _PaperMaps(gray, [l.bbox for l in lines])
    for l in lines:
        l.fg, l.bg = extract_colors(img_bgr, l.seg.quad)
        dark = _luma(l.bg) < 128.0
        l.comp = (dark, maps.label_at(dark, l.bbox))
    _mark_furigana(lines)
    groups = _group(lines)
    furigana_by_main: Dict[int, List[int]] = {}
    for i, l in enumerate(lines):
        if l.furigana_of is not None:
            furigana_by_main.setdefault(l.furigana_of, []).append(i)
    blocks: List[TextBlock] = []
    for members in groups:
        furi = [f for m in members for f in furigana_by_main.get(m, [])]
        blocks.append(_make_block(img_bgr, lines, members, furi, maps))
    _split_shared_bubbles(blocks)
    _uniform_font_sizes(blocks)
    t1 = time.perf_counter()
    erase.apply(img_bgr, blocks, all_segments if all_segments is not None else segments, gray=gray)
    t2 = time.perf_counter()
    place.prepare(img_bgr, blocks, gray=gray)
    t3 = time.perf_counter()
    if timings is not None:
        timings.update(group=1000 * (t1 - t0), erase=1000 * (t2 - t1), place=1000 * (t3 - t2))
    return blocks


# Imported last: both modules refer to TextBlock in annotations only, so the
# circular import is harmless.
from . import erase, place  # noqa: E402
