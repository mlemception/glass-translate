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
4. **Erasing** - inside a bubble the text is painted out with the paper
   colour; over artwork the glyph strokes are inpainted so the drawing under
   the letters is kept, and the translation is rendered with a halo.

The output is a :class:`~glasstranslate.core.types.SegmentStyle` per block
with its typesetting fields filled (``layout_box``, ``layout_mask``,
``clean_patch``, ``outline``, ``max_font_px``...).  Renderers that do not
understand blocks still work: the block's segment quad is the union of its
lines, and the plain style fields are valid.
"""
from __future__ import annotations

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
_BUBBLE_MAX_AREA_RATIO = 20.0  # paper area / block area beyond which it is not a bubble
_BUBBLE_MAX_DIM_RATIO = 6.0  # paper w or h / block w or h beyond which it is not a bubble
_BUBBLE_MARGIN = 0.35  # inset of the layout region from the bubble outline (glyph sizes)
_OPEN_EXPAND_X = 0.6  # horizontal growth of the layout box for text without a bubble (glyphs)
_OPEN_EXPAND_Y = 0.3
_FLAT_STD = 14.0  # grey std-dev below which the area around free text counts as flat
_MAX_FONT_RATIO = 0.72  # translation font size ceiling relative to the source glyph size
_INPAINT_RADIUS = 3
_TEXT_PAD = 2  # pixels added around a line's bbox when erasing / masking


@dataclass
class TextBlock:
    """One typeset unit: the merged segment, its style and its source lines."""

    segment: Segment
    style: SegmentStyle
    members: List[Segment]
    furigana: List[Segment] = field(default_factory=list)
    bubble: Optional[Rect] = None  # bounding box of the detected bubble, if any


@dataclass
class _Line:
    seg: Segment
    bbox: Rect
    glyph: float  # em size of the glyphs, pixels
    vertical: Optional[bool]  # None: single glyph, orientation unknown
    kana: bool
    fg: RGB = (0, 0, 0)
    bg: RGB = (255, 255, 255)
    comp: Tuple[bool, int] = (False, 0)  # (dark paper?, component label)
    furigana_of: Optional[int] = None

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
    """Em size of a line's glyphs: the column width for vertical text, the
    line height for horizontal text, the smaller side for a lone glyph."""
    w = quad_text_width(seg.quad)
    h = quad_text_height(seg.quad)
    if len(seg.text.strip()) <= 1:
        return max(1.0, min(w, h))
    return max(1.0, w if is_vertical(seg.quad, seg.text) else h)


def _line(seg: Segment, width: int, height: int) -> _Line:
    text = seg.text.strip()
    vertical: Optional[bool]
    if len(text) <= 1:
        vertical = None
    else:
        vertical = is_vertical(seg.quad, seg.text)
    return _Line(seg, seg.bbox.clamp(width, height), glyph_size(seg), vertical, is_kana_only(text))


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


# ------------------------------------------------------------ erasing
def _stroke_mask(crop_bgr: np.ndarray, boxes: Sequence[Rect], origin: Rect, fg: RGB, bg: RGB) -> np.ndarray:
    """Pixels inside ``boxes`` that are closer to ``fg`` than to ``bg``."""
    h, w = crop_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    fg_bgr = np.array(fg[::-1], dtype=np.float32)
    bg_bgr = np.array(bg[::-1], dtype=np.float32)
    for box in boxes:
        b = _grow(box, _TEXT_PAD, _TEXT_PAD, 10**9, 10**9)
        x1, y1 = max(0, b.x - origin.x), max(0, b.y - origin.y)
        x2, y2 = min(w, b.x2 - origin.x), min(h, b.y2 - origin.y)
        if x2 <= x1 or y2 <= y1:
            continue
        region = crop_bgr[y1:y2, x1:x2].astype(np.float32)
        d_fg = np.linalg.norm(region - fg_bgr, axis=2)
        d_bg = np.linalg.norm(region - bg_bgr, axis=2)
        mask[y1:y2, x1:x2] |= (d_fg < d_bg).astype(np.uint8)
    if mask.any():
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _erase_in_bubble(img: np.ndarray, rect: Rect, labels: np.ndarray, comp: int, bg: RGB) -> np.ndarray:
    patch = img[rect.y : rect.y2, rect.x : rect.x2].copy()
    inside = labels[rect.y : rect.y2, rect.x : rect.x2] == comp
    patch[inside] = np.array(bg[::-1], dtype=np.uint8)
    return patch


def _erase_open(img: np.ndarray, rect: Rect, boxes: Sequence[Rect], fg: RGB, bg: RGB) -> Tuple[np.ndarray, bool]:
    """Erase glyph strokes inside ``rect``.  Returns ``(patch, flat)`` where
    ``flat`` tells whether the surroundings were a plain colour (then the
    strokes were simply painted over) or artwork (then they were inpainted
    and the translation should carry a halo)."""
    crop = img[rect.y : rect.y2, rect.x : rect.x2]
    strokes = _stroke_mask(crop, boxes, rect, fg, bg)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    rest = gray[strokes == 0]
    flat = rest.size == 0 or float(rest.std()) < _FLAT_STD
    if flat:
        patch = crop.copy()
        patch[strokes > 0] = np.array(bg[::-1], dtype=np.uint8)
        return patch, True
    return cv2.inpaint(crop, strokes, _INPAINT_RADIUS, cv2.INPAINT_TELEA), False


# ------------------------------------------------------------ blocks
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
    fg, bg = extract_colors(img, quad)
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
    contains = comp_rect.x <= source_rect.x + 2 and comp_rect.y <= source_rect.y + 2 and (
        comp_rect.x2 >= source_rect.x2 - 2 and comp_rect.y2 >= source_rect.y2 - 2
    )
    is_bubble = (
        comp > 0
        and contains
        and carea <= _BUBBLE_MAX_AREA_RATIO * max(1, source_rect.w * source_rect.h)
        and cw <= _BUBBLE_MAX_DIM_RATIO * source_rect.w + 2 * glyph
        and ch <= _BUBBLE_MAX_DIM_RATIO * source_rect.h + 2 * glyph
    )

    layout_box: Rect
    layout_mask: Optional[np.ndarray] = None
    clean_rect = _grow(source_rect, _TEXT_PAD + 1, _TEXT_PAD + 1, width, height)
    outline = False
    bubble: Optional[Rect] = None
    if is_bubble:
        margin = max(3, int(round(_BUBBLE_MARGIN * glyph)))
        region = (labels[cy : cy + ch, cx : cx + cw] == comp).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        eroded = cv2.erode(region, kernel).astype(bool)
        if eroded.any():
            layout_box = comp_rect
            layout_mask = eroded
            bubble = comp_rect
            clean_patch = _erase_in_bubble(img, clean_rect, labels, comp, bg)
        else:
            is_bubble = False
    if not is_bubble:
        layout_box = _grow(source_rect, _OPEN_EXPAND_X * glyph, _OPEN_EXPAND_Y * glyph, width, height)
        boxes = [l.bbox for l in members + furigana]
        clean_patch, flat = _erase_open(img, clean_rect, boxes, fg, bg)
        outline = not flat

    style = SegmentStyle(
        fg=fg,
        bg=bg,
        angle_deg=angle,
        text_height_px=glyph,
        vertical=vertical,
        layout_box=layout_box,
        layout_mask=layout_mask,
        upright=True,
        clean_patch=clean_patch,
        clean_rect=clean_rect,
        outline=outline,
        max_font_px=_MAX_FONT_RATIO * glyph,
        source_quads=np.stack([l.seg.quad for l in members]).astype(np.float32),
    )
    confidence = min(l.seg.confidence for l in members)
    segment = Segment(text=text, quad=quad, confidence=confidence, lang_hint=members[0].seg.lang_hint)
    return TextBlock(segment, style, [l.seg for l in members], [l.seg for l in furigana], bubble)


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


def build_blocks(img_bgr: np.ndarray, segments: Sequence[Segment]) -> List[TextBlock]:
    """Group ``segments`` (raw OCR lines in the coordinates of ``img_bgr``)
    into typeset-ready blocks; see the module docstring."""
    if not segments:
        return []
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
    return blocks
