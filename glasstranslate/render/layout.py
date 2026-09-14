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
   text itself painted over) to find the speech bubble.  Joined balloons, and
   balloons whose outline opens into the page, share one patch of paper;
   :mod:`.bubbles` cuts it into one interior per block first.  When the
   block's own interior is bubble-sized the translation is typeset inside its
   outline; otherwise the text is treated as free text over artwork.
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
from . import bubbles
from .style import extract_colors, is_vertical, quad_angle_deg, quad_text_height, quad_text_width

# --- tunables --------------------------------------------------------------
_LIGHT_THRESHOLD = 200  # grey level above which a pixel is "paper" (light backgrounds)
_DARK_THRESHOLD = 60  # grey level below which a pixel is "paper" (dark backgrounds)
_FURIGANA_MAX_RATIO = 0.7  # glyph size relative to the neighbouring main text
# The same limit on the character em, which is the sharp test.  An OCR box
# runs about 1.4 ems across, so a dialogue column standing beside a slightly
# bigger one clears the glyph ratio easily; ruby, printed at half the pitch of
# the kanji it reads, does not come near it.  Over the five reference pages
# real ruby reaches 0.71 of its parent's em and the dialogue columns the glyph
# rule alone ate (``すまんな``, ``ところで``, ``これだ``...) start at 0.85.
_FURIGANA_MAX_EM_RATIO = 0.78
_FURIGANA_REACH = 1.4  # how far (in main glyph sizes) furigana may sit from its text
_GROUP_REACH = 0.7  # gap (in glyph sizes) that still joins two lines into a block
_GROUP_SIZE_RATIO = 1.7  # max glyph-size ratio between the lines of one block
_BUBBLE_MAX_AREA_RATIO = 12.0  # paper area / block area beyond which it is not a bubble
# A bubble hugs its text: its width/height may exceed the text's by at most
# this factor plus a two-glyph margin.  Larger paper regions are the panel
# background, and the text on them is free text laid out where it was.
_BUBBLE_MAX_DIM_RATIO = 2.5
# ...but only along the reading direction, where the OCR measures the text.
# Across it the OCR reports a lower bound, because whole columns go missing
# (1ja yields 25 raw lines for the whole page, and four of the five columns
# in one balloon are simply absent), so across the reading direction a
# balloon may also be this many columns wide whatever the OCR found.  The
# columns of a balloon are set at the character pitch, so one column is one
# em across - not one ``glyph_size``, which is the OCR box and runs 1.4 em.
_BUBBLE_MAX_COLUMNS = 5.0
_BUBBLE_MIN_SOLIDITY = 0.75  # paper area / convex hull area: bubbles are convex-ish
# A balloon is walled in by its own outline and leaks into the paper behind it
# through a tail or a gap at most; at least this much of its edge has to be
# that wall (ink, artwork, or the balloon next door), the rest being paper the
# cut in :mod:`.bubbles` left to nobody.  Without it a room cut out of the
# paper a panel is drawn on - the white between a scaffold's beams, a slice of
# a flat panel - reads as a balloon on every other gate, and the eraser then
# paints flat paper over the artwork inside it.  Measured on the reference
# pages: real balloons 0.69 - 1.00, rooms cut out of panel paper 0.03 - 0.48.
_BUBBLE_MIN_WALLED = 0.5
# Blocks are searched together when one's text reaches into another's window,
# because only then can they share a room.  That relation is transitive, and
# on a screen whose paper is all one component a chain of blocks can drag the
# whole frame into one cluster; a cluster spreading over more than this many
# times its biggest member's own window is not a room, and its blocks are
# searched one at a time instead (see :func:`_spread`).
_CLUSTER_MAX_SPREAD = 4.0
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


def has_kanji(text: str) -> bool:
    """True when ``text`` carries a CJK ideograph.  Furigana is the reading of
    kanji, so a line of pure kana never carries ruby of its own - which is what
    keeps a small kana column standing beside a bigger kana one (2ja
    ``ねえ``/``って``, 3jp ``こねーよ``/``いちいち``) out of the ruby rule."""
    return any(0x3400 <= ord(c) <= 0x4DBF or 0x4E00 <= ord(c) <= 0x9FFF or 0xF900 <= ord(c) <= 0xFAFF for c in text)


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
    """Point every ruby line at the line it reads.  Ruby is kana-only, it
    annotates kanji (:func:`has_kanji`), it is printed much smaller than that
    kanji - on the glyph box *and* on the character em, the one the OCR does
    not inflate - and it sits on the same paper within ``_FURIGANA_REACH``.
    A line with a parent is dropped from the translation but still erased, so
    the size rules are kept tight: what they wave through is lost speech."""
    for i, s in enumerate(lines):
        if not s.kana:
            continue
        best: Optional[Tuple[float, int]] = None
        for j, m in enumerate(lines):
            if j == i or not has_kanji(m.seg.text):
                continue
            if s.glyph >= _FURIGANA_MAX_RATIO * m.glyph or s.comp != m.comp:
                continue
            if s.em >= _FURIGANA_MAX_EM_RATIO * m.em:
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


def _outsized(one: float, other: float) -> bool:
    """Do these two measurements differ by more than ``_GROUP_SIZE_RATIO``?"""
    big, small = max(one, other), min(one, other)
    return big > _GROUP_SIZE_RATIO * max(small, 1e-6)


def _compatible(a: _Line, b: _Line) -> bool:
    if a.comp != b.comp:
        return False
    if a.vertical is not None and b.vertical is not None and a.vertical != b.vertical:
        return False
    # Two lines are different sizes only when *both* ways of measuring them
    # say so.  The OCR box is inflated sideways over a neighbouring column
    # and carries padding that varies with the glyphs in it; the character em
    # is the pitch along the line (:func:`glyph_em`) and is coarse on a line
    # of one or two characters, where a single cell sets it.  Each measure
    # splits an utterance the other keeps: 3jp's `迷っても` / `なくても` are one
    # sentence whose boxes run 46.0 and 25.0 (ratio 1.84, over the limit) but
    # whose ems run 22.0 and 18.8 (1.17); 2ja's `って` sits in the middle of
    # `ねえ...有名なの？` with a box that matches its neighbours' and an em of
    # 36.5 against their 16.6.  Splitting only when both agree keeps both.
    if _outsized(a.em, b.em) and _outsized(a.glyph, b.glyph):
        return False
    reach = _GROUP_REACH * max(a.glyph, b.glyph)
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
@dataclass
class _Spec:
    """What the paper pass needs to know about a group of lines before the
    block itself is made: who is in it, how big and which way its characters
    run, where it sits and whether its paper is dark."""

    members: List[_Line]
    furigana: List[_Line]
    vertical: bool
    glyph: float
    em: float
    source_rect: Rect
    fg: RGB
    bg: RGB
    angle: float
    dark: bool


def _spec(lines: List[_Line], member_idx: List[int], furigana_idx: List[int]) -> _Spec:
    members = [lines[i] for i in member_idx]
    furigana = [lines[i] for i in furigana_idx]
    votes = [l.vertical for l in members if l.vertical is not None]
    vertical = bool(votes) and sum(votes) * 2 >= len(votes)
    # The block's colours are the medians of its lines' (measured once in
    # build_blocks); re-clustering the union quad gave the same answer for
    # single-colour blocks at 0.6 ms a block.
    fg = tuple(int(v) for v in np.median([l.fg for l in members], axis=0))
    bg = tuple(int(v) for v in np.median([l.bg for l in members], axis=0))
    if _luma(fg) < _FG_SNAP_LUMA:
        fg = (0, 0, 0)
    angle = 0.0
    if not vertical:
        angles = [quad_angle_deg(l.seg.quad) for l in members if l.vertical is not None]
        median = float(np.median(angles)) if angles else 0.0
        angle = median if abs(median) > 4.0 else 0.0
    return _Spec(
        members=members,
        furigana=furigana,
        vertical=vertical,
        glyph=float(np.median([l.glyph for l in members])),
        em=float(np.median([l.em for l in members])),
        source_rect=_union([l.bbox for l in members + furigana]),
        fg=fg,
        bg=bg,
        angle=angle,
        dark=_luma(bg) < 128.0,
    )


def _component(maps: _PaperMaps, spec: _Spec) -> Tuple[int, Rect, int]:
    """The paper component under the spec's first line, as ``(label, bounding
    box, area)``.  Label 0 means the component is just the text boxes painted
    into the paper map (a backdrop whose grey is neither light nor dark
    paper): there is no paper around the text, so the block is neither in a
    bubble nor bounded."""
    _, stats = maps.get(spec.dark)
    comp = maps.label_at(spec.dark, spec.members[0].bbox)
    cx, cy, cw, ch, carea = (int(v) for v in stats[comp][:5])
    src = spec.source_rect
    if comp > 0 and cw <= src.w + 2 * _TEXT_PAD + 2 and ch <= src.h + 2 * _TEXT_PAD + 2:
        comp = 0
    return comp, Rect(cx, cy, cw, ch), carea


def _whole(maps: _PaperMaps, spec: _Spec, comp: int, rect: Rect, area: int) -> bubbles.Interior:
    """A paper component taken as one interior, uncut."""
    labels, _ = maps.get(spec.dark)
    return bubbles.Interior(rect, labels[rect.y : rect.y2, rect.x : rect.x2] == comp, area)


def _search_window(spec: _Spec, comp_rect: Rect, width: int, height: int) -> Rect:
    """The paper worth searching for this block's balloon: the source
    footprint grown by as far as a balloon may reach past it on each side
    (:func:`_dim_limits`) with a glyph of slack, clipped to the component.
    Paper beyond it cannot be part of anything the gate would accept, and
    cutting the search down to it keeps a page-sized background component
    from costing a page-sized distance transform per block."""
    src, glyph = spec.source_rect, spec.glyph
    max_w, max_h = _dim_limits(spec)
    grown = _grow(src, max_w - src.w + glyph, max_h - src.h + glyph, width, height)
    x, y = max(grown.x, comp_rect.x), max(grown.y, comp_rect.y)
    return Rect(x, y, min(grown.x2, comp_rect.x2) - x, min(grown.y2, comp_rect.y2) - y)


def _clusters(idx: Sequence[int], specs: Sequence[_Spec], windows: Dict[int, Rect]) -> List[List[int]]:
    """Blocks that have to be looked at together, because only they can turn
    out to share a room: a shared room is one block's whole interior, so it
    lies in that block's search window and holds the other's text, which
    therefore reaches into that window too."""
    parent = {i: i for i in idx}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pos, a in enumerate(idx):
        for b in idx[pos + 1 :]:
            ra, rb = find(a), find(b)
            near = windows[a].intersects(specs[b].source_rect) or windows[b].intersects(specs[a].source_rect)
            if ra != rb and near:
                parent[rb] = ra
    out: Dict[int, List[int]] = {}
    for i in idx:
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _spread(groups: Sequence[Sequence[int]], windows: Dict[int, Rect]) -> List[List[int]]:
    """Break up a cluster whose blocks are too far apart to be in one room.

    :func:`_clusters` merges transitively, so on a screen whose paper is one
    connected component - a document viewer, a white page behind a manga
    page - a chain of blocks reaching one after another can put the whole
    frame in a single cluster.  The union of their windows is then the whole
    frame, and :mod:`.bubbles` pays a distance transform per block over it:
    24 blocks on a 4K frame measured 1.7 s, against a 100 ms pipeline tick.

    A cluster that spreads over more than ``_CLUSTER_MAX_SPREAD`` times its
    biggest member's own search window is not one room by any reading, so
    every block in it is searched alone.  Blocks that then turn out to claim
    the same paper are dropped to free text by :func:`_uncross`, which is
    what keeps the interiors disjoint when the clustering is skipped."""
    out: List[List[int]] = []
    for group in groups:
        union = _union([windows[i] for i in group])
        largest = max(windows[i].w * windows[i].h for i in group)
        if len(group) > 1 and union.w * union.h > _CLUSTER_MAX_SPREAD * largest:
            out.extend([i] for i in group)
        else:
            out.append(list(group))
    return out


def _uncross(out: List[Optional[bubbles.Interior]]) -> None:
    """Interiors are disjoint by construction inside one cluster, and two
    blocks in different clusters are too far apart to meet.  When
    :func:`_spread` has taken a cluster apart that guarantee is gone, so any
    pair that does overlap gives up: the smaller one is no balloon."""
    live = [i for i, interior in enumerate(out) if interior is not None]
    for pos, i in enumerate(live):
        for j in live[pos + 1 :]:
            a, b = out[i], out[j]
            if a is None or b is None or not a.rect.intersects(b.rect):
                continue
            if _touching(a, b):
                loser = i if a.area <= b.area else j
                out[loser] = None


def _touching(a: bubbles.Interior, b: bubbles.Interior) -> bool:
    """Do two interiors share a pixel?"""
    x0, y0 = max(a.rect.x, b.rect.x), max(a.rect.y, b.rect.y)
    x1, y1 = min(a.rect.x2, b.rect.x2), min(a.rect.y2, b.rect.y2)
    left = a.region[y0 - a.rect.y : y1 - a.rect.y, x0 - a.rect.x : x1 - a.rect.x]
    right = b.region[y0 - b.rect.y : y1 - b.rect.y, x0 - b.rect.x : x1 - b.rect.x]
    return bool((left & right).any())


def _escapes(interior: bubbles.Interior, window: Rect, comp_rect: Rect) -> bool:
    """The interior runs into an edge of the search window that is not an
    edge of the component itself: the paper goes on past what the gate would
    allow, so this is no balloon."""
    r = interior.rect
    return (
        (r.x <= window.x and window.x > comp_rect.x)
        or (r.y <= window.y and window.y > comp_rect.y)
        or (r.x2 >= window.x2 and window.x2 < comp_rect.x2)
        or (r.y2 >= window.y2 and window.y2 < comp_rect.y2)
    )


def _interiors(maps: _PaperMaps, specs: Sequence[_Spec], width: int, height: int) -> List[Optional[bubbles.Interior]]:
    """One balloon interior per spec.  A component holding a single block that
    already reads as a balloon is taken as it is - that is most of the
    balloons on a page, and it costs nothing.  Every other one (joined
    balloons, a balloon that leaks into the page, the page itself) goes to
    :mod:`.bubbles` a search window at a time, to be cut into one room per
    block."""
    out: List[Optional[bubbles.Interior]] = [None] * len(specs)
    paper: Dict[Tuple[bool, int], Tuple[Rect, int, List[int]]] = {}
    for i, spec in enumerate(specs):
        comp, rect, area = _component(maps, spec)
        if comp > 0:
            paper.setdefault((spec.dark, comp), (rect, area, []))[2].append(i)
    for (_, comp), (rect, area, idx) in paper.items():
        if len(idx) == 1:
            whole = _whole(maps, specs[idx[0]], comp, rect, area)
            if _is_bubble(whole, specs[idx[0]]):
                out[idx[0]] = whole
                continue
        labels, _ = maps.get(specs[idx[0]].dark)
        windows = {i: _search_window(specs[i], rect, width, height) for i in idx}
        for group in _spread(_clusters(idx, specs, windows), windows):
            window = _union([windows[i] for i in group])
            region = (labels[window.y : window.y2, window.x : window.x2] == comp).view(np.uint8)
            cut = bubbles.interiors(
                region,
                (window.x, window.y),
                [specs[i].source_rect for i in group],
                [specs[i].em for i in group],
            )
            for i, interior in zip(group, cut):
                out[i] = None if interior is None or _escapes(interior, window, rect) else interior
    _uncross(out)
    return out


def _dim_limits(spec: _Spec) -> Tuple[float, float]:
    """How wide and how tall a patch of paper may be and still be the balloon
    the spec's text sits in, in pixels.

    A bubble hugs its text, and along the reading direction the OCR measures
    that text: a column's length is the column's length.  Across it the OCR
    reports a lower bound only, so the balloon may also be
    ``_BUBBLE_MAX_COLUMNS`` columns wide there whatever the OCR found.  Both
    limits carry a two-glyph margin for the outline and the air inside it."""
    src, glyph = spec.source_rect, spec.glyph
    along, across = (src.h, src.w) if spec.vertical else (src.w, src.h)
    along_limit = _BUBBLE_MAX_DIM_RATIO * along + 2 * glyph
    across_limit = max(_BUBBLE_MAX_DIM_RATIO * across, _BUBBLE_MAX_COLUMNS * spec.em) + 2 * glyph
    return (across_limit, along_limit) if spec.vertical else (along_limit, across_limit)


def _holds(rect: Rect, src: Rect) -> bool:
    """Does ``rect`` cover the whole source footprint (two pixels of slack)?"""
    return rect.x <= src.x + 2 and rect.y <= src.y + 2 and rect.x2 >= src.x2 - 2 and rect.y2 >= src.y2 - 2


def _is_bubble(interior: Optional[bubbles.Interior], spec: _Spec) -> bool:
    """Does this patch of paper read as the balloon the block's text sits in?
    It has to hold the whole source footprint, stay within
    ``_BUBBLE_MAX_AREA_RATIO`` of its area, hug it (:func:`_dim_limits`), be
    walled in (``_BUBBLE_MIN_WALLED``) and be convex-ish
    (``_BUBBLE_MIN_SOLIDITY``)."""
    if interior is None or interior.area <= 0:
        return False
    src = spec.source_rect
    max_w, max_h = _dim_limits(spec)
    return (
        _holds(interior.rect, src)
        and interior.area <= _BUBBLE_MAX_AREA_RATIO * max(1, src.w * src.h)
        and interior.rect.w <= max_w
        and interior.rect.h <= max_h
        and interior.walled >= _BUBBLE_MIN_WALLED
        and interior.solidity() >= _BUBBLE_MIN_SOLIDITY
    )


def _bubble_region(interior: bubbles.Interior, spec: _Spec) -> Optional[np.ndarray]:
    """The lettering area inside a balloon: the interior with its holes
    closed, inset by ``_BUBBLE_MARGIN_EM`` ems and reduced to the one piece
    the block's text is on - an inset pinches a waisted balloon in two, and
    the lettering is anchored on the inscribed circle of whatever it is
    handed.  None when the inset leaves nothing (a caption box barely roomier
    than its text).

    The holes the source glyphs punch in the region are *kept*, although the
    eraser paints over them (``erase._analyse_bubble`` fills them itself) and
    although they drop the region's inscribed circle into a gap between two
    characters - 1ja block 13 measures 7.0 times its inscribed circle with
    them and 1.2 without.  Closing them was tried and measured worse on every
    score that compares us with the letterer (overflowing blocks 2 -> 9,
    containment 0.022 -> 0.052, centre offset 0.71 -> 0.78 em), because the
    ground truth derives its own interior from the same paper map and has the
    same holes: filling ours alone only makes the two disagree."""
    margin = max(3, int(round(_BUBBLE_MARGIN_EM * spec.em)))
    inset = bubbles.inset(interior.region, margin)
    if not inset.any():
        return None
    # The gaps the paper map leaves between a block's own OCR boxes are left
    # alone too.  A hairline close (2 - 4 px) adds no pixel anywhere and moves
    # no row's widest run; reaching 3jp's merged balloon takes ~20 px, which
    # is the hole filling above.  Painting the block's own footprint in does
    # reach it (widest run 50 -> 97 px, type 9.0 -> 10.4 px) but fattens every
    # mask towards its text: 0.36 -> 0.44 em of mean centre offset, for one
    # block.
    return bubbles.one_piece(inset, spec.source_rect, (interior.rect.x, interior.rect.y))


def _open_layout(spec: _Spec, maps: _PaperMaps, width: int, height: int) -> Tuple[Rect, Optional[np.ndarray], Rect]:
    """Free text: ``(layout_box, layout_mask, layout_seed)``.  The seed is
    where the text was, the box how far it may reach without placement data
    (see the module docstring, step 6), and the mask - when the text is on
    paper bounded by an outline, a face or a panel edge - which part of the
    box is that paper."""
    src, glyph = spec.source_rect, spec.glyph
    seed = _grow(src, _OPEN_SEED_X * glyph, _OPEN_SEED_Y * glyph, width, height)
    limit_x = max(_OPEN_LIMIT_X * src.w, _OPEN_LIMIT_MIN_X * glyph)
    box = _grow(src, limit_x, _OPEN_LIMIT_Y * glyph, width, height)
    comp, rect, _ = _component(maps, spec)
    if comp <= 0 or not _holds(rect, src):
        return box, None, seed
    labels, _ = maps.get(spec.dark)
    mask = labels[box.y : box.y2, box.x : box.x2] == comp
    # The text itself was painted into the paper map; make sure the seed is
    # entirely usable even if the map missed a glyph.
    sy, sx = seed.y - box.y, seed.x - box.x
    mask[sy : sy + seed.h, sx : sx + seed.w] = True
    return box, mask, seed


def _make_block(
    img: np.ndarray, spec: _Spec, interior: Optional[bubbles.Interior], maps: _PaperMaps
) -> TextBlock:
    height, width = img.shape[:2]
    layout_box: Rect
    layout_mask: Optional[np.ndarray] = None
    layout_seed: Optional[Rect] = None
    bubble: Optional[Rect] = None
    inset = _bubble_region(interior, spec) if _is_bubble(interior, spec) else None
    if inset is not None and interior is not None:
        layout_box = bubble = interior.rect
        layout_mask = inset
    else:
        layout_box, layout_mask, layout_seed = _open_layout(spec, maps, width, height)

    members = spec.members
    style = SegmentStyle(
        fg=spec.fg,
        bg=spec.bg,
        angle_deg=spec.angle,
        text_height_px=spec.glyph,
        vertical=spec.vertical,
        layout_box=layout_box,
        layout_mask=layout_mask,
        upright=True,
        max_font_px=_MAX_FONT_RATIO * spec.em,
        # Members *and* furigana: the footprint the placement reads off the
        # style alone (``place._source_rect``) has to be the ink that was
        # erased, which is what ``spec.source_rect`` already is.
        source_quads=np.stack([l.seg.quad for l in members + spec.furigana]).astype(np.float32),
        layout_seed=layout_seed,
        in_bubble=bubble is not None,
    )
    segment = Segment(
        text=_ordered_text(members, spec.vertical),
        quad=_rect_quad(spec.source_rect),
        confidence=min(l.seg.confidence for l in members),
        lang_hint=members[0].seg.lang_hint,
    )
    return TextBlock(segment, style, [l.seg for l in members], [l.seg for l in spec.furigana], bubble, spec.em)


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
    specs = [_spec(lines, members, [f for m in members for f in furigana_by_main.get(m, [])]) for members in groups]
    interiors = _interiors(maps, specs, width, height)
    blocks = [_make_block(img_bgr, spec, interior, maps) for spec, interior in zip(specs, interiors)]
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
