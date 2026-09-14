"""Letterer-style placement of translated text blocks.

``layout.build_blocks`` decides *what* a block is (its words, its bubble, its
size ceiling); this module decides *where and in what shape* the English is
lettered, the way a human letterer does:

* text inside a speech bubble / caption box is centred in the bubble and
  flowed along its outline (:func:`typeset.typeset`);
* free text (no bubble) is set as a compact block near the original text,
  on the quietest patch of the panel: every line count ``n`` at the ceiling
  size gives a candidate shape (the narrowest ``n``-line block), every
  shape is slid over a grid of positions around the source, and the
  position covering the least artwork (weighted towards the block centre)
  while staying close to the source, about six em wide, with evenly filled
  lines wins.  Hard constraints: never over another block's text or bubble,
  never across a panel border or into another panel, never off the page
  (the paper-coloured halo may overlap the clearance margins, the letters
  may not).  The size is kept at the ceiling unless a smaller size gives a
  clearly better block: one 5 % step costs more than a hyphen, so a word
  that fits nowhere intact is hyphenated first.

:func:`prepare` runs once per page with the image and computes the
image-derived data (ink map, blocked map, search box) into the blocks'
:class:`~glasstranslate.core.types.SegmentStyle`; :func:`typeset_block` then
needs only the style and a ``measure`` callable, so the PIL renderer and the
Qt overlay share it unchanged.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.types import RGB, Rect, SegmentStyle
from .fit import Measure
from .typeset import (
    PlacedLine,
    Typeset,
    balanced_breaks,
    clip_spans,
    hyphenation_points,
    ink_offsets,
    line_pitch,
    mask_spans,
    memoize_measure,
    min_max_width,
    rect_spans,
    span_widths,
    split_at,
    typeset,
    typeset_lobes,
)

# --- size ------------------------------------------------------------------
# The font size ceiling is ``style.max_font_px`` (the source em, set by
# ``layout.build_blocks`` and uniform over the page).  Styles without it
# (plain segments, tests) fall back to a fraction of the OCR box height:
# the reference lettering has an em of about 0.73 x the source glyph box.
_FALLBACK_EM_RATIO = 0.73
# Halo width around lettering on artwork (fraction of the em); the block
# footprint used for scoring is grown by it.  Mirrors ``compose.HALO_RATIO``.
HALO_RATIO = 0.11

# --- prepare: search region, ink and blocked maps ---------------------------
_REACH_X_EM = 5.0  # search box: source rect grown by this many em sideways...
_REACH_Y_EM = 3.0  # ...and vertically
_INK_LIGHT = 200  # grey level below which a pixel is ink on light paper
_INK_DARK = 60  # grey level above which a pixel is ink on dark paper
_BORDER_DARK = 90  # panel borders are this dark...
_BORDER_LEN_FRAC = 0.08  # ...and straight for this fraction of the page's short side
_BORDER_THICK = 2  # rows/cols a border run must span (rejects hairlines, tolerates tilt)
_BLOB_PX = 7  # dark areas at least this thick are artwork (hair, shadow), not border lines
_MIN_PANEL_FRAC = 0.04  # a panel component smaller than this fraction of the page is ignored
_OBSTACLE_MARGIN_EM = 0.5  # clearance kept from other blocks' text / bubbles
_ZONE_GAP_EM = 0.6  # gap left between the lettering of two neighbouring free-text blocks
_EDGE_MARGIN_EM = 0.3  # clearance kept between lettering and panel border lines / the page edge
_TEXT_PAD = 2  # pixels around a source line treated as paper (erased)

# --- typeset_block: shape and position search --------------------------------
# All soft costs are in units of "ink density under the block" (0..1).
_MAX_LINES = 14
_MAX_LINES_H = 3  # horizontal captions / titles are set in at most this many lines
_GRID_STEP_EM = 0.18  # candidate positions every ~5 px at dialogue size
_DENSITY_EM = 0.8  # ink is averaged over this window before being squared: hatching
#                    (density ~0.2 -> 0.04) is cheap to cover, solid black (1.0) is not
_LAMBDA_OUT = 0.1  # cost per em the block's edge is from covering the source centre (dialogue;
#                    horizontal captions must keep covering it)
_LAMBDA_CENTRE = 0.006  # cost per em between the block centre and the source centre
_LINES_PER_WORD = 0.6  # a letterer sets about one and a half to two words per line...
_LINE_WEIGHT = 0.006  # ...each extra line beyond that costs this much
_WIDTH_IDEAL_EM = 6.5  # dialogue blocks are about this wide (reference pages: 5.5-6.5 em)...
_WIDTH_WEIGHT = 0.004  # ...each em beyond costs this (mild: the space decides)
_BALANCE_WEIGHT = 0.6  # x normalised squared shortfall of the balanced lines (ragged blocks lose)
_ASPECT_BAND_V = (0.8, 2.4)  # preferred width/height for dialogue (vertical source)
_ASPECT_BAND_H = (2.0, 8.0)  # ...and for horizontal captions
_ASPECT_WEIGHT = 0.02  # cost per unit of aspect outside the band
_EXT_FREE_EM = 2.0  # the block may reach this far beyond the source footprint on any side for free...
_EXT_WEIGHT = 0.03  # ...and pays this per squared em beyond (a letterer stays near the original text)
_HYPHEN_COST = 0.04  # hyphenating the widest word (tried when nothing fits intact) costs this
_INNER_FRAC = 0.5  # the central part of the footprint whose ink counts twice
_SHRINK_STEP = 0.95
_SHRINK_WEIGHT = 1.5  # cost of shrinking: 1.5 x (1 - size / ceiling); one 5 % step (0.075) is worse than a hyphen
_MIN_SCALE = 0.75  # never shrink below this fraction of the ceiling once a place was found
_OVAL_END_FRAC = 0.9  # target width of the first and last line relative to the widest


def size_ceiling(style: SegmentStyle, min_size: float = 0.0) -> float:
    """The block's font size ceiling in pixels: ``style.max_font_px`` when
    the layout set it, else ``_FALLBACK_EM_RATIO`` x the OCR box height (at
    least ``min_size``).  The single place this decision is made, shared by
    :func:`prepare`, :func:`typeset_block` and ``compose``."""
    if style.max_font_px:
        return float(style.max_font_px)
    return max(min_size, style.text_height_px * _FALLBACK_EM_RATIO)


def _luma(rgb: RGB) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def _grow(rect: Rect, dx: float, dy: float) -> Rect:
    dx_i, dy_i = int(round(dx)), int(round(dy))
    return Rect(rect.x - dx_i, rect.y - dy_i, rect.w + 2 * dx_i, rect.h + 2 * dy_i)


def _paint(mask: np.ndarray, rect: Rect, origin: Rect, value) -> None:
    """Set ``mask`` (aligned with ``origin``) to ``value`` inside ``rect``."""
    x1, y1 = max(0, rect.x - origin.x), max(0, rect.y - origin.y)
    x2, y2 = min(origin.w, rect.x2 - origin.x), min(origin.h, rect.y2 - origin.y)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = value


def _rect_distance_sq(rect: Rect, origin: Rect) -> np.ndarray:
    """Squared Euclidean distance (int64) from every pixel of ``origin`` to
    ``rect``; integer arithmetic is ~5x cheaper than a float ``hypot`` over
    the search box and the zone test only needs comparisons."""
    xs = np.arange(origin.x, origin.x2, dtype=np.int64)
    ys = np.arange(origin.y, origin.y2, dtype=np.int64)
    dx = np.maximum(np.maximum(rect.x - xs, xs - (rect.x2 - 1)), 0)
    dy = np.maximum(np.maximum(rect.y - ys, ys - (rect.y2 - 1)), 0)
    return dx[None, :] ** 2 + dy[:, None] ** 2


def _rect_distance(rect: Rect, origin: Rect) -> np.ndarray:
    """Euclidean distance from every pixel of ``origin`` to ``rect``."""
    return np.sqrt(_rect_distance_sq(rect, origin).astype(np.float64))


def panel_borders(gray: np.ndarray, text_boxes: Sequence[Rect]) -> Tuple[np.ndarray, np.ndarray]:
    """``(separators, lines)``: boolean masks of the page's long straight
    dark runs (horizontal or vertical, at least ``_BORDER_THICK`` thick), with
    the text lines painted out first so a column of glyphs never counts.
    ``separators`` also contains solid dark areas (they satisfy the straight
    kernels too) and is what splits the page into panels; ``lines`` has
    those blobs removed and is the part lettering must never cross: a
    haloed block may sit on black hair, never on a gutter line."""
    h, w = gray.shape[:2]
    dark = (gray < _BORDER_DARK).astype(np.uint8)
    for tb in text_boxes:
        g = _grow(tb, _TEXT_PAD, _TEXT_PAD).clamp(w, h)
        dark[g.y : g.y2, g.x : g.x2] = 0
    length = max(20, int(_BORDER_LEN_FRAC * min(h, w)))
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((_BORDER_THICK, length), np.uint8))
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((length, _BORDER_THICK), np.uint8))
    separators = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
    blob = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((_BLOB_PX, _BLOB_PX), np.uint8))
    blob = cv2.dilate(blob, np.ones((_BLOB_PX, _BLOB_PX), np.uint8))
    lines = separators & (1 - blob)
    return separators.astype(bool), lines.astype(bool)


def _panel_label(labels: np.ndarray, stats: np.ndarray, src: Rect) -> int:
    """Label of the panel component holding ``src`` (its centre, else the
    majority of its pixels); 0 when none is large enough to trust."""
    h, w = labels.shape
    cy = min(h - 1, max(0, src.y + src.h // 2))
    cx = min(w - 1, max(0, src.x + src.w // 2))
    label = int(labels[cy, cx])
    if label == 0:
        r = src.clamp(w, h)
        inside = labels[r.y : r.y2, r.x : r.x2]
        if inside.size:
            vals, counts = np.unique(inside[inside > 0], return_counts=True)
            if vals.size:
                label = int(vals[np.argmax(counts)])
    if label > 0 and stats[label, cv2.CC_STAT_AREA] < _MIN_PANEL_FRAC * h * w:
        return 0
    return label


def _panel_rect(stats: np.ndarray, label: int) -> Optional[Rect]:
    """Bounding rectangle (frame coordinates) of panel component ``label``;
    None for label 0 (no panel could be identified)."""
    if label <= 0:
        return None
    return Rect(
        int(stats[label, cv2.CC_STAT_LEFT]),
        int(stats[label, cv2.CC_STAT_TOP]),
        int(stats[label, cv2.CC_STAT_WIDTH]),
        int(stats[label, cv2.CC_STAT_HEIGHT]),
    )


def _room(src: Rect, dx: int, dy: int, lines_wide: np.ndarray, labels: np.ndarray, panel: int, limit: int) -> int:
    """Free pixels beyond the edge of ``src`` in direction ``(dx, dy)`` (one
    of them zero) along the line through its centre, up to ``limit``: until
    a border line, another panel or the page edge."""
    h, w = lines_wide.shape
    cx, cy = src.x + src.w // 2, src.y + src.h // 2
    if dx:
        y = cy
        if not (0 <= y < h):
            return 0
        if dx > 0:
            x0, x1 = src.x2, min(w, src.x2 + limit)
            if x1 <= x0:
                return 0
            stop = lines_wide[y, x0:x1] | ((labels[y, x0:x1] != 0) & (labels[y, x0:x1] != panel) if panel > 0 else False)
        else:
            x1, x0 = src.x, max(0, src.x - limit)
            if x1 <= x0:
                return 0
            seg_l, seg_lab = lines_wide[y, x0:x1][::-1], labels[y, x0:x1][::-1]
            stop = seg_l | ((seg_lab != 0) & (seg_lab != panel) if panel > 0 else False)
    else:
        x = cx
        if not (0 <= x < w):
            return 0
        if dy > 0:
            y0, y1 = src.y2, min(h, src.y2 + limit)
            if y1 <= y0:
                return 0
            stop = lines_wide[y0:y1, x] | ((labels[y0:y1, x] != 0) & (labels[y0:y1, x] != panel) if panel > 0 else False)
        else:
            y1, y0 = src.y, max(0, src.y - limit)
            if y1 <= y0:
                return 0
            seg_l, seg_lab = lines_wide[y0:y1, x][::-1], labels[y0:y1, x][::-1]
            stop = seg_l | ((seg_lab != 0) & (seg_lab != panel) if panel > 0 else False)
    stop = np.asarray(stop, dtype=bool)
    hit = np.flatnonzero(stop)
    return int(hit[0]) if hit.size else int(stop.size)


def _zone_shift(src: Rect, other: Rect, lines_wide: np.ndarray, labels: np.ndarray, panel: int, em: float, gap: float) -> float:
    """How far (pixels of distance difference) the zone boundary between
    ``src`` and ``other`` moves towards ``other``: the block with less room
    on its far side (panel border, page edge) gets more of the corridor,
    the boundary never crossing either source edge (minus the gap)."""
    sep_x = max(other.x - src.x2, src.x - other.x2)
    sep_y = max(other.y - src.y2, src.y - other.y2)
    if max(sep_x, sep_y) <= gap:
        return 0.0  # sources touch or overlap along the axis: plain midway split
    limit = int(round(_REACH_X_EM * em))
    if sep_x >= sep_y:
        away = 1 if src.x > other.x else -1  # from the other block towards this one
        room_self = _room(src, away, 0, lines_wide, labels, panel, limit)
        room_other = _room(other, -away, 0, lines_wide, labels, panel, limit)
        sep = sep_x
    else:
        away = 1 if src.y > other.y else -1
        room_self = _room(src, 0, away, lines_wide, labels, panel, limit)
        room_other = _room(other, 0, -away, lines_wide, labels, panel, limit)
        sep = sep_y
    shift = float(room_other - room_self)
    bound = float(sep - gap)
    return max(-bound, min(bound, shift))


def prepare(img_bgr: np.ndarray, blocks: Sequence, gray: Optional[np.ndarray] = None) -> None:
    """Compute, for every free-text block in ``blocks`` (``layout.TextBlock``
    objects), the data :func:`typeset_block` places with, and store it in
    the block's style: ``search_box``, ``ink_map`` and ``blocked_map``.
    Every block (bubbles included) also gets its ``panel_box``, the panel
    component its source text sits in, which the quality renderer groups its
    inpainting jobs by.
    Bubble blocks are otherwise left alone (they are centred in their bubble).  Cheap:
    two morphological openings on the page plus small per-block crops.
    ``gray`` may be the page's grey image (saves a conversion)."""
    blocks = [b for b in blocks if b.style.layout_box is not None]
    if not blocks:
        return
    h, w = img_bgr.shape[:2]
    if gray is None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ink_light = gray < _INK_LIGHT
    text_boxes = [b.segment.bbox for b in blocks]
    separators, lines = panel_borders(gray, text_boxes)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((~separators).astype(np.uint8), connectivity=4)
    panels = [_panel_label(labels, stats, b.segment.bbox) for b in blocks]
    # The panel a block lies in is also what ``render/quality.py`` groups its
    # inpainting jobs by, so every block gets it - bubbles included.
    for bi, b in enumerate(blocks):
        b.style.panel_box = _panel_rect(stats, panels[bi])
    # Lettering keeps a margin from border lines and from the page edge,
    # unless the source text itself hugs a border (a caption in a gutter):
    # then the letterer is as tight as the original was.
    ems = [size_ceiling(b.style) for b in blocks]
    edge = max(2, int(round(_EDGE_MARGIN_EM * float(np.median(ems)))))
    lines_u8 = lines.astype(np.uint8)
    lines_u8[:edge, :] = 1
    lines_u8[-edge:, :] = 1
    lines_u8[:, :edge] = 1
    lines_u8[:, -edge:] = 1
    lines_wide = cv2.dilate(lines_u8, np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)).astype(bool)
    lines = lines_u8.astype(bool)

    for bi, b in enumerate(blocks):
        st: SegmentStyle = b.style
        if st.in_bubble:
            continue
        em = size_ceiling(st)
        src = b.segment.bbox
        box = _grow(src, _REACH_X_EM * em, _REACH_Y_EM * em).clamp(w, h)
        if box.w <= 0 or box.h <= 0:
            continue
        crop = (slice(box.y, box.y2), slice(box.x, box.x2))
        dark_bg = _luma(st.bg) < 128.0
        ink = (gray[crop] > _INK_DARK) if dark_bg else ink_light[crop]
        ink = ink.astype(np.uint8)
        # The block's own glyphs are erased before lettering: they are paper.
        for line in list(getattr(b, "members", [])) + list(getattr(b, "furigana", [])):
            _paint(ink, _grow(line.bbox, _TEXT_PAD, _TEXT_PAD), box, 0)
        # Flat-paper text: whatever layout.py excluded from the paper counts as ink.
        if st.layout_mask is not None and st.layout_box is not None and not st.outline:
            lb = st.layout_box
            inter = Rect(max(lb.x, box.x), max(lb.y, box.y), 0, 0)
            inter = Rect(inter.x, inter.y, min(lb.x2, box.x2) - inter.x, min(lb.y2, box.y2) - inter.y)
            if inter.w > 0 and inter.h > 0 and st.layout_mask.shape[:2] == (lb.h, lb.w):
                sub = st.layout_mask[inter.y - lb.y : inter.y2 - lb.y, inter.x - lb.x : inter.x2 - lb.x]
                ink[inter.y - box.y : inter.y2 - box.y, inter.x - box.x : inter.x2 - box.x] |= (~sub).astype(np.uint8)

        src_clamped = src.clamp(w, h)
        hugs_border = bool(lines_wide[src_clamped.y : src_clamped.y2, src_clamped.x : src_clamped.x2].any())
        blocked = (lines if hugs_border else lines_wide)[crop].copy()
        panel = panels[bi]
        if panel > 0:
            # Other panels are out of bounds; separator pixels themselves
            # (label 0) stay allowed unless they are thin lines: solid
            # artwork may carry haloed lettering, a gutter line may not.
            crop_labels = labels[crop]
            blocked |= (crop_labels != panel) & (crop_labels != 0)
        margin = _OBSTACLE_MARGIN_EM * em
        gap = _ZONE_GAP_EM * em
        d_self: Optional[np.ndarray] = None
        near = _grow(box, box.w, box.h)
        for oi, other in enumerate(blocks):
            if other is b:
                continue
            if panel > 0 and panels[oi] > 0 and panels[oi] != panel:
                continue  # another panel: already out of bounds
            ost: SegmentStyle = other.style
            obstacle = ost.layout_box if (ost.in_bubble and ost.layout_box is not None) else other.segment.bbox
            if not near.intersects(obstacle):
                continue
            _paint(blocked, _grow(obstacle, margin, margin), box, True)
            if ost.in_bubble:
                continue  # a bubble's lettering stays inside it: the rect is enough
            # Free-text neighbours: the space between two sources is split so
            # neither block's lettering can reach the other's.  The split is
            # midway, shifted towards the block with more room on its far
            # side (a block against a panel border gets the corridor; the
            # letterer makes the other one yield), never past a source edge.
            if d_self is None:
                d_self = _rect_distance(src, box)
            osrc = other.segment.bbox
            shift = _zone_shift(src, osrc, lines_wide, labels, panel, em, gap)
            thr = d_self + gap - shift  # d_other < thr  <=>  d_other^2 < thr^2 (thr > 0)
            blocked |= (thr > 0) & (_rect_distance_sq(osrc, box) < thr * thr)
        st.search_box = box
        st.ink_map = ink
        st.blocked_map = blocked


# ------------------------------------------------------------ typeset_block
def _block_spans(style: SegmentStyle, box: Rect, cx: Optional[float] = None) -> np.ndarray:
    if style.layout_mask is not None and style.layout_mask.shape[:2] == (box.h, box.w):
        return mask_spans(style.layout_mask, box, cx)
    return rect_spans(box)


def _source_rect(style: SegmentStyle) -> Rect:
    if style.source_quads is not None and len(style.source_quads):
        pts = np.asarray(style.source_quads, dtype=np.float64).reshape(-1, 2)
        return Rect.from_quad(pts)
    if style.layout_seed is not None:
        return style.layout_seed
    assert style.layout_box is not None
    return style.layout_box


def _box_sums(integral: np.ndarray, xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> np.ndarray:
    """Sums of the map under a ``w x h`` window at every ``(y, x)`` in
    ``ys x xs`` (top-left corners), from its integral image."""
    y0, y1 = ys[:, None], ys[:, None] + h
    x0, x1 = xs[None, :], xs[None, :] + w
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def _grid(extent: int, size: int, step: int) -> np.ndarray:
    """Top-left offsets of a ``size`` window inside ``extent``, every ``step``
    pixels, always including the last valid offset."""
    last = extent - size
    if last < 0:
        return np.zeros(0, dtype=np.int64)
    xs = np.arange(0, last + 1, step, dtype=np.int64)
    if xs[-1] != last:
        xs = np.append(xs, last)
    return xs


class _Candidate:
    __slots__ = ("cost", "size", "n", "width", "x", "y", "glyph_h", "pitch", "words", "forced")

    def __init__(
        self, cost: float, size: float, n: int, width: float, x: int, y: int, glyph_h: float, pitch: float,
        words: Sequence[str], forced: Sequence[int],
    ) -> None:
        self.cost, self.size, self.n, self.width = cost, size, n, width
        self.x, self.y, self.glyph_h, self.pitch = x, y, glyph_h, pitch
        self.words, self.forced = tuple(words), tuple(forced)


def hyphenate_widest(words: Sequence[str], size: float, measure: Measure, lang: str) -> Optional[Tuple[List[str], int]]:
    """Split the widest word at the dictionary point that best balances its
    halves: ``(new words, index of the head)`` (the head ends with ``-`` and
    a line must end after it), or None when it has no legal break."""
    if not words:
        return None
    widest = max(range(len(words)), key=lambda i: measure(words[i], size)[0])
    word = words[widest]
    points = hyphenation_points(word, lang)
    if not points:
        return None
    def widest_half(p: int) -> float:
        head, tail = split_at(word, p)
        return max(measure(head, size)[0], measure(tail, size)[0])

    head, tail = split_at(word, min(points, key=widest_half))
    out = list(words[:widest]) + [head, tail] + list(words[widest + 1 :])
    return out, widest


def _search(
    words: Sequence[str],
    text: str,
    size: float,
    measure: Measure,
    search: Rect,
    ii_ink: np.ndarray,
    ii_blocked: np.ndarray,
    source: Rect,
    band: Tuple[float, float],
    trace: Optional[list] = None,
    must_cover: bool = False,
    forced: Sequence[int] = (),
    max_lines: int = _MAX_LINES,
    min_aspect: float = 0.0,
) -> Optional[_Candidate]:
    """Best (lowest-cost) feasible shape + position at ``size``, or None.
    ``trace`` (debugging) receives one dict per line count with the cost
    components of that shape's best position.

    Cost of a footprint (block + halo margin) = mean squared ink density
    under it (the central half counted again) + displacement from the
    source (free while the footprint still covers the source centre, plus a
    small centring term) + reach beyond the source footprint past
    ``_EXT_FREE_EM`` (squared) + extra lines beyond ``_LINES_PER_WORD`` +
    width beyond ``_WIDTH_IDEAL_EM`` + raggedness of the balanced lines
    (``_BALANCE_WEIGHT``) + aspect outside the preferred band.  Shapes whose
    lettering (the footprint without its halo margin: a paper-coloured halo
    may overlap a margin) touches a blocked pixel are infeasible (and, with
    ``must_cover``, those not covering the source centre).  Shapes with more
    than ``max_lines`` lines or narrower than ``min_aspect`` (width /
    height) are never considered."""
    glyph_h = measure("Mg", size)[1]
    pitch = line_pitch(text, glyph_h)
    ink_top, ink_bottom = ink_offsets(text, glyph_h)
    ink_h1 = ink_bottom - ink_top
    margin = HALO_RATIO * size + 1.0
    inset = int(margin)  # the lettering itself starts this far inside the footprint
    step = max(2, int(round(_GRID_STEP_EM * size)))
    width_of = span_widths(words, size, measure)
    count = len(words)
    scx, scy = source.x + source.w / 2.0, source.y + source.h / 2.0
    n_ideal = math.ceil(count * _LINES_PER_WORD)

    best: Optional[_Candidate] = None
    prev_w: Optional[float] = None
    for n in range(1, min(count, max_lines) + 1):
        w_line = min_max_width(count, n, width_of, forced)
        if w_line is None:
            continue
        if prev_w is not None and w_line >= prev_w - 0.5:
            continue  # n-1 lines already gave this width: same shape
        prev_w = w_line
        ink_h = (n - 1) * pitch + ink_h1
        if ink_h > 0 and w_line / ink_h < min_aspect:
            break  # taller shapes only get narrower
        fw = int(math.ceil(w_line + 2 * margin))
        fh = int(math.ceil(ink_h + 2 * margin))
        xs = _grid(search.w, fw, step)
        ys = _grid(search.h, fh, step)
        if xs.size == 0 or ys.size == 0:
            continue
        blocked = _box_sums(ii_blocked, xs + inset, ys + inset, fw - 2 * inset, fh - 2 * inset)
        feasible = blocked == 0
        if not feasible.any():
            continue
        area = float(fw * fh)
        ink_full = _box_sums(ii_ink, xs, ys, fw, fh) / area
        iw, ih = max(1, int(fw * _INNER_FRAC)), max(1, int(fh * _INNER_FRAC))
        ix, iy = xs + (fw - iw) // 2, ys + (fh - ih) // 2
        ink_inner = _box_sums(ii_ink, ix, iy, iw, ih) / float(iw * ih)
        ink_cost = 0.5 * ink_full + 0.5 * ink_inner
        cx = search.x + xs[None, :] + fw / 2.0
        cy = search.y + ys[:, None] + fh / 2.0
        dx, dy = np.abs(cx - scx), np.abs(cy - scy)
        outside = np.hypot(np.maximum(dx - fw / 2.0, 0.0), np.maximum(dy - fh / 2.0, 0.0))
        if must_cover:
            feasible &= outside <= 0.0
            if not feasible.any():
                continue
        dist_cost = (_LAMBDA_OUT * outside + _LAMBDA_CENTRE * np.hypot(dx, dy)) / size
        # Reach beyond the source footprint (the letterer stays near it).
        fx, fy = search.x + xs[None, :], search.y + ys[:, None]
        ext = np.maximum.reduce([
            np.maximum(source.x - fx, 0.0) + np.zeros_like(fy, dtype=np.float64),
            np.maximum(fx + fw - source.x2, 0.0) + np.zeros_like(fy, dtype=np.float64),
            np.maximum(source.y - fy, 0.0) + np.zeros_like(fx, dtype=np.float64),
            np.maximum(fy + fh - source.y2, 0.0) + np.zeros_like(fx, dtype=np.float64),
        ]) / size
        ext_cost = _EXT_WEIGHT * np.maximum(ext - _EXT_FREE_EM, 0.0) ** 2
        line_cost = _LINE_WEIGHT * max(0, n - n_ideal)
        width_cost = _WIDTH_WEIGHT * max(0.0, w_line / size - _WIDTH_IDEAL_EM)
        balance_cost = _BALANCE_WEIGHT * _raggedness(count, n, w_line, width_of, forced)
        aspect = w_line / ink_h if ink_h > 0 else float("inf")
        aspect_cost = _ASPECT_WEIGHT * (max(0.0, band[0] - aspect) + max(0.0, aspect - band[1]))
        shape_cost = line_cost + width_cost + balance_cost + aspect_cost
        cost = np.where(feasible, ink_cost + dist_cost + ext_cost + shape_cost, np.inf)
        k = int(np.argmin(cost))
        c = float(cost.flat[k])
        if not np.isfinite(c):
            continue
        yi, xi = divmod(k, xs.size)
        if trace is not None:
            trace.append(dict(n=n, width=round(w_line), fp=(fw, fh), x=search.x + int(xs[xi]), y=search.y + int(ys[yi]),
                              feasible=round(float(feasible.mean()), 3), ink=round(float(ink_cost[yi, xi]), 4),
                              dist=round(float(dist_cost[yi, xi]), 4), ext=round(float(ext_cost[yi, xi]), 4),
                              lines=round(line_cost, 4), wide=round(width_cost, 4), balance=round(balance_cost, 4),
                              aspect=round(aspect_cost, 4), total=round(c, 4)))
        if best is None or c < best.cost:
            best = _Candidate(c, size, n, w_line, int(xs[xi]), int(ys[yi]), glyph_h, pitch, words, forced)
    return best


def _oval_targets(n: int, width: float) -> List[float]:
    targets = [width] * n
    if n >= 3:
        targets[0] = targets[-1] = width * _OVAL_END_FRAC
    return targets


def _breaks(count: int, n: int, width: float, width_of, forced: Sequence[int]) -> List[int]:
    """Balanced line starts for ``n`` lines of at most ``width`` (mildly
    oval targets); one word per line if the DP has no solution (it always
    has one when ``width`` came from :func:`typeset.min_max_width`)."""
    starts = balanced_breaks(count, n, width + 0.01, width_of, _oval_targets(n, width), forced)
    return starts if starts is not None else list(range(count + 1))


def _raggedness(count: int, n: int, width: float, width_of, forced: Sequence[int]) -> float:
    """Normalised squared shortfall of the balanced lines against their
    (oval) targets: 0 for a perfectly even block, about 0.1 when a single
    word sticks out over lines half its width."""
    if n < 2 or width <= 0:
        return 0.0
    starts = _breaks(count, n, width, width_of, forced)
    targets = _oval_targets(len(starts) - 1, width)
    short = [max(0.0, t - width_of(starts[i], starts[i + 1])) for i, t in enumerate(targets)]
    return sum(v * v for v in short) / (len(short) * width * width)


def _flow_candidate(text: str, cand: _Candidate, measure: Measure, search: Rect) -> Typeset:
    """Set the words in the chosen shape at the chosen place: balanced lines
    (mildly oval), centred on the footprint."""
    words = cand.words
    width_of = span_widths(words, cand.size, measure)
    count = len(words)
    starts = _breaks(count, cand.n, cand.width, width_of, cand.forced)
    lines = [" ".join(words[starts[i] : starts[i + 1]]) for i in range(len(starts) - 1)]
    margin = HALO_RATIO * cand.size + 1.0
    ink_top, _ = ink_offsets(text, cand.glyph_h)
    cx = search.x + cand.x + margin + cand.width / 2.0
    top0 = search.y + cand.y + margin - ink_top
    placed = [
        PlacedLine(line, cx, top0 + i * cand.pitch, measure(line, cand.size)[0]) for i, line in enumerate(lines)
    ]
    return Typeset(cand.size, cand.pitch, placed, True, 0, cand.glyph_h)


def _maps(style: SegmentStyle, box: Rect, em: float) -> Tuple[Rect, np.ndarray, np.ndarray]:
    """``(search box, squared-ink-density integral, blocked integral)``;
    without :func:`prepare` data the layout box is the search box and
    everything in it is quiet paper.  The ink map is averaged over a
    ``_DENSITY_EM`` window and squared so that texture (hatching, tone) is
    cheap to letter over and solid ink is expensive."""
    search = style.search_box
    ink = style.ink_map
    blocked = style.blocked_map
    if search is None or ink is None or ink.shape[:2] != (search.h, search.w):
        search = box
        ink = np.zeros((box.h, box.w), dtype=np.uint8)
        blocked = None
    if blocked is None or blocked.shape[:2] != (search.h, search.w):
        blocked = np.zeros((search.h, search.w), dtype=np.uint8)
    k = max(1, int(round(_DENSITY_EM * em)))
    density = cv2.blur(np.ascontiguousarray(ink, dtype=np.float32), (k, k), borderType=cv2.BORDER_REPLICATE)
    ii_ink = cv2.integral(density * density)
    ii_blocked = cv2.integral(np.ascontiguousarray(blocked, dtype=np.uint8))
    return search, ii_ink, ii_blocked


def typeset_block(text: str, style: SegmentStyle, measure: Measure, *, min_size: float = 7.0, lang: str = "en") -> Typeset:
    """Typeset ``text`` for a block (drop-in for ``compose.typeset_block``).

    Bubbles (``style.in_bubble``): the text is flowed through the bubble
    mask and centred in it at the largest size that fits (tight leading);
    two joined boxes holding a sentence with an ellipsis boundary get one
    part per box (:func:`typeset.typeset_lobes`).

    Free text: the block shape and position are chosen like a letterer's
    (see the module docstring), using the maps from :func:`prepare` when
    present, else the layout box.  Horizontal captions (titles) keep
    covering their source and stay wide: at most ``_MAX_LINES_H`` lines, at
    least the aspect ``_ASPECT_BAND_H[0]``.  The size is the ceiling
    (:func:`size_ceiling`) unless a smaller size gives a better block: the
    size shrinks by ``_SHRINK_STEP`` steps while the shrink penalty alone
    (``_SHRINK_WEIGHT``) stays below the best total so far (never below
    ``_MIN_SCALE`` of the ceiling once a place exists, never below
    ``min_size``); when the widest word fits nowhere at a size it is
    hyphenated at a dictionary point (``_HYPHEN_COST``) before the size is
    reduced.  When nothing is feasible at all the text is flowed
    through the search box with :func:`typeset.typeset` (``fitted`` may be
    False), so a result is always produced.
    """
    box = style.layout_box
    assert box is not None
    measure = memoize_measure(measure)
    ceiling = max(size_ceiling(style, min_size), min_size)
    words = text.split()
    if not words:
        return Typeset(ceiling, ceiling, [], True)
    if style.in_bubble or (style.layout_seed is None and style.search_box is None):
        spans = _block_spans(style, box)
        if style.in_bubble and not style.vertical:
            # A horizontal source is a *line* - a caption band's text sits on
            # a printed rule and the letterer keeps the English on it - so the
            # block is anchored where the Japanese was, exactly as a
            # horizontal free-text caption is by ``must_cover`` below.  The
            # optical centre is the middle of the *shape*, which on a band the
            # text fills a third of (1ja's caption strip: 660 x 152 of band
            # for Japanese 86 px deep) drops the lettering ~17 px below its
            # own rule.  Vertical text runs the height of its balloon and so
            # says nothing about where the English belongs; those keep the
            # optical centre, and a balloon its text does fill gives the same
            # answer either way.
            src = _source_rect(style)
            return typeset(text, spans, float(box.y), measure, max_size=ceiling, min_size=min_size,
                           anchor_y=src.y + src.h / 2.0, lang=lang)
        return typeset_lobes(text, spans, float(box.y), measure, max_size=ceiling, min_size=min_size, lang=lang)

    search, ii_ink, ii_blocked = _maps(style, box, ceiling)
    src = _source_rect(style)
    centre = (src.x + src.w / 2.0, src.y + src.h / 2.0)
    band = _ASPECT_BAND_V if style.vertical else _ASPECT_BAND_H
    # Horizontal captions stay where they were (the block keeps covering the
    # source centre) and stay wide; dialogue may move, at a cost.
    must_cover = not style.vertical
    max_lines = _MAX_LINES if style.vertical else _MAX_LINES_H
    min_aspect = 0.0 if style.vertical else band[0]

    best: Optional[_Candidate] = None
    best_total = float("inf")
    size = ceiling
    while True:
        cand = _search(words, text, size, measure, search, ii_ink, ii_blocked, src, band, must_cover=must_cover,
                       max_lines=max_lines, min_aspect=min_aspect)
        if cand is None:
            # Before shrinking, try the letterer's other move: hyphenate the
            # widest word (the reference page does this in narrow boxes).
            split = hyphenate_widest(words, size, measure, lang)
            if split is not None:
                cand = _search(split[0], text, size, measure, search, ii_ink, ii_blocked, src, band,
                               must_cover=must_cover, forced=(split[1],), max_lines=max_lines, min_aspect=min_aspect)
                if cand is not None:
                    cand.cost += _HYPHEN_COST
        if cand is not None:
            total = cand.cost + _SHRINK_WEIGHT * (1.0 - size / ceiling)
            if total < best_total:
                best, best_total = cand, total
        if size <= min_size:
            break
        next_size = max(size * _SHRINK_STEP, min_size)
        if best is not None and (next_size < _MIN_SCALE * ceiling or _SHRINK_WEIGHT * (1.0 - next_size / ceiling) >= best_total):
            break  # smaller cannot win: its shrink penalty alone is already worse
        size = next_size

    if best is not None:
        return _flow_candidate(text, best, measure, search)
    # Nothing feasible: flow through the search box, anchored on the source.
    spans = rect_spans(search)
    if style.layout_mask is not None and style.layout_mask.shape[:2] == (box.h, box.w):
        spans, top = clip_spans(_block_spans(style, box, centre[0]), float(box.y), box)
        if spans.shape[0] == 0:
            spans, top = rect_spans(search), float(search.y)
    else:
        top = float(search.y)
    return typeset(text, spans, top, measure, max_size=ceiling, min_size=min_size, anchor_y=centre[1], lang=lang)
