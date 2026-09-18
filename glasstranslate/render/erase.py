"""Glyph eraser for manga text blocks.

Computes ``clean_patch`` / ``clean_rect`` / ``outline`` for the blocks of
:mod:`glasstranslate.render.layout`.  The goal is a page that looks as if
the original letterer had never put the Japanese on it: paper stays paper,
glyphs vanish completely (including the ones the OCR box was too short to
cover), and the line-art and screentone around and under the text is kept.

Per block the work happens in a window around the source text, in *column
coordinates* (horizontal lines are transposed so that text always runs along
``y``, with furigana on the ``+x`` side for vertical script and the ``-x``
side for horizontal script):

1. **Ink components** - the tight ink mask (pixels closer to the text colour
   than to the paper colour) is split into 8-connected components.
2. **Classification** - a component mostly inside the OCR line/furigana boxes
   is a glyph; a glyph-sized component poking out of a box is a glyph; a
   large component that only passes through the boxes is *art*; a thin
   diagonal stroke parallel to a hatching field around the text is art too.
   Components that belong to the boxes of another block are left alone.
3. **Glyph completion** - along every column axis the sweep extends the
   column over glyph-bodied components centred on the axis within
   ``SWEEP_GAP`` glyphs (up to ``SWEEP_MAX`` glyphs), collects the small
   pieces inside the extended column, then checks the furigana strip beside
   it.  Pieces that touch art are never collected.
4. **Art protection** - only ink inside the *corridor* (the glyph columns,
   not the whole of an inflated multi-column OCR box) is erased, so art in
   the margin of such a box survives; inside the corridor the solid
   bodies (thicker than any glyph stroke) and the straight lines that enter
   the text from outside (``HoughLinesP`` on the art, followed through the
   text with drift correction; the line keeps its own width where a glyph
   stroke crosses it) survive.
5. **Fill** - the erase set is dilated by ``ERASE_DILATE`` px to swallow
   anti-aliased edges (never into kept ink), and filled with the *local
   background*: exact paper colour on paper, the local mean grey on a
   screentone, so neither white blotches nor grey smears appear.  Near-paper
   JPEG ringing in a ring around the erase set is flattened on paper.
6. **Bubbles** are redrawn, never inpainted: the paper interior (the paper
   component holding the text, every enclosed glyph, furigana mark or stray
   filled in) is painted with the paper colour, clipped to the outline; the
   outline and its anti-aliased fringe are copied from the source, so it keeps
   its own thickness and its tail.  Nothing outside the outline changes.

Public API: :func:`apply` (whole page, in place on the block styles),
:func:`erase_block` (one block -> ``(patch, rect, outline)``),
:func:`erase_block_masked` (the same plus the glyph mask that was painted
over, which the quality renderer of ``render/quality.py`` regenerates) and
:func:`page_glyph` (the page's lettering scale, which bounds every reach above).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.types import RGB, Rect, Segment
from .layout import TextBlock, glyph_em

# --- tunables --------------------------------------------------------------
PAD = 2  # px added around OCR boxes when deciding what is "inside" a line
RECT_PAD = 2  # px margin of clean_rect around the erase set
GLYPH_MAX = 1.4  # a component with both sides <= this many glyphs is glyph-sized
GLYPH_MIN = 0.2  # completion candidates must be at least this big (glyphs)
PIECE_MIN = 0.1  # small pieces collected inside a swept column / furigana strip (glyphs)
MIN_FILL = 0.18  # ink area / bbox area below which a candidate is a thin diagonal art stroke
FULL_INSIDE = 0.8  # fraction of a component inside the boxes that makes it a glyph outright
NEAR_ART = 3  # px: pieces this close to art (big components) are art fragments, not glyphs
# Hatching rescue.  A broken speed line inside an OCR box is glyph-sized, partly
# inside and not touching the border, so the free-text branch of `_classify`
# claims it as a glyph; `_hatching` is the only thing that gives it back.  These
# three were measured together over the six worst-damaged corpus pages: the
# artwork we destroy falls 51.1 % -> 43.7 % and erase precision rises 0.648 ->
# 0.657, at a recall cost of 0.939 -> 0.921.  Swept one at a time first, so the
# contribution of each is known: reach 47.6 %, min-len 49.0 %, field 49.8 %.
#
# `HATCH_REACH` is the strongest and saturates at 2.0 (3.0 measured slightly
# worse on both axes).  `HATCH_FILL` and `HATCH_ANGLE` were swept and do
# essentially nothing - which is the useful finding: these strokes are not
# failing the direction test, they are failing to find a FIELD to belong to,
# and reach is what finds it.
HATCH_MIN_LEN = 0.20  # glyphs: shortest stroke that can be a hatching stroke
HATCH_FILL = 0.35  # ink / bbox area above which a stroke is not a thin diagonal
HATCH_ELONG = 3.0  # major / minor extent of a hatching stroke
HATCH_AXIS = 15.0  # degrees from horizontal/vertical below which a stroke may be a glyph stroke
HATCH_ANGLE = 15.0  # degrees a stroke may deviate from a hatching direction of the field
HATCH_FIELD = 2  # strokes of one direction outside the text (within HATCH_REACH) that make a hatching field
HATCH_REACH = 2.0  # glyphs around the text boxes in which the field is looked for
SWEEP_GAP = 0.8  # gap (glyphs) bridged when completing a column
SWEEP_MAX = 2.5  # how far (glyphs) a column may be extended in one direction
SWEEP_BODY = 0.4  # only components at least this big (glyphs) extend a column
SWEEP_ALIGN = 0.4  # ...and only when centred within this many glyphs of the column axis
SWEEP_CORRIDOR = 0.3  # corridor widening on each side, as a fraction of the line width
AXIS_BOX_MAX = 1.35  # a vertical box wider than this many glyphs spans several columns
CORRIDOR_PAD = 0.1  # glyphs: margin of the erase corridor around the glyph columns
FURIGANA_MAX = 0.75  # furigana components are at most this many glyphs
FURIGANA_ZONE = (-0.15, 0.7)  # furigana strip beside the column edge (glyphs from the glyph edge)
FURIGANA_NOISE = 0.08  # ink fraction of the strip (besides the furigana) above which the strip is art
SOLID_DT = 3.0  # minimum half-thickness (px) of a solid body; raised to the glyph stroke + 1
HOUGH_MIN_LEN = 0.8  # glyphs
HOUGH_OUTSIDE = 0.4  # glyphs of a straight line outside the text zone to count as continuing art
HOUGH_OUTSIDE_HATCH = 0.15  # ...for a line in a direction of the hatching field around the text
HOUGH_EXTEND = 1.2  # glyphs a kept art line is extrapolated on each side through the text zone
# Longest candidate lines followed per block.  This is a COST BOUND, not a
# classifier: every line that reaches it has already been detected on the art
# mask (glyphs removed) and must still show clear run outside the text zone, so
# following more of them can only keep ink, never erase more.
#
# It was 30, and hatching does yield hundreds - measured over 11 blocks on the
# four worst-damaged corpus pages, `_art_lines` detects a median of 182
# candidates and up to 1097, with 30 of 33 calls over the old cap.  The worst
# block protected 2.7 % of the art it had already identified and the rest was
# erased.  Sweeping the cap moved the share of the letterer's kept artwork we
# destroy: 59.4 % at 30, 55.4 % at 100, 51.4 % at 300, 49.9 % at 1200, for
# +19 % erase time at 300 on those four pages and nothing at all on pages that
# never reach 30 candidates.  300 is where the curve turns; past it the cost
# keeps rising and the recovery does not.
#
# Raising this does NOT fix the defect on its own - half the artwork still dies
# at 1200, so the remaining damage is somewhere else.
HOUGH_MAX_LINES = 300
# px: the largest glyph the art-line DETECTOR scales its thresholds by.
#
# Everything else in this module is a multiple of the block's glyph because it describes the
# lettering.  These two do not: they describe the ARTWORK, and artwork does not get bigger
# because the lettering beside it does.  Measured over the 50-page sample, the straight-line
# lengths of a page's art are the same everywhere regardless of its lettering scale - p50 14
# to 24 px, p75 23 to 47 px - while `HOUGH_MIN_LEN * glyph` ranges from 24 px to 193 px.  So
# on `v21_p145_5879c9` (page lettering 241 px) art protection demanded a 193 px straight run
# and **0.2 % of that page's art lines could supply one**, against 46 % on a page lettered at
# 33 px.  Art protection was therefore switched off on exactly the pages where the erase
# reach is largest - the same compounding failure the page-glyph cap fixed in the other half.
#
# The glyph strokes are already removed from the mask Hough runs on (`art[sel_mask] = 0` in
# `_keep_art`), so a lower minimum cannot start reading lettering as artwork.  38 px is the
# corpus-median page lettering scale, i.e. the requirement a normally lettered page already
# makes; 25 of the 49 pages are bit-identical under it.
HOUGH_GLYPH_MAX = 38.0
LINE_REACH = 4  # px looked at on each side of a followed line to measure the ink run across it
LINE_SNAP = 2  # px the followed line may drift off the ink and still be snapped onto it
LINE_SLACK = 1.5  # px a run inside the text may exceed the line's own thickness and still be line
LINE_GAP = 3  # px of missing ink along a line that do not end it
LINE_DENSITY = 0.6  # fraction of the detected line (outside the text) that must be ink
ART_MIN_PX = 12  # art pixels inside the corridor below which no lines are followed
ERASE_DILATE = 2  # px
RING = 3  # px beyond the erase set in which near-paper ringing is flattened
RING_TOL = 12  # grey levels from the paper colour that count as ringing
TONE_RADIUS = 0.5  # glyphs: radius of the local background estimate used for the fill
TONE_DOT = 4  # px: ink components up to this size are screentone dots and count as background
TONE_TOL = 12  # grey levels: a local background this close to the paper colour is paper
TONE_MIN_SAMPLES = 0.1  # fraction of the estimate's square that must be background to trust the mean
PAPER_TOL = 45  # grey levels from the paper colour that still count as paper (bubbles)
OUTLINE_MIN = 0.35  # bubbles: an ink component at least this fraction of the window's short side may be the outline
# ...but only if it is not the lettering itself: a component at least this far inside the block's
# own text boxes is the text, which is glyph-sized on light paper but runs the length of a column
# on dark paper, where the ink mask selects light pixels.  Measured over the corpus the two
# populations do not overlap - outlines sit at 0.000-0.013 of their area inside the boxes and
# lettering at 0.422-1.000 - and the result is identical anywhere in 0.30-0.40.  Above 1.0 nothing
# is excluded, which is the disabled setting.
OUTLINE_IN_TEXT = 0.35
OUTLINE_FRINGE = 1  # px of anti-aliased edge kept with a bubble outline
# Bubbles: the paper fill reaches this many glyphs beyond the text boxes (strays and furigana
# next to the column are filled; art drawn across the balloon farther away is kept).
BUBBLE_ZONE_EM = 1.5
# ...and, however far from the boxes, over any glyph-sized ink component
# lying this far inside the paper interior: a balloon holds paper and
# lettering and nothing else, while whole columns go missing from the OCR
# (1ja returns one of the five columns of one balloon, so the zone around the
# box it did return reaches nowhere near the rest).  Art drawn across a
# balloon is larger than a glyph or joined to the outline, so it is not
# collected here and the zone still keeps it.
BUBBLE_STRAY_INSIDE = 0.9
# Bubbles: the fill never paints past the balloon layout.py found - its lettering mask with the
# glyph holes closed and its inset undone - except over the words themselves.  The eraser's own
# interior is the paper component the text sits on, and where a dark balloon runs into dark
# artwork that component runs on into the drawing, taking its hatching with it.
LAYOUT_BOUND = True  # False reproduces the unbounded fill exactly
BOUND_MARGIN_EM = 0.18  # the inset layout.py applies (_BUBBLE_MARGIN_EM); a test keeps them equal
BOUND_LETTER_EM = 0.3  # the words: text and furigana boxes padded this many glyphs
# Low-confidence OCR lines as erase evidence (:func:`apply`).  Ruby is the
# smallest, faintest thing on a page, so it is the first line the detector
# loses to the confidence floor - 4ja returns ``めぐみ`` at 0.26 against a floor
# of 0.5 while its kanji comes back at 0.56.  A dropped line counts as a
# block's own ink when it is not already covered by anybody's boxes, sits
# within this many ems of that block's source footprint...
EVIDENCE_REACH_EM = 1.5
# ...and its own characters are smaller than this fraction of the block's em.
# The test is on the character em (``layout.glyph_em``), never on the box: a
# Japanese column is one em wide however long it runs, so a box measure calls
# a whole un-grouped column of speech the same size as the ruby beside it and
# erases it untranslated (3jp drops ``っーか`` at 0.41, one full-size kana
# column, right next to the block that should have had it).  The ratio is
# ``layout._FURIGANA_MAX_EM_RATIO``: dropped ink that belongs to a block is
# ruby or a stray mark, and both are printed well under the text they sit by.
EVIDENCE_MAX_EM_RATIO = 0.78
# A dropped line this far covered by the blocks' own boxes is one of them
# (the same line, re-offered), not new evidence.
EVIDENCE_COVERED = 0.6
OUTLINE_REACH = 0.3  # glyphs around the source footprint checked for remaining art
OUTLINE_MIN_INK = 40  # kept ink pixels in that area that make the block "over art"
WINDOW_ALONG = 0.5  # window margin along the reading axis beyond the sweep reach (glyphs)
WINDOW_ACROSS = 1.4  # window margin across the reading axis (glyphs)
GLYPH_CAP = 1.25  # block glyph size is capped at this multiple of the page lettering scale


@dataclass
class _Ctx:
    """Analysis input in column coordinates (text runs along ``y``)."""

    gray: np.ndarray  # (h, w) uint8 window
    fg_l: float
    bg_l: float
    glyph: float
    main: List[Rect]  # member line boxes (window coords)
    furi: List[Rect]  # furigana boxes
    foreign: List[Rect]  # boxes of other blocks
    furi_side: int  # +1: furigana at larger x, -1: at smaller x
    bubble: bool = False
    hatch_dirs: Optional[np.ndarray] = None  # directions (degrees) of a hatching field around the text
    bound: Optional[np.ndarray] = None  # bubbles: bool (h, w), the balloon layout.py found


@dataclass
class _Analysis:
    erase: np.ndarray  # bool (h, w): pixels to fill
    kept: np.ndarray  # bool: ink that stays
    ring: np.ndarray  # bool: near-paper pixels around the erase set to flatten to paper
    fill: Optional[np.ndarray]  # float32 grey per pixel for the erased area (None: paper colour)


# --------------------------------------------------------------- helpers
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


def _local(rect: Rect, origin: Rect) -> Rect:
    return Rect(rect.x - origin.x, rect.y - origin.y, rect.w, rect.h)


def _transposed(rect: Rect) -> Rect:
    return Rect(rect.y, rect.x, rect.h, rect.w)


def _paint(shape: Tuple[int, int], rects: Sequence[Rect], pad: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    _paint_into(mask, rects, pad)
    return mask


def _paint_into(mask: np.ndarray, rects: Sequence[Rect], pad: int) -> None:
    h, w = mask.shape
    for r in rects:
        g = _grow(r, pad, pad, w, h)
        if g.w > 0 and g.h > 0:
            mask[g.y : g.y2, g.x : g.x2] = 1


def _bbox(mask: np.ndarray) -> Optional[Rect]:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    x1, x2 = int(cols[0]), int(cols[-1]) + 1
    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    return Rect(x1, y1, x2 - x1, y2 - y1)


def _ellipse(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.astype(bool)
    src = mask.view(np.uint8) if mask.dtype == bool else mask.astype(np.uint8)
    return cv2.dilate(src, _ellipse(radius)).view(bool)


def _ink_mask(gray: np.ndarray, fg_l: float, bg_l: float) -> np.ndarray:
    """Pixels closer to the text colour than to the paper colour (uint8 0/1)."""
    mid = (fg_l + bg_l) / 2.0
    ink = (gray < mid) if fg_l < bg_l else (gray > mid)
    return ink.view(np.uint8)


def _fraction_inside(labels: np.ndarray, region: np.ndarray, area: np.ndarray) -> np.ndarray:
    n = area.shape[0]
    inside = np.bincount(labels[region > 0], minlength=n).astype(np.float64)
    return inside / np.maximum(area, 1)


# --------------------------------------------------------------- analysis
class _Components:
    """Connected components of the ink mask with the per-component numbers
    the classifier and the sweep need (all arrays indexed by label)."""

    def __init__(self, ink: np.ndarray) -> None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        self.n = n
        self.labels = labels
        h, w = ink.shape
        self.x = stats[:, cv2.CC_STAT_LEFT].astype(np.int32)
        self.y = stats[:, cv2.CC_STAT_TOP].astype(np.int32)
        self.w = stats[:, cv2.CC_STAT_WIDTH].astype(np.int32)
        self.h = stats[:, cv2.CC_STAT_HEIGHT].astype(np.int32)
        self.area = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h
        self.cx = self.x + self.w / 2.0
        self.cy = self.y + self.h / 2.0
        self.maxdim = np.maximum(self.w, self.h)
        self.mindim = np.minimum(self.w, self.h)
        self.fill = self.area / np.maximum(self.w * self.h, 1)
        self.touch = (self.x == 0) | (self.y == 0) | (self.x2 >= w) | (self.y2 >= h)
        self.border = self.touch.copy()  # ink components joined to the window edge
        self.border[0] = False
        # Label 0 is the background: never select it.
        self.touch[0] = True
        self.area[0] = 1

    def mask_of(self, selected: np.ndarray, rect: Optional[Rect] = None) -> np.ndarray:
        """Bool mask of the pixels whose component is flagged in ``selected``
        (of the whole window, or of ``rect`` only)."""
        if rect is None:
            return selected[self.labels]
        return selected[self.labels[rect.y : rect.y2, rect.x : rect.x2]]

    def rects(self, selected: np.ndarray) -> List[Rect]:
        return [Rect(int(self.x[i]), int(self.y[i]), int(self.w[i]), int(self.h[i])) for i in np.nonzero(selected)[0]]


def _column_axes(ctx: _Ctx, sel_mask: np.ndarray) -> List[Tuple[Rect, float, float, bool]]:
    """``(box, axis_x, corridor_half_width, wide)`` for every column of text.
    A box no wider than a glyph is one column; a wider (inflated) box is
    split at the peaks of its ink profile."""
    g = ctx.glyph
    out: List[Tuple[Rect, float, float, bool]] = []
    for box in ctx.main:
        if box.w <= 0 or box.h <= 0:
            continue
        if box.w <= AXIS_BOX_MAX * g:
            half = (0.5 + SWEEP_CORRIDOR) * max(box.w, 0.6 * g)
            out.append((box, box.x + box.w / 2.0, half, False))
            continue
        sub = sel_mask[max(0, box.y) : box.y2, max(0, box.x) : box.x2]
        if sub.size == 0:
            continue
        profile = sub.sum(axis=0).astype(np.float64)
        k = max(3, int(0.25 * g) | 1)
        profile = np.convolve(profile, np.ones(k) / k, mode="same")
        peak0 = float(profile.max())
        if peak0 <= 0:
            continue
        clear = max(1, int(0.8 * g))
        while True:
            i = int(np.argmax(profile))
            if profile[i] < max(2.0, 0.35 * peak0):
                break
            out.append((box, max(0, box.x) + i + 0.5, (0.5 + SWEEP_CORRIDOR) * g, True))
            profile[max(0, i - clear) : i + clear + 1] = 0.0
    return out


def _sweep(
    ctx: _Ctx,
    comps: _Components,
    ink: np.ndarray,
    sel: np.ndarray,
    pool: np.ndarray,
    pieces: np.ndarray,
    axes: Sequence[Tuple[Rect, float, float, bool]],
) -> List[Rect]:
    """Glyph completion along each column axis: extend the column over
    glyph-bodied components centred on the axis, collect the small pieces
    inside the extended column, then look for furigana beside it.  Marks
    accepted components in ``sel``.  Returns the erase corridor of every
    column: the OCR box over the column's final extent, or - for a column of
    an inflated multi-column box - the span of its glyph bodies, so that art
    in the box margin is not text."""
    g = ctx.glyph
    h, w = ctx.gray.shape
    gap = SWEEP_GAP * g
    reach = SWEEP_MAX * g
    lo, hi = FURIGANA_ZONE
    pad = max(2, int(round(CORRIDOR_PAD * g)))
    corridors: List[Rect] = []
    for box, axis, half, wide in axes:
        x0, x1 = max(0, int(axis - half)), min(w, int(axis + half) + 1)
        if x1 <= x0:
            continue
        r0, r1 = max(0, int(box.y - 0.5 * g)), min(h, int(box.y2 + 0.5 * g))
        stripe = Rect(x0, r0, x1 - x0, r1 - r0)
        rows = np.nonzero(comps.mask_of(sel, stripe).any(axis=1))[0]
        if rows.size == 0:
            continue
        top, bottom = r0 + int(rows[0]), r0 + int(rows[-1]) + 1
        top0, bottom0 = top, bottom
        body = pool & (comps.maxdim >= SWEEP_BODY * g) & (np.abs(comps.cx - axis) <= SWEEP_ALIGN * g)
        cand = np.nonzero(body)[0]
        # Upwards.
        while cand.size:
            c = cand[~sel[cand]]
            c = c[
                (comps.cy[c] < top)
                & (comps.y2[c] >= top - gap)
                & (comps.y2[c] <= top + 0.3 * g)
                & (comps.y[c] >= top0 - reach)
            ]
            if c.size == 0:
                break
            sel[c] = True
            top = min(top, int(comps.y[c].min()))
        # Downwards.
        while cand.size:
            c = cand[~sel[cand]]
            c = c[
                (comps.cy[c] > bottom)
                & (comps.y[c] <= bottom + gap)
                & (comps.y[c] >= bottom - 0.3 * g)
                & (comps.y2[c] <= bottom0 + reach)
            ]
            if c.size == 0:
                break
            sel[c] = True
            bottom = max(bottom, int(comps.y2[c].max()))
        # Small pieces inside the (extended) column: dots, marks, radicals.
        inside = pieces & ~sel & (np.abs(comps.cx - axis) <= half)
        inside &= (comps.y2 >= top - 0.3 * g) & (comps.y <= bottom + 0.3 * g)
        sel |= inside
        # Furigana strip beside the column, over its extent.
        stripe = Rect(x0, top, x1 - x0, bottom - top)
        cols = np.nonzero(comps.mask_of(sel, stripe).any(axis=0))[0]
        if cols.size:
            if ctx.furi_side > 0:
                edge = x0 + int(cols[-1]) + 1
                zx0, zx1 = edge + lo * g, edge + hi * g
            else:
                edge = x0 + int(cols[0])
                zx0, zx1 = edge - hi * g, edge - lo * g
            f = pieces & ~sel & (comps.maxdim <= FURIGANA_MAX * g)
            f &= (comps.cx >= zx0) & (comps.cx <= zx1)
            f &= (comps.cy >= top - 0.2 * g) & (comps.cy <= bottom + 0.2 * g)
            if f.any():
                # Furigana sit on paper: a strip full of other ink is art.
                sx0, sx1 = max(0, int(zx0)), min(w, int(zx1) + 1)
                strip = ink[top:bottom, sx0:sx1]
                other = int(strip.sum()) - int(comps.area[f].sum())
                if other <= FURIGANA_NOISE * max(1, strip.size):
                    sel |= f
        if wide:
            bodies = sel & (comps.maxdim >= SWEEP_BODY * g) & (np.abs(comps.cx - axis) <= half)
            bodies &= (comps.y2 > top) & (comps.y < bottom)
            if not bodies.any():
                continue
            cx0, cx1 = int(comps.x[bodies].min()), int(comps.x2[bodies].max())
            corridors.append(Rect(cx0 - pad, top - pad, cx1 - cx0 + 2 * pad, bottom - top + 2 * pad).clamp(w, h))
        else:
            y0, y1 = min(box.y, top) - pad, max(box.y2, bottom) + pad
            corridors.append(Rect(box.x - pad, y0, box.w + 2 * pad, y1 - y0).clamp(w, h))
    return corridors


def _follow_lines(
    ink: np.ndarray,
    zone: np.ndarray,
    lines: np.ndarray,
    t_lo: np.ndarray,
    t_hi: np.ndarray,
    min_outside: np.ndarray,
    keep: np.ndarray,
) -> bool:
    """Follow straight art lines through the text zone and mark, in ``keep``,
    their pixels inside the zone (all lines at once).

    ``lines`` are ``(x1, y1, x2, y2)`` rows; each is sampled over the
    parameter range ``[t_lo, t_hi]`` (covering the detected segment
    ``[0, length]`` and its extrapolation towards the zone).  The line is
    snapped onto the nearest ink within ``LINE_SNAP`` px (Hough angles drift
    over a long extrapolation); where the run of ink across it is as thin as
    the line is outside the zone the whole run is kept, where a glyph stroke
    crosses it only the line's own width is kept, and the kept part must be
    joined (gaps <= ``LINE_GAP``) to the line's support outside the zone.  A
    line needs ``min_outside`` ink samples outside the zone, ``LINE_DENSITY``
    of its detected part inked, and must enter the zone.  Returns whether any
    pixel was kept."""
    h, w = ink.shape
    n = lines.shape[0]
    x1, y1 = lines[:, 0], lines[:, 1]
    length = np.maximum(np.hypot(lines[:, 2] - x1, lines[:, 3] - y1), 1e-6)
    dx, dy = (lines[:, 2] - x1) / length, (lines[:, 3] - y1) / length
    nx, ny = -dy, dx
    t_lo = np.floor(t_lo)
    n_t = int(np.ceil(t_hi - t_lo).max()) + 1
    t = t_lo[:, None] + np.arange(n_t)[None, :]  # (lines, points)
    valid_t = t <= t_hi[:, None] + 1.0
    k = np.arange(-LINE_REACH, LINE_REACH + 1)
    n_k = k.size
    px = x1[:, None] + dx[:, None] * t
    py = y1[:, None] + dy[:, None] * t
    xs = np.rint(px[:, :, None] + k[None, None, :] * nx[:, None, None]).astype(np.intp)
    ys = np.rint(py[:, :, None] + k[None, None, :] * ny[:, None, None]).astype(np.intp)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h) & valid_t[:, :, None]
    xs, ys = np.clip(xs, 0, w - 1), np.clip(ys, 0, h - 1)
    on = (ink[ys, xs] > 0) & valid  # (lines, points, offsets)
    # Nearest ink offset to the centre within the snap distance.
    hit = np.full((n, n_t), -1, dtype=np.intp)
    for d in range(LINE_SNAP, -1, -1):
        for j in (LINE_REACH + d, LINE_REACH - d):
            hit = np.where(on[:, :, j], j, hit)
    has = hit >= 0
    if not has.any():
        return False
    # Run of ink across the line through the hit offset.
    left = np.zeros(on.shape, dtype=np.int32)
    right = np.zeros(on.shape, dtype=np.int32)
    left[:, :, 0] = on[:, :, 0]
    for j in range(1, n_k):
        left[:, :, j] = (left[:, :, j - 1] + 1) * on[:, :, j]
    right[:, :, -1] = on[:, :, -1]
    for j in range(n_k - 2, -1, -1):
        right[:, :, j] = (right[:, :, j + 1] + 1) * on[:, :, j]
    hj = np.where(has, hit, LINE_REACH)
    lo = hj - (np.take_along_axis(left, hj[:, :, None], 2)[:, :, 0] - 1)
    hi = hj + (np.take_along_axis(right, hj[:, :, None], 2)[:, :, 0] - 1)
    run = np.where(has, hi - lo + 1, 0)
    cx, cy = xs[:, :, LINE_REACH], ys[:, :, LINE_REACH]
    in_zone = (zone[cy, cx] > 0) & valid[:, :, LINE_REACH]
    own = (t >= 0) & (t <= length[:, None]) & valid_t  # the detected part of the line
    support = own & ~in_zone & has
    n_support = support.sum(axis=1)
    ok = (n_support >= min_outside) & in_zone.any(axis=1)
    ok &= n_support >= LINE_DENSITY * np.maximum(1, (own & ~in_zone).sum(axis=1))
    if not ok.any():
        return False
    # Line thickness outside the zone: median run over the support samples.
    ranked = np.sort(np.where(support, run, 1 << 20), axis=1)
    t_line = ranked[np.arange(n), np.maximum(n_support - 1, 0) // 2].astype(np.float64)
    t_line = np.where(n_support > 0, t_line, 1.0)
    t_max = np.maximum(3.0, t_line + LINE_SLACK)
    thin = has & (run <= t_max[:, None])
    # Segments of consecutive ink along the line (gaps <= LINE_GAP bridged);
    # only segments joined to thin support outside the zone continue art.
    span_k = 2 * ((LINE_GAP + 1) // 2) + 1
    closed = cv2.morphologyEx(has.view(np.uint8), cv2.MORPH_CLOSE, np.ones((1, span_k), np.uint8)).astype(bool)
    closed &= valid_t
    starts = closed.copy()
    starts[:, 1:] &= ~closed[:, :-1]
    seg = np.cumsum(starts, axis=1) * closed  # 0 = no segment
    ids = seg + (np.arange(n) * (n_t + 1))[:, None]
    marked = np.bincount(ids[support & thin], minlength=n * (n_t + 1) + 1) > 0
    good = marked[ids] & (seg > 0)
    take = good & in_zone & has & ok[:, None]
    if not take.any():
        return False
    # Line centre offset: run centre where thin, otherwise carried over from
    # the nearest thin samples before and after (a glyph stroke crossing).
    centre = (lo + hi) / 2.0
    pos = np.arange(n_t)[None, :]
    fwd = np.maximum.accumulate(np.where(thin, pos, -1), axis=1)
    bwd = np.minimum.accumulate(np.where(thin, pos, n_t)[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(n)[:, None]
    c_f = np.where(fwd >= 0, centre[rows, np.clip(fwd, 0, n_t - 1)], np.nan)
    c_b = np.where(bwd < n_t, centre[rows, np.clip(bwd, 0, n_t - 1)], np.nan)
    nan_f, nan_b = np.isnan(c_f), np.isnan(c_b)
    off = np.where(nan_f, c_b, np.where(nan_b, c_f, (c_f + c_b) / 2.0))
    off = np.where(nan_f & nan_b, float(LINE_REACH), off)
    half = np.maximum(1.0, np.ceil(t_line / 2.0))
    kk = np.arange(n_k)[None, None, :]
    span_thin = (kk >= lo[:, :, None]) & (kk <= hi[:, :, None])
    span_thick = on & (np.abs(kk - off[:, :, None]) <= half[:, None, None])
    span = np.where(thin[:, :, None], span_thin, span_thick) & take[:, :, None]
    keep[ys[span], xs[span]] = True
    return True


def detector_glyph(glyph: float) -> float:
    """The glyph size the art-line *detector* scales its thresholds by.

    What a run of ink must look like to BE a drawn line - how long it is
    (``HOUGH_MIN_LEN``) and how much support it has outside the text
    (``HOUGH_OUTSIDE``) - is a fact about the **artwork**, and artwork does not
    get bigger because the lettering beside it does.  So those thresholds stop
    scaling once the block is lettered larger than a normal page
    (:data:`HOUGH_GLYPH_MAX`).

    Everything about the *text* keeps the block's own glyph: how far a kept
    line is extrapolated through the text zone (``HOUGH_EXTEND``), the sweep,
    the window, the corridor.  Only the two detector thresholds are capped, and
    capping them can only keep MORE art, never erase more.
    """
    return min(float(glyph), HOUGH_GLYPH_MAX)


def _art_lines(ctx: _Ctx, art: np.ndarray, ink: np.ndarray, zone: np.ndarray) -> Optional[np.ndarray]:
    """Pixels (bool mask) of straight art lines that enter the text zone:
    detected on ``art`` (uint8, glyphs removed) with real support outside the
    zone, followed through it on the full ``ink``."""
    g = ctx.glyph
    gd = detector_glyph(g)
    ext = HOUGH_EXTEND * g
    zb = _bbox(zone)
    if zb is None:
        return None
    # Lines further than their extrapolation from the zone cannot reach it:
    # detect only inside the zone's bounding box grown by the extrapolation.
    h, w = art.shape
    roi = _grow(zb, ext, ext, w, h)
    lines = cv2.HoughLinesP(
        np.ascontiguousarray(art[roi.y : roi.y2, roi.x : roi.x2]) * 255,
        1,
        np.pi / 180.0,
        threshold=max(8, int(0.4 * gd)),
        minLineLength=max(8, int(HOUGH_MIN_LEN * gd)),
        maxLineGap=3,
    )
    if lines is None:
        return None
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 4) + np.array([roi.x, roi.y, roi.x, roi.y], dtype=np.float64)
    # Only lines whose extension crosses the zone's bounding box are followed.
    lengths = np.maximum(np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]), 1e-6)
    dx = (lines[:, 2] - lines[:, 0]) / lengths
    dy = (lines[:, 3] - lines[:, 1]) / lengths
    t0, t1 = np.full(len(lines), -ext), lengths + ext
    for p, d, lo_b, hi_b in ((lines[:, 0], dx, zb.x - 1, zb.x2), (lines[:, 1], dy, zb.y - 1, zb.y2)):
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo_b - p) / d, (hi_b - p) / d
        par = np.abs(d) < 1e-9
        inside = (p >= lo_b) & (p <= hi_b)
        t_lo = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(ta, tb))
        t_hi = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(ta, tb))
        t0, t1 = np.maximum(t0, t_lo), np.minimum(t1, t_hi)
    cand = np.nonzero(t0 <= t1)[0]
    if cand.size == 0:
        return None
    cand = cand[np.argsort(-lengths[cand])[:HOUGH_MAX_LINES]]
    # Strokes of a hatching field around the text need less support outside
    # the text: they are mostly under it.
    # How much support a line needs OUTSIDE the text zone is a fact about the artwork too:
    # on a page lettered at 241 px this demanded 96 px of clear run before a line counted as
    # art at all, which is most of a panel.  Capped with the detector's glyph for the same
    # reason - and capping it can only keep MORE art, never erase more.
    min_outside = np.full(cand.size, HOUGH_OUTSIDE * gd)
    if ctx.hatch_dirs is not None and ctx.hatch_dirs.size:
        ang = np.degrees(np.arctan2(dy[cand], dx[cand])) % 180.0
        diff = np.abs(ang[:, None] - ctx.hatch_dirs[None, :])
        diff = np.minimum(diff, 180.0 - diff)
        min_outside[(diff <= HATCH_ANGLE).any(axis=1)] = HOUGH_OUTSIDE_HATCH * gd
    keep = np.zeros(zone.shape, dtype=bool)
    t_lo = np.minimum(0.0, t0[cand] - 2.0)
    t_hi = np.maximum(lengths[cand], t1[cand] + 2.0)
    if not _follow_lines(ink, zone, lines[cand], t_lo, t_hi, min_outside, keep):
        return None
    return keep


def _keep_art(ctx: _Ctx, sel_mask: np.ndarray, amb_mask: np.ndarray, ink: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Ink inside the text zone that is art and must survive: solid bodies
    (thicker than any glyph stroke of this block) and straight lines
    continuing from outside the zone.  ``sel_mask`` / ``amb_mask`` are the
    pixel masks of the glyph / art components.  Returns the keep mask."""
    h, w = ink.shape
    keep = np.zeros((h, w), dtype=bool)
    amb_zone = int((amb_mask & (zone > 0)).sum())
    if amb_zone == 0:
        return keep
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    stroke = float(np.percentile(dt[sel_mask], 99)) if sel_mask.any() else 0.0
    solid_dt = max(SOLID_DT, stroke + 1.0)
    solid = (dt >= solid_dt) & amb_mask
    if solid.any():
        keep |= _dilate(solid, ERASE_DILATE) & amb_mask
    if amb_zone < ART_MIN_PX:
        return keep  # nothing worth following: the art only brushes the text
    art = ink.copy()
    art[sel_mask] = 0
    lines = _art_lines(ctx, art, ink, zone)
    if lines is not None:
        keep |= lines
    return keep


def _hatching(ctx: _Ctx, comps: _Components, frac_in: np.ndarray, own: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(hatch, directions)``: bool per component flagging thin, straight,
    diagonal strokes inside the text boxes that run parallel to a *hatching
    field* - at least ``HATCH_FIELD`` such strokes of the same direction just
    outside the boxes - and the field's directions (degrees, mod 180).
    Glyph strokes are mostly axis-aligned, so these strokes are art (hatched
    shading under the text)."""
    g = ctx.glyph
    h, w = own.shape
    out = np.zeros(comps.n, dtype=bool)
    none = np.zeros(0, dtype=np.float64)
    cand = (comps.maxdim >= HATCH_MIN_LEN * g) & (comps.maxdim <= GLYPH_MAX * g) & (comps.fill <= HATCH_FILL)
    cand &= comps.mindim >= 3  # a diagonal stroke spans both axes
    cand[0] = False
    if cand.sum() < HATCH_FIELD + 1:
        return out, none
    near = _paint((h, w), ctx.main + ctx.furi, int(HATCH_REACH * g))
    cy = np.clip(comps.cy.astype(int), 0, h - 1)
    cx = np.clip(comps.cx.astype(int), 0, w - 1)
    cand &= (frac_in > 0) | (near[cy, cx] > 0)
    if cand.sum() < HATCH_FIELD + 1:
        return out, none
    # Second moments of the candidate components -> orientation and elongation.
    idx = np.nonzero(cand)[0]
    remap = np.zeros(comps.n, dtype=np.int64)
    remap[idx] = np.arange(1, idx.size + 1)
    bx, by = int(comps.x[idx].min()), int(comps.y[idx].min())
    sub = remap[comps.labels[by : int(comps.y2[idx].max()), bx : int(comps.x2[idx].max())]]
    ys, xs = np.nonzero(sub)
    lab = sub[ys, xs]
    ys, xs = ys + by, xs + bx
    n = np.bincount(lab, minlength=idx.size + 1)[1:].astype(np.float64)
    n = np.maximum(n, 1)
    x, y = xs.astype(np.float64), ys.astype(np.float64)
    mx = np.bincount(lab, x, idx.size + 1)[1:] / n
    my = np.bincount(lab, y, idx.size + 1)[1:] / n
    cxx = np.bincount(lab, x * x, idx.size + 1)[1:] / n - mx * mx + 1 / 12.0
    cyy = np.bincount(lab, y * y, idx.size + 1)[1:] / n - my * my + 1 / 12.0
    cxy = np.bincount(lab, x * y, idx.size + 1)[1:] / n - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    elong = np.sqrt((tr / 2.0 + disc) / np.maximum(tr / 2.0 - disc, 1e-6))
    angle = np.degrees(0.5 * np.arctan2(2.0 * cxy, cxx - cyy)) % 180.0
    off_axis = np.minimum(angle % 90.0, 90.0 - angle % 90.0) >= HATCH_AXIS
    stroke = (elong >= HATCH_ELONG) & off_axis
    outside = stroke & (frac_in[idx] == 0)
    if outside.sum() < HATCH_FIELD:
        return out, none
    # Hatching directions: angles (mod 180) around which enough outside strokes cluster.
    ang_out = angle[outside]
    diff = np.abs(angle[:, None] - ang_out[None, :])
    diff = np.minimum(diff, 180.0 - diff)
    support = (diff <= HATCH_ANGLE).sum(axis=1) - outside.astype(int)
    inside = stroke & (frac_in[idx] > 0) & (support >= HATCH_FIELD)
    out[idx[inside]] = True
    return out, ang_out[support[outside] >= HATCH_FIELD - 1]


def _classify(ctx: _Ctx, comps: _Components) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(sel, amb, pool, pieces, own)``: glyph components, art components
    crossing the boxes, completion candidates (glyph bodies), small-piece
    candidates, and the painted own boxes."""
    h, w = ctx.gray.shape
    g = ctx.glyph
    own = _paint((h, w), ctx.main + ctx.furi, PAD)
    frac_in = _fraction_inside(comps.labels, own, comps.area)
    if ctx.foreign:
        frac_f = _fraction_inside(comps.labels, _paint((h, w), ctx.foreign, PAD), comps.area)
    else:
        frac_f = np.zeros(comps.n)
    foreign_c = (frac_f > 0.5) & (frac_f > frac_in)
    glyph_sized = comps.maxdim <= GLYPH_MAX * g
    if ctx.bubble:
        # Inside a bubble everything that is not joined to the outline is text.
        sel = ~foreign_c & ~comps.touch & (frac_in > 0)
        amb = np.zeros(comps.n, dtype=bool)
    else:
        sel = ~foreign_c & ((frac_in >= FULL_INSIDE) | ((frac_in > 0) & glyph_sized & ~comps.touch))
        hatch, ctx.hatch_dirs = _hatching(ctx, comps, frac_in, own)
        sel &= ~hatch
        amb = ~foreign_c & (frac_in > 0) & ~sel & ~hatch
    sel[0] = False
    amb[0] = False
    pieces = ~foreign_c & (frac_in == 0) & ~comps.touch & glyph_sized
    pieces &= (comps.maxdim >= PIECE_MIN * g) & (comps.mindim >= 2) & (comps.fill >= MIN_FILL)
    if pieces.any():
        # Fragments lying against art (hatching, borders) are art, not stray glyphs.
        big = ~glyph_sized | comps.touch
        big[0] = False
        if big.any():
            near = _dilate(comps.mask_of(big), NEAR_ART).astype(np.uint8)
            pieces &= _fraction_inside(comps.labels, near, comps.area) < 0.3
    pool = pieces & (comps.maxdim >= GLYPH_MIN * g)
    return sel, amb, pool, pieces, own


def _complete(
    ctx: _Ctx, comps: _Components, ink: np.ndarray, sel: np.ndarray, pool: np.ndarray, pieces: np.ndarray, own: np.ndarray
) -> np.ndarray:
    """Run glyph completion, then return the *corridor* (uint8): the glyph
    columns (see :func:`_sweep`), the furigana boxes and the boxes of every
    component the completion added, all grown by ``CORRIDOR_PAD`` glyphs.
    Only ink inside the corridor is erased, so art in the margin of an
    inflated OCR box survives; the corridor is also the text zone the art
    lines must have their support outside of."""
    before = sel.copy()
    corridors = _sweep(ctx, comps, ink, sel, pool, pieces, _column_axes(ctx, comps.mask_of(sel)))
    pad = max(2, int(round(CORRIDOR_PAD * ctx.glyph)))
    corridor = _paint(own.shape, corridors + ctx.furi, pad)
    _paint_into(corridor, comps.rects(sel & ~before), pad)
    if not corridors:
        corridor |= own
    return corridor


def _background(gray: np.ndarray, exclude: np.ndarray, radius: int, bg_l: float) -> np.ndarray:
    """Local mean grey of the background (pixels not in ``exclude``) in a
    square of ``radius``; paper colour where nothing is left to average."""
    valid = (~exclude).astype(np.float32)
    k = (2 * radius + 1, 2 * radius + 1)
    num = cv2.boxFilter(gray.astype(np.float32) * valid, -1, k, normalize=False)
    den = cv2.boxFilter(valid, -1, k, normalize=False)
    enough = den >= TONE_MIN_SAMPLES * k[0] * k[1]
    out = np.where(enough, num / np.maximum(den, 1e-6), bg_l).astype(np.float32)
    out[np.abs(out - bg_l) <= TONE_TOL] = bg_l
    return out


def _analyse_open(ctx: _Ctx) -> _Analysis:
    gray = ctx.gray
    h, w = gray.shape
    ink = _ink_mask(gray, ctx.fg_l, ctx.bg_l)
    empty = np.zeros((h, w), dtype=bool)
    if not ink.any():
        return _Analysis(empty, empty.copy(), empty.copy(), None)
    comps = _Components(ink)
    sel, amb, pool, pieces, own = _classify(ctx, comps)
    corridor = _complete(ctx, comps, ink, sel, pool, pieces, own)
    sel_mask, amb_mask = comps.mask_of(sel), comps.mask_of(amb)
    keep = _keep_art(ctx, sel_mask, amb_mask, ink, corridor)
    erase = (sel_mask | amb_mask) & (corridor > 0) & ~keep
    kept = ink.astype(bool) & ~erase
    erase_d = _dilate(erase, ERASE_DILATE) & ~_dilate(kept, 1)
    erase_d |= erase
    ring = _dilate(erase_d, RING) & ~erase_d & ~kept
    ring &= np.abs(gray.astype(np.int16) - ctx.bg_l) <= RING_TOL
    # Local background for the fill: paper on paper, the tone's mean grey on
    # a screentone.  Ringing is only flattened on paper.
    radius = max(2, int(TONE_RADIUS * ctx.glyph))
    box = _bbox(erase_d)
    fill: Optional[np.ndarray] = None
    if box is not None:
        sub = _grow(box, radius + RING, radius + RING, w, h)
        sl = (slice(sub.y, sub.y2), slice(sub.x, sub.x2))
        # Ink and its anti-aliased fringe do not count as background, so the
        # grey edges of hatching lines cannot tint the fill; the dots of a
        # screentone (tiny components) do.
        dots = comps.mask_of(comps.maxdim <= TONE_DOT, sub)
        fringe = _dilate((ink[sl] > 0) & ~dots, 1)
        local = _background(gray[sl], erase_d[sl] | ring[sl] | fringe, radius, ctx.bg_l)
        if not np.all(local[erase_d[sl]] == ctx.bg_l):
            fill = np.full((h, w), ctx.bg_l, dtype=np.float32)
            fill[sl] = local
            ring &= fill == ctx.bg_l
    return _Analysis(erase_d, kept, ring, fill)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """``mask`` with every enclosed hole filled (components of the complement
    that do not touch the window border)."""
    inv = (~mask).astype(np.uint8)
    n, labels = cv2.connectedComponents(inv, connectivity=4)
    open_to_border = np.zeros(n, dtype=bool)
    open_to_border[0] = True
    for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        open_to_border[np.unique(edge)] = True
    return mask | ~open_to_border[labels]


def _analyse_bubble(ctx: _Ctx) -> _Analysis:
    """Redraw the bubble: its paper interior (the paper component holding the
    text, as ``layout.py`` finds it, with every enclosed glyph, furigana, stray
    mark or hole filled) is painted with the paper colour, clipped to the
    outline.  The outline - the large ink components bordering the interior
    from outside, tails included - and its ``OUTLINE_FRINGE`` px anti-aliased
    edge are copied from the source untouched, so it keeps its own thickness.
    Nothing outside the outline is touched.

    Inside it the fill reaches ``BUBBLE_ZONE_EM`` glyphs past the text boxes,
    and then as far as the lettering goes: any ink component of glyph size
    (``GLYPH_MAX``) lying ``BUBBLE_STRAY_INSIDE`` inside the interior, clear of
    the window edge, extends the zone to itself, because a balloon holds paper
    and lettering and nothing else and whole columns go missing from the OCR
    (1ja returns one of the five columns of one balloon).  Art drawn across the
    balloon's edge - 4ja's chibi, whose head reaches in - is larger than a
    glyph or joined to the outline, fails that escape, stays outside the zone
    and survives."""
    gray = ctx.gray
    h, w = gray.shape
    ink = _ink_mask(gray, ctx.fg_l, ctx.bg_l)
    comps = _Components(ink)
    own = _paint((h, w), ctx.main + ctx.furi, PAD)
    zone_pad = max(PAD, int(round(BUBBLE_ZONE_EM * max(ctx.glyph, 1.0))))
    zone = _paint((h, w), ctx.main + ctx.furi, zone_pad)
    # The outline candidates, and the ones the paper map is carved with.  A
    # component mostly inside the block's own text boxes is that block's
    # lettering, not the balloon's edge: on dark paper the ink mask selects
    # light pixels, so a column of glyphs joins into one component the length of
    # the column and clears OUTLINE_MIN on its own.  Carving it would punch the
    # lettering back out of the paper map and leave the interior a handful of
    # specks between the strokes.  It is dropped from the CARVE only: `big`
    # still carries it into the outline test below, so a box that overlaps a
    # real edge cannot cost that edge its protection.
    big = comps.maxdim >= OUTLINE_MIN * min(h, w)
    big[0] = False
    lettering = _fraction_inside(comps.labels, own, comps.area) >= OUTLINE_IN_TEXT
    big_mask = comps.mask_of(big & ~lettering)

    # Paper component the text sits on: text boxes painted over (glyphs count as
    # paper), but never the outline, even where a box overlaps it.
    paper = (np.abs(gray.astype(np.int16) - ctx.bg_l) <= PAPER_TOL).astype(np.uint8)
    paper[own > 0] = 1
    paper[big_mask] = 0
    _, plabels = cv2.connectedComponents(paper, connectivity=4)
    label = 0
    for box in ctx.main + ctx.furi:
        cy = min(h - 1, max(0, box.y + box.h // 2))
        cx = min(w - 1, max(0, box.x + box.w // 2))
        label = int(plabels[cy, cx])
        if label:
            break
    interior = _fill_holes(plabels == label) if label else own.astype(bool)

    # The outline: big ink components that border the interior mostly from
    # outside (a big glyph enclosed by the interior lies inside it entirely).
    adjacent = np.zeros(comps.n, dtype=bool)
    adjacent[np.unique(comps.labels[_dilate(interior, 1) & (ink > 0)])] = True
    inside = _fraction_inside(comps.labels, interior.view(np.uint8), comps.area)
    outline = comps.mask_of(adjacent & big & (inside < 0.5))

    # The columns the detector never reported: glyph-sized ink lying on the
    # interior and clear of the outline is lettering wherever it is, so the
    # zone follows it out to the far side of the balloon.
    stray = ~big & ~comps.touch & (comps.maxdim <= GLYPH_MAX * max(ctx.glyph, 1.0))
    stray &= inside >= BUBBLE_STRAY_INSIDE
    stray[0] = False
    if stray.any():
        _paint_into(zone, comps.rects(stray), PAD)
    erase = interior & zone.astype(bool) & ~_dilate(outline, OUTLINE_FRINGE)
    if ctx.bound is not None:
        words = _paint((h, w), ctx.main + ctx.furi, max(PAD, int(round(BOUND_LETTER_EM * max(ctx.glyph, 1.0)))))
        erase &= ctx.bound | (words > 0)
    kept = ink.astype(bool) & ~erase
    empty = np.zeros((h, w), dtype=bool)
    return _Analysis(erase, kept, empty, None)


# --------------------------------------------------------------- per block
def _layout_bound(block: TextBlock, win: Rect) -> Optional[np.ndarray]:
    """The balloon ``layout.py`` found, in the coordinates of ``win``: its
    lettering mask (``style.layout_mask`` over ``style.layout_box``) with the
    glyph holes closed and the inset undone.  None when there is none."""
    style = block.style
    if style.layout_mask is None or style.layout_box is None:
        return None
    mask = np.asarray(style.layout_mask).astype(bool)
    lb = style.layout_box
    x0, y0 = max(lb.x, win.x), max(lb.y, win.y)
    x1, y1 = min(lb.x + mask.shape[1], win.x2), min(lb.y + mask.shape[0], win.y2)
    if x1 <= x0 or y1 <= y0:
        return None
    region = np.zeros((win.h, win.w), dtype=bool)
    region[y0 - win.y : y1 - win.y, x0 - win.x : x1 - win.x] = mask[y0 - lb.y : y1 - lb.y, x0 - lb.x : x1 - lb.x]
    em = float(block.em_px or style.text_height_px or 1.0)
    return _dilate(_fill_holes(region), max(3, int(round(BOUND_MARGIN_EM * em))) + 2)


def _boxes(block: TextBlock) -> Tuple[List[Rect], List[Rect]]:
    return [m.bbox for m in block.members], [f.bbox for f in block.furigana]


def _no_mask(rect: Rect) -> np.ndarray:
    """All-False glyph mask for a patch that erased nothing."""
    return np.zeros((max(0, rect.h), max(0, rect.w)), dtype=bool)


def _gap(a: Rect, b: Rect) -> float:
    """Distance between two rectangles; 0 when they touch or overlap."""
    dx = max(0, b.x - a.x2, a.x - b.x2)
    dy = max(0, b.y - a.y2, a.y - b.y2)
    return float(np.hypot(dx, dy))


def _covered(box: Rect, boxes: Sequence[Rect]) -> float:
    """Largest fraction of ``box`` that any single rectangle of ``boxes``
    covers."""
    area = float(max(1, box.w * box.h))
    best = 0.0
    for other in boxes:
        iw = min(box.x2, other.x2) - max(box.x, other.x)
        ih = min(box.y2, other.y2) - max(box.y, other.y)
        if iw > 0 and ih > 0:
            best = max(best, iw * ih / area)
    return best


def _evidence(
    blocks: Sequence[TextBlock], all_segments: Sequence[Segment], width: int, height: int
) -> List[List[Rect]]:
    """Per block, the boxes of ``all_segments`` that no block owns: the lines
    the detector did see and the confidence floor dropped, ruby first of all.
    They are the block's ink and nothing else - they never reach its text -
    so the eraser treats them exactly like its furigana boxes.  Each goes to
    the one block it is nearest, and only when its characters are small enough
    and it is close enough to be that block's own ruby or strays
    (``EVIDENCE_MAX_EM_RATIO`` / ``EVIDENCE_REACH_EM``).  A line printed at
    the block's own size is a line of speech the grouping did not get, not
    ink to paint over, and is left alone."""
    out: List[List[Rect]] = [[] for _ in blocks]
    if not all_segments or not blocks:
        return out
    owned: List[Rect] = []
    sources: List[Rect] = []
    for b in blocks:
        main, furi = _boxes(b)
        own = [r.clamp(width, height) for r in main + furi if r.w > 0 and r.h > 0]
        owned.extend(own)
        sources.append(_union(own) if own else b.segment.bbox.clamp(width, height))
    ems = [max(1.0, float(b.em_px) or float(b.style.text_height_px)) for b in blocks]
    for seg in all_segments:
        box = seg.bbox.clamp(width, height)
        if box.w <= 0 or box.h <= 0 or _covered(box, owned) >= EVIDENCE_COVERED:
            continue
        em_seg = glyph_em(seg)
        best: Optional[Tuple[float, int]] = None
        for i, em in enumerate(ems):
            if em_seg >= EVIDENCE_MAX_EM_RATIO * em:
                continue
            dist = _gap(box, sources[i])
            if dist <= EVIDENCE_REACH_EM * em and (best is None or dist < best[0]):
                best = (dist, i)
        if best is not None:
            out[best[1]].append(box)
    return out


def erase_block(
    img_bgr: np.ndarray,
    block: TextBlock,
    gray: Optional[np.ndarray] = None,
    foreign: Sequence[Rect] = (),
    glyph: Optional[float] = None,
    extra: Sequence[Rect] = (),
) -> Tuple[np.ndarray, Rect, bool]:
    """Erase the source glyphs of ``block``: ``(patch, rect, outline)``.

    See :func:`erase_block_masked`, of which this is the three-value form kept
    for every existing caller.
    """
    patch, rect, outline, _mask = erase_block_masked(img_bgr, block, gray, foreign, glyph, extra)
    return patch, rect, outline


def erase_block_masked(
    img_bgr: np.ndarray,
    block: TextBlock,
    gray: Optional[np.ndarray] = None,
    foreign: Sequence[Rect] = (),
    glyph: Optional[float] = None,
    extra: Sequence[Rect] = (),
) -> Tuple[np.ndarray, Rect, bool, np.ndarray]:
    """Erase the source glyphs of ``block``.

    Returns ``(patch, rect, outline, mask)``: ``patch`` is a BGR copy of
    ``rect`` (frame coordinates) with the glyphs removed, ``outline`` tells
    whether ink or screentone remains under/near the text footprint (the
    renderer then draws the translation with a halo), and ``mask`` is the bool
    array of ``rect``'s shape flagging the pixels that were painted over -
    the glyph classification the quality renderer regenerates
    (``render/quality.py``).  ``gray`` may be the page's grey image (saves a
    conversion); ``foreign`` are the OCR boxes of the other blocks, which are
    never erased; ``glyph`` overrides the block's glyph size
    (``style.text_height_px``); ``extra`` are further boxes of this block's own
    ink - the low-confidence lines :func:`_evidence` hands it - erased like its
    furigana and, like it, never translated.
    """
    height, width = img_bgr.shape[:2]
    if gray is None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    style = block.style
    main, furi = _boxes(block)
    main = [r.clamp(width, height) for r in main if r.w > 0 and r.h > 0]
    furi = [r.clamp(width, height) for r in furi if r.w > 0 and r.h > 0]
    furi += [r.clamp(width, height) for r in extra if r.w > 0 and r.h > 0]
    source = _union(main + furi) if (main or furi) else block.segment.bbox.clamp(width, height)
    if not main:
        main = [source]
    g = max(4.0, float(glyph if glyph is not None else style.text_height_px))
    vertical = bool(style.vertical)
    fg_l, bg_l = _luma(style.fg), _luma(style.bg)
    if abs(fg_l - bg_l) < 20:
        fg_l = 0.0 if bg_l >= 128 else 255.0

    bubble = block.bubble is not None and style.in_bubble
    if bubble:
        assert block.bubble is not None
        win = _grow(block.bubble.union(source), 3, 3, width, height)
    else:
        along, across = (SWEEP_MAX + WINDOW_ALONG) * g, WINDOW_ACROSS * g
        win = _grow(source, across if vertical else along, along if vertical else across, width, height)
    if win.w <= 0 or win.h <= 0:
        rect = _grow(source, PAD + 1, PAD + 1, width, height)
        return img_bgr[rect.y : rect.y2, rect.x : rect.x2].copy(), rect, False, _no_mask(rect)

    gray_w = gray[win.y : win.y2, win.x : win.x2]

    def loc(rs: Sequence[Rect]) -> List[Rect]:
        out = [_local(r, win) for r in rs]
        return out if vertical else [_transposed(r) for r in out]

    ctx = _Ctx(
        np.ascontiguousarray(gray_w if vertical else gray_w.T),
        fg_l,
        bg_l,
        g,
        loc(main),
        loc(furi),
        loc(foreign),
        +1 if vertical else -1,
        bubble,
    )
    if bubble and LAYOUT_BOUND:
        bound = _layout_bound(block, win)
        if bound is not None:
            ctx.bound = np.ascontiguousarray(bound if vertical else bound.T)
    res = _analyse_bubble(ctx) if bubble else _analyse_open(ctx)
    erase, kept, ring, fill = res.erase, res.kept, res.ring, res.fill
    if not vertical:
        erase, kept, ring = erase.T, kept.T, ring.T
        fill = None if fill is None else fill.T

    # Halo needed?  Kept ink or a screentone under/right next to the text.
    outline = False
    if not bubble:
        foot = _local(_grow(source, OUTLINE_REACH * g, OUTLINE_REACH * g, width, height), win).clamp(win.w, win.h)
        outline = int(kept[foot.y : foot.y2, foot.x : foot.x2].sum()) >= OUTLINE_MIN_INK
        if not outline and fill is not None:
            outline = bool((fill[foot.y : foot.y2, foot.x : foot.x2] != bg_l).any())

    box = _bbox(erase)
    if box is None:
        rect = _grow(source, PAD + 1, PAD + 1, width, height)
        return img_bgr[rect.y : rect.y2, rect.x : rect.x2].copy(), rect, outline, _no_mask(rect)

    local_rect = _grow(box, RECT_PAD + (0 if bubble else RING), RECT_PAD + (0 if bubble else RING), win.w, win.h)
    sl = (slice(local_rect.y, local_rect.y2), slice(local_rect.x, local_rect.x2))
    rect = Rect(local_rect.x + win.x, local_rect.y + win.y, local_rect.w, local_rect.h)
    patch = img_bgr[rect.y : rect.y2, rect.x : rect.x2].copy()
    bg_bgr = np.array(style.bg[::-1], dtype=np.uint8)
    patch[ring[sl]] = bg_bgr
    erase_l = erase[sl]
    if fill is None:
        patch[erase_l] = bg_bgr
    else:
        scale = fill[sl][erase_l][:, None] / max(bg_l, 1.0)
        patch[erase_l] = np.clip(bg_bgr[None, :].astype(np.float32) * scale + 0.5, 0, 255).astype(np.uint8)
    return patch, rect, outline, np.ascontiguousarray(erase_l)


def _share_erased(blocks: Sequence[TextBlock]) -> None:
    """Keep a block's erase out of its neighbour's patch.  Two blocks in one
    corner of a page can end up with overlapping ``clean_rect``s, and a
    renderer paints them one after another: in the overlap the last patch
    wins and puts the other block's glyphs back on the page (2ja: the ``E`` of
    ``SASUKE`` came back once the balloon beside it grew).  Every patch takes
    over what the others erased inside it, so the painting order stops
    mattering.  ``erase_mask`` is left alone - it is this block's own glyph
    classification, and the quality renderer regenerates exactly that."""
    for i, a in enumerate(blocks):
        ra, ma, pa = a.style.clean_rect, a.style.erase_mask, a.style.clean_patch
        if ra is None or ma is None or pa is None:
            continue
        for b in blocks[i + 1 :]:
            rb, mb, pb = b.style.clean_rect, b.style.erase_mask, b.style.clean_patch
            if rb is None or mb is None or pb is None:
                continue
            x, y = max(ra.x, rb.x), max(ra.y, rb.y)
            x2, y2 = min(ra.x2, rb.x2), min(ra.y2, rb.y2)
            if x2 <= x or y2 <= y:
                continue
            sa = (slice(y - ra.y, y2 - ra.y), slice(x - ra.x, x2 - ra.x))
            sb = (slice(y - rb.y, y2 - rb.y), slice(x - rb.x, x2 - rb.x))
            only_a, only_b = ma[sa] & ~mb[sb], mb[sb] & ~ma[sa]
            pb[sb][only_a] = pa[sa][only_a]
            pa[sa][only_b] = pb[sb][only_b]


def page_glyph(blocks: Sequence[TextBlock]) -> float:
    """The page's lettering scale: the median block glyph size, each block
    weighted by the number of source characters it carries.

    Every reach in this module is a multiple of the block's glyph size - how far
    a column is swept (``SWEEP_MAX``), how wide the window is (``WINDOW_ACROSS``),
    how big a component may be and still read as a glyph (``GLYPH_MAX``) - so an
    over-measured glyph does not merely mis-erase the text, it enlarges the area
    the eraser is willing to delete.  A *plain* median is set by a one-character
    stylised sound effect as readily as by a line of dialogue, and a page of one
    or two blocks used to get no cap at all: measured over the 50-page corpus
    sample, 63 % of the source ink this module removed that the official release
    kept was in blocks whose glyph exceeded 1.25x the page median, a single
    one-character block accounting for 103k pixels of erased artwork.  Weighting
    by character count lets the page's actual lettering set the scale.

    Returns 0.0 when no block has a measured glyph size.
    """
    sizes: List[float] = []
    for b in blocks:
        g = float(b.style.text_height_px or 0.0)
        if g > 0:
            sizes.extend([g] * max(1, len(b.segment.text)))
    return float(np.median(sizes)) if sizes else 0.0


def apply(
    img_bgr: np.ndarray, blocks: List[TextBlock], all_segments: Sequence[Segment] = (), gray: Optional[np.ndarray] = None
) -> None:
    """Recompute ``clean_patch``, ``clean_rect``, ``outline`` and
    ``erase_mask`` of every block in place.  ``all_segments`` - the raw OCR
    lines of *any* confidence, a superset of the lines the blocks were built
    from - is read as evidence of ink: whatever in it no block owns goes to
    the block it belongs to (:func:`_evidence`) and is erased with that
    block's own glyphs, never translated.  ``gray`` may be the page's grey
    image (saves a conversion)."""
    if not blocks:
        return
    if gray is None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    height, width = img_bgr.shape[:2]
    evidence = _evidence(blocks, all_segments, width, height)
    boxes_per_block: List[List[Rect]] = []
    for i, b in enumerate(blocks):
        main, furi = _boxes(b)
        boxes_per_block.append(main + furi + evidence[i])
    scale = page_glyph(blocks)
    for i, b in enumerate(blocks):
        foreign = [r for j, rs in enumerate(boxes_per_block) if j != i for r in rs]
        g = b.style.text_height_px
        if scale > 0:
            g = min(g, GLYPH_CAP * scale)
        patch, rect, outline, mask = erase_block_masked(
            img_bgr, b, gray=gray, foreign=foreign, glyph=g, extra=evidence[i]
        )
        b.style.clean_patch = patch
        b.style.clean_rect = rect
        b.style.outline = outline
        b.style.erase_mask = mask
    _share_erased(blocks)
