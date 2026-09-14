"""Where a block of lettering is aimed, and the search that lands it there.

A speech balloon is not symmetric about the bounding box of its rows: a
tail, a flat side, or the neighbour it was cut away from drags that box one
way, so a block centred on the box sits visibly high or low in the balloon.
The letterer's middle is the centre of the largest circle that fits inside
the interior - what :func:`interior_centre` returns, and the definition
``demo/typeset_metrics.interior_centre`` scores the render against.

:class:`Placement` is the vertical half of the search that puts a block
there.  For one line count it takes the candidate offset nearest the anchor
(:func:`offset_ladder`); it settles that count when the flow uses fewer
lines than it was given, because a four line block dropped into a five line
slot hangs half a line high; it prefers the fewest lines that land *on* the
anchor to the fewest that land anywhere; and only then does it widen a
banner into an oval, by narrowing the measure rather than by sliding the
block into a narrower part of the balloon.  It knows nothing about fonts or
line breaking: ``flow``, ``budgets`` and ``finish`` all come from
``render/typeset.py``, which owns those.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - annotations only, no import cycle
    from .typeset import Typeset

Budgets = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
Finish = Callable[[np.ndarray, np.ndarray, float, Any], "Typeset"]

# Candidate vertical placements of a block are this many per line pitch: the
# block starts on the anchor and is nudged away in quarter-line steps until
# its lines flow.
_OFFSET_STEPS = 4
# A block whose inked centre lands within this fraction of the font size of
# the anchor is centred, and no taller block is tried in the hope of doing
# better.  It is the bound ``centre_offset_em`` is accepted at in
# ``demo/typeset_metrics`` (0.25 source ems).
ANCHOR_SLACK_EM = 0.25
# A block wider than this many times its height gets an extra line instead,
# when that still fits: comic lettering is oval, not a banner.
_MAX_ASPECT = 2.2
# Fractions of the line budgets (widest first) tried when a banner is set
# one line taller: the letterer shortens the measure until the extra line
# appears, and 0.6 of the bubble's width is as narrow as a block ever gets.
_BALANCE_FACTORS = tuple(round(1.0 - 0.05 * i, 2) for i in range(9))  # 1.0 .. 0.6
# A distance transform counts in pixel *indexes*; placed lines live in the
# continuous frame, where a pixel's middle is half a pixel further on.  With
# the half added a plain rectangle's optical centre is its exact centre.
_PIXEL_CENTRE = 0.5


def interior_centre(region: np.ndarray) -> Optional[Tuple[float, float]]:
    """Optical centre ``(x, y)`` of a bubble interior: the peak of its
    distance transform, i.e. the centre of the largest circle inscribed in
    it (the mean of the peak pixels when several tie).  Balloons have tails
    and flat sides, so neither the bounding-box centre nor the area centroid
    is where a letterer puts the middle of the block; the inscribed circle
    is.  The same definition as ``demo/typeset_metrics.interior_centre``,
    which scores the lettering against it.  None when ``region`` is empty.

    The region is padded with one row/column of background first.  A panel
    that runs to the edge of the page has no zeros beyond it, so an unpadded
    transform keeps rising all the way out and puts the peak *on the border*
    (4ja block 8, a dark panel at ``46,734,348,466`` on a 1200-row page,
    scores its centre at ``(241, 1199)`` instead of ``(171, 933)``).  A
    letterer centres the block in the part of the panel that is on the page."""
    region = region.astype(bool)
    if not region.any():
        return None
    padded = np.pad(region, 1).view(np.uint8)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt >= dt.max() - 1e-6)
    return float(xs.mean()) - 1.0, float(ys.mean()) - 1.0


def spans_centre(spans: np.ndarray, top: float) -> Optional[Tuple[float, float]]:
    """:func:`interior_centre` of the region described by row ``spans``
    (``[x1, x2)`` per row, the first row at absolute y ``top``), in absolute
    frame pixels.  The rows are rasterised into their own bounding box;
    :func:`interior_centre` pads it, so the transform sees the same open
    space outside the region that it sees on the page.

    Every row between the first and last inked one is rasterised, empty ones
    included: a region in two pieces, one above the other, must read as two
    pieces, or the gap closes up and the centre slides into it."""
    filled = np.flatnonzero(spans[:, 1] > spans[:, 0])
    if filled.size == 0:
        return None
    r0, r1 = int(filled[0]), int(filled[-1]) + 1
    left, right = spans[r0:r1, 0], spans[r0:r1, 1]
    x0 = int(np.floor(float(spans[filled, 0].min())))
    x1 = max(int(np.ceil(float(spans[filled, 1].max()))), x0 + 1)
    cols = np.arange(x0, x1, dtype=np.float64)
    region = (cols[None, :] >= left[:, None]) & (cols[None, :] < right[:, None])
    centre = interior_centre(region)
    if centre is None:
        return None
    return x0 + centre[0] + _PIXEL_CENTRE, top + float(r0) + centre[1] + _PIXEL_CENTRE


def line_centre(budget_centre: float, budget_w: float, line_w: float, anchor_x: Optional[float]) -> float:
    """Where a line of width ``line_w`` sits inside a budget of width
    ``budget_w`` centred on ``budget_centre``: on ``anchor_x`` when the line
    reaches it without leaving the budget, else as near it as it can get.
    Without an anchor the line keeps the budget's own centre, which is what
    a rectangle wants and what a caller-anchored block (free text) keeps."""
    if anchor_x is None or line_w > budget_w:
        return budget_centre
    slack = (budget_w - line_w) / 2.0
    return min(max(anchor_x, budget_centre - slack), budget_centre + slack)


def offset_ladder(base: float, lo: float, hi: float, step: float) -> np.ndarray:
    """Candidate block tops, nearest ``base`` first: ``base`` itself, then
    ``base`` one step down and up, two steps down and up, and so on; tops
    outside ``[lo, hi]`` are dropped."""
    k = np.arange(1, int((hi - lo) / step) + 2, dtype=np.float64)
    k = k[k * step <= (hi - lo)]
    offsets = np.concatenate(([0.0], np.stack((k * step, -k * step), axis=1).ravel()))
    tops = base + offsets
    return tops[(tops >= lo - 1e-6) & (tops <= hi + 1e-6)]


def ink_centre(ts: "Typeset", ink: Tuple[float, float]) -> float:
    """The y of a block's inked centroid, each line weighing its own width
    (``ink`` is the ``(top, bottom)`` offset of the inked rows within a line
    box).  That is the centroid ``demo/typeset_metrics.centre_offset_em``
    measures the block by, and a block whose last line is the longest inks
    lower than the middle of its box suggests.  ``inf`` when it has no
    lines."""
    weight = sum(line.width for line in ts.lines)
    if weight <= 0:
        return float("inf")
    mid = (ink[0] + ink[1]) / 2.0
    return sum(line.width * (line.top + mid) for line in ts.lines) / weight


def aspect(ts: "Typeset") -> float:
    """Width of the widest line over the height of the block."""
    widest = max((line.width for line in ts.lines), default=0.0)
    height = ts.line_h * max(1, len(ts.lines))
    return widest / height if height > 0 else 0.0


class Placement:
    """The search for where a block of lettering sits in a region at one
    font size.  ``flow`` flows a tuple of line widths (``typeset._Flower``),
    ``budgets`` gives the ``(widths, centres)`` of lines inking given row
    ranges (``typeset._Spans.budgets``), and ``finish`` turns a flow into a
    positioned ``Typeset``.  The rest is geometry: the line pitch, the
    ``(top, bottom)`` offsets of the inked rows within a line box, the
    region's first and last inked row, and the anchor to aim at."""

    def __init__(
        self,
        flow: Any,
        budgets: Budgets,
        finish: Finish,
        *,
        line_h: float,
        ink: Tuple[float, float],
        region: Tuple[float, float],
        anchor_y: float,
        slack: float,
        width_quantum: float,
    ) -> None:
        self.flow = flow
        self.budgets = budgets
        self.finish = finish
        self.line_h = line_h
        self.ink = ink
        self.ink_h1 = ink[1] - ink[0]
        self.region_top, self.region_bottom = region
        self.anchor_y = anchor_y
        self.slack = max(slack, line_h / (2 * _OFFSET_STEPS))
        self.quantum = width_quantum
        self._rows_cache: Dict[int, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._placed: Dict[Tuple[int, bool, bool, float], Optional["Typeset"]] = {}

    def _rows(self, n: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """``(block tops, line widths, line centres)`` of every candidate
        placement of an ``n``-line block, nearest the anchor first; one
        vectorised range query for the lot.  Cached: :meth:`settle` and
        :meth:`widen` ask for the same counts again."""
        if n in self._rows_cache:
            return self._rows_cache[n]
        block_h = (n - 1) * self.line_h + self.ink_h1
        lo = self.region_top - self.ink[0]
        hi = self.region_bottom - self.ink[0] - block_h
        out: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        if hi >= lo:
            base = min(max(self.anchor_y - block_h / 2.0 - self.ink[0], lo), hi)
            tops = offset_ladder(base, lo, hi, self.line_h / _OFFSET_STEPS)
            if tops.size:
                line_tops = tops[:, None] + np.arange(n, dtype=np.float64)[None, :] * self.line_h
                widths, centres = self.budgets(line_tops + self.ink[0], line_tops + self.ink[1])
                out = (tops, widths, centres)
        self._rows_cache[n] = out
        return out

    def _walk(
        self,
        rows: Tuple[np.ndarray, np.ndarray, np.ndarray],
        quantised: np.ndarray,
        order: np.ndarray,
        hyphenate: bool,
        allow_force: bool,
    ) -> Tuple[int, Optional["Typeset"]]:
        """The first placement in ``order`` whose lines flow, with the index
        it came from.  Widths are floored to the quantum, so offsets sharing
        a tuple share one cached flow."""
        tops, widths, centres = rows
        for o in order:
            flowed = self.flow.quantised(tuple(quantised[o].tolist()), hyphenate, allow_force)
            if flowed is not None:
                return int(o), self.finish(widths[o], centres[o], float(tops[o]), flowed)
        return -1, None

    def place(self, n: int, *, hyphenate: bool, allow_force: bool, factor: float = 1.0) -> Optional["Typeset"]:
        """The placement of an ``n``-line block that puts its ink on the
        anchor, or None.  ``factor`` narrows every line budget (see
        :meth:`widen`).  Cached: :meth:`settle`, :meth:`search` and
        :meth:`widen` between them ask for the same block again and again.

        The ladder aims the block's *box* at the anchor, but a block whose
        lines differ in width does not ink the middle of its box, so a
        placement that lands further than ``slack`` from the anchor is
        measured and the ladder walked again from the top that would cancel
        that error; the better of the two is kept.  Without hyphens the
        offsets that cannot hold the text even wall-to-wall are skipped
        before any lookup."""
        key = (n, hyphenate, allow_force, factor)
        if key in self._placed:
            return self._placed[key]
        out = self._place(n, hyphenate, allow_force, factor)
        self._placed[key] = out
        return out

    def _place(self, n: int, hyphenate: bool, allow_force: bool, factor: float) -> Optional["Typeset"]:
        """:meth:`place` without the cache."""
        rows = self._rows(n)
        if rows is None:
            return None
        tops, widths, centres = rows
        if factor != 1.0:
            widths = widths * factor
            rows = (tops, widths, centres)
        quantised = np.floor(widths / self.quantum) * self.quantum
        if hyphenate:
            order = np.arange(int(tops.size))
        else:
            order = np.flatnonzero(quantised.sum(axis=1) >= self.flow.total_w * 0.999)
        o, first = self._walk(rows, quantised, order, hyphenate, allow_force)
        if first is None:
            return None
        error = ink_centre(first, self.ink) - self.anchor_y
        if abs(error) <= self.slack:
            return first  # centred, or as close as a rung of the ladder gets
        target = np.abs(tops[order] - (tops[o] - error))
        _, second = self._walk(rows, quantised, order[np.argsort(target, kind="stable")],
                               hyphenate, allow_force)
        if second is None:
            return first
        return min(first, second, key=self.key)

    def settle(self, n: int, *, hyphenate: bool, allow_force: bool) -> Optional["Typeset"]:
        """:meth:`place` an ``n``-line block, then place it again for the
        number of lines the flow actually used: a block given five rows that
        fills only four is a four line block hanging half a line above the
        anchor.  The count only ever falls, so this ends."""
        settled: Optional["Typeset"] = None
        while n >= 1:
            ts = self.place(n, hyphenate=hyphenate, allow_force=allow_force)
            if ts is None:
                return settled
            if settled is None or self.key(ts) < self.key(settled):
                settled = ts
            if len(ts.lines) >= n:
                return settled
            n = len(ts.lines)
        return settled

    def offset(self, ts: "Typeset") -> float:
        """How far the block's inked centroid sits from the anchor."""
        return abs(ink_centre(ts, self.ink) - self.anchor_y)

    def key(self, ts: "Typeset") -> Tuple[int, float]:
        """How two placements are ranked: fewest hyphens first (the module's
        rule, see :func:`~glasstranslate.render.typeset.typeset`), then
        nearest the anchor."""
        return ts.hyphens, self.offset(ts)

    def _one_taller(self, want: int, *, hyphenate: bool, allow_force: bool, hyphens: int) -> Optional["Typeset"]:
        """The block of exactly ``want`` lines, set by narrowing the line
        budgets (``_BALANCE_FACTORS``, widest first) until the extra line
        appears.  None when no factor gains the line, when the narrowest
        budgets stop flowing, or when the taller block needs a hyphen the
        shorter one did not."""
        for factor in _BALANCE_FACTORS:
            cand = self.place(want, hyphenate=hyphenate, allow_force=allow_force, factor=factor)
            if cand is None:
                break  # narrower budgets only get harder from here
            if len(cand.lines) == want:
                return cand if cand.hyphens <= hyphens else None
        return None

    def widen(self, ts: "Typeset", max_lines: int, slack: float, *, hyphenate: bool, allow_force: bool) -> "Typeset":
        """Prefer an oval to a banner: while the block is wider than
        ``_MAX_ASPECT`` times its height, set it one line taller
        (:meth:`_one_taller`), keeping it on the anchor.  Stops as soon as a
        taller block cannot be had, or would sit further than ``slack`` from
        the anchor and further than the block it replaces: the shape of the
        block is worth less than its place in the balloon."""
        while aspect(ts) > _MAX_ASPECT and len(ts.lines) < max_lines:
            taller = self._one_taller(len(ts.lines) + 1, hyphenate=hyphenate,
                                      allow_force=allow_force, hyphens=ts.hyphens)
            if taller is None:
                break
            off = self.offset(taller)
            if off > slack and off > self.offset(ts):
                break
            ts = taller
        return ts

    def search(
        self, max_lines: int, widest_span: float, slack: float, *, hyphenate: bool, allow_force: bool
    ) -> Optional["Typeset"]:
        """The block to set at this size: the fewest lines that flow with no
        hyphen *and* land within ``slack`` of the anchor; when no count
        manages that, the one with the fewest hyphens that lands closest to
        the anchor (a block that merely fits somewhere no longer beats one
        that fits where it belongs).  The winner is then widened
        (:meth:`widen`).  None when nothing flows."""
        best: Optional["Typeset"] = None
        best_key = (1 << 30, float("inf"))
        for n in range(1, max_lines + 1):
            if not hyphenate and n * widest_span < self.flow.total_w * 0.999:
                continue  # cannot hold the text even wall-to-wall
            cand = self.settle(n, hyphenate=hyphenate, allow_force=allow_force)
            if cand is None:
                continue
            key = self.key(cand)
            if key < best_key:
                best, best_key = cand, key
            if best_key[0] == 0 and best_key[1] <= slack:
                break
        if best is None:
            return None
        return self.widen(best, max_lines, slack, hyphenate=hyphenate, allow_force=allow_force)
