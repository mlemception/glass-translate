"""Shape-aware typesetting: flow text into an arbitrary region (a speech
bubble, or a plain rectangle) so the lines follow the region's outline.

Pure Python + numpy, no Qt/PIL: the caller supplies ``measure(text, size)``
returning ``(width, height)``.  Both renderers (PIL ``compose`` and the Qt
overlay) share this module so the demo output and the live glass agree.

The region is described by *row spans*: for every pixel row of the layout
rectangle, the leftmost and rightmost x that text may occupy.  A rectangle
is the special case where every row has the same span; an oval bubble has
short spans at the top and bottom and long ones in the middle, which is
exactly what makes typeset manga text look like it belongs in the bubble.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Rect
from .fit import Measure

# Font sizes are tried from the ceiling downwards in these multiplicative steps.
_SIZE_STEP = 0.93
# Extra leading between lines as a fraction of the font size.
_LINE_GAP = 0.08
# Words at least this long may be hyphenated when they do not fit a line.
_MIN_HYPHENATE_LEN = 6
# Width scale factors tried (largest first) when balancing line lengths.
_BALANCE_FACTORS = tuple(round(1.0 - 0.05 * i, 2) for i in range(9))  # 1.0 .. 0.6


@dataclass(frozen=True)
class PlacedLine:
    """One typeset line: ``text`` centred at ``cx`` with its top at ``top``."""

    text: str
    cx: float
    top: float
    width: float


@dataclass(frozen=True)
class Typeset:
    """Result of :func:`typeset`: the chosen ``size`` and positioned lines."""

    size: float
    line_h: float
    lines: List[PlacedLine]

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)


def rect_spans(rect: Rect) -> np.ndarray:
    """Row spans for a plain rectangle: shape ``(rect.h, 2)`` of ``[x1, x2)``."""
    spans = np.empty((max(rect.h, 0), 2), dtype=np.float64)
    spans[:, 0] = rect.x
    spans[:, 1] = rect.x2
    return spans


def mask_spans(mask: np.ndarray, rect: Rect, cx: Optional[float] = None) -> np.ndarray:
    """Row spans of a boolean ``mask`` (shape ``rect.h x rect.w``) placed at
    ``rect``.  For each row the run of True pixels containing ``cx`` (default:
    the rect centre) is used; when that column is False the run nearest to
    ``cx`` is taken; rows with no True pixel get a zero-width span.  Runs are
    used, rather than the row's overall extent, so a concave bubble (one with
    a tail, or two bubbles joined) never lets a line cross empty space."""
    h, w = mask.shape[:2]
    spans = np.zeros((h, 2), dtype=np.float64)
    centre = (rect.w / 2.0) if cx is None else (cx - rect.x)
    for r in range(h):
        row = mask[r]
        if not row.any():
            continue
        padded = np.concatenate(([False], row.astype(bool), [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2]  # runs are [start, end)
        # Prefer the run containing the centre column, else the nearest one.
        inside = (starts <= centre) & (centre < ends)
        if inside.any():
            i = int(np.flatnonzero(inside)[0])
        else:
            dist = np.minimum(np.abs(starts - centre), np.abs(ends - 1 - centre))
            i = int(np.argmin(dist))
        spans[r] = (rect.x + starts[i], rect.x + ends[i])
    return spans


def _line_budget(spans: np.ndarray, top: float, bottom: float, base_y: float) -> Tuple[float, float]:
    """``(width, centre_x)`` available to a line occupying rows ``[top, bottom)``
    (absolute y).  The width is the *intersection* of the row spans so every
    glyph of the line stays inside the region."""
    r0 = max(0, int(np.floor(top - base_y)))
    r1 = min(spans.shape[0], int(np.ceil(bottom - base_y)))
    if r1 <= r0:
        return 0.0, 0.0
    rows = spans[r0:r1]
    left = float(rows[:, 0].max())
    right = float(rows[:, 1].min())
    if right <= left:
        return 0.0, 0.0
    return right - left, (left + right) / 2.0


def _hyphenate(word: str, size: float, width: float, measure: Measure) -> Optional[Tuple[str, str]]:
    """Split ``word`` into ``(head + '-', tail)`` with the longest head that
    fits ``width``; None when no split leaves at least two letters each side."""
    if len(word) < _MIN_HYPHENATE_LEN:
        return None
    for cut in range(len(word) - 2, 1, -1):
        head = word[:cut] + "-"
        if measure(head, size)[0] <= width:
            return head, word[cut:]
    return None


def _flow(words: Sequence[str], size: float, widths: Sequence[float], measure: Measure) -> Optional[List[str]]:
    """Greedily place ``words`` into consecutive lines of the given ``widths``.
    Returns the line texts, or None when the words do not fit."""
    lines: List[str] = []
    pending = list(words)
    for width in widths:
        if not pending:
            break
        if width <= 0:
            lines.append("")
            continue
        current = ""
        while pending:
            word = pending[0]
            candidate = f"{current} {word}" if current else word
            if measure(candidate, size)[0] <= width:
                current = candidate
                pending.pop(0)
                continue
            if current:
                break  # line full, next word starts the next line
            split = _hyphenate(word, size, width, measure)
            if split is None:
                return None  # a word that cannot go anywhere: this size fails
            head, tail = split
            current = head
            pending[0] = tail
            break
        lines.append(current)
    if pending:
        return None
    while lines and not lines[-1]:
        lines.pop()
    return lines


def typeset(
    text: str,
    spans: np.ndarray,
    top: float,
    measure: Measure,
    *,
    max_size: float,
    min_size: float = 6.0,
    line_gap: float = _LINE_GAP,
) -> Typeset:
    """Flow ``text`` into the region described by ``spans`` (row spans whose
    first row is at absolute y ``top``), choosing the largest font size not
    above ``max_size`` at which every word fits, then balancing line lengths.

    The line block is centred vertically in the region.  For each candidate
    size the smallest line count that holds the text wins; within that line
    count the lines are re-flowed at progressively narrower widths as long as
    the count does not grow, which evens the lines out (a centred pyramid or
    oval instead of a long first line and an orphan).  When nothing fits even
    at ``min_size`` the text is flowed at ``min_size`` through a rectangle as
    wide as the widest span, so a result is always produced.
    """
    words = text.split()
    height = float(spans.shape[0])
    if not words or height <= 0:
        return Typeset(max_size, max_size, [])

    size = max(max_size, min_size)
    while True:
        glyph_h = measure("Mg", size)[1]
        line_h = glyph_h * (1.0 + line_gap)
        max_lines = int(height // line_h)
        for n in range(1, max_lines + 1):
            block_top = top + (height - n * line_h) / 2.0
            budgets = [_line_budget(spans, block_top + i * line_h, block_top + (i + 1) * line_h, top) for i in range(n)]
            widths = [b[0] for b in budgets]
            lines = _flow(words, size, widths, measure)
            if lines is None:
                continue
            # Balance: narrow every line's budget until the text needs more lines.
            for factor in _BALANCE_FACTORS[1:]:
                narrower = _flow(words, size, [w * factor for w in widths], measure)
                if narrower is None or len(narrower) > len(lines):
                    break
                lines = narrower
            placed = [
                PlacedLine(line, budgets[i][1], block_top + i * line_h, measure(line, size)[0])
                for i, line in enumerate(lines)
            ]
            return Typeset(size, line_h, placed)
        if size <= min_size:
            break
        size = max(size * _SIZE_STEP, min_size)

    # Nothing fits: flow at min_size through the widest rectangle and centre it.
    widest = float(np.max(spans[:, 1] - spans[:, 0])) if spans.size else 0.0
    cx = float(np.mean(spans[:, :2])) if spans.size else 0.0
    glyph_h = measure("Mg", min_size)[1]
    line_h = glyph_h * (1.0 + line_gap)
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if measure(candidate, min_size)[0] <= widest or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    block_top = top + (height - len(lines) * line_h) / 2.0
    placed = [PlacedLine(l, cx, block_top + i * line_h, measure(l, min_size)[0]) for i, l in enumerate(lines)]
    return Typeset(min_size, line_h, placed)
