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

Lettering conventions followed here, all borrowed from professional comic
typesetting:

* the largest size not above the caller's ceiling is used, but lines are
  then narrowed until the block is a compact oval/diamond rather than a few
  wall-to-wall lines;
* all-caps lettering is set with tight leading (line pitch about 1.22 x cap
  height, see :func:`line_pitch`); text with lower-case letters keeps the
  font's full ascent + descent pitch so descenders never touch the next line;
* words are only hyphenated at dictionary syllable boundaries (``pyphen``)
  or after a hyphen they already contain (``RE-CRE-`` / ``ATING``), at most
  once per word, and a layout without hyphens is preferred over one with
  them at the same size, and over one at a slightly larger size (each hyphen
  costs ``_HYPHEN_SIZE_COST`` of the size); a word is split arbitrarily only
  as a last resort,
  when it is wider than the region and the size has already dropped to
  ``_FORCE_SPLIT_SCALE`` of the ceiling;
* lines are balanced (a dynamic programme evens out their fill, so no
  single word is left alone on the last line when a rebreak avoids it);
* the line block is centred on the region's *optical* centre - the middle
  of the largest circle inscribed in it, not the middle of its bounding box,
  which a tail or a flat side drags off the balloon - or on the caller's
  anchor when it passes one (:mod:`.anchor`);
* a region made of two joined boxes (two lobes, e.g. two caption boxes
  sharing an edge) holding a sentence with an ellipsis boundary
  (``... ...``) gets the first part in the upper box and the rest in the
  lower one (:func:`typeset_lobes`).

Vertical metrics
----------------
``measure`` only reports the font's ascent + descent (the *line height*,
``glyph_h`` below).  The cap height, the cap top and the deepest all-caps
descender are expressed as documented fractions of that line height
(``CAP_RATIO`` etc.), measured on the bundled Anime Ace 2.0 BB; PIL and Qt
report the same ascent + descent for the same font, so the ratios hold for
both renderers.  ``PlacedLine.top`` keeps its meaning as the top of the
font's line box (PIL ``draw.text`` origin; Qt baseline = top + ascent), so
renderers need no change when the pitch is tightened.
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.types import Rect
from .anchor import ANCHOR_SLACK_EM, Placement, ink_centre, interior_centre, line_centre, spans_centre
from .fit import Measure

try:  # dictionary hyphenation; optional so the package imports without it
    import pyphen as _pyphen
except Exception:  # pragma: no cover - depends on the environment
    _pyphen = None

# --- vertical metrics (fractions of the measured line height ascent+descent) --
# Anime Ace 2.0 BB (regular / italic): ascent 1.07 em, descent 0.15 em ->
# line height 1.21 em; cap height 0.84 / 0.82 em (H) -> 0.69 of the line
# height; the cap top sits 0.22 / 0.24 em below the line top -> 0.18-0.20;
# the deepest all-caps descender (the comma, 0.15 em below the baseline)
# reaches the bottom of the line box -> 1.0.
CAP_RATIO = 0.69
CAP_TOP_RATIO = 0.19
DESCENT_RATIO = 1.0
# Leading of all-caps comic lettering: line pitch = 1.22 x cap height
# (measured on professionally lettered pages) = 0.73 x line height.
CAPS_LEADING = 1.22
CAPS_PITCH_RATIO = round(CAPS_LEADING * CAP_RATIO, 3)  # 0.732
# Text with lower-case letters keeps the full ascent + descent pitch.
MIXED_PITCH_RATIO = 1.0

# Font sizes are tried from the ceiling downwards in these multiplicative steps.
_SIZE_STEP = 0.93
# A hyphenated layout is kept only while no smaller size sets the text with
# fewer hyphens at a lower cost: each hyphen costs this fraction of the size
# (one hyphen costs about 1.4 size steps; two hyphens lose to three steps).
_HYPHEN_SIZE_COST = 0.1
# Extra leading between lines as a fraction of the line pitch (default none).
_LINE_GAP = 0.0
# Words at least this long may be split *without* a dictionary point, and
# only when they are wider than the region itself and the size has already
# fallen to this fraction of the ceiling: shrinking beats an ugly split.
_MIN_FORCE_SPLIT_LEN = 8
_FORCE_SPLIT_SCALE = 0.7
# Two joined boxes: a row whose span centre jumps by more than this fraction
# of the (narrower) span width starts a new lobe...
_LOBE_JUMP_FRAC = 0.25
# ...and the rows where the boxes' interiors join (the run spans both) may
# take up to this fraction of the region's height; each lobe must have at
# least this fraction of the rows (a tail or a rounded corner is no lobe).
_LOBE_JOINT_FRAC = 0.15
_LOBE_MIN_ROWS_FRAC = 0.2
# The two-lobe split is used when its size is at least this fraction of the
# size the continuous flow reaches (it costs at most two size steps).
_LOBE_MIN_SCALE = 0.85
_ELLIPSIS_BOUNDARY = re.compile(r"(\.{3}|…)\s+(?=(\.{3}|…))")
# Minimum letters left on each side of a hyphen.
_HYPHEN_MIN_SIDE = 2
# Characters stripped from a word before looking it up for hyphenation.
_WORD_PUNCT = ".,;:!?\"'()[]…-–—“”‘’"
# A block that cannot be centred at one size may be centred a size or two
# smaller: a balloon whose wide part is short holds fewer big lines than
# small ones, and the big block can only sit above it.  Every em the block
# inks away from the anchor, past the ``anchor.ANCHOR_SLACK_EM`` a centred
# block is allowed, costs this fraction of the size - the same price as a
# hyphen, so one ``_SIZE_STEP`` step buys about 0.7 em of centring and no
# more.  Measured over the five reference pages, 0.1 is the knee: it moves
# four more blocks inside 0.25 em and three out of the 0.6 em tail for one
# per cent of cap height, and paying more only shrinks more blocks.
_OFF_CENTRE_SIZE_COST = 0.1
# Line widths are floored to this many pixels before flowing, so nearby
# vertical offsets of the same block share one cached flow (conservative:
# a flow that fits the floored widths fits the real ones).
_WIDTH_QUANTUM = 2.0

# Hyphenation dictionaries by language.  The only module-level cache in the
# typesetting path; it is shared by the PIL renderer (pipeline / demo thread)
# and the Qt overlay (GUI thread), hence the lock.  Every other cache
# (:func:`memoize_measure`, ``_Flower``) lives inside one call.
_DICTS: dict = {}
_DICTS_LOCK = threading.Lock()


def _dictionary(lang: str):
    """A ``pyphen.Pyphen`` for ``lang`` (``en`` -> ``en_US``), cached; None
    when pyphen or the language is unavailable.  Thread-safe."""
    if _pyphen is None:
        return None
    key = lang.lower()
    with _DICTS_LOCK:
        if key in _DICTS:
            return _DICTS[key]
    dic = None
    # The regional dictionary first: plain "en" resolves to British patterns
    # (AMP-LI-FIC-A-TION); American ones (AM-PLI-FI-CA-TION) are the comics norm.
    for code in ({"en": "en_US", "de": "de_DE", "fr": "fr", "es": "es", "pt": "pt_BR", "it": "it_IT", "nl": "nl_NL", "ru": "ru_RU"}.get(key, lang), lang):
        try:
            dic = _pyphen.Pyphen(lang=code, left=_HYPHEN_MIN_SIDE, right=_HYPHEN_MIN_SIDE)
            break
        except Exception:
            continue
    with _DICTS_LOCK:
        return _DICTS.setdefault(key, dic)


# ------------------------------------------------------------ metrics
def is_caps(text: str) -> bool:
    """True when ``text`` is all-caps lettering: it has upper-case letters
    and no lower-case one.  Text without cased letters at all (CJK, digits)
    is *not* caps: its glyphs fill the whole em, so it keeps the full pitch
    and the whole line box counts as inked."""
    return any(c.isupper() for c in text) and not any(c.islower() for c in text)


def line_pitch(text: str, glyph_h: float, line_gap: float = _LINE_GAP) -> float:
    """Distance between consecutive line tops for ``text`` drawn with a font
    whose ascent + descent is ``glyph_h``: tight (``CAPS_PITCH_RATIO``) for
    all-caps text, the full line height otherwise, times ``1 + line_gap``."""
    ratio = CAPS_PITCH_RATIO if is_caps(text) else MIXED_PITCH_RATIO
    return glyph_h * ratio * (1.0 + line_gap)


def ink_offsets(text: str, glyph_h: float) -> Tuple[float, float]:
    """``(top, bottom)`` of the pixels a line of ``text`` can ink, relative to
    the line top: the cap top and the deepest descender for all-caps text,
    the whole line box otherwise."""
    if is_caps(text):
        return CAP_TOP_RATIO * glyph_h, DESCENT_RATIO * glyph_h
    return 0.0, glyph_h


def ink_height(n: int, pitch: float, glyph_h: float, caps: bool = True) -> float:
    """Height of the inked area of ``n`` lines at ``pitch`` (first cap top to
    last descender)."""
    top, bottom = (CAP_TOP_RATIO * glyph_h, DESCENT_RATIO * glyph_h) if caps else (0.0, glyph_h)
    return max(0, n - 1) * pitch + (bottom - top)


def memoize_measure(measure: Measure) -> Measure:
    """Wrap ``measure`` in a per-call dictionary cache keyed on ``(text, size)``.
    Typesetting re-measures the same strings hundreds of times while it
    searches; font measurement is by far the dominant cost."""
    if getattr(measure, "_memoized", False):
        return measure
    cache: Dict[Tuple[str, float], Tuple[float, float]] = {}

    def cached(text: str, size: float) -> Tuple[float, float]:
        key = (text, size)
        hit = cache.get(key)
        if hit is None:
            hit = cache[key] = measure(text, size)
        return hit

    cached._memoized = True  # type: ignore[attr-defined]
    return cached


@dataclass(frozen=True)
class PlacedLine:
    """One typeset line: ``text`` centred at ``cx`` with its top at ``top``
    (top of the font's line box: PIL text origin, Qt baseline - ascent)."""

    text: str
    cx: float
    top: float
    width: float


@dataclass(frozen=True)
class Typeset:
    """Result of :func:`typeset`: the chosen ``size`` and positioned lines.
    ``line_h`` is the line *pitch* (distance between consecutive tops);
    ``glyph_h`` the font's ascent + descent at ``size`` (0 = unknown, treated
    as ``line_h``).  ``fitted`` is False when nothing fitted even at the
    minimum size and the text was flowed through the widest rectangle
    instead (it may overflow)."""

    size: float
    line_h: float
    lines: List[PlacedLine]
    fitted: bool = True
    hyphens: int = 0
    glyph_h: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)

    @property
    def bbox(self) -> Optional[Rect]:
        """Axis-aligned box around the inked pixels of the placed lines
        (frame pixels): from the first line's cap top to the last line's
        deepest descender for all-caps text."""
        if not self.lines:
            return None
        x1 = min(l.cx - l.width / 2.0 for l in self.lines)
        x2 = max(l.cx + l.width / 2.0 for l in self.lines)
        gh = self.glyph_h if self.glyph_h > 0 else self.line_h
        top, bottom = ink_offsets(self.text, gh) if self.glyph_h > 0 else (0.0, self.line_h)
        y1 = min(l.top for l in self.lines) + top
        y2 = max(l.top for l in self.lines) + bottom
        return Rect(int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2 - x1)), int(np.ceil(y2 - y1)))


def rect_spans(rect: Rect) -> np.ndarray:
    """Row spans for a plain rectangle: shape ``(rect.h, 2)`` of ``[x1, x2)``."""
    spans = np.empty((max(rect.h, 0), 2), dtype=np.float64)
    spans[:, 0] = rect.x
    spans[:, 1] = rect.x2
    return spans


def mask_spans(mask: np.ndarray, rect: Rect, cx: Optional[float] = None) -> np.ndarray:
    """Row spans of a boolean ``mask`` (shape ``rect.h x rect.w``) placed at
    ``rect``.  For each row the run of True pixels containing ``cx`` (default:
    the mask's optical centre column, :func:`.anchor.interior_centre`) is
    used; when that column is False the run nearest to ``cx`` is taken; rows
    with no True pixel get a zero-width span.  The optical column is the
    balloon's own, where the rect centre is only the middle of the box the
    layout cut around it: on a mask holed by un-OCR'd Japanese, or one whose
    balloon sits off to one side of its box, the rect centre picks a run in
    the wrong part of the region and the block is then flowed and anchored
    there (1ja block 13, 2.95 em of anchor error against
    ``demo/typeset_metrics``, 0.05 em on the optical column).  Runs are
    used, rather than the row's overall extent, so a concave bubble (one with
    a tail, or two bubbles joined) never lets a line cross empty space.

    All runs of all rows are found with one 2-D diff of the padded mask and
    the run to keep per row is chosen with a single sort (rows, then a key
    that puts the run containing the centre first, then distance, then
    order)."""
    h, w = mask.shape[:2]
    spans = np.zeros((h, 2), dtype=np.float64)
    if h == 0 or w == 0:
        return spans
    if cx is not None:
        centre = cx - rect.x
    else:
        optical = interior_centre(mask)
        centre = rect.w / 2.0 if optical is None else optical[0]
    padded = np.zeros((h, w + 2), dtype=bool)
    padded[:, 1:-1] = mask.astype(bool, copy=False)
    edges = padded[:, 1:] != padded[:, :-1]  # (h, w + 1): run starts and ends
    rows, cols = np.nonzero(edges)  # row-major: every row's edges in order, in pairs
    if rows.size == 0:
        return spans
    run_rows = rows[0::2]
    starts = cols[0::2]
    ends = cols[1::2]  # runs are [start, end)
    inside = (starts <= centre) & (centre < ends)
    dist = np.minimum(np.abs(starts - centre), np.abs(ends - 1 - centre))
    key = np.where(inside, -1.0, dist)
    order = np.lexsort((np.arange(run_rows.size), key, run_rows))
    first = np.ones(order.size, dtype=bool)
    first[1:] = run_rows[order][1:] != run_rows[order][:-1]
    pick = order[first]
    spans[run_rows[pick], 0] = rect.x + starts[pick]
    spans[run_rows[pick], 1] = rect.x + ends[pick]
    return spans


def clip_spans(spans: np.ndarray, top: float, rect: Rect) -> Tuple[np.ndarray, float]:
    """Restrict ``spans`` (first row at absolute y ``top``) to ``rect``:
    rows outside are dropped and every span is intersected with the rect's
    x range.  Returns ``(spans, new_top)``."""
    r0 = max(0, int(round(rect.y - top)))
    r1 = min(spans.shape[0], int(round(rect.y2 - top)))
    if r1 <= r0:
        return np.zeros((0, 2), dtype=np.float64), float(rect.y)
    out = spans[r0:r1].copy()
    out[:, 0] = np.maximum(out[:, 0], rect.x)
    out[:, 1] = np.minimum(out[:, 1], rect.x2)
    empty = out[:, 1] <= out[:, 0]
    out[empty] = 0.0
    return out, top + r0


def _line_budget(spans: np.ndarray, top: float, bottom: float, base_y: float,
                 anchor_x: Optional[float] = None) -> Tuple[float, float]:
    """``(width, centre_x)`` available to a line inking rows ``[top, bottom)``
    (absolute y).  The width is the *intersection* of the row spans so every
    glyph of the line stays inside the region.  Reference implementation of
    :meth:`_Spans.budgets` (kept for tests and single queries).

    With an ``anchor_x`` the intersection is cut **symmetric about that axis**
    instead, ``half = min(anchor_x - left, right - anchor_x)``.  A block is
    already centred on one axis - :func:`.anchor.line_centre` aims every line
    at it - but that function can only clamp a line into the budget it is
    given, and the intersection's centre is not the axis wherever the region
    is lopsided about it, which on a balloon is most rows.  A line as wide as
    its budget then has no slack and takes the row's centre, so the block
    wanders: measured std 7.56 px against a professionally lettered 0.45 px.

    Making the budget symmetric puts the axis at its centre by construction,
    so the clamp can no longer pull a line off it, and the budget stays a
    strict subset of the intersection, so nothing leaves the region.  The
    taper a letterer draws round an oval falls out of it - the symmetric
    half-width is small at the top and bottom and widest at the waist - with
    no rule anywhere saying to make one."""
    r0 = max(0, int(np.floor(top - base_y)))
    r1 = min(spans.shape[0], int(np.ceil(bottom - base_y)))
    if r1 <= r0:
        return 0.0, 0.0
    rows = spans[r0:r1]
    left = float(rows[:, 0].max())
    right = float(rows[:, 1].min())
    if right <= left:
        return 0.0, 0.0
    if anchor_x is None:
        return right - left, (left + right) / 2.0
    half = min(anchor_x - left, right - anchor_x)
    if half <= 0.0:
        return 0.0, 0.0
    return 2.0 * half, float(anchor_x)


class _Spans:
    """Row spans of a region with sparse tables for O(1) range queries, so
    the line budgets of every candidate vertical offset of a block are
    computed in one vectorised call instead of one ``max``/``min`` per line
    per offset (the dominant cost of flowing bubble text)."""

    def __init__(self, spans: np.ndarray, top: float, anchor_x: Optional[float] = None) -> None:
        self.spans = spans
        self.top = float(top)
        self.anchor_x = None if anchor_x is None else float(anchor_x)
        self.h = int(spans.shape[0])
        left = np.ascontiguousarray(spans[:, 0], dtype=np.float64)
        right = np.ascontiguousarray(spans[:, 1], dtype=np.float64)
        self._left: List[np.ndarray] = [left]
        self._right: List[np.ndarray] = [right]
        j = 1
        while (1 << j) <= self.h:
            half = 1 << (j - 1)
            pl, pr = self._left[-1], self._right[-1]
            self._left.append(np.maximum(pl[:-half], pl[half:]))
            self._right.append(np.minimum(pr[:-half], pr[half:]))
            j += 1

    def budgets(self, tops: np.ndarray, bottoms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(widths, centres)`` of the lines inking rows ``[tops, bottoms)``
        (absolute y, arrays of any common shape); zero where a line has no
        room.  Bit-for-bit the same as :func:`_line_budget` per element."""
        tops = np.asarray(tops, dtype=np.float64)
        bottoms = np.asarray(bottoms, dtype=np.float64)
        r0 = np.maximum(0, np.floor(tops - self.top)).astype(np.int64)
        r1 = np.minimum(self.h, np.ceil(bottoms - self.top)).astype(np.int64)
        k = r1 - r0
        valid = k >= 1
        widths = np.zeros(tops.shape, dtype=np.float64)
        centres = np.zeros(tops.shape, dtype=np.float64)
        if not valid.any():
            return widths, centres
        kk = np.where(valid, k, 1)
        j_all = np.frexp(kk.astype(np.float64))[1] - 1  # floor(log2(k)), exact
        left = np.zeros(tops.shape, dtype=np.float64)
        right = np.zeros(tops.shape, dtype=np.float64)
        for j in np.unique(j_all[valid]):
            sel = valid & (j_all == j)
            span = 1 << int(j)
            a, b = r0[sel], r1[sel] - span
            tl, tr = self._left[int(j)], self._right[int(j)]
            left[sel] = np.maximum(tl[a], tl[b])
            right[sel] = np.minimum(tr[a], tr[b])
        ok = valid & (right > left)
        if self.anchor_x is None:
            widths[ok] = right[ok] - left[ok]
            centres[ok] = (left[ok] + right[ok]) / 2.0
            return widths, centres
        # Symmetric about the axis; see :func:`_line_budget`.
        half = np.minimum(self.anchor_x - left, right - self.anchor_x)
        ok &= half > 0.0
        widths[ok] = 2.0 * half[ok]
        centres[ok] = self.anchor_x
        return widths, centres


def _split_word(word: str) -> Tuple[str, str, str]:
    """``word`` -> ``(leading punctuation, core, trailing punctuation)``."""
    start, end = 0, len(word)
    while start < end and word[start] in _WORD_PUNCT:
        start += 1
    while end > start and word[end - 1] in _WORD_PUNCT:
        end -= 1
    return word[:start], word[start:end], word[end:]


def hyphenation_points(word: str, lang: str = "en") -> List[int]:
    """Indexes of ``word`` where a line may break, ascending: dictionary
    syllable boundaries (a hyphen is inserted there, see :func:`split_at`)
    and, for a word that already contains a hyphen (``RE-CREATING``), the
    positions right after each hyphen plus the syllable boundaries of every
    part.  Empty when the word has no legal break: short words, numbers, or
    when no dictionary is available."""
    lead, core, _ = _split_word(word)
    if "-" in core:
        points: List[int] = []
        offset = len(lead)
        for part in core.split("-"):
            if offset > len(lead):
                points.append(offset)  # break after the existing hyphen
            points.extend(offset + p for p in _syllable_points(part, lang))
            offset += len(part) + 1
        return points
    return [len(lead) + p for p in _syllable_points(core, lang)]


def _syllable_points(core: str, lang: str) -> List[int]:
    if len(core) < 2 * _HYPHEN_MIN_SIDE + 1 or not core.isalpha():
        return []
    dic = _dictionary(lang)
    if dic is None:
        return []
    return [p for p in dic.positions(core) if _HYPHEN_MIN_SIDE <= p <= len(core) - _HYPHEN_MIN_SIDE]


def split_at(word: str, cut: int) -> Tuple[str, str]:
    """``(head, tail)`` of ``word`` broken at ``cut``: the head ends with a
    hyphen (its own when the cut follows one the word already had)."""
    head = word[:cut]
    if not head.endswith("-"):
        head += "-"
    return head, word[cut:]


def _hyphenate(
    word: str, size: float, width: float, measure: Measure, lang: str, *, force: bool
) -> Optional[Tuple[str, str, bool, bool]]:
    """Split ``word`` into ``(head, tail, inserted, forced)`` at the latest
    legal point (:func:`hyphenation_points`) whose head fits ``width``;
    ``inserted`` is False when the break follows a hyphen the word already
    had.  With ``force`` (the word is wider than the whole region, so it
    cannot be placed intact anywhere, and shrinking has been exhausted) any
    position leaving ``_HYPHEN_MIN_SIDE + 1`` letters on both sides is
    allowed (``forced`` True)."""
    points = hyphenation_points(word, lang)
    for cut in reversed(points):
        head, tail = split_at(word, cut)
        if measure(head, size)[0] <= width:
            return head, tail, not word[:cut].endswith("-"), False
    if force and len(word) >= _MIN_FORCE_SPLIT_LEN:
        lead, core, _ = _split_word(word)
        for cut in range(len(lead) + len(core) - _HYPHEN_MIN_SIDE - 1, len(lead) + _HYPHEN_MIN_SIDE, -1):
            head, tail = split_at(word, cut)
            if measure(head, size)[0] <= width:
                return head, tail, True, True
    return None


def _flow(
    words: Sequence[str],
    size: float,
    widths: Sequence[float],
    measure: Measure,
    *,
    hyphenate: bool,
    allow_force: bool,
    lang: str,
) -> Optional[Tuple[List[str], int, int]]:
    """Greedily place ``words`` into consecutive lines of the given ``widths``.
    Returns ``(line texts, hyphens inserted, forced splits)``, or None when
    the words do not fit (or a line in the middle of the block has no room
    at all).  A word is broken at most once; a word wider than the region is
    broken even without ``hyphenate``, arbitrarily only with ``allow_force``."""
    lines: List[str] = []
    pending: List[Tuple[str, int]] = [(w, 0) for w in words]
    hyphens = forced = 0
    widest = max(widths) if widths else 0.0
    for width in widths:
        if not pending:
            break
        if width <= 0:
            return None  # a blank line inside the block: try another placement
        current = ""
        while pending:
            word, depth = pending[0]
            candidate = f"{current} {word}" if current else word
            if measure(candidate, size)[0] <= width:
                current = candidate
                pending.pop(0)
                continue
            if current:
                break  # line full, next word starts the next line
            if depth >= 1:
                return None  # already broken once: shrink instead of splitting again
            too_wide = measure(word, size)[0] > widest
            split = _hyphenate(word, size, width, measure, lang, force=too_wide and allow_force) if (hyphenate or too_wide) else None
            if split is None:
                return None  # a word that cannot go anywhere: this size fails
            head, tail, inserted, was_forced = split
            current = head
            pending[0] = (tail, depth + 1)
            hyphens += int(inserted)
            forced += int(was_forced)
            break
        lines.append(current)
    if pending:
        return None
    return lines, hyphens, forced


class _Flower:
    """Cached greedy flow of one word list at one size: identical (floored)
    width tuples are flowed once, however many vertical offsets produce them."""

    def __init__(self, words: Sequence[str], size: float, measure: Measure, lang: str) -> None:
        self.words = tuple(words)
        self.size = size
        self.measure = measure
        self.lang = lang
        self.cache: Dict[Tuple[bool, bool, Tuple[float, ...]], Optional[Tuple[List[str], int, int]]] = {}
        # Cheap lower bound: the text needs at least this much total width.
        self.total_w = sum(measure(w, size)[0] for w in self.words) + measure(" ", size)[0] * max(0, len(self.words) - 1)

    def __call__(self, widths: Sequence[float], hyphenate: bool, allow_force: bool = False) -> Optional[Tuple[List[str], int, int]]:
        q = tuple(math.floor(w / _WIDTH_QUANTUM) * _WIDTH_QUANTUM for w in widths)
        if sum(q) < self.total_w * 0.999 and not hyphenate:
            # Even wall-to-wall the text cannot fit these lines without hyphens.
            return None
        return self.quantised(q, hyphenate, allow_force)

    def quantised(self, q: Tuple[float, ...], hyphenate: bool, allow_force: bool = False) -> Optional[Tuple[List[str], int, int]]:
        """Flow for widths already floored to the quantum (cached)."""
        key = (hyphenate, allow_force, q)
        if key in self.cache:
            return self.cache[key]
        out = _flow(self.words, self.size, q, self.measure, hyphenate=hyphenate, allow_force=allow_force, lang=self.lang)
        self.cache[key] = out
        return out


def _rebalance(lines: List[str], budgets: Sequence[float], size: float, measure: Measure) -> List[str]:
    """Re-break the tokens of the greedily flowed ``lines`` into the same
    number of lines so that every line fills the same fraction of its
    budget (the letterer's balanced block, no orphan last word).  Tokens
    ending with a hyphen (heads of broken words) keep ending their line."""
    n = len(lines)
    if n < 2:
        return lines
    tokens = [t for line in lines for t in line.split()]
    if len(tokens) <= n:
        return lines
    forced = [i for i, t in enumerate(tokens) if t.endswith("-")]
    width = span_widths(tokens, size, measure)
    total = sum(width(i, i + 1) for i in range(len(tokens))) + measure(" ", size)[0] * (len(tokens) - n)
    avail = sum(budgets[:n])
    fill = min(1.0, total / avail) if avail > 0 else 1.0
    caps = [b + 1e-6 for b in budgets[:n]]
    tgts = [fill * b for b in budgets[:n]]
    # The letterer's order: end a line at every sentence end first, and balance
    # what is left inside each piece.  When those breaks do not fit the lines
    # available the DP says so (None, or the wrong count) and the plain
    # balanced block stands, exactly as before.
    beats = sorted(set(forced) | set(sentence_breaks(tokens)))
    # ...and among the breaks that remain, prefer the ones a letterer prefers.
    sense = sense_costs(tokens)
    starts = balanced_breaks(len(tokens), n, caps, width, tgts, beats, sense)
    if starts is None or len(starts) != n + 1:
        starts = balanced_breaks(len(tokens), n, caps, width, tgts, forced, sense)
    if starts is None or len(starts) != n + 1:
        return lines
    return [" ".join(tokens[starts[i] : starts[i + 1]]) for i in range(n)]


def _finish(
    flow: _Flower,
    line_h: float,
    glyph_h: float,
    widths: np.ndarray,
    centres: np.ndarray,
    block_top: float,
    flowed: Tuple[List[str], int, int],
    anchor_x: Optional[float] = None,
) -> Typeset:
    """Balance the flowed lines over their budgets and position them: each
    line is centred on ``anchor_x`` as far as its own budget lets it reach
    (:func:`.anchor.line_centre`), and on the budget's centre when the block
    has no horizontal anchor."""
    lines, hyphens, forced = flowed
    lines = _rebalance(lines, widths.tolist(), flow.size, flow.measure)
    placed: List[PlacedLine] = []
    for i, line in enumerate(lines):
        width = flow.measure(line, flow.size)[0]
        cx = line_centre(float(centres[i]), float(widths[i]), width, anchor_x)
        placed.append(PlacedLine(line, cx, block_top + i * line_h, width))
    return Typeset(flow.size, line_h, placed, True, hyphens + forced, glyph_h)


def typeset(
    text: str,
    spans: np.ndarray,
    top: float,
    measure: Measure,
    *,
    max_size: float,
    min_size: float = 6.0,
    line_gap: float = _LINE_GAP,
    anchor_y: Optional[float] = None,
    lang: str = "en",
) -> Typeset:
    """Flow ``text`` into the region described by ``spans`` (row spans whose
    first row is at absolute y ``top``), choosing the largest font size not
    above ``max_size`` at which every word fits, then balancing line lengths.

    The inked block is centred on ``anchor_y`` (default: the region's
    *optical* centre, the middle of the largest circle inscribed in it -
    :func:`.anchor.spans_centre` - which a tail or a flat side cannot drag
    off the balloon the way it drags the middle of the rows' bounding box),
    sliding away from it only as far as needed to find rows with room; when
    the anchor is the default the lines are centred on its x as well, as far
    as each line's own budget lets it reach.  For each candidate size the
    smallest line count that holds the text *and* lands on the anchor is
    found first; when that block is wider than ``anchor._MAX_ASPECT`` times
    its height, the measure is narrowed until it takes another line, so a
    wide bubble gets an oval rather than a banner.  At a given size
    a layout without hyphens (dictionary syllable breaks, at most one per
    word) beats one with them, even if it needs more lines; and a layout
    with hyphens at one size is only kept when no size a little smaller sets
    the text with fewer of them: every hyphen counts as ``_HYPHEN_SIZE_COST``
    of the size (so one hyphen still beats one ``_SIZE_STEP`` step, three do
    not).  Arbitrary splits of a word wider than the region are only tried
    once the size has fallen to ``_FORCE_SPLIT_SCALE`` of ``max_size``.  When
    nothing fits even at ``min_size`` the text is flowed at ``min_size``
    through a rectangle as wide as the widest span (``fitted=False``), so a
    result is always produced.
    """
    words = text.split()
    height = float(spans.shape[0])
    if not words or height <= 0:
        return Typeset(max_size, max_size, [], True)
    measure = memoize_measure(measure)

    filled = np.flatnonzero(spans[:, 1] > spans[:, 0])
    if filled.size == 0:
        return _fallback(words, spans, top, measure, min_size, line_gap)
    region_top = top + float(filled[0])
    region_bottom = top + float(filled[-1]) + 1.0
    # Aim at the region's optical centre, computed once per block; a caller
    # that passes ``anchor_y`` keeps its own aim and its own horizontal
    # placement (free text belongs over the Japanese it replaces, not over
    # the middle of the shape it was given).
    optical = spans_centre(spans, top) if anchor_y is None else None
    anchor_x = None if optical is None else optical[0]
    if anchor_y is not None:
        centre_y = float(anchor_y)
    else:
        centre_y = optical[1] if optical is not None else (region_top + region_bottom) / 2.0
    centre_y = min(max(centre_y, region_top), region_bottom)
    if anchor_x is None:
        widest_span = float(np.max(spans[:, 1] - spans[:, 0]))
    else:
        # The widest line that any budget can supply, which under symmetric
        # budgets is narrower than the widest row: handing the search the raw
        # value would let it believe in a width no row can actually offer.
        widest_span = float(max(0.0, 2.0 * np.max(
            np.minimum(anchor_x - spans[:, 0], spans[:, 1] - anchor_x))))
    sp = _Spans(spans, top, anchor_x)

    size0 = max(max_size, min_size)
    size = size0
    best: Optional[Typeset] = None
    best_score = float("inf")
    while True:
        allow_force = size <= _FORCE_SPLIT_SCALE * max_size + 1e-6 or size <= min_size + 1e-6
        result = _typeset_at(words, text, size, measure, sp, region_top, region_bottom, centre_y, anchor_x,
                             widest_span, line_gap, lang, allow_force)
        if result is not None:
            off = _off_centre_em(result, centre_y)
            score = (1.0 - size / size0) + _HYPHEN_SIZE_COST * result.hyphens + _OFF_CENTRE_SIZE_COST * off
            if score < best_score:
                best, best_score = result, score
            if result.hyphens == 0 and off <= 0.0:
                break  # hyphen-free and centred: no smaller size can score better
        if size <= min_size:
            break
        next_size = max(size * _SIZE_STEP, min_size)
        if best is not None and (1.0 - next_size / size0) >= best_score:
            break  # shrinking further costs more than the hyphens saved
        size = next_size
    if best is not None:
        return best
    return _fallback(words, spans, top, measure, min_size, line_gap)


def _off_centre_em(ts: Typeset, anchor_y: float) -> float:
    """How far ``ts`` inks from ``anchor_y``, in ems of its own size, past
    the ``ANCHOR_SLACK_EM`` a centred block is allowed; 0 when it is centred.
    The size loop pays ``_OFF_CENTRE_SIZE_COST`` of the size per em of it."""
    if not ts.lines or ts.size <= 0:
        return 0.0
    glyph_h = ts.glyph_h if ts.glyph_h > 0 else ts.line_h
    centre = ink_centre(ts, ink_offsets(ts.text, glyph_h))
    return max(0.0, abs(centre - anchor_y) / ts.size - ANCHOR_SLACK_EM)


def _typeset_at(
    words: Sequence[str],
    text: str,
    size: float,
    measure: Measure,
    sp: _Spans,
    region_top: float,
    region_bottom: float,
    centre_y: float,
    anchor_x: Optional[float],
    widest_span: float,
    line_gap: float,
    lang: str,
    allow_force: bool,
) -> Optional[Typeset]:
    """The best layout of ``words`` at exactly ``size`` (see :func:`typeset`):
    hyphen-free first, then hyphenated; the fewest lines that flow and land
    on the anchor, then widened while the block is banner-shaped.  The
    vertical search lives in :class:`.anchor.Placement`.  None when nothing
    fits."""
    glyph_h = measure("Mg", size)[1]
    line_h = line_pitch(text, glyph_h, line_gap)
    ink = ink_offsets(text, glyph_h)
    span_h = region_bottom - region_top
    ink_h1 = ink[1] - ink[0]  # inked height of a single line
    max_lines = int((span_h - ink_h1) // line_h) + 1 if span_h >= ink_h1 else 0
    flow = _Flower(words, size, measure, lang)

    def finish(widths: np.ndarray, centres: np.ndarray, block_top: float,
               flowed: Tuple[List[str], int, int]) -> Typeset:
        return _finish(flow, line_h, glyph_h, widths, centres, block_top, flowed, anchor_x)

    slack = ANCHOR_SLACK_EM * size
    search = Placement(flow, sp.budgets, finish, line_h=line_h, ink=ink,
                       region=(region_top, region_bottom), anchor_y=centre_y,
                       slack=slack, width_quantum=_WIDTH_QUANTUM)
    for hyphenate in (False, True):
        found = search.search(max_lines, widest_span, slack,
                              hyphenate=hyphenate, allow_force=allow_force)
        if found is not None:
            return found
    return None


def lobe_split_row(spans: np.ndarray) -> Optional[int]:
    """Row index (into ``spans``) where a region's second lobe begins: the
    first filled row after the span centre jumps by more than
    ``_LOBE_JUMP_FRAC`` of the narrower of the two rows' widths.  Two joined
    boxes share a few rows where the run spans both (the joint); a second
    jump within ``_LOBE_JOINT_FRAC`` of the region's height ends that joint
    and the lower lobe starts there.  None when the region has one lobe, or
    more than two."""
    filled = np.flatnonzero(spans[:, 1] > spans[:, 0])
    if filled.size < 2:
        return None
    centres = (spans[filled, 0] + spans[filled, 1]) / 2.0
    widths = spans[filled, 1] - spans[filled, 0]
    jump = np.abs(np.diff(centres)) > _LOBE_JUMP_FRAC * np.minimum(widths[1:], widths[:-1])
    rows = np.flatnonzero(jump)
    if rows.size == 1:
        split = int(rows[0] + 1)
    elif rows.size == 2 and rows[1] - rows[0] <= max(2, _LOBE_JOINT_FRAC * filled.size):
        # The joint rows in between belong to neither lobe; both are
        # short, and a line covering them takes the narrower budget anyway.
        split = int(rows[1] + 1)
    else:
        return None
    # Both lobes must be real boxes, not a tail or a rounded corner's rows.
    min_rows = max(3, int(_LOBE_MIN_ROWS_FRAC * filled.size))
    if split < min_rows or filled.size - split < min_rows:
        return None
    return int(filled[split])


def typeset_lobes(
    text: str,
    spans: np.ndarray,
    top: float,
    measure: Measure,
    *,
    max_size: float,
    min_size: float = 6.0,
    line_gap: float = _LINE_GAP,
    lang: str = "en",
) -> Typeset:
    """:func:`typeset`, but when the region is two joined boxes
    (:func:`lobe_split_row`) and ``text`` has an ellipsis boundary
    (``"... ..."``), the sentence is split there: the first part is set in
    the upper lobe and the rest in the lower one, both at one common size,
    each centred in its box.  The split is used only when that size is at
    least ``_LOBE_MIN_SCALE`` of the size the continuous flow reaches."""
    joint = typeset(text, spans, top, measure, max_size=max_size, min_size=min_size, line_gap=line_gap, lang=lang)
    row = lobe_split_row(spans)
    m = _ELLIPSIS_BOUNDARY.search(text)
    if row is None or m is None or not joint.fitted:
        return joint
    part1, part2 = text[: m.end()].strip(), text[m.end() :].strip()
    if not part1 or not part2:
        return joint
    kw = dict(min_size=min_size, line_gap=line_gap, lang=lang)
    a = typeset(part1, spans[:row], top, measure, max_size=max_size, **kw)
    b = typeset(part2, spans[row:], top + row, measure, max_size=a.size, **kw)
    if b.size < a.size - 1e-6:
        a = typeset(part1, spans[:row], top, measure, max_size=b.size, **kw)
    size = min(a.size, b.size)
    if not (a.fitted and b.fitted and a.lines and b.lines) or abs(a.size - b.size) > 0.01 * size:
        return joint
    if size < _LOBE_MIN_SCALE * joint.size:
        return joint
    return Typeset(size, a.line_h, a.lines + b.lines, True, a.hyphens + b.hyphens, a.glyph_h)


def _fallback(
    words: Sequence[str], spans: np.ndarray, top: float, measure: Measure, min_size: float, line_gap: float
) -> Typeset:
    """Nothing fits: flow at ``min_size`` through the widest rectangle and
    centre it on the region (the result may overflow; ``fitted`` is False)."""
    widest = float(np.max(spans[:, 1] - spans[:, 0])) if spans.size else 0.0
    filled = spans[spans[:, 1] > spans[:, 0]] if spans.size else spans
    cx = float(np.mean(filled[:, :2])) if filled.size else (float(np.mean(spans[:, :2])) if spans.size else 0.0)
    text = " ".join(words)
    glyph_h = measure("Mg", min_size)[1]
    line_h = line_pitch(text, glyph_h, line_gap)
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
    height = float(spans.shape[0])
    block_top = top + (height - ((len(lines) - 1) * line_h + glyph_h)) / 2.0
    placed = [PlacedLine(l, cx, block_top + i * line_h, measure(l, min_size)[0]) for i, l in enumerate(lines)]
    return Typeset(min_size, line_h, placed, False, 0, glyph_h)


# ------------------------------------------------------------ line breaking
WidthOf = Callable[[int, int], float]


def span_widths(words: Sequence[str], size: float, measure: Measure) -> WidthOf:
    """``width(i, j)`` of the line made of ``words[i:j]`` (``j`` exclusive)
    at ``size``, measured as the joined string (so kerning is honoured) and
    cached."""
    cache: Dict[Tuple[int, int], float] = {}

    def width(i: int, j: int) -> float:
        key = (i, j)
        w = cache.get(key)
        if w is None:
            w = cache[key] = measure(" ".join(words[i:j]), size)[0]
        return w

    return width


# A sentence ends at one of these, possibly behind a closing quote or bracket.
_SENTENCE_END = ".!?…"
_SENTENCE_CLOSERS = "\"')]}»”’"


def sentence_breaks(tokens: Sequence[str]) -> List[int]:
    """Indexes of the tokens that end a sentence, and so end a line.

    A letterer does not trade sense against rectangle-ness - they break the
    block at the beat first and balance what is left inside each piece.  The
    release sets `NO... / YOU DON'T / UNDERSTAND, / MAKI.` and `WHO... /
    ...ARE / YOU?!`, keeping the interjection whole on its own line;
    minimising raggedness alone gives `NO. YOU / DON'T / UNDERSTAND, / MAKI.`,
    which reads as a stumble.

    The last token is never returned - the block already ends there - and nor
    is a hyphenated word head, which ends its line for a different reason and
    is forced separately."""
    out: List[int] = []
    for i, token in enumerate(tokens[:-1]):
        stripped = token.rstrip(_SENTENCE_CLOSERS)
        if stripped and not stripped.endswith("-") and stripped[-1] in _SENTENCE_END:
            out.append(i)
    return out


# A phrase ends at one of these too, but weakly: a letterer will break there when the
# shapes are close, and will not distort the block to do it (that is what a sentence end
# gets, through `forced`).
_PHRASE_END = ",;:—–"
# Never stranded at the end of a line, away from the noun they introduce.
_ARTICLES = frozenset({"a", "an", "the"})
#
# THE CONJUNCTION/PREPOSITION RULE IS NOT HERE, AND THAT IS DELIBERATE.
# The obvious companion rule - reward breaking *before* a conjunction or preposition, and
# penalise stranding one at a line end - was built, measured against the release's own
# lettering on every block it moved, and **refuted**: 2 blocks better, 3 worse, 1 unchanged.
# VIZ is perfectly willing to strand a short function word when it balances the block
# better, and does so in `SPEECH OR / JUGON?`, `SPECIFICALLY FOR / ENTERTAINMENT` and
# `ON THE DAY IN / QUESTION.` - all three of which this rule "corrected" away from the
# release.  See the perf note, tag `sense`.  What survives below are the two rules the
# release does not contradict.
# One unit of sense cost, as a fraction of the line's target width squared.  The balance
# term is (target - width)**2, so 0.04 means a preferred break wins only when it costs the
# rectangle less than 20 % of the target in width - i.e. when the shapes really are
# comparable, which is the letterer's own rule.  It is deliberately far below the 0.40 a
# previous attempt needed to flip a SENTENCE break: sentence ends are handled by `forced`
# before this is consulted, so all this has to do is order the breaks that remain.
_SENSE_SPAN = 0.04


def sense_costs(tokens: Sequence[str]) -> List[float]:
    """Cost of ending a line before each token, in units of ``_SENSE_SPAN``.

    Negative is a break a letterer prefers, positive one they avoid.  Index ``j``
    is the cost of the break that puts ``tokens[j]`` at the start of the next
    line; index 0 is never a break and is always 0.

    Two rules, both of which the release's own lettering supports (or at least
    never contradicts on the 50-page sample):

    * **after a phrase end** (``,`` ``;`` ``:`` em dash) - a natural pause, so a
      small reward;
    * **after an article** - a penalty, because ``THE / GIRL`` separates a word
      from the noun it introduces and no letterer sets that on purpose.

    A third rule - reward breaking before a conjunction or preposition - was
    built and **refuted by the release**; see the comment on :data:`_ARTICLES`.

    A token is judged on its letters only, so quotation marks and brackets do
    not hide the word inside them.
    """
    costs = [0.0] * (len(tokens) + 1)

    def word(token: str) -> str:
        return "".join(c for c in token if c.isalpha()).lower()

    for j in range(1, len(tokens)):
        before = tokens[j - 1]
        cost = 0.0
        stripped = before.rstrip(_SENTENCE_CLOSERS)
        if stripped and stripped[-1] in _PHRASE_END:
            cost -= 1.0
        if word(before) in _ARTICLES:
            cost += 2.0
        costs[j] = cost * _SENSE_SPAN
    return costs


def _line_limits(count: int, forced: Sequence[int]) -> List[int]:
    """``limit[i]``: exclusive upper bound of the end of a line starting at
    word ``i`` so that no forced break (a line must end after word ``k`` for
    each ``k`` in ``forced``, e.g. a hyphenated word head) is skipped."""
    limit = [count] * (count + 1)
    for k in sorted(set(forced), reverse=True):
        if 0 <= k < count:
            for i in range(0, k + 1):
                limit[i] = min(limit[i], k + 1)
    return limit


def min_max_width(count: int, n: int, width: WidthOf, forced: Sequence[int] = ()) -> Optional[float]:
    """Smallest width ``W`` such that ``count`` words can be broken into at
    most ``n`` lines each no wider than ``W`` (no hyphenation), honouring
    ``forced`` breaks (see :func:`_line_limits`).  None when ``count`` is 0
    or the forced breaks cannot be met in ``n`` lines."""
    if count <= 0:
        return None
    n = max(1, min(n, count))
    inf = float("inf")
    limit = _line_limits(count, forced)
    # best[k][i]: minimal max line width to set words[i:] in exactly k lines.
    best = [inf] * (count + 1)
    best[count] = 0.0
    for i in range(count - 1, -1, -1):
        best[i] = width(i, count) if limit[i] >= count else inf
    for _k in range(2, n + 1):
        nxt = [inf] * (count + 1)
        nxt[count] = 0.0
        for i in range(count - 1, -1, -1):
            b = best[i]  # fewer lines is always allowed (a line may be dropped)
            for j in range(i + 1, min(count, limit[i] + 1)):
                w = width(i, j)
                if w >= b:
                    break  # the first line alone is already worse; longer only grows
                v = max(w, best[j])
                if v < b:
                    b = v
            nxt[i] = b
        best = nxt
    return best[0] if best[0] < inf else None


def balanced_breaks(
    count: int,
    n: int,
    max_w: Union[float, Sequence[float]],
    width: WidthOf,
    targets: Optional[Sequence[float]] = None,
    forced: Sequence[int] = (),
    sense: Optional[Sequence[float]] = None,
) -> Optional[List[int]]:
    """Break ``count`` words into at most ``n`` lines no wider than ``max_w``
    (one limit, or one per line), minimising the summed squared shortfall
    of each line against its target width (``targets[k]`` for line k,
    default the line's limit): the letterer's "balanced" block.  ``forced``
    lists words after which a line must end.  Returns the list of line
    start indexes (plus the final ``count``), or None when a word alone is
    wider than its line's limit.

    ``sense`` optionally carries one cost per break position (``sense[j]`` is
    charged when a line ends just before word ``j``), in the units
    :func:`sense_costs` produces: a *fraction of the line's target width
    squared*, so it is comparable with the balance term on any page at any
    size.  Negative is a preferred break.  It is deliberately small - the
    rectangle term still decides the shape, and a sense break only wins when
    the shapes are comparable."""
    if count <= 0:
        return None
    n = max(1, min(n, count))
    inf = float("inf")
    limit = _line_limits(count, forced)
    maxes = [float(m) for m in max_w] if isinstance(max_w, (list, tuple, np.ndarray)) else [float(max_w)] * n
    maxes += [maxes[-1]] * (n - len(maxes))
    tgt = list(targets) if targets is not None else list(maxes[:n])
    tgt += maxes[len(tgt) : n]
    # cost[k][i]: minimal cost to set words[i:] in exactly k lines (k = 1..n);
    # when exactly n lines are used, the line set at level k is line n - k.
    cost = [[inf] * (count + 1) for _ in range(n + 1)]
    nxt_break = [[count] * (count + 1) for _ in range(n + 1)]
    for k in range(1, n + 1):
        target = tgt[n - k]
        cap = maxes[n - k]
        for i in range(count - 1, -1, -1):
            best_c, best_j = inf, count
            if k == 1:
                w = width(i, count)
                if w <= cap + 1e-6 and limit[i] >= count:
                    best_c = (target - w) ** 2
            else:
                for j in range(i + 1, min(count, limit[i] + 1)):
                    w = width(i, j)
                    if w > cap + 1e-6:
                        break
                    rest = cost[k - 1][j]
                    if rest == inf:
                        continue
                    c = (target - w) ** 2 + rest
                    if sense is not None and j < len(sense):
                        # Scaled by this line's own target, so one "unit" of sense means
                        # the same thing on a wide caption and a narrow balloon.
                        c += float(sense[j]) * target * target
                    if c < best_c:
                        best_c, best_j = c, j
            cost[k][i], nxt_break[k][i] = best_c, best_j
    # Exactly n lines when possible, else the largest feasible count below.
    k_best = next((k for k in range(n, 0, -1) if cost[k][0] < inf), None)
    if k_best is None:
        return None
    starts = [0]
    i, k = 0, k_best
    while i < count:
        i = nxt_break[k][i]
        k -= 1
        starts.append(i)
    return starts
