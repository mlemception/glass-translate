"""Defect B - the lettering must be optically centred in the balloon.

``render/typeset.typeset`` used to centre the block on the *bounding box of
the filled row spans* (the first and last row with any run in it) and then
slide it in quarter-line steps until a line count flowed.  A real balloon is
not symmetric about that box - a tail, a flat side or a joined neighbour
drags the box one way - and the search kept the first line count that fitted
*anywhere*, so a block that only flowed near one end won over a taller one
that would have flowed centred.

The rule asserted here: the block's ink centroid sits on the interior's
optical centre, the middle of the largest circle inscribed in it
(``render/anchor.interior_centre``, the same definition
``demo/typeset_metrics.interior_centre`` scores the render against), within
``CENTRE_TOLERANCE_EM``.  Both axes count: the distance measured is 2-D.

Containment is checked here as a guard rather than as the defect: the flow
takes the *intersection* of each line's row spans, so glyphs never leave the
region the block was given.  When lettering does cross a balloon outline on a
real page it is because that region is not the balloon - see
``tests/test_layout_shared_bubbles.py``.

Measures come from a fake linear measurer, as in ``tests/test_typeset.py``:
no fonts, deterministic.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from glasstranslate.core.types import Rect, Segment
from glasstranslate.render import build_blocks, typeset_block

T = importlib.import_module("glasstranslate.render.typeset")
A = importlib.import_module("glasstranslate.render.anchor")

FIXTURES = Path(__file__).resolve().parent / "fixtures"
# How far the block's ink centroid may sit from the interior's optical
# centre.  This is the acceptance bound of ``centre_offset_em`` in
# ``demo/typeset_metrics``: at or below 0.25 source ems.  Every case in this
# module is a single balloon interior, so none of them needs the looser 0.6
# em hard cap that a genuinely awkward (two-lobed) interior is allowed.
CENTRE_TOLERANCE_EM = 0.25
TEXT = "I THOUGHT I TOLD YOU TO STAY AWAY FROM THIS PLACE"

# ``1ja_tailed_balloon.png``: the OCR lines of that crop (text, x, y, w, h, confidence).
LINES_1JA = [
    ("ウケるね（笑）", 45, 25, 30, 109, 0.899),
    ("マジで？", 68, 26, 27, 76, 0.976),
]


def fake_measure(text: str, size: float) -> tuple[float, float]:
    """Every glyph is half an em wide; the line box is 1.15 em."""
    return 0.5 * size * len(text), 1.15 * size


def quad(r: Rect) -> np.ndarray:
    return np.array([[r.x, r.y], [r.x2, r.y], [r.x2, r.y2], [r.x, r.y2]], np.float32)


def fixture(name: str) -> np.ndarray:
    gray = cv2.imread(str(FIXTURES / name), cv2.IMREAD_GRAYSCALE)
    assert gray is not None, f"missing fixture {name}"
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def ink_mask(ts, shape) -> np.ndarray:
    """Upper bound of the lettering's glyph pixels: every placed line's inked
    box (cap top to deepest descender, see ``typeset.ink_offsets``).  The flow
    guarantees each such box fits its rows' spans, so a layout that keeps the
    region also keeps this mask inside it."""
    out = np.zeros(shape[:2], bool)
    glyph_h = ts.glyph_h if ts.glyph_h > 0 else ts.line_h
    top, bottom = T.ink_offsets(ts.text, glyph_h)
    for line in ts.lines:
        y0, y1 = int(np.floor(line.top + top)), int(np.ceil(line.top + bottom))
        x0, x1 = int(np.floor(line.cx - line.width / 2.0)), int(np.ceil(line.cx + line.width / 2.0))
        out[max(0, y0): max(0, y1), max(0, x0): max(0, x1)] = True
    return out


def inscribed_centre(region: np.ndarray) -> tuple[float, float]:
    """Centre of the largest circle inscribed in ``region`` (the mean of the
    distance transform's peak pixels).  Unlike the bounding-box centre it
    ignores a tail, and unlike the area centroid it is not dragged by one."""
    dt = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt >= dt.max() - 1e-6)
    return float(xs.mean()), float(ys.mean())


def centroid(mask: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    assert m["m00"] > 0
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def offset_em(ts, region: np.ndarray, em: float) -> float:
    """``centre_offset_em``: the distance from the block's ink centroid to
    ``region``'s optical centre, in ems."""
    dx, dy = centroid(ink_mask(ts, region.shape))
    cx, cy = inscribed_centre(region)
    return float(np.hypot(dx - cx, dy - cy)) / em


def region_of(block, shape) -> np.ndarray:
    st = block.style
    out = np.zeros(shape[:2], bool)
    box = st.layout_box
    out[box.y: box.y2, box.x: box.x2] = True if st.layout_mask is None else st.layout_mask
    return out


def tailed_balloon():
    """The one bubble block of ``1ja_tailed_balloon.png`` (page 1ja cropped to
    585, 810, 135, 160: a balloon whose tail runs off the bottom edge)."""
    img = fixture("1ja_tailed_balloon.png")
    segs = [Segment(t, quad(Rect(x, y, w, h)), c) for t, x, y, w, h, c in LINES_1JA]
    blocks = build_blocks(img, segs, segs)
    bubbles = [b for b in blocks if b.style.in_bubble]
    assert len(bubbles) == 1, "the fixture holds exactly one bubble block"
    return img, bubbles[0]


def tail_mask() -> np.ndarray:
    """An ellipse with a tail running off the bottom: the filled rows reach
    the tip, so their middle sits well below the balloon's own."""
    mask = np.zeros((220, 160), np.uint8)
    cv2.ellipse(mask, (80, 70), (70, 60), 0, 0, 360, 1, -1)
    cv2.fillPoly(mask, [np.array([[60, 110], [100, 110], [55, 210]], np.int32)], 1)
    return mask.astype(bool)


def lobed_mask() -> np.ndarray:
    """Two balloons joined by a narrow throat: the shape ``render/layout``
    leaves behind when it cannot split a shared patch of paper (defect A).
    The bounding box of the rows is centred on the throat, which is inside
    neither balloon."""
    mask = np.zeros((240, 200), np.uint8)
    cv2.ellipse(mask, (70, 70), (60, 55), 0, 0, 360, 1, -1)
    cv2.ellipse(mask, (140, 175), (55, 50), 0, 0, 360, 1, -1)
    cv2.line(mask, (95, 115), (120, 130), 1, 7)
    return mask.astype(bool)


# ------------------------------------------------------------ centring
# Three texts through the one real balloon, with the offset each is expected
# to reach.  ``TEXT`` fills the balloon with three lines and lands about one
# quarter-line ladder rung out (3.8 px against a 3.4 px rung: the flow
# re-breaks between rungs, so the block cannot be parked exactly on the
# anchor), which is most of the 4.3 px ``CENTRE_TOLERANCE_EM`` allows at this
# em.  The other two have room to spare and are held to ``TIGHT_EM``, so the
# real-page evidence does not rest on the one near-threshold number.
TIGHT_EM = 0.05
REAL_CASES = [
    (TEXT, CENTRE_TOLERANCE_EM),
    ("HM?", TIGHT_EM),
    ("SUPERCALIFRAGILISTIC EXPIALIDOCIOUS EXTRAORDINARY MAGNIFICENT WORDS", TIGHT_EM),
]


@pytest.mark.parametrize("text, bound", REAL_CASES)
def test_lettering_is_centred_on_the_balloon_interior(text: str, bound: float) -> None:
    """A real tailed balloon, through the whole layout: the block lands on
    the interior's optical centre, not on the middle of the rows' box, which
    the tail drags some 3 px down and which the slide then over-corrected by
    35 px upwards.  One line, three lines and six lines of it, so a change to
    ``ink_offsets``, to the line pitch or to ``fake_measure`` cannot slip
    through on the one case that has no headroom."""
    img, block = tailed_balloon()
    ts = typeset_block(text, block.style, fake_measure)
    assert ts.lines and ts.fitted
    offset = offset_em(ts, region_of(block, img.shape), block.em_px)
    assert offset <= bound, (
        f"the block sits {offset:.3f} em from the balloon's optical centre ({len(ts.lines)} lines)"
    )


def test_a_tail_does_not_pull_the_block_off_centre() -> None:
    """The mechanism on its own, without the layout: an ellipse with a tail.
    The filled rows run to the tip of the tail, so their middle sits well
    below the balloon's own; the block must still ink the balloon's centre,
    on both axes."""
    mask = tail_mask()
    spans = T.mask_spans(mask, Rect(0, 0, 160, 220))
    ts = T.typeset("A FEW WORDS OF DIALOGUE HERE", spans, 0.0, fake_measure, max_size=16.0)
    assert ts.lines and ts.fitted
    dx, dy = centroid(ink_mask(ts, mask.shape))
    cx, cy = inscribed_centre(mask)
    assert abs(dy - cy) <= CENTRE_TOLERANCE_EM * 16.0, (
        f"the block's ink centre is {dy:.1f}, the balloon's inscribed centre {cy:.1f}"
    )
    assert abs(dx - cx) <= CENTRE_TOLERANCE_EM * 16.0, (
        f"the block's ink centre is {dx:.1f}, the balloon's inscribed centre {cx:.1f}"
    )


def test_the_anchor_is_not_dragged_into_an_empty_lobe() -> None:
    """Two balloons joined by a throat: the optical centre is the middle of
    the larger balloon, never the throat the bounding box points at, and the
    block sets there."""
    mask = lobed_mask()
    spans = T.mask_spans(mask, Rect(0, 0, 200, 240))
    ts = T.typeset("A FEW WORDS OF DIALOGUE HERE", spans, 0.0, fake_measure, max_size=16.0)
    assert ts.lines and ts.fitted
    dx, dy = centroid(ink_mask(ts, mask.shape))
    cx, cy = inscribed_centre(mask)
    assert float(np.hypot(dx - cx, dy - cy)) <= CENTRE_TOLERANCE_EM * 16.0


def test_a_caller_anchor_still_wins() -> None:
    """Free text is deliberately anchored on the Japanese it replaces, so an
    ``anchor_y`` from the caller beats the region's optical centre; only the
    default changed."""
    mask = tail_mask()
    spans = T.mask_spans(mask, Rect(0, 0, 160, 220))
    kw = dict(max_size=12.0, min_size=7.0)
    free = T.typeset("A FEW WORDS OF DIALOGUE HERE", spans, 0.0, fake_measure, anchor_y=40.0, **kw)
    bubble = T.typeset("A FEW WORDS OF DIALOGUE HERE", spans, 0.0, fake_measure, **kw)
    assert free.lines and free.fitted and bubble.lines and bubble.fitted
    _, free_y = centroid(ink_mask(free, mask.shape))
    _, bubble_y = centroid(ink_mask(bubble, mask.shape))
    _, cy = inscribed_centre(mask)
    assert abs(free_y - 40.0) < abs(free_y - cy)  # aimed where the caller said
    assert free_y < bubble_y - 5.0  # and well above the block that was not


def test_the_optical_centre_does_not_depend_on_the_margin_around_it() -> None:
    """``anchor.interior_centre`` pads before its distance transform, so a
    region that runs to the edge of the array scores the same centre as the
    same region with room around it.  Without the pad the transform keeps
    rising past the border and puts the peak *on* it (4ja block 8, a dark
    panel at the foot of a 1200-row page, scored ``(241, 1199)`` instead of
    ``(171, 933)``), and ``demo/typeset_metrics.interior_centre`` - which
    scores us - pads for the same reason."""
    disc = np.zeros((200, 120), np.uint8)
    cv2.ellipse(disc, (60, 150), (62, 52), 0, 0, 360, 1, -1)
    tight = disc[110:, :].astype(bool)  # the ellipse cut off on three sides
    assert tight[0].any() and tight[-1].any() and tight[:, 0].any()  # it touches them
    cut = A.interior_centre(tight)
    roomy = A.interior_centre(np.pad(tight, 40))
    assert cut is not None and roomy is not None
    assert cut[0] == pytest.approx(roomy[0] - 40, abs=0.5)
    assert cut[1] == pytest.approx(roomy[1] - 40, abs=0.5)


def test_an_empty_row_is_a_gap_not_a_join() -> None:
    """``anchor.spans_centre`` rasterises every row between the first and the
    last inked one, empty ones included.  Compacting the inked rows instead
    would close the gap up and put the anchor inside it: here the region is
    eight rows, four empty, then ten, and the centre belongs in the taller
    lower piece (y 15), not at y 9 where the compacted raster puts it.  No
    layout region reaching :func:`~glasstranslate.render.typeset.typeset`
    today has an interior empty row, so this is a precondition rather than a
    live bug - and the way interiors are cut is exactly the thing that
    changes."""
    spans = np.zeros((20, 2))
    spans[0:8] = (0.0, 10.0)
    spans[10:20] = (0.0, 10.0)
    centre = A.spans_centre(spans, 0.0)
    assert centre is not None
    assert centre[1] > 12.0, f"the anchor landed at y {centre[1]:.1f}, inside the gap"
    assert centre == pytest.approx((5.0, 15.0), abs=0.5)


def test_rows_are_taken_from_the_balloons_own_column() -> None:
    """``mask_spans`` picks each row's run on the mask's optical column, not
    on the middle of the box the layout cut around it.  A balloon pushed to
    one side of its box, with a sliver of a neighbour on the other, would
    otherwise be flowed and anchored in the sliver.  The sliver has to sit
    *on* the rect centre column for this to discriminate: put it out at the
    edge and the nearest-run fallback picks the balloon either way."""
    mask = np.zeros((160, 300), np.uint8)
    cv2.circle(mask, (70, 80), 60, 1, -1)  # the balloon, on the left
    mask[60: 100, 145: 165] = 1  # a sliver of the neighbour, over the box's middle
    box = Rect(0, 0, 300, 160)
    assert mask[80, int(box.w / 2)], "the sliver must cover the rect centre column, or nothing is proved"
    spans = T.mask_spans(mask.astype(bool), box)
    row = spans[80]
    assert row[0] < 70 < row[1], f"row 80 was taken from {row}, not from the balloon"
    centre = A.spans_centre(spans, 0.0)
    assert centre is not None and abs(centre[0] - 70) <= 2.0 and abs(centre[1] - 80) <= 2.0


def band_balloon() -> np.ndarray:
    """A balloon wide only in a short band: a flat belly 230 px across and 64
    high, on a neck a quarter of that wide.  Its optical centre is in the
    belly, and a block set at the ceiling is taller than the belly, so it can
    only sit up the neck - the shape that makes ``typeset`` buy centring with
    a size step (``_OFF_CENTRE_SIZE_COST``)."""
    mask = np.zeros((240, 260), np.uint8)
    cv2.ellipse(mask, (130, 160), (115, 32), 0, 0, 360, 1, -1)  # the belly
    cv2.ellipse(mask, (130, 80), (32, 75), 0, 0, 360, 1, -1)  # the neck
    return mask.astype(bool)


def test_a_smaller_size_is_bought_when_the_big_one_cannot_be_centred() -> None:
    """The largest size that flows is not always the one to set.  In
    :func:`band_balloon` the ceiling gives a seven line block that only fits
    up the neck, 1.21 em above the balloon's optical centre; one
    ``_SIZE_STEP`` smaller it is four lines that sit in the belly, 0.08 em
    off.  ``typeset``'s size loop prices that trade at
    ``_OFF_CENTRE_SIZE_COST`` per em off centre, and this is the only test of
    it: with the constant at 0 the block comes back at the ceiling and 1.21
    em out, failing both assertions below.

    Both halves are asserted on purpose - the size must actually drop, so a
    mechanism that fires without buying anything fails too."""
    mask = band_balloon()
    spans = T.mask_spans(mask, Rect(0, 0, 260, 240))
    max_size = 22.0
    ts = T.typeset("THIS BALLOON IS WIDE ONLY NEAR THE BOTTOM OF IT", spans, 0.0,
                   fake_measure, max_size=max_size, min_size=7.0)
    assert ts.lines and ts.fitted
    assert ts.size < max_size, "the block was set at the ceiling, where it cannot be centred"
    dx, dy = centroid(ink_mask(ts, mask.shape))
    cx, cy = inscribed_centre(mask)
    offset = float(np.hypot(dx - cx, dy - cy)) / max_size
    assert offset <= CENTRE_TOLERANCE_EM, (
        f"the block sits {offset:.2f} em from the balloon's optical centre at size {ts.size:.2f}"
    )
    assert int((ink_mask(ts, mask.shape) & ~mask).sum()) == 0  # and it stayed inside


# ------------------------------------------------------------ containment guard
def test_lettering_never_leaves_the_region_it_was_given() -> None:
    """A guard, not the defect: whatever the centring fix does, no glyph pixel
    may end up outside the layout region, and the block still fits."""
    img, block = tailed_balloon()
    region = region_of(block, img.shape)
    for text in (TEXT, "HM?", "SUPERCALIFRAGILISTIC EXPIALIDOCIOUS EXTRAORDINARY MAGNIFICENT WORDS"):
        ts = typeset_block(text, block.style, fake_measure)
        assert ts.fitted, f"the fitter gave up on {text!r}"
        outside = int((ink_mask(ts, img.shape) & ~region).sum())
        assert outside == 0, f"{outside} glyph px outside the balloon for {text!r}"
