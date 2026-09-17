"""A line may sit slightly off the block's axis to reach width the strictly
centred budget cannot offer.

``typeset._line_budget`` cuts each line's row intersection symmetric about the
block's optical axis, so the axis is the budget's centre by construction and no
line can be dragged off it (``test_typeset_symmetric_budget``).  That rule is
*exact*, not cautious: for a line of width ``W`` centred on the axis,
``W <= 2 * min(anchor_x - left, right - anchor_x)`` is the whole of what fits.

The cost lands unevenly.  A balloon drawn around a vertical Japanese column is
lopsided about its own optical centre, and over the 50-page corpus the
symmetric width is 0.98 of the widest row on blocks set at the ceiling but only
0.82 on blocks that had to shrink - the loss falls on exactly the blocks that
could least afford it.  Moving the axis instead was measured and refuted: it
trades width for height, because rows near the top and bottom of the balloon
stop being symmetric about the new axis and drop out of the usable region, so
fewer lines fit and the size falls (one block went 61 -> 102 px of width and
0.748 -> 0.484 of its ceiling).

So the term that moves is the centring itself.  A line may sit up to
``_AXIS_SLACK_EM`` ems off the axis, which widens its budget to

    W = min(right - left, 2 * (half + slack))

capped at the true intersection - so containment is unchanged - and narrower
lines are slid back onto the axis by ``anchor.line_centre``, so the wander is
only paid on the lines that spend it.

Synthetic spans only; no corpus imagery enters the repository.
"""
from __future__ import annotations

import importlib

import numpy as np

from glasstranslate.render.anchor import line_centre

from tests.test_typeset_symmetric_budget import lopsided_spans

T = importlib.import_module("glasstranslate.render.typeset")


def test_slack_widens_the_budget_on_a_lopsided_region() -> None:
    """The defect: the centred budget is far narrower than the rows really are.

    Fails against the old code, where ``budgets`` took no ``slack`` at all and
    every width was exactly ``2 * min(reach)``.
    """
    spans = lopsided_spans()
    sp = T._Spans(spans, 0.0, anchor_x=50.0)
    tops = np.arange(0, 70, 7, dtype=np.float64)
    tight, _ = sp.budgets(tops, tops + 6.0)
    loose, _ = sp.budgets(tops, tops + 6.0, 8.0)
    ok = tight > 0
    assert ok.sum() >= 5
    assert (loose[ok] >= tight[ok] - 1e-9).all(), "slack made a budget narrower"
    assert (loose[ok] > tight[ok] + 1e-9).any(), "slack bought no width anywhere"


def test_slack_never_leaves_the_region() -> None:
    """The invariant the whole idea stands on: ``c_contain`` passes at 0 only.

    However much slack is asked for, the widened budget stays inside the rows'
    own intersection - so no line can be placed outside the balloon.
    """
    spans = lopsided_spans()
    anchor = 50.0
    sp = T._Spans(spans, 0.0, anchor_x=anchor)
    plain = T._Spans(spans, 0.0)
    tops = np.arange(0, 70, 3, dtype=np.float64)
    pw, pc = plain.budgets(tops, tops + 6.0)
    for slack in (0.0, 2.0, 8.0, 40.0, 1000.0):
        w, c = sp.budgets(tops, tops + 6.0, slack)
        for i in range(len(tops)):
            if w[i] <= 0:
                continue
            assert w[i] <= pw[i] + 1e-9, f"slack {slack} exceeded the intersection"
            assert c[i] - w[i] / 2 >= pc[i] - pw[i] / 2 - 1e-9, f"slack {slack} ran off the left"
            assert c[i] + w[i] / 2 <= pc[i] + pw[i] / 2 + 1e-9, f"slack {slack} ran off the right"


def test_a_line_is_never_further_off_the_axis_than_the_slack() -> None:
    """What the block pays.  A line filling its budget has no room to be slid
    back, so its centre is the budget's centre - and that is what bounds the
    wander to the slack asked for, rather than to the region's lopsidedness."""
    spans = lopsided_spans()
    anchor = 50.0
    sp = T._Spans(spans, 0.0, anchor_x=anchor)
    tops = np.arange(0, 70, 3, dtype=np.float64)
    for slack in (2.0, 8.0):
        w, c = sp.budgets(tops, tops + 6.0, slack)
        at = [line_centre(cc, ww, ww, anchor) for ww, cc in zip(w, c) if ww > 0]
        assert at, "no line had any room"
        assert max(abs(x - anchor) for x in at) <= slack + 1e-9


def test_a_narrower_line_is_slid_back_onto_the_axis() -> None:
    """The wander is self-limiting: only lines that spend the width pay for it.

    A line comfortably narrower than its budget still centres exactly on the
    axis, so a block whose lines all fit is where it was before.
    """
    spans = lopsided_spans()
    anchor = 50.0
    sp = T._Spans(spans, 0.0, anchor_x=anchor)
    tops = np.arange(0, 70, 3, dtype=np.float64)
    w, c = sp.budgets(tops, tops + 6.0, 8.0)
    narrow = [line_centre(cc, ww, 4.0, anchor) for ww, cc in zip(w, c) if ww > 12.0]
    assert len(narrow) >= 5
    np.testing.assert_allclose(narrow, anchor)


def test_zero_slack_is_the_shipped_symmetric_rule() -> None:
    """The control that makes every measurement above believable."""
    spans = lopsided_spans()
    anchor = 50.0
    sp = T._Spans(spans, 0.0, anchor_x=anchor)
    tops = np.arange(0, 70, 3, dtype=np.float64)
    w0, c0 = sp.budgets(tops, tops + 6.0)
    w1, c1 = sp.budgets(tops, tops + 6.0, 0.0)
    np.testing.assert_array_equal(w0, w1)
    np.testing.assert_array_equal(c0, c1)
    for t, w, c in zip(tops, w1, c1):
        assert (w, c) == T._line_budget(spans, t, t + 6.0, 0.0, anchor, 0.0)


def test_the_vectorised_budget_still_matches_the_reference_with_slack() -> None:
    """``_Spans.budgets`` is documented as bit-for-bit ``_line_budget``, and
    that has to keep holding on the new branch or the reference implementation
    stops being a reference."""
    spans = lopsided_spans()
    anchor = 50.0
    sp = T._Spans(spans, 0.0, anchor_x=anchor)
    tops = np.arange(0, 70, 3, dtype=np.float64)
    for slack in (0.0, 1.5, 8.0, 40.0):
        widths, centres = sp.budgets(tops, tops + 6.0, slack)
        for t, w, c in zip(tops, widths, centres):
            assert (w, c) == T._line_budget(spans, t, t + 6.0, 0.0, anchor, slack)


def test_free_text_is_untouched_by_slack() -> None:
    """No ``anchor_x`` means no axis to be off: the free-text and horizontal
    caption paths keep the whole intersection and its own centre, whatever
    slack is passed."""
    spans = lopsided_spans()
    plain = T._Spans(spans, 0.0)
    tops = np.arange(0, 70, 3, dtype=np.float64)
    base = plain.budgets(tops, tops + 6.0)
    for slack in (0.0, 8.0, 100.0):
        w, c = plain.budgets(tops, tops + 6.0, slack)
        np.testing.assert_array_equal(w, base[0])
        np.testing.assert_array_equal(c, base[1])


def fake_measure(text: str, size: float) -> tuple[float, float]:
    """Every glyph is half an em wide; the line box is 1.15 em.  Same fake as
    ``tests/test_typeset``: no fonts, no images."""
    return 0.5 * size * len(text), 1.15 * size


def drifting_balloon(n: int = 90, half: float = 35.0, drift: float = 0.45) -> np.ndarray:
    """An oval whose waist slides sideways as it descends - the shape a balloon
    takes when it is drawn around a vertical Japanese column that is not itself
    upright.  Its rows are wide, but they are lopsided about the region's
    optical centre, so a strictly centred line can use only part of them."""
    spans = np.zeros((n, 2))
    for i in range(n):
        t = (i - n / 2.0) / (n / 2.0)
        w = half * max(0.05, (1.0 - t * t) ** 0.5)
        centre = 60.0 + drift * i
        spans[i] = (centre - w, centre + w)
    return spans


def test_a_lopsided_balloon_holds_its_size_end_to_end() -> None:
    """The behavioural regression, through ``typeset`` itself.

    The budget tests above fail against the old code because ``budgets`` had no
    ``slack`` parameter - a signature failure, which proves nothing about what a
    reader sees.  This one fails on *behaviour*: the old rule cannot offer the
    width for two words at the ceiling on this shape, so it steps down the size
    ladder to 17.30 of a 20.00 ceiling.  With the slack the block holds the
    ceiling, in the same two lines.
    """
    spans = drifting_balloon()
    got = T.typeset("MMMMM MMMMM", spans, 0.0, fake_measure, max_size=20.0, min_size=6.0)
    assert got.fitted
    assert len(got.lines) == 2
    assert got.size == 20.0, f"the old rule reaches only 17.30 here; got {got.size:.2f}"


def test_a_lopsided_balloon_needs_one_line_fewer() -> None:
    """The same defect seen as line count rather than size: at the ceiling the
    old rule needs three lines for text the widened budget sets in two."""
    spans = drifting_balloon(120, 40.0, 0.6)
    got = T.typeset("AB CD EF GH", spans, 0.0, fake_measure, max_size=20.0, min_size=6.0)
    assert got.fitted and got.size == 20.0
    assert len(got.lines) == 2, f"the old rule takes 3 lines here; got {len(got.lines)}"


def test_an_axis_outside_a_row_still_gives_that_row_no_room() -> None:
    """Slack widens a budget; it does not resurrect a row the axis has left.

    Pinned because the tempting generalisation - letting slack pull the axis
    back into rows it has left - would let a block drift out of the lobe it
    belongs to.
    """
    spans = np.array([[0.0, 40.0], [60.0, 100.0], [10.0, 90.0]])
    sp = T._Spans(spans, 0.0, anchor_x=50.0)
    w, _ = sp.budgets(np.array([0.0, 1.0]), np.array([1.0, 2.0]), 20.0)
    assert w[0] == 0.0 and w[1] == 0.0
