"""Line budgets centred on the block's optical axis, not on each row's own middle.

``typeset._finish`` already centres every line of a bubble block on one
``anchor_x`` - the region's optical centre - via ``anchor.line_centre``.  That
function clamps the line into its row's budget:

    slack = (budget_w - line_w) / 2
    return min(max(anchor_x, budget_centre - slack), budget_centre + slack)

so a line nearly as wide as its rows' intersection has no slack and takes *that
intersection's* centre instead of the axis.  The intersection's centre is not
the axis wherever the region is not symmetric about it, which on a real balloon
is most rows.  Measured on one page, per-line centres wandered std 7.56 px,
range 23 px, against a professionally lettered std 0.45 px, range 1.5 px.

The budget is therefore cut symmetric about the axis:

    half = min(anchor_x - left, right - anchor_x)

which makes the axis the budget's centre by construction, so the clamp can
never pull a line off it - and which is a strict subset of the intersection, so
containment cannot get worse.

Synthetic spans only; no corpus imagery enters the repository.
"""
from __future__ import annotations

import importlib

import numpy as np

from glasstranslate.render.anchor import line_centre

# ``glasstranslate.render`` exports a ``typeset`` function that shadows the
# submodule of the same name, so the module is fetched explicitly.
T = importlib.import_module("glasstranslate.render.typeset")


def lopsided_spans(n: int = 80) -> np.ndarray:
    """Rows whose runs lie further right the further down you go.

    Every row contains x = 50, so an axis there is inside the region
    throughout, but no row is centred on it.
    """
    spans = np.zeros((n, 2))
    for i in range(n):
        spans[i, 0] = 10 + 0.3 * i
        spans[i, 1] = 90 + 0.9 * i
    return spans


def test_budget_is_centred_on_the_anchor_when_one_is_given() -> None:
    spans = lopsided_spans()
    sp = T._Spans(spans, 0.0, anchor_x=50.0)
    tops = np.arange(0, 70, 7, dtype=np.float64)
    widths, centres = sp.budgets(tops, tops + 6.0)
    assert (widths > 0).any(), "no line had any room at all"
    ok = widths > 0
    np.testing.assert_allclose(centres[ok], 50.0)


def test_without_an_anchor_the_budget_is_unchanged() -> None:
    """The free-text path keeps the intersection's own centre."""
    spans = lopsided_spans()
    plain = T._Spans(spans, 0.0)
    tops = np.arange(0, 70, 7, dtype=np.float64)
    widths, centres = plain.budgets(tops, tops + 6.0)
    for t, b, w, c in zip(tops, tops + 6.0, widths, centres):
        assert (w, c) == T._line_budget(spans, t, b, 0.0)


def test_the_symmetric_budget_never_leaves_the_region() -> None:
    """Claim 2: a strict subset of the intersection, so containment holds."""
    spans = lopsided_spans()
    anchor = 50.0
    sym = T._Spans(spans, 0.0, anchor_x=anchor)
    plain = T._Spans(spans, 0.0)
    tops = np.arange(0, 70, 3, dtype=np.float64)
    sw, sc = sym.budgets(tops, tops + 6.0)
    pw, pc = plain.budgets(tops, tops + 6.0)
    for i in range(len(tops)):
        if sw[i] <= 0:
            continue
        assert sw[i] <= pw[i] + 1e-9, "symmetric budget wider than the intersection"
        assert sc[i] - sw[i] / 2 >= pc[i] - pw[i] / 2 - 1e-9
        assert sc[i] + sw[i] / 2 <= pc[i] + pw[i] / 2 + 1e-9


def test_a_line_that_fills_its_budget_still_sits_on_the_axis() -> None:
    """The defect end to end: the clamp used to win, now it cannot.

    A line as wide as its budget has zero slack, so ``line_centre`` returns the
    budget's centre whatever the anchor is.  The fix is that the budget's
    centre IS the anchor.
    """
    spans = lopsided_spans()
    anchor = 50.0
    tops = np.arange(0, 70, 7, dtype=np.float64)

    sym_w, sym_c = T._Spans(spans, 0.0, anchor_x=anchor).budgets(tops, tops + 6.0)
    plain_w, plain_c = T._Spans(spans, 0.0).budgets(tops, tops + 6.0)

    # Each line exactly fills its own budget: no slack anywhere.
    sym_at = [line_centre(c, w, w, anchor) for w, c in zip(sym_w, sym_c) if w > 0]
    plain_at = [line_centre(c, w, w, anchor) for w, c in zip(plain_w, plain_c) if w > 0]

    assert len(sym_at) >= 5 and len(plain_at) >= 5
    assert float(np.std(sym_at)) == 0.0, f"symmetric budgets still wander: {sym_at}"
    np.testing.assert_allclose(sym_at, anchor)
    # And the old behaviour really did wander, so this test can fail.
    assert float(np.std(plain_at)) > 5.0, (
        f"the intersection budget was expected to wander, got std "
        f"{float(np.std(plain_at)):.2f}")


def test_line_budget_takes_the_anchor_too() -> None:
    """``_Spans.budgets`` is documented as bit-for-bit ``_line_budget``."""
    spans = lopsided_spans()
    anchor = 50.0
    sp = T._Spans(spans, 0.0, anchor_x=anchor)
    tops = np.arange(0, 70, 3, dtype=np.float64)
    widths, centres = sp.budgets(tops, tops + 6.0)
    for t, w, c in zip(tops, widths, centres):
        assert (w, c) == T._line_budget(spans, t, t + 6.0, 0.0, anchor)


def test_an_axis_outside_a_row_gives_that_row_no_room() -> None:
    """Rows the axis has left behind cannot host a line centred on it."""
    spans = np.array([[0.0, 40.0], [60.0, 100.0], [10.0, 90.0]])
    sp = T._Spans(spans, 0.0, anchor_x=50.0)
    widths, _ = sp.budgets(np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert widths[0] == 0.0, "axis right of the row, yet the row offered room"
    assert widths[1] == 0.0, "axis left of the row, yet the row offered room"
    assert widths[2] == 80.0, "axis inside the row should give 2 x min(50-10, 90-50)"
