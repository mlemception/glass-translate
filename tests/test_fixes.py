"""Regression tests for the review-fix round: grey backdrops are not
bubbles, bubble margins are applied on every side, all-CJK text is not
all-caps, fallback flows centre on the region, the vectorised span / budget
code matches its per-row reference, hyphens trade against size, joined
boxes split at their joint."""
from __future__ import annotations

import importlib

import cv2
import numpy as np
import pytest

from glasstranslate.core.types import Rect, Segment
from glasstranslate.render import build_blocks

T = importlib.import_module("glasstranslate.render.typeset")
C = importlib.import_module("glasstranslate.render.compose")


def fake_measure(text: str, size: float) -> tuple[float, float]:
    return 0.5 * size * len(text), 1.15 * size


def _quad(r: Rect) -> np.ndarray:
    return np.array([[r.x, r.y], [r.x2, r.y], [r.x2, r.y2], [r.x, r.y2]], np.float32)


def _column_page(level: int, box: Rect = Rect(320, 200, 20, 80)) -> tuple[np.ndarray, list[Segment]]:
    img = np.full((400, 600, 3), level, np.uint8)
    for k in range(4):
        cv2.rectangle(img, (box.x + 2, box.y + 2 + 20 * k), (box.x2 - 2, box.y + 16 + 20 * k), (0, 0, 0), -1)
    return img, [Segment("外の文です", _quad(box), 0.9)]


# ------------------------------------------------------------ layout
@pytest.mark.parametrize("level", [120, 150, 190])
def test_mid_grey_backdrop_is_free_text_not_a_bubble(level):
    img, segs = _column_page(level)
    st = build_blocks(img, segs)[0].style
    assert not st.in_bubble
    assert st.layout_box is not None and st.layout_box.w > 2 * 20  # wider than the source column
    ts = C.typeset_block("THE TEXT OUTSIDE THE BUBBLE", st, fake_measure)
    assert ts.size == pytest.approx(st.max_font_px) and ts.fitted
    assert ts.bbox is not None and ts.bbox.w > 20  # not squeezed into the 20-px source column


def test_bubble_margin_is_applied_on_every_side():
    img = np.full((300, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (109, 49), (190, 250), (0, 0, 0), 2)  # a caption box: interior x 111..188
    box = Rect(140, 80, 20, 140)
    for k in range(6):
        cv2.rectangle(img, (box.x + 2, box.y + 2 + 22 * k), (box.x2 - 2, box.y + 18 + 22 * k), (0, 0, 0), -1)
    blk = build_blocks(img, [Segment("縦書きの文章です", _quad(box), 0.9)])[0]
    st = blk.style
    assert st.in_bubble and st.layout_mask is not None and st.layout_box is not None
    cols = np.flatnonzero(st.layout_mask.any(axis=0)) + st.layout_box.x
    rows = np.flatnonzero(st.layout_mask.any(axis=1)) + st.layout_box.y
    # Inset from the interior on the left AND the right (the component's own bounding box edge).
    assert cols[0] >= 111 + 3 and cols[-1] <= 188 - 3
    assert abs((cols[0] - 111) - (188 - cols[-1])) <= 1
    assert rows[0] >= 51 + 3 and rows[-1] <= 248 - 3


# ------------------------------------------------------------ metrics / fallback
def test_all_cjk_text_is_not_caps_and_keeps_full_pitch():
    assert T.is_caps("ALL CAPS, 247!") and not T.is_caps("Mixed") and not T.is_caps("人外魔境新宿決戦") and not T.is_caps("247")
    assert T.line_pitch("人外魔境新宿決戦", 23.0) == pytest.approx(23.0)
    assert T.ink_offsets("人外魔境", 23.0) == (0.0, 23.0)


def test_fallback_centres_on_the_filled_rows():
    spans = np.zeros((140, 2))
    spans[20:80] = (300, 400)
    ts = T._fallback(["WORD"] * 3, spans, 0.0, fake_measure, 6.0, 0.0)
    assert all(l.cx == pytest.approx(350.0) for l in ts.lines)


# ------------------------------------------------------------ vectorised spans / budgets
def _optical_column(mask):
    """The mask's optical centre column, in mask coordinates: the x of its
    distance transform's peak, padded so a region touching the edge of the
    array does not peak on the border.  Spelled out here rather than imported
    so the reference stays independent of the code under test."""
    if not mask.any():
        return mask.shape[1] / 2.0
    dt = cv2.distanceTransform(np.pad(mask.astype(bool), 1).view(np.uint8), cv2.DIST_L2, 5)
    return float(np.nonzero(dt >= dt.max() - 1e-6)[1].mean()) - 1.0


def _mask_spans_reference(mask, rect, centre):
    """``centre`` is in mask coordinates; ``mask_spans`` takes a page ``cx``."""
    h, w = mask.shape[:2]
    spans = np.zeros((h, 2), dtype=np.float64)
    for r in range(h):
        row = mask[r]
        if not row.any():
            continue
        padded = np.concatenate(([False], row.astype(bool), [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        starts, ends = edges[0::2], edges[1::2]
        inside = (starts <= centre) & (centre < ends)
        if inside.any():
            i = int(np.flatnonzero(inside)[0])
        else:
            dist = np.minimum(np.abs(starts - centre), np.abs(ends - 1 - centre))
            i = int(np.argmin(dist))
        spans[r] = (rect.x + starts[i], rect.x + ends[i])
    return spans


def test_mask_spans_matches_per_row_reference():
    rng = np.random.default_rng(3)
    rect = Rect(17, 5, 60, 40)
    for _ in range(20):
        mask = rng.random((40, 60)) < 0.5
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8)).astype(bool)
        mask[rng.integers(0, 40, 3)] = False  # some empty rows
        for cx in (None, 17.0, 40.5, 90.0):
            # cx=None means the mask's own optical column, not the rect centre:
            # the balloon may sit off to one side of the box the layout cut.
            centre = _optical_column(mask) if cx is None else cx - rect.x
            np.testing.assert_array_equal(T.mask_spans(mask, rect, cx), _mask_spans_reference(mask, rect, centre))


def test_spans_budgets_match_line_budget():
    rng = np.random.default_rng(7)
    spans = np.zeros((93, 2))
    x1 = rng.integers(0, 40, 93)
    spans[:, 0] = x1
    spans[:, 1] = x1 + rng.integers(0, 60, 93)
    spans[[0, 1, 50, 92]] = 0.0
    sp = T._Spans(spans, 12.5)
    tops = 12.5 + rng.random(400) * 100 - 5
    bottoms = tops + rng.random(400) * 30
    widths, centres = sp.budgets(tops, bottoms)
    for t, b, w, c in zip(tops, bottoms, widths, centres):
        assert (w, c) == T._line_budget(spans, t, b, 12.5)


# ------------------------------------------------------------ hyphens vs size, lobes
def test_smaller_size_without_hyphens_beats_three_hyphens():
    pytest.importorskip("pyphen")
    text = "CONSTITUTIONALLY CONSTITUTIONALLY CONSTITUTIONALLY"
    # 16 letters at 10 px each do not fit 150 px at size 20; one 7 % step does.
    ts = T.typeset(text, T.rect_spans(Rect(0, 0, 150, 400)), 0.0, fake_measure, max_size=20)
    assert ts.hyphens == 0 and 18.0 < ts.size < 20.0
    assert [l.text for l in ts.lines] == ["CONSTITUTIONALLY"] * 3


def test_lobe_split_survives_the_joint_rows():
    spans = np.zeros((200, 2))
    spans[0:100] = (200, 380)
    spans[100:105] = (100, 380)  # the boxes' shared edge: one run across both
    spans[105:200] = (100, 250)
    assert T.lobe_split_row(spans) == 105
    spans[:] = (100, 380)  # a single box: no split
    assert T.lobe_split_row(spans) is None
    spans[190:200] = (100, 160)  # a short tail at the bottom is no lobe either
    assert T.lobe_split_row(spans) is None
