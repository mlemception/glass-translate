"""``HOUGH_MAX_LINES`` - the cap on how much art the eraser protects.

``erase._art_lines`` detects the straight art entering a text zone, keeps the
lines whose extension crosses it, then follows **the longest ``HOUGH_MAX_LINES``
of them** through the text.  Whatever it detected beyond that count is erased.

**These are guards, not a regression test, and the difference is stated on
purpose.**  The slice that raised the cap from 30 to 300 is carried by a corpus
measurement, not by anything here:

* over 11 blocks on the four worst-damaged pages, ``_art_lines`` detected a
  median of **182** candidate lines and up to **1097**, and **30 of 33** calls
  were over the cap - the worst block protected 2.7 % of the art it had
  already identified;
* sweeping the cap over those pages moved the share of the letterer's kept
  artwork that we destroy: **59.4 %** at 30, 55.4 % at 100, **51.4 %** at 300,
  49.9 % at 1200.

A synthetic page cannot reproduce that size of effect and this file does not
pretend otherwise.  Clean unbroken lines are large ink components that
*classification* already keeps, so the cap never binds on them
(:func:`test_unbroken_art_lines_never_needed_the_cap` pins exactly that).  Real
speed lines are fragmented by screentone, neighbouring art and the glyphs
themselves into many small components that classification does not protect, and
only then does the cap decide.  The dashed fixture below reproduces the
*mechanism* at a fraction of the magnitude - 0.70 -> 0.85 -> 0.87 surviving as
the cap goes 5 -> 30 -> 300 - which is enough to pin the direction and the
monotonicity, and not enough to size the defect.

The fan is deliberately **radial, not parallel**: ``erase._hatching`` protects
thin strokes sharing a direction with a hatching field, so a parallel grating
would be rescued by that path instead of this one.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from glasstranslate.core.types import Rect, Segment, SegmentStyle
from glasstranslate.render import erase
from glasstranslate.render.layout import TextBlock

PAPER = 250
GLYPH = 30
N_LINES = 90


def _quad(r: Rect) -> np.ndarray:
    return np.array([[r.x, r.y], [r.x2, r.y], [r.x2, r.y2], [r.x, r.y2]], np.float32)


def _block(box: Rect, glyph: float) -> TextBlock:
    seg = Segment("テスト", _quad(box), 0.9)
    style = SegmentStyle(fg=(0, 0, 0), bg=(PAPER, PAPER, PAPER), angle_deg=0.0,
                         text_height_px=glyph, vertical=True, in_bubble=False)
    return TextBlock(seg, style, [seg], [], None)


def _draw_column(img: np.ndarray, x: int, y0: int, n: int, glyph: int, gap: int = 6) -> Rect:
    for i in range(n):
        y = y0 + i * (glyph + gap)
        cv2.rectangle(img, (x, y), (x + glyph - 1, y + 4), (0, 0, 0), -1)
        cv2.rectangle(img, (x + glyph // 2 - 2, y), (x + glyph // 2 + 2, y + glyph - 1), (0, 0, 0), -1)
        cv2.rectangle(img, (x, y + glyph - 5), (x + glyph - 1, y + glyph - 1), (0, 0, 0), -1)
    return Rect(x, y0, glyph, n * (glyph + gap) - gap)


def _burst(shape: tuple[int, int], *, dash: int = 0, n: int | None = None) -> np.ndarray:
    """Speed lines radiating from a focus outside the frame.

    ``dash`` 0 draws them unbroken; a positive ``dash`` breaks each into
    segments, which is what stops classification from keeping them wholesale.
    """
    h, w = shape
    count = N_LINES if n is None else n
    layer = np.zeros((h, w), np.uint8)
    fx, fy = -w * 0.6, h * 0.5
    for i in range(count):
        ang = np.deg2rad(-52.0 + 104.0 * i / (count - 1))
        dx, dy = float(np.cos(ang)), float(np.sin(ang))
        if not dash:
            cv2.line(layer, (int(fx), int(fy)),
                     (int(fx + 4 * w * dx), int(fy + 4 * w * dy)), 255, 2, cv2.LINE_8)
            continue
        t = 0.0
        while t < 4 * w:
            cv2.line(layer, (int(fx + t * dx), int(fy + t * dy)),
                     (int(fx + (t + dash) * dx), int(fy + (t + dash) * dy)), 255, 2, cv2.LINE_8)
            t += dash + 7
    return layer


def _surviving_art_fraction(*, dash: int = 26) -> float:
    """Share of the burst's ink inside the erased patch that is still ink."""
    h, w = 600, 700
    img = np.full((h, w, 3), PAPER, np.uint8)
    lines = _burst((h, w), dash=dash)
    img[lines > 0] = (0, 0, 0)
    box = _draw_column(img, 330, 120, 8, GLYPH)
    block = _block(box, float(GLYPH))

    patch, rect, _ = erase.erase_block(img, block)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    art = lines[rect.y:rect.y2, rect.x:rect.x2] > 0
    # Art the glyph column sits on is legitimately ambiguous; exclude it.
    col = np.zeros(lines.shape, bool)
    col[box.y:box.y2, box.x:box.x2] = True
    art &= ~col[rect.y:rect.y2, rect.x:rect.x2]
    assert art.sum() > 500, "the fixture stopped putting art inside the erased patch"
    return float(((gray < 128) & art).sum()) / float(art.sum())


def test_the_cap_is_above_what_a_real_page_detects():
    """The cap is a cost bound, not a classifier - keep it clear of the data.

    The only assertion here that fails against the pre-slice code, and it fails
    on the constant rather than on behaviour.  Corpus measurement: median 182
    candidate lines per call, max 1097.  A cap at or below that median silently
    discards most of the art on the pages that have the most of it.
    """
    assert erase.HOUGH_MAX_LINES >= 300


def test_unbroken_art_lines_never_needed_the_cap():
    """Why a clean synthetic cannot show this defect.

    Whole lines are large ink components that classification keeps outright, so
    the Hough path is not what saves them and the cap does not bind.  If this
    ever starts depending on the cap, the fixture has drifted away from the
    thing it is documenting.
    """
    assert _surviving_art_fraction(dash=0) > 0.95


@pytest.mark.parametrize("cap, floor", [(5, 0.60), (30, 0.80), (300, 0.85)])
def test_a_fragmented_burst_survives_better_as_the_cap_rises(cap, floor, monkeypatch):
    """The mechanism, at the magnitude a synthetic can reach.

    Fragmented strokes are glyph-sized components, so only ``_art_lines`` can
    keep them and the cap throttles how many it keeps.
    """
    monkeypatch.setattr(erase, "HOUGH_MAX_LINES", cap)
    assert _surviving_art_fraction() > floor


def test_raising_the_cap_never_erases_more(monkeypatch):
    """The safety argument, asserted rather than assumed.

    Lines reach this stage already classified as art - detected on the ink mask
    with glyph components removed, and required to show clear run outside the
    text zone.  Following more of them can only *keep* ink, never erase more,
    so the surviving fraction must not fall as the cap rises.  If this ever
    fails, the slice's whole recall-safety argument is wrong.
    """
    monkeypatch.setattr(erase, "HOUGH_MAX_LINES", 30)
    at_30 = _surviving_art_fraction()
    monkeypatch.undo()
    assert _surviving_art_fraction() >= at_30 - 1e-9
