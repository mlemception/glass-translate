"""The Qt overlay condenses balloon lettering, as ``compose`` already does.

``compose`` squeezes balloon dialogue to ``CONDENSE_RATIO`` (0.854) because our
face is ~14 % wider than the reference's, so blocks that did not fit stepped
*down* in size instead.  Over the corpus that change put blocks at the ceiling
35 % -> 41 %, median size/ceiling 0.887 -> 0.936, total lines 1125 -> 1089, and
a blind comparison picked it 6-0-2.

The app letters through ``ui/overlay.py``, a parallel Qt implementation, which
had no condensation of any kind - so none of that reached a user.  Qt does it
with ``QFont::setStretch``, measured offscreen against the bundled
``animeace2_reg.ttf``:

    stretch 85 -> drawn ink width ratio 0.8503   (compose: 0.854, agree to 0.4 %)
                  drawn ink height ratio 1.0000  (exactly, at every stretch)
                  QFontMetricsF vs drawn ink      agree to 0.02 %

The height row is the point.  Qt transforms outlines; the PIL route's
supersampling shifted glyph height 19 px -> 23 px and corrupted the cap
measurement the layout is judged on.

BALLOONS ONLY.  ``compose.condenses`` gates on ``style.in_bubble`` because
condensing free text over artwork failed the corpus gate - panel-border
collisions 0 -> 1 on two pages.  The overlay imports that function rather than
copying the rule, so the two renderers cannot drift apart.

Qt runs offscreen here; no window is shown, and the overlay window itself is
not touched by any of this.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtGui import QFontMetricsF  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from glasstranslate.core.types import Rect, SegmentStyle  # noqa: E402
from glasstranslate.render.compose import CONDENSE_RATIO  # noqa: E402
from glasstranslate.ui import overlay as O  # noqa: E402

SAMPLE = "THE QUICK BROWN FOX"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def family(qapp):
    fam = O.load_manga_font()
    if not fam:
        pytest.skip("the bundled lettering face is unavailable")
    return fam


def _style(*, in_bubble: bool) -> SegmentStyle:
    box = Rect(0, 0, 200, 120)
    return SegmentStyle(
        fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, text_height_px=20.0,
        vertical=True, layout_box=box, clean_rect=box, in_bubble=in_bubble,
        outline=not in_bubble, max_font_px=16.0,
    )


def test_a_condensed_font_is_narrower_and_exactly_as_tall(family) -> None:
    """The defect: the overlay drew balloon lettering at full width.

    Fails against the previous code, where ``make_font`` took no ``condensed``
    argument at all.
    """
    plain = O.make_font(family, 40)
    tight = O.make_font(family, 40, condensed=True)
    wp = QFontMetricsF(plain).horizontalAdvance(SAMPLE)
    wt = QFontMetricsF(tight).horizontalAdvance(SAMPLE)
    assert wt < wp, "condensed lettering was not narrower"
    ratio = wt / wp
    assert abs(ratio - CONDENSE_RATIO) < 0.02, (
        f"Qt condensed to {ratio:.4f}, compose uses {CONDENSE_RATIO}; the two renderers "
        f"must letter a block at the same width")
    # The property the PIL route never had: squeezing must not touch cap height.
    assert QFontMetricsF(tight).capHeight() == QFontMetricsF(plain).capHeight()
    assert QFontMetricsF(tight).ascent() == QFontMetricsF(plain).ascent()


def test_the_measurer_reports_the_width_that_will_be_drawn(family) -> None:
    """Measuring and drawing must agree, or the block is laid out against widths
    the painter will not produce."""
    for condensed in (False, True):
        measure = O.qt_measurer(family, condensed=condensed)
        drawn = QFontMetricsF(O.make_font(family, 40, condensed=condensed))
        assert abs(measure(SAMPLE, 40)[0] - drawn.horizontalAdvance(SAMPLE)) < 0.51


def test_a_condensed_measurer_is_narrower_than_a_plain_one(family) -> None:
    """The measurer must carry the flag too - it caches a font per size, so a
    plain measurer would silently lay the block out at full width."""
    plain = O.qt_measurer(family)
    tight = O.qt_measurer(family, condensed=True)
    assert tight(SAMPLE, 40)[0] < plain(SAMPLE, 40)[0]
    # Height is the line box, which condensing must leave alone.
    assert tight(SAMPLE, 40)[1] == plain(SAMPLE, 40)[1]


def test_balloons_condense_and_free_text_does_not() -> None:
    """Slice 8's gate, which this must carry: condensing free text over artwork
    failed the corpus gate with panel-border collisions 0 -> 1 on two pages."""
    assert O.condenses(_style(in_bubble=True)) is True
    assert O.condenses(_style(in_bubble=False)) is False


def test_the_overlay_uses_composes_own_gate_not_a_copy() -> None:
    """If the two renderers ever disagree about which blocks condense, the app
    and the corpus stop being comparable - so the rule has one home."""
    import importlib

    # ``glasstranslate.render`` exports a *function* named ``compose`` that
    # shadows the submodule, as it does for ``typeset``; fetch the module.
    compose = importlib.import_module("glasstranslate.render.compose")
    assert O.condenses is compose.condenses


def test_plain_is_still_the_default(family) -> None:
    """Everything that does not ask for it is untouched."""
    assert O.make_font(family, 40).stretch() != O.CONDENSED_STRETCH
    assert O.make_font(family, 40, condensed=True).stretch() == O.CONDENSED_STRETCH
