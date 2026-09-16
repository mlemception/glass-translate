"""Horizontal condensation of the lettering face (``compose.CONDENSE_RATIO``).

No condensed cut of Anime Ace is licensed to us, so the glyphs are squeezed at
draw time.  The two things that can go wrong are both covered here:

* **measure and draw disagreeing.**  The fitter chooses a size and a line
  breaking from measured widths; if the drawn glyphs are wider than the widths
  the fitter used, text overflows the balloon it was fitted to.
* **a block's lines not sharing one axis.**  ``typeset.py`` gives every line its
  own ``cx`` from its own row span, so condensing a whole block about a single
  origin would slide lines sideways.  Condensation is per line for that reason,
  and the two-line test below is what fails if that is ever "simplified".

The module is imported by path: ``glasstranslate.render`` re-exports a
*function* named ``compose``, so ``from ... import compose`` does not give the
module (the same trap ``test_typeset.py`` documents).
"""

from __future__ import annotations

import importlib
from typing import Optional, Tuple

import pytest
from PIL import Image, ImageDraw, ImageFont

from glasstranslate.render.typeset import PlacedLine, Typeset

compose = importlib.import_module("glasstranslate.render.compose")

TEXT = "POSSIBLE."
SIZE = 64


def _font(size: int = SIZE):
    return ImageFont.truetype(compose.MANGA_FONT_PATH, size)


def _canvas(w: int = 1400, h: int = 400) -> Image.Image:
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


def _ink(img: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of everything drawn, or None when nothing was."""
    return img.getchannel("A").getbbox()


def _drawn_width(text: str = TEXT, size: int = SIZE) -> float:
    canvas = _canvas()
    compose._blit_condensed(canvas, text, _font(size), 700.0, 100.0, (0, 0, 0, 255))
    box = _ink(canvas)
    assert box is not None, "nothing was drawn"
    return box[2] - box[0]


def test_the_ratio_is_the_measured_gap_to_the_reference() -> None:
    """6.6 cap-heights against our 7.727 - see the perf doc, cycle 4."""
    assert compose.CONDENSE_RATIO == pytest.approx(6.6 / 7.727, abs=0.002)
    assert 0.8 < compose.CONDENSE_RATIO < 1.0, "below ~0.8 synthetic condensation looks squashed"


def test_drawn_letters_are_narrower_than_the_face_draws_them() -> None:
    """The behavioural change: fails against the old code, which drew natural."""
    natural = _font().getlength(TEXT)
    assert _drawn_width() == pytest.approx(natural * compose.CONDENSE_RATIO, rel=0.03)


def test_condensation_does_not_touch_the_height() -> None:
    canvas_plain, canvas_cond = _canvas(), _canvas()
    ImageDraw.Draw(canvas_plain).text((100, 100), TEXT, font=_font(), fill=(0, 0, 0, 255))
    compose._blit_condensed(canvas_cond, TEXT, _font(), 700.0, 100.0, (0, 0, 0, 255))
    plain, cond = _ink(canvas_plain), _ink(canvas_cond)
    assert plain is not None and cond is not None
    assert (cond[3] - cond[1]) == pytest.approx(plain[3] - plain[1], abs=1)


def test_the_measurer_reports_what_will_actually_be_drawn() -> None:
    """Measure/draw agreement - the overflow bug this class of change invites."""
    measured = compose.pil_measurer(compose.MANGA_FONT_PATH, condense=True)(TEXT, SIZE)[0]
    assert measured == pytest.approx(_drawn_width(), rel=0.04)


def test_the_plain_measurer_is_unchanged() -> None:
    """Only blocks condense; the plain-segment branch keeps natural metrics."""
    natural = compose.pil_measurer(compose.MANGA_FONT_PATH)(TEXT, SIZE)[0]
    assert natural == pytest.approx(_font().getlength(TEXT), rel=0.01)
    condensed = compose.pil_measurer(compose.MANGA_FONT_PATH, condense=True)(TEXT, SIZE)[0]
    assert condensed == pytest.approx(natural * compose.CONDENSE_RATIO, rel=0.001)


def test_a_line_is_centred_on_its_own_cx() -> None:
    canvas = _canvas()
    compose._blit_condensed(canvas, TEXT, _font(), 700.0, 100.0, (0, 0, 0, 255))
    box = _ink(canvas)
    assert box is not None
    assert (box[0] + box[2]) / 2.0 == pytest.approx(700.0, abs=2.0)


def test_lines_with_different_axes_stay_on_their_own_axes() -> None:
    """Why condensation is per line and not one squeeze of the whole block.

    A single scale origin would pull the off-axis line toward the other one;
    the further apart they are, the larger the error.
    """
    left_cx, right_cx = 300.0, 1000.0
    font = _font(48)
    ts = Typeset(
        size=48.0,
        line_h=60.0,
        lines=[
            PlacedLine("AAAA", left_cx, 60.0, font.getlength("AAAA") * compose.CONDENSE_RATIO),
            PlacedLine("BBBB", right_cx, 160.0, font.getlength("BBBB") * compose.CONDENSE_RATIO),
        ],
    )
    canvas = _canvas()
    compose.draw_typeset(canvas, ts, font, (0, 0, 0), (255, 255, 255), False)

    top = canvas.crop((0, 0, canvas.width, 140))
    bottom = canvas.crop((0, 140, canvas.width, canvas.height))
    top_box, bottom_box = _ink(top), _ink(bottom)
    assert top_box is not None and bottom_box is not None
    assert (top_box[0] + top_box[2]) / 2.0 == pytest.approx(left_cx, abs=3.0)
    assert (bottom_box[0] + bottom_box[2]) / 2.0 == pytest.approx(right_cx, abs=3.0)


def test_a_line_running_off_the_page_edge_is_clipped_not_raised() -> None:
    """``alpha_composite`` raises on an out-of-range box; ``draw.text`` did not."""
    canvas = _canvas(300, 200)
    compose._blit_condensed(canvas, TEXT, _font(), 290.0, 20.0, (0, 0, 0, 255))
    compose._blit_condensed(canvas, TEXT, _font(), 5.0, 20.0, (0, 0, 0, 255))
    assert _ink(canvas) is not None


def test_condensing_lets_a_width_bound_block_hold_a_larger_size() -> None:
    """The point of the slice, at the level the fitter works on."""
    T = importlib.import_module("glasstranslate.render.fit")

    text = "A BLOCK OF DIALOGUE THAT HAS TO FIT"
    natural = compose.pil_measurer(compose.MANGA_FONT_PATH)
    condensed = compose.pil_measurer(compose.MANGA_FONT_PATH, condense=True)
    fit_natural = T.fit_text(text, 420, 200, natural)
    fit_condensed = T.fit_text(text, 420, 200, condensed)
    assert fit_condensed.size >= fit_natural.size
    assert (fit_condensed.size, len(fit_condensed.lines)) != (fit_natural.size, len(fit_natural.lines))
