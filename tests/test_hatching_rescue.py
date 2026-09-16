"""The hatching rescue's three reach constants, and why they are what they are.

A broken speed line inside an OCR box is glyph-sized, partly inside and not
touching the border, so the free-text branch of ``erase._classify`` claims it as
a glyph (``erase.py:745``).  ``_hatching`` is the only thing that gives it back,
and it needs ``HATCH_FIELD`` strokes of one direction, within ``HATCH_ANGLE``
degrees, found **outside** the text within ``HATCH_REACH`` glyphs.

**These are guards, not a regression test** - the same honest label as
``test_art_line_cap.py``, and for the same reason.  The synthetic fixture there
scores **0.870 under both the old and the new constants**: once the Hough path
keeps those strokes the hatching rescue is redundant on a clean page, so nothing
a unit test can build distinguishes them.  The corpus sweep is the evidence,
over the six worst-damaged pages:

| setting | artwork destroyed | ``erase_recall`` |
|---|---|---|
| old (0.35 / 4 / 1.0) | 51.1 % | 0.939 |
| ``HATCH_FIELD`` 4 -> 2 alone | 49.8 % | 0.933 |
| ``HATCH_MIN_LEN`` 0.35 -> 0.20 alone | 49.0 % | 0.933 |
| ``HATCH_REACH`` 1.0 -> 2.0 alone | 47.6 % | 0.933 |
| **all three** | **43.7 %** | **0.921** |

Two levers were swept and left alone deliberately: ``HATCH_FILL`` (0.35 -> 0.50)
changed the destroyed share by **0.0 points** and ``HATCH_ANGLE`` (15 -> 30) by
0.2.  That is the useful finding rather than a null result - these strokes are
not failing the *direction* test, they are failing to find a field to belong to,
which is why reach is what moves the number.  Do not re-sweep them.
"""
from __future__ import annotations

from glasstranslate.render import erase


def test_the_field_search_reaches_past_the_text():
    """``HATCH_REACH`` is the strongest of the three and saturates at 2.0.

    3.0 measured slightly *worse* on both axes (44.1 % destroyed, recall 0.920),
    so this is a measured optimum and not a "bigger is better" knob.
    """
    assert erase.HATCH_REACH >= 2.0


def test_a_field_can_form_from_a_pair():
    """Four strokes of one direction is more than a fragmented burst supplies."""
    assert erase.HATCH_FIELD <= 2


def test_short_fragments_can_still_be_hatching():
    """The strokes this rescues are broken, so the length floor has to be low."""
    assert erase.HATCH_MIN_LEN <= 0.20


def test_the_two_levers_that_do_nothing_are_left_alone():
    """Pins the swept-and-rejected pair so they are not re-tuned on a hunch.

    Raising either was measured and bought nothing; changing them should be a
    deliberate edit here with a fresh measurement, not a guess.
    """
    assert erase.HATCH_FILL == 0.35
    assert erase.HATCH_ANGLE == 15.0
