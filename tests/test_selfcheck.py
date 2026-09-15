"""Tests for ``glasstranslate/render/selfcheck.py`` - the three reference-free
invariants.  Synthetic strings and tiny masks only: no corpus page, no OCR, no
GPU, no network, no torch."""
from __future__ import annotations

import numpy as np
import pytest

from glasstranslate.render import selfcheck as SC


# ------------------------------------------------------------ text_complete
def test_text_complete_accepts_a_block_broken_and_hyphenated() -> None:
    """Breaking and hyphenating move characters between lines; they never lose any."""
    text = "THE PRECIOUSNESS OF THE WEAK."
    assert SC.text_complete(text, ["THE", "PRECIOUS-", "NESS OF", "THE WEAK."])
    assert SC.text_complete(text, [text])
    # Nothing drawn is not a fragment - it is a different state, and None.
    assert SC.text_complete("", []) is None
    assert SC.text_complete("80", []) is None


def test_text_complete_catches_the_dropped_word() -> None:
    """Fragment lettering: the typesetter silently set less than it was handed,
    and the reader sees half a sentence in the balloon."""
    text = "BESIDES, YOU GUYS CAN'T EVEN GO TO HAUNTED SPOTS WITHOUT ME."
    assert not SC.text_complete(text, ["TO HAUNTED", "SPOTS"])
    assert not SC.text_complete("ONE TWO THREE", ["ONE TWO"])
    assert SC.text_complete("ONE TWO", []) is None  # not lettered at all: answered's job


# ------------------------------------------------------------- breaks_clean
def test_breaks_clean_accepts_breaks_at_spaces_and_at_hyphens() -> None:
    assert SC.breaks_clean("BEING AN ADULT IS SO CONFUSING!",
                           ["BEING AN", "ADULT IS", "SO CONFUSING!"])
    # A hyphen the typesetter inserted: on the page, absent from the source.
    assert SC.breaks_clean("THE PRECIOUSNESS OF THE WEAK.",
                           ["THE", "PRECIOUS-", "NESS OF", "THE WEAK."])
    # A hyphen the text already carried, broken after.
    assert SC.breaks_clean("AWW, SEN-PAI.", ["AWW, SEN-", "PAI."])
    assert SC.breaks_clean("ONE", ["ONE"])
    assert SC.breaks_clean("", []) is None
    assert SC.breaks_clean("183", []) is None


def test_breaks_clean_catches_the_break_that_ate_its_space() -> None:
    """The defect that sets ``BEINGAN ADULTISSO CONFUSING!``: every character is
    present, so ``text_complete`` is happy, and the page is still unreadable."""
    text = "BEING AN ADULT IS SO CONFUSING!"
    lines = ["BEINGAN", "ADULTISSO", "CONFUSING!"]
    assert SC.text_complete(text, lines)      # nothing was lost...
    assert not SC.breaks_clean(text, lines)   # ...and it still reads wrong
    # A break mid-word with no hyphen at all.
    assert not SC.breaks_clean("CONFUSING", ["CONFU", "SING"])


def test_breaks_clean_catches_invented_and_reordered_text() -> None:
    assert not SC.breaks_clean("ONE TWO", ["ONE", "TWO", "THREE"])
    assert not SC.breaks_clean("ONE TWO THREE", ["TWO THREE", "ONE"])
    assert not SC.breaks_clean("ONE TWO", ["ONE"])


# -------------------------------------------------------- ink_inside_bubble
def _interior(size: int = 60) -> np.ndarray:
    mask = np.zeros((size, size), bool)
    mask[5:-5, 5:-5] = True
    return mask


def test_ink_inside_bubble_is_zero_for_a_cleanly_erased_balloon() -> None:
    page = np.full((60, 60), 255, np.uint8)
    assert SC.ink_inside_bubble(page, _interior(), em_px=8.0) == 0


def test_ink_inside_bubble_finds_the_glyph_the_eraser_never_saw() -> None:
    """Furigana the eraser missed survives as a stray Japanese glyph next to
    English lettering - the leftover the reader sees immediately."""
    page = np.full((60, 60), 255, np.uint8)
    page[28:36, 28:36] = 0          # a surviving glyph, well inside the interior
    found = SC.ink_inside_bubble(page, _interior(), em_px=8.0)
    assert found == 64


def test_ink_inside_bubble_ignores_the_outline_the_eraser_keeps() -> None:
    """The eraser deliberately preserves the balloon outline, so the band at the
    interior's edge must not read as leftover source ink."""
    page = np.full((60, 60), 255, np.uint8)
    interior = _interior()
    page[5:7, 5:-5] = 0             # the outline, right at the interior's edge
    assert SC.ink_inside_bubble(page, interior, em_px=8.0) == 0


def test_ink_inside_bubble_is_none_for_free_text_with_no_balloon() -> None:
    page = np.full((60, 60), 255, np.uint8)
    assert SC.ink_inside_bubble(page, None, em_px=8.0) is None
    assert SC.ink_inside_bubble(page, np.zeros((60, 60), bool), em_px=8.0) is None


def test_ink_inside_bubble_reads_dark_paper_the_other_way_round() -> None:
    page = np.zeros((60, 60), np.uint8)      # dark balloon
    page[28:36, 28:36] = 255                 # light ink on it
    assert SC.ink_inside_bubble(page, _interior(), em_px=8.0, dark=True) == 64
    assert SC.ink_inside_bubble(page, _interior(), em_px=8.0, dark=False) > 0


# -------------------------------------------------- what the old code could not see
def test_these_invariants_are_not_derivable_from_the_counters_they_join() -> None:
    """The gate already had overflow / leftover / uncontained / collisions, and
    none of them can see any of the three defects above: a fragment fits its
    balloon, a break that ate its space fits it too, and both leave the page
    geometrically perfect."""
    text = "BEING AN ADULT IS SO CONFUSING!"
    assert SC.text_complete(text, ["BEINGAN", "ADULTISSO", "CONFUSING!"])
    assert not SC.breaks_clean(text, ["BEINGAN", "ADULTISSO", "CONFUSING!"])
    assert not SC.text_complete(text, ["SO CONFUSING!"])
    assert SC.breaks_clean(text, ["BEING AN ADULT IS SO CONFUSING!"])


@pytest.mark.parametrize("field", ["fragments", "bad_breaks", "bubble_ink_px"])
def test_the_gate_enforces_each_new_invariant(field: str) -> None:
    """Each one is a hard invariant: zero to non-zero on two pages fails the gate."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
    import corpus_score as CS

    assert field in [f for _, f in CS._HARD_INVARIANTS]
    clean = {"overflow_px": 0, "leftover_px": 0, "uncontained": 0, "collisions": 0,
             "fragments": 0, "bad_breaks": 0, "bubble_ink_px": 0}
    baseline = {"mean_R": 50.0, "pages": {"a": dict(clean), "b": dict(clean)}}
    broken = {"mean_R": 50.0, "pages": {"a": {**clean, field: 3}, "b": {**clean, field: 1}}}
    verdict = CS.compare_baseline(broken, baseline)
    assert not verdict["passed"]
    assert len(verdict["invariant_regressions"]) == 2
    assert CS.compare_baseline(baseline, baseline)["passed"]
