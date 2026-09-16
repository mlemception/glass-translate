"""Glyph coverage: the renderer must never draw a ``.notdef`` box.

The bundled comic face carries no bullet, em dash, inverted question mark or accented
Latin, so any of those reached the canvas as tofu.  Measured over the 50-page corpus
sample: 9 blocks of 316 (2.8 %) drew at least one box.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
from PIL import Image

from glasstranslate.core.types import Rect, Segment, SegmentStyle, StyledSegment, TranslatedSegment

# The package re-exports a *function* named ``compose``; import the module.
compose = importlib.import_module("glasstranslate.render.compose")

# Every codepoint the corpus sample actually hit, with what it must become.
MISSING_IN_FACE = {
    "•": "-",      # BULLET, 8 of the 13 occurrences
    "—": "--",     # EM DASH
    "¿": "",       # INVERTED QUESTION MARK - dropped, it is OCR noise here
    "⊥": "",       # UP TACK - likewise
    "Ć": "C",      # LATIN CAPITAL C WITH ACUTE - folds
}


@pytest.fixture(scope="module")
def face_path() -> str:
    return compose.MANGA_FONT_PATH


def test_the_face_really_lacks_these_codepoints(face_path: str) -> None:
    """Guards the premise.  If a future face covers them, the table above is stale."""
    assert compose.unrenderable_chars("".join(MISSING_IN_FACE), face_path)


@pytest.mark.parametrize("ch,expected", sorted(MISSING_IN_FACE.items()))
def test_each_missing_codepoint_folds_to_something_renderable(
    ch: str, expected: str, face_path: str
) -> None:
    folded = compose.fold_to_face(ch, face_path)
    assert folded == expected
    assert compose.unrenderable_chars(folded, face_path) == []


def test_plain_ascii_is_untouched(face_path: str) -> None:
    text = "THE QUICK BROWN FOX, 12345 -- DON'T PANIC!"
    assert compose.fold_to_face(text, face_path) == text


def test_latin1_accents_the_face_does_carry_are_left_alone(face_path: str) -> None:
    """Pins where the face's coverage actually stops: Latin-1 mostly yes, Extended-A no."""
    text = "CAFÉ NAÏVE SEÑOR À Ü"
    assert compose.unrenderable_chars(text, face_path) == []
    assert compose.fold_to_face(text, face_path) == text


@pytest.mark.parametrize("ch,expected", [
    ("Ć", "C"),   # C WITH ACUTE - the one the corpus hit
    ("Ā", "A"),   # A WITH MACRON
    ("Ę", "E"),   # E WITH OGONEK
    ("Ś", "S"),   # S WITH ACUTE
    ("Ż", "Z"),   # Z WITH DOT ABOVE
    ("Ç", "C"),   # C WITH CEDILLA - Latin-1, but this face lacks it
    ("Ł", "L"),   # L WITH STROKE - no decomposition; needs the table
    ("Ø", "O"),   # O WITH STROKE - likewise
])
def test_accented_latin_folds_generally_not_just_the_one_we_saw(
    ch: str, expected: str, face_path: str
) -> None:
    assert compose.unrenderable_chars(ch, face_path), "premise: the face lacks it"
    assert compose.fold_to_face(ch, face_path) == expected


def test_nothing_unrenderable_survives_a_fold(face_path: str) -> None:
    messy = "• ITEM — ¿⊥ ĆAFÉ…"
    assert compose.unrenderable_chars(compose.fold_to_face(messy, face_path), face_path) == []


def test_a_face_that_covers_everything_changes_nothing() -> None:
    """The fallback CJK face has all five, so folding against it is a no-op."""
    path = compose.default_font_path()
    if not path:
        pytest.skip("no system fallback font on this machine")
    if compose.unrenderable_chars("".join(MISSING_IN_FACE), path):
        pytest.skip("system fallback face does not cover the probe set")
    text = "• ITEM — Ć"
    assert compose.fold_to_face(text, path) == text


def _segment(text: str) -> TranslatedSegment:
    quad = np.array([[10, 10], [210, 10], [210, 90], [10, 90]], dtype=np.float64)
    style = SegmentStyle(
        fg=(0, 0, 0),
        bg=(255, 255, 255),
        angle_deg=0.0,
        vertical=False,
        text_height_px=18.0,
        layout_box=Rect(10, 10, 200, 80),
        max_font_px=18.0,
    )
    return TranslatedSegment(StyledSegment(Segment("x", quad, 0.9), style, "ja"), text, "en")


def _draw(text: str) -> np.ndarray:
    canvas = Image.new("RGBA", (240, 120), (255, 255, 255, 255))
    path = compose.MANGA_FONT_PATH
    compose._draw_block(
        canvas, _segment(text), path, compose.pil_measurer(path),
        hide_original=False, uppercase=False,
    )
    return np.asarray(canvas.convert("L"))


def test_a_bullet_draws_the_same_pixels_as_the_hyphen_it_folds_to() -> None:
    """The behavioural regression.

    Against the old renderer these differ: the bullet reaches the face, misses, and draws
    a solid ``.notdef`` rectangle, while the hyphen draws a hyphen.  Folding before
    measurement makes the two identical -- and identical is the assertion, because it also
    proves the fold happens before line breaking rather than at draw time.
    """
    assert np.array_equal(_draw("• ONE"), _draw("- ONE"))


def test_an_accent_draws_the_same_pixels_as_its_base_letter() -> None:
    assert np.array_equal(_draw("ĆAT"), _draw("CAT"))


# --- Cycle 3: one stray glyph must not re-face a whole block --------------------

MOSTLY_LATIN = "兴 THEY SAY MEN-ARE WOLVES, BUT..."   # 1 ideograph in 33 chars
GENUINELY_CJK = "それはきっと天使に"


def test_has_cjk_still_reports_any_cjk_at_all() -> None:
    """The primitive is unchanged; only the face decision moves off it."""
    assert compose.has_cjk(MOSTLY_LATIN)
    assert compose.has_cjk(GENUINELY_CJK)
    assert not compose.has_cjk("PLAIN ENGLISH")


def test_one_stray_ideograph_does_not_claim_the_cjk_face() -> None:
    """The regression: 1 CJK character in 33 used to re-face the whole block."""
    assert not compose.needs_cjk_face(MOSTLY_LATIN)


def test_genuine_cjk_still_takes_the_cjk_face() -> None:
    assert compose.needs_cjk_face(GENUINELY_CJK)


def test_an_evenly_split_block_stays_on_the_fallback() -> None:
    """Deliberately conservative: at the threshold we do not move the block."""
    assert compose.needs_cjk_face("A二")


def test_whitespace_does_not_dilute_the_ratio() -> None:
    assert compose.needs_cjk_face("天使   天使")
    assert not compose.needs_cjk_face("")


def test_the_face_chosen_for_a_mostly_latin_block_is_the_comic_one() -> None:
    style = SegmentStyle(fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, vertical=False,
                         text_height_px=18.0, layout_box=Rect(10, 10, 200, 80), max_font_px=18.0)
    assert compose.block_font_path_for(style, MOSTLY_LATIN, "FALLBACK.ttf") == \
        compose.manga_font_path(False)
    assert compose.block_font_path_for(style, GENUINELY_CJK, "FALLBACK.ttf") == "FALLBACK.ttf"


def test_the_stray_ideograph_is_then_dropped_rather_than_drawn_as_tofu() -> None:
    """Composes with the fold: the comic face cannot draw it, so it goes."""
    folded = compose.fold_to_face(MOSTLY_LATIN, compose.MANGA_FONT_PATH)
    assert "兴" not in folded
    assert folded.strip().startswith("THEY SAY")
    assert compose.unrenderable_chars(folded, compose.MANGA_FONT_PATH) == []
