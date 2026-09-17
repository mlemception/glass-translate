"""Tests for glasstranslate.render.style (angles, heights, colours, vertical)."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from glasstranslate.core.types import Segment
from glasstranslate.render.style import (
    extract_colors,
    is_vertical,
    measure_style,
    quad_angle_deg,
    quad_text_height,
)


def rotated_quad(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    """Axis-aligned w x h rectangle centred at (cx, cy), rotated CCW-on-screen
    by ``angle_deg`` (screen y grows downward).  Points ordered TL, TR, BR, BL
    in the text's reading frame, matching RapidOCR."""
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    a = math.radians(angle_deg)
    # CCW on screen with y-down: x' = x cos a + y sin a ; y' = -x sin a + y cos a
    rot = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    return (local @ rot.T + np.array([cx, cy])).astype(np.float32)


# ---------------------------------------------------------------- angle


def test_axis_aligned_quad_has_zero_angle():
    q = rotated_quad(100, 50, 200, 30, 0.0)
    assert quad_angle_deg(q) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("angle", [20.0, -20.0, 45.0])
def test_quad_angle_matches_generated_rotation(angle):
    q = rotated_quad(300, 200, 240, 40, angle)
    assert quad_angle_deg(q) == pytest.approx(angle, abs=0.5)


def test_positive_angle_means_right_end_higher_on_screen():
    # Baseline from (0, 100) to (100, 50): the right end is higher (smaller y).
    q = np.array([[0, 100], [100, 50], [110, 70], [10, 120]], dtype=np.float32)
    assert quad_angle_deg(q) > 0


def test_angle_normalised_into_half_open_range():
    # Baseline pointing straight left (180 deg) is equivalent to 0 deg.
    q = np.array([[100, 0], [0, 0], [0, 20], [100, 20]], dtype=np.float32)
    assert quad_angle_deg(q) == pytest.approx(0.0, abs=1e-6)
    # Baseline pointing straight down: -90 -> normalised to +90.
    q = np.array([[0, 0], [0, 100], [-20, 100], [-20, 0]], dtype=np.float32)
    assert quad_angle_deg(q) == pytest.approx(90.0, abs=1e-6)


def test_angle_accepts_plain_lists():
    assert quad_angle_deg([[0, 0], [10, 0], [10, 5], [0, 5]]) == 0.0


# ---------------------------------------------------------------- height


def test_text_height_is_short_edge_length():
    q = rotated_quad(0, 0, 200, 30, 33.0)
    assert quad_text_height(q) == pytest.approx(30.0, abs=1e-3)


# ---------------------------------------------------------------- colours


def make_text_image(bg_rgb, fg_rgb, size=(120, 40), text_box=(20, 10, 100, 30)):
    """Solid background with a filled rectangle of foreground colour, as BGR."""
    h, w = size[1], size[0]
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = bg_rgb[::-1]
    x0, y0, x1, y1 = text_box
    img[y0:y1, x0:x1] = fg_rgb[::-1]
    return img


def quad_from_box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def test_white_on_blue_colors():
    blue, white = (10, 40, 200), (255, 255, 255)
    img = make_text_image(blue, white)
    fg, bg = extract_colors(img, quad_from_box(15, 5, 105, 35))
    assert fg == white
    assert bg == blue


def test_black_on_white_colors():
    img = make_text_image((255, 255, 255), (0, 0, 0))
    fg, bg = extract_colors(img, quad_from_box(15, 5, 105, 35))
    assert fg == (0, 0, 0)
    assert bg == (255, 255, 255)


def test_colors_are_rgb_not_bgr():
    # Red background in RGB is (255, 0, 0) -> BGR bytes (0, 0, 255).
    img = make_text_image((255, 0, 0), (0, 255, 0))
    fg, bg = extract_colors(img, quad_from_box(15, 5, 105, 35))
    assert bg == (255, 0, 0)
    assert fg == (0, 255, 0)


def test_colors_tolerate_noise():
    rng = np.random.default_rng(0)
    img = make_text_image((30, 30, 30), (240, 240, 240)).astype(np.int16)
    img += rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    fg, bg = extract_colors(img, quad_from_box(15, 5, 105, 35))
    assert all(abs(c - 240) <= 8 for c in fg)
    assert all(abs(c - 30) <= 8 for c in bg)


def test_colors_empty_crop_falls_back():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    fg, bg = extract_colors(img, quad_from_box(100, 100, 120, 120))
    assert (fg, bg) == ((0, 0, 0), (255, 255, 255))


def test_colors_uniform_crop_returns_contrasting_fg():
    img = np.full((40, 40, 3), 255, dtype=np.uint8)
    fg, bg = extract_colors(img, quad_from_box(5, 5, 30, 30))
    assert bg == (255, 255, 255)
    assert fg == (0, 0, 0)


def test_colors_quad_partially_outside_image():
    img = make_text_image((255, 255, 255), (0, 0, 0))
    fg, bg = extract_colors(img, quad_from_box(-10, -10, 105, 35))
    assert fg == (0, 0, 0)
    assert bg == (255, 255, 255)


# ---------------------------------------------------------------- vertical / measure_style


def test_vertical_detection():
    tall = quad_from_box(0, 0, 20, 100)
    assert is_vertical(tall, "縦書き")
    assert not is_vertical(tall, "I")  # single char: never vertical
    wide = quad_from_box(0, 0, 100, 20)
    assert not is_vertical(wide, "hello")
    almost_square = quad_from_box(0, 0, 20, 28)
    assert not is_vertical(almost_square, "ab")


def test_measure_style_combines_everything():
    img = make_text_image((10, 40, 200), (255, 255, 255))
    seg = Segment(text="Hello", quad=quad_from_box(15, 5, 105, 35), confidence=0.9)
    style = measure_style(img, seg)
    assert style.fg == (255, 255, 255)
    assert style.bg == (10, 40, 200)
    assert style.angle_deg == pytest.approx(0.0)
    assert style.text_height_px == pytest.approx(30.0)
    assert style.vertical is False


def test_measure_style_rotated_vertical_segment():
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    quad = rotated_quad(150, 150, 20, 120, 10.0)
    seg = Segment(text="縦書き", quad=quad, confidence=0.9)
    style = measure_style(img, seg)
    assert style.vertical is True
    assert style.angle_deg == pytest.approx(10.0, abs=0.5)
    assert style.text_height_px == pytest.approx(120.0, abs=1e-3)


def antialiased_text_image(bg: tuple[int, int, int], ink: tuple[int, int, int],
                           *, thickness: int = 1) -> np.ndarray:
    """Fine strokes drawn with ``cv2.LINE_AA``, so most ink pixels are fringe.

    This is what small CJK lettering looks like to ``extract_colors``: the
    solid centres of the strokes are outnumbered by part-ink, part-paper edge
    pixels, and the median of the foreground cluster is a blend nobody
    printed.
    """
    img = np.full((40, 120, 3), bg[::-1], np.uint8)  # BGR
    for k in range(6):
        x = 12 + 18 * k
        cv2.line(img, (x, 8), (x + 10, 31), ink[::-1], thickness, cv2.LINE_AA)
        cv2.line(img, (x + 10, 8), (x, 31), ink[::-1], thickness, cv2.LINE_AA)
    return img


def test_antialiased_black_text_is_read_as_black_not_grey() -> None:
    """The ink colour is the ink, not the average of the ink and the paper.

    A guard, not a regression: synthetic anti-aliased strokes already read as
    black before ``_ink_core`` existed (luma 54 at the thinnest stroke that
    draws).  Reproducing the real failure takes a real page - a crop there
    carries furigana, neighbouring columns, screentone and the balloon wall,
    and the two-way split lands differently.  The evidence for that is in
    docs/perf/2026-09-15-typeset-corpus.md; this pins the property.
    """
    img = antialiased_text_image((255, 255, 255), (0, 0, 0))
    fg, bg = extract_colors(img, quad_from_box(5, 4, 110, 32))
    assert max(bg) > 200, f"paper misread as {bg}"
    # Anime Ace is lettered in black; anything above the layout layer's
    # _FG_SNAP_LUMA = 90 floor is drawn as visible grey on the page.
    luma = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
    assert luma < 90, f"black lettering read as grey {fg} (luma {luma:.0f})"


def test_antialiased_white_on_black_is_read_as_white_not_grey() -> None:
    """The same, the other way up: inverted balloons and SFX on black.

    This direction is what stops ``_ink_core`` being written as "take the
    darkest quarter": white-on-black lettering has to get whiter, not darker.
    """
    img = antialiased_text_image((0, 0, 0), (255, 255, 255))
    fg, bg = extract_colors(img, quad_from_box(5, 4, 110, 32))
    assert min(bg) < 60, f"dark paper misread as {bg}"
    luma = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
    assert luma > 165, f"white lettering read as grey {fg} (luma {luma:.0f})"


# ------------------------------------------------- upright dialogue

def test_dialogue_is_lettered_upright_not_italic() -> None:
    """The release sets ordinary dialogue in a roman face, not an oblique one.

    Measured over the corpus: the reference's own English ink de-shears to
    -2.0, +1.5, -2.5, +0.0 and -5.0 degrees on five pages, while the bundled
    ``animeace2_ital`` measures +14.0 and ``animeace2_reg`` +0.0.  Two blind
    two reviews raised the slant independently before it was measured.
    """
    from glasstranslate.core.types import Rect, SegmentStyle
    from glasstranslate.render.compose import block_font_path_for, block_italic, manga_font_path

    dialogue = SegmentStyle(fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0,
                            text_height_px=20.0, vertical=True,
                            layout_box=Rect(0, 0, 100, 100), in_bubble=True)
    assert block_italic(dialogue) is False, "vertical dialogue is still lettered in the italic"
    assert block_font_path_for(dialogue, "HELLO", None) == manga_font_path(False)


def test_the_italic_face_is_still_available() -> None:
    """Kept for the real convention - thought and flashback - if a signal ever
    distinguishes them."""
    from glasstranslate.render.compose import manga_font_path

    assert manga_font_path(True) is not None
    assert manga_font_path(True) != manga_font_path(False)
