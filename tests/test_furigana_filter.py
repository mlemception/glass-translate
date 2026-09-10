"""Geometric furigana (ruby) detection over raw OCR segments
(``glasstranslate.ocr.furigana``).  Pure geometry + kana test, no models."""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from glasstranslate.core.types import Segment
from glasstranslate.ocr import furigana as F


def seg(text: str, x: int, y: int, w: int, h: int) -> Segment:
    quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    return Segment(text=text, quad=quad, confidence=0.9)


# A vertical kanji column 30 px wide, 200 px tall.
PARENT_COLUMN = seg("漢字の縦書き", 100, 50, 30, 200)
# A horizontal kanji line 300 px wide, 40 px tall.
PARENT_LINE = seg("漢字の横書き文章", 50, 100, 300, 40)


# ------------------------------------------------------------- kana test
@pytest.mark.parametrize(
    "text, expected",
    [
        ("ふりがな", True),
        ("カタカナー", True),
        ("ふり がな", True),
        ("漢字", False),
        ("かん字", False),
        ("abc", False),
        ("", False),
        ("   ", False),
        ("。、", False),
    ],
)
def test_is_kana_text(text: str, expected: bool) -> None:
    assert F.is_kana_text(text) is expected


# ------------------------------------------------------------- vertical
def test_small_kana_column_beside_a_tall_column_is_furigana() -> None:
    ruby = seg("かんじ", 132, 60, 12, 80)  # right of the column, 2 px gap
    assert F.furigana_indices([PARENT_COLUMN, ruby]) == frozenset({1})


def test_left_side_furigana_is_detected() -> None:
    ruby = seg("かんじ", 86, 60, 12, 80)  # left of the column, 2 px gap
    assert F.furigana_indices([ruby, PARENT_COLUMN]) == frozenset({0})


def test_same_size_kana_column_is_not_furigana() -> None:
    kana_column = seg("ひらがなだけ", 135, 50, 30, 200)
    assert F.furigana_indices([PARENT_COLUMN, kana_column]) == frozenset()


def test_kana_column_far_from_its_column_is_not_furigana() -> None:
    far = seg("かんじ", 300, 60, 12, 80)  # 170 px gap
    assert F.furigana_indices([PARENT_COLUMN, far]) == frozenset()


def test_kana_column_without_vertical_overlap_is_not_furigana() -> None:
    below = seg("かんじ", 132, 260, 12, 80)  # right of the column but under it
    assert F.furigana_indices([PARENT_COLUMN, below]) == frozenset()


def test_tiny_parent_does_not_make_ruby() -> None:
    parent = seg("漢字", 100, 50, 6, 40)  # glyphs below MIN_PARENT_GLYPH_PX
    ruby = seg("か", 107, 55, 3, 20)
    assert F.furigana_indices([parent, ruby]) == frozenset()


# ------------------------------------------------------------- horizontal
def test_horizontal_ruby_above_a_line_is_detected() -> None:
    ruby = seg("かんじ", 60, 84, 60, 14)  # above the line, 2 px gap
    assert F.furigana_indices([PARENT_LINE, ruby]) == frozenset({1})


def test_kana_line_below_a_line_is_not_furigana() -> None:
    below = seg("かんじ", 60, 142, 60, 14)  # ruby never sits under horizontal text
    assert F.furigana_indices([PARENT_LINE, below]) == frozenset()


def test_kana_line_far_above_is_not_furigana() -> None:
    far = seg("かんじ", 60, 20, 60, 14)  # 66 px gap
    assert F.furigana_indices([PARENT_LINE, far]) == frozenset()


# ------------------------------------------------------------- never ruby
def test_filter_is_a_no_op_for_latin_text() -> None:
    line = seg("Hello world", 50, 100, 300, 40)
    small = seg("tiny", 60, 84, 40, 14)
    segs = [line, small]
    assert F.furigana_indices(segs) == frozenset()
    assert F.strip_furigana(segs) == segs


def test_kanji_text_is_never_removed() -> None:
    small_kanji = seg("漢", 132, 60, 12, 14)
    segs = [PARENT_COLUMN, small_kanji]
    assert F.furigana_indices(segs) == frozenset()
    assert len(F.strip_furigana(segs)) == 2


# ------------------------------------------------------------- strip
def test_strip_preserves_order_and_does_not_mutate_input() -> None:
    ruby_a = seg("かんじ", 132, 60, 12, 80)
    other = seg("外の文", 320, 200, 20, 80)
    ruby_b = seg("よこ", 60, 84, 60, 14)
    segs: List[Segment] = [ruby_a, PARENT_COLUMN, other, PARENT_LINE, ruby_b]
    snapshot = list(segs)
    out = F.strip_furigana(segs)
    assert out == [PARENT_COLUMN, other, PARENT_LINE]
    assert out is not segs
    assert segs == snapshot  # input untouched
    assert all(a is b for a, b in zip(segs, snapshot))


def test_empty_input() -> None:
    assert F.furigana_indices([]) == frozenset()
    assert F.strip_furigana([]) == []


def test_constants_are_the_documented_defaults() -> None:
    assert F.FURIGANA_MAX_HEIGHT_RATIO == 0.6
    assert F.FURIGANA_MAX_GAP_RATIO == 0.6
    assert F.MIN_PARENT_GLYPH_PX == 8
