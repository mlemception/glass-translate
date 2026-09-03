"""Tests for glasstranslate.render.fit with a fake linear measurer:
width = 0.5 * size * len(text), height = size."""
from __future__ import annotations

import pytest

from glasstranslate.render.fit import FitResult, fit_text, wrap_lines


def measure(text: str, size: float) -> tuple[float, float]:
    return 0.5 * size * len(text), size


def total_height(res: FitResult) -> float:
    return sum(measure(line, res.size)[1] for line in res.lines)


def max_width(res: FitResult) -> float:
    return max(measure(line, res.size)[0] for line in res.lines)


def test_single_line_fits_at_start_size():
    # box 200x20 -> start 17; "Hello" width = 0.5*17*5 = 42.5 <= 200
    res = fit_text("Hello", 200, 20, measure)
    assert res.lines == ["Hello"]
    assert res.size == pytest.approx(17.0)


def test_explicit_start_size_is_used():
    res = fit_text("Hi", 200, 20, measure, start_size=12)
    assert res.size == 12
    assert res.lines == ["Hi"]


def test_single_line_preferred_when_it_fits():
    # One line at start size (34) beats two lines at 20 each.
    res = fit_text("ab cd", 200, 40, measure)
    assert res.lines == ["ab cd"]
    assert res.size == pytest.approx(34.0)
    # With a small explicit start size, splitting lets the text grow instead.
    res = fit_text("ab cd", 200, 40, measure, start_size=10)
    assert res.lines == ["ab cd"]
    assert res.size == 10


def test_wraps_into_two_lines_when_too_wide():
    # 16 chars at size 34 (0.85*40) = 272 > 200; two lines at size 20:
    # "Hello wonderful" = 15*10 = 150 <= 200; "world" fits -> 2 lines
    res = fit_text("Hello wonderful world", 200, 40, measure)
    assert len(res.lines) == 2
    assert " ".join(res.lines) == "Hello wonderful world"
    assert max_width(res) <= 200
    assert total_height(res) <= 40
    # wrapping is preferred over shrinking: two lines at box_h/2 each
    assert res.size == pytest.approx(20.0)


def test_respects_max_lines_and_shrinks_instead():
    text = "one two three four five six seven eight nine ten"
    res = fit_text(text, 120, 60, measure, max_lines=2)
    assert len(res.lines) <= 2
    assert max_width(res) <= 120
    assert total_height(res) <= 60
    assert " ".join(res.lines) == text


def test_shrinks_to_fit_narrow_box():
    res = fit_text("Supercalifragilistic", 60, 40, measure, max_lines=1)
    assert res.lines == ["Supercalifragilistic"]
    assert max_width(res) <= 60
    assert res.size >= 6.0


def test_breaks_long_word_when_wider_than_box():
    # At size 6 a single char is 3 wide; box 30 holds 10 chars per line.
    res = fit_text("abcdefghijklmnopqrst", 30, 100, measure, max_lines=3, min_size=6)
    assert "".join(res.lines) == "abcdefghijklmnopqrst"
    assert len(res.lines) >= 2
    assert max_width(res) <= 30


def test_hits_min_size_and_still_returns_lines():
    text = "x" * 500
    res = fit_text(text, 10, 5, measure, max_lines=2, min_size=6.0)
    assert res.size == 6.0
    assert res.lines
    assert all(res.lines)


def test_never_returns_empty_lines():
    assert fit_text("", 100, 20, measure).lines == [""]
    assert fit_text("   ", 100, 20, measure).lines == [""]
    assert fit_text("a", 1, 1, measure).lines == ["a"]


def test_result_is_deterministic():
    a = fit_text("The quick brown fox jumps over the lazy dog", 90, 50, measure)
    b = fit_text("The quick brown fox jumps over the lazy dog", 90, 50, measure)
    assert a == b


def test_wrap_lines_greedy():
    assert wrap_lines("aa bb cc dd", 4, 10, measure) == ["aa bb", "cc dd"]
    assert wrap_lines("aaaaaaaa", 4, 8, measure) == ["aaaa", "aaaa"]
    assert wrap_lines("", 4, 8, measure) == [""]


def test_cjk_without_spaces_is_character_broken():
    res = fit_text("翻訳された文章です", 40, 60, measure, max_lines=3)
    assert "".join(res.lines) == "翻訳された文章です"
    assert max_width(res) <= 40
    assert total_height(res) <= 60
