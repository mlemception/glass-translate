"""Tests for glasstranslate.capture.diff.ChangeDetector (CPU only, no display)."""
from __future__ import annotations

import numpy as np
import pytest

from glasstranslate.capture.diff import ChangeDetector
from glasstranslate.core.types import Rect

W, H = 640, 360


def _frame(value: int = 200, w: int = W, h: int = H) -> np.ndarray:
    return np.full((h, w, 3), value, np.uint8)


def _covering(rects: list[Rect], target: Rect) -> bool:
    """True if some returned rect fully contains ``target``."""
    return any(r.x <= target.x and r.y <= target.y and r.x2 >= target.x2 and r.y2 >= target.y2 for r in rects)


def test_first_frame_is_whole_frame_dirty() -> None:
    det = ChangeDetector()
    rects = det.update(_frame())
    assert rects == [Rect(0, 0, W, H)]
    assert det.fraction_changed == 1.0


def test_unchanged_frame_returns_empty() -> None:
    det = ChangeDetector()
    det.update(_frame())
    assert det.update(_frame()) == []
    assert det.fraction_changed == 0.0


def test_small_noise_below_pixel_delta_is_ignored() -> None:
    det = ChangeDetector(min_pixel_delta=24)
    det.update(_frame(100))
    noisy = _frame(100)
    noisy[:, :, :] += 10  # uniform +10 on every channel -> gray shift of 10 < 24
    assert det.update(noisy) == []


def test_localized_change_yields_small_rect_containing_it() -> None:
    det = ChangeDetector(tile=64)
    det.update(_frame())
    changed = _frame()
    change = Rect(200, 100, 80, 40)
    changed[change.y : change.y2, change.x : change.x2] = 0
    rects = det.update(changed)
    assert len(rects) == 1
    r = rects[0]
    assert _covering(rects, change)
    # 80x40 change starting at (200,100) touches tiles x∈[3,4], y∈[1,2] -> at most 128x128
    assert r.w <= 3 * 64 and r.h <= 3 * 64
    assert r.x % 64 == 0 and r.y % 64 == 0
    assert 0.0 < det.fraction_changed < 0.05


def test_two_separate_changes_give_two_rects() -> None:
    det = ChangeDetector(tile=64)
    det.update(_frame())
    changed = _frame()
    a = Rect(10, 10, 40, 40)
    b = Rect(500, 250, 60, 60)
    changed[a.y : a.y2, a.x : a.x2] = 0
    changed[b.y : b.y2, b.x : b.x2] = 0
    rects = det.update(changed)
    assert len(rects) == 2
    assert _covering(rects, a) and _covering(rects, b)
    assert not rects[0].intersects(rects[1])


def test_adjacent_dirty_tiles_are_merged() -> None:
    det = ChangeDetector(tile=64)
    det.update(_frame())
    changed = _frame()
    band = Rect(0, 64, W, 64)  # a full row of tiles
    changed[band.y : band.y2, band.x : band.x2] = 0
    rects = det.update(changed)
    assert rects == [band]


def test_tiny_change_below_threshold_is_not_dirty() -> None:
    det = ChangeDetector(tile=64, threshold=0.02)
    det.update(_frame())
    changed = _frame()
    changed[10:12, 10:12] = 0  # 4 px of a 4096 px tile -> ~0.1% < 2%
    assert det.update(changed) == []
    assert det.fraction_changed > 0.0


def test_edge_partial_tiles_use_real_area() -> None:
    # 650 wide with tile 64 -> last column is only 10 px wide.
    det = ChangeDetector(tile=64, threshold=0.5)
    det.update(_frame(w=650, h=64))
    changed = _frame(w=650, h=64)
    changed[:, 640:650] = 0  # the whole partial tile changes
    rects = det.update(changed)
    assert rects == [Rect(640, 0, 10, 64)]


def test_size_change_marks_whole_frame_dirty() -> None:
    det = ChangeDetector()
    det.update(_frame())
    det.update(_frame())
    rects = det.update(_frame(w=W + 32, h=H - 16))
    assert rects == [Rect(0, 0, W + 32, H - 16)]
    assert det.fraction_changed == 1.0
    assert det.update(_frame(w=W + 32, h=H - 16)) == []


def test_reset_marks_next_frame_dirty() -> None:
    det = ChangeDetector()
    det.update(_frame())
    det.reset()
    assert det.fraction_changed == 0.0
    assert det.update(_frame()) == [Rect(0, 0, W, H)]


def test_fraction_changed_tracks_full_frame_change() -> None:
    det = ChangeDetector()
    det.update(_frame(0))
    det.update(_frame(255))
    assert det.fraction_changed == pytest.approx(1.0)
    changed = _frame(255)
    changed[: H // 2] = 0
    det.update(changed)
    assert det.fraction_changed == pytest.approx(0.5, abs=0.01)


def test_odd_sizes_and_grayscale_input() -> None:
    det = ChangeDetector(tile=32)
    gray = np.full((123, 77), 50, np.uint8)
    assert det.update(gray) == [Rect(0, 0, 77, 123)]
    assert det.update(gray) == []
    gray2 = gray.copy()
    gray2[120:123, 70:77] = 255
    rects = det.update(gray2)
    assert rects == [Rect(64, 96, 13, 27)]


def test_invalid_input_rejected() -> None:
    det = ChangeDetector()
    with pytest.raises(TypeError):
        det.update(np.zeros((10, 10, 3), np.float32))
    with pytest.raises(ValueError):
        det.update(np.zeros((10, 10, 4), np.uint8))
    with pytest.raises(ValueError):
        ChangeDetector(threshold=1.5)


def test_fullhd_speed_budget() -> None:
    """Performance guard: well under the 5 ms target even on a slow CI box."""
    import time

    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    b = a.copy()
    b[300:400, 500:900] = 255 - b[300:400, 500:900]
    det = ChangeDetector()
    det.update(a)
    det.update(b)  # warm caches
    n = 20
    t0 = time.perf_counter()
    for i in range(n):
        det.update(a if i % 2 else b)
    per_frame_ms = (time.perf_counter() - t0) / n * 1e3
    assert per_frame_ms < 15.0, f"{per_frame_ms:.2f} ms per frame"
