"""Tests for ``demo/typeset_reference.py`` on synthetic pages: homography
recovery, the letterer's erase mask, and the lettering statistics (cap
height, line count, inset, centring).  No OCR engine, GPU or files."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import typeset_reference as TR  # noqa: E402


def _art_page(h: int = 600, w: int = 400, seed: int = 3) -> np.ndarray:
    """A synthetic monochrome page: paper with panel borders, random line art and blobs."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 245, np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), 0, 3)
    cv2.line(img, (10, h // 2), (w - 10, h // 2), 0, 3)
    for _ in range(40):
        p1 = (int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20)))
        p2 = (int(rng.integers(20, w - 20)), int(rng.integers(20, h - 20)))
        cv2.line(img, p1, p2, int(rng.integers(0, 90)), int(rng.integers(1, 4)), cv2.LINE_AA)
    for _ in range(25):
        c = (int(rng.integers(30, w - 30)), int(rng.integers(30, h - 30)))
        cv2.circle(img, c, int(rng.integers(4, 18)), int(rng.integers(0, 120)), int(rng.integers(1, 3)), cv2.LINE_AA)
    return img


def test_align_pages_recovers_a_scaled_rotated_homography() -> None:
    ja = _art_page()
    h, w = ja.shape
    # "eng" = the same art scanned larger, slightly rotated and shifted (eng -> ja is the inverse).
    scale = 1.33
    ang = np.deg2rad(0.8)
    a = np.array([[scale * np.cos(ang), -scale * np.sin(ang), 12.0], [scale * np.sin(ang), scale * np.cos(ang), 7.0], [0, 0, 1.0]])
    eng = cv2.warpPerspective(ja, a, (int(w * scale) + 30, int(h * scale) + 30), borderValue=245)
    eng = cv2.GaussianBlur(eng, (3, 3), 0)
    res = TR.align_pages(ja, eng)
    assert res.method in ("ecc", "orb")
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    src = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), a).reshape(-1, 2)  # ja corners in eng space
    back = cv2.perspectiveTransform(src.reshape(-1, 1, 2), res.homography).reshape(-1, 2)
    assert np.abs(back - corners).max() < 1.5, (res.method, np.abs(back - corners).max())
    assert res.score > 0.8
    warped = TR.warp_reference(cv2.cvtColor(eng, cv2.COLOR_GRAY2BGR), res.homography, ja.shape)
    assert warped.shape == (h, w, 3)
    diff = np.abs(warped[..., 0].astype(int) - ja.astype(int))
    assert np.median(diff) == 0 and diff.mean() < 16.0  # paper is paper; thin anti-aliased lines resample imperfectly


def test_align_pages_from_hand_picked_pairs() -> None:
    ja = _art_page()
    a = np.array([[1.25, 0.0, 20.0], [0.0, 1.25, 5.0], [0, 0, 1.0]])
    pts_ja = [(50, 60), (350, 80), (330, 540), (40, 520)]
    pts_eng = [tuple(cv2.perspectiveTransform(np.array([[p]], np.float64), a).reshape(2)) for p in pts_ja]
    res = TR.align_pages(ja, np.zeros((760, 520), np.uint8), pairs=list(zip(pts_ja, pts_eng)))
    assert res.method == "pairs"
    back = cv2.perspectiveTransform(np.array(pts_eng, np.float64).reshape(-1, 1, 2), res.homography).reshape(-1, 2)
    assert np.abs(back - np.array(pts_ja, np.float64)).max() < 0.5


def test_erase_ground_truth_marks_removed_japanese_keeps_art_and_ignores_english() -> None:
    ja = np.full((200, 300), 245, np.uint8)
    eng = ja.copy()
    cv2.line(ja, (20, 100), (280, 100), 0, 3)  # art line in both pages
    cv2.line(eng, (20, 100), (280, 100), 0, 3)
    # Japanese glyphs: two blobs crossing the line, gone in the reference.
    cv2.rectangle(ja, (120, 60), (140, 140), 0, -1)
    cv2.rectangle(ja, (160, 60), (180, 140), 0, -1)
    # English lettering in the reference, elsewhere in the same window.
    cv2.rectangle(eng, (200, 30), (260, 50), 0, -1)
    near = np.zeros(ja.shape, bool)
    near[20:180, 90:290] = True
    english_boxes = [TR.Box(195, 25, 70, 30)]
    gt = TR.erase_ground_truth(ja, eng, near, english_boxes)
    japanese = np.zeros(ja.shape, bool)
    japanese[60:141, 120:141] = True
    japanese[60:141, 160:181] = True
    japanese &= ~(np.abs(np.arange(200)[:, None] - 100) <= 1)  # the art line pixels stay
    assert (gt.erase & japanese).sum() > 0.9 * japanese.sum()
    assert (gt.erase & ~near).sum() == 0
    assert gt.erase[25:55, 195:265].sum() == 0  # never under the English lettering
    assert gt.english[30:51, 200:261].mean() > 0.95
    line = np.zeros(ja.shape, bool)
    line[99:102, 100:270] = True  # the part of the art line inside the near-text window
    assert (gt.kept_art & line).sum() > 0.8 * line.sum()
    assert (gt.kept_art & ~near).sum() == 0  # nothing is judged outside the window
    assert (gt.erase & line).sum() < 0.1 * line.sum()


def test_lettering_stats_measures_caps_lines_inset_and_centring() -> None:
    h, w = 300, 300
    interior = np.zeros((h, w), np.uint8)
    cv2.ellipse(interior, (150, 150), (120, 100), 0, 0, 360, 1, -1)
    interior = interior.astype(bool)
    ink = np.zeros((h, w), bool)
    cap, pitch = 14, 22
    tops = [150 - pitch - cap // 2, 150 - cap // 2, 150 + pitch - cap // 2]
    for top in tops:
        ink[top: top + cap, 100:200] = True
    stats = TR.lettering_stats(ink, interior, em=20.0)
    assert stats["line_count"] == 3
    assert abs(stats["cap_height_px"] - cap) <= 1
    assert abs(stats["line_pitch_px"] - pitch) <= 1
    assert stats["bbox"] == [100, tops[0], 100, tops[-1] + cap - tops[0]]
    assert abs(stats["centre_dx_px"]) <= 1 and abs(stats["centre_dy_px"]) <= 1
    # Inset: the line ends are 50 px left/right of the ellipse centre; the nearest outline point from the
    # top-left corner (100, 121) of the block is about 55 px away (a diagonal, closer than the 70 px along x).
    assert 50 <= stats["inset_min_px"] <= 60
    assert stats["inset_min_em"] == pytest.approx(stats["inset_min_px"] / 20.0)
    assert stats["cap_em"] == pytest.approx(cap / 20.0, abs=0.06)
    assert stats["ink_px"] == int(ink.sum())
    empty = TR.lettering_stats(np.zeros((h, w), bool), interior, em=20.0)
    assert empty["line_count"] == 0 and empty["bbox"] is None and empty["cap_height_px"] is None


def test_line_clusters_splits_on_gaps_and_drops_specks() -> None:
    profile = np.zeros(100, int)
    profile[10:20] = 5
    profile[22:24] = 1  # a two-row speck below the first line: merged (gap 2 < min gap 3)
    profile[40:52] = 8
    profile[80:81] = 1  # a one-row speck: dropped (shorter than min height 3)
    assert TR.line_clusters(profile, min_gap=3, min_height=3) == [(10, 24), (40, 52)]
