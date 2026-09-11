"""Tests for ``demo/typeset_metrics.py`` on synthetic pages: SSIM, the erase
scores of a perfect and of a lazy render, lettering comparisons and the
markdown table.  No files, OCR or GPU."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import typeset_metrics as TM  # noqa: E402
import typeset_reference as TR  # noqa: E402


def _pages():
    """ja: art line + two Japanese blobs; eng: the line + English lettering elsewhere."""
    ja = np.full((200, 300), 245, np.uint8)
    eng = ja.copy()
    cv2.line(ja, (20, 100), (280, 100), 0, 3)
    cv2.line(eng, (20, 100), (280, 100), 0, 3)
    cv2.rectangle(ja, (120, 60), (140, 140), 0, -1)
    cv2.rectangle(ja, (160, 60), (180, 140), 0, -1)
    cv2.rectangle(eng, (200, 30), (260, 50), 0, -1)
    near = np.zeros(ja.shape, bool)
    near[20:180, 90:290] = True
    gt = TR.erase_ground_truth(ja, eng, near, [TR.Box(195, 25, 70, 30)])
    return ja, eng, gt


def _record(window=(90, 20, 200, 160), kind="art", em=20.0):
    return {"index": 0, "kind": kind, "text": "字", "em_px": em, "dark": False, "window": list(window), "bubble": None,
            "source_bbox": [120, 60, 60, 80], "stats": TR.lettering_stats(np.zeros((200, 300), bool), None, em)}


def test_ssim_is_one_for_identical_and_lower_for_blurred() -> None:
    ja, eng, _ = _pages()
    same = TM.ssim_map(eng.astype(np.float64), eng.astype(np.float64))
    assert same.shape == eng.shape and same.min() > 0.999
    blurred = cv2.GaussianBlur(eng, (9, 9), 3).astype(np.float64)
    region = np.zeros(eng.shape, bool)
    region[20:180, 90:290] = True
    assert TM.masked_mean(TM.ssim_map(eng.astype(np.float64), blurred), region) < 0.9
    assert TM.masked_mean(same, np.zeros(eng.shape, bool)) is None


def test_perfect_erase_scores_one_and_lazy_erase_scores_zero_recall() -> None:
    ja, eng, gt = _pages()
    rec = _record()
    perfect = TM.score_erase(ja, eng, eng, gt, rec)
    assert perfect["erase_iou"] == pytest.approx(1.0)
    assert perfect["erase_recall"] == pytest.approx(1.0) and perfect["erase_precision"] == pytest.approx(1.0)
    assert perfect["art_kept"] == pytest.approx(1.0)
    assert perfect["ssim"] > 0.99 and perfect["mae"] < 0.5
    lazy = TM.score_erase(ja, ja, eng, gt, rec)  # nothing erased
    assert lazy["erase_recall"] == pytest.approx(0.0) and lazy["erase_iou"] == pytest.approx(0.0)
    assert lazy["art_kept"] == pytest.approx(1.0)
    assert lazy["ssim"] < perfect["ssim"]
    # Erasing the art line too: recall stays 1, precision and art_kept drop.
    clumsy = eng.copy()
    clumsy[95:106, 100:200] = 245
    over = TM.score_erase(ja, clumsy, eng, gt, rec)
    assert over["erase_recall"] == pytest.approx(1.0) and over["erase_precision"] < 0.9 and over["art_kept"] < 0.7


def test_score_lettering_compares_ours_with_the_reference() -> None:
    h, w = 300, 300
    interior = np.zeros((h, w), np.uint8)
    cv2.ellipse(interior, (150, 150), (120, 100), 0, 0, 360, 1, -1)
    interior = interior.astype(bool)
    ref_ink = np.zeros((h, w), bool)
    for top in (120, 142, 164):
        ref_ink[top: top + 14, 100:200] = True
    ours_ink = np.zeros((h, w), bool)
    for top in (126, 146):  # two lines, 10 px caps; block centre at y = 141 (9 px above the bubble centre)
        ours_ink[top: top + 10, 110:190] = True
    ref_stats = TR.lettering_stats(ref_ink, interior, 20.0)
    out = TM.score_lettering(ours_ink, interior, 20.0, ref_stats, blocked=np.zeros((h, w), bool))
    assert out["lines"] == 2 and out["lines_ref"] == 3
    assert out["cap_ratio"] == pytest.approx(10 / 14, abs=0.08)
    assert out["inset_em"] > out["inset_ref_em"] > 0
    assert out["centre_dy_em"] == pytest.approx(-9 / 20.0, abs=0.1)
    assert out["centre_dy_ref_em"] == pytest.approx(-1 / 20.0, abs=0.1)
    assert out["overflow_px"] == 0 and out["collision_px"] == 0
    spill = ours_ink.copy()
    spill[140:150, 5:25] = True  # outside the bubble (its left edge is at x = 30), over a blocked area
    blocked = np.zeros((h, w), bool)
    blocked[:, :50] = True
    out2 = TM.score_lettering(spill, interior, 20.0, ref_stats, blocked=blocked)
    assert out2["overflow_px"] == 200 and out2["collision_px"] == 200


def test_page_table_lists_blocks_and_summary() -> None:
    rows = [
        {"index": 0, "kind": "bubble", "erase_iou": 0.5, "erase_recall": 0.6, "erase_precision": 0.7, "art_kept": 0.8,
         "ssim": 0.9, "mae": 3.0, "cap_ratio": 1.1, "inset_em": 0.3, "inset_ref_em": 0.25, "centre_dx_em": 0.0,
         "centre_dy_em": 0.1, "lines": 3, "lines_ref": 3, "overflow_px": 0, "collision_px": 0},
        {"index": 1, "kind": "art", "erase_iou": 0.3, "erase_recall": 0.4, "erase_precision": 0.9, "art_kept": 0.6,
         "ssim": 0.8, "mae": 5.0, "cap_ratio": None, "inset_em": None, "inset_ref_em": None, "centre_dx_em": None,
         "centre_dy_em": None, "lines": 2, "lines_ref": None, "overflow_px": 0, "collision_px": 12},
    ]
    summary = TM.summarize(rows)
    assert summary["blocks"] == 2 and summary["erase_iou"] == pytest.approx(0.4)
    assert summary["art_kept"] == pytest.approx(0.7) and summary["ssim"] == pytest.approx(0.85)
    assert summary["cap_ratio"] == pytest.approx(1.1) and summary["collisions"] == 1 and summary["overflows"] == 0
    md = TM.page_table("3jp", "base", rows, summary)
    assert "| # | kind |" in md and "| 0 | bubble |" in md and "| 1 | art |" in md
    assert "0.40" in md and "**mean**" in md
