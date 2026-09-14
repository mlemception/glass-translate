"""Tests for ``demo/typeset_metrics.py`` on synthetic pages: SSIM, the erase
scores of a perfect and of a lazy render, lettering comparisons, the block
geometry scores (containment, centring, leftover ink, blocks per bubble) and
the markdown table.  No files, OCR or GPU."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import typeset_metrics as TM  # noqa: E402
import typeset_reference as TR  # noqa: E402

from glasstranslate.core.types import Rect  # noqa: E402


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


# --------------------------------------------------------------- geometry
def _bubble(w: int = 300, h: int = 300, tail: bool = False) -> np.ndarray:
    """An elliptical interior, optionally with a tail running to the bottom."""
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (150, 120), (120, 90), 0, 0, 360, 1, -1)
    if tail:
        cv2.fillPoly(mask, [np.array([[130, 190], [175, 190], [120, 295]], np.int32)], 1)
    return mask.astype(bool)


def test_interior_core_erodes_by_the_margin_and_never_empties() -> None:
    interior = _bubble()
    core = TM.interior_core(interior, 20.0)  # 0.18 em -> 4 px
    assert core.sum() < interior.sum()
    dt = cv2.distanceTransform(interior.view(np.uint8), cv2.DIST_L2, 5)
    assert dt[core].min() >= 3.0  # every core pixel is at least the margin from the outline
    # A slit thinner than the margin still yields a region to judge against.
    slit = np.zeros((40, 40), bool)
    slit[18:21, 5:35] = True
    assert TM.interior_core(slit, 40.0).sum() == slit.sum()


def test_containment_is_the_share_of_glyph_ink_outside_the_eroded_interior() -> None:
    interior = _bubble()
    inside = np.zeros(interior.shape, bool)
    inside[90:150, 80:220] = True
    assert TM.score_geometry(inside, interior, 20.0)["containment"] == pytest.approx(0.0)
    assert TM.score_geometry(inside, interior, 20.0)["outside_px"] == 0
    spill = inside.copy()
    spill[100:120, 0:20] = True  # 400 px in the page margin, clear of the ellipse
    out = TM.score_geometry(spill, interior, 20.0)
    assert out["outside_px"] == 400
    assert out["containment"] == pytest.approx(400 / (60 * 140 + 400))
    # Free text: no interior, no score, and nothing to fold into the bubble means.
    blank = TM.score_geometry(spill, None, 20.0)
    assert blank["containment"] is None and blank["centre_offset_em"] is None


def test_centre_offset_is_measured_from_the_inscribed_centre() -> None:
    interior = _bubble(tail=True)
    cx, cy = TM.interior_centre(interior)
    assert cx == pytest.approx(150, abs=6) and cy == pytest.approx(120, abs=10)
    rows = np.flatnonzero(interior.any(axis=1))
    assert (rows[0] + rows[-1]) / 2.0 > cy + 30  # the tail drags the bounding box well below
    centred = np.zeros(interior.shape, bool)
    centred[int(cy) - 25: int(cy) + 25, int(cx) - 70: int(cx) + 70] = True
    assert TM.score_geometry(centred, interior, 20.0)["centre_offset_em"] == pytest.approx(0.0, abs=0.1)
    low = np.zeros(interior.shape, bool)
    low[int(cy) + 15: int(cy) + 65, int(cx) - 70: int(cx) + 70] = True
    assert TM.score_geometry(low, interior, 20.0)["centre_offset_em"] == pytest.approx(40 / 20.0, abs=0.05)


def test_interior_centre_of_a_panel_running_off_the_page_is_not_on_the_border() -> None:
    """A dark panel that reaches the edge of the page has no background beyond
    it, so an unpadded distance transform keeps rising and peaks on the border.
    Stands for 4ja block 8, the panel at ``46,734,348,466`` on a 1200-row page,
    which scored its centre at ``(241, 1199)`` and 10.26 em of offset."""
    region = np.zeros((200, 300), bool)
    region[120:200, 60:240] = True  # flush against the bottom edge
    cx, cy = TM.interior_centre(region)
    assert cx == pytest.approx(150, abs=3)
    assert cy == pytest.approx(160, abs=3)  # the middle of the visible part...
    assert cy < 198  # ...not the border row


def test_centre_offset_uses_the_blocks_own_part_of_a_shared_bubble() -> None:
    interior = _bubble()
    own = interior.copy()
    own[:, 150:] = False  # the left half is ours
    ink = np.zeros(interior.shape, bool)
    ink[100:140, 50:140] = True
    whole = TM.score_geometry(ink, interior, 20.0)["centre_offset_em"]
    mine = TM.score_geometry(ink, interior, 20.0, own)["centre_offset_em"]
    assert mine < whole


def test_leftover_ink_counts_source_ink_left_in_the_bubble_but_not_the_outline() -> None:
    interior = _bubble()
    erased = np.full(interior.shape, 245, np.uint8)
    assert TM.score_leftover(erased, interior, 20.0, False)["leftover_px"] == 0
    with_ruby = erased.copy()
    with_ruby[100:130, 200:210] = 20  # a ruby column the eraser missed: 300 px
    assert TM.score_leftover(with_ruby, interior, 20.0, False)["leftover_px"] == 300
    outlined = erased.copy()
    cv2.ellipse(outlined, (150, 120), (120, 90), 0, 0, 360, 0, 3)  # the balloon's own outline
    assert TM.score_leftover(outlined, interior, 20.0, False)["leftover_px"] == 0
    assert TM.score_leftover(erased, None, 20.0, False)["leftover_px"] is None  # free text


def _block(bg, seed: Rect):
    """A stand-in for ``render.layout.TextBlock`` with the fields the metric reads."""
    member = SimpleNamespace(bbox=seed)
    return SimpleNamespace(style=SimpleNamespace(bg=bg), members=[member], furigana=[], segment=SimpleNamespace(bbox=seed))


def test_free_text_has_no_interior_and_so_no_blocks_per_bubble() -> None:
    gray = np.full((300, 400), 40, np.uint8)
    cv2.ellipse(gray, (110, 150), (95, 90), 0, 0, 360, 255, -1)  # a joined pair: one white region
    cv2.ellipse(gray, (200, 150), (95, 90), 0, 0, 360, 255, -1)
    cv2.circle(gray, (350, 60), 40, 255, -1)  # a balloon of its own, clear of both
    blocks = [_block((255, 255, 255), Rect(80, 130, 24, 60)),
              _block((255, 255, 255), Rect(220, 130, 24, 60)),
              _block((255, 255, 255), Rect(340, 45, 20, 30))]
    assert TM.blocks_per_bubble(blocks) == [None, None, None]  # free text: no interior


def _bubble_block(bg, seed: Rect, box: Rect):
    """A block the layout put in a bubble: ``box`` is its own interior, filled."""
    block = _block(bg, seed)
    block.style.in_bubble = True
    block.style.layout_box = box
    block.style.layout_mask = np.ones((box.h, box.w), bool)
    return block


def test_blocks_of_one_joined_component_count_as_one_once_it_is_cut() -> None:
    """Two balloons the artist joined share a paper component, so the raw count
    says 2; once ``render.bubbles`` has given each block its own interior they
    no longer letter into each other's half and the count is 1.  Stands for
    2ja page 11 and 4ja page 4."""
    gray = np.full((300, 400), 40, np.uint8)
    cv2.ellipse(gray, (110, 150), (95, 90), 0, 0, 360, 255, -1)  # one white region...
    cv2.ellipse(gray, (200, 150), (95, 90), 0, 0, 360, 255, -1)  # ...spanning both balloons
    left = _bubble_block((255, 255, 255), Rect(80, 130, 24, 60), Rect(20, 70, 130, 160))
    right = _bubble_block((255, 255, 255), Rect(220, 130, 24, 60), Rect(160, 70, 130, 160))
    assert TM.blocks_per_bubble([left, right]) == [1, 1]


def test_interiors_that_still_overlap_are_counted_as_shared() -> None:
    """The cut is the whole point: interiors that still reach into each other
    are the defect, and the metric must keep saying so."""
    gray = np.full((300, 400), 40, np.uint8)
    cv2.ellipse(gray, (110, 150), (95, 90), 0, 0, 360, 255, -1)
    cv2.ellipse(gray, (200, 150), (95, 90), 0, 0, 360, 255, -1)
    shared = Rect(20, 70, 270, 160)
    left = _bubble_block((255, 255, 255), Rect(80, 130, 24, 60), shared)
    right = _bubble_block((255, 255, 255), Rect(220, 130, 24, 60), shared)
    assert TM.blocks_per_bubble([left, right]) == [2, 2]


def test_summary_leaves_free_text_out_of_the_geometry_means() -> None:
    rows = [
        {"index": 0, "kind": "bubble", "containment": 0.2, "centre_offset_em": 1.5, "leftover_px": 40,
         "blocks_per_bubble": 3},
        {"index": 1, "kind": "bubble", "containment": 0.0, "centre_offset_em": 0.1, "leftover_px": 0,
         "blocks_per_bubble": 3},
        {"index": 2, "kind": "art", "containment": None, "centre_offset_em": None, "leftover_px": None,
         "blocks_per_bubble": 1},
    ]
    s = TM.summarize(rows)
    assert s["containment"] == pytest.approx(0.1) and s["containment_worst"] == pytest.approx(0.2)
    assert s["uncontained"] == 1
    assert s["centre_offset_em"] == pytest.approx(0.8) and s["centre_offset_worst_em"] == pytest.approx(1.5)
    assert s["leftover_px"] == 40 and s["leftover_blocks"] == 1
    assert s["blocks_per_bubble_max"] == 3 and s["shared_blocks"] == 2


def test_page_table_lists_blocks_and_summary() -> None:
    rows = [
        {"index": 0, "kind": "bubble", "erase_iou": 0.5, "erase_recall": 0.6, "erase_precision": 0.7, "art_kept": 0.8,
         "ssim": 0.9, "mae": 3.0, "cap_ratio": 1.1, "inset_em": 0.3, "inset_ref_em": 0.25, "centre_dx_em": 0.0,
         "centre_dy_em": 0.1, "lines": 3, "lines_ref": 3, "overflow_px": 0, "collision_px": 0,
         "containment": 0.05, "centre_offset_em": 1.2, "leftover_px": 40, "blocks_per_bubble": 2},
        {"index": 1, "kind": "art", "erase_iou": 0.3, "erase_recall": 0.4, "erase_precision": 0.9, "art_kept": 0.6,
         "ssim": 0.8, "mae": 5.0, "cap_ratio": None, "inset_em": None, "inset_ref_em": None, "centre_dx_em": None,
         "centre_dy_em": None, "lines": 2, "lines_ref": None, "overflow_px": 0, "collision_px": 12,
         "containment": None, "centre_offset_em": None, "leftover_px": None, "blocks_per_bubble": 1},
    ]
    summary = TM.summarize(rows)
    assert summary["blocks"] == 2 and summary["erase_iou"] == pytest.approx(0.4)
    assert summary["art_kept"] == pytest.approx(0.7) and summary["ssim"] == pytest.approx(0.85)
    assert summary["cap_ratio"] == pytest.approx(1.1) and summary["collisions"] == 1 and summary["overflows"] == 0
    md = TM.page_table("3jp", "base", rows, summary)
    assert "| # | kind |" in md and "| 0 | bubble |" in md and "| 1 | art |" in md
    assert "0.40" in md and "**mean**" in md
    # The geometry columns: values for the bubble, "-" for the free-text block.
    assert "contain | centre off (em) | leftover px | blk/bubble |" in md
    assert "| 0.050 | 1.20 | 40 | 2 |" in md and "| - | - | - | 1 |" in md
    assert "worst 0.050" in md and "max 2 in 1" in md
    page = TM.summary_table("base", [{"stem": "3jp", "blocks": rows, "summary": summary}])
    assert "contain mean/worst" in page and "0.050/0.050" in page
