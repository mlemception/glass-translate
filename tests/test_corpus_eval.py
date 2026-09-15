"""Tests for the corpus evaluation driver: the sampler's determinism under a seed, the
selectors, the raw-metric -> score-component mapping, the shared-ink gate, the triage
report and the baseline payload the regression gate compares.

Synthetic inputs only - no corpus page, no archive, no OCR, no GPU, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import corpus_eval as CE  # noqa: E402
import corpus_score as CS  # noqa: E402


def _pairs(n: int = 40) -> List[Dict[str, Any]]:
    out = []
    for i in range(n):
        volume = i % 5
        out.append({
            "page_id": f"v{volume:02d}_p{i:03d}_{i:06x}",
            "volume": volume,
            "ja_entry": f"ja/{i}.jpg",
            "ja_half": None,
            "en_entry": (f"Jujutsu Kaisen - c{i % 9 + 1:03d} (v{volume:02d}) - p{i:03d} "
                         "[VIZ Media] [Digital] [1r0n].png"),
            "en_half": None,
            "status": "VERIFIED_FAST",
            "combined": 0.8,
        })
    return out


# --------------------------------------------------------------------- sampling
def test_sample_is_identical_for_the_same_seed() -> None:
    pool = _pairs()
    first = CE.sample_pairs(pool, 10, seed=1)
    second = CE.sample_pairs(pool, 10, seed=1)
    assert [p["page_id"] for p in first] == [p["page_id"] for p in second]
    assert len(first) == 10


def test_sample_does_not_depend_on_the_order_of_the_input() -> None:
    """pairs.json order must not change which pages a seed selects."""
    pool = _pairs()
    shuffled = list(reversed(pool))
    assert ([p["page_id"] for p in CE.sample_pairs(pool, 10, seed=7)] ==
            [p["page_id"] for p in CE.sample_pairs(shuffled, 10, seed=7)])


def test_a_different_seed_selects_a_different_sample() -> None:
    pool = _pairs()
    a = {p["page_id"] for p in CE.sample_pairs(pool, 10, seed=1)}
    b = {p["page_id"] for p in CE.sample_pairs(pool, 10, seed=2)}
    assert a != b


def test_sample_larger_than_the_pool_returns_everything_once() -> None:
    pool = _pairs(6)
    got = CE.sample_pairs(pool, 50, seed=1)
    assert len(got) == 6
    assert len({p["page_id"] for p in got}) == 6


def test_sample_is_sorted_so_runs_are_comparable() -> None:
    ids = [p["page_id"] for p in CE.sample_pairs(_pairs(), 12, seed=3)]
    assert ids == sorted(ids)


# --------------------------------------------------------------------- selectors
def test_select_by_volume_chapter_and_page() -> None:
    pool = _pairs()
    by_volume = CE.select_pairs(pool, volumes=[2])
    assert by_volume and all(p["volume"] == 2 for p in by_volume)

    chapter = CE.chapter_of(pool[3])
    by_chapter = CE.select_pairs(pool, chapters=[chapter])
    assert by_chapter and all(CE.chapter_of(p) == chapter for p in by_chapter)

    wanted = pool[5]["page_id"]
    assert [p["page_id"] for p in CE.select_pairs(pool, pages=[wanted])] == [wanted]


def test_select_from_a_pages_file_ignores_blanks_and_comments(tmp_path: Path) -> None:
    pool = _pairs()
    listing = tmp_path / "pages.txt"
    listing.write_text(f"# a comment\n\n{pool[2]['page_id']}\n{pool[9]['page_id']}\n",
                       encoding="utf-8")
    got = CE.select_pairs(pool, pages_file=listing)
    assert {p["page_id"] for p in got} == {pool[2]["page_id"], pool[9]["page_id"]}


def test_selectors_compose_and_can_return_nothing() -> None:
    pool = _pairs()
    assert CE.select_pairs(pool, volumes=[2], pages=[pool[0]["page_id"]]) == []


# --------------------------------------------------------------------- mapping
def test_to_components_maps_every_raw_key_onto_a_score_component() -> None:
    raw = {"containment_mean": 0.0, "erase_iou": 1.0, "art_kept": 1.0,
           "lpips_excess": 0.0, "centre_offset_em_mean": 0.0,
           "leftover_em2_mean": 0.0, "size_logratio_rms": 0.0,
           "group_f1": 1.0, "line_exact": 1.0, "text_iou_mean": 1.0}
    components = CE.to_components(raw)
    assert set(components) == set(CS.COMPONENT_WEIGHTS)
    assert all(v == pytest.approx(1.0) for v in components.values())
    assert CS.page_score(components, 1.0)["R"] == pytest.approx(100.0)


def test_to_components_passes_none_through_so_the_weights_renormalise() -> None:
    raw = {"containment_mean": None, "erase_iou": 1.0, "art_kept": 1.0,
           "lpips_excess": None, "centre_offset_em_mean": None,
           "leftover_em2_mean": None, "size_logratio_rms": 0.0,
           "group_f1": 1.0, "line_exact": 1.0, "text_iou_mean": 1.0}
    components = CE.to_components(raw)
    assert components["c_contain"] is None
    assert components["c_centre"] is None
    assert components["c_leftover"] is None
    assert components["c_art"] is None
    result = CS.page_score(components, 1.0)
    assert sorted(result["missing"]) == ["c_art", "c_centre", "c_contain", "c_leftover"]
    assert result["R"] == pytest.approx(100.0)


def test_to_components_tolerates_a_missing_key() -> None:
    assert CE.to_components({})["c_contain"] is None


# --------------------------------------------------------------------- shared-ink gate
def _art_page() -> np.ndarray:
    page = np.full((300, 220), 245, np.uint8)
    cv2.rectangle(page, (20, 20), (200, 280), 0, 3)
    cv2.rectangle(page, (40, 40), (120, 140), 0, 2)
    cv2.circle(page, (150, 200), 35, 0, 3)
    return page


def test_shared_ink_is_high_when_the_artwork_matches() -> None:
    ja = _art_page()
    en = ja.copy()
    cv2.rectangle(en, (60, 60), (100, 75), 0, -1)  # different lettering, same art
    assert CE.shared_ink_ratio(ja, en) > 0.9


def test_shared_ink_is_low_when_every_mark_was_replaced() -> None:
    """The author-note / bonus-text class gate 5 exists to catch."""
    ja = np.full((300, 220), 245, np.uint8)
    cv2.rectangle(ja, (30, 40), (190, 60), 0, -1)
    cv2.rectangle(ja, (30, 90), (190, 110), 0, -1)
    en = np.full((300, 220), 245, np.uint8)
    cv2.rectangle(en, (30, 200), (190, 220), 0, -1)
    assert CE.shared_ink_ratio(ja, en) < CE.SHARED_INK_MIN


def test_shared_ink_of_a_blank_page_does_not_divide_by_zero() -> None:
    blank = np.full((50, 50), 255, np.uint8)
    assert CE.shared_ink_ratio(blank, blank) == 1.0


# --------------------------------------------------------------------- report / gate
def _rows() -> List[Dict[str, Any]]:
    return [
        {"page_id": "v01_p010_aaaaaa", "status": "scored", "volume": 1, "R": 80.0,
         "components": {"c_contain": 1.0, "c_size": 0.4}, "overflow_px": 0,
         "leftover_px": 0, "bubble_heavy": True, "free_text_only": False,
         "verified_weak": False},
        {"page_id": "v02_p020_bbbbbb", "status": "scored", "volume": 2, "R": 40.0,
         "components": {"c_contain": 0.5, "c_size": 0.2}, "overflow_px": 12,
         "leftover_px": 0, "bubble_heavy": False, "free_text_only": True,
         "verified_weak": False},
        {"page_id": "v03_p030_cccccc", "status": "unverifiable", "reason": "no shared art"},
    ]


def test_report_carries_numbers_and_ids_but_never_pixels(tmp_path: Path,
                                                         monkeypatch) -> None:
    monkeypatch.setattr(CE, "OUT_ROOT", tmp_path)
    rows = _rows()
    summary = CS.corpus_summary([r for r in rows if r["status"] == "scored"])
    path = CE.write_report(rows, summary, "t1", seed=1, sample=3)
    text = path.read_text(encoding="utf-8")
    assert "release-likeness R" in text
    assert "v01_p010_aaaaaa" in text and "v02_p020_bbbbbb" in text
    assert "unverifiable: 1" in text
    # A committed report must never embed image data.
    assert "data:image" not in text and "base64" not in text


def test_report_records_the_gate_verdict(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(CE, "OUT_ROOT", tmp_path)
    rows = _rows()
    summary = CS.corpus_summary([r for r in rows if r["status"] == "scored"])
    gate = {"passed": False, "message": "headline score 60.00 -> 50.00",
            "invariant_regressions": [{"page_id": "v02_p020_bbbbbb",
                                       "invariant": "overflow_px", "was": 0, "now": 12}]}
    text = CE.write_report(rows, summary, "t2", seed=1, sample=3,
                           gate=gate).read_text(encoding="utf-8")
    assert "**FAIL**" in text and "headline score 60.00 -> 50.00" in text
    assert "overflow_px: 0 -> 12" in text


def test_baseline_payload_round_trips_and_feeds_the_gate(tmp_path: Path,
                                                         monkeypatch) -> None:
    monkeypatch.setattr(CE, "OUT_ROOT", tmp_path)
    rows = _rows()
    summary = CS.corpus_summary([r for r in rows if r["status"] == "scored"])
    CE.save_baseline(rows, summary, "base")
    loaded = CE.load_baseline("base")
    assert loaded is not None
    assert set(loaded["pages"]) == {"v01_p010_aaaaaa", "v02_p020_bbbbbb"}
    assert loaded["pages"]["v02_p020_bbbbbb"]["overflow_px"] == 12
    # Comparing a run against itself must pass.
    assert CS.compare_baseline(CE._gate_payload(rows, summary, "base"), loaded)["passed"]


def test_missing_baseline_reads_as_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(CE, "OUT_ROOT", tmp_path)
    assert CE.load_baseline("nope") is None


def test_gate_fails_when_a_clean_page_starts_overflowing(tmp_path: Path,
                                                         monkeypatch) -> None:
    monkeypatch.setattr(CE, "OUT_ROOT", tmp_path)
    rows = _rows()
    summary = CS.corpus_summary([r for r in rows if r["status"] == "scored"])
    baseline = CE._gate_payload(rows, summary, "base")
    # Two pages: one alone is inside measured run-to-run noise (INVARIANT_MIN_PAGES).
    baseline["pages"]["v02_p020_bbbbbb"]["overflow_px"] = 0  # it used to be clean
    baseline["pages"]["v01_p010_aaaaaa"]["uncontained"] = 0
    current = CE._gate_payload(rows, summary, "now")
    current["pages"]["v01_p010_aaaaaa"]["uncontained"] = 4
    verdict = CS.compare_baseline(current, baseline)
    assert not verdict["passed"]
    assert {r["page_id"] for r in verdict["invariant_regressions"]} == {
        "v01_p010_aaaaaa", "v02_p020_bbbbbb"}
