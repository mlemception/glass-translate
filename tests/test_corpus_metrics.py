"""Tests for ``demo/corpus_metrics.py`` on synthetic inputs: the dilated text
IoU (and why it is dilated), the grouping F1 over assigned / unassigned English
lines, the resolution-free leftover, the symmetric size log-ratio, the graded
line count, the anti-degenerate answered fraction, the LPIPS floor windows and
excess, the weak-reference guard, and the ``page_components`` / ``page_extras``
glue that feeds ``demo/corpus_score.py``.  No corpus page, no OCR, no GPU, no
network, no torch."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import corpus_metrics as CM  # noqa: E402
import typeset_reference as TR  # noqa: E402

from glasstranslate.core.types import Rect  # noqa: E402

SHAPE = (200, 300)


def _bars(xs: Sequence[int], top: int, bottom: int, width: int, shape: Tuple[int, int] = SHAPE) -> np.ndarray:
    """A row of vertical strokes: one line of lettering, crudely."""
    mask = np.zeros(shape, bool)
    for x in xs:
        mask[top:bottom, x: x + width] = True
    return mask


def _raw_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / float(union) if union else 0.0


# --------------------------------------------------------------- text IoU
def test_text_iou_is_one_for_identical_and_zero_for_disjoint() -> None:
    ink = _bars(range(40, 140, 8), 80, 100, 3)
    assert CM.text_iou(ink, ink.copy(), 20.0) == pytest.approx(1.0)
    far = _bars(range(200, 280, 8), 20, 40, 3)  # another corner, far past the dilation reach
    assert CM.text_iou(ink, far, 20.0) == pytest.approx(0.0)


def test_text_iou_dilation_scores_the_same_sentence_in_another_font() -> None:
    """The release letters in its own typeface.  Two *correct* renders of one
    sentence therefore share almost no glyph pixels - strokes land a few px
    apart and are drawn at another weight - while covering the same text area.
    Undilated, that reads as a total miss; dilated, as the match it is."""
    ours = _bars(range(40, 140, 8), 80, 100, 3)
    theirs = _bars(range(43, 143, 8), 82, 98, 2)  # shifted 3 px, thinner, slightly shorter caps
    assert _raw_iou(ours, theirs) < 0.10  # glyph for glyph: nothing in common
    score = CM.text_iou(ours, theirs, 20.0)
    assert score is not None and score > 0.60  # as text areas: the same sentence, in the same place
    assert score > _raw_iou(ours, theirs) + 0.5


def test_text_iou_is_none_when_either_side_has_no_text() -> None:
    ink = _bars(range(40, 140, 8), 80, 100, 3)
    blank = np.zeros(SHAPE, bool)
    assert CM.text_iou(blank, ink, 20.0) is None  # never 0.0: nothing to compare is not a total miss
    assert CM.text_iou(ink, blank, 20.0) is None
    assert CM.text_iou(blank, blank, 20.0) is None


# --------------------------------------------------------------- grouping
def _line(x: int, y: int, w: int = 40, h: int = 16, text: str = "WORDS",
          confidence: float = 1.0) -> SimpleNamespace:
    """A stand-in for an English ``Segment``: the bbox the metric reads, plus the
    text and confidence ``trustworthy_english`` reads.  A real reference segment
    always carries all three (``typeset_reference._seg_record``)."""
    return SimpleNamespace(bbox=Rect(x, y, w, h), text=text, confidence=confidence)


def test_group_f1_is_one_when_every_balloon_holds_exactly_one_block() -> None:
    owned = [[_line(10, 10)], [_line(10, 60)], [_line(10, 110), _line(10, 130)]]
    out = CM.group_f1(owned, [], [1, 1, None], 20.0)
    assert out == {"group_f1": pytest.approx(1.0), "tp": 3, "fp": 0, "fn": 0}


def test_two_of_our_blocks_in_one_balloon_lower_group_f1_through_fp() -> None:
    owned = [[_line(10, 10)], [_line(10, 60)], [_line(10, 80)]]
    out = CM.group_f1(owned, [], [1, 2, 2], 20.0)
    assert out["tp"] == 1 and out["fp"] == 2 and out["fn"] == 0
    assert out["group_f1"] == pytest.approx(2 / 4.0)
    # A block we lettered where the release lettered nothing is an FP too.
    empty = CM.group_f1([[_line(10, 10)], []], [], [1, 1], 20.0)
    assert empty["tp"] == 1 and empty["fp"] == 1 and empty["group_f1"] == pytest.approx(2 / 3.0)


def test_a_missed_balloon_lowers_group_f1_through_fn() -> None:
    owned = [[_line(10, 10)], [_line(10, 60)]]
    missed = [_line(200, 20), _line(200, 40)]  # one balloon's worth of lines, nobody claimed them
    out = CM.group_f1(owned, missed, [1, 1], 20.0)
    assert out["tp"] == 2 and out["fp"] == 0 and out["fn"] == 1
    assert out["group_f1"] == pytest.approx(4 / 5.0) and out["group_f1"] < 1.0


def test_unassigned_lines_cluster_by_distance_into_one_or_two_misses() -> None:
    owned = [[_line(10, 10)]]
    near = [_line(200, 20), _line(200, 40)]  # centres 20 px apart: one em, one balloon
    far = [_line(200, 20), _line(200, 140)]  # 120 px: six ems, two balloons
    assert CM.group_f1(owned, near, [1], 20.0)["fn"] == 1
    assert CM.group_f1(owned, far, [1], 20.0)["fn"] == 2


def test_untranslated_blocks_are_left_out_of_group_f1() -> None:
    """A logo or an author's name the release leaves in Japanese: lettering it
    is not a win and not lettering it is not a miss."""
    owned = [[_line(10, 10)], []]  # block 1 has no English lines - it is the logo
    assert CM.group_f1(owned, [], [1, None], 20.0)["group_f1"] == pytest.approx(2 / 3.0)
    out = CM.group_f1(owned, [], [1, None], 20.0, untranslated={1})
    assert out == {"group_f1": pytest.approx(1.0), "tp": 1, "fp": 0, "fn": 0}


def test_group_f1_is_none_when_there_is_nothing_to_score() -> None:
    assert CM.group_f1([], [], [], 20.0)["group_f1"] is None


# --------------------------------------------------------------- scalar forms
def test_leftover_em2_is_resolution_free() -> None:
    assert CM.leftover_em2(400, 10.0) == pytest.approx(4.0)
    assert CM.leftover_em2(400, 20.0) == pytest.approx(1.0)  # the same ink at 2x em: a quarter the value
    assert CM.leftover_em2(0, 20.0) == pytest.approx(0.0)
    assert CM.leftover_em2(400, 0.0) is None
    assert CM.leftover_em2(None, 20.0) is None


def test_size_logratio_rms_is_zero_when_perfect_and_symmetric_otherwise() -> None:
    assert CM.size_logratio_rms([1.0, 1.0, 1.0]) == pytest.approx(0.0)
    assert CM.size_logratio_rms([2.0]) == pytest.approx(CM.size_logratio_rms([0.5]))
    assert CM.size_logratio_rms([2.0, 0.5]) == pytest.approx(CM.size_logratio_rms([2.0]))
    assert CM.size_logratio_rms([1.0, None, 0.0, -1.0, float("nan")]) == pytest.approx(0.0)
    assert CM.size_logratio_rms([]) is None
    assert CM.size_logratio_rms([None, 0.0]) is None


def test_line_closeness_counts_only_the_compared_blocks() -> None:
    rows = [{"lines": 2, "lines_ref": 2}, {"lines": 3, "lines_ref": 2}, {"lines": 1, "lines_ref": None}]
    # 1.0 for the exact block, 1 - 1/2 for the one line over: mean 0.75.
    assert CM.line_closeness(rows) == pytest.approx(0.75)
    assert CM.line_closeness(rows + [{"lines": 9, "lines_ref": 2, "untranslated": True}]) == pytest.approx(0.75)
    assert CM.line_closeness([{"lines": 1, "lines_ref": None}]) is None
    assert CM.line_closeness([]) is None


def test_line_closeness_grades_a_near_miss_instead_of_zeroing_the_page() -> None:
    """The v24_p065_3745d5 case: two lines where the release set three.

    Under the old exact-match ``line_exact`` this returned 0.0, and because R is a
    geometric mean that took an otherwise near-perfect page to R = 0.00.  Graded,
    one line out of three costs a third.  An exact match still scores 1.0, and a
    genuinely wrong shape still reaches a true 0.
    """
    assert CM.line_closeness([{"lines": 2, "lines_ref": 3}]) == pytest.approx(2.0 / 3.0)
    assert CM.line_closeness([{"lines": 3, "lines_ref": 3}]) == pytest.approx(1.0)
    assert CM.line_closeness([{"lines": 1, "lines_ref": 5}]) == pytest.approx(0.2)
    # At or past twice the reference count there is no credit left.
    assert CM.line_closeness([{"lines": 6, "lines_ref": 3}]) == pytest.approx(0.0)
    assert CM.line_closeness([{"lines": 9, "lines_ref": 3}]) == pytest.approx(0.0)
    # A reference count of 0 has no scale to divide by, so it is scored by equality.
    assert CM.line_closeness([{"lines": 0, "lines_ref": 0}]) == pytest.approx(1.0)
    assert CM.line_closeness([{"lines": 2, "lines_ref": 0}]) == pytest.approx(0.0)


def test_trustworthy_english_drops_the_noise_and_keeps_the_lettering() -> None:
    """Three definitional rules, measured on the 50-page sample as 28 % of the pile."""
    kept = {"text": "WITHOUT ME.", "bbox": [0, 0, 10, 10], "confidence": 1.0}
    short = {"text": "P!", "bbox": [0, 0, 10, 10], "confidence": 0.95}
    segments = [
        kept,
        short,
        {"text": "00000", "bbox": [0, 0, 10, 10], "confidence": 0.9},    # screentone, no letter
        {"text": "8889", "bbox": [0, 0, 10, 10], "confidence": 0.69},    # ditto
        {"text": "会", "bbox": [0, 0, 10, 10], "confidence": 0.95},  # CJK read off the English page
        {"text": "uそ!", "bbox": [0, 0, 10, 10], "confidence": 0.9},  # ditto, mixed
        {"text": "A11T7", "bbox": [0, 0, 10, 10], "confidence": 0.43},   # OCR says it does not know
    ]
    assert CM.trustworthy_english(segments) == [kept, short]
    # The bar is exactly the one typeset_reference.derive already uses for a block.
    assert CM.ENGLISH_MIN_CONFIDENCE == 0.6
    edge = {"text": "OK", "bbox": [0, 0, 10, 10], "confidence": CM.ENGLISH_MIN_CONFIDENCE}
    assert CM.trustworthy_english([edge]) == [edge]
    assert CM.trustworthy_english([]) == []


def test_noise_clusters_no_longer_count_as_balloons_we_missed() -> None:
    """The end-to-end point of the filter: screentone read as ``00000`` used to be a
    missed balloon in ``group_f1``'s FN term and in ``answered``'s denominator."""
    noise = [{"text": "00000", "bbox": [500, 500, 40, 12], "confidence": 0.9},
             {"text": "省", "bbox": [700, 700, 30, 30], "confidence": 0.49}]
    assert CM.trustworthy_english(noise) == []
    assert CM._cluster_count(noise, 20.0) == 2
    assert CM._cluster_count(CM.trustworthy_english(noise), 20.0) == 0


# --------------------------------------------------------------- answered
def _gt(ink: int) -> Dict[str, Any]:
    return {"stats": {"ink_px": ink, "line_count": 1 if ink else 0}}


def test_answered_fraction_counts_the_lettered_ground_truth_we_answered() -> None:
    records = [_gt(500), _gt(500), _gt(500), _gt(0)]  # the last one the release left alone
    rows = [{"ink_px": 400}, {"ink_px": 300}, {"missed": True}, {"ink_px": 0}]
    assert CM.answered_fraction(records, rows) == pytest.approx(2 / 3.0)


def test_abstaining_scores_zero_and_a_page_with_nothing_to_letter_scores_none() -> None:
    records = [_gt(500), _gt(500)]
    assert CM.answered_fraction(records, [{"missed": True}, {"ink_px": 0}]) == pytest.approx(0.0)
    assert CM.answered_fraction([_gt(0), _gt(0)], [{"ink_px": 0}, {"ink_px": 0}]) is None
    # Untranslated blocks are nobody's to answer.
    assert CM.answered_fraction([_gt(500)], [{"ink_px": 0, "untranslated": True}]) is None


# --------------------------------------------------------------- LPIPS glue
def _hits(window: Tuple[int, int, int, int], box: Sequence[int]) -> bool:
    x, y, w, h = window
    bx, by, bw, bh = box
    return not (x + w <= bx or bx + bw <= x or y + h <= by or by + bh <= y)


def test_text_free_windows_avoid_text_and_stay_on_the_page() -> None:
    boxes = [[0, 0, 800, 220], [300, 400, 200, 120]]
    windows = CM.text_free_windows((600, 800), [Rect(*boxes[0]), boxes[1]], count=4, size=256)
    assert 0 < len(windows) <= 4
    for x, y, w, h in windows:
        assert (w, h) == (256, 256)
        assert 0 <= x <= 800 - 256 and 0 <= y <= 600 - 256
        assert not any(_hits((x, y, w, h), box) for box in boxes)
    assert len({(x, y) for x, y, _, _ in windows}) == len(windows)  # spread, not the same corner four times


def test_text_free_windows_run_out_on_a_page_that_is_all_text() -> None:
    assert CM.text_free_windows((600, 800), [Rect(0, 0, 800, 600)], count=6, size=256) == []
    assert CM.text_free_windows((100, 100), [], count=6, size=256) == []  # no room for a window at all


def test_lpips_excess_subtracts_the_floor_and_clamps_at_zero() -> None:
    assert CM.lpips_excess(0.20, 0.12) == pytest.approx(0.08)
    assert CM.lpips_excess(0.10, 0.12) == pytest.approx(0.0)  # beating the floor is noise, not credit
    assert CM.lpips_excess(None, 0.12) is None
    assert CM.lpips_excess(0.20, None) is None


def test_reference_weak_fires_above_the_threshold_only() -> None:
    assert CM.reference_weak([_line(0, 0)] * 2, 20) is False  # 10 %: rapidocr misses a line here and there
    assert CM.reference_weak([_line(0, 0)] * 5, 20) is True  # 25 %: the reference under-covers the page
    assert CM.reference_weak(5, 20) is True  # a plain count works too
    assert CM.reference_weak([], 0) is False


# --------------------------------------------------------------- components
def _scored(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> Dict[str, Any]:
    return {"stem": "v01p003", "tag": "corpus", "blocks": list(rows), "summary": summary}


def test_page_components_always_carries_every_key() -> None:
    rows = [{"kind": "bubble", "leftover_px": 100, "cap_ratio": 1.0, "lines": 2, "lines_ref": 2}]
    summary = {"containment": 0.0, "erase_iou": 0.8, "art_kept": 0.9, "centre_offset_em": 0.2}
    out = CM.page_components(_scored(rows, summary), {"em_px": [10.0], "answered": 1.0,
                                                      "lpips_art_mean": 0.2, "lpips_floor": 0.1,
                                                      "group_f1": {"group_f1": 0.75}, "text_iou": [0.8, 0.6]})
    assert set(CM.COMPONENT_KEYS) <= set(out) and "answered" in out
    assert out["leftover_em2_mean"] == pytest.approx(1.0) and out["lpips_excess"] == pytest.approx(0.1)
    assert out["group_f1"] == pytest.approx(0.75) and out["text_iou_mean"] == pytest.approx(0.7)
    assert out["line_closeness"] == pytest.approx(1.0) and out["size_logratio_rms"] == pytest.approx(0.0)
    assert out["erase_iou"] == pytest.approx(0.8) and out["art_kept"] == pytest.approx(0.9)


def test_a_page_with_no_balloon_blocks_reports_none_not_zero() -> None:
    """Free text has no interior, so there is nothing to contain, centre or
    leave ink inside: ``page_score`` must renormalise those weights away rather
    than read the page as perfectly contained."""
    rows = [{"kind": "art", "containment": None, "centre_offset_em": None, "leftover_px": None,
             "cap_ratio": 1.2, "lines": 1, "lines_ref": 1}]
    summary = {"containment": None, "centre_offset_em": None, "erase_iou": 0.7, "art_kept": 0.8}
    out = CM.page_components(_scored(rows, summary), {"em_px": [20.0]})
    for key in CM.BALLOON_ONLY_KEYS:
        assert out[key] is None
    assert set(CM.COMPONENT_KEYS) <= set(out)
    assert out["erase_iou"] == pytest.approx(0.7)
    assert out["answered"] == pytest.approx(1.0)  # nothing to answer: the gate stays open


def test_a_weak_reference_blanks_the_metrics_that_depend_on_it() -> None:
    rows = [{"kind": "bubble", "leftover_px": 100, "cap_ratio": 1.0, "lines": 2, "lines_ref": 2}]
    summary = {"containment": 0.0, "erase_iou": 0.8, "art_kept": 0.9, "centre_offset_em": 0.2}
    extras = {"em_px": 10.0, "group_f1": 0.5, "text_iou": [0.8], "reference_weak": True}
    out = CM.page_components(_scored(rows, summary), extras)
    for key in CM.WEAK_REFERENCE_KEYS:
        assert out[key] is None
    assert out["containment_mean"] == pytest.approx(0.0) and out["leftover_em2_mean"] == pytest.approx(1.0)


def test_untranslated_blocks_stay_out_of_the_page_components() -> None:
    rows = [{"kind": "bubble", "leftover_px": 100, "cap_ratio": 1.0, "lines": 2, "lines_ref": 2},
            {"kind": "art", "leftover_px": 900, "cap_ratio": 4.0, "lines": 1, "lines_ref": 3,
             "untranslated": True}]
    out = CM.page_components(_scored(rows, {"erase_iou": 0.8, "art_kept": 0.9}), {"em_px": [10.0, 10.0]})
    assert out["leftover_em2_mean"] == pytest.approx(1.0)
    assert out["size_logratio_rms"] == pytest.approx(0.0) and out["line_closeness"] == pytest.approx(1.0)


# --------------------------------------------------------------- page extras
def _write_page(tmp_path: Path, kind: str) -> Tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    """A one-block synthetic page on disk: the JA scan, our erased and typeset
    renders, and the ground truth whose English lettering sits where ours does."""
    height, width = SHAPE
    ja = np.full(SHAPE, 245, np.uint8)
    cv2.rectangle(ja, (120, 80), (180, 100), 0, -1)  # the Japanese line we are meant to erase
    erased = np.full(SHAPE, 245, np.uint8)
    typeset = erased.copy()
    cv2.rectangle(typeset, (122, 82), (178, 98), 0, -1)  # our lettering
    english = np.zeros(SHAPE, bool)
    english[80:100, 120:180] = True  # the release's lettering, a hair wider
    render_dir = tmp_path / "render"
    render_dir.mkdir()
    ja_path = tmp_path / "page_ja.png"
    cv2.imwrite(str(ja_path), ja)
    cv2.imwrite(str(render_dir / "corpus_erased.png"), erased)
    cv2.imwrite(str(render_dir / "corpus_typeset.png"), typeset)
    (render_dir / "corpus_blocks.json").write_text(json.dumps({"blocks": [{"bbox": [122, 82, 56, 16]}]}), encoding="utf-8")

    record = {"index": 0, "kind": kind, "text": "字", "em_px": 20.0, "dark": False,
              "window": [100, 50, 140, 100], "source_bbox": [120, 80, 60, 20],
              "english_lines": [{"text": "HI", "bbox": [120, 80, 60, 20], "confidence": 0.9}],
              "stats": {"ink_px": 1200, "line_count": 1, "cap_height_px": 16.0}}
    bubble = kind == "bubble"
    row: Dict[str, Any] = {"index": 0, "kind": kind, "ink_px": 900, "overflow_px": 0, "cap_ratio": 1.0,
                           "lines": 1, "lines_ref": 1, "erase_iou": 0.9, "art_kept": 0.95,
                           "containment": 0.0 if bubble else None,
                           "centre_offset_em": 0.2 if bubble else None,
                           "leftover_px": 100 if bubble else None,
                           "blocks_per_bubble": 1 if bubble else None}
    summary = {"erase_iou": 0.9, "art_kept": 0.95, "unpaired_blocks": 0, "missed_blocks": 0,
               "containment": 0.0 if bubble else None, "centre_offset_em": 0.2 if bubble else None}
    truth = {"gt": TR.EraseGroundTruth(np.zeros(SHAPE, bool), np.zeros(SHAPE, bool), english),
             "eng_gray": np.full(SHAPE, 245, np.uint8), "records": [record], "unassigned_english": []}
    return _scored([row], summary), truth, render_dir, ja_path


def test_stylish_onomatopoeia_is_excluded_not_penalised(tmp_path: Path) -> None:
    """SFX the release did not re-letter is out of scope.

    An art-kind block with no English lines is onomatopoeia VIZ left in Japanese.
    Scoring it would reward us on pages where the release left the SFX and punish us on
    pages where it redrew them, for identical behaviour - so it is dropped from the
    scored set entirely rather than counted as a false positive.
    """
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "art")
    truth["records"][0]["english_lines"] = []  # the release left this SFX alone
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus",
                            ja_path=ja_path)
    assert extras["sfx_excluded"] == 1
    assert extras["scored_blocks"] == 0
    # It must NOT become a false positive in the grouping score.
    assert extras["group_fp"] == 0
    assert extras["text_iou_mean"] is None
    assert extras["overflow_px"] == 0 and extras["leftover_px"] == 0


def test_the_sfx_exclusion_survives_into_page_components(tmp_path: Path) -> None:
    """page_components is what feeds the score, so the exclusion has to reach it.

    page_components recomputes line_closeness / size_logratio_rms / leftover_em2_mean from
    scored["blocks"], where it cannot tell an SFX block from a real one - and an SFX
    block's `lines_ref` is 0, not None, so it really would be compared.  The values
    page_extras computed over the in-scope blocks must win.
    """
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "art")
    truth["records"][0]["english_lines"] = []
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus",
                            ja_path=ja_path)
    assert extras["sfx_excluded"] == 1
    components = CM.page_components(scored, extras)
    assert components["line_closeness"] == extras["line_closeness"]
    assert components["size_logratio_rms"] == extras["size_logratio_rms"]
    assert components["leftover_em2_mean"] == extras["leftover_em2_mean"]


def test_page_components_without_extras_still_computes_locally(tmp_path: Path) -> None:
    """The local computation is the fallback for a caller that has no extras."""
    scored, _truth, _render_dir, _ja_path = _write_page(tmp_path, "bubble")
    components = CM.page_components(scored)
    assert components["line_closeness"] == pytest.approx(1.0)
    assert components["size_logratio_rms"] == pytest.approx(0.0)


def test_a_lettered_free_text_block_is_still_scored(tmp_path: Path) -> None:
    """Only UNLETTERED art blocks are SFX; free text the release did translate stays in."""
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "art")
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus",
                            ja_path=ja_path)
    assert extras["sfx_excluded"] == 0
    assert extras["scored_blocks"] == 1
    assert extras["text_iou_mean"] is not None


def test_a_bubble_we_lettered_but_the_release_did_not_is_still_a_false_positive(
        tmp_path: Path) -> None:
    """The SFX exemption is for artwork only - an empty balloon is a real error."""
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "bubble")
    truth["records"][0]["english_lines"] = []
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus",
                            ja_path=ja_path)
    assert extras["sfx_excluded"] == 0
    assert extras["scored_blocks"] == 1
    assert extras["group_fp"] == 1


def test_page_extras_reads_the_render_and_scores_the_page(tmp_path: Path) -> None:
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "bubble")
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus", ja_path=ja_path)
    assert extras["bubble_heavy"] is True and extras["free_text_only"] is False
    assert extras["reference_weak"] is False
    assert extras["overflow_px"] == 0 and isinstance(extras["overflow_px"], int)
    assert extras["leftover_px"] == 100 and isinstance(extras["leftover_px"], int)
    assert extras["group_f1"] == pytest.approx(1.0) and extras["group_tp"] == 1
    assert extras["text_iou_mean"] is not None and extras["text_iou_mean"] > 0.6
    assert extras["leftover_em2_mean"] == pytest.approx(100 / 400.0)
    assert extras["line_closeness"] == pytest.approx(1.0) and extras["size_logratio_rms"] == pytest.approx(0.0)
    assert extras["answered"] == pytest.approx(1.0)
    components = CM.page_components(scored, extras)
    assert set(CM.COMPONENT_KEYS) <= set(components)
    assert components["leftover_em2_mean"] == pytest.approx(extras["leftover_em2_mean"])


def test_page_extras_without_a_sidecar_result_leaves_lpips_absent(tmp_path: Path) -> None:
    """LPIPS needs torch and runs out of process; no result file means the art
    component renormalises away, never that the page scored a perfect 0."""
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "bubble")
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus", ja_path=ja_path)
    assert extras["lpips_excess"] is None and extras["lpips_art_mean"] is None
    assert CM.page_components(scored, extras)["lpips_excess"] is None
    (render_dir / "corpus_lpips.json").write_text(json.dumps({"mean": 0.12, "mean_art": 0.20, "blocks": []}),
                                                  encoding="utf-8")
    with_lpips = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus", ja_path=ja_path)
    assert with_lpips["lpips_excess"] == pytest.approx(0.08)
    assert CM.page_components(scored, with_lpips)["lpips_excess"] == pytest.approx(0.08)


def test_page_extras_of_a_free_text_page_has_no_balloon_components(tmp_path: Path) -> None:
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "art")
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus", ja_path=ja_path)
    assert extras["free_text_only"] is True and extras["bubble_heavy"] is False
    assert extras["leftover_px"] == 0 and extras["leftover_em2_mean"] is None
    components = CM.page_components(scored, extras)
    for key in CM.BALLOON_ONLY_KEYS:
        assert components[key] is None
    assert components["text_iou_mean"] is not None  # free text is still compared as text


def test_page_extras_turns_a_missed_block_into_a_miss(tmp_path: Path) -> None:
    """The ground truth's records are blocks of this same pipeline, so a record
    our render produced nothing for is a balloon the release lettered and we did
    not: its English lines count as an unassigned cluster, not a true positive."""
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "bubble")
    scored["blocks"][0] = {"index": 0, "kind": "bubble", "missed": True}
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus", ja_path=ja_path)
    assert extras["group_tp"] == 0 and extras["group_fn"] == 1
    assert extras["group_f1"] == pytest.approx(0.0)
    assert extras["answered"] == pytest.approx(0.0)  # and abstaining is punished by the gate
    assert extras["text_iou"] == [] and extras["text_iou_mean"] is None


def test_page_extras_counts_an_unpaired_block_of_ours_as_a_false_positive(tmp_path: Path) -> None:
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "bubble")
    scored["summary"]["unpaired_blocks"] = 2  # two blocks we lettered that no ground-truth block claims
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus", ja_path=ja_path)
    assert extras["group_tp"] == 1 and extras["group_fp"] == 2
    assert extras["group_f1"] == pytest.approx(2 / 4.0)


def test_page_extras_flags_a_weak_english_reference(tmp_path: Path) -> None:
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "bubble")
    # Three unassigned lines: over the fraction AND over the absolute-count floor.
    truth["unassigned_english"] = [_line(10, 10), _line(10, 40), _line(10, 70)]
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus", ja_path=ja_path)
    assert extras["reference_weak"] is True and extras["unassigned_en_lines"] == 3
    components = CM.page_components(scored, extras)
    for key in CM.WEAK_REFERENCE_KEYS:
        assert components[key] is None
    # group_f1 must SURVIVE a weak reference: its FN term is the only penalty in the whole
    # score for a balloon we never detected, and unassigned lines are what that produces.
    assert "group_f1" not in CM.WEAK_REFERENCE_KEYS
    assert components["group_f1"] is not None


def test_a_single_dropped_line_on_a_short_page_is_not_a_weak_reference(tmp_path: Path) -> None:
    """One unassigned line on a two-line page is 50 % and means nothing.

    The fraction alone flagged 11 of 50 real pages, six of them on 1-2 lines total, and
    several of those "lines" were rapidocr firing on screentone ("8889", "00000").
    """
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "bubble")
    truth["unassigned_english"] = [_line(10, 10)]
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus", ja_path=ja_path)
    assert extras["unassigned_en_lines"] == 1
    assert extras["reference_weak"] is False


def test_page_extras_survives_a_render_it_cannot_read(tmp_path: Path) -> None:
    scored, truth, render_dir, ja_path = _write_page(tmp_path, "bubble")
    (render_dir / "corpus_typeset.png").unlink()
    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag="corpus", ja_path=ja_path)
    assert extras["text_iou"] == [] and extras["text_iou_mean"] is None
    assert extras["group_f1"] == pytest.approx(1.0)  # the structural scores still hold
    assert isinstance(extras["overflow_px"], int)


def test_component_keys_match_what_corpus_score_consumes() -> None:
    """The nine weighted components of ``corpus_score`` read exactly these ten
    keys (``c_erase`` takes two of them)."""
    import corpus_score as CS

    assert len(CS.COMPONENT_WEIGHTS) == 9
    assert len(CM.COMPONENT_KEYS) == 10 and len(set(CM.COMPONENT_KEYS)) == 10
    rows: List[Dict[str, Any]] = [{"kind": "bubble", "leftover_px": 0, "cap_ratio": 1.0, "lines": 1, "lines_ref": 1}]
    out = CM.page_components(_scored(rows, {"containment": 0.0, "erase_iou": 1.0, "art_kept": 1.0,
                                            "centre_offset_em": 0.0}),
                             {"em_px": 20.0, "group_f1": 1.0, "text_iou": 1.0, "lpips_art_mean": 0.1,
                              "lpips_floor": 0.1, "answered": 1.0})
    components = {
        "c_contain": CS.c_contain(out["containment_mean"]),
        "c_erase": CS.c_erase(out["erase_iou"], out["art_kept"]),
        "c_art": CS.c_art(out["lpips_excess"]),
        "c_centre": CS.c_centre(out["centre_offset_em_mean"]),
        "c_leftover": CS.c_leftover(out["leftover_em2_mean"]),
        "c_size": CS.c_size(out["size_logratio_rms"]),
        "c_group": CS.c_group(out["group_f1"]),
        "c_lines": CS.c_lines(out["line_closeness"]),
        "c_textiou": CS.c_textiou(out["text_iou_mean"]),
    }
    assert set(components) == set(CS.COMPONENT_WEIGHTS)
    assert CS.page_score(components, out["answered"])["R"] == pytest.approx(100.0)
