"""Tests for ``demo/corpus_score.py``: the weighted geometric-mean page
score, the nine component helpers, corpus-level aggregation (mean / p05 /
zero-count split by ``bubble_heavy`` vs ``free_text_only``), the markdown
table helpers and the ``compare_baseline`` regression gate.  Synthetic
component numbers only - no corpus page, no image file, no network."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import corpus_score as CS  # noqa: E402

ALL_ONES: dict[str, float] = {key: 1.0 for key in CS.COMPONENT_WEIGHTS}


def _page(r: float | None, **flags: bool) -> dict[str, object]:
    """A minimal ``corpus_summary`` page record."""
    page: dict[str, object] = {"R": r, "free_text_only": False, "bubble_heavy": False, "verified_weak": False}
    page.update(flags)
    return page


def _run(mean_r: float | None, pages: dict[str, dict[str, int]]) -> dict[str, object]:
    """A minimal ``compare_baseline`` run record."""
    return {"mean_R": mean_r, "pages": pages}


# --------------------------------------------------------------- components
@pytest.mark.parametrize(
    "func, arity",
    [
        (CS.c_contain, 1), (CS.c_art, 1), (CS.c_centre, 1), (CS.c_leftover, 1),
        (CS.c_size, 1), (CS.c_group, 1), (CS.c_lines, 1), (CS.c_textiou, 1),
    ],
)
def test_component_helpers_pass_none_through(func, arity: int) -> None:
    assert func(None) is None


def test_c_erase_falls_back_to_erase_iou_when_there_was_no_art_to_keep() -> None:
    """``art_kept`` is None on text over plain paper - there was no artwork to preserve.

    Discarding the whole erase axis there renormalised 0.15 of the weight away on some of
    the best pages in the corpus, all of which had an excellent ``erase_iou``. A measured
    half of the component beats no component.
    """
    assert CS.c_erase(None, 1.0) is None       # nothing measured at all
    assert CS.c_erase(None, None) is None
    assert CS.c_erase(0.92, None) == pytest.approx(0.92)
    assert CS.c_erase(0.0, None) == pytest.approx(0.0)  # erased nothing: still a zero
    assert CS.c_erase(1.0, 0.25) == pytest.approx(0.5)  # both present: geometric mean


def test_component_formulas_hit_their_named_knees() -> None:
    assert CS.c_contain(0.0) == pytest.approx(1.0)
    assert CS.c_contain(CS.CONTAIN_KNEE) == pytest.approx(1.0 / np.e)
    assert CS.c_art(0.0) == pytest.approx(1.0)
    assert CS.c_art(CS.LPIPS_SPAN) == pytest.approx(0.0)
    assert CS.c_centre(0.0) == pytest.approx(1.0)
    assert CS.c_centre(CS.CENTRE_SPAN) == pytest.approx(0.0)
    assert CS.c_leftover(0.0) == pytest.approx(1.0)
    assert CS.c_leftover(CS.LEFTOVER_SCALE) == pytest.approx(1.0 / np.e)
    assert CS.c_size(0.0) == pytest.approx(1.0)
    assert CS.c_size(CS.SIZE_SPAN) == pytest.approx(0.0)
    assert CS.c_size(CS.SIZE_SPAN / 2.0) == pytest.approx(0.5)
    assert CS.c_erase(1.0, 1.0) == pytest.approx(1.0)
    assert CS.c_erase(0.0, 1.0) == pytest.approx(0.0)


def test_identity_components_clamp_out_of_range_input() -> None:
    assert CS.c_group(1.5) == pytest.approx(1.0)
    assert CS.c_lines(-0.5) == pytest.approx(0.0)
    assert CS.c_textiou(0.4) == pytest.approx(0.4)


# --------------------------------------------------------------- page_score
def test_perfect_page_scores_exactly_100() -> None:
    result = CS.page_score(dict(ALL_ONES), 1.0)
    assert result["R"] == pytest.approx(100.0)
    assert result["missing"] == []
    assert sum(result["weights_used"].values()) == pytest.approx(1.0)


# ----------------------------------------------------------- degenerate audit
def test_draw_nothing_zero_answered_scores_zero() -> None:
    """A page we chose not to render at all: every component could be
    perfect, but the anti-degenerate gate alone must zero it."""
    result = CS.page_score(dict(ALL_ONES), 0.0)
    assert result["answered"] == 0.0
    assert result["R"] == 0.0


def test_erase_nothing_zero_iou_scores_zero() -> None:
    """Nothing erased: erase_iou is 0, so c_erase (not answered) is the
    component that is exactly 0 and drives R to 0."""
    components = dict(ALL_ONES)
    components["c_erase"] = CS.c_erase(0.0, 1.0)
    assert components["c_erase"] == 0.0
    result = CS.page_score(components, 1.0)
    assert result["R"] == 0.0


def test_bleach_every_balloon_zero_art_kept_scores_zero() -> None:
    """Everything erased, including the art: art_kept is 0, so c_erase is
    again exactly 0 even though erase_iou (recall) is perfect."""
    components = dict(ALL_ONES)
    components["c_erase"] = CS.c_erase(1.0, 0.0)
    assert components["c_erase"] == 0.0
    result = CS.page_score(components, 1.0)
    assert result["R"] == 0.0


def test_badly_placed_lettering_hurts_hard_without_annihilating_the_page() -> None:
    """``c_centre`` is a BOUNDED ramp that reaches 0 at ``CENTRE_SPAN``, and
    ``CENTRE_SPAN`` is the release's OWN p90 (1.10 em) - a tenth of VIZ's balloons
    sit at or past the point where this component used to zero a whole page.  It
    held five of the eleven R = 0 pages down, every one of them with lettering
    actually drawn: badly placed, not missing.  So it is floored like the other
    non-degenerate components - it must cost a great deal and still leave the page
    rankable against the rest of the bad half.
    """
    assert "c_centre" in CS._FLOORED_COMPONENTS
    components = dict(ALL_ONES)
    components["c_centre"] = CS.c_centre(CS.CENTRE_SPAN * 2.0)
    assert components["c_centre"] == 0.0  # the component itself still reads a true zero
    result = CS.page_score(components, 1.0)
    assert result["R"] > 0.0          # ...but the page stays on the scale
    assert result["R"] < 60.0         # ...and it is punished hard
    # The real degenerate guards are untouched: they live in answered and c_erase.
    assert CS.page_score(dict(ALL_ONES), 0.0)["R"] == 0.0
    bleached = dict(ALL_ONES)
    bleached["c_erase"] = CS.c_erase(1.0, 0.0)
    assert CS.page_score(bleached, 1.0)["R"] == 0.0


def test_one_dot_at_optical_centre_scores_near_zero() -> None:
    """Containment and centring are perfect (the dot sits dead centre of its
    balloon) but a single dot is not a line of lettered text.

    ``c_size`` is a bounded ramp, so a glyph size this wrong is a true 0 and the page is
    zeroed outright - which is the intended behaviour: a render that letters nothing
    legible is not partially correct, whatever its geometry looks like."""
    components = dict(ALL_ONES)
    components["c_size"] = CS.c_size(5.0)  # a wildly wrong glyph size
    components["c_lines"] = 0.001          # no real text line matched
    components["c_textiou"] = 0.001        # negligible overlap with the reference ink
    result = CS.page_score(components, 1.0)
    assert result["components"]["c_size"] == pytest.approx(0.0, abs=1e-4)
    assert result["R"] == pytest.approx(0.0)

    # ...and a merely poor size, rather than an absurd one, must NOT zero the page: the
    # old Gaussian reached 1e-29 there and silently owned the whole score.
    fair = dict(ALL_ONES)
    fair["c_size"] = CS.c_size(0.40)
    fair_result = CS.page_score(fair, 1.0)
    assert 80.0 < fair_result["R"] < 100.0


def test_copy_ja_page_unchanged_scores_zero() -> None:
    """The lazy render: nothing was erased (erase_iou 0) and nothing was
    answered (answered 0) - both mechanisms independently zero the page."""
    components = dict(ALL_ONES)
    components["c_erase"] = CS.c_erase(0.0, 1.0)
    result = CS.page_score(components, 0.0)
    assert result["R"] == 0.0


def test_zero_component_does_not_warn_or_raise() -> None:
    """The 0 ** x guard: a component at exactly 0 must give a clean 0.0, not
    a math.log domain error or a numpy warning."""
    components = dict(ALL_ONES)
    components["c_lines"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = CS.page_score(components, 1.0)
    assert result["R"] == 0.0


# --------------------------------------------------------------- renormalisation
def test_renormalises_over_present_weights_when_three_keys_are_absent() -> None:
    """A page with no balloon blocks: c_contain / c_centre / c_leftover are
    absent, but the page still scores on the rest."""
    components = dict(ALL_ONES)
    components["c_contain"] = None
    components["c_centre"] = None
    components["c_leftover"] = None
    result = CS.page_score(components, 1.0)
    assert set(result["missing"]) == {"c_contain", "c_centre", "c_leftover"}
    assert result["weights_used"].keys() == {"c_erase", "c_art", "c_size", "c_group", "c_lines", "c_textiou"}
    assert sum(result["weights_used"].values()) == pytest.approx(1.0)
    assert result["R"] == pytest.approx(100.0)  # every present component is still 1.0


# --------------------------------------------------------------- missing R
def test_no_components_present_gives_r_none() -> None:
    result = CS.page_score({}, 1.0)
    assert result["R"] is None
    assert result["missing"] == list(CS.COMPONENT_WEIGHTS)


def test_zero_block_page_excluded_from_corpus_summary_mean_never_counted_as_100() -> None:
    pages = [_page(None), _page(50.0)]
    summary = CS.corpus_summary(pages)
    assert summary["mean_R"] == pytest.approx(50.0)


# --------------------------------------------------------------- monotonicity
@pytest.mark.parametrize("key", list(CS.COMPONENT_WEIGHTS))
def test_worsening_any_single_component_strictly_lowers_r(key: str) -> None:
    baseline_r = CS.page_score(dict(ALL_ONES), 1.0)["R"]
    worsened = dict(ALL_ONES)
    worsened[key] = 0.9
    worsened_r = CS.page_score(worsened, 1.0)["R"]
    assert worsened_r < baseline_r


# --------------------------------------------------------------- geometric mean
def test_geometric_mean_a_clean_erase_does_not_buy_back_text_hanging_out_of_every_balloon() -> None:
    """One component (the highest-weighted, c_contain) blown to 0.01 with
    every other component perfect must score far below the arithmetic mean
    of the same nine values: the geometric mean punishes the single bad
    component instead of averaging it away."""
    components = dict(ALL_ONES)
    components["c_contain"] = 0.01
    result = CS.page_score(components, 1.0)
    arithmetic_mean_r = 100.0 * ((len(components) - 1) * 1.0 + 0.01) / len(components)
    assert arithmetic_mean_r == pytest.approx(89.0, abs=0.1)
    assert result["R"] < arithmetic_mean_r / 2.0


# --------------------------------------------------------------- corpus_summary
def test_corpus_summary_splits_bubble_heavy_vs_free_text_only_and_excludes_verified_weak_from_p05() -> None:
    pages = [
        _page(90.0, bubble_heavy=True),
        _page(0.0, bubble_heavy=True),
        _page(10.0, bubble_heavy=True, verified_weak=True),  # accepted-weak outlier
        _page(80.0, free_text_only=True),
        _page(20.0, free_text_only=True),
    ]
    summary = CS.corpus_summary(pages)

    assert summary["bubble_heavy"]["mean_R"] == pytest.approx((90.0 + 0.0 + 10.0) / 3.0)
    assert summary["bubble_heavy"]["zero_count"] == 1
    assert summary["free_text_only"]["mean_R"] == pytest.approx((80.0 + 20.0) / 2.0)
    assert summary["free_text_only"]["zero_count"] == 0

    # verified_weak counts toward mean_R (10.0 is included above) but is
    # dropped from the bad-tail p05: only [90.0, 0.0] remain for the tail.
    expected_bubble_p05 = float(np.percentile([0.0, 90.0], 5))
    assert summary["bubble_heavy"]["p05_R"] == pytest.approx(expected_bubble_p05)

    overall_tail = [page["R"] for page in pages if not page["verified_weak"]]
    assert summary["p05_R"] == pytest.approx(float(np.percentile(sorted(overall_tail), 5)))
    assert summary["mean_R"] == pytest.approx(sum(page["R"] for page in pages) / len(pages))


# --------------------------------------------------------------- markdown
def test_component_table_md_lists_every_component_with_weight_and_contribution() -> None:
    pages = [_page(80.0, components={key: 0.5 for key in CS.COMPONENT_WEIGHTS})]
    summary = CS.corpus_summary(pages)
    table = CS.component_table_md(summary)
    assert "| component | mean | weight | contribution |" in table
    for key, weight in CS.COMPONENT_WEIGHTS.items():
        assert f"| {key} | 0.500 | {weight:.2f} | {weight * 0.5:.3f} |" in table


def test_distribution_md_lists_deciles_and_worst_best_page_ids() -> None:
    pages = [
        {"R": 10.0, "page_id": "v08/p001"},
        {"R": 90.0, "page_id": "v08/p002"},
        {"R": None, "page_id": "v08/p003"},  # excluded: no R
    ]
    table = CS.distribution_md(pages)
    assert "| decile | R |" in table
    assert "worst: v08/p001 (10.00); best: v08/p002 (90.00)" in table


def test_distribution_md_of_no_scored_pages_does_not_crash() -> None:
    table = CS.distribution_md([{"R": None}])
    assert "No scored pages." in table


# --------------------------------------------------------------- compare_baseline
def test_compare_baseline_passes_on_identical_run() -> None:
    baseline = _run(60.0, {"p1": {"overflow_px": 0, "leftover_px": 0}})
    current = _run(60.0, {"p1": {"overflow_px": 0, "leftover_px": 0}})
    result = CS.compare_baseline(current, baseline)
    assert result["passed"] is True
    assert result["headline_drop"] is False
    assert result["invariant_regressions"] == []


def test_compare_baseline_fails_on_headline_drop_beyond_tolerance_message_has_both_numbers() -> None:
    baseline = _run(61.20, {})
    current = _run(58.94, {})
    result = CS.compare_baseline(current, baseline)
    assert result["headline_drop"] is True
    assert result["passed"] is False
    assert "61.20" in result["message"] and "58.94" in result["message"]


# The render-to-render spread measured in docs/perf/2026-09-15-typeset-corpus.md.
MEASURED_SPREAD_R = 0.0335


def test_tolerance_sits_above_the_measured_spread_and_far_below_the_old_one() -> None:
    """The gate has to be able to see a real slice.

    ``TOLERANCE`` was 1.0, set from "two identical runs scored 45.72 and 46.48".
    That 0.76 was an artefact: the second run never had its LPIPS pass, so
    ``c_art`` was None and the other eight weights renormalised.  The true spread
    is 0.0335 R.  Keep headroom over it - the LPIPS pass is not bit-deterministic -
    but never so much that an improvement is indistinguishable from noise.
    """
    assert MEASURED_SPREAD_R < CS.TOLERANCE <= 0.30


def test_compare_baseline_fails_on_a_drop_the_old_tolerance_hid() -> None:
    """A half-point drop is 15x the measured spread; TOLERANCE = 1.0 waved it through."""
    baseline = _run(46.21, {})
    current = _run(45.71, {})
    result = CS.compare_baseline(current, baseline)
    assert result["headline_drop"] is True
    assert result["passed"] is False


def test_compare_baseline_passes_on_drop_within_tolerance() -> None:
    baseline = _run(60.0, {})
    current = _run(60.0 - CS.TOLERANCE, {})  # exactly at the edge: not "beyond" tolerance
    result = CS.compare_baseline(current, baseline)
    assert result["headline_drop"] is False
    assert result["passed"] is True


def test_compare_baseline_reports_but_tolerates_a_single_invariant_page() -> None:
    """One page is inside measured run-to-run noise - see INVARIANT_MIN_PAGES.

    Two identical runs of the real sample moved 3 of 200 integer counters, so a lone page
    is not evidence. It is still reported, just not failed on.
    """
    baseline = _run(70.0, {"v08/p012": {"overflow_px": 0, "leftover_px": 0}})
    current = _run(70.0, {"v08/p012": {"overflow_px": 143, "leftover_px": 0}})
    result = CS.compare_baseline(current, baseline)
    assert result["invariant_regressions"] == [
        {"page_id": "v08/p012", "invariant": "overflow", "was": 0, "now": 143}
    ]
    assert result["passed"] is True


def test_compare_baseline_fails_on_overflow_invariant_regression_with_page_id_in_message() -> None:
    baseline = _run(70.0, {"v08/p012": {"overflow_px": 0, "leftover_px": 0},
                           "v08/p013": {"overflow_px": 0, "leftover_px": 0}})
    current = _run(70.0, {"v08/p012": {"overflow_px": 143, "leftover_px": 0},
                          "v08/p013": {"overflow_px": 22, "leftover_px": 0}})
    result = CS.compare_baseline(current, baseline)
    assert result["passed"] is False
    assert result["invariant_regressions"] == [
        {"page_id": "v08/p012", "invariant": "overflow", "was": 0, "now": 143},
        {"page_id": "v08/p013", "invariant": "overflow", "was": 0, "now": 22},
    ]
    assert "v08/p012 overflow 0 -> 143" in result["message"]


def test_compare_baseline_fails_on_leftover_invariant_regression_with_page_id_in_message() -> None:
    baseline = _run(70.0, {"v08/p012": {"overflow_px": 0, "leftover_px": 0},
                           "v08/p013": {"overflow_px": 0, "leftover_px": 0}})
    current = _run(70.0, {"v08/p012": {"overflow_px": 0, "leftover_px": 55},
                          "v08/p013": {"overflow_px": 0, "leftover_px": 7}})
    result = CS.compare_baseline(current, baseline)
    assert result["passed"] is False
    assert result["invariant_regressions"] == [
        {"page_id": "v08/p012", "invariant": "leftover", "was": 0, "now": 55},
        {"page_id": "v08/p013", "invariant": "leftover", "was": 0, "now": 7},
    ]
    assert "v08/p012" in result["message"] and "55" in result["message"]


def test_compare_baseline_message_reports_regression_count_and_first_page_in_order() -> None:
    baseline = _run(70.0, {
        "v08/p012": {"overflow_px": 0, "leftover_px": 0},
        "v08/p013": {"overflow_px": 0, "leftover_px": 0},
    })
    current = _run(70.0, {
        "v08/p012": {"overflow_px": 143, "leftover_px": 0},
        "v08/p013": {"overflow_px": 12, "leftover_px": 0},
    })
    result = CS.compare_baseline(current, baseline)
    assert len(result["invariant_regressions"]) == 2
    assert result["message"] == "2 pages regressed a hard invariant: v08/p012 overflow 0 -> 143"


def test_compare_baseline_only_current_and_only_baseline_pages_do_not_fail_it() -> None:
    baseline = _run(70.0, {"p_old": {"overflow_px": 0, "leftover_px": 0}})
    current = _run(70.0, {"p_new": {"overflow_px": 0, "leftover_px": 0}})
    result = CS.compare_baseline(current, baseline)
    assert result["passed"] is True
    assert result["only_current"] == ["p_new"]
    assert result["only_baseline"] == ["p_old"]
