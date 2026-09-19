"""The pure judging layer of the glass profiling driver (``tools/glass_report.py``).

Everything here works on an event list alone, so a JSONL recorded by any run can be
re-judged later without Qt, a window or the machine that produced it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from glasstranslate.ui.glass import profile as P
from tools import glass_report as G

Event = Tuple[float, str, Dict[str, Any]]


def _phase(t: float, name: str, edge: str, rep: int = 1) -> Event:
    return (t, "phase", {"name": name, "edge": edge, "rep": rep})


# ------------------------------------------------------------------------------ gap stats
# A 1000 ms "frame" keeps every delta exactly representable, so the bucket edges are tested
# and not the binary expansion of 0.015.
SLOW_FRAME_MS = 1000.0


def test_gap_stats_buckets_deltas_and_never_crosses_windows() -> None:
    stats = G.gap_stats([[0.0, 1.0, 3.0], [10.0, 20.0]], frame_ms=SLOW_FRAME_MS)
    assert stats["n"] == 3  # the 7 s between the two windows is not a delta
    assert stats["p50_ms"] == pytest.approx(2000.0) and stats["p90_ms"] == pytest.approx(10000.0)
    assert stats["largest_ms"] == pytest.approx(10000.0) and stats["largest_frames"] == pytest.approx(10.0)
    assert stats["over"] == {"2.0": 1, "2.5": 1}
    assert stats["histogram"] == {"<=1.5": 1, "1.5-2.0": 1, "2.0-2.5": 0, "2.5-4.0": 0, "4.0-8.0": 0, ">8.0": 1}


def test_gap_stats_upper_bucket_edges_are_inclusive() -> None:
    # exactly 1.5, 2.0, 2.5, 4.0, 8.0 and 9.0 frame intervals, one window each
    stats = G.gap_stats([[0.0, gap] for gap in (1.5, 2.0, 2.5, 4.0, 8.0, 9.0)], frame_ms=SLOW_FRAME_MS)
    assert stats["histogram"] == {"<=1.5": 1, "1.5-2.0": 1, "2.0-2.5": 1, "2.5-4.0": 1, "4.0-8.0": 1, ">8.0": 1}
    assert stats["over"] == {"2.0": 4, "2.5": 3}


def test_gap_stats_takes_extra_factors() -> None:
    stats = G.gap_stats([[0.0, 3.0]], frame_ms=SLOW_FRAME_MS, factors=(2.0, 2.5, 4.0))
    assert stats["over"] == {"2.0": 1, "2.5": 1, "4.0": 0}


def test_gap_stats_is_all_zeros_without_deltas_or_a_frame_time() -> None:
    zero = G.gap_stats([], frame_ms=10.0)
    assert zero["n"] == 0 and zero["largest_ms"] == 0.0 and zero["largest_frames"] == 0.0
    assert zero["over"] == {"2.0": 0, "2.5": 0}
    assert set(zero["histogram"]) == set(G.gap_stats([[0.0, 0.01]], 10.0)["histogram"])
    assert all(v == 0 for v in zero["histogram"].values())
    assert G.gap_stats([[0.0, 0.01]], frame_ms=0.0) == zero
    assert G.gap_stats([[0.5]], frame_ms=10.0) == zero  # a single timestamp has no delta


# ------------------------------------------------------------------------------- windows
def test_phase_windows_pair_by_repetition() -> None:
    events = [_phase(0.0, "drag", "start", 1), _phase(1.0, "drag", "stop", 1),
              _phase(2.0, "tabs", "start", 1), _phase(3.0, "tabs", "stop", 1),
              _phase(4.0, "drag", "start", 2), _phase(5.0, "drag", "stop", 2),
              _phase(6.0, "drag", "start", 3)]  # never closed: not a window
    assert G.phase_windows(events, "drag") == [(0.0, 1.0, 1), (4.0, 5.0, 2)]
    assert G.phase_windows(events, "tabs") == [(2.0, 3.0, 1)]
    assert G.phase_windows(events, "idle") == []


def test_tab_windows_only_cover_switches_inside_the_tabs_phase() -> None:
    events = [_phase(2.0, "tabs", "start", 1), (2.1, "tab", {"index": 1}), (2.5, "tab", {"index": 2}),
              _phase(3.0, "tabs", "stop", 1), (3.5, "tab", {"index": 3})]
    windows = G.tab_windows(events, window_ms=200.0)
    assert [w[0] for w in windows] == [2.1, 2.5]
    assert [round(w[1], 6) for w in windows] == [2.3, 2.7]


# ------------------------------------------------------------------------------ per page
def _page_events() -> List[Event]:
    return [
        (0.0, "overlay_apply_ms", {"ms": 1.0, "blocks": 2, "fresh": 2}),
        # The pipeline's next "nothing changed" result is delivered in the same event-loop turn,
        # BEFORE the page is painted: it neither closes the page's span nor counts as a page.
        (0.0002, "overlay_apply_ms", {"ms": 0.1, "blocks": 2, "fresh": 0}),
        (0.1, "overlay_typeset_ms", {"ms": 10.0}),
        (0.2, "overlay_typeset_ms", {"ms": 20.0}),
        (0.3, "overlay_layer_ms", {"ms": 5.0}),
        (0.4, "paint_ms", {"ms": 3.0}),
        (0.5, "paint_ms", {"ms": 7.0}),
        (2.0, "overlay_apply_ms", {"ms": 2.0, "blocks": 1, "fresh": 1}),
        (2.1, "overlay_typeset_ms", {"ms": 4.0}),
        (2.2, "overlay_layer_ms", {"ms": 6.0}),
        (2.3, "paint_ms", {"ms": 8.0}),
    ]


def test_marks_per_page_sums_the_span_after_each_fresh_apply() -> None:
    pages = G.marks_per_page(_page_events())
    assert pages["pages"] == 2
    assert (pages["typeset_ms"]["n"], pages["typeset_ms"]["max"]) == (2, pytest.approx(30.0))
    assert pages["layer_ms"]["max"] == pytest.approx(6.0)
    assert (pages["layer_ms"]["n"], pages["layer_ms"]["p50"]) == (2, pytest.approx(5.0))
    assert pages["paint_ms"]["max"] == pytest.approx(8.0)  # page 1's largest paint is 7 ms
    assert pages["apply_ms"]["max"] == pytest.approx(2.0)  # the fresh=0 apply is not a page
    assert pages["blocks"]["max"] == 2


def test_marks_per_page_reports_the_fresh_counts() -> None:
    """A partial re-read shows up as a page with fewer fresh segments than blocks."""
    pages = G.marks_per_page(_page_events())
    assert pages["fresh"]["n"] == 2 and pages["fresh"]["max"] == 2.0 and pages["fresh"]["p50"] == 1.0


def test_marks_per_page_without_any_page() -> None:
    pages = G.marks_per_page([(0.0, "paint_ms", {"ms": 4.0})])
    assert pages["pages"] == 0 and pages["typeset_ms"]["n"] == 0 and pages["paint_ms"]["max"] == 0.0
    assert pages["fresh"]["n"] == 0


# ------------------------------------------------------------------------------- analyse
META = {"mode": "control", "refresh_hz": 100.0, "frame_ms": 10.0, "steps": 2, "step_px": 8,
        "repeat": 1, "heartbeat_ms": 4, "backend": None, "ocr_engine": "mangaocr",
        "capacity": 200000, "dropped": 0, "overlay": None}


def _run_events() -> List[Event]:
    events: List[Event] = [
        _phase(0.0, "drag", "start"), _phase(0.05, "drag", "stop"),
        (0.00, "swap", {}), (0.01, "swap", {}), (0.02, "swap", {}), (0.05, "swap", {}),
        (0.000, "tick", {}), (0.004, "tick", {}), (0.008, "tick", {}),
        _phase(1.0, "tabs", "start"), _phase(1.5, "tabs", "stop"),
        (1.0, "tab", {"index": 1}), (1.00, "swap", {}), (1.01, "swap", {}), (1.02, "swap", {}),
        (1.3, "page", {"index": 1, "name": "1ja.jpg"}),
        _phase(2.0, "idle", "start"), _phase(2.5, "idle", "stop"),
        (2.1, "swap", {}), (2.2, "swap", {}),
        (3.0, "page", {"index": 0, "name": "before.jpg"}),  # outside every phase
        (100.0, "meta", dict(META)),
    ]
    return events + _page_events()


def test_analyse_reads_the_meta_event_and_splits_the_phases() -> None:
    record = G.analyse(_run_events())
    assert record["meta"]["mode"] == "control" and record["meta"]["frame_ms"] == 10.0
    assert record["events"] == len(_run_events()) and record["dropped"] == 0
    assert record["swap"]["drag"]["n"] == 3 and record["swap"]["tabs"]["n"] == 2
    assert record["swap"]["all"]["n"] == 5
    assert record["swap"]["all"]["largest_frames"] == pytest.approx(3.0)  # the 30 ms drag gap
    assert record["tick"]["drag"]["n"] == 2 and record["tick"]["all"]["n"] == 2
    assert record["tick"]["tabs"]["n"] == 0 and record["tick"]["idle"]["n"] == 0
    assert record["swaps_per_tab_window"]["n"] == 1 and record["swaps_per_tab_window"]["max"] == 3
    assert record["idle_swaps"] == 2
    assert record["page_swaps"] == 1  # only the page change inside the tabs phase
    assert record["pages"]["pages"] == 2
    assert record["gap_factor"] == 2.5
    assert record["over_gap_factor"] == {"swap": 1, "tick": 0}


def test_analyse_honours_a_custom_gap_factor() -> None:
    record = G.analyse(_run_events(), gap_factor=4.0)
    assert record["gap_factor"] == 4.0 and record["over_gap_factor"]["swap"] == 0
    assert "4.0" in record["swap"]["all"]["over"] and "2.0" in record["swap"]["all"]["over"]


def test_the_gap_factor_is_rounded_before_de_duplicating() -> None:
    """2.04 formats to the same ``2.0`` key as the standard factor; counting it twice would
    double every gap in that bucket."""
    events = [(0.0, "phase", {"name": "drag", "edge": "start", "rep": 1}),
              (1.0, "phase", {"name": "drag", "edge": "stop", "rep": 1}),
              (0.0, "swap", {}), (0.021, "swap", {}),  # exactly 2.1 frame intervals at 10 ms
              (9.0, "meta", dict(META))]
    record = G.analyse(events, gap_factor=2.04)
    assert record["gap_factor"] == 2.0
    assert record["swap"]["all"]["over"] == {"2.0": 1, "2.5": 0}
    assert record["over_gap_factor"]["swap"] == 1


def test_analyse_needs_a_meta_event_with_a_usable_frame_time() -> None:
    with pytest.raises(ValueError, match="meta"):
        G.analyse([(0.0, "swap", {})])
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="frame_ms"):
            G.analyse([(0.0, "meta", {"mode": "control", "frame_ms": bad})])


# A 10 ms frame: a tick gap above 20 ms is late.  Ticks every 4 ms except where a stall sits.
def _stall_events() -> List[Event]:
    ticks = [0.000, 0.004, 0.008,
             0.108,             # 100 ms late: a 98 ms paint ended inside it
             0.112, 0.162,      # 50 ms late: a garbage collection
             0.166, 0.196,      # 30 ms late: a new page was delivered right after it
             0.200, 0.300,      # 100 ms late: nothing of ours ran
             0.304]
    return [(t, "tick", {}) for t in ticks] + [
        (0.107, "paint_ms", {"ms": 98.0}),
        (0.150, "gc", {"ms": 40.0, "gen": 2, "thread": "pipeline"}),
        (0.197, "overlay_apply_ms", {"ms": 0.3, "blocks": 9, "fresh": 9}),
        (0.2005, "paint_ms", {"ms": 0.4}),  # a repaint too small to explain anything
        (100.0, "meta", dict(META)),
    ]


def test_stall_causes_attributes_every_late_tick() -> None:
    stalls = G.stall_causes(_stall_events(), frame_ms=10.0)
    assert stalls["late_ticks"] == 4
    assert stalls["causes"]["paint"] == {"n": 1, "total_ms": pytest.approx(100.0), "largest_ms": pytest.approx(100.0)}
    assert stalls["causes"]["gc"]["n"] == 1 and stalls["causes"]["gc"]["largest_ms"] == pytest.approx(50.0)
    assert stalls["causes"]["delivery"]["n"] == 1 and stalls["causes"]["delivery"]["largest_ms"] == pytest.approx(30.0)
    assert stalls["causes"]["unattributed"]["n"] == 1
    assert stalls["causes"]["unattributed"]["largest_ms"] == pytest.approx(100.0)


def test_stall_causes_ignores_a_repaint_that_only_followed_the_stall() -> None:
    """Marks are stamped at the END of their span.  A stall is always followed by a repaint, so a
    mark that merely lands near the gap explains nothing: the span itself has to overlap it."""
    events = [(t, "tick", {}) for t in (0.000, 0.004, 0.055, 0.059)] + [
        (0.0550, "gc", {"ms": 25.0, "gen": 2}),   # ran 0.030 .. 0.055: inside the late gap
        (0.0585, "paint_ms", {"ms": 3.0}),        # ran 0.0555 .. 0.0585: entirely after it
    ]
    causes = G.stall_causes(events, frame_ms=10.0)["causes"]
    assert causes["gc"]["n"] == 1 and causes["paint"]["n"] == 0


def test_stall_causes_charges_a_paint_only_to_the_gap_it_really_covers() -> None:
    events = [(t, "tick", {}) for t in (0.000, 0.004, 0.044, 0.080)] + [
        (0.047, "paint_ms", {"ms": 40.0}),        # ran 0.007 .. 0.047: 37 ms of gap one, 3 ms of gap two
    ]
    causes = G.stall_causes(events, frame_ms=10.0)["causes"]
    assert causes["paint"]["n"] == 1 and causes["paint"]["largest_ms"] == pytest.approx(40.0)
    assert causes["unattributed"]["n"] == 1      # less than a frame of overlap explains no missed frame


def test_stall_causes_prefers_the_paint_over_a_collection_inside_it() -> None:
    events = [(0.0, "tick", {}), (0.1, "tick", {}), (0.05, "gc", {"ms": 5.0, "gen": 0}),
              (0.099, "paint_ms", {"ms": 95.0})]
    causes = G.stall_causes(events, frame_ms=10.0)["causes"]
    assert causes["paint"]["n"] == 1 and causes["gc"]["n"] == 0


def test_stall_causes_is_empty_without_late_ticks_or_a_frame_time() -> None:
    quiet = [(0.000, "tick", {}), (0.004, "tick", {}), (0.008, "tick", {})]
    for stalls in (G.stall_causes(quiet, frame_ms=10.0), G.stall_causes(_stall_events(), frame_ms=0.0)):
        assert stalls["late_ticks"] == 0
        assert set(stalls["causes"]) == {"paint", "gc", "delivery", "unattributed"}
        assert all(c == {"n": 0, "total_ms": 0.0, "largest_ms": 0.0} for c in stalls["causes"].values())


def test_analyse_carries_the_stall_causes_and_the_collections() -> None:
    record = G.analyse(_stall_events())
    assert record["stalls"]["late_ticks"] == 4
    assert record["gc"]["n"] == 1 and record["gc"]["max"] == pytest.approx(40.0)


def test_analyse_marks_a_finished_run_complete() -> None:
    assert G.analyse(_run_events())["complete"] is True


def test_analyse_marks_a_hard_stopped_run_incomplete() -> None:
    assert G.analyse(_run_events() + [(8.0, "hard_stop", {})])["complete"] is False


def test_analyse_marks_an_unclosed_phase_incomplete() -> None:
    cut_short = _run_events() + [(8.0, "phase", {"name": "drag", "edge": "start", "rep": 2})]
    assert G.analyse(cut_short)["complete"] is False


# ---------------------------------------------------------------------------------- gate
def _stats(n: int, largest_frames: float, over2: int) -> Dict[str, Any]:
    return {"n": n, "p50_ms": 0.0, "p90_ms": 0.0, "largest_ms": 0.0, "largest_frames": largest_frames,
            "over": {"2.0": over2, "2.5": 0}, "histogram": {}}


def _record(swap_largest: float = 1.0, swap_over2: int = 0, swap_n: int = 6, tick_n: int = 4,
            tick_largest: float = 1.0, tick_over2: int = 0, dropped: int = 0,
            complete: bool = True) -> Dict[str, Any]:
    return {"meta": dict(META),
            "events": 42,
            "swap": {"all": _stats(swap_n, swap_largest, swap_over2)},
            "tick": {"all": _stats(tick_n, tick_largest, tick_over2)},
            "dropped": dropped,
            "complete": complete,
            "stalls": G.stall_causes([], frame_ms=10.0),
            "gc": P.percentiles([]),
            "pages": {"pages": 1, "apply_ms": P.percentiles([1.0]), "typeset_ms": P.percentiles([2.0]),
                      "layer_ms": P.percentiles([3.0]), "paint_ms": P.percentiles([4.0]),
                      "blocks": P.percentiles([5.0]), "fresh": P.percentiles([5.0])}}


def _check(gate: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(c for c in gate["checks"] if c["name"] == name)


def test_judge_gate_passes_an_equal_run() -> None:
    gate = G.judge_gate(_record(), _record())
    assert gate["pass"] is True
    assert [c["name"] for c in gate["checks"]] == ["swap_largest", "swap_over_2", "tick_largest",
                                                   "tick_over_2", "measured_enough", "complete",
                                                   "no_dropped_events"]
    assert not any(c.get("skipped") for c in gate["checks"])


def test_judge_gate_allows_one_extra_frame_but_not_two() -> None:
    assert G.judge_gate(_record(swap_largest=2.0), _record(swap_largest=3.0))["pass"] is True
    gate = G.judge_gate(_record(swap_largest=2.0), _record(swap_largest=3.5))
    assert gate["pass"] is False
    assert _check(gate, "swap_largest") == {"name": "swap_largest", "ok": False, "observed": 3.5, "limit": 3.0}


def test_judge_gate_counts_the_two_frame_gaps() -> None:
    assert G.judge_gate(_record(swap_over2=3), _record(swap_over2=3))["pass"] is True
    gate = G.judge_gate(_record(swap_over2=1), _record(swap_over2=2))
    assert gate["pass"] is False and _check(gate, "swap_over_2")["observed"] == 2
    gate = G.judge_gate(_record(tick_over2=1), _record(tick_over2=9))
    assert gate["pass"] is False and _check(gate, "tick_over_2")["ok"] is False
    gate = G.judge_gate(_record(tick_largest=1.0), _record(tick_largest=5.0))
    assert gate["pass"] is False and _check(gate, "tick_largest")["ok"] is False


@pytest.mark.parametrize("run_ticks", [0, 4])
def test_judge_gate_skips_the_tick_checks_when_the_baseline_has_none(run_ticks: int) -> None:
    """No heartbeat in the baseline means there is nothing to compare the run's ticks against."""
    gate = G.judge_gate(_record(tick_n=0, tick_largest=9.0),
                        _record(tick_n=run_ticks, tick_largest=9.0, tick_over2=9))
    assert gate["pass"] is True
    for name in ("tick_largest", "tick_over_2"):
        assert _check(gate, name)["ok"] is True and _check(gate, name)["skipped"] is True


def test_judge_gate_fails_when_only_the_run_lost_its_heartbeat() -> None:
    """A run whose GUI thread never ticked measured nothing: that is a failure, not a skip."""
    gate = G.judge_gate(_record(tick_n=40), _record(tick_n=0, tick_largest=0.0))
    assert gate["pass"] is False
    assert not _check(gate, "tick_largest").get("skipped")
    assert _check(gate, "measured_enough")["ok"] is False


def test_judge_gate_needs_the_run_to_have_measured_half_the_baseline() -> None:
    baseline = _record(swap_n=100, tick_n=100)
    assert G.judge_gate(baseline, _record(swap_n=50, tick_n=50))["pass"] is True
    gate = G.judge_gate(baseline, _record(swap_n=49, tick_n=100))
    assert gate["pass"] is False and _check(gate, "measured_enough")["ok"] is False
    assert _check(gate, "measured_enough")["observed"] == {"swap": 49, "tick": 100}
    assert _check(gate, "measured_enough")["limit"] == {"swap": 50.0, "tick": 50.0}
    gate = G.judge_gate(baseline, _record(swap_n=100, tick_n=49))
    assert gate["pass"] is False and _check(gate, "measured_enough")["ok"] is False
    # a baseline without ticks puts no floor under the run's tick count
    gate = G.judge_gate(_record(swap_n=100, tick_n=0), _record(swap_n=100, tick_n=0))
    assert gate["pass"] is True and _check(gate, "measured_enough")["limit"]["tick"] is None


def test_judge_gate_fails_a_frozen_run() -> None:
    """Nothing presented, nothing ticked: the gap checks are vacuously fine and must not pass."""
    frozen = _record(swap_n=0, swap_largest=0.0, tick_n=0, tick_largest=0.0)
    gate = G.judge_gate(_record(swap_n=100, tick_n=100), frozen)
    assert gate["pass"] is False
    # "largest gap of nothing" is 0 frames and would clear every limit: the n == 0 guard and
    # measured_enough are what actually catch it.
    assert _check(gate, "swap_largest")["ok"] is False
    assert _check(gate, "tick_largest")["ok"] is False
    assert _check(gate, "measured_enough")["ok"] is False


def test_judge_gate_requires_both_records_complete() -> None:
    assert _check(G.judge_gate(_record(), _record()), "complete")["ok"] is True
    gate = G.judge_gate(_record(), _record(complete=False))
    assert gate["pass"] is False
    assert _check(gate, "complete")["observed"] == {"run": False, "baseline": True}
    assert G.judge_gate(_record(complete=False), _record())["pass"] is False


@pytest.mark.parametrize("baseline_dropped,run_dropped", [(0, 5), (5, 0)])
def test_judge_gate_fails_when_either_run_dropped_events(baseline_dropped: int, run_dropped: int) -> None:
    gate = G.judge_gate(_record(dropped=baseline_dropped), _record(dropped=run_dropped))
    assert gate["pass"] is False and _check(gate, "no_dropped_events")["ok"] is False


# -------------------------------------------------------------------------------- report
def _report() -> Dict[str, Any]:
    runs = {"baseline": _record(), "pipeline": _record(swap_largest=2.0)}
    gate = G.judge_gate(runs["baseline"], runs["pipeline"])
    return G.build_report(runs, {"baseline": "demo/output/perf/a.jsonl"}, gate)


def test_build_report_carries_the_schema_and_a_timestamp() -> None:
    report = _report()
    assert report["schema"] == G.REPORT_SCHEMA_VERSION == 1
    assert len(report["created"]) >= 19 and "T" in report["created"]
    assert set(report["runs"]) == {"baseline", "pipeline"} and report["gate"]["pass"] is True
    assert report["jsonl"] == {"baseline": "demo/output/perf/a.jsonl"}
    assert G.validate_report(report) == []


def test_validate_report_lists_every_missing_piece() -> None:
    report = _report()
    report["schema"] = 2
    assert any("schema" in p for p in G.validate_report(report))
    report = _report()
    del report["runs"]["baseline"]
    assert any("baseline" in p for p in G.validate_report(report))
    report = _report()
    del report["runs"]["pipeline"]["swap"]["all"]["histogram"]
    assert any("histogram" in p for p in G.validate_report(report))
    report = _report()
    del report["runs"]["pipeline"]["tick"]["all"]["over"]["2.5"]
    assert any("2.5" in p for p in G.validate_report(report))
    report = _report()
    del report["runs"]["pipeline"]["pages"]["layer_ms"]["p90"]
    assert any("layer_ms" in p and "p90" in p for p in G.validate_report(report))
    report = _report()
    del report["runs"]["pipeline"]["pages"]
    assert any("pages" in p for p in G.validate_report(report))


@pytest.mark.parametrize("key", ["events", "dropped", "complete", "gc", "stalls"])
def test_validate_report_requires_the_run_totals(key: str) -> None:
    report = _report()
    del report["runs"]["pipeline"][key]
    assert any(key in problem for problem in G.validate_report(report))


def test_validate_report_requires_the_page_totals_and_the_meta_mode() -> None:
    report = _report()
    del report["runs"]["pipeline"]["pages"]["pages"]
    assert any("pages.pages" in p for p in G.validate_report(report))
    report = _report()
    del report["runs"]["pipeline"]["pages"]["paint_ms"]
    assert any("paint_ms" in p for p in G.validate_report(report))
    report = _report()
    del report["runs"]["pipeline"]["meta"]["mode"]
    assert any("meta.mode" in p for p in G.validate_report(report))


def test_validate_report_checks_the_gate_shape() -> None:
    report = _report()
    report["gate"] = None  # a run judged on its own carries no gate
    assert G.validate_report(report) == []
    report["gate"] = {"checks": []}
    assert any("gate" in problem for problem in G.validate_report(report))


def test_format_comparison_names_every_run_and_check() -> None:
    text = G.format_comparison(_report())
    assert "baseline" in text and "pipeline" in text
    for name in ("swap_largest", "swap_over_2", "measured_enough", "complete", "no_dropped_events"):
        assert name in text
    assert "PASS" in text.upper()


def test_format_comparison_shows_the_totals_and_completeness() -> None:
    header, _rule, first = G.format_comparison(_report()).splitlines()[:3]
    for column in ("events", "dropped", "complete"):
        assert column in header
    assert "42" in first and "yes" in first
    incomplete = _report()
    incomplete["runs"]["pipeline"]["complete"] = False
    assert "NO" in G.format_comparison(incomplete).splitlines()[3]


# -------------------------------------------------------------------------- load_events
def test_load_events_round_trips_a_real_dump(tmp_path: Path) -> None:
    prof = P.Profiler(maxlen=10)
    prof.mark("phase", name="drag", edge="start", rep=1)
    prof.mark("swap")
    prof.end("paint_ms", prof.begin())
    out = tmp_path / "run.jsonl"
    assert prof.dump(out) == 3
    events = G.load_events(out)
    assert [k for _t, k, _p in events] == ["phase", "swap", "paint_ms"]
    assert events[0][2] == {"name": "drag", "edge": "start", "rep": 1}
    assert events[1][2] == {} and events[2][2]["ms"] >= 0.0
    assert events[0][0] == prof.events()[0][0]


def test_append_events_adds_lines_load_events_reads_back(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    out.write_text('{"t": 1.0, "kind": "tick"}\n', encoding="utf-8")
    written = G.append_events(out, [(0.5, "gc", {"gen": 2, "ms": 40.0, "thread": "MainThread"})])
    assert written == 1
    assert G.load_events(out) == [(1.0, "tick", {}), (0.5, "gc", {"gen": 2, "ms": 40.0, "thread": "MainThread"})]
    assert G.append_events(out, []) == 0


def test_load_events_skips_blank_lines_and_reports_a_bad_one(tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text('{"t": 1.0, "kind": "swap"}\n\n  \n{"t": 2.0, "kind": "tick"}\n', encoding="utf-8")
    assert [k for _t, k, _p in G.load_events(good)] == ["swap", "tick"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"t": 1.0, "kind": "swap"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        G.load_events(bad)
    missing = tmp_path / "missing_kind.jsonl"
    missing.write_text('{"t": 1.0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        G.load_events(missing)


def test_load_events_reports_a_non_numeric_timestamp(tmp_path: Path) -> None:
    bad = tmp_path / "when.jsonl"
    bad.write_text('{"t": 1.0, "kind": "swap"}\n{"t": "soon", "kind": "tick"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        G.load_events(bad)
