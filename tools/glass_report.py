"""Judging layer of the glass profiling driver (dev only): events in, verdict out.

``tools/profile_glass.py`` opens windows and records events; this module never touches Qt,
never opens a window and has no import side effects, so a ``.jsonl`` written by any run - on
any machine, through any shell - can be re-judged later with ``--analyse``.

The event stream is the one :mod:`glasstranslate.ui.glass.profile` writes: one JSON object
per line, ``{"t": perf_counter, "kind": ..., **payload}``.  What this module reads from it:

* ``meta``   one per run: ``mode``, ``refresh_hz``, ``frame_ms``, ``steps``, ``step_px``,
             ``repeat``, ``heartbeat_ms``, ``backend``, ``ocr_engine``, ``capacity``,
             ``dropped``, ``overlay``.
* ``phase``  ``name`` (``drag`` / ``tabs`` / ``idle``), ``edge`` (``start`` / ``stop``),
             ``rep``: one measured window per completed pair.
* ``swap``   a presented frame; ``tick`` the driver's heartbeat; ``tab`` a page switch.
* ``overlay_apply_ms`` / ``overlay_typeset_ms`` / ``overlay_layer_ms`` / ``paint_ms``: the
  overlay's own cost, grouped into "pages" by the applies that carried fresh segments.
* ``gc``     a garbage collection (``gen``, ``ms``, ``thread``): it holds the interpreter lock, so
             it is one of the causes :func:`stall_causes` sorts the late heartbeats into.
* ``hard_stop``  the run overran its budget and was cut off; together with a phase that never
  stopped this is what makes a record ``complete: False``.

Every gap is measured **within** one window and expressed in frame intervals, so the numbers
compare across refresh rates: a delta of 1.0 frames is a frame presented on time, 2.0 frames
is one frame missed.  :func:`judge_gate` turns two records into the proposed lag-gate criteria;
it refuses to pass a run that recorded far less than the baseline or never finished its script,
because every gap check is vacuously fine when nothing was measured.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from glasstranslate.ui.glass.profile import percentiles

__all__ = [
    "GAP_FACTORS",
    "HISTOGRAM_EDGES",
    "META_KEYS",
    "REPORT_SCHEMA_VERSION",
    "Event",
    "analyse",
    "append_events",
    "build_report",
    "format_comparison",
    "gap_stats",
    "histogram_keys",
    "is_complete",
    "judge_gate",
    "load_events",
    "marks_per_page",
    "meta_payload",
    "phase_windows",
    "stall_causes",
    "tab_windows",
    "validate_report",
]

Event = Tuple[float, str, Dict[str, Any]]
Window = Tuple[float, float]
META_KEYS: Tuple[str, ...] = ("mode", "refresh_hz", "frame_ms", "steps", "step_px", "repeat",
                              "heartbeat_ms", "backend", "ocr_engine", "capacity", "dropped",
                              "overlay")

HISTOGRAM_EDGES: Tuple[float, ...] = (1.5, 2.0, 2.5, 4.0, 8.0)  # frame intervals
GAP_FACTORS: Tuple[float, ...] = (2.0, 2.5)
TAB_WINDOW_MS = 200.0
PHASE_NAMES: Tuple[str, ...] = ("drag", "tabs", "idle")
_PAGE_KEYS: Tuple[str, ...] = ("apply_ms", "typeset_ms", "layer_ms", "paint_ms", "blocks", "fresh")


# ------------------------------------------------------------------------------------ io
def load_events(path: Union[str, "os.PathLike[str]"]) -> List[Event]:
    """Read a profiler JSONL dump; raises ``ValueError`` naming the offending line."""
    target = Path(path)
    out: List[Event] = []
    with target.open("r", encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{target}: line {number} is not JSON: {exc}") from exc
            if not isinstance(record, dict) or "t" not in record or "kind" not in record:
                raise ValueError(f"{target}: line {number} has no 't' / 'kind' field")
            try:
                timestamp = float(record["t"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{target}: line {number} has a non-numeric 't': "
                                 f"{record['t']!r}") from exc
            payload = {k: v for k, v in record.items() if k not in ("t", "kind")}
            out.append((timestamp, str(record["kind"]), payload))
    return out


def append_events(path: Union[str, "os.PathLike[str]"], events: Sequence[Event]) -> int:
    """Append events recorded outside the profiler (the driver's ``gc`` list) to a dump.

    Same line format as ``Profiler.dump``; the order in the file does not matter, every reader
    here sorts by ``t``.  Returns the number of lines written.
    """
    if not events:
        return 0
    with Path(path).open("a", encoding="utf-8") as fh:
        for t, kind, payload in events:
            fh.write(json.dumps({"t": float(t), "kind": str(kind), **payload}, allow_nan=False) + "\n")
    return len(events)


# --------------------------------------------------------------------------- gap arithmetic
def histogram_keys(edges: Sequence[float] = HISTOGRAM_EDGES) -> Tuple[str, ...]:
    """Bucket labels in frame intervals, upper edge inclusive (``<=1.5`` .. ``>8.0``)."""
    keys = [f"<={edges[0]:.1f}"]
    keys += [f"{low:.1f}-{high:.1f}" for low, high in zip(edges, edges[1:])]
    keys.append(f">{edges[-1]:.1f}")
    return tuple(keys)


def _bucket(frames: float, keys: Sequence[str], edges: Sequence[float] = HISTOGRAM_EDGES) -> str:
    for index, edge in enumerate(edges):
        if frames <= edge:
            return keys[index]
    return keys[-1]


def _window_deltas_ms(windows: Iterable[Sequence[float]]) -> List[float]:
    """Consecutive deltas inside each window; never a delta across two windows."""
    out: List[float] = []
    for window in windows:
        times = list(window)
        out.extend((b - a) * 1000.0 for a, b in zip(times, times[1:]))
    return out


def gap_stats(windows: Sequence[Sequence[float]], frame_ms: float,
              factors: Sequence[float] = GAP_FACTORS) -> Dict[str, Any]:
    """Distribution of the gaps between consecutive timestamps, in ms and in frames."""
    keys = histogram_keys()
    over = {f"{factor:.1f}": 0 for factor in factors}
    histogram = {key: 0 for key in keys}
    deltas = _window_deltas_ms(windows) if frame_ms > 0 else []
    if not deltas:
        return {"n": 0, "p50_ms": 0.0, "p90_ms": 0.0, "largest_ms": 0.0, "largest_frames": 0.0,
                "over": over, "histogram": histogram}
    for delta in deltas:
        frames = delta / frame_ms
        histogram[_bucket(frames, keys)] += 1
        for factor in factors:
            if frames > factor:
                over[f"{factor:.1f}"] += 1
    pct = percentiles(deltas)
    largest = max(deltas)
    return {"n": len(deltas), "p50_ms": pct["p50"], "p90_ms": pct["p90"], "largest_ms": largest,
            "largest_frames": largest / frame_ms, "over": over, "histogram": histogram}


# --------------------------------------------------------------------------------- windows
def phase_windows(events: Sequence[Event], name: str) -> List[Tuple[float, float, int]]:
    """``(t0, t1, rep)`` for every completed ``phase`` start/stop pair called ``name``."""
    out: List[Tuple[float, float, int]] = []
    open_reps: Dict[int, float] = {}
    for t, kind, payload in sorted(events, key=lambda e: e[0]):
        if kind != "phase" or payload.get("name") != name:
            continue
        rep = int(payload.get("rep", 0))
        edge = payload.get("edge")
        if edge == "start":
            open_reps[rep] = t
        elif edge == "stop" and rep in open_reps:
            out.append((open_reps.pop(rep), t, rep))
    return out


def _spans(events: Sequence[Event], name: str) -> List[Window]:
    return [(t0, t1) for t0, t1, _rep in phase_windows(events, name)]


def _all_spans(events: Sequence[Event]) -> List[Window]:
    names = {p.get("name") for _t, k, p in events if k == "phase"}
    return [span for name in sorted(str(n) for n in names if n) for span in _spans(events, name)]


def tab_windows(events: Sequence[Event], window_ms: float = TAB_WINDOW_MS) -> List[Window]:
    """``window_ms`` after every ``tab`` switch that happened inside a ``tabs`` phase."""
    spans = _spans(events, "tabs")
    out: List[Window] = []
    for t, kind, _payload in sorted(events, key=lambda e: e[0]):
        if kind == "tab" and any(t0 <= t <= t1 for t0, t1 in spans):
            out.append((t, t + window_ms / 1000.0))
    return out


def _times_in(events: Sequence[Event], kind: str, spans: Sequence[Window]) -> List[List[float]]:
    """The ``kind`` timestamps of each span, one sorted list per span."""
    times = sorted(t for t, k, _p in events if k == kind)
    return [[t for t in times if t0 <= t <= t1] for t0, t1 in spans]


def _count_in(events: Sequence[Event], kind: str, spans: Sequence[Window]) -> int:
    return sum(1 for t, k, _p in events if k == kind and any(t0 <= t <= t1 for t0, t1 in spans))


# -------------------------------------------------------------------------------- per page
def _ms(payload: Mapping[str, Any]) -> float:
    try:
        return float(payload.get("ms", 0.0))
    except (TypeError, ValueError):
        return 0.0


def marks_per_page(events: Sequence[Event]) -> Dict[str, Any]:
    """Overlay cost per *new page*: an ``overlay_apply_ms`` whose ``fresh`` payload is > 0.

    A page's span reaches to the next *new page*.  A "nothing changed" apply (``fresh == 0``) neither
    opens nor closes one: it is often delivered in the same event-loop turn, before the page's paint.
    ``paint_ms`` is therefore the largest overlay paint before the next new page, which is the
    page's own first paint in practice (the later ones reuse its layouts and layers).
    """
    ordered = sorted(events, key=lambda e: e[0])
    applies = [i for i, (_t, kind, p) in enumerate(ordered)
               if kind == "overlay_apply_ms" and float(p.get("fresh", 0) or 0) > 0]
    values: Dict[str, List[float]] = {key: [] for key in _PAGE_KEYS}
    for number, index in enumerate(applies):
        payload = ordered[index][2]
        end = applies[number + 1] if number + 1 < len(applies) else len(ordered)
        span = ordered[index + 1:end]
        values["apply_ms"].append(_ms(payload))
        values["blocks"].append(float(payload.get("blocks", 0) or 0))
        values["fresh"].append(float(payload.get("fresh", 0) or 0))
        values["typeset_ms"].append(sum(_ms(p) for _t, k, p in span if k == "overlay_typeset_ms"))
        values["layer_ms"].append(sum(_ms(p) for _t, k, p in span if k == "overlay_layer_ms"))
        values["paint_ms"].append(max([_ms(p) for _t, k, p in span if k == "paint_ms"], default=0.0))
    out: Dict[str, Any] = {"pages": len(values["apply_ms"])}
    out.update({key: percentiles(values[key]) for key in _PAGE_KEYS})
    return out


# ---------------------------------------------------------------------------------- stalls
STALL_CAUSES: Tuple[str, ...] = ("paint", "gc", "delivery", "unattributed")
LATE_TICK_FRAMES = 2.0
_DELIVERY_SLACK_S = 0.015  # a new page is applied right after the turn that delivered it


def _overlap_ms(t: float, payload: Mapping[str, Any], start: float, end: float) -> float:
    """How much of a span mark (stamped at its END, ``ms`` long) lies inside ``start..end``."""
    return max(0.0, min(t, end) - max(t - _ms(payload) / 1000.0, start)) * 1000.0


def _cause_of(start: float, end: float, ordered: Sequence[Event], frame_ms: float) -> str:
    """Why the GUI thread missed its heartbeat between ``start`` and ``end`` (first match wins).

    A paint or a collection counts only when its own span covers at least one frame of the gap:
    every stall is followed by a repaint, and a mark that merely lands nearby explains nothing.
    """
    for cause, kind in (("paint", "paint_ms"), ("gc", "gc")):
        if any(k == kind and _overlap_ms(t, p, start, end) >= frame_ms for t, k, p in ordered):
            return cause
    if any(k == "overlay_apply_ms" and start < t <= end + _DELIVERY_SLACK_S
           and float(p.get("fresh", 0) or 0) > 0 for t, k, p in ordered):
        return "delivery"
    return "unattributed"


def stall_causes(events: Sequence[Event], frame_ms: float) -> Dict[str, Any]:
    """Every heartbeat gap above two frames over the whole run, grouped by what caused it.

    ``paint`` = an overlay ``paintEvent`` ran inside the gap; ``gc`` = a garbage collection did (it
    holds the interpreter lock on whichever thread started it); ``delivery`` = a new page was applied
    right after it (the result reaching the GUI thread); ``unattributed`` = none of our marks.
    """
    causes = {name: {"n": 0, "total_ms": 0.0, "largest_ms": 0.0} for name in STALL_CAUSES}
    if frame_ms <= 0:
        return {"late_ticks": 0, "causes": causes}
    ordered = sorted((e for e in events if e[1] in ("paint_ms", "gc", "overlay_apply_ms")), key=lambda e: e[0])
    ticks = sorted(t for t, kind, _p in events if kind == "tick")
    late = [(a, b) for a, b in zip(ticks, ticks[1:]) if (b - a) * 1000.0 > LATE_TICK_FRAMES * frame_ms]
    for start, end in late:
        gap_ms = (end - start) * 1000.0
        cause = causes[_cause_of(start, end, ordered, frame_ms)]
        cause["n"] += 1
        cause["total_ms"] += gap_ms
        cause["largest_ms"] = max(cause["largest_ms"], gap_ms)
    return {"late_ticks": len(late), "causes": causes}


# --------------------------------------------------------------------------------- analyse
def _meta(events: Sequence[Event]) -> Dict[str, Any]:
    for _t, kind, payload in events:
        if kind == "meta":
            return dict(payload)
    raise ValueError("the event stream carries no 'meta' event: it was not written by "
                     "tools/profile_glass.py, or the run died before its final mark")


def meta_payload(*, mode: str, refresh_hz: float, frame_ms: float, steps: int, step_px: int,
                 repeat: int, heartbeat_ms: int, backend: Optional[str], ocr_engine: Optional[str],
                 capacity: int, dropped: int, overlay: Optional[List[int]]) -> Dict[str, Any]:
    """The ``meta`` mark's payload: :data:`META_KEYS` exactly, and nothing else.

    It lives here because this module is what reads it back.  Nothing in it may identify the
    machine: the scripted parameters, the screen's timing and the engine names only - no path
    outside the repo, no URL, no key.
    """
    values = (mode, float(refresh_hz), float(frame_ms), int(steps), int(step_px), int(repeat),
              int(heartbeat_ms), backend, ocr_engine, int(capacity), int(dropped), overlay)
    return dict(zip(META_KEYS, values))


def is_complete(events: Sequence[Event]) -> bool:
    """False when the run was cut short: a ``hard_stop`` mark, or a phase that never stopped."""
    if any(kind == "hard_stop" for _t, kind, _p in events):
        return False
    open_phases = set()
    for _t, kind, payload in sorted(events, key=lambda e: e[0]):
        if kind != "phase":
            continue
        key = (str(payload.get("name")), int(payload.get("rep", 0)))
        if payload.get("edge") == "start":
            open_phases.add(key)
        elif payload.get("edge") == "stop":
            open_phases.discard(key)
    return not open_phases


def analyse(events: Sequence[Event], gap_factor: float = 2.5) -> Dict[str, Any]:
    """One run's record: the meta line, the swap / tick gaps per phase and the page costs."""
    meta = _meta(events)
    try:
        frame_ms = float(meta.get("frame_ms") or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"the meta event carries an unusable frame_ms: {meta.get('frame_ms')!r}") from exc
    if not math.isfinite(frame_ms) or frame_ms <= 0:
        raise ValueError(f"the meta event carries no usable frame_ms ({frame_ms!r}): "
                         "the run could not read its screen's refresh rate")
    # Rounded first: 2.04 and 2.0 would format to the same ``over`` key and be counted twice.
    factor = round(float(gap_factor), 1)
    factors = tuple(sorted({*GAP_FACTORS, factor}))
    drag, tabs, idle = (_spans(events, name) for name in PHASE_NAMES)
    tabbed = tab_windows(events)

    swap_drag, swap_tabs = _times_in(events, "swap", drag), _times_in(events, "swap", tabbed)
    swap = {"drag": gap_stats(swap_drag, frame_ms, factors),
            "tabs": gap_stats(swap_tabs, frame_ms, factors),
            "all": gap_stats(swap_drag + swap_tabs, frame_ms, factors)}
    ticks = {name: _times_in(events, "tick", spans)
             for name, spans in zip(PHASE_NAMES, (drag, tabs, idle))}
    tick = {name: gap_stats(window, frame_ms, factors) for name, window in ticks.items()}
    tick["all"] = gap_stats([w for name in PHASE_NAMES for w in ticks[name]], frame_ms, factors)

    key = f"{factor:.1f}"
    return {
        "meta": meta,
        "events": len(events),
        "dropped": int(meta.get("dropped", 0) or 0),
        "complete": is_complete(events),
        "swap": swap,
        "tick": tick,
        "swaps_per_tab_window": percentiles([len(w) for w in swap_tabs]),
        "idle_swaps": _count_in(events, "swap", idle),
        "pages": marks_per_page(events),
        "stalls": stall_causes(events, frame_ms),
        "gc": percentiles([_ms(p) for _t, kind, p in events if kind == "gc"]),
        "page_swaps": _count_in(events, "page", _all_spans(events)),
        "gap_factor": factor,
        "over_gap_factor": {"swap": swap["all"]["over"][key], "tick": tick["all"]["over"][key]},
    }


# ------------------------------------------------------------------------------------ gate
def _check(name: str, ok: bool, observed: Any, limit: Any, skipped: bool = False) -> Dict[str, Any]:
    check: Dict[str, Any] = {"name": name, "ok": bool(ok), "observed": observed, "limit": limit}
    if skipped:
        check["skipped"] = True
    return check


def _measured_enough(baseline: Mapping[str, Any], run: Mapping[str, Any]) -> Dict[str, Any]:
    """A run that recorded far fewer marks than the baseline cannot be compared to it.

    Both runs play the same script for the same scripted duration, so half the baseline's marks
    is a generous floor.  Without it a frozen run - nothing presented, nothing ticked - would
    pass every gap check vacuously.
    """
    counts = {group: int(run[group]["all"]["n"]) for group in ("swap", "tick")}
    base = {group: int(baseline[group]["all"]["n"]) for group in ("swap", "tick")}
    limit: Dict[str, Optional[float]] = {"swap": 0.5 * base["swap"],
                                         "tick": 0.5 * base["tick"] if base["tick"] > 0 else None}
    ok = counts["swap"] >= float(limit["swap"] or 0.0)
    if limit["tick"] is not None:
        ok = ok and counts["tick"] >= limit["tick"]
    return _check("measured_enough", ok, counts, limit)


def judge_gate(baseline: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    """The proposed lag gate: the pipeline may cost one frame, never a second one."""
    checks: List[Dict[str, Any]] = []
    for group in ("swap", "tick"):
        base, measured = baseline[group]["all"], run[group]["all"]
        # Only a baseline without a heartbeat leaves nothing to compare against.  A run that
        # lost its own ticks while the baseline had them is a failure, not a skip.
        if group == "tick" and base["n"] == 0:
            checks.append(_check("tick_largest", True, measured["largest_frames"], None, skipped=True))
            checks.append(_check("tick_over_2", True, measured["over"]["2.0"], None, skipped=True))
            continue
        limit = base["largest_frames"] + 1.0
        checks.append(_check(f"{group}_largest", measured["n"] > 0 and measured["largest_frames"] <= limit,
                             measured["largest_frames"], limit))
        checks.append(_check(f"{group}_over_2", measured["over"]["2.0"] <= base["over"]["2.0"],
                             measured["over"]["2.0"], base["over"]["2.0"]))
    checks.append(_measured_enough(baseline, run))
    complete = bool(run.get("complete", False)), bool(baseline.get("complete", False))
    checks.append(_check("complete", all(complete), {"run": complete[0], "baseline": complete[1]}, True))
    dropped = int(run.get("dropped", 0) or 0), int(baseline.get("dropped", 0) or 0)
    checks.append(_check("no_dropped_events", dropped == (0, 0), {"run": dropped[0], "baseline": dropped[1]}, 0))
    return {"pass": all(check["ok"] for check in checks), "checks": checks}


# ---------------------------------------------------------------------------------- report
REPORT_SCHEMA_VERSION = 1


def build_report(runs: Mapping[str, Dict[str, Any]], jsonl: Mapping[str, str],
                 gate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The persisted comparison: every run's record, where its events came from, the gate."""
    return {
        "schema": REPORT_SCHEMA_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "runs": dict(runs),
        "jsonl": dict(jsonl),
        "gate": gate,
    }


def _validate_gaps(name: str, run: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    for group in ("swap", "tick"):
        pooled = (run.get(group) or {}).get("all")
        if not isinstance(pooled, dict):
            problems.append(f"runs.{name}.{group}.all is missing")
            continue
        problems += [f"runs.{name}.{group}.all.{key} is missing"
                     for key in ("histogram", "largest_ms", "largest_frames") if key not in pooled]
        over = pooled.get("over") or {}
        problems += [f"runs.{name}.{group}.all.over['{factor}'] is missing"
                     for factor in ("2.0", "2.5") if factor not in over]
    return problems


def _validate_pages(name: str, run: Mapping[str, Any]) -> List[str]:
    pages = run.get("pages")
    if not isinstance(pages, dict):
        return [f"runs.{name}.pages is missing"]
    problems: List[str] = [] if "pages" in pages else [f"runs.{name}.pages.pages is missing"]
    for key in ("apply_ms", "typeset_ms", "layer_ms", "paint_ms"):
        block = pages.get(key)
        if not isinstance(block, dict):
            problems.append(f"runs.{name}.pages.{key} is missing")
            continue
        problems += [f"runs.{name}.pages.{key}.{p} is missing" for p in ("p50", "p90", "max") if p not in block]
    return problems


def _validate_totals(name: str, run: Mapping[str, Any]) -> List[str]:
    problems = [f"runs.{name}.{key} is missing" for key in ("events", "dropped", "complete", "gc")
                if key not in run]
    stalls = run.get("stalls")
    if not isinstance(stalls, dict) or "late_ticks" not in stalls or set(stalls.get("causes") or {}) != set(STALL_CAUSES):
        problems.append(f"runs.{name}.stalls must carry late_ticks and one entry per cause")
    meta = run.get("meta")
    if not isinstance(meta, dict) or "mode" not in meta:
        problems.append(f"runs.{name}.meta.mode is missing")
    return problems


def validate_report(report: Mapping[str, Any]) -> List[str]:
    """Everything wrong with ``report`` as a schema-1 document (empty list = valid)."""
    problems: List[str] = []
    if report.get("schema") != REPORT_SCHEMA_VERSION:
        problems.append(f"schema is {report.get('schema')!r}, expected {REPORT_SCHEMA_VERSION}")
    gate = report.get("gate")
    if gate is not None and not (isinstance(gate, dict) and "pass" in gate and "checks" in gate):
        problems.append("gate must be null or carry 'pass' and 'checks'")
    runs = report.get("runs")
    if not isinstance(runs, dict):
        return problems + ["runs is missing"]
    if "baseline" not in runs:
        problems.append("runs has no 'baseline' entry")
    for name, run in runs.items():
        if not isinstance(run, dict):
            problems.append(f"runs.{name} is not a record")
            continue
        problems += (_validate_gaps(str(name), run) + _validate_pages(str(name), run)
                     + _validate_totals(str(name), run))
    return problems


# ----------------------------------------------------------------------------- comparison
_COLUMNS: Tuple[Tuple[str, int], ...] = (
    ("run", 14), ("swap max", 9), ("swap>2.0", 9), ("swap>2.5", 9), ("tick max", 9),
    ("tick>2.0", 9), ("tick>2.5", 9), ("pages", 6), ("typeset", 16), ("layer", 16),
    ("apply", 16), ("paint max", 10), ("events", 9), ("dropped", 9), ("complete", 9),
)


def _pair(block: Mapping[str, Any]) -> str:
    return f"{float(block.get('p50', 0.0)):.1f}/{float(block.get('max', 0.0)):.1f}"


def _row(name: str, run: Mapping[str, Any]) -> List[str]:
    swap, tick = run.get("swap", {}).get("all", {}), run.get("tick", {}).get("all", {})
    pages = run.get("pages", {})
    return [
        name,
        f"{float(swap.get('largest_frames', 0.0)):.2f}f",
        str(swap.get("over", {}).get("2.0", 0)), str(swap.get("over", {}).get("2.5", 0)),
        f"{float(tick.get('largest_frames', 0.0)):.2f}f",
        str(tick.get("over", {}).get("2.0", 0)), str(tick.get("over", {}).get("2.5", 0)),
        str(pages.get("pages", 0)),
        _pair(pages.get("typeset_ms", {})), _pair(pages.get("layer_ms", {})),
        _pair(pages.get("apply_ms", {})),
        f"{float(pages.get('paint_ms', {}).get('max', 0.0)):.1f}",
        str(run.get("events", 0)), str(run.get("dropped", 0)),
        "yes" if run.get("complete") else "NO",
    ]


def _stall_lines(runs: Mapping[str, Any]) -> List[str]:
    """Late heartbeats of the whole run by cause: ``n x largest ms`` per cause."""
    lines = ["late heartbeats (> 2.0 frames, whole run) by cause:"]
    for name, run in runs.items():
        stalls = run.get("stalls") or {}
        cells = [f"{cause} {int(c.get('n', 0))} x <= {float(c.get('largest_ms', 0.0)):.1f} ms"
                 for cause, c in (stalls.get("causes") or {}).items()]
        lines.append(f"  {str(name):<13} {int(stalls.get('late_ticks', 0)):>3}: " + ", ".join(cells))
    return lines


def format_comparison(report: Mapping[str, Any]) -> str:
    """One plain-text row per run (gaps in frame intervals, page costs in ms) plus the gate."""
    widths = [width for _label, width in _COLUMNS]
    lines = ["".join(label.ljust(width) for label, width in _COLUMNS),
             "-" * sum(widths)]
    for name, run in (report.get("runs") or {}).items():
        lines.append("".join(cell.ljust(width) for cell, width in zip(_row(str(name), run), widths)))
    lines.append("")
    lines.append("p50/max in ms; 'swap max' / 'tick max' are the largest gap in frame intervals")
    lines.extend(_stall_lines(report.get("runs") or {}))
    gate = report.get("gate")
    if not gate:
        lines.append("gate: not evaluated (no baseline to compare against)")
        return "\n".join(lines)
    lines.append(f"gate: {'PASS' if gate.get('pass') else 'FAIL'}")
    for check in gate.get("checks", []):
        state = "skip" if check.get("skipped") else ("ok  " if check.get("ok") else "FAIL")
        lines.append(f"  [{state}] {str(check.get('name')):<20} observed {check.get('observed')!r}"
                     f"  limit {check.get('limit')!r}")
    return "\n".join(lines)
