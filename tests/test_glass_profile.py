"""Opt-in glass profiler (feature batch F2-0, ``GLASSTRANSLATE_PROFILE``).

Off by default: the module singleton is a no-op object and no window attribute or
connection exists.  On: events are recorded, dumped as JSONL and summarised; the
``frame_gui`` hook carries the stale-geometry offset (hypothesis a).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage, QMoveEvent

from glasstranslate.core.types import Rect
from glasstranslate.ui.glass import profile as P
from glasstranslate.ui.glass.backdrop import BackdropFrame

pytestmark = pytest.mark.usefixtures("qapp")


@pytest.fixture()
def profiling_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(P.PROFILE_ENV, raising=False)
    P.install()
    yield P.profiler
    P.install()


@pytest.fixture()
def profiling_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(P.PROFILE_ENV, "1")
    prof = P.install()
    yield prof
    monkeypatch.delenv(P.PROFILE_ENV, raising=False)
    P.install()


def _frame(x: int, y: int, ts: float = 1.0) -> BackdropFrame:
    return BackdropFrame(QImage(4, 4, QImage.Format.Format_RGB32), (x - 32, y - 32), Rect(x, y, 832, 640), 1.0, 0.5, ts)


# ------------------------------------------------------------------------------ disabled
@pytest.mark.parametrize("value", [None, "", "0", " 0 "])
def test_enabled_reads_env(monkeypatch: pytest.MonkeyPatch, value) -> None:
    if value is None:
        monkeypatch.delenv(P.PROFILE_ENV, raising=False)
    else:
        monkeypatch.setenv(P.PROFILE_ENV, value)
    assert P.enabled() is False
    monkeypatch.setenv(P.PROFILE_ENV, "1")
    assert P.enabled() is True


def test_profile_disabled_is_noop_and_records_nothing(profiling_off, tmp_path: Path) -> None:
    prof = profiling_off
    assert isinstance(prof, P.NoopProfiler) and not isinstance(prof, P.Profiler)
    assert prof.enabled is False
    for _ in range(1000):
        P.mark("move", x=1, y=2)
        prof.geometry("poke", Rect(0, 0, 1, 1))
        prof.frame_gui(_frame(0, 0), lambda: Rect(5, 5, 1, 1))
    assert prof.timed_call("push_backdrop_ms", lambda a, b: a + b, 2, 3) == 5
    assert prof.events() == [] and prof.summary() == {}
    assert prof.dump(tmp_path / "never.jsonl") == 0 and not (tmp_path / "never.jsonl").exists()
    # the window has no profiling attribute / frameSwapped connection when profiling is off
    from glasstranslate.config.settings import AppConfig
    from glasstranslate.ui import control as C

    w = C.ControlWindow(AppConfig(), tmp_path / "c.json")
    try:
        assert not hasattr(w, "_profile_hooks")
    finally:
        w.shutdown()
        w.deleteLater()


# ------------------------------------------------------------------------------- enabled
def test_profile_records_events_and_dumps_jsonl(profiling_on, tmp_path: Path) -> None:
    prof = profiling_on
    assert isinstance(prof, P.Profiler) and prof.enabled is True and P.mark == prof.mark
    P.mark("move", x=10, y=20)
    prof.geometry("poke", Rect(1, 2, 3, 4))
    prof.geometry("grab", None)
    assert prof.timed_call("push_backdrop_ms", lambda: "ok") == "ok"
    events = prof.events()
    assert [e[1] for e in events] == ["move", "poke", "grab", "push_backdrop_ms"]
    assert events[0][2] == {"x": 10, "y": 20} and events[1][2] == {"x": 1, "y": 2, "w": 3, "h": 4}
    assert events[2][2] == {} and events[3][2]["ms"] >= 0.0
    assert all(events[i][0] <= events[i + 1][0] for i in range(len(events) - 1))
    out = tmp_path / "perf" / "p.jsonl"
    assert prof.dump(out) == 4
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [line["kind"] for line in lines] == ["move", "poke", "grab", "push_backdrop_ms"]
    assert lines[0] == {"t": events[0][0], "kind": "move", "x": 10, "y": 20}
    prof.clear()
    assert prof.events() == []


def test_profiler_is_bounded_and_thread_safe(profiling_on) -> None:
    import threading

    prof = P.Profiler(maxlen=50)

    def worker() -> None:
        for i in range(100):
            prof.mark("swap", i=i)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(prof.events()) == 50 and prof.summary()["counts"] == {"swap": 50}


def test_frame_gui_event_carries_stale_offset(profiling_on) -> None:
    prof = profiling_on
    assert P.stale_offset(Rect(100, 100, 8, 6), Rect(140, 100, 8, 6)) == (40, 0)
    prof.frame_gui(_frame(100, 100, ts=7.5), lambda: Rect(140, 100, 832, 640))
    (t, kind, payload), = prof.events()
    assert kind == "frame_gui"
    assert payload == {"ts": 7.5, "fx": 100, "fy": 100, "cx": 140, "cy": 100, "dx": 40, "dy": 0}


def test_control_window_frame_hook_records_offset(profiling_on, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real hook: ``_on_backdrop_frame`` records the frame vs the window's current rect."""
    from glasstranslate.config.settings import AppConfig
    from glasstranslate.ui import control as C
    from glasstranslate.ui.glass import win32

    prof = profiling_on
    monkeypatch.setattr(win32, "physical_rect", lambda _w: Rect(140, 100, 832, 640))
    w = C.ControlWindow(AppConfig(), tmp_path / "c.json")
    try:
        assert hasattr(w, "_profile_hooks")
        w._on_backdrop_frame(_frame(100, 100, ts=3.0))
        kinds = [e[1] for e in prof.events()]
        assert "frame_gui" in kinds and "push_backdrop_ms" in kinds
        frame_ev = next(p for _t, k, p in prof.events() if k == "frame_gui")
        assert (frame_ev["dx"], frame_ev["dy"], frame_ev["ts"]) == (40, 0, 3.0)
        assert w.bridge.backdropSerial == 1  # the frame still reached the bridge
        prof.clear()
        w.moveEvent(QMoveEvent(QPoint(1, 1), QPoint(0, 0)))
        assert [e[1] for e in prof.events()][:1] == ["move"]
    finally:
        w.shutdown()
        w.deleteLater()


def test_grabber_records_grab_events(profiling_on, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    import numpy as np

    from glasstranslate.ui.glass import backdrop as BD

    class Shot:
        def __init__(self, w: int, h: int) -> None:
            self.width, self.height = w, h
            self.bgra = np.zeros((h, w, 4), np.uint8).tobytes()

    class FakeMSS:
        monitors = [{"left": 0, "top": 0, "width": 400, "height": 300}]

        def grab(self, r: Dict[str, int]) -> Shot:
            return Shot(r["width"], r["height"])

    prof = profiling_on
    geometry = BD.WindowGeometry()
    geometry.update(Rect(50, 60, 100, 50), 1.0)
    frames: List[BackdropFrame] = []
    grabber = BD.BackdropGrabber(geometry, frames.append)
    grabber._grab_once(FakeMSS(), time.perf_counter())  # forced first grab -> emitted
    grabber._grab_once(FakeMSS(), time.perf_counter())  # unchanged -> not emitted
    grabs = [p for _t, k, p in prof.events() if k == "grab"]
    assert len(grabs) == 2 and len(frames) == 1
    assert grabs[0]["emitted"] is True and grabs[1]["emitted"] is False
    assert (grabs[0]["x"], grabs[0]["y"]) == (50, 60) and grabs[0]["ms"] >= 0.0


# ------------------------------------------------------------------------- cap and spans
def test_eviction_is_counted_and_reset_by_clear(profiling_on) -> None:
    """A full deque silently dropped the oldest events; ``dropped`` makes that visible."""
    prof = P.Profiler(maxlen=4)
    assert prof.capacity == 4 and prof.dropped == 0
    for i in range(4):
        prof.mark("swap", i=i)
    assert prof.dropped == 0 and len(prof.events()) == 4
    for i in range(3):
        prof.mark("swap", i=100 + i)
    assert prof.dropped == 3 and len(prof.events()) == 4
    summary = prof.summary()
    assert summary["dropped"] == 3 and summary["capacity"] == 4
    prof.clear()
    assert prof.dropped == 0 and prof.events() == []
    assert prof.summary()["dropped"] == 0


def test_install_passes_the_capacity_through(profiling_on) -> None:
    try:
        prof = P.install(force=True, maxlen=7)
        assert isinstance(prof, P.Profiler) and prof.capacity == 7
        assert P.install(force=True).capacity == P.MAX_EVENTS  # the in-app default is unchanged
        off = P.install(force=False, maxlen=7)
        assert isinstance(off, P.NoopProfiler) and not isinstance(off, P.Profiler)
        assert off.capacity == 0 and off.dropped == 0
    finally:
        P.install(force=True)


@pytest.mark.parametrize("maxlen", [0, -5])
def test_a_capacity_below_one_is_clamped(profiling_on, maxlen: int) -> None:
    """A deque with maxlen 0 records nothing at all; one event is the floor."""
    assert P.Profiler(maxlen=maxlen).capacity == 1
    try:
        prof = P.install(force=True, maxlen=maxlen)
        assert prof.capacity == 1
        prof.mark("swap")
        assert len(prof.events()) == 1
    finally:
        P.install(force=True)


def test_begin_end_records_a_span_with_its_payload(profiling_on) -> None:
    prof = profiling_on
    t0 = prof.begin()
    assert t0 > 0.0
    prof.end("overlay_apply_ms", t0, blocks=3, fresh=2)
    (_t, kind, payload), = prof.events()
    assert kind == "overlay_apply_ms" and payload["blocks"] == 3 and payload["fresh"] == 2
    assert payload["ms"] >= 0.0
    assert prof.summary()["ms"]["overlay_apply_ms"]["n"] == 1


def test_begin_end_is_a_noop_when_profiling_is_off(profiling_off) -> None:
    prof = profiling_off
    assert prof.begin() == 0.0
    prof.end("overlay_apply_ms", 0.0, blocks=1, fresh=1)
    prof.end("paint_ms", prof.begin())
    assert prof.events() == []


# ------------------------------------------------------------------------------- overlay
def _overlay_with_block(text: str = "HELLO THERE FRIEND"):
    """A 400x300 glass overlay plus one block segment (the helper of test_overlay_cache)."""
    from glasstranslate.ui import overlay as O

    from tests.test_overlay_cache import _block_segment

    glass = O.GlassOverlay()
    glass.setGeometry(0, 0, 400, 300)
    return glass, _block_segment(text, 60)


def _rendered_bytes(glass, seg) -> bytes:
    glass._apply_segments([seg])
    image = glass.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    return bytes(image.constBits())


def test_overlay_marks_nothing_when_profiling_is_off(profiling_off) -> None:
    glass, seg = _overlay_with_block()
    try:
        assert len(_rendered_bytes(glass, seg)) > 0
    finally:
        glass.deleteLater()
    assert profiling_off.events() == []


def test_overlay_marks_apply_typeset_layer_and_paint(profiling_on) -> None:
    prof = profiling_on
    glass, seg = _overlay_with_block()
    try:
        glass._apply_segments([seg])
        applies = [p for _t, k, p in prof.events() if k == "overlay_apply_ms"]
        assert len(applies) == 1 and applies[0]["blocks"] == 1 and applies[0]["fresh"] == 1
        assert applies[0]["ms"] >= 0.0
        glass.grab()
        kinds = [k for _t, k, _p in prof.events()]
        assert kinds.count("overlay_apply_ms") == 1
        assert kinds.count("overlay_typeset_ms") >= 1 and kinds.count("overlay_layer_ms") >= 1
        assert kinds.count("paint_ms") >= 1
        counted = (kinds.count("overlay_typeset_ms"), kinds.count("overlay_layer_ms"), kinds.count("paint_ms"))
        glass.grab()  # the layout and the lettering layer are cached: only the paint repeats
        kinds = [k for _t, k, _p in prof.events()]
        assert (kinds.count("overlay_typeset_ms"), kinds.count("overlay_layer_ms")) == counted[:2]
        assert kinds.count("paint_ms") > counted[2]
        prof.clear()
        glass._apply_segments([seg])  # the very same object: nothing is fresh
        applies = [p for _t, k, p in prof.events() if k == "overlay_apply_ms"]
        assert len(applies) == 1 and (applies[0]["blocks"], applies[0]["fresh"]) == (1, 0)
    finally:
        glass.deleteLater()


def test_overlay_paints_the_same_pixels_with_and_without_profiling(monkeypatch: pytest.MonkeyPatch) -> None:
    def rendered() -> bytes:
        glass, seg = _overlay_with_block()
        try:
            return _rendered_bytes(glass, seg)
        finally:
            glass.deleteLater()

    try:
        monkeypatch.delenv(P.PROFILE_ENV, raising=False)
        P.install()
        off = rendered()
        monkeypatch.setenv(P.PROFILE_ENV, "1")
        P.install()
        on = rendered()
    finally:  # a failed assertion must not leak a live recorder into the next test
        monkeypatch.delenv(P.PROFILE_ENV, raising=False)
        P.install()
    assert len(off) > 0 and on == off


def test_a_failing_typeset_still_paints_and_only_loses_its_mark(
    profiling_on, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_paint_block``'s per-block ``except`` keeps the glass alive; the open span is dropped."""
    from glasstranslate.ui import overlay as O

    def boom(*_args, **_kwargs):
        raise RuntimeError("typeset exploded")

    glass, seg = _overlay_with_block()
    try:
        glass._apply_segments([seg])
        monkeypatch.setattr(O, "typeset_block", boom)
        profiling_on.clear()
        assert len(glass.grab().toImage().constBits()) > 0  # the window still produced pixels
        kinds = [k for _t, k, _p in profiling_on.events()]
        assert kinds.count("paint_ms") == 1  # the paint finished and was measured
        assert "overlay_typeset_ms" not in kinds and "overlay_layer_ms" not in kinds
    finally:
        glass.deleteLater()


# ------------------------------------------------------------------------------- summary
def test_summary_percentiles(profiling_on) -> None:
    prof = profiling_on
    assert P.percentiles([]) == {"n": 0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    pct = P.percentiles(list(range(1, 11)))  # nearest-rank
    assert pct["n"] == 10 and pct["p50"] == 5 and pct["p90"] == 9 and pct["max"] == 10 and pct["mean"] == 5.5
    assert P.percentiles([7.0]) == {"n": 1, "p50": 7.0, "p90": 7.0, "max": 7.0, "mean": 7.0}
    # synthetic timeline: move at t, poke +1 ms, grab +5 ms, frame_gui +9 ms; two swaps
    base = 100.0
    timeline = [(base, "move", {}), (base + 0.001, "poke", {}), (base + 0.005, "grab", {"ms": 4.0}),
                (base + 0.009, "frame_gui", {"dx": 3}), (base + 0.010, "swap", {}), (base + 0.014, "swap", {}),
                (base + 0.020, "move", {}), (base + 0.022, "poke", {}), (base + 0.030, "frame_gui", {"dx": 0})]
    with prof._lock:
        prof._events.extend(timeline)
    assert P.pair_latencies_ms(timeline, "move", "poke") == pytest.approx([1.0, 2.0])
    assert P.pair_latencies_ms(timeline, "move", "frame_gui") == pytest.approx([9.0, 10.0])
    s = prof.summary()
    assert s["counts"] == {"move": 2, "poke": 2, "grab": 1, "frame_gui": 2, "swap": 2}
    assert s["ms"]["grab"]["n"] == 1 and s["ms"]["grab"]["max"] == 4.0
    assert s["latency_ms"]["move->poke"]["n"] == 2 and s["latency_ms"]["move->poke"]["max"] == pytest.approx(2.0)
    assert s["latency_ms"]["move->frame_gui"]["p50"] == pytest.approx(9.0)  # nearest-rank of [9, 10]
    assert s["latency_ms"]["move->frame_gui"]["max"] == pytest.approx(10.0)
    assert "poke->grab" in s["latency_ms"] and "grab->frame_gui" in s["latency_ms"]
