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
