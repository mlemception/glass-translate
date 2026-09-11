"""Capture mode (feature batch 2026-09-09, F1; extended to the control window 2026-09-10).

A runtime-only toggle that makes the overlay *and* the control window visible to screenshot /
screen-capture tools (clears ``WDA_EXCLUDEFROMCAPTURE`` on both), freezes the pipeline so the
glass cannot re-capture its own lettering, and freezes the control window's backdrop grabber so
the panel cannot sample itself.  Never persisted; always starts off.  Headless: no real Win32
calls, every ``SetWindowDisplayAffinity`` is recorded by a fake ``user32``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
from PySide6.QtGui import QImage, QWindow

from glasstranslate.config.settings import AppConfig
from glasstranslate.core.types import Rect
from glasstranslate.ui import control as C
from glasstranslate.ui import control_window as CW
from glasstranslate.ui import overlay as OV
from glasstranslate.ui.capture_mode import CaptureMode
from glasstranslate.ui.glass import win32 as W
from glasstranslate.ui.glass.appearance import Appearance, AppearanceSignals
from glasstranslate.ui.glass.backdrop import BackdropFrame

pytestmark = pytest.mark.usefixtures("qapp")

WIN = sys.platform == "win32"


class _FakeUser32:
    """Records ``SetWindowDisplayAffinity`` calls; every other call succeeds.

    Functions are plain attributes (not methods) because the overlay assigns ``.restype`` to
    ``GetWindowLongW`` / ``SetWindowLongW`` the way it does on the real ``ctypes`` handles.
    """

    def __init__(self, ok: bool = True) -> None:
        self.calls: List[Tuple[int, int]] = []
        self.ok = ok

        def set_affinity(hwnd, flag) -> int:
            self.calls.append((int(getattr(hwnd, "value", hwnd) or 0), int(flag)))
            return 1 if self.ok else 0

        self.SetWindowDisplayAffinity = set_affinity
        self.GetWindowLongW = lambda hwnd, index: 0
        self.SetWindowLongW = lambda hwnd, index, value: 0
        self.GetWindowRect = lambda hwnd, rect: 0


# --------------------------------------------------------------------------- win32 primitive
@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_set_capture_excluded_calls_affinity_with_wda_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeUser32()
    monkeypatch.setattr(W, "_user32", fake)
    win = QWindow()
    try:
        assert W.set_capture_excluded(win, False) is True
    finally:
        win.destroy()
    assert [flag for _, flag in fake.calls] == [W.WDA_NONE]
    assert W.WDA_NONE == 0


@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_set_capture_excluded_calls_affinity_with_exclude_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeUser32()
    monkeypatch.setattr(W, "_user32", fake)
    win = QWindow()
    try:
        assert W.set_capture_excluded(win, True) is True
    finally:
        win.destroy()
    assert [flag for _, flag in fake.calls] == [W.WDA_EXCLUDEFROMCAPTURE]


@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_exclude_from_capture_delegates_to_set_capture_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeUser32()
    monkeypatch.setattr(W, "_user32", fake)
    win = QWindow()
    try:
        assert W.exclude_from_capture(win) is True
    finally:
        win.destroy()
    assert [flag for _, flag in fake.calls] == [W.WDA_EXCLUDEFROMCAPTURE]


@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_set_capture_excluded_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(W, "_user32", _FakeUser32(ok=False))
    win = QWindow()
    try:
        assert W.set_capture_excluded(win, False) is False
    finally:
        win.destroy()


def test_set_capture_excluded_is_a_noop_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(W, "IS_WINDOWS", False)
    win = QWindow()
    try:
        assert W.set_capture_excluded(win, False) is False
        assert W.set_capture_excluded(win, True) is False
    finally:
        win.destroy()


# ----------------------------------------------------------------------------------- overlay
@pytest.fixture()
def overlay(qapp):
    ov = OV.GlassOverlay()
    yield ov
    ov.hide()
    ov.deleteLater()
    qapp.processEvents()


def test_overlay_defaults_to_excluded(overlay: OV.GlassOverlay) -> None:
    assert overlay.capture_excluded is True


@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_overlay_set_capture_excluded_applies_immediately_when_shown(
    overlay: OV.GlassOverlay, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeUser32()
    monkeypatch.setattr(OV, "_user32", lambda: fake)
    overlay.show()
    assert [flag for _, flag in fake.calls] == [OV._WDA_EXCLUDEFROMCAPTURE]
    overlay.set_capture_excluded(False)
    assert overlay.capture_excluded is False
    assert [flag for _, flag in fake.calls][-1] == OV._WDA_NONE
    overlay.set_capture_excluded(True)
    assert [flag for _, flag in fake.calls][-1] == OV._WDA_EXCLUDEFROMCAPTURE


@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_overlay_show_reapplies_stored_affinity(overlay: OV.GlassOverlay, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeUser32()
    monkeypatch.setattr(OV, "_user32", lambda: fake)
    overlay.set_capture_excluded(False)  # before the native window exists: stored only
    overlay.show()
    assert [flag for _, flag in fake.calls] == [OV._WDA_NONE]
    overlay.hide()
    overlay.show()
    assert [flag for _, flag in fake.calls] == [OV._WDA_NONE, OV._WDA_NONE]


# ------------------------------------------------------------------------------------ bridge
def _bridge(tmp_path: Path):
    appearance = Appearance(watch=False, overrides="")
    appearance.apply(AppearanceSignals(dark_mode=True))
    cfg = AppConfig()
    bridge = C.ControlBridge(cfg, tmp_path / "config.json", appearance=appearance)
    fired: List[AppConfig] = []
    bridge.config_changed.connect(fired.append)
    return bridge, fired


def test_capture_mode_defaults_off(tmp_path: Path) -> None:
    bridge, _ = _bridge(tmp_path)
    assert bridge.overlayCaptureMode is False


def test_capture_mode_setter_emits_request_once_and_notifies(tmp_path: Path) -> None:
    bridge, _ = _bridge(tmp_path)
    requests: List[bool] = []
    notified: List[int] = []
    bridge.capture_mode_requested.connect(requests.append)
    bridge.overlayCaptureModeChanged.connect(lambda: notified.append(1))
    bridge.overlayCaptureMode = True
    bridge.overlayCaptureMode = True  # idempotent
    bridge.overlayCaptureMode = False
    assert requests == [True, False]
    assert len(notified) == 2


def test_capture_mode_is_not_written_to_config_json(tmp_path: Path) -> None:
    bridge, fired = _bridge(tmp_path)
    bridge.overlayCaptureMode = True
    assert fired == []  # not a config change
    bridge.save_now()
    data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert not any("capture" in key.lower() for key in data)
    assert "overlayCaptureMode" not in AppConfig().to_dict()


def test_capture_mode_never_persists_across_a_reload(tmp_path: Path) -> None:
    bridge, _ = _bridge(tmp_path)
    bridge.overlayCaptureMode = True
    bridge.load_config(AppConfig())
    assert bridge.overlayCaptureMode is True  # load_config leaves runtime state alone ...
    appearance = Appearance(watch=False, overrides="")
    appearance.apply(AppearanceSignals(dark_mode=True))
    fresh = C.ControlBridge(AppConfig.load(tmp_path / "config.json"), tmp_path / "config.json", appearance=appearance)
    assert fresh.overlayCaptureMode is False  # ... and a new session always starts off


def test_control_window_relays_capture_mode_request(qapp, tmp_path: Path) -> None:
    window = C.ControlWindow(AppConfig(), tmp_path / "c.json")
    try:
        requests: List[bool] = []
        window.capture_mode_requested.connect(requests.append)
        window.bridge.overlayCaptureMode = True
        assert requests == [True]
    finally:
        window.shutdown()
        window.deleteLater()
        qapp.processEvents()


# ----------------------------------------------------------------------------- control window
class _FakeGrabber:
    """Stands in for ``BackdropGrabber``: records pause / resume, never touches the screen."""

    def __init__(self, geometry, on_frame, **_kw) -> None:
        self.log: List[str] = []
        self._paused = True

    @property
    def paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        self.log.append("start")

    def resume(self) -> None:
        self._paused = False
        self.log.append("resume")

    def pause(self) -> None:
        self._paused = True
        self.log.append("pause")

    def poke(self) -> None:
        self.log.append("poke")

    def stop(self) -> None:
        self.log.append("stop")

    def join(self, timeout: Optional[float] = None) -> None:
        self.log.append("join")


def _frame() -> BackdropFrame:
    return BackdropFrame(QImage(4, 4, QImage.Format.Format_RGB32), (68, 68), Rect(100, 100, 832, 640), 1.0, 0.5,
                         time.perf_counter())


@pytest.fixture()
def control(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A ``ControlWindow`` with a recording fake grabber and a recording fake ``user32``."""
    monkeypatch.setenv("GLASSTRANSLATE_APPEARANCE", "glass")  # glass allowed: the grabber exists once shown
    monkeypatch.setattr(CW, "BackdropGrabber", _FakeGrabber)
    fake = _FakeUser32()
    monkeypatch.setattr(W, "_user32", fake, raising=False)
    win = C.ControlWindow(AppConfig(), tmp_path / "c.json")
    yield win, fake
    win.shutdown()
    win.deleteLater()
    qapp.processEvents()


def test_control_window_defaults_to_excluded(control) -> None:
    win, _ = control
    assert win.capture_excluded is True


@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_control_window_set_capture_excluded_applies_immediately_when_shown(control) -> None:
    win, fake = control
    win.show()
    assert [flag for _, flag in fake.calls] == [W.WDA_EXCLUDEFROMCAPTURE]
    win.set_capture_excluded(False)
    assert win.capture_excluded is False
    assert [flag for _, flag in fake.calls][-1] == W.WDA_NONE
    win.set_capture_excluded(True)
    assert win.capture_excluded is True
    assert [flag for _, flag in fake.calls][-1] == W.WDA_EXCLUDEFROMCAPTURE


@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_control_window_show_reapplies_stored_affinity(control) -> None:
    win, fake = control
    win.set_capture_excluded(False)  # before the native window exists: stored only
    assert fake.calls == []
    win.show()
    assert [flag for _, flag in fake.calls] == [W.WDA_NONE]
    win.hide()
    win.show()
    assert [flag for _, flag in fake.calls] == [W.WDA_NONE, W.WDA_NONE]


def test_control_window_freezes_its_backdrop_while_capturable(control, qapp) -> None:
    """Capturable means the grabber would sample the panel itself: pause it, drop in-flight frames."""
    win, _ = control
    win.show()
    grabber = win.grabber
    assert isinstance(grabber, _FakeGrabber) and grabber.log == ["start", "resume"]
    win.set_capture_excluded(False)
    assert grabber.log[-1] == "pause"
    serial = win.bridge.backdropSerial
    win._frame_ready.emit(_frame())  # a grab that was already in flight when the affinity flipped
    qapp.processEvents()
    assert win.bridge.backdropSerial == serial  # dropped
    win.set_capture_excluded(True)
    assert grabber.log[-1] == "resume"  # re-excluded first, then a forced fresh grab
    win._frame_ready.emit(_frame())
    qapp.processEvents()
    assert win.bridge.backdropSerial == serial + 1


def test_control_window_stays_frozen_while_hidden_and_capturable(control) -> None:
    win, _ = control
    win.show()
    grabber = win.grabber
    win.set_capture_excluded(False)
    win.hide()
    win.show()  # show would normally resume the grabber
    assert grabber.log.count("resume") == 1
    assert grabber.paused is True


@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_control_window_freezes_before_unhiding_and_rehides_before_resuming(control) -> None:
    """The ordering invariant: no moment exists where the grabber runs on a capturable window."""
    win, fake = control
    win.show()
    grabber = win.grabber
    timeline: List[str] = []
    record_affinity = fake.SetWindowDisplayAffinity

    def set_affinity(hwnd, flag) -> int:
        timeline.append("WDA_NONE" if int(flag) == W.WDA_NONE else "EXCLUDE")
        return record_affinity(hwnd, flag)

    fake.SetWindowDisplayAffinity = set_affinity
    grabber.pause = lambda: timeline.append("pause")  # instance attributes shadow the fake's methods
    grabber.resume = lambda: timeline.append("resume")
    win.set_capture_excluded(False)
    win.set_capture_excluded(True)
    assert timeline == ["pause", "WDA_NONE", "EXCLUDE", "resume"]


@pytest.mark.skipif(not WIN, reason="Win32 only")
def test_control_window_close_forgets_the_native_window(control) -> None:
    """close() destroys the platform window; a later restore (shutdown) must not re-create it."""
    win, fake = control
    win.show()
    win.set_capture_excluded(False)
    calls = len(fake.calls)
    win.close()
    win.set_capture_excluded(True)  # what CaptureMode.restore() does at shutdown
    assert win.capture_excluded is True
    assert len(fake.calls) == calls  # stored only: no winId() on a destroyed window
    win.show()  # a re-show applies the stored state again
    assert [flag for _, flag in fake.calls][calls:] == [W.WDA_EXCLUDEFROMCAPTURE]


# -------------------------------------------------------------------------------- controller
class _FakeTarget:
    """A capturable window (the overlay or the control window): records every exclusion request."""

    def __init__(self) -> None:
        self.excluded: List[bool] = []
        self.segments_set: List[list] = []

    def set_capture_excluded(self, excluded: bool) -> None:
        self.excluded.append(bool(excluded))

    def set_segments(self, segments) -> None:
        self.segments_set.append(list(segments))


class _FakePipeline:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.log: List[str] = []

    def is_alive(self) -> bool:
        return self.alive

    def pause(self) -> None:
        self.log.append("pause")

    def resume(self) -> None:
        self.log.append("resume")


def _controller(pipeline: Optional[_FakePipeline], wants_running: bool):
    overlay, control = _FakeTarget(), _FakeTarget()
    status: List[str] = []
    state = {"pipeline": pipeline, "wants_running": wants_running}
    mode = CaptureMode(
        overlay,
        control,
        pipeline=lambda: state["pipeline"],
        wants_running=lambda: state["wants_running"],
        status=status.append,
    )
    return mode, overlay, control, status, state


def test_enabling_pauses_pipeline_and_unexcludes_both_windows() -> None:
    pipe = _FakePipeline()
    mode, overlay, control, status, _ = _controller(pipe, wants_running=True)
    assert mode.active is False
    mode.set_active(True)
    assert mode.active is True
    assert overlay.excluded == [False]
    assert control.excluded == [False]
    assert pipe.log == ["pause"]
    assert status and "capture" in status[-1].lower()
    assert "window" in status[-1].lower()  # the message names the control window too


def test_disabling_restores_previous_running_state() -> None:
    pipe = _FakePipeline()
    mode, overlay, control, status, _ = _controller(pipe, wants_running=True)
    mode.set_active(True)
    mode.set_active(False)
    assert overlay.excluded == [False, True]
    assert control.excluded == [False, True]
    assert pipe.log == ["pause", "resume"]
    assert mode.active is False
    assert len(status) == 2
    assert "off" in status[-1].lower() and "window" in status[-1].lower()  # the off message names both


def test_disabling_does_not_start_a_stopped_pipeline() -> None:
    pipe = _FakePipeline()
    mode, overlay, control, _, state = _controller(pipe, wants_running=False)
    mode.set_active(True)
    state["wants_running"] = False  # the user pressed Stop meanwhile (or it was never running)
    mode.set_active(False)
    assert pipe.log == ["pause"]
    assert overlay.excluded == [False, True]
    assert control.excluded == [False, True]


def test_toggle_without_a_pipeline_only_touches_the_windows() -> None:
    mode, overlay, control, _, _ = _controller(None, wants_running=False)
    mode.set_active(True)
    mode.set_active(False)
    assert overlay.excluded == [False, True]
    assert control.excluded == [False, True]


def test_set_active_is_idempotent() -> None:
    pipe = _FakePipeline()
    mode, overlay, control, status, _ = _controller(pipe, wants_running=True)
    mode.set_active(True)
    mode.set_active(True)
    assert overlay.excluded == [False]
    assert control.excluded == [False]
    assert pipe.log == ["pause"]
    assert len(status) == 1


def test_capture_mode_keeps_last_segments() -> None:
    pipe = _FakePipeline()
    mode, overlay, _, _, _ = _controller(pipe, wants_running=True)
    mode.set_active(True)
    mode.set_active(False)
    assert overlay.segments_set == []  # never clears what is painted


def test_hold_pauses_a_pipeline_started_while_active() -> None:
    mode, _, _, _, state = _controller(None, wants_running=False)
    mode.set_active(True)
    started = _FakePipeline()
    state["pipeline"] = started
    state["wants_running"] = True
    mode.hold()
    assert started.log == ["pause"]
    mode.set_active(False)
    assert started.log == ["pause", "resume"]


def test_hold_is_a_noop_when_inactive() -> None:
    pipe = _FakePipeline()
    mode, _, _, _, _ = _controller(pipe, wants_running=True)
    mode.hold()
    assert pipe.log == []


def test_restore_reexcludes_both_windows_when_mode_on() -> None:
    pipe = _FakePipeline()
    mode, overlay, control, _, _ = _controller(pipe, wants_running=True)
    mode.set_active(True)
    mode.restore()
    assert overlay.excluded == [False, True]
    assert control.excluded == [False, True]
    assert mode.active is False
    assert pipe.log == ["pause"]  # restore is for teardown: nothing is resumed
    mode.restore()
    assert overlay.excluded == [False, True]  # idempotent
    assert control.excluded == [False, True]


def test_restore_is_a_noop_when_mode_off() -> None:
    mode, overlay, control, _, _ = _controller(None, wants_running=False)
    mode.restore()
    assert overlay.excluded == []
    assert control.excluded == []
