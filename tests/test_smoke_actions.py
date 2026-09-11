"""Smoke-action hook (``GLASSTRANSLATE_SMOKE_ACTIONS``): whitelist, download record, early finish.

The hook lets ``packaging/smoke_test.py`` run D drive the Engines-page manga-ocr download inside the
frozen exe.  These tests drive ``SmokeReport`` with a fake session so no window, pipeline or network
is needed.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal

from glasstranslate.core.types import Rect
from glasstranslate.ui import app as A
from glasstranslate.ui import smoke_report as SR

pytestmark = pytest.mark.usefixtures("qapp")


class _FakeBridge(QObject):
    downloadChanged = Signal()

    def __init__(self, models_dir: str) -> None:
        super().__init__()
        self.modelsDir = models_dir
        self.downloadLabel = ""
        self.downloadProgress = -1
        self.downloadActive = False
        self.qualityStatus = "Ready"
        self.calls = 0

    def downloadMangaOcr(self) -> None:
        self.calls += 1
        self.downloadActive = True
        self.downloadLabel = "Downloading manga-ocr models…"
        self.downloadChanged.emit()

    def progress(self, done_mb: float, total_mb: float, pct: int) -> None:
        self.downloadLabel = f"Downloading manga-ocr models: {done_mb:.1f} / {total_mb:.1f} MB"
        self.downloadProgress = pct
        self.downloadChanged.emit()

    def finish(self) -> None:
        self.downloadActive = False
        self.downloadProgress = 100
        self.downloadChanged.emit()


class _FakeControl(QObject):
    frameSwapped = Signal()

    def __init__(self, bridge: _FakeBridge) -> None:
        super().__init__()
        self.bridge = bridge


class _FakeConfig:
    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self.quality_renderer = "auto"
        self.quality_sidecar_python = ""


class _FakeOverlay:
    """Only what the report reads: the patch-cache counters of ``GlassOverlay``."""

    def __init__(self) -> None:
        self.upgrades = 0

    @property
    def patch_stats(self) -> dict:
        return {"conversions": 7, "upgrades": self.upgrades}


class _FakePipeline:
    SNAPSHOT = {"active": True, "available": True, "stats": {"done": 2, "failed": 0}}

    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.replaced: List = []
        self.stops = 0

    def is_alive(self) -> bool:
        return self.alive

    def replace_capture(self, factory) -> None:
        self.replaced.append(factory)

    def stop(self, timeout: float = 0.0) -> None:
        self.stops += 1

    def quality_snapshot(self) -> dict:
        return dict(self.SNAPSHOT)


class _FakeSession(QObject):
    """Stands in for ``GlassTranslateApp``: the pieces the smoke actions touch."""

    _result_ready = Signal(object, object)

    def __init__(self, bridge: _FakeBridge, config_path: Path = Path("config.json")) -> None:
        super().__init__()
        self.control = _FakeControl(bridge)
        self.overlay = _FakeOverlay()
        self.pipeline = _FakePipeline()
        self.cfg = _FakeConfig(bridge.modelsDir)
        self.config_path = config_path
        self.capture_factory = None
        self.starts = 0

    def start_pipeline(self) -> None:
        self.starts += 1
        self.pipeline = _FakePipeline()


def _report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actions: str):
    monkeypatch.setenv(SR.SMOKE_ACTIONS_ENV, actions)
    monkeypatch.setattr(SR, "SMOKE_ACTION_SETTLE_MS", 0)
    bridge = _FakeBridge(str(tmp_path / "models"))
    session = _FakeSession(bridge, tmp_path / "config.json")
    report = SR.SmokeReport(session, tmp_path / "smoke.json", None)
    return report, bridge, session


def _settle(qapp, done, timeout_s: float = 5.0) -> None:
    """Pump the event loop until ``done()`` or the timeout (the probe runs on a worker thread)."""
    deadline = time.monotonic() + timeout_s
    while not done() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_only_whitelisted_actions_are_queued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report, bridge, _ = _report(tmp_path, monkeypatch, " deleteEverything ,downloadMangaOcr, ,close,feedPage")
    assert report._pending_actions == ["downloadMangaOcr", "feedPage"]
    assert SR._SMOKE_ACTIONS == ("downloadMangaOcr", "probeSidecar", "feedPage")
    assert bridge.calls == 0  # nothing runs until the delayed timer fires


def test_no_env_means_no_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SR.SMOKE_ACTIONS_ENV, raising=False)
    bridge = _FakeBridge(str(tmp_path / "models"))
    report = SR.SmokeReport(_FakeSession(bridge), tmp_path / "smoke.json", None)
    assert report._pending_actions == [] and report._actions == {}


def test_download_action_records_progress_and_finishes_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    report, bridge, _ = _report(tmp_path, monkeypatch, "downloadMangaOcr")
    ready = {"value": False}
    monkeypatch.setattr("glasstranslate.ocr.models.models_ready", lambda models_dir, files=None: ready["value"])

    report._run_actions()
    record = report._actions["downloadMangaOcr"]
    assert bridge.calls == 1
    assert record["started"] is True and record["finished"] is False
    assert record["models_dir"] == str(tmp_path / "models") and record["models_ready_before"] is False
    assert report._pending_actions == []

    bridge.progress(50.0, 200.0, 25)
    assert record["progress"] == 25 and "50.0 / 200.0" in record["label"]

    ready["value"] = True
    bridge.finish()
    assert record["finished"] is True and record["progress"] == 100
    assert record["models_ready_after"] is True and record["seconds"] >= 0.0

    bridge.progress(1.0, 2.0, 3)  # a later downloadChanged must not reopen the record
    assert record["progress"] == 100 and record["finished"] is True

    qapp.processEvents()  # the 0 ms settle timer: asks for the grab; no frame swapped yet -> deferred, no crash
    assert report._pending_grab is True and report.written is False


# ------------------------------------------------------------------------------- probeSidecar
def _install_sidecar(monkeypatch: pytest.MonkeyPatch, launcher: Optional[Path], *, fail: str = "",
                     block: Optional[threading.Event] = None) -> List:
    """Point the app's sidecar lookup at ``launcher`` and replace ``QualityClient`` by a recorder.

    ``block`` makes ``start()`` hang until the event is set, standing in for a sidecar that
    never prints READY.
    """
    import glasstranslate.render.quality as Q

    made: List = []

    class _Client:
        def __init__(self, python, models_dir, **kwargs) -> None:
            self.python, self.models_dir, self.kwargs = python, models_dir, kwargs
            self.started = self.shut_down = False
            made.append(self)

        def start(self) -> int:
            if block is not None:
                block.wait(30.0)
            if fail:
                raise RuntimeError(fail)
            self.started = True
            return 51234

        def health(self) -> dict:
            return {"ok": True, "mode": "fake", "models": {"lama": "ready"}}

        def shutdown(self) -> None:
            self.shut_down = True

    monkeypatch.setattr(Q, "find_sidecar_python", lambda cfg: launcher)
    monkeypatch.setattr(Q, "QualityClient", _Client)
    return made


def test_main_connects_the_report_before_the_shutdown_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    """Qt runs ``aboutToQuit`` slots in connection order, so the report - and its grabWindow -
    must be written before the session tears the pipeline and the quality sidecar down (a real
    sidecar takes tens of seconds to exit after a job)."""
    from PySide6.QtGui import QSurfaceFormat

    order: List[str] = []

    class _Session:
        def __init__(self, cfg, config_path=None) -> None:
            self.control = SimpleNamespace(closed=SimpleNamespace(connect=lambda slot: None))

        def show(self) -> None:
            pass

        def shutdown(self) -> None:
            order.append("shutdown")

        def teardown(self) -> None:
            pass

    class _Report:
        def __init__(self, session, path, autoexit_ms, parent=None) -> None:
            order.append("report created")
            qapp.aboutToQuit.connect(lambda: order.append("report written"))

        def write_if_needed(self) -> None:
            pass

    monkeypatch.setattr(A, "GlassTranslateApp", _Session)
    monkeypatch.setattr(A, "SmokeReport", _Report)
    monkeypatch.setattr(A, "_configure_logging", lambda *a, **k: [])
    monkeypatch.setattr(A, "qInstallMessageHandler", lambda handler: None)
    monkeypatch.setenv(A.AUTOEXIT_ENV, "0")  # quit as soon as the loop starts
    monkeypatch.setenv(A.SMOKE_LOG_ENV, str(tmp_path / "smoke.json"))
    surface, quit_on_close = QSurfaceFormat.defaultFormat(), qapp.quitOnLastWindowClosed()
    try:
        assert A.main(tmp_path / "config.json", argv=["test"]) == 0
    finally:
        QSurfaceFormat.setDefaultFormat(surface)
        qapp.setQuitOnLastWindowClosed(quit_on_close)
    assert order == ["report created", "report written", "shutdown"]


def test_app_re_exports_the_smoke_report_names() -> None:
    """``app`` stays the import site for the tests and the packaging scripts after the split."""
    assert A.SmokeReport is SR.SmokeReport and A.exercise_pages is SR.exercise_pages
    for name in ("SMOKE_LOG_ENV", "SMOKE_ACTIONS_ENV", "SMOKE_PAGE_ENV", "SMOKE_ACTION_DELAY_MS",
                 "SMOKE_ACTION_SETTLE_MS", "SMOKE_GRAB_LEAD_MS", "SMOKE_PAGES_LEAD_MS",
                 "PROBE_READY_TIMEOUT_S"):
        assert getattr(A, name) == getattr(SR, name), name


def test_probe_sidecar_records_a_venv_launcher_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    launcher = tmp_path / "renderer" / ".venv" / "Scripts" / "python.exe"
    _install_sidecar(monkeypatch, launcher)
    report, _, _ = _report(tmp_path, monkeypatch, "probeSidecar")
    report._run_actions()
    record = report._actions["probeSidecar"]
    _settle(qapp, lambda: bool(record["ok"] or record["error"]))
    assert record["kind"] == "venv" and record["ok"] is True


def test_probe_sidecar_starts_a_fake_client_and_records_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    launcher = tmp_path / "renderer" / "glassrenderer.exe"
    made = _install_sidecar(monkeypatch, launcher)
    report, bridge, _ = _report(tmp_path, monkeypatch, "probeSidecar")

    report._run_actions()
    record = report._actions["probeSidecar"]
    _settle(qapp, lambda: bool(record["ok"] or record["error"]))

    assert record["launcher"] == str(launcher) and record["kind"] == "frozen"
    assert record["ok"] is True and record["error"] == ""
    assert record["health"]["mode"] == "fake" and record["seconds"] >= 0.0
    client = made[0]
    assert str(client.python) == str(launcher) and client.models_dir == bridge.modelsDir
    assert client.kwargs["fake"] is True and client.kwargs["log_path"] is None
    assert client.kwargs["ready_timeout"] == SR.PROBE_READY_TIMEOUT_S
    assert client.shut_down is True


def test_probe_sidecar_without_a_launcher_records_no_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    made = _install_sidecar(monkeypatch, None)
    report, _, _ = _report(tmp_path, monkeypatch, "probeSidecar")
    report._run_actions()
    record = report._actions["probeSidecar"]
    assert record["launcher"] is None and record["kind"] is None
    assert record["ok"] is False and record["error"] == "no sidecar" and not made


def test_probe_sidecar_watchdog_records_a_hang_and_releases_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    """A sidecar that never prints READY must fail a field, not hold the app to its autoexit."""
    release = threading.Event()
    made = _install_sidecar(monkeypatch, tmp_path / "renderer" / "glassrenderer.exe", block=release)
    monkeypatch.setattr(SR, "PROBE_WATCHDOG_MS", 0)
    report, _, _ = _report(tmp_path, monkeypatch, "probeSidecar")

    report._run_actions()
    record = report._actions["probeSidecar"]
    _settle(qapp, lambda: bool(record["error"]))
    assert record["ok"] is False and record["error"] == "probe did not finish within 30 s"
    assert report._running_actions == set()  # the report may be written now

    release.set()  # the worker thread finally comes back: its late result must not overwrite
    _settle(qapp, lambda: bool(made) and made[0].shut_down)
    assert record["error"] == "probe did not finish within 30 s" and record["ok"] is False


def test_probe_ready_timeout_fits_inside_the_portable_autoexit() -> None:
    assert SR.PROBE_READY_TIMEOUT_S == 30.0
    assert SR.PROBE_WATCHDOG_MS == 30_000


def test_probe_sidecar_records_a_failure_and_still_shuts_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    made = _install_sidecar(monkeypatch, tmp_path / "renderer" / "glassrenderer.exe", fail="no CUDA here")
    report, _, _ = _report(tmp_path, monkeypatch, "probeSidecar")
    report._run_actions()
    record = report._actions["probeSidecar"]
    _settle(qapp, lambda: bool(record["error"]))
    assert record["ok"] is False and "no CUDA here" in record["error"]
    assert made[0].shut_down is True


# ----------------------------------------------------------------------------------- feedPage
def _page(tmp_path: Path, name: str = "page.png") -> Path:
    """The fixture lives under a non-ASCII directory on purpose: ``cv2.imread`` cannot open
    such a path on Windows, so ``StaticPageCapture`` must decode from memory."""
    folder = tmp_path / "pàge ü"
    folder.mkdir(exist_ok=True)
    path = folder / name
    image = np.full((40, 60, 3), 210, np.uint8)
    image[10:20, 5:25] = 30
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    assert ok
    path.write_bytes(buffer.tobytes())
    return path


def _block(serial: int, is_block: bool = True) -> SimpleNamespace:
    return SimpleNamespace(style=SimpleNamespace(is_block=is_block, clean_patch_serial=serial))


def test_static_page_capture_serves_the_whole_page_from_a_non_ascii_path(tmp_path: Path) -> None:
    from glasstranslate.capture import StaticPageCapture

    page = _page(tmp_path)
    assert not page.parent.name.isascii()  # the portable sandbox is such a path
    capture = StaticPageCapture(page)
    assert capture.name == "static" and capture.size == (60, 40)
    frame = capture.grab(Rect(7, 9, 10, 10))
    assert frame is not None and frame.image.shape == (40, 60, 3)
    assert (frame.origin_x, frame.origin_y) == (7, 9) and frame.timestamp > 0
    capture.close()


def test_static_page_capture_rejects_an_undecodable_file(tmp_path: Path) -> None:
    from glasstranslate.capture import StaticPageCapture

    bad = tmp_path / "not-an-image.png"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        StaticPageCapture(bad)


def test_static_page_capture_rejects_a_missing_file(tmp_path: Path) -> None:
    from glasstranslate.capture import StaticPageCapture

    with pytest.raises(ValueError):
        StaticPageCapture(tmp_path / "pàge ü" / "never-written.png")


def test_feed_page_swaps_the_capture_of_the_running_pipeline_and_finishes_on_an_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    page = _page(tmp_path)
    monkeypatch.setenv(SR.SMOKE_PAGE_ENV, str(page))
    report, _, session = _report(tmp_path, monkeypatch, "feedPage")
    pipeline = session.pipeline

    report._run_actions()
    record = report._actions["feedPage"]
    assert record["page"] == page.name  # the basename only, never the directory
    assert record["started"] is True and record["status"] == "running"
    # A live worker swaps its own capture: restarting it would race two engine loads.
    assert pipeline.replaced == [session.capture_factory]
    assert session.starts == 0 and pipeline.stops == 0
    capture = session.capture_factory(session.cfg)
    assert capture.name == "static" and capture.grab(Rect(0, 0, 5, 5)).image.shape == (40, 60, 3)

    session._result_ready.emit([_block(0), _block(0, is_block=False)], None)
    assert record["finished"] is False and record["blocks"] == 1 and record["upgraded"] == 0

    session.overlay.upgrades = 1  # the overlay re-converted the upgraded patch
    session._result_ready.emit([_block(1)], None)
    assert record["finished"] is True and record["status"] == "upgraded"
    assert record["upgraded"] == 1 and record["overlay_upgrades"] == 1
    assert record["quality_stats"] == _FakePipeline.SNAPSHOT and record["seconds"] >= 0.0


@pytest.mark.parametrize("state", ["none", "dead"])
def test_feed_page_starts_a_pipeline_when_none_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp, state: str
) -> None:
    """Without a live worker the factory is set first and the pipeline built around it."""
    monkeypatch.setenv(SR.SMOKE_PAGE_ENV, str(_page(tmp_path)))
    report, _, session = _report(tmp_path, monkeypatch, "feedPage")
    session.pipeline = None if state == "none" else _FakePipeline(alive=False)

    report._run_actions()
    assert report._actions["feedPage"]["started"] is True
    assert session.starts == 1 and session.capture_factory is not None
    assert session.capture_factory(session.cfg).name == "static"
    assert session.pipeline.replaced == []  # the new worker was constructed with the factory


def test_feed_page_waits_for_the_overlay_to_re_convert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    monkeypatch.setenv(SR.SMOKE_PAGE_ENV, str(_page(tmp_path)))
    report, _, session = _report(tmp_path, monkeypatch, "feedPage")
    report._run_actions()
    record = report._actions["feedPage"]
    session._result_ready.emit([_block(3)], None)  # the serial advanced, the glass has not caught up
    assert record["upgraded"] == 1 and record["overlay_upgrades"] == 0 and record["finished"] is False
    assert report._feed_rechecks == 1  # ...so a re-read is armed


def test_feed_page_finishes_from_the_re_check_when_the_overlay_catches_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    """The overlay converts the upgraded patch one event-loop turn after the result that
    carried it, and a still page emits nothing more - the re-check is what finishes the run."""
    monkeypatch.setenv(SR.SMOKE_PAGE_ENV, str(_page(tmp_path)))
    monkeypatch.setattr(SR, "FEED_RECHECK_MS", 0)
    report, _, session = _report(tmp_path, monkeypatch, "feedPage")
    report._run_actions()
    record = report._actions["feedPage"]

    session._result_ready.emit([_block(1)], None)
    assert record["finished"] is False and record["overlay_upgrades"] == 0

    session.overlay.upgrades = 1  # the glass caught up; no further result will ever arrive
    _settle(qapp, lambda: bool(record["finished"]), timeout_s=2.0)
    assert record["finished"] is True and record["overlay_upgrades"] == 1
    assert record["status"] == "upgraded" and record["seconds"] >= 0.0
    assert report._running_actions == set()  # the settle timer is armed


def test_feed_page_re_check_gives_up_after_the_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    """A conversion that never happens still ends at the autoexit, with the counts recorded."""
    monkeypatch.setenv(SR.SMOKE_PAGE_ENV, str(_page(tmp_path)))
    monkeypatch.setattr(SR, "FEED_RECHECK_MS", 0)
    monkeypatch.setattr(SR, "FEED_RECHECK_MAX", 3)
    report, _, session = _report(tmp_path, monkeypatch, "feedPage")
    report._run_actions()
    record = report._actions["feedPage"]

    session._result_ready.emit([_block(1)], None)
    _settle(qapp, lambda: report._feed_rechecks >= 3, timeout_s=2.0)
    assert report._feed_rechecks == 3 and record["finished"] is False
    assert record["upgraded"] == 1 and record["overlay_upgrades"] == 0
    assert report._running_actions == {"feedPage"}


def test_feed_page_without_the_page_env_records_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    monkeypatch.delenv(SR.SMOKE_PAGE_ENV, raising=False)
    report, _, session = _report(tmp_path, monkeypatch, "feedPage")
    report._run_actions()
    record = report._actions["feedPage"]
    assert record["started"] is False and record["status"] == "error"
    assert SR.SMOKE_PAGE_ENV in record["error"]
    assert session.starts == 0 and session.pipeline.replaced == []


def test_feed_page_refuses_a_file_that_is_not_an_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    text = tmp_path / "notes.txt"
    text.write_text("not a page", encoding="utf-8")
    monkeypatch.setenv(SR.SMOKE_PAGE_ENV, str(text))
    report, _, session = _report(tmp_path, monkeypatch, "feedPage")
    report._run_actions()
    record = report._actions["feedPage"]
    assert record["status"] == "error" and "not an image" in record["error"]
    assert record["page"] == "notes.txt" and session.starts == 0


def test_feed_page_refuses_an_oversized_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    page = _page(tmp_path)
    monkeypatch.setenv(SR.SMOKE_PAGE_ENV, str(page))
    monkeypatch.setattr(SR, "PAGE_MAX_BYTES", 8)  # the real cap is 64 MiB
    report, _, session = _report(tmp_path, monkeypatch, "feedPage")
    report._run_actions()
    record = report._actions["feedPage"]
    assert record["status"] == "error" and "over the 8 byte cap" in record["error"]
    assert session.starts == 0 and session.pipeline.replaced == []


def test_feed_page_with_an_unreadable_page_records_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    bad = tmp_path / "not-an-image.png"
    bad.write_text("nope", encoding="utf-8")
    monkeypatch.setenv(SR.SMOKE_PAGE_ENV, str(bad))
    report, _, session = _report(tmp_path, monkeypatch, "feedPage")
    report._run_actions()
    record = report._actions["feedPage"]
    assert record["started"] is False and record["status"] == "error"
    assert session.starts == 0 and session.pipeline.replaced == []


def test_two_actions_quit_only_once_both_have_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp
) -> None:
    """``probeSidecar,feedPage``: the probe finishing first must not cut the page run short."""
    _install_sidecar(monkeypatch, tmp_path / "renderer" / "glassrenderer.exe")
    monkeypatch.setenv(SR.SMOKE_PAGE_ENV, str(_page(tmp_path)))
    report, _, session = _report(tmp_path, monkeypatch, "probeSidecar,feedPage")

    report._run_actions()
    assert report._running_actions == {"probeSidecar", "feedPage"}
    _settle(qapp, lambda: bool(report._actions["probeSidecar"]["ok"]))
    # The probe is done, the page is not: no quit is armed yet.
    assert report._running_actions == {"feedPage"}
    qapp.processEvents()
    assert report._pending_grab is False and report.written is False

    session.overlay.upgrades = 1
    session._result_ready.emit([_block(1)], None)
    assert report._actions["feedPage"]["finished"] is True and report._running_actions == set()
    qapp.processEvents()  # now the 0 ms settle timer runs (no frame swapped -> deferred grab)
    assert report._pending_grab is True


# ------------------------------------------------------------------- always-present report blocks
def test_report_blocks_carry_quality_overlay_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    launcher = tmp_path / "renderer" / "glassrenderer.exe"
    _install_sidecar(monkeypatch, launcher)
    monkeypatch.setattr("glasstranslate.render.quality_models.models_ready",
                        lambda models_dir, files=None: True)
    monkeypatch.setattr(SR, "user_data_dir", lambda: tmp_path / "profile")
    monkeypatch.setattr(SR, "logs_dir", lambda: tmp_path / "profile" / "logs")
    monkeypatch.setattr(SR, "portable_root", lambda: tmp_path / "root")
    report, bridge, session = _report(tmp_path, monkeypatch, "")

    quality = report._quality_block()
    assert quality["setting"] == "auto" and quality["launcher"] == str(launcher)
    assert quality["kind"] == "frozen" and quality["models_ready"] is True
    assert quality["hint"] == bridge.qualityStatus
    assert quality["scheduler_active"] is True and quality["scheduler_available"] is True
    assert quality["scheduler_stats"] == _FakePipeline.SNAPSHOT["stats"]

    assert report._overlay_block() == {"patch_conversions": 7, "patch_upgrades": 0}
    paths = report._paths_block()
    assert paths["config"] == str(session.config_path) and paths["models_dir"] == session.cfg.models_dir
    assert paths["portable_root"] == str(tmp_path / "root")
    assert paths["user_data_dir"] == str(tmp_path / "profile")
    assert paths["logs"] == str(tmp_path / "profile" / "logs")


def test_paths_block_reports_no_portable_root_outside_a_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    monkeypatch.setattr(SR, "portable_root", lambda: None)
    report, _, _ = _report(tmp_path, monkeypatch, "")
    assert report._paths_block()["portable_root"] is None


def test_quality_block_without_a_sidecar_or_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    _install_sidecar(monkeypatch, None)
    monkeypatch.setattr("glasstranslate.render.quality_models.models_ready",
                        lambda models_dir, files=None: False)
    report, _, session = _report(tmp_path, monkeypatch, "")
    session.pipeline = None
    quality = report._quality_block()
    assert quality["launcher"] is None and quality["kind"] is None and quality["models_ready"] is False
    assert quality["scheduler_active"] is None and quality["scheduler_available"] is None
    assert quality["scheduler_stats"] is None
