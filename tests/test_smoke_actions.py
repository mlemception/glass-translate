"""Smoke-action hook (``GLASSTRANSLATE_SMOKE_ACTIONS``): whitelist, download record, early finish.

The hook lets ``packaging/smoke_test.py`` run D drive the Engines-page manga-ocr download inside the
frozen exe.  These tests drive ``SmokeReport`` with a fake session so no window, pipeline or network
is needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QObject, Signal

from glasstranslate.ui import app as A

pytestmark = pytest.mark.usefixtures("qapp")


class _FakeBridge(QObject):
    downloadChanged = Signal()

    def __init__(self, models_dir: str) -> None:
        super().__init__()
        self.modelsDir = models_dir
        self.downloadLabel = ""
        self.downloadProgress = -1
        self.downloadActive = False
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


class _FakeSession:
    def __init__(self, bridge: _FakeBridge) -> None:
        self.control = _FakeControl(bridge)


def _report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, actions: str):
    monkeypatch.setenv(A.SMOKE_ACTIONS_ENV, actions)
    monkeypatch.setattr(A, "SMOKE_ACTION_SETTLE_MS", 0)
    bridge = _FakeBridge(str(tmp_path / "models"))
    report = A.SmokeReport(_FakeSession(bridge), tmp_path / "smoke.json", None)
    return report, bridge


def test_only_whitelisted_actions_are_queued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report, bridge = _report(tmp_path, monkeypatch, " deleteEverything ,downloadMangaOcr, ,close")
    assert report._pending_actions == ["downloadMangaOcr"]
    assert bridge.calls == 0  # nothing runs until the delayed timer fires


def test_no_env_means_no_actions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(A.SMOKE_ACTIONS_ENV, raising=False)
    bridge = _FakeBridge(str(tmp_path / "models"))
    report = A.SmokeReport(_FakeSession(bridge), tmp_path / "smoke.json", None)
    assert report._pending_actions == [] and report._actions == {}


def test_download_action_records_progress_and_finishes_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp) -> None:
    report, bridge = _report(tmp_path, monkeypatch, "downloadMangaOcr")
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
