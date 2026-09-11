"""``QualityScheduler``: dedupe, supersede, one job at a time, failure back-off, stop.

No sidecar and no sockets: the client is a stub the test drives.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import pytest

from glasstranslate.core.types import Rect
from glasstranslate.render import quality as Q


def _job(key: str, members: Tuple[str, ...] = ("a",)) -> Q.Job:
    rect = Rect(0, 0, 8, 8)
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 255
    return Q.Job(
        key=key,
        rect=rect,
        image=np.full((8, 8, 3), 10, np.uint8),
        mask=mask,
        members=tuple((m, 0, 0, 8, 8) for m in members),
    )


class _StubClient:
    """Records the jobs it is asked to inpaint; can be made to fail or block."""

    def __init__(self, *, fail: Optional[Exception] = None, device: str = "cpu") -> None:
        self.fail = fail
        self.device = device
        self.seen: List[np.ndarray] = []
        self.started = 0
        self.stopped = 0
        self.gate: Optional[threading.Event] = None

    def start(self) -> int:
        self.started += 1
        return 1234

    def health(self) -> dict:
        return {"ok": True, "mode": "fake", "device": self.device}

    def inpaint(self, image, mask, params=None):
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail is not None:
            raise self.fail
        self.seen.append(image.copy())
        out = image.copy()
        out[mask > 0] = 99
        return out

    def shutdown(self) -> None:
        self.stopped += 1


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _scheduler(client, results: list, statuses: list, **kwargs) -> Q.QualityScheduler:
    return Q.QualityScheduler(
        lambda: client,
        on_result=lambda job, crop: results.append((job.key, crop)),
        status=statuses.append,
        **kwargs,
    )


def test_one_job_runs_and_the_result_reaches_the_callback() -> None:
    client = _StubClient(device="cuda:0")
    results: list = []
    statuses: List[str] = []
    sched = _scheduler(client, results, statuses)
    try:
        sched.submit([_job("k1")])
        assert _wait(lambda: len(results) == 1)
        assert results[0][0] == "k1" and (results[0][1][3, 3] == 99).all()
        assert sched.available is True
        assert sched.stats["done"] == 1 and sched.stats["failed"] == 0
        assert any("Quality renderer: ready on cuda" in s for s in statuses)
    finally:
        sched.stop()
    assert client.stopped == 1


def test_the_same_content_key_is_never_resubmitted() -> None:
    client = _StubClient()
    results: list = []
    sched = _scheduler(client, results, [])
    try:
        sched.submit([_job("same")])
        assert _wait(lambda: len(results) == 1)
        sched.submit([_job("same")])
        sched.submit([_job("other")])
        assert _wait(lambda: len(results) == 2)
        time.sleep(0.05)
        assert [key for key, _ in results] == ["same", "other"]
    finally:
        sched.stop()


def test_a_queued_job_is_superseded_by_a_newer_one_for_the_same_members() -> None:
    client = _StubClient()
    client.gate = threading.Event()
    results: list = []
    sched = _scheduler(client, results, [])
    try:
        sched.submit([_job("running", ("a",))])          # taken by the worker, blocked on the gate
        assert _wait(lambda: sched.stats["in_flight"] == 1)
        sched.submit([_job("stale", ("a",))])            # queued
        sched.submit([_job("fresh", ("a",))])            # supersedes "stale"
        assert _wait(lambda: sched.stats["queued"] == 1)
        assert sched.stats["superseded"] == 1
        client.gate.set()
        assert _wait(lambda: len(results) == 2)
        assert [key for key, _ in results] == ["running", "fresh"]
    finally:
        client.gate.set()
        sched.stop()


def test_a_failing_client_reports_once_and_retries_after_the_back_off() -> None:
    client = _StubClient(fail=RuntimeError("stage failed"))
    results: list = []
    statuses: List[str] = []
    sched = _scheduler(client, results, statuses, retry_s=0.0)
    try:
        sched.submit([_job("k1")])
        assert _wait(lambda: sched.stats["failed"] >= 1)
        sched.submit([_job("k2")])
        assert _wait(lambda: sched.stats["failed"] >= 2)
        # One status line per distinct error, not one per job.
        unavailable = [s for s in statuses if "Quality renderer unavailable" in s]
        assert len(unavailable) == 1
        assert "quick fill" in unavailable[0] and "stage failed" in unavailable[0]
        assert results == []
        # The client is rebuilt for the retry (back-off 0 in this test).
        assert client.started >= 2
    finally:
        sched.stop()


def test_a_client_that_cannot_be_built_leaves_the_scheduler_unavailable() -> None:
    def factory():
        raise Q.QualityUnavailable("sidecar not installed")

    statuses: List[str] = []
    sched = Q.QualityScheduler(factory, on_result=lambda job, crop: None, status=statuses.append,
                               retry_s=0.0)
    try:
        sched.submit([_job("k1")])
        assert _wait(lambda: sched.stats["failed"] >= 1)
        assert sched.available is False
        assert any("sidecar not installed" in s for s in statuses)
    finally:
        sched.stop()


def test_stop_is_idempotent_and_drops_pending_work() -> None:
    client = _StubClient()
    sched = _scheduler(client, [], [])
    sched.submit([_job("k1")])
    sched.stop()
    sched.stop()
    assert client.stopped <= 1
    sched.submit([_job("k2")])  # after stop: ignored, never raises
    assert sched.stats["queued"] == 0


def test_submitting_nothing_is_free() -> None:
    sched = _scheduler(_StubClient(), [], [])
    try:
        sched.submit([])
        assert sched.stats["queued"] == 0
    finally:
        sched.stop()


def test_missing_models_reported_by_health_send_no_job_and_back_off_long() -> None:
    class _NoModels(_StubClient):
        def health(self) -> dict:
            return {"ok": True, "mode": "real", "device": "cuda:0",
                    "models": {"lama": "missing", "sdxl": "ready", "controlnet": "missing"},
                    "load_error": None}

    client = _NoModels()
    results: list = []
    statuses: List[str] = []
    sched = _scheduler(client, results, statuses, retry_s=0.0)
    try:
        sched.submit([_job("k1")])
        assert _wait(lambda: sched.stats["failed"] >= 1)
        assert client.seen == [] and results == []
        assert client.stopped == 1  # the half-started client is shut down, not leaked
        assert any("models not usable: controlnet, lama" in s for s in statuses)
        assert sched._next_attempt - time.monotonic() > 60  # MODELS_RETRY_S, not the 0 s retry
        assert not any("ready on" in s for s in statuses)
    finally:
        sched.stop()


def test_a_job_error_keeps_the_warm_sidecar_but_a_transport_error_drops_it() -> None:
    client = _StubClient(fail=Q.QualityUnavailable("/inpaint: HTTP 500 (stage failed)", transport=False))
    sched = _scheduler(client, [], [], retry_s=0.0)
    try:
        sched.submit([_job("k1")])
        assert _wait(lambda: sched.stats["failed"] >= 1)
        assert client.stopped == 0 and client.started == 1
        client.fail = Q.QualityUnavailable("/inpaint: ConnectionRefusedError")  # transport
        sched.submit([_job("k2")])
        assert _wait(lambda: sched.stats["failed"] >= 2)
        assert client.stopped == 1
    finally:
        sched.stop()


def test_stop_during_a_slow_start_still_shuts_the_sidecar_down() -> None:
    class _SlowStart(_StubClient):
        def __init__(self) -> None:
            super().__init__()
            self.release = threading.Event()

        def start(self) -> int:
            self.started += 1
            self.release.wait(5)
            return 1234

    client = _SlowStart()
    results: list = []
    sched = _scheduler(client, results, [])
    sched.submit([_job("k1")])
    assert _wait(lambda: client.started == 1)
    stopper = threading.Thread(target=sched.stop)
    stopper.start()
    time.sleep(0.2)
    client.release.set()
    stopper.join(10)
    assert _wait(lambda: client.stopped >= 1)
    assert results == []  # a job that outlives stop() delivers nothing


def test_default_factory_returns_none_without_the_model_store(tmp_path, monkeypatch) -> None:
    from glasstranslate.config.settings import AppConfig
    from glasstranslate.core import engines
    from glasstranslate.render import quality as QQ

    monkeypatch.setattr(QQ, "find_sidecar_python", lambda cfg: tmp_path / "python.exe")
    cfg = AppConfig()
    cfg.quality_renderer = "auto"
    cfg.models_dir = str(tmp_path)
    statuses: List[str] = []
    assert engines._default_quality_factory(cfg, on_result=lambda *a: None, status=statuses.append) is None
    assert statuses and "models not downloaded" in statuses[0]


@pytest.mark.parametrize("value", ["off", "", "nonsense"])
def test_default_factory_returns_none_unless_auto(value: str) -> None:
    from glasstranslate.config.settings import AppConfig
    from glasstranslate.core.engines import _default_quality_factory

    cfg = AppConfig()
    cfg.quality_renderer = value
    assert _default_quality_factory(cfg, on_result=lambda *a: None, status=lambda s: None) is None
