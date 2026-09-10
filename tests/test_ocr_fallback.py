"""``FallbackOCREngine``: primary (manga-ocr) with an always-available
fallback (PaddleOCR via rapidocr), observable through ``status`` and
``name``.  Fake engines and a fake clock only."""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import pytest

from glasstranslate.core.interfaces import OCREngine
from glasstranslate.core.types import Segment
from glasstranslate.ocr.chain import FallbackOCREngine

IMG = np.zeros((16, 32, 3), np.uint8)


def seg(text: str) -> Segment:
    return Segment(text, np.zeros((4, 2), np.float32), 0.9)


class FakeEngine(OCREngine):
    def __init__(self, name: str, results: Optional[List[Segment]] = None, device: str = "cpu") -> None:
        self.name = name
        self.device = device
        self.results = results or []
        self.error: Optional[Exception] = None  # raised by recognize while set
        self.warmup_error: Optional[Exception] = None
        self.calls = 0
        self.warmups = 0
        self.closed = False

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.results)

    def warmup(self) -> None:
        self.warmups += 1
        if self.warmup_error is not None:
            raise self.warmup_error

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class Factory:
    """Primary constructor that can be told to fail; counts its calls."""

    def __init__(self, engine: FakeEngine, error: Optional[Exception] = None) -> None:
        self.engine = engine
        self.error = error
        self.calls = 0

    def __call__(self) -> OCREngine:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.engine


def chain(
    factory: Callable[[], OCREngine], fallback: FakeEngine, clock: FakeClock, retry: float = 5.0
) -> tuple[FallbackOCREngine, List[str]]:
    statuses: List[str] = []
    eng = FallbackOCREngine(
        factory, fallback, primary_name="mangaocr", status=statuses.append, retry_after_s=retry, clock=clock
    )
    return eng, statuses


@pytest.fixture
def primary() -> FakeEngine:
    return FakeEngine("mangaocr", [seg("primary")], device="gpu")


@pytest.fixture
def fallback() -> FakeEngine:
    return FakeEngine("paddleocr", [seg("fallback")], device="cpu")


# ------------------------------------------------------------------ happy path
def test_primary_result_is_used_when_it_succeeds(primary: FakeEngine, fallback: FakeEngine) -> None:
    eng, statuses = chain(Factory(primary), fallback, FakeClock())
    out = eng.recognize(IMG)
    assert [s.text for s in out] == ["primary"]
    assert fallback.calls == 0 and primary.calls == 1
    assert eng.name == "mangaocr" and eng.active_engine == "mangaocr" and eng.primary_available
    assert statuses == []  # nothing to report on the happy path


def test_primary_is_constructed_lazily(primary: FakeEngine, fallback: FakeEngine) -> None:
    factory = Factory(primary)
    eng, _ = chain(factory, fallback, FakeClock())
    assert factory.calls == 0 and not eng.primary_available
    eng.recognize(IMG)
    eng.recognize(IMG)
    assert factory.calls == 1  # built once, reused


def test_empty_primary_output_is_not_a_fallback_trigger(primary: FakeEngine, fallback: FakeEngine) -> None:
    primary.results = []
    eng, statuses = chain(Factory(primary), fallback, FakeClock())
    assert eng.recognize(IMG) == []
    assert fallback.calls == 0 and statuses == []


# ------------------------------------------------------------------ failures
def test_construction_failure_falls_back_and_reports_once(primary: FakeEngine, fallback: FakeEngine) -> None:
    factory = Factory(primary, FileNotFoundError("models missing"))
    eng, statuses = chain(factory, fallback, FakeClock())
    for _ in range(3):
        assert [s.text for s in eng.recognize(IMG)] == ["fallback"]
    assert factory.calls == 1  # sticky on the fallback inside the backoff
    assert statuses == ["OCR: mangaocr failed (FileNotFoundError); using paddleocr"]
    assert eng.name == "paddleocr" and eng.active_engine == "paddleocr" and not eng.primary_available


def test_per_call_exception_falls_back_for_that_call(primary: FakeEngine, fallback: FakeEngine) -> None:
    clock = FakeClock()
    eng, statuses = chain(Factory(primary), fallback, clock)
    primary.error = RuntimeError("DML boom")
    assert [s.text for s in eng.recognize(IMG)] == ["fallback"]
    assert statuses == ["OCR: mangaocr failed (RuntimeError); using paddleocr"]
    assert eng.name == "paddleocr"
    primary.error = None
    clock.t += 5.0
    assert [s.text for s in eng.recognize(IMG)] == ["primary"]
    assert statuses[-1] == "OCR: mangaocr restored" and len(statuses) == 2
    assert eng.name == "mangaocr" and eng.primary_available


def test_fallback_does_not_spam_status_every_frame(primary: FakeEngine, fallback: FakeEngine) -> None:
    clock = FakeClock()
    factory = Factory(primary, RuntimeError("still broken"))
    eng, statuses = chain(factory, fallback, clock)
    for _ in range(10):
        eng.recognize(IMG)
        clock.t += 5.0  # every frame is past the backoff: the primary is retried each time
    assert factory.calls == 10
    assert len(statuses) == 1  # one message per transition, not per retry
    assert fallback.calls == 10


def test_primary_is_retried_after_the_backoff(primary: FakeEngine, fallback: FakeEngine) -> None:
    clock = FakeClock()
    factory = Factory(primary, FileNotFoundError("not yet"))
    eng, statuses = chain(factory, fallback, clock, retry=5.0)
    eng.recognize(IMG)
    clock.t += 4.9
    eng.recognize(IMG)
    assert factory.calls == 1  # inside the backoff: no retry
    factory.error = None  # models arrived
    clock.t += 0.2
    out = eng.recognize(IMG)
    assert factory.calls == 2 and [s.text for s in out] == ["primary"]
    assert statuses == [
        "OCR: mangaocr failed (FileNotFoundError); using paddleocr",
        "OCR: mangaocr restored",
    ]
    assert eng.primary_available


def test_both_engines_failing_returns_empty_and_reports_once(primary: FakeEngine, fallback: FakeEngine) -> None:
    factory = Factory(primary, FileNotFoundError("missing"))
    fallback.error = RuntimeError("no provider")
    eng, statuses = chain(factory, fallback, FakeClock())
    for _ in range(3):
        assert eng.recognize(IMG) == []  # never raises: the pipeline must not die
    assert len(statuses) == 2
    assert statuses[0].startswith("OCR: mangaocr failed (FileNotFoundError)")
    assert "paddleocr failed" in statuses[1] and "RuntimeError" in statuses[1]


def test_fallback_recovery_allows_a_new_report_later(primary: FakeEngine, fallback: FakeEngine) -> None:
    factory = Factory(primary, FileNotFoundError("missing"))
    fallback.error = RuntimeError("flaky")
    eng, statuses = chain(factory, fallback, FakeClock())
    eng.recognize(IMG)
    fallback.error = None
    assert [s.text for s in eng.recognize(IMG)] == ["fallback"]
    fallback.error = RuntimeError("flaky again")
    eng.recognize(IMG)
    assert sum("paddleocr failed" in s for s in statuses) == 2


# ------------------------------------------------------------------ identity
def test_name_and_device_reflect_the_active_engine(primary: FakeEngine, fallback: FakeEngine) -> None:
    clock = FakeClock()
    eng, _ = chain(Factory(primary), fallback, clock)
    eng.recognize(IMG)
    assert (eng.name, eng.device) == ("mangaocr", "gpu")
    primary.error = RuntimeError("x")
    eng.recognize(IMG)
    assert (eng.name, eng.device) == ("paddleocr", "cpu")
    assert eng.fallback is fallback and eng.primary_name == "mangaocr"


def test_status_callback_is_optional(primary: FakeEngine, fallback: FakeEngine) -> None:
    eng = FallbackOCREngine(Factory(primary, RuntimeError("x")), fallback, primary_name="mangaocr")
    assert [s.text for s in eng.recognize(IMG)] == ["fallback"]


# ------------------------------------------------------------------ lifecycle
def test_warmup_warms_both_engines(primary: FakeEngine, fallback: FakeEngine) -> None:
    eng, statuses = chain(Factory(primary), fallback, FakeClock())
    eng.warmup()
    assert primary.warmups == 1 and fallback.warmups == 1
    assert statuses == [] and eng.name == "mangaocr" and eng.primary_available


def test_warmup_handles_a_failing_primary(primary: FakeEngine, fallback: FakeEngine) -> None:
    primary.warmup_error = RuntimeError("DML init")
    eng, statuses = chain(Factory(primary), fallback, FakeClock())
    eng.warmup()
    assert fallback.warmups == 1
    assert statuses == ["OCR: mangaocr failed (RuntimeError); using paddleocr"]
    assert eng.name == "paddleocr" and not eng.primary_available
    assert [s.text for s in eng.recognize(IMG)] == ["fallback"]
    assert len(statuses) == 1


def test_warmup_propagates_a_failing_fallback(primary: FakeEngine, fallback: FakeEngine) -> None:
    fallback.warmup_error = RuntimeError("baseline engine broken")
    eng, _ = chain(Factory(primary), fallback, FakeClock())
    with pytest.raises(RuntimeError, match="baseline"):
        eng.warmup()


def test_close_closes_both_engines(primary: FakeEngine, fallback: FakeEngine) -> None:
    eng, _ = chain(Factory(primary), fallback, FakeClock())
    eng.warmup()
    eng.close()
    assert primary.closed and fallback.closed


def test_close_does_not_construct_the_primary(primary: FakeEngine, fallback: FakeEngine) -> None:
    factory = Factory(primary)
    eng, _ = chain(factory, fallback, FakeClock())
    eng.close()
    assert factory.calls == 0 and fallback.closed and not primary.closed
