"""Regression: switching translation backends (Argos -> Libre -> Argos) and
then closing the window froze the GUI for about a second.

``AppWindow.shutdown`` runs ``Pipeline.stop()`` on the GUI thread, and its
``join`` waited up to 5 s for the worker's current pass.  A backend switch
rebuilds the translator and clears the translation cache, so the very next
pass re-translates everything while stuck inside an uninterruptible call
(LibreTranslate ``urlopen`` or the CTranslate2 model reload) -- exactly when
the user closes the window.

Fakes only: a blocking translator/OCR stands in for the network or model call.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional, Sequence

import numpy as np

from glasstranslate.config.settings import AppConfig
from glasstranslate.core.interfaces import LanguageDetector, OCREngine, ScreenCapture, Translator
from glasstranslate.core.pipeline import Pipeline
from glasstranslate.core.types import Frame, Rect, Segment

W, H = 200, 120

_STOP_LATENCY_BUDGET_S = 1.0  # user-visible freeze budget for Pipeline.stop()
_BLOCK_S = 10.0  # how long the fake "network call" would block if never released


def _page() -> tuple[np.ndarray, List[Segment]]:
    img = np.full((H, W, 3), 255, np.uint8)
    quad = np.array([[10, 10], [110, 10], [110, 40], [10, 40]], dtype=np.float32)
    return img, [Segment("こんにちは", quad, 0.9)]


# ----------------------------------------------------------------- fakes
class _Capture(ScreenCapture):
    name = "fake"

    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def grab(self, region: Rect) -> Optional[Frame]:
        return Frame(self.image.copy(), region.x, region.y, 0.0)


class _OCR(OCREngine):
    """Optionally blocks inside ``recognize`` until ``gate`` is set, so a test
    can park the worker mid-pass at the OCR stage."""

    name = "fake"
    device = "cpu"

    def __init__(self, segments: Sequence[Segment], gate: Optional[threading.Event] = None) -> None:
        self.segments = list(segments)
        self.gate = gate
        self.entered = threading.Event()

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(_BLOCK_S)
        return [Segment(s.text, s.quad.copy(), s.confidence) for s in self.segments]

    def warmup(self) -> None:
        pass


class _Detector(LanguageDetector):
    def detect(self, text: str) -> Optional[str]:
        return "ja"


class _BlockingTranslator(Translator):
    """Blocks inside ``translate_batch`` like an in-flight LibreTranslate
    request or a CTranslate2 model reload; set ``release`` to let it finish."""

    name = "blocking"
    device = "cpu"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.closed = False

    def supported_pairs(self) -> List[tuple[str, str]]:
        return [("ja", "en")]

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        self.calls += 1
        self.entered.set()
        self.release.wait(_BLOCK_S)
        return [f"EN:{t}" for t in texts]

    def close(self) -> None:
        self.closed = True


class _NamedTranslator(Translator):
    """Records which backend name it was built for, its batches and close()."""

    device = "cpu"

    def __init__(self, backend: str) -> None:
        self.name = backend
        self.calls: List[List[str]] = []
        self.closed = False

    def supported_pairs(self) -> List[tuple[str, str]]:
        return [("ja", "en")]

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        self.calls.append(list(texts))
        return [f"{self.name}:{t}" for t in texts]

    def close(self) -> None:
        self.closed = True


def _make(img: np.ndarray, segs: Sequence[Segment], translator_factory, ocr: Optional[_OCR] = None) -> Pipeline:
    cfg = AppConfig(translation_backend="argos", source_lang="ja", target_lang="en", manga_mode=False)
    ocr = ocr or _OCR(segs)
    return Pipeline(
        cfg,
        region_provider=lambda: Rect(0, 0, W, H),
        on_result=lambda segments, stats: None,
        on_status=lambda msg: None,
        capture_factory=lambda c: _Capture(img),
        ocr_factory=lambda c: ocr,
        translator_factory=translator_factory,
        detector_factory=_Detector,
    )


def _switched(pipe: Pipeline, backend: str) -> AppConfig:
    d = pipe.config.to_dict()
    d["translation_backend"] = backend
    return AppConfig.from_dict(d)


# ----------------------------------------------------------------- tests
def test_stop_returns_quickly_while_translator_is_blocked() -> None:
    """The GUI thread must not freeze behind an uninterruptible translate call."""
    img, segs = _page()
    tr = _BlockingTranslator()
    pipe = _make(img, segs, lambda c: tr)
    pipe.start()
    try:
        assert tr.entered.wait(5.0), "worker never reached translate_batch"

        started = time.perf_counter()
        pipe.stop()
        elapsed = time.perf_counter() - started

        assert elapsed < _STOP_LATENCY_BUDGET_S, f"stop() blocked the caller for {elapsed:.2f}s"
    finally:
        tr.release.set()
        pipe.join(5.0)
    assert not pipe.is_alive()
    assert tr.closed  # run() releases the engines when the worker drains


def test_pending_stop_skips_translation_of_discarded_work() -> None:
    """Once stop is requested, the pass must not start new translate batches:
    their results are thrown away anyway, and each one can cost a network
    round-trip or a model load."""
    img, segs = _page()
    gate = threading.Event()
    ocr = _OCR(segs, gate=gate)
    tr = _BlockingTranslator()
    tr.release.set()  # would return instantly if (wrongly) called
    pipe = _make(img, segs, lambda c: tr, ocr=ocr)
    pipe.start()
    assert ocr.entered.wait(5.0), "worker never reached the OCR stage"

    stopper = threading.Thread(target=pipe.stop)
    stopper.start()  # sets the stop event immediately, then joins the worker
    time.sleep(0.3)
    gate.set()  # let the pass continue past OCR towards the translate stage
    stopper.join(_BLOCK_S)
    pipe.join(5.0)

    assert not pipe.is_alive()
    assert tr.calls == 0, "translate_batch was called for work that a pending stop discards"


def test_argos_libre_argos_switch_closes_old_and_uses_new_translator() -> None:
    img, segs = _page()
    created: List[_NamedTranslator] = []

    def factory(cfg: AppConfig) -> _NamedTranslator:
        tr = _NamedTranslator(cfg.translation_backend)
        created.append(tr)
        return tr

    pipe = _make(img, segs, factory)
    pipe.step()
    assert [t.name for t in created] == ["argos"]
    assert created[0].calls, "first pass did not translate"

    pipe.set_config(_switched(pipe, "libretranslate"))
    pipe.step()
    assert [t.name for t in created] == ["argos", "libretranslate"]
    assert created[0].closed, "old translator was not closed on switch"
    assert created[1].calls, "cache clear should force a real re-translation"

    pipe.set_config(_switched(pipe, "argos"))
    pipe.step()
    assert [t.name for t in created] == ["argos", "libretranslate", "argos"]
    assert created[1].closed
    assert created[2].calls and not created[2].closed


def test_stop_when_idle_closes_engines_promptly() -> None:
    img, segs = _page()
    tr = _BlockingTranslator()
    tr.release.set()  # never blocks: the worker is idle between passes
    pipe = _make(img, segs, lambda c: tr)
    pipe.start()
    assert tr.entered.wait(5.0), "worker never completed a pass"

    started = time.perf_counter()
    pipe.stop()
    elapsed = time.perf_counter() - started
    pipe.join(5.0)

    assert elapsed < _STOP_LATENCY_BUDGET_S
    assert not pipe.is_alive()
    assert tr.closed
