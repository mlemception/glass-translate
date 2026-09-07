"""Backdrop helpers and the grabber thread with a fake ``mss`` (docs/GLASS_DESIGN.md 1.2)."""
from __future__ import annotations

import sys
import threading
import time
import types
from typing import List

import numpy as np
import pytest
from PySide6.QtGui import QImage

from glasstranslate.core.types import Rect
from glasstranslate.ui.glass import backdrop as BD

pytestmark = pytest.mark.usefixtures("qapp")


def test_clamp_and_slab_rect() -> None:
    assert BD.clamp_rect(Rect(-10, -10, 50, 50), Rect(0, 0, 100, 100)) == Rect(0, 0, 40, 40)
    assert BD.clamp_rect(Rect(90, 90, 50, 50), Rect(0, 0, 100, 100)) == Rect(90, 90, 10, 10)
    assert BD.clamp_rect(Rect(200, 200, 5, 5), Rect(0, 0, 100, 100)).w == 0
    assert BD.slab_rect(Rect(100, 100, 832, 640), 1.0) == Rect(136, 128, 760, 560)
    assert BD.slab_rect(Rect(0, 0, 1664, 1280), 2.0) == Rect(72, 56, 1520, 1120)


def test_downsample_and_luma() -> None:
    arr = np.zeros((64, 80, 4), np.uint8)
    arr[..., :3] = 255
    sample = BD.downsample(arr)
    assert sample.shape == (8, 10, 4) and sample.flags["C_CONTIGUOUS"] and int(sample[0, 0, 0]) == 255
    assert BD.mean_luma(sample) == pytest.approx(1.0)
    arr[:32] = 0  # top half black
    sample = BD.downsample(arr)
    assert BD.mean_luma(sample) == pytest.approx(0.5)
    assert BD.mean_luma(sample, Rect(0, 0, 80, 32)) == pytest.approx(0.0)
    assert BD.mean_luma(sample, Rect(0, 32, 80, 32)) == pytest.approx(1.0)
    assert BD.mean_luma(sample, Rect(-100, -100, 180, 132)) == pytest.approx(0.0)  # negative origin clamps
    assert BD.mean_luma(sample, Rect(-100, -100, 80, 132)) == pytest.approx(0.5)  # fully off-sample -> whole
    assert BD.mean_luma(sample, Rect(500, 500, 10, 10)) == pytest.approx(0.5)  # empty crop -> whole
    # a change on a sampled pixel is detected; the sample is a copy, not a view
    before = BD.downsample(arr)
    arr[40, 40, 1] = 0
    assert not np.array_equal(BD.downsample(arr), before)
    # channel weights: pure blue is darker than pure green in BT.601
    blue = np.zeros((8, 8, 4), np.uint8)
    blue[..., 0] = 255
    green = np.zeros((8, 8, 4), np.uint8)
    green[..., 1] = 255
    assert BD.mean_luma(BD.downsample(blue)) < BD.mean_luma(BD.downsample(green))
    tiny = np.full((3, 5, 4), 255, np.uint8)
    assert BD.downsample(tiny).shape == (1, 1, 4) and BD.mean_luma(BD.downsample(tiny)) == pytest.approx(1.0)


def test_luma_smoother_tau() -> None:
    s = BD.LumaSmoother(0.25)
    assert s.update(1.0, 0.0) == 1.0
    assert s.update(0.0, 0.25) == pytest.approx(np.exp(-1), abs=1e-6)  # one tau -> 1/e
    assert s.update(0.0, 5.0) == pytest.approx(0.0, abs=1e-6)
    s.reset()
    assert s.value is None and s.update(0.3, 1.0) == 0.3


class _FakeShot:
    def __init__(self, arr: np.ndarray) -> None:
        self.height, self.width = arr.shape[:2]
        self.bgra = arr.tobytes()


class _FakeMSS:
    """Stands in for mss.MSS: a 400x300 virtual screen whose content is a settable colour."""

    colour = 40
    created = 0
    closed = 0
    grabs: List[dict] = []

    def __init__(self) -> None:
        _FakeMSS.created += 1
        self.monitors = [{"left": 0, "top": 0, "width": 400, "height": 300}]

    def grab(self, region: dict) -> _FakeShot:
        _FakeMSS.grabs.append(dict(region))
        arr = np.full((region["height"], region["width"], 4), _FakeMSS.colour, np.uint8)
        arr[..., 3] = 255
        return _FakeShot(arr)

    def close(self) -> None:
        _FakeMSS.closed += 1


@pytest.fixture()
def fake_mss(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("mss")
    module.MSS = _FakeMSS
    monkeypatch.setitem(sys.modules, "mss", module)
    _FakeMSS.colour, _FakeMSS.created, _FakeMSS.closed, _FakeMSS.grabs = 40, 0, 0, []
    return _FakeMSS


def _wait(pred, timeout: float = 2.0) -> bool:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_grabber_thread(fake_mss) -> None:
    frames: List[BD.BackdropFrame] = []
    lock = threading.Lock()

    def on_frame(frame: BD.BackdropFrame) -> None:
        with lock:
            frames.append(frame)

    geometry = BD.WindowGeometry()
    grabber = BD.BackdropGrabber(geometry, on_frame, hz=60.0)
    grabber.start()
    try:
        assert grabber.paused  # starts paused
        time.sleep(0.1)
        assert grabber.grab_count == 0 and fake_mss.created == 0
        grabber.resume()
        time.sleep(0.1)
        assert grabber.grab_count == 0  # no geometry yet: nothing to grab
        geometry.update(Rect(10, 20, 100, 50), 1.0)
        assert _wait(lambda: len(frames) >= 1)
        frame = frames[0]
        # window (10,20,100,50) + 32 px margin, clamped at the virtual screen's left/top edge
        assert frame.origin == (0, 0)
        assert (frame.image.width(), frame.image.height()) == (142, 102)
        assert frame.image.format() == QImage.Format.Format_RGB32 and frame.image.devicePixelRatio() == 1.0
        assert frame.origin_logical == (-10.0, -20.0)
        assert frame.luma == pytest.approx(40 / 255, abs=1e-3)
        assert fake_mss.grabs[0] == {"left": 0, "top": 0, "width": 142, "height": 102}
        # unchanged desktop: grabs continue, no new frames
        assert _wait(lambda: grabber.grab_count >= 5)
        assert len(frames) == 1
        # poke forces an emit even though nothing changed
        grabber.poke()
        assert _wait(lambda: len(frames) == 2)
        # a change in the desktop is emitted
        fake_mss.colour = 200
        assert _wait(lambda: len(frames) == 3)
        assert frames[2].luma == pytest.approx(200 / 255, abs=1e-3)
        # pause stops grabbing
        grabber.pause()
        time.sleep(0.05)
        n = grabber.grab_count
        time.sleep(0.1)
        assert grabber.grab_count == n and grabber.paused
        # resume re-emits (forced) with the DPR stamped from the geometry cache
        geometry.update(Rect(50, 60, 100, 50), 2.0)
        grabber.resume()
        assert _wait(lambda: len(frames) == 4)
        assert frames[3].image.devicePixelRatio() == 2.0 and frames[3].origin == (18, 28)
        assert frames[3].origin_logical == (-16.0, -16.0)
    finally:
        grabber.stop()
        grabber.join(2.0)
    assert not grabber.is_alive() and fake_mss.created == 1 and fake_mss.closed == 1


def test_provider_serves_latest_image() -> None:
    from PySide6.QtCore import QSize

    provider = BD.BackdropProvider()
    img = QImage(7, 5, QImage.Format.Format_RGB32)
    provider.set_image(img)
    size = QSize()
    out = provider.requestImage("42", size, QSize())
    assert (out.width(), out.height()) == (7, 5) and (size.width(), size.height()) == (7, 5)
    assert provider.request_count == 1 and provider.image().width() == 7
