"""The desktop pixels *behind* the control window, for the glass shader.

``BackdropGrabber`` is a plain ``threading.Thread`` that owns an ``mss.MSS`` instance
(created in the thread: Win32 device contexts are thread-bound) and loops at 15 Hz.  Each
pass grabs the window's physical rectangle expanded by ``MARGIN`` physical pixels (clamped to
the virtual screen), compares a 1/8 strided downsample of the raw BGRA buffer with the previous
one and, only when something changed, builds a ``QImage`` (``Format_RGB32`` - D3D11 uploads
BGRA8 without conversion, no ``cvtColor``) *in the grabber thread*, copies it so it owns its
memory, stamps the window's device pixel ratio and hands it to the ``on_frame`` callback.
``poke()`` forces an immediate grab (move / resize / expose / screen change), ``pause()`` stops
grabbing while the window is hidden, minimised or glass is not allowed, ``stop()`` ends the
thread.

The GUI thread stores the frame in ``BackdropProvider`` (``image://backdrop/<serial>``) and
bumps ``bridge.backdropSerial`` so QML rebinds ``Image.source``.  The frame also carries the
mean luma over the **slab rect only** (the transparent shadow margin never carries ink);
``LumaSmoother`` (EMA, tau 250 ms) and ``InkPolarity`` (flip to light ink over dark material
below .38, to dark ink over light material above .62) live here too so they can be unit
tested without a window.

Never use dxcam here: it is a per-process singleton (one Desktop Duplication per output) and a
second consumer breaks the pipeline's capture (verified, ``reports/behind-capture.md``).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider

from ...core.types import Rect
from . import profile

__all__ = [
    "GRAB_HZ",
    "MARGIN",
    "POLARITY_HIGH",
    "POLARITY_LOW",
    "SLAB_INSET",
    "BackdropFrame",
    "BackdropGrabber",
    "BackdropProvider",
    "InkPolarity",
    "LumaSmoother",
    "WindowGeometry",
    "clamp_rect",
    "downsample",
    "mean_luma",
    "slab_rect",
]

log = logging.getLogger(__name__)

MARGIN = 32  # physical px grabbed around the window for the rim refraction
GRAB_HZ = 15.0
BLOCK = 8  # stride of the 1/8 downsample used for change detection and luma
# Slab inset inside the window in logical px: left, top, right, bottom (GLASS_DESIGN section 2.1).
SLAB_INSET = (36, 28, 36, 52)
LUMA_TAU_S = 0.25
POLARITY_LOW, POLARITY_HIGH = 0.38, 0.62
# BT.601 luma weights for the B, G, R, (A) channel order of a BGRA buffer.
_LUMA_BGR = np.array([0.114, 0.587, 0.299, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class BackdropFrame:
    """One grabbed backdrop, ready for the GUI thread."""

    image: QImage  # Format_RGB32, owns its memory, devicePixelRatio set
    origin: Tuple[int, int]  # physical virtual-screen position of the grab
    window_rect: Rect  # physical window rect the grab was computed from
    dpr: float
    luma: float  # raw mean luma over the slab rect, 0..1
    timestamp: float  # time.perf_counter() at grab time

    @property
    def origin_logical(self) -> Tuple[float, float]:
        """Grab origin relative to the window, in logical px (normally ``(-32/dpr, -32/dpr)``)."""
        return ((self.origin[0] - self.window_rect.x) / self.dpr, (self.origin[1] - self.window_rect.y) / self.dpr)


class WindowGeometry:
    """Thread-safe cache of the window's physical rect and DPR, written by the GUI thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rect: Optional[Rect] = None
        self._dpr = 1.0

    def update(self, rect: Rect, dpr: float) -> None:
        with self._lock:
            self._rect = rect
            self._dpr = float(dpr) if dpr and dpr > 0 else 1.0

    def get(self) -> Tuple[Optional[Rect], float]:
        with self._lock:
            return self._rect, self._dpr


def clamp_rect(rect: Rect, bounds: Rect) -> Rect:
    """Intersection of ``rect`` with ``bounds`` (zero-sized when disjoint)."""
    x1, y1 = max(rect.x, bounds.x), max(rect.y, bounds.y)
    x2, y2 = min(rect.x2, bounds.x2), min(rect.y2, bounds.y2)
    return Rect(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def slab_rect(window_rect: Rect, dpr: float) -> Rect:
    """The visible slab (window minus the shadow margin) in absolute physical px."""
    left, top, right, bottom = (int(round(v * dpr)) for v in SLAB_INSET)
    return Rect(window_rect.x + left, window_rect.y + top,
                max(1, window_rect.w - left - right), max(1, window_rect.h - top - bottom))


def downsample(bgra: np.ndarray, step: int = BLOCK) -> np.ndarray:
    """A contiguous 1/``step``-scale strided sample of the BGRA buffer, shape ``(h/step, w/step, 4)``.

    This is what the change test compares (~0.05 ms for a 900x700 grab against ~2.6 ms for a
    true area downsample; the grab itself is ~4 ms).  Details narrower than ``step`` device px
    can slip through unnoticed, which the frosted interior hides anyway.
    """
    return np.ascontiguousarray(bgra[::step, ::step])


def mean_luma(sample: np.ndarray, region: Optional[Rect] = None, step: int = BLOCK) -> float:
    """Mean BT.601 luma (0..1) of a downsampled buffer, restricted to ``region`` (physical px
    relative to the grab origin) when given; falls back to the whole sample if the crop is empty."""
    sel = sample
    if region is not None:
        y0, y1 = region.y // step, -(-region.y2 // step)
        x0, x1 = region.x // step, -(-region.x2 // step)
        crop = sample[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if crop.size:
            sel = crop
    per_px = sel.reshape(-1, 4).astype(np.float64) @ _LUMA_BGR
    return float(per_px.mean() / 255.0)


class LumaSmoother:
    """Exponential moving average with a time constant (default 250 ms)."""

    def __init__(self, tau_s: float = LUMA_TAU_S) -> None:
        self.tau_s = float(tau_s)
        self.value: Optional[float] = None
        self._t: Optional[float] = None

    def update(self, sample: float, now: Optional[float] = None) -> float:
        now = time.perf_counter() if now is None else now
        if self.value is None or self._t is None:
            self.value = float(sample)
        else:
            dt = max(0.0, now - self._t)
            alpha = 1.0 - math.exp(-dt / self.tau_s) if self.tau_s > 0 else 1.0
            self.value += alpha * (float(sample) - self.value)
        self._t = now
        return self.value

    def reset(self) -> None:
        self.value = None
        self._t = None


class InkPolarity:
    """Hysteresis state machine for the material / ink polarity.

    ``0`` = dark material + light ink, ``1`` = light material + dark ink.  Starts from the
    system polarity (``0`` in dark mode, ``1`` in light mode), flips to ``1`` when the smoothed
    luma exceeds ``POLARITY_HIGH`` and back to ``0`` below ``POLARITY_LOW``; the wide band keeps a
    half-dark / half-light desktop on the system polarity.
    """

    def __init__(self, dark_mode: bool) -> None:
        self.value = 0 if dark_mode else 1

    @staticmethod
    def initial_for(dark_mode: bool) -> int:
        return 0 if dark_mode else 1

    def reset(self, dark_mode: bool) -> bool:
        """Return to the system polarity; True when the value changed."""
        new = self.initial_for(dark_mode)
        changed = new != self.value
        self.value = new
        return changed

    def update(self, luma: float) -> bool:
        """Feed a (smoothed) luma; True when the polarity flipped."""
        if self.value == 0 and luma > POLARITY_HIGH:
            self.value = 1
            return True
        if self.value == 1 and luma < POLARITY_LOW:
            self.value = 0
            return True
        return False


class BackdropGrabber(threading.Thread):
    """15 Hz mss grabber of the desktop around the window; see the module docstring.

    ``on_frame`` is called from the grabber thread (hand it a Qt signal's ``emit`` for a queued
    hop to the GUI thread).  ``geometry`` is the GUI thread's :class:`WindowGeometry` cache.
    """

    def __init__(
        self,
        geometry: WindowGeometry,
        on_frame: Callable[[BackdropFrame], None],
        *,
        hz: float = GRAB_HZ,
        margin: int = MARGIN,
        name: str = "backdrop-grab",
    ) -> None:
        super().__init__(daemon=True, name=name)
        self._geometry = geometry
        self._on_frame = on_frame
        self._period = 1.0 / float(hz)
        self._margin = int(margin)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # start paused; the window resumes it on show
        self._force = True
        self._last_sample: Optional[np.ndarray] = None
        self.grab_count = 0
        self.emit_count = 0
        self.last_grab_ms = 0.0

    # -- control (any thread)
    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def poke(self) -> None:
        """Grab immediately and emit even if the pixels did not change."""
        self._force = True
        self._wake.set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        if self._paused.is_set():
            self._paused.clear()
            self._force = True
            self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    # -- thread body
    def run(self) -> None:  # noqa: D401 - Thread API
        import mss  # noqa: WPS433 - the instance must be born in this thread

        sct = None
        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    self._wake.wait(0.5)
                    self._wake.clear()
                    continue
                t0 = time.perf_counter()
                try:
                    if sct is None:
                        sct = mss.MSS()
                    self._grab_once(sct, t0)
                except Exception:  # pragma: no cover - a failing DC is recreated next pass
                    log.warning("backdrop grab failed; re-creating the screen handle", exc_info=True)
                    if sct is not None:
                        try:
                            sct.close()
                        finally:
                            sct = None
                self.last_grab_ms = (time.perf_counter() - t0) * 1000.0
                remaining = self._period - (time.perf_counter() - t0)
                if remaining > 0:
                    self._wake.wait(remaining)
                self._wake.clear()
        finally:
            if sct is not None:
                try:
                    sct.close()
                except Exception:  # pragma: no cover - best effort
                    log.debug("mss close failed", exc_info=True)

    def _grab_once(self, sct, now: float) -> None:
        rect, dpr = self._geometry.get()
        if rect is None or rect.w <= 0 or rect.h <= 0:
            return
        virt = sct.monitors[0]
        bounds = Rect(int(virt["left"]), int(virt["top"]), int(virt["width"]), int(virt["height"]))
        m = self._margin
        region = clamp_rect(Rect(rect.x - m, rect.y - m, rect.w + 2 * m, rect.h + 2 * m), bounds)
        if region.w <= 0 or region.h <= 0:
            return
        shot = sct.grab({"left": region.x, "top": region.y, "width": region.w, "height": region.h})
        self.grab_count += 1
        emitted = self._emit_if_changed(shot, region, rect, dpr, now)
        profile.profiler.mark("grab", x=rect.x, y=rect.y, emitted=emitted, ms=(time.perf_counter() - now) * 1000.0)

    def _emit_if_changed(self, shot, region: Rect, rect: Rect, dpr: float, now: float) -> bool:
        """Change test + QImage build + emit; True when a frame was handed to ``on_frame``."""
        w, h = int(shot.width), int(shot.height)
        data = shot.bgra  # one copy out of mss, shared by the change test and the QImage
        bgra = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
        sample = downsample(bgra)
        force, self._force = self._force, False
        if not force and self._last_sample is not None and self._last_sample.shape == sample.shape \
                and np.array_equal(sample, self._last_sample):
            return False
        self._last_sample = sample
        slab = slab_rect(rect, dpr)
        luma = mean_luma(sample, Rect(slab.x - region.x, slab.y - region.y, slab.w, slab.h))
        # QImage construction is reentrant; copy() so the image owns its pixels once data dies.
        image = QImage(data, w, h, w * 4, QImage.Format.Format_RGB32).copy()
        image.setDevicePixelRatio(dpr)
        self.emit_count += 1
        self._on_frame(BackdropFrame(image, (region.x, region.y), rect, dpr, luma, now))
        return True


class BackdropProvider(QQuickImageProvider):
    """Serves the latest backdrop as ``image://backdrop/<serial>`` (the id is a cache-buster)."""

    def __init__(self) -> None:
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._lock = threading.Lock()
        self._image = QImage(1, 1, QImage.Format.Format_RGB32)
        self._image.fill(0xFF808080)
        self.request_count = 0

    def set_image(self, image: QImage) -> None:
        with self._lock:
            self._image = image

    def image(self) -> QImage:
        with self._lock:
            return self._image

    def requestImage(self, id: str, size: QSize, requestedSize: QSize) -> QImage:  # noqa: N802,A002 - Qt API
        with self._lock:
            img = self._image
            self.request_count += 1
        if size is not None:
            size.setWidth(img.width())
            size.setHeight(img.height())
        return img
