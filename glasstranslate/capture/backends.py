"""Screen-capture backends.

Two implementations of :class:`~glasstranslate.core.interfaces.ScreenCapture`:

* :class:`DXCamCapture` - Windows Desktop Duplication through ``dxcam``.  Very
  fast (sub-millisecond grabs once warmed up) and tells us when nothing on the
  screen changed.  Each instance owns one lazily created ``dxcam`` camera bound
  to the monitor that contains the requested region; the camera is re-created
  if the region moves to another monitor.
* :class:`MSSCapture` - portable GDI/X11 grabs through ``mss``.  Slower
  (a few ms) but works everywhere and can grab across monitor boundaries.

Both accept regions in *virtual-screen physical pixels* (the coordinate system
used by ``Rect`` throughout the pipeline), clamp them to what is actually on
screen and return contiguous ``HxWx3`` uint8 BGR frames.

The overlay window is excluded from capture on Windows via
``SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`` (done by the UI layer);
neither backend needs to know about it.  On other platforms there is no
equivalent, so the overlay may appear in its own capture - documented limitation.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import numpy as np

from glasstranslate.core.interfaces import ScreenCapture
from glasstranslate.core.types import Frame, Rect

__all__ = ["DXCamCapture", "MSSCapture"]

logger = logging.getLogger(__name__)

# How often to poll Desktop Duplication for a fresh frame when the requested
# region changed and the cached frame no longer applies.
_DXCAM_REGION_CHANGE_RETRIES = 4
_DXCAM_REGION_CHANGE_RETRY_S = 0.004


# --------------------------------------------------------------------------- helpers
def _clamp_to(region: Rect, bounds: Rect) -> Rect:
    """Intersect ``region`` with ``bounds``; ``w``/``h`` are 0 when disjoint."""
    x = max(region.x, bounds.x)
    y = max(region.y, bounds.y)
    x2 = min(region.x2, bounds.x2)
    y2 = min(region.y2, bounds.y2)
    return Rect(x, y, max(0, x2 - x), max(0, y2 - y))


def _overlap_area(a: Rect, b: Rect) -> int:
    c = _clamp_to(a, b)
    return c.w * c.h


@dataclass(frozen=True)
class _Monitor:
    """A dxcam output and where it sits on the virtual screen."""

    device_idx: int
    output_idx: int
    rect: Rect  # virtual-screen physical pixels


def _enumerate_dxcam_outputs(dxcam_module: Any) -> List[_Monitor]:
    """Map every dxcam output to its virtual-screen rectangle.

    dxcam keeps the DXGI output descriptors (which carry the desktop
    coordinates) on its module-level factory.  If that private attribute is
    ever removed we fall back to ``EnumDisplayMonitors`` and assume the same
    ordering as dxcam's output list on adapter 0.
    """
    factory = getattr(dxcam_module, "__factory", None)
    outputs = getattr(factory, "outputs", None)
    monitors: List[_Monitor] = []
    if outputs:
        for device_idx, device_outputs in enumerate(outputs):
            for output_idx, out in enumerate(device_outputs):
                dc = out.desc.DesktopCoordinates
                rect = Rect(int(dc.left), int(dc.top), int(dc.right - dc.left), int(dc.bottom - dc.top))
                if rect.w > 0 and rect.h > 0:
                    monitors.append(_Monitor(device_idx, output_idx, rect))
        if monitors:
            return monitors
    logger.warning("dxcam factory outputs unavailable; falling back to EnumDisplayMonitors ordering")
    return [_Monitor(0, i, r) for i, r in enumerate(_enum_display_monitors())]


def _enum_display_monitors() -> List[Rect]:
    """Virtual-screen rectangles of all monitors via Win32 ``EnumDisplayMonitors``."""
    from ctypes import wintypes

    rects: List[Rect] = []
    proc_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
    )

    def _cb(_hmon: int, _hdc: int, prect: Any, _lparam: int) -> int:
        r = prect.contents
        rects.append(Rect(int(r.left), int(r.top), int(r.right - r.left), int(r.bottom - r.top)))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(None, None, proc_type(_cb), 0)
    return rects


# --------------------------------------------------------------------------- dxcam
class DXCamCapture(ScreenCapture):
    """Desktop Duplication capture (Windows only).

    ``grab`` semantics:

    * Returns a fresh :class:`Frame` whenever the desktop changed since the
      previous grab.
    * When Desktop Duplication reports *no change*, the previous frame is
      returned again with a new timestamp (the pipeline does its own diffing).
    * Returns ``None`` only when no frame for the requested region exists yet
      (first grab on a completely static desktop, region fully off-screen, or a
      transient duplication failure before any frame was received).
    """

    name = "dxcam"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("DXCamCapture requires Windows (Desktop Duplication API)")
        import dxcam  # noqa: WPS433 - optional, Windows-only dependency

        self._dxcam = dxcam
        self._monitors: List[_Monitor] = _enumerate_dxcam_outputs(dxcam)
        if not self._monitors:
            raise RuntimeError("dxcam found no display outputs")
        self._cam: Any = None
        self._cam_monitor: Optional[_Monitor] = None
        self._last: Optional[Frame] = None
        self._failing = False  # rate-limits error logging

    # ----------------------------------------------------------------- api
    def grab(self, region: Rect) -> Optional[Frame]:
        monitor = self._monitor_for(region)
        if monitor is None:
            return self._stale_or_none()
        clamped = _clamp_to(region, monitor.rect)
        if clamped.w <= 0 or clamped.h <= 0:
            return self._stale_or_none()

        try:
            cam = self._camera_for(monitor)
            local = (
                clamped.x - monitor.rect.x,
                clamped.y - monitor.rect.y,
                clamped.x2 - monitor.rect.x,
                clamped.y2 - monitor.rect.y,
            )
            img = cam.grab(region=local)
            if img is None and not self._last_matches(clamped):
                img = self._retry_grab(cam, local)
        except Exception as exc:  # COMError, ValueError from dxcam, ...
            self._on_failure(exc)
            return self._stale_or_none()

        if self._failing:
            logger.info("dxcam capture recovered")
            self._failing = False

        now = time.perf_counter()
        if img is None:
            if self._last is None or not self._last_matches(clamped):
                return None
            return Frame(self._last.image, self._last.origin_x, self._last.origin_y, now)
        frame = Frame(np.ascontiguousarray(img), clamped.x, clamped.y, now)
        self._last = frame
        return frame

    def close(self) -> None:
        self._release_camera()
        self._last = None

    # ------------------------------------------------------------ internals
    def _monitor_for(self, region: Rect) -> Optional[_Monitor]:
        """The monitor overlapping ``region`` the most, or None if off-screen."""
        best: Optional[_Monitor] = None
        best_area = 0
        for mon in self._monitors:
            area = _overlap_area(region, mon.rect)
            if area > best_area:
                best, best_area = mon, area
        return best

    def _camera_for(self, monitor: _Monitor) -> Any:
        if self._cam is not None and self._cam_monitor == monitor:
            return self._cam
        self._release_camera()
        logger.info(
            "dxcam: opening device %d output %d at %s", monitor.device_idx, monitor.output_idx, monitor.rect
        )
        self._cam = self._dxcam.create(
            device_idx=monitor.device_idx, output_idx=monitor.output_idx, output_color="BGR"
        )
        self._cam_monitor = monitor
        return self._cam

    def _release_camera(self) -> None:
        cam, self._cam, self._cam_monitor = self._cam, None, None
        if cam is not None:
            try:
                cam.release()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("dxcam release failed", exc_info=True)

    @staticmethod
    def _retry_grab(cam: Any, local: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Poll briefly for a new desktop frame after the region changed.

        Desktop Duplication only hands us pixels when *something* on the
        desktop was presented; right after the glass moves the cached frame is
        for the wrong region, so give the compositor a few milliseconds.
        """
        for _ in range(_DXCAM_REGION_CHANGE_RETRIES):
            time.sleep(_DXCAM_REGION_CHANGE_RETRY_S)
            img = cam.grab(region=local)
            if img is not None:
                return img
        return None

    def _last_matches(self, region: Rect) -> bool:
        last = self._last
        return (
            last is not None
            and last.origin_x == region.x
            and last.origin_y == region.y
            and last.width == region.w
            and last.height == region.h
        )

    def _stale_or_none(self) -> Optional[Frame]:
        if self._last is None:
            return None
        return Frame(self._last.image, self._last.origin_x, self._last.origin_y, time.perf_counter())

    def _on_failure(self, exc: Exception) -> None:
        if not self._failing:
            logger.warning("dxcam grab failed (%s: %s); will retry with a fresh camera", type(exc).__name__, exc)
            self._failing = True
        else:
            logger.debug("dxcam grab failed again: %s", exc)
        self._release_camera()


# --------------------------------------------------------------------------- mss
class MSSCapture(ScreenCapture):
    """Portable capture via ``mss``.

    The ``mss.MSS`` instance is created lazily on the first ``grab`` so it is
    born in the thread that uses it (Win32 device contexts are thread-bound).
    Regions are clamped to the virtual screen; grabs spanning several monitors
    are fine.  ``None`` is returned when the clamped region is empty or ``mss``
    raised (the instance is then re-created on the next call).
    """

    name = "mss"

    def __init__(self) -> None:
        self._sct: Any = None
        self._virtual: Optional[Rect] = None
        self._failing = False

    # ----------------------------------------------------------------- api
    def grab(self, region: Rect) -> Optional[Frame]:
        try:
            sct = self._screen()
            assert self._virtual is not None
            clamped = _clamp_to(region, self._virtual)
            if clamped.w <= 0 or clamped.h <= 0:
                return None
            shot = sct.grab({"left": clamped.x, "top": clamped.y, "width": clamped.w, "height": clamped.h})
        except Exception as exc:
            if not self._failing:
                logger.warning("mss grab failed (%s: %s); will re-create the screen handle", type(exc).__name__, exc)
                self._failing = True
            self.close()
            return None

        if self._failing:
            logger.info("mss capture recovered")
            self._failing = False
        bgra = np.frombuffer(shot.bgra, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)  # allocates a fresh contiguous array
        return Frame(bgr, clamped.x, clamped.y, time.perf_counter())

    def close(self) -> None:
        sct, self._sct, self._virtual = self._sct, None, None
        if sct is not None:
            try:
                sct.close()
            except Exception:  # pragma: no cover - best effort cleanup
                logger.debug("mss close failed", exc_info=True)

    # ------------------------------------------------------------ internals
    def _screen(self) -> Any:
        if self._sct is None:
            import mss  # noqa: WPS433 - imported lazily to keep module import cheap

            self._sct = mss.MSS()
            virt = self._sct.monitors[0]  # index 0 is the whole virtual screen
            self._virtual = Rect(int(virt["left"]), int(virt["top"]), int(virt["width"]), int(virt["height"]))
        return self._sct
