"""Win32 helpers for the frameless, per-pixel-translucent control window.

Everything here was verified by the probes in ``demo/output/probes/behind-capture``
and ``demo/output/probes/review/qtquick`` (see the reports next to them):

* :func:`exclude_from_capture` - ``SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)``
  makes every same-process screen grabber (mss, ``QScreen.grabWindow``, dxcam)
  see *through* the window, which is what the backdrop grabber and the
  pipeline's capture need.  It is a per-HWND flag that survives hide/show,
  ``setGeometry`` and flag changes; re-applying it is harmless.
* :func:`physical_rect` - the window rectangle in **physical** virtual-screen
  pixels from ``GetWindowRect`` (what mss expects), with a DPR-scaled
  ``frameGeometry()`` fallback that agrees exactly at DPR 1.0 and 2.0.
* :func:`keyboard_move` / :func:`keyboard_size` - post ``WM_SYSCOMMAND`` with
  ``SC_MOVE`` / ``SC_SIZE`` so ``DefWindowProc`` runs its keyboard move / size
  loop (arrow keys, Enter, Esc) on a window that has no native title bar.

On other platforms every function degrades to a no-op / Qt fallback.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from typing import Optional

from PySide6.QtGui import QWindow

from ...core.types import Rect

__all__ = [
    "SC_MOVE",
    "SC_SIZE",
    "WDA_EXCLUDEFROMCAPTURE",
    "WM_SYSCOMMAND",
    "dpi_awareness",
    "exclude_from_capture",
    "hwnd_of",
    "keyboard_move",
    "keyboard_size",
    "physical_rect",
]

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

WDA_EXCLUDEFROMCAPTURE = 0x11
WM_SYSCOMMAND = 0x0112
SC_SIZE = 0xF000
SC_MOVE = 0xF010

if IS_WINDOWS:
    import ctypes.wintypes as wt

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.SetWindowDisplayAffinity.argtypes = [wt.HWND, wt.DWORD]
    _user32.SetWindowDisplayAffinity.restype = wt.BOOL
    _user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
    _user32.GetWindowRect.restype = wt.BOOL
    _user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
    _user32.PostMessageW.restype = wt.BOOL
    _user32.IsWindow.argtypes = [wt.HWND]
    _user32.IsWindow.restype = wt.BOOL
    try:
        _user32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        _user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        _user32.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
        _HAS_DPI_API = True
    except AttributeError:  # pragma: no cover - pre-1607 Windows
        _HAS_DPI_API = False


def hwnd_of(window: QWindow) -> int:
    """Native handle of ``window``; forces native creation (``winId``)."""
    return int(window.winId())


def exclude_from_capture(window: QWindow) -> bool:
    """Hide ``window`` from screen capture so grabs show what is *behind* it.

    Returns True on success.  Safe to call before ``show()`` (``winId()``
    creates the native window) and cheap to repeat on every ``showEvent``.
    """
    if not IS_WINDOWS:
        return False
    hwnd = wt.HWND(hwnd_of(window))
    ok = bool(_user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))
    if not ok:
        log.warning("SetWindowDisplayAffinity failed: error %d", ctypes.get_last_error())
    return ok


def _qt_physical_rect(window: QWindow) -> Rect:
    """``frameGeometry()`` mapped to physical pixels through the screen origin."""
    dpr = float(window.devicePixelRatio() or 1.0)
    g = window.frameGeometry()
    screen = window.screen()
    if screen is not None:
        sg = screen.geometry()
        x = sg.x() + (g.x() - sg.x()) * dpr
        y = sg.y() + (g.y() - sg.y()) * dpr
    else:
        x, y = g.x() * dpr, g.y() * dpr
    return Rect(int(round(x)), int(round(y)), int(round(g.width() * dpr)), int(round(g.height() * dpr)))


def physical_rect(window: QWindow) -> Rect:
    """Window rectangle in physical virtual-screen pixels.

    ``GetWindowRect`` is authoritative on Windows (it is already up to date
    inside ``moveEvent`` / ``resizeEvent``); elsewhere, or when the native
    window does not exist yet, the DPR-scaled Qt geometry is returned.
    """
    if IS_WINDOWS:
        hwnd = wt.HWND(hwnd_of(window))
        r = wt.RECT()
        if _user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return Rect(int(r.left), int(r.top), int(r.right - r.left), int(r.bottom - r.top))
        log.debug("GetWindowRect failed: error %d", ctypes.get_last_error())
    return _qt_physical_rect(window)


def _post_syscommand(window: QWindow, command: int) -> bool:
    if not IS_WINDOWS:
        return False
    hwnd = wt.HWND(hwnd_of(window))
    if not _user32.IsWindow(hwnd):
        return False
    ok = bool(_user32.PostMessageW(hwnd, WM_SYSCOMMAND, command, 0))
    if not ok:
        log.warning("PostMessageW(WM_SYSCOMMAND, 0x%X) failed: error %d", command, ctypes.get_last_error())
    return ok


def keyboard_move(window: QWindow) -> bool:
    """Start the system keyboard move loop (arrow keys, Enter / Esc)."""
    return _post_syscommand(window, SC_MOVE)


def keyboard_size(window: QWindow) -> bool:
    """Start the system keyboard size loop (arrow keys pick an edge)."""
    return _post_syscommand(window, SC_SIZE)


def dpi_awareness() -> Optional[int]:
    """Thread DPI awareness (0 unaware, 1 system, 2 per-monitor); None off Windows."""
    if not IS_WINDOWS or not _HAS_DPI_API:
        return None
    ctx = _user32.GetThreadDpiAwarenessContext()
    return int(_user32.GetAwarenessFromDpiAwarenessContext(ctx))
