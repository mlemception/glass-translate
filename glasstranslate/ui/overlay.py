"""The translucent, click-through "glass" window that shows translations.

The overlay is a frameless always-on-top tool window with a per-pixel alpha
background.  On Windows it is made click-through with ``WS_EX_TRANSPARENT``
and hidden from screen capture with ``SetWindowDisplayAffinity`` so the
pipeline can grab the pixels *under* it.  In *grab mode* click-through is
switched off and the user can drag the glass around (anywhere) or resize it
(bottom-right handle); ``Escape`` leaves grab mode.

Coordinate conventions
----------------------
Segments arrive in *frame pixels*: physical pixels relative to the top-left of
the glass, which is exactly what :meth:`GlassOverlay.physical_rect` reports to
the pipeline.  Painting happens in Qt logical pixels, so every frame
coordinate is divided by ``devicePixelRatioF()``.

Rotation sign: :func:`glasstranslate.render.style.quad_angle_deg` is
counter-clockwise-positive on screen, while ``QPainter.rotate`` is clockwise
for positive angles in Qt's y-down coordinates, so the painter rotates by
``-angle_deg``.
"""
from __future__ import annotations

import ctypes
import threading
import unicodedata
import logging
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal, Slot, QThread
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
    QResizeEvent,
    QMoveEvent,
    QShowEvent,
)
from PySide6.QtWidgets import QWidget

from ..core.types import Rect, TranslatedSegment
from ..render.fit import FitResult, Measure, fit_text
from ..render.style import quad_text_height, quad_text_width

__all__ = ["GlassOverlay", "qt_measurer"]

log = logging.getLogger(__name__)

# Win32 constants (see winuser.h).
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000
_WDA_EXCLUDEFROMCAPTURE = 0x00000011

_HANDLE_PX = 16  # logical size of the resize handle square in grab mode
_BORDER_PX = 2
_ACCENT = QColor(0, 170, 255)
_GRAB_MIN_ALPHA = 0.18  # make the glass visible while the user drags it
_MIN_W, _MIN_H = 120, 60
_LINE_GAP = 0.1  # extra spacing between stacked characters, fraction of size
_MIN_FONT_PX = 6.0
_VERTICAL_MAX_LINES = 6  # a tall column translated to Latin may need many short lines


def stacks_vertically(text: str) -> bool:
    """True when ``text`` should be drawn one glyph per row (CJK-style);
    scripts written with spaces between words read badly stacked and are
    laid out horizontally instead."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    cjk = sum(1 for c in letters if unicodedata.east_asian_width(c) in ("W", "F"))
    return cjk >= len(letters) / 2


if sys.platform == "win32":
    import ctypes.wintypes as _wt


_USER32: Optional[ctypes.WinDLL] = None


def _user32() -> Optional[ctypes.WinDLL]:
    """user32 loaded with ``use_last_error`` so ``ctypes.get_last_error()``
    reports the real Win32 error after a failed call."""
    global _USER32
    if sys.platform != "win32":
        return None
    if _USER32 is None:
        _USER32 = ctypes.WinDLL("user32", use_last_error=True)
    return _USER32


def _hwnd(widget: QWidget) -> "_wt.HWND":
    return _wt.HWND(int(widget.winId()))


def qt_measurer(family: str) -> Measure:
    """A :data:`~glasstranslate.render.fit.Measure` for ``family`` based on
    ``QFontMetricsF``.  ``size`` is the font pixel size; results are logical
    pixels.  Fonts and metrics are cached per rounded size."""
    cache: Dict[int, QFontMetricsF] = {}

    def metrics(size: float) -> QFontMetricsF:
        px = max(1, int(round(size)))
        fm = cache.get(px)
        if fm is None:
            font = QFont(family)
            font.setPixelSize(px)
            fm = QFontMetricsF(font)
            cache[px] = fm
        return fm

    def measure(text: str, size: float) -> Tuple[float, float]:
        fm = metrics(size)
        return fm.horizontalAdvance(text), fm.height()

    return measure


@dataclass
class _Drag:
    """State of an in-progress move/resize drag in grab mode."""

    resizing: bool
    origin_global: QPoint
    start_geometry: QRect


class GlassOverlay(QWidget):
    """Translucent always-on-top window painting translated segments."""

    geometry_changed = Signal(QRect)  # logical geometry after a move/resize
    grab_mode_changed = Signal(bool)
    _segments_changed = Signal(object)  # list[TranslatedSegment], any thread -> GUI thread

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._rect_lock = threading.Lock()
        self._cached_rect = Rect(0, 0, 0, 0)
        self.setMinimumSize(_MIN_W, _MIN_H)
        self.setMouseTracking(True)
        self.setWindowTitle("GlassTranslate glass")

        self._segments: List[TranslatedSegment] = []
        self._opacity = 0.1
        self._font_family = "Segoe UI"
        self._hide_original = True
        self._click_through = True
        self._grab_mode = False
        self._drag: Optional[_Drag] = None
        self._native_ready = False
        self._measure: Measure = qt_measurer(self._font_family)
        self._segments_changed.connect(self._apply_segments, Qt.ConnectionType.QueuedConnection)

    # ------------------------------------------------------------ properties
    @property
    def grab_mode(self) -> bool:
        return self._grab_mode

    @property
    def click_through(self) -> bool:
        return self._click_through

    @property
    def segments(self) -> List[TranslatedSegment]:
        return list(self._segments)

    # ------------------------------------------------------------------ api
    def set_segments(self, segments: Sequence[TranslatedSegment]) -> None:
        """Replace the displayed segments.  Safe to call from any thread."""
        self._segments_changed.emit(list(segments))

    def set_background_opacity(self, opacity: float) -> None:
        self._opacity = min(1.0, max(0.0, float(opacity)))
        self.update()

    def set_font_family(self, family: str) -> None:
        if family and family != self._font_family:
            self._font_family = family
            self._measure = qt_measurer(family)
            self.update()

    def set_hide_original(self, hide: bool) -> None:
        self._hide_original = bool(hide)
        self.update()

    def set_click_through(self, enabled: bool) -> None:
        """Let mouse input pass through the glass (``True``) or catch it."""
        self._click_through = bool(enabled)
        if sys.platform == "win32":
            self._apply_win32_styles()
        else:  # pragma: no cover - X11/macOS path
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, self._click_through)
            if self.isVisible():
                self.show()

    def set_grab_mode(self, enabled: bool) -> None:
        """Enter/leave grab mode: interactive move/resize with a visible frame."""
        enabled = bool(enabled)
        if enabled == self._grab_mode:
            return
        self._grab_mode = enabled
        self._drag = None
        self.set_click_through(not enabled)
        if enabled:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            self.activateWindow()
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        else:
            self.unsetCursor()
        self.update()
        self.grab_mode_changed.emit(enabled)

    def toggle_grab_mode(self) -> None:
        self.set_grab_mode(not self._grab_mode)

    def physical_rect(self) -> Rect:
        """The glass rectangle in virtual-screen *physical* pixels.

        Safe to call from any thread: the pipeline worker polls this every
        tick, so the value is computed on the GUI thread whenever the window
        is shown, moved or resized and merely read here under a lock.
        """
        if QThread.currentThread() is not self.thread():
            with self._rect_lock:
                return self._cached_rect
        return self._refresh_physical_rect()

    def _refresh_physical_rect(self) -> Rect:
        """GUI thread only: recompute and cache the physical rectangle."""
        rect_out: Optional[Rect] = None
        user32 = _user32()
        if user32 is not None and self._native_ready:
            rect = _wt.RECT()
            if user32.GetWindowRect(_hwnd(self), ctypes.byref(rect)):
                rect_out = Rect(
                    int(rect.left), int(rect.top), int(rect.right - rect.left), int(rect.bottom - rect.top)
                )
        if rect_out is None:
            rect_out = self._qt_physical_rect()
        with self._rect_lock:
            self._cached_rect = rect_out
        return rect_out

    # -------------------------------------------------------------- Qt events
    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._native_ready = True
        if sys.platform == "win32":
            self._apply_win32_styles()
            self._exclude_from_capture()
        else:  # pragma: no cover
            self.set_click_through(self._click_through)
        self._refresh_physical_rect()

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)
        self._refresh_physical_rect()
        self.geometry_changed.emit(self.geometry())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_physical_rect()
        self.geometry_changed.emit(self.geometry())

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._grab_mode:
            self.set_grab_mode(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._grab_mode or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self._drag = _Drag(
            resizing=self._handle_rect().contains(event.position().toPoint()),
            origin_global=event.globalPosition().toPoint(),
            start_geometry=self.geometry(),
        )
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._grab_mode:
            return super().mouseMoveEvent(event)
        if self._drag is None:
            over_handle = self._handle_rect().contains(event.position().toPoint())
            self.setCursor(Qt.CursorShape.SizeFDiagCursor if over_handle else Qt.CursorShape.SizeAllCursor)
            return
        delta = event.globalPosition().toPoint() - self._drag.origin_global
        g = self._drag.start_geometry
        if self._drag.resizing:
            self.resize(max(_MIN_W, g.width() + delta.x()), max(_MIN_H, g.height() + delta.y()))
        else:
            self.move(g.topLeft() + delta)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        alpha = max(self._opacity, _GRAB_MIN_ALPHA) if self._grab_mode else self._opacity
        painter.fillRect(self.rect(), QColor(0, 0, 0, int(round(255 * alpha))))

        dpr = self.devicePixelRatioF() or 1.0
        for seg in self._segments:
            try:
                self._paint_segment(painter, seg, dpr)
            except Exception:  # one bad segment must not blank the glass
                log.exception("failed to paint segment %r", seg.source_text)

        if self._grab_mode:
            self._paint_grab_frame(painter)
        painter.end()

    # ---------------------------------------------------------- painting
    def _paint_segment(self, painter: QPainter, seg: TranslatedSegment, dpr: float) -> None:
        quad = np.asarray(seg.quad, dtype=np.float64).reshape(4, 2) / dpr
        style = seg.style
        if self._hide_original:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*style.bg))
            painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in quad]))
        text = seg.translation.strip()
        if not text:
            return
        cx, cy = (float(v) for v in quad.mean(axis=0))
        box_w = max(1.0, quad_text_width(quad))
        box_h = max(1.0, quad_text_height(quad))

        painter.save()
        painter.translate(cx, cy)
        painter.rotate(-style.angle_deg)
        painter.setPen(QColor(*style.fg))
        if style.vertical and stacks_vertically(text):
            self._draw_vertical(painter, text, box_w, box_h)
        elif style.vertical:
            # Vertical CJK column translated into a horizontal script (the
            # manga case): lay the Latin text out horizontally, wrapped to the
            # column's height as the line width and its width as the height.
            fit = fit_text(text, box_h, box_w, self._measure, max_lines=_VERTICAL_MAX_LINES)
            painter.rotate(90)  # CW on screen: reads top-to-bottom like the column it replaces
            self._draw_lines(painter, fit, box_h, box_w)
        else:
            fit = fit_text(text, box_w, box_h, self._measure)
            self._draw_lines(painter, fit, box_w, box_h)
        painter.restore()

    def _font(self, size: float) -> QFont:
        font = QFont(self._font_family)
        font.setPixelSize(max(1, int(round(size))))
        return font

    def _draw_lines(self, painter: QPainter, fit: FitResult, box_w: float, box_h: float) -> None:
        font = self._font(fit.size)
        painter.setFont(font)
        line_h = QFontMetricsF(font).height()
        total = line_h * len(fit.lines)
        y = -total / 2.0
        flags = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        for line in fit.lines:
            painter.drawText(QRectF(-box_w / 2.0, y, box_w, line_h), flags, line)
            y += line_h

    def _draw_vertical(self, painter: QPainter, text: str, box_w: float, box_h: float) -> None:
        chars = [c for c in text if not c.isspace()] or [" "]
        n = len(chars)

        def fits(size: float) -> bool:
            widest = max(self._measure(c, size)[0] for c in chars)
            glyph_h = self._measure("M", size)[1]
            return widest <= box_w and glyph_h * n + size * _LINE_GAP * (n - 1) <= box_h

        size = max(_MIN_FONT_PX, min(box_w * 0.9, box_h / n))
        while size > _MIN_FONT_PX and not fits(size):
            size = max(size * 0.9, _MIN_FONT_PX)
        font = self._font(size)
        painter.setFont(font)
        glyph_h = QFontMetricsF(font).height()
        step = glyph_h + size * _LINE_GAP
        y = -(step * n - size * _LINE_GAP) / 2.0
        flags = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        for c in chars:
            painter.drawText(QRectF(-box_w / 2.0, y, box_w, glyph_h), flags, c)
            y += step

    def _paint_grab_frame(self, painter: QPainter) -> None:
        pen = QPen(_ACCENT, _BORDER_PX)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inset = _BORDER_PX / 2.0
        painter.drawRect(QRectF(self.rect()).adjusted(inset, inset, -inset, -inset))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_ACCENT)
        painter.drawRect(self._handle_rect())
        painter.setPen(QColor(255, 255, 255, 220))
        painter.setFont(QFont(self._font_family, 10))
        painter.drawText(
            self.rect().adjusted(8, 6, -8, -6),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
            "Grab mode: drag to move, corner to resize, Esc to finish",
        )

    def _handle_rect(self) -> QRect:
        return QRect(self.width() - _HANDLE_PX, self.height() - _HANDLE_PX, _HANDLE_PX, _HANDLE_PX)

    # -------------------------------------------------------------- Win32
    def _apply_win32_styles(self) -> None:
        user32 = _user32()
        if user32 is None or not self._native_ready:
            return
        hwnd = _hwnd(self)
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.restype = ctypes.c_long
        ex = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE) & 0xFFFFFFFF
        ex |= _WS_EX_LAYERED | _WS_EX_TOOLWINDOW
        if self._click_through:
            ex |= _WS_EX_TRANSPARENT | _WS_EX_NOACTIVATE
        else:
            ex &= ~(_WS_EX_TRANSPARENT | _WS_EX_NOACTIVATE)
        user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, ctypes.c_long(ctypes.c_int32(ex).value))

    def _exclude_from_capture(self) -> None:
        user32 = _user32()
        if user32 is None or not self._native_ready:
            return
        if not user32.SetWindowDisplayAffinity(_hwnd(self), _WDA_EXCLUDEFROMCAPTURE):
            log.warning("SetWindowDisplayAffinity failed (error %d); glass may appear in captures",
                        ctypes.get_last_error())

    def _qt_physical_rect(self) -> Rect:
        """Portable approximation used when ``GetWindowRect`` is unavailable:
        scale the frame geometry by the screen's device pixel ratio."""
        g = self.frameGeometry()
        screen = self.screen()
        dpr = screen.devicePixelRatio() if screen is not None else 1.0
        return Rect(
            int(round(g.x() * dpr)),
            int(round(g.y() * dpr)),
            int(round(g.width() * dpr)),
            int(round(g.height() * dpr)),
        )

    # ------------------------------------------------------------- slots
    @Slot(object)
    def _apply_segments(self, segments: object) -> None:
        self._segments = list(segments) if isinstance(segments, (list, tuple)) else []
        self.update()
