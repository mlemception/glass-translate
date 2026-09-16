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

Typeset blocks
--------------
Segments whose ``style.is_block`` is set (manga mode) are drawn the way the
PIL renderer :func:`glasstranslate.render.compose.compose` draws them: the
block's ``clean_patch`` (the frame with the original glyphs erased) is
painted at ``clean_rect``, then the translation is placed with the shared
:func:`glasstranslate.render.compose.typeset_block` (bubble text flowed
through the bubble outline, free text set on the quietest nearby paper by
``render.place``) using a ``QFontMetricsF`` measurer of the same bundled
Anime Ace font.  The whole block is laid out in *frame* pixels under a
``1/dpr`` painter scale, so its placement is identical to the demo output
at any DPI.  Pixel drawing mirrors :func:`glasstranslate.render.compose.draw_typeset`:
first every line's paper-coloured rounded halo (``compose.halo_px`` +
``compose.weight_px`` outward, as a pen of twice that width centred on the
outline), then the glyphs filled in the foreground colour with a thin
foreground pen (``2 x weight_px``) for the extra stroke weight.

Caching: laying a block out (``typeset_block``) and stroking its glyph
outlines are both far more expensive than a repaint budget allows (about 80
ms each for a page of blocks), so per result every block's ``Typeset`` and
its lettering, rendered once into a small ARGB layer at frame resolution,
are kept by segment index; ``paintEvent`` only blits the layers.  Both
caches survive pipeline passes that re-emit the same ``TranslatedSegment``
objects (the frequent "nothing changed" ticks) and are dropped for a
segment whose object changed, or when the case / font setting changes.
The patch cache additionally keys on ``style.clean_patch_serial``, which the
quality renderer bumps whenever it swaps an upgraded ``clean_patch`` into a
live block: that patch is re-converted while the block's typeset and its
lettering layer are kept.
"""
from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
import threading
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QFile, QIODevice, QPoint, QPointF, QRect, QRectF, Qt, Signal, Slot, QThread
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPolygonF,
    QResizeEvent,
    QMoveEvent,
    QShowEvent,
)
from PySide6.QtWidgets import QWidget

from ..core.types import Rect, SegmentStyle, TranslatedSegment
from ..render.compose import (MANGA_FONT_PATH, block_italic, halo_px, needs_cjk_face,
                              typeset_block, weight_px)
from ..render.fit import FitResult, Measure, fit_text
from ..render.style import quad_text_height, quad_text_width
from ..render.typeset import Typeset

__all__ = ["GlassOverlay", "bgr_to_qimage", "load_manga_font", "make_font", "qt_measurer"]

log = logging.getLogger(__name__)

# Win32 constants (see winuser.h).
_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000
_WDA_NONE = 0x00000000
_WDA_EXCLUDEFROMCAPTURE = 0x00000011

_HANDLE_PX = 16  # logical size of the resize handle square in grab mode
_BORDER_PX = 2
_ACCENT = QColor(0, 170, 255)
_GRAB_MIN_ALPHA = 0.18  # make the glass visible while the user drags it
_MIN_W, _MIN_H = 120, 60
_LINE_GAP = 0.1  # extra spacing between stacked characters, fraction of size
_MIN_FONT_PX = 6.0
_VERTICAL_MAX_LINES = 6  # a tall column translated to Latin may need many short lines
# Anime Ace styles bundled next to the PIL renderer's default; registered
# with Qt once per process.
_MANGA_FONT_FILES = ("animeace2_reg.ttf", "animeace2_ital.ttf")
_MANGA_FONT_DIR = os.path.dirname(MANGA_FONT_PATH)
_manga_font_family: Optional[str] = None


def bgr_to_qimage(patch: np.ndarray) -> QImage:
    """Deep-copy an ``HxWx3`` BGR array into a QImage that owns its pixels."""
    rows, cols = patch.shape[:2]
    data = np.ascontiguousarray(patch[:, :, :3], dtype=np.uint8)
    return QImage(data.data, cols, rows, int(data.strides[0]), QImage.Format.Format_BGR888).copy()


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


def _resource_bytes(path: str) -> Optional[bytes]:
    """Contents of a Qt resource (``:/fonts/...``), or None when it is not compiled in."""
    f = QFile(path)
    if not f.open(QIODevice.OpenModeFlag.ReadOnly):
        return None
    try:
        return bytes(f.readAll().data())
    finally:
        f.close()


def load_manga_font() -> Optional[str]:
    """Register the bundled Anime Ace files with Qt (once) and return the
    family name, or None when the files are missing or Qt rejected them.

    The fonts are read from the compiled Qt resources (``:/fonts/``, see
    ``tools/build_resources.py``) so the packaged exe needs no loose files;
    a source checkout without the generated resources falls back to the
    ``.ttf`` files next to the PIL renderer.  Needs a ``QGuiApplication``."""
    global _manga_font_family
    if _manga_font_family is not None:
        return _manga_font_family
    for name in _MANGA_FONT_FILES:
        data = _resource_bytes(f":/fonts/{name}")
        if data is None:
            path = os.path.join(_MANGA_FONT_DIR, name)
            if not os.path.exists(path):
                continue
            with open(path, "rb") as fh:
                data = fh.read()
        if not data:
            continue
        font_id = QFontDatabase.addApplicationFontFromData(data)
        families = QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []
        if families:
            _manga_font_family = str(families[0])
    if _manga_font_family is None:
        log.warning("bundled comic font not available; typeset blocks use the overlay font")
    return _manga_font_family


def make_font(family: str, size: float, *, bold: bool = False, italic: bool = False) -> QFont:
    """A ``family`` font at ``size`` pixels, unhinted so glyph advances scale
    linearly with size (the flow algorithm relies on that)."""
    font = QFont(family)
    font.setPixelSize(max(1, int(round(size))))
    font.setBold(bold)
    font.setItalic(italic)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


def qt_measurer(family: str, *, bold: bool = False, italic: bool = False) -> Measure:
    """A :data:`~glasstranslate.render.fit.Measure` for ``family`` based on
    ``QFontMetricsF``: ``(horizontal advance, ascent + descent)`` like
    :func:`glasstranslate.render.compose.pil_measurer`, so the shared
    typesetter (which derives cap height and leading from the line height)
    lays a block out identically for both renderers.  ``size`` is the font
    pixel size; results are in the same pixel unit.  Fonts and metrics are
    cached per rounded size in this measurer (one per font, used on the GUI
    thread only)."""
    cache: Dict[int, QFontMetricsF] = {}

    def metrics(size: float) -> QFontMetricsF:
        px = max(1, int(round(size)))
        fm = cache.get(px)
        if fm is None:
            fm = QFontMetricsF(make_font(family, px, bold=bold, italic=italic))
            cache[px] = fm
        return fm

    def measure(text: str, size: float) -> Tuple[float, float]:
        fm = metrics(size)
        # FreeType (and so PIL's ``getmetrics``) reports the ascender rounded
        # up and the descender rounded down to whole pixels; do the same so
        # both renderers see one line height and take the same layout
        # decisions (line pitch, cap height and the size search depend on it).
        return fm.horizontalAdvance(text), float(math.ceil(fm.ascent()) + math.ceil(fm.descent()))

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
        self._patches: Dict[int, QImage] = {}  # clean patches of the current blocks, by index
        self._patch_serials: Dict[int, int] = {}  # style.clean_patch_serial each cached patch was made from
        # Cache counters for the smoke report (docs/GLASS_DESIGN.md section 5): how many patches
        # were converted to a QImage and how many of those replaced a cached image of the same
        # block, i.e. a quality-renderer upgrade the overlay actually picked up.
        self._patch_conversions = 0
        self._patch_upgrades = 0
        self._opacity = 0.1
        self._font_family = "Segoe UI"
        self._hide_original = True
        self._uppercase = True
        self._click_through = True
        self._grab_mode = False
        self._capture_excluded = True  # capture mode (F1) clears this; re-applied on every show
        self._drag: Optional[_Drag] = None
        self._native_ready = False
        self._measure: Measure = qt_measurer(self._font_family)
        self._block_family = load_manga_font() or self._font_family
        self._block_measure: Measure = qt_measurer(self._block_family)
        self._block_measure_italic: Measure = qt_measurer(self._block_family, italic=True)
        # Typeset results and rendered lettering layers per segment index;
        # laying a block out means a search over sizes and boxes and stroking
        # its outlines is slow, so both are done once per result, not on
        # every repaint (see the module docstring).
        self._typesets: Dict[int, Typeset] = {}
        self._layers: Dict[int, Tuple[QImage, int, int]] = {}  # (image, frame x, frame y)
        self._segments_changed.connect(self._apply_segments, Qt.ConnectionType.QueuedConnection)

    # ------------------------------------------------------------ properties
    @property
    def grab_mode(self) -> bool:
        return self._grab_mode

    @property
    def click_through(self) -> bool:
        return self._click_through

    @property
    def capture_excluded(self) -> bool:
        """``False`` while capture mode lets screen-capture tools see the glass."""
        return self._capture_excluded

    @property
    def segments(self) -> List[TranslatedSegment]:
        return list(self._segments)

    @property
    def patch_stats(self) -> Dict[str, int]:
        """Clean-patch cache counters: ``conversions`` (BGR patch -> QImage) and ``upgrades``
        (a conversion that replaced the cached image of the same block, so a sidecar result
        reached the glass).  Read by the smoke report; never reset."""
        return {"conversions": self._patch_conversions, "upgrades": self._patch_upgrades}

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
            self._typesets = {}  # blocks with CJK text are set in this font
            self._layers = {}
            self.update()

    def set_hide_original(self, hide: bool) -> None:
        self._hide_original = bool(hide)
        self.update()

    def set_uppercase(self, uppercase: bool) -> None:
        """Letter typeset blocks in capitals (comic convention) or as translated."""
        if bool(uppercase) != self._uppercase:
            self._uppercase = bool(uppercase)
            self._typesets = {}  # the cached layouts were made for the other case
            self._layers = {}
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

    def set_capture_excluded(self, excluded: bool) -> None:
        """Hide the glass from screen capture (default) or make it capturable.

        The desired state is remembered and re-applied on every ``showEvent``, so
        hiding / showing the glass never resets capture mode.
        """
        self._capture_excluded = bool(excluded)
        if sys.platform == "win32":
            self._apply_capture_affinity()

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
            self._apply_capture_affinity()
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
        # Blocks first erase, then all draw (a later block's patch must not
        # cover an earlier block's lettering), then the plain segments.
        blocks = [(i, seg) for i, seg in enumerate(self._segments) if seg.style.is_block and not seg.untranslated]
        if self._hide_original:
            for i, seg in blocks:
                self._paint_clean_patch(painter, i, seg.style, dpr)
        for i, seg in blocks:
            try:
                self._paint_block(painter, seg, dpr, i)
            except Exception:  # one bad segment must not blank the glass
                log.exception("failed to paint block %r", seg.source_text)
        for seg in self._segments:
            if seg.style.is_block or seg.untranslated:
                continue
            try:
                self._paint_segment(painter, seg, dpr)
            except Exception:
                log.exception("failed to paint segment %r", seg.source_text)

        if self._grab_mode:
            self._paint_grab_frame(painter)
        painter.end()

    # ---------------------------------------------------------- painting
    def _paint_clean_patch(self, painter: QPainter, index: int, style: SegmentStyle, dpr: float) -> None:
        image = self._patches.get(index)
        rect = style.clean_rect
        if image is None or rect is None:
            return
        painter.drawImage(QRectF(rect.x / dpr, rect.y / dpr, rect.w / dpr, rect.h / dpr), image)

    def _paint_block(self, painter: QPainter, seg: TranslatedSegment, dpr: float, index: int = -1) -> None:
        """Flow the translation through the block's layout region and letter
        it in the comic font, with a halo when it sits on artwork.  Everything
        is computed in frame pixels; the painter is scaled by ``1/dpr``."""
        text = seg.translation.strip()
        if not text:
            return
        # As :func:`compose.compose`: capitals in the comic font, unless the
        # translation is substantially CJK (Anime Ace has no such glyphs): then
        # the overlay font, upright and as translated, like the PIL renderer's
        # CJK system-font fallback.  The test is a ratio, not `has_cjk`: one
        # misread ideograph used to re-face a whole balloon into the system sans.
        cjk = needs_cjk_face(text)
        if self._uppercase and not cjk:
            text = text.upper()
        style = seg.style
        italic = block_italic(style) and not cjk
        family = self._font_family if cjk else self._block_family
        ts = self._typesets.get(index) if index >= 0 else None
        if ts is None:
            if cjk:
                measure = self._measure
            else:
                measure = self._block_measure_italic if italic else self._block_measure
            ts = typeset_block(text, style, measure, lang=seg.tgt_lang or "en")
            if index >= 0:
                self._typesets[index] = ts
        if not ts.lines:
            return
        # Dialogue (vertical source text) is italic like printed comics;
        # horizontal captions such as chapter titles stay upright.
        font = make_font(family, ts.size, italic=italic)
        layer = self._layers.get(index) if index >= 0 else None
        if layer is None:
            layer = self._render_layer(ts, font, style)
            if index >= 0:
                self._layers[index] = layer
        image, x0, y0 = layer
        painter.save()
        painter.scale(1.0 / dpr, 1.0 / dpr)
        if abs(style.angle_deg) >= 0.5:
            box = style.layout_box
            assert box is not None
            cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
            painter.translate(cx, cy)
            painter.rotate(-style.angle_deg)
            painter.translate(-cx, -cy)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QPointF(x0, y0), image)
        painter.restore()

    def _render_layer(self, ts: Typeset, font: QFont, style: SegmentStyle) -> Tuple[QImage, int, int]:
        """Render the lettering of ``ts`` once into a transparent ARGB image
        at frame resolution: ``(image, x, y)`` with ``(x, y)`` the frame pixel
        of the image's top-left corner (integers, so blitting it reproduces
        the direct drawing's sub-pixel positions exactly).  The image covers
        the glyph outlines plus the halo / weight pens."""
        paths = self._glyph_paths(ts, font)
        bounds = QRectF()
        for path in paths:
            bounds = bounds.united(path.boundingRect())
        pad = halo_px(ts.size) + weight_px(ts.size) + 2
        x0 = int(math.floor(bounds.left())) - pad
        y0 = int(math.floor(bounds.top())) - pad
        w = int(math.ceil(bounds.right())) - x0 + pad + 1
        h = int(math.ceil(bounds.bottom())) - y0 + pad + 1
        image = QImage(max(1, w), max(1, h), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)
        p = QPainter(image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.translate(-x0, -y0)
        self._stroke_paths(p, paths, ts.size, style)
        p.end()
        return image, x0, y0

    def _draw_typeset(self, painter: QPainter, ts: Typeset, font: QFont, style: SegmentStyle) -> None:
        """Draw the placed lines of ``ts`` at their frame positions, exactly
        as :func:`glasstranslate.render.compose.draw_typeset` does with PIL.

        ``PlacedLine.top`` is the top of the font's line box (the PIL text
        origin), so the Qt baseline is ``top + ascent``.  Two passes over all
        lines, as glyph outlines (``QPainterPath.addText``): first, when the
        block sits on artwork (``style.outline``), every line's halo - the
        paper colour, reaching ``halo_px + weight_px`` outward, which for a
        pen centred on the outline means twice that width, round joins and
        caps like PIL's stroker - then every line's glyphs filled with the
        foreground colour under a thin foreground pen of ``2 x weight_px``
        (the same outward reach as PIL's ``stroke_width=weight_px``).
        Halos first keeps a line's halo from cutting into the descenders of
        the line above.  (``paintEvent`` blits the cached :meth:`_render_layer`
        image made with this same code instead of re-stroking every paint.)"""
        self._stroke_paths(painter, self._glyph_paths(ts, font), ts.size, style)

    @staticmethod
    def _glyph_paths(ts: Typeset, font: QFont) -> List[QPainterPath]:
        """One glyph-outline path per placed line, in frame pixels."""
        ascent = QFontMetricsF(font).ascent()
        paths: List[QPainterPath] = []
        for line in ts.lines:
            path = QPainterPath()
            path.addText(QPointF(line.cx - line.width / 2.0, line.top + ascent), font, line.text)
            paths.append(path)
        return paths

    @staticmethod
    def _stroke_paths(painter: QPainter, paths: Sequence[QPainterPath], size: float, style: SegmentStyle) -> None:
        """The two drawing passes of :meth:`_draw_typeset` over prebuilt paths."""
        weight = weight_px(size)
        if style.outline:
            bg = QColor(*style.bg)
            halo = QPen(bg, 2.0 * (halo_px(size) + weight))
            halo.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            halo.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(halo)
            painter.setBrush(bg)
            for path in paths:
                painter.drawPath(path)
        fg = QColor(*style.fg)
        if weight > 0:
            stroke = QPen(fg, 2.0 * weight)
            stroke.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            stroke.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(stroke)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fg)
        for path in paths:
            painter.drawPath(path)

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

    def _apply_capture_affinity(self) -> None:
        """Push ``self._capture_excluded`` to the native window (no-op before it exists)."""
        user32 = _user32()
        if user32 is None or not self._native_ready:
            return
        affinity = _WDA_EXCLUDEFROMCAPTURE if self._capture_excluded else _WDA_NONE
        if not user32.SetWindowDisplayAffinity(_hwnd(self), affinity):
            log.warning("SetWindowDisplayAffinity(0x%X) failed (error %d); glass capture state unknown",
                        affinity, ctypes.get_last_error())

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
        new = list(segments) if isinstance(segments, (list, tuple)) else []
        # The pipeline re-emits the very same TranslatedSegment objects on
        # every pass that changed nothing (and for the untouched blocks of a
        # partial re-read), so cached layouts, lettering layers and patch
        # images are carried over by object identity; anything else is
        # recomputed.  ``self._segments`` keeps the old objects alive while
        # the ids are matched, so an id cannot be reused by a new object.
        old_index = {id(seg): i for i, seg in enumerate(self._segments)}
        typesets: Dict[int, Typeset] = {}
        layers: Dict[int, Tuple[QImage, int, int]] = {}
        patches: Dict[int, QImage] = {}
        serials: Dict[int, int] = {}
        for i, seg in enumerate(new):
            j = old_index.get(id(seg))
            if j is not None:
                if j in self._typesets:
                    typesets[i] = self._typesets[j]
                if j in self._layers:
                    layers[i] = self._layers[j]
            if seg.style.is_block and seg.style.clean_patch is not None:
                # The quality renderer replaces a block's clean_patch in place and
                # bumps its serial, so identity alone is not enough to reuse the
                # converted image: the pair (segment, serial) is the cache key.
                serial = int(seg.style.clean_patch_serial)
                reused = self._patches.get(j) if j is not None and self._patch_serials.get(j) == serial else None
                if reused is None:
                    reused = bgr_to_qimage(seg.style.clean_patch)
                    self._patch_conversions += 1
                    if j is not None and j in self._patches:
                        self._patch_upgrades += 1  # same block, new serial: a sidecar upgrade
                patches[i] = reused
                serials[i] = serial
        self._segments = new
        self._typesets = typesets
        self._layers = layers
        self._patches = patches
        self._patch_serials = serials
        self.update()
