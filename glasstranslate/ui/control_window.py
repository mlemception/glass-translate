"""``ControlWindow`` - the frameless ``QQuickView`` hosting ``Main.qml`` and the backdrop grabber.

Split out of ``ui/control.py`` (which still re-exports every name here); see that module's
docstring for the window/bridge/backdrop design.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from PySide6.QtCore import QEvent, QSize, Qt, QUrl, Signal, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QExposeEvent,
    QHideEvent,
    QIcon,
    QMoveEvent,
    QPlatformSurfaceEvent,
    QResizeEvent,
    QScreen,
    QShowEvent,
    QWindow,
)
from PySide6.QtQuick import QQuickItem, QQuickView

from ..config.settings import AppConfig
from ..core.types import PipelineStats
from .control_bridge import WINDOW_TITLE, ControlBridge
from .glass import profile, win32
from .glass.appearance import Appearance
from .glass.backdrop import BackdropFrame, BackdropGrabber, BackdropProvider, WindowGeometry
from .resources_guard import RESOURCES_OK

__all__ = ["ControlWindow", "MAIN_QML_URL", "WINDOW_DEFAULT_SIZE", "WINDOW_FLAGS"]

log = logging.getLogger(__name__)

MAIN_QML_URL = "qrc:/qml/Main.qml"
WINDOW_DEFAULT_SIZE = QSize(832, 640)  # slab 760x560 inside the 36/28/36/52 shadow margin; fixed-size, not resizable
WINDOW_FLAGS = (
    Qt.WindowType.Window
    | Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowMinimizeButtonHint
    | Qt.WindowType.WindowSystemMenuHint
)


# ------------------------------------------------------------------------------------ window
class ControlWindow(QQuickView):
    """Main settings window; see the module docstring."""

    config_changed = Signal(object)  # AppConfig
    start_stop_requested = Signal(bool)  # True = start
    grab_mode_requested = Signal()
    toggle_glass_requested = Signal()
    capture_mode_requested = Signal(bool)  # capture mode on/off (the overlay and this window)
    models_changed = Signal()  # a package was downloaded
    closed = Signal()  # the user closed the window
    _frame_ready = Signal(object)  # BackdropFrame, emitted from the grabber thread (queued)

    def __init__(
        self,
        cfg: AppConfig,
        config_path: Optional[Path] = None,
        parent: Optional[QWindow] = None,
    ) -> None:
        super().__init__(parent)
        self.setTitle(WINDOW_TITLE)
        self.setFlags(WINDOW_FLAGS)
        self.setColor(Qt.GlobalColor.transparent)
        self.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
        self.setMinimumSize(WINDOW_DEFAULT_SIZE)
        self.setMaximumSize(WINDOW_DEFAULT_SIZE)
        self.resize(WINDOW_DEFAULT_SIZE)
        if RESOURCES_OK:
            self.setIcon(QIcon(":/icons/app.png"))

        self.appearance = Appearance(self)
        self.bridge = ControlBridge(cfg, config_path, window=self, appearance=self.appearance, parent=self)
        for name in ("config_changed", "start_stop_requested", "grab_mode_requested", "toggle_glass_requested",
                     "capture_mode_requested", "models_changed"):
            getattr(self.bridge, name).connect(getattr(self, name))

        self._provider = BackdropProvider()
        self.engine().addImageProvider("backdrop", self._provider)
        self.qml_warnings: List[str] = []
        self.engine().warnings.connect(self._on_qml_warnings)
        context = self.rootContext()
        context.setContextProperty("bridge", self.bridge)
        context.setContextProperty("appearance", self.appearance)

        self._geometry = WindowGeometry()
        self._grabber: Optional[BackdropGrabber] = None
        self._capture_excluded = True  # capture mode clears this; re-applied on every show
        self._native_ready = False  # True while the platform window exists (showEvent sets, surface destruction clears)
        self._close_seen = False
        self._frame_ready.connect(self._on_backdrop_frame)
        self.windowStateChanged.connect(self._on_window_state_changed)
        self.screenChanged.connect(self._on_screen_changed)
        self.appearance.changed.connect(self._update_grabber)

        self.setSource(QUrl(MAIN_QML_URL))
        if self.status() == QQuickView.Status.Error:
            for err in self.errors():
                log.error("QML: %s", err.toString())
        elif self.rootObject() is None:
            log.error("QML root object missing (status %s)", self.status())
        if profile.profiler.enabled:
            self._install_profile_hooks()

    def _install_profile_hooks(self) -> None:
        """``GLASSTRANSLATE_PROFILE`` only: count swaps and tab switches (F2-0, never on by default)."""
        prof = profile.profiler
        # frameSwapped is emitted on the render thread.  A direct Python slot there needs the GIL
        # while the GUI thread may hold it inside C++ (exposeEvent -> render-loop sync waiting for
        # that same render thread) -> deadlock on the first frame.  Queue it: counts stay exact,
        # the timestamp becomes GUI-dispatch time (documented in profile.py).
        self._profile_hooks: List[Any] = [
            self.frameSwapped.connect(lambda: prof.mark("swap"), Qt.ConnectionType.QueuedConnection)
        ]
        tab = self.rootObject().findChild(QQuickItem, "tabBar") if self.rootObject() is not None else None
        if tab is not None:
            self._profile_hooks.append(tab.currentIndexChanged.connect(lambda: prof.mark("tab", index=tab.property("currentIndex"))))

    # ---------------------------------------------------------------- public
    @property
    def config(self) -> AppConfig:
        return self.bridge.config

    @property
    def status_history(self) -> List[str]:
        """Every ``show_status`` message so far (smoke report)."""
        return self.bridge.status_history

    @property
    def grabber(self) -> Optional[BackdropGrabber]:
        return self._grabber

    @property
    def capture_excluded(self) -> bool:
        """``False`` while capture mode lets screen-capture tools see this window."""
        return self._capture_excluded

    def set_capture_excluded(self, excluded: bool) -> None:
        """Hide this window from screen capture (default) or make it capturable (capture mode).

        The desired state is remembered and re-applied on every ``showEvent``.  While the window
        is capturable its backdrop grabber is frozen (mss would sample the panel itself), the last
        backdrop stays on the glass, and any frame still in flight is dropped; re-excluding resumes
        the grabber, which forces a fresh grab.
        """
        self._capture_excluded = bool(excluded)
        if self._capture_excluded:
            self._apply_capture_affinity()  # excluded again before the grabber may sample
            self._update_grabber()
            return
        self._update_grabber()  # frozen before the panel becomes visible to capture
        self._apply_capture_affinity()

    def load_config(self, cfg: AppConfig) -> None:
        """Populate every control from ``cfg`` without emitting changes."""
        self.bridge.load_config(cfg)

    def set_running(self, running: bool) -> None:
        """Reflect the pipeline state on the Start/Stop button and the running dot."""
        self.bridge.set_running(running)

    @Slot(object)
    def update_stats(self, stats: PipelineStats) -> None:
        """Refresh the latency readout from a pipeline pass."""
        if isinstance(stats, PipelineStats):
            self.bridge.stats.update(stats)

    @Slot(str)
    def show_status(self, message: str) -> None:
        self.bridge.show_status(message)

    def save_soon(self) -> None:
        """Schedule a debounced save (for changes made outside this window,
        e.g. the glass geometry)."""
        self.bridge.save_soon()

    @Slot()
    def save_now(self) -> None:
        """Persist the config immediately (normally driven by the debounce timer)."""
        self.bridge.save_now()

    # ---------------------------------------------------------------- events
    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._native_ready = True
        self._apply_capture_affinity()
        self._refresh_geometry()
        self._update_grabber()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self._update_grabber()

    def event(self, event: QEvent) -> bool:
        if (
            isinstance(event, QPlatformSurfaceEvent)
            and event.surfaceEventType() == QPlatformSurfaceEvent.SurfaceEventType.SurfaceAboutToBeDestroyed
        ):
            # close() / destroy(): applying the affinity now would re-create the native window
            # through winId(); the stored flag is applied again by the next showEvent instead.
            self._native_ready = False
        return super().event(event)

    def exposeEvent(self, event: QExposeEvent) -> None:
        super().exposeEvent(event)
        if self.isExposed():
            self._refresh_geometry()
            self._poke()

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)
        self._refresh_geometry()
        rect = self._geometry.get()[0]
        profile.profiler.geometry("move", rect)
        if rect is not None:
            self.bridge.set_window_origin(rect.x, rect.y)
        self._poke()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_geometry()
        self._poke()

    def close(self) -> bool:
        """Close the window; flushes the pending save and emits ``closed``.

        ``QWindow.close()`` returns True *without* delivering ``closeEvent`` when the window has
        no platform window yet (never shown, e.g. offscreen tests), so that case is handled here.
        """
        self._close_seen = False
        ok = super().close()
        if ok and not self._close_seen:
            self._on_close()
        return ok

    def closeEvent(self, event: QCloseEvent) -> None:
        super().closeEvent(event)
        if event.isAccepted():
            self._close_seen = True
            self._on_close()

    def _on_close(self) -> None:
        self.bridge.flush_pending_save()
        self.shutdown()
        self.closed.emit()

    # -------------------------------------------------------------- backdrop
    def shutdown(self) -> None:
        """Stop the grabber thread and the appearance watcher (idempotent).

        Called from ``closeEvent`` and by the application before the ``QApplication`` is
        destroyed, so no thread or native event filter outlives the Qt objects it touches.
        """
        self.stop_grabber()
        self.appearance.stop()

    def stop_grabber(self, timeout_s: float = 1.0) -> None:
        """Stop and join the grabber thread (idempotent)."""
        grabber, self._grabber = self._grabber, None
        if grabber is not None:
            grabber.stop()
            grabber.join(timeout_s)

    def _refresh_geometry(self) -> None:
        if not self.isVisible():
            return
        self._geometry.update(win32.physical_rect(self), self.devicePixelRatio())

    def _apply_capture_affinity(self) -> None:
        """Push ``self._capture_excluded`` to the native window (stored only before the first show)."""
        if self._native_ready:
            win32.set_capture_excluded(self, self._capture_excluded)

    def _update_grabber(self) -> None:
        minimized = bool(self.windowStates() & Qt.WindowState.WindowMinimized)
        active = self.isVisible() and not minimized and self.appearance.glassAllowed and self._capture_excluded
        if active:
            if self._grabber is None:
                self._grabber = BackdropGrabber(self._geometry, self._frame_ready.emit)
                self._grabber.start()
            self._grabber.resume()
        elif self._grabber is not None:
            self._grabber.pause()

    def _poke(self) -> None:
        if self._grabber is not None and not self._grabber.paused:
            profile.profiler.mark("poke")
            self._grabber.poke()

    @Slot(object)
    def _on_backdrop_frame(self, frame: object) -> None:
        if not isinstance(frame, BackdropFrame) or not self._capture_excluded:
            return  # a grab still in flight when the window became capturable may contain the panel
        profile.profiler.frame_gui(frame, lambda: win32.physical_rect(self))
        self._provider.set_image(frame.image)
        profile.profiler.timed_call("push_backdrop_ms", self.bridge.push_backdrop, frame)

    @Slot(Qt.WindowState)
    def _on_window_state_changed(self, _state: Qt.WindowState) -> None:
        self._update_grabber()

    @Slot(QScreen)
    def _on_screen_changed(self, _screen: QScreen) -> None:
        self._refresh_geometry()
        self._poke()

    @Slot(list)
    def _on_qml_warnings(self, warnings: list) -> None:
        for warning in warnings:
            text = warning.toString() if hasattr(warning, "toString") else str(warning)
            self.qml_warnings.append(text)
            log.warning("QML: %s", text)
