"""Application entry point: wires config, control window, glass overlay,
pipeline and global hotkeys together.

``main(config_path=None)`` blocks until the Qt event loop exits.  Two
environment variables help automated smoke tests:

* ``GLASSTRANSLATE_AUTOEXIT_MS`` - quit automatically after that many ms.
* ``GLASSTRANSLATE_CONFIG`` - config file path (overridden by ``config_path``).
"""
from __future__ import annotations

import logging
import os
from dataclasses import replace
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QRect, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication

from ..config.settings import AppConfig, Hotkeys, OverlayGeometry, default_config_path
from ..core.pipeline import Pipeline
from ..core.types import PipelineStats, Rect, TranslatedSegment
from .control import ControlWindow
from .hotkeys import HotkeyManager
from .overlay import GlassOverlay

__all__ = ["GlassTranslateApp", "main"]

log = logging.getLogger(__name__)

AUTOEXIT_ENV = "GLASSTRANSLATE_AUTOEXIT_MS"
CONFIG_ENV = "GLASSTRANSLATE_CONFIG"


class GlassTranslateApp(QObject):
    """Owns the windows, the pipeline and the hotkeys for one app session.

    Pipeline callbacks run on the worker thread; they are turned into queued
    signals here so every widget access happens on the GUI thread.
    """

    _result_ready = Signal(object, object)  # list[TranslatedSegment], PipelineStats
    _status_ready = Signal(str)

    def __init__(self, cfg: AppConfig, config_path: Optional[Path] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.config_path = config_path
        self.control = ControlWindow(cfg, config_path)
        self.overlay = GlassOverlay()
        self.hotkeys = HotkeyManager(self)
        self.pipeline: Optional[Pipeline] = None
        self.last_stats: Optional[PipelineStats] = None
        self._bound_hotkeys: Optional[Hotkeys] = None

        geom = cfg.overlay
        self.overlay.setGeometry(QRect(geom.x, geom.y, geom.w, geom.h))
        self._apply_overlay_settings(cfg)

        self._result_ready.connect(self._on_result_gui)
        self._status_ready.connect(self._on_status_gui)
        self.control.config_changed.connect(self._on_config_changed)
        self.control.start_stop_requested.connect(self._on_start_stop)
        self.control.grab_mode_requested.connect(self.overlay.toggle_grab_mode)
        self.control.toggle_glass_requested.connect(self.toggle_glass)
        self.control.models_changed.connect(self._on_models_changed)
        self.overlay.geometry_changed.connect(self._on_overlay_geometry)
        self.overlay.grab_mode_changed.connect(self._on_grab_mode_changed)
        self.hotkeys.error.connect(self.control.show_status)
        self._bind_hotkeys(cfg)

    # -------------------------------------------------------------- lifecycle
    def show(self) -> None:
        self.control.show()
        self.overlay.show()
        if self.cfg.running_on_start:
            self.start_pipeline()

    def shutdown(self) -> None:
        """Stop everything; safe to call more than once."""
        self.hotkeys.stop()
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.control.save_now()

    # -------------------------------------------------------------- pipeline
    @property
    def running(self) -> bool:
        return self.pipeline is not None and self.pipeline.is_alive() and not self.pipeline.paused

    def start_pipeline(self) -> None:
        if self.pipeline is None or not self.pipeline.is_alive():
            self.pipeline = Pipeline(
                self.cfg,
                region_provider=self.overlay.physical_rect,
                on_result=self._result_ready.emit,
                on_status=self._status_ready.emit,
            )
            self.pipeline.start()
        else:
            self.pipeline.resume()
        self.cfg.running_on_start = True
        self.control.set_running(True)

    def stop_pipeline(self) -> None:
        if self.pipeline is not None:
            self.pipeline.pause()
        self.cfg.running_on_start = False
        self.control.set_running(False)
        self.overlay.set_segments([])
        self.control.show_status("Paused")

    def toggle_pipeline(self) -> None:
        if self.running:
            self.stop_pipeline()
        else:
            self.start_pipeline()

    def toggle_glass(self) -> None:
        self.overlay.setVisible(not self.overlay.isVisible())

    # ------------------------------------------------------------------ slots
    @Slot(object, object)
    def _on_result_gui(self, segments: object, stats: object) -> None:
        if isinstance(stats, PipelineStats):
            self.last_stats = stats
            self.control.update_stats(stats)
        if isinstance(segments, list) and not self.overlay.grab_mode:
            self.overlay.set_segments(segments)

    @Slot(str)
    def _on_status_gui(self, message: str) -> None:
        self.control.show_status(message)

    @Slot(object)
    def _on_config_changed(self, cfg: object) -> None:
        if not isinstance(cfg, AppConfig):
            return
        self.cfg = cfg
        self._apply_overlay_settings(cfg)
        if cfg.hotkeys != self._bound_hotkeys:
            # Restarting the global keyboard hook is slow; skip it for the
            # many config edits (slider drags) that leave hotkeys untouched.
            self._bind_hotkeys(cfg)
        if self.pipeline is not None:
            self.pipeline.set_config(cfg)

    @Slot(bool)
    def _on_start_stop(self, start: bool) -> None:
        if start:
            self.start_pipeline()
        else:
            self.stop_pipeline()

    @Slot()
    def _on_models_changed(self) -> None:
        if self.pipeline is not None:
            self.pipeline.refresh_models()

    @Slot(QRect)
    def _on_overlay_geometry(self, rect: QRect) -> None:
        self.cfg.overlay = OverlayGeometry(rect.x(), rect.y(), rect.width(), rect.height())
        if self.pipeline is not None:
            self.pipeline.invalidate()
        self.control.save_soon()

    @Slot(bool)
    def _on_grab_mode_changed(self, enabled: bool) -> None:
        if enabled:
            self.overlay.set_segments([])
            self.control.show_status("Grab mode: drag the glass, Esc to finish")
        elif self.pipeline is not None:
            self.pipeline.invalidate()

    # -------------------------------------------------------------- helpers
    def _apply_overlay_settings(self, cfg: AppConfig) -> None:
        self.overlay.set_background_opacity(cfg.overlay_opacity)
        self.overlay.set_font_family(cfg.font_family)
        self.overlay.set_hide_original(cfg.hide_original)

    def _bind_hotkeys(self, cfg: AppConfig) -> None:
        self._bound_hotkeys = replace(cfg.hotkeys)
        self.hotkeys.rebind(
            {
                cfg.hotkeys.toggle_grab: self.overlay.toggle_grab_mode,
                cfg.hotkeys.toggle_running: self.toggle_pipeline,
                cfg.hotkeys.toggle_hidden: self.toggle_glass,
            }
        )


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("GLASSTRANSLATE_LOGLEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def main(config_path: Optional[Path] = None, argv: Optional[List[str]] = None) -> int:
    """Run the application; returns the process exit code."""
    _configure_logging()
    if config_path is None:
        env_path = os.environ.get(CONFIG_ENV)
        config_path = Path(env_path) if env_path else default_config_path()
    config_path = Path(config_path)
    cfg = AppConfig.load(config_path)

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("GlassTranslate")
    app.setQuitOnLastWindowClosed(False)
    session = GlassTranslateApp(cfg, config_path)
    session.control.closed.connect(app.quit)
    app.aboutToQuit.connect(session.shutdown)
    session.show()

    autoexit = os.environ.get(AUTOEXIT_ENV)
    if autoexit:
        QTimer.singleShot(max(0, int(autoexit)), app.quit)

    return app.exec()
