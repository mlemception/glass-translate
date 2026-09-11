"""Application entry point: wires config, control window, glass overlay,
pipeline and global hotkeys together.

``main(config_path=None)`` blocks until the Qt event loop exits.  Environment
variables for automated runs:

* ``GLASSTRANSLATE_AUTOEXIT_MS`` - quit automatically after that many ms.
* ``GLASSTRANSLATE_CONFIG`` - config file path (overridden by ``config_path``).
* ``GLASSTRANSLATE_SMOKE_LOG`` - write the JSON smoke report of
  docs/GLASS_DESIGN.md section 5 to this path (screenshot PNG next to it).
* ``GLASSTRANSLATE_LOGLEVEL`` - root logging level (default INFO).
* ``GLASSTRANSLATE_SMOKE_ACTIONS`` - comma list of whitelisted actions to drive
  (``downloadMangaOcr``, ``probeSidecar``, ``feedPage``); the app reports and quits
  once they are done.
* ``GLASSTRANSLATE_SMOKE_PAGE`` - still image the ``feedPage`` action feeds the pipeline
  instead of the screen.

The report itself and the actions live in ``smoke_report.py``; the names above are re-exported
here, which is where the tests and ``packaging/smoke_test.py`` expect them.

Frozen (PyInstaller, no console) specifics: logging goes to a rotating file under
``settings.logs_dir()`` - the portable root's ``logs`` folder, else
``%LOCALAPPDATA%/GlassTranslate/logs`` - whenever ``sys.stderr`` is missing or the app is
frozen, and Qt's own messages are routed to the ``qt`` logger.
"""
from __future__ import annotations

import gc
import logging
import logging.handlers
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QRect, QTimer, QtMsgType, Signal, Slot, qInstallMessageHandler
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from .. import __version__
from ..config.settings import (
    AppConfig,
    Hotkeys,
    OverlayGeometry,
    cache_dir,
    default_config_path,
    is_portable,
    logs_dir,
)
from ..core.interfaces import ScreenCapture
from ..core.pipeline import Pipeline
from ..core.types import PipelineStats
from .capture_mode import CaptureMode
from .control import ControlWindow, is_frozen
from .hotkeys import HotkeyManager
from .overlay import GlassOverlay
from .smoke_report import (  # noqa: F401 - re-exported: the tests and the packaging scripts read them here
    PROBE_READY_TIMEOUT_S,
    SMOKE_ACTION_DELAY_MS,
    SMOKE_ACTION_SETTLE_MS,
    SMOKE_ACTIONS_ENV,
    SMOKE_GRAB_LEAD_MS,
    SMOKE_LOG_ENV,
    SMOKE_PAGE_ENV,
    SMOKE_PAGES_LEAD_MS,
    SmokeReport,
    exercise_pages,
)

__all__ = [
    "GlassTranslateApp",
    "PROBE_READY_TIMEOUT_S",
    "SMOKE_ACTIONS_ENV",
    "SMOKE_ACTION_DELAY_MS",
    "SMOKE_ACTION_SETTLE_MS",
    "SMOKE_GRAB_LEAD_MS",
    "SMOKE_LOG_ENV",
    "SMOKE_PAGES_LEAD_MS",
    "SMOKE_PAGE_ENV",
    "SmokeReport",
    "exercise_pages",
    "main",
]

log = logging.getLogger(__name__)

AUTOEXIT_ENV = "GLASSTRANSLATE_AUTOEXIT_MS"
CONFIG_ENV = "GLASSTRANSLATE_CONFIG"
LOGLEVEL_ENV = "GLASSTRANSLATE_LOGLEVEL"
LOG_FILE_NAME = "glasstranslate.log"
LOG_MAX_BYTES = 2_000_000
LOG_BACKUP_COUNT = 3
QML_CACHE_ENV = "QML_DISK_CACHE_PATH"  # where Qt puts the compiled QML cache
RHI_CACHE_ENV = "QSG_RHI_DISABLE_DISK_CACHE"  # ...and whether it keeps a pipeline cache at all
_QT_LEVELS = {
    QtMsgType.QtDebugMsg: logging.DEBUG,
    QtMsgType.QtInfoMsg: logging.INFO,
    QtMsgType.QtWarningMsg: logging.WARNING,
    QtMsgType.QtCriticalMsg: logging.ERROR,
    QtMsgType.QtFatalMsg: logging.CRITICAL,
}


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
        # None = the pipeline's own default (a real screen-capture backend).  The feedPage smoke
        # action puts a StaticPageCapture factory here before the pipeline starts; a pipeline
        # that is already running is handed the factory through ``Pipeline.replace_capture``.
        self.capture_factory: Optional[Callable[[AppConfig], ScreenCapture]] = None
        self.last_stats: Optional[PipelineStats] = None
        self._bound_hotkeys: Optional[Hotkeys] = None
        self._torn_down = False
        self.capture_mode = CaptureMode(
            self.overlay,
            self.control,
            pipeline=lambda: self.pipeline,
            wants_running=lambda: self.cfg.running_on_start,
            status=self.control.show_status,
        )

        geom = cfg.overlay
        self.overlay.setGeometry(QRect(geom.x, geom.y, geom.w, geom.h))
        self._apply_overlay_settings(cfg)

        self._result_ready.connect(self._on_result_gui)
        self._status_ready.connect(self._on_status_gui)
        self.control.config_changed.connect(self._on_config_changed)
        self.control.start_stop_requested.connect(self._on_start_stop)
        self.control.grab_mode_requested.connect(self.overlay.toggle_grab_mode)
        self.control.toggle_glass_requested.connect(self.toggle_glass)
        self.control.capture_mode_requested.connect(self.capture_mode.set_active)
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
        self.capture_mode.restore()
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self.control.save_now()

    def teardown(self) -> None:
        """Destroy the windows explicitly, before the ``QApplication`` goes away.

        Letting Python garbage-collect a ``QQuickView``/``QWidget`` after the application object
        is gone crashes at exit (native teardown order); deleting them here while the event loop
        can still run their deferred deletes avoids that.  Queued pipeline callbacks that are
        already posted (the final "Pipeline stopped" status) are delivered *before* the windows
        go, and the slots ignore anything that arrives later.
        """
        self.shutdown()
        app = QCoreApplication.instance()
        if app is not None:
            app.processEvents()  # deliver the queued _status_ready / _result_ready calls now
        self._torn_down = True
        self.control.shutdown()
        self.control.hide()
        self.overlay.hide()
        self.overlay.deleteLater()
        self.control.deleteLater()
        if app is not None:
            app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            app.processEvents()

    # -------------------------------------------------------------- pipeline
    @property
    def running(self) -> bool:
        return self.pipeline is not None and self.pipeline.is_alive() and not self.pipeline.paused

    def start_pipeline(self) -> None:
        if self.pipeline is None or not self.pipeline.is_alive():
            extra: Dict[str, Any] = {}
            if self.capture_factory is not None:
                extra["capture_factory"] = self.capture_factory
            self.pipeline = Pipeline(
                self.cfg,
                region_provider=self.overlay.physical_rect,
                on_result=self._result_ready.emit,
                on_status=self._status_ready.emit,
                **extra,
            )
            self.pipeline.start()
        else:
            self.pipeline.resume()
        self.cfg.running_on_start = True
        self.control.set_running(True)
        self.capture_mode.hold()  # a pipeline started during capture mode waits until it ends

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
        if self._torn_down:
            return
        if isinstance(stats, PipelineStats):
            self.last_stats = stats
            self.control.update_stats(stats)
        if isinstance(segments, list) and not self.overlay.grab_mode:
            self.overlay.set_segments(segments)

    @Slot(str)
    def _on_status_gui(self, message: str) -> None:
        if self._torn_down:
            return
        self.control.show_status(message)

    @Slot(object)
    def _on_config_changed(self, cfg: object) -> None:
        if not isinstance(cfg, AppConfig):
            return
        log.debug("config changed (backend=%s, opacity=%.2f, hotkeys=%s)", cfg.translation_backend,
                  cfg.overlay_opacity, cfg.hotkeys)
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
        self.overlay.set_uppercase(cfg.uppercase)

    def _bind_hotkeys(self, cfg: AppConfig) -> None:
        self._bound_hotkeys = replace(cfg.hotkeys)
        log.info("binding hotkeys: %s", cfg.hotkeys)
        self.hotkeys.rebind(
            {
                cfg.hotkeys.toggle_grab: self.overlay.toggle_grab_mode,
                cfg.hotkeys.toggle_running: self.toggle_pipeline,
                cfg.hotkeys.toggle_hidden: self.toggle_glass,
            }
        )


# ------------------------------------------------------------------------------- logging
def _configure_logging(logger: Optional[logging.Logger] = None, *, frozen: Optional[bool] = None) -> List[logging.Handler]:
    """Install the handlers of docs/GLASS_DESIGN.md section 5 on ``logger`` (default: root).

    A ``StreamHandler`` only when ``sys.stderr`` exists (it is ``None`` in a no-console exe, where
    ``basicConfig`` would silently discard everything); a ``RotatingFileHandler`` under
    ``settings.logs_dir()`` when frozen or without stderr.  Returns the handlers
    added (empty when the logger was already configured).
    """
    root = logger if logger is not None else logging.getLogger()
    if root.handlers:
        return []
    frozen = is_frozen() if frozen is None else frozen
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: List[logging.Handler] = []
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    if sys.stderr is None or frozen:
        log_dir = logs_dir()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(
                logging.handlers.RotatingFileHandler(
                    log_dir / LOG_FILE_NAME, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
                )
            )
        except OSError as exc:  # pragma: no cover - unwritable profile
            if sys.stderr is not None:
                print(f"cannot open log file in {log_dir}: {exc}", file=sys.stderr)
    for handler in handlers:
        handler.setFormatter(fmt)
        root.addHandler(handler)
    root.setLevel(os.environ.get(LOGLEVEL_ENV, "INFO"))
    return handlers


def apply_portable_qt_environment(env: MutableMapping[str, str] = os.environ) -> Dict[str, str]:
    """Keep Qt's two disk caches out of the user profile in portable mode.

    Qt writes both to ``QStandardPaths::CacheLocation`` - ``%LOCALAPPDATA%\\GlassTranslate\\
    cache`` - whatever the app's own paths say: the **QML compilation cache**
    (``qmlcache/*.qmlc``) and the scene graph's **automatic RHI pipeline cache**
    (``qtpipelinecache-*``), about 6 MB together.  A portable bundle may not leave that behind,
    so the QML cache is redirected into :func:`~glasstranslate.config.settings.cache_dir` and
    the pipeline cache is switched off: every shader ships as a precompiled ``.qsb``, so there
    is nothing measurable to regain from caching the pipelines.

    Outside portable mode nothing is touched, and a variable the environment already carries is
    never overridden.  Returns what it set.  **Call before the QApplication exists.**
    """
    if not is_portable():
        return {}
    wanted = {QML_CACHE_ENV: str(cache_dir() / "qmlcache"), RHI_CACHE_ENV: "1"}
    applied = {key: value for key, value in wanted.items() if key not in env}
    for key, value in applied.items():
        env[key] = value
    return applied


def _qt_message_handler(mode: QtMsgType, context: Any, message: str) -> None:
    """Route Qt / QML messages to the ``qt`` logger (there is no console in the exe)."""
    category = getattr(context, "category", "") or "default"
    logging.getLogger("qt").log(_QT_LEVELS.get(mode, logging.WARNING), "[%s] %s", category, message)



# ----------------------------------------------------------------------------------- main
def main(config_path: Optional[Path] = None, argv: Optional[List[str]] = None) -> int:
    """Run the application; returns the process exit code."""
    _configure_logging()
    qInstallMessageHandler(_qt_message_handler)
    log.info("GlassTranslate %s starting, frozen=%s, python %s", __version__, is_frozen(), sys.version.split()[0])
    if config_path is None:
        env_path = os.environ.get(CONFIG_ENV)
        config_path = Path(env_path) if env_path else default_config_path()
    config_path = Path(config_path)
    cfg = AppConfig.load(config_path)

    # Before the QApplication: both variables are read when Qt builds its engines.
    applied = apply_portable_qt_environment()
    if applied:
        log.info("portable mode: %s", ", ".join(f"{k}={v}" for k, v in sorted(applied.items())))

    # Per-pixel alpha for the translucent QQuickView: the default surface format must carry an
    # alpha channel before the application (and its scene graph) exists.
    fmt = QSurfaceFormat()
    fmt.setAlphaBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("GlassTranslate")
    app.setApplicationVersion(__version__)
    app.setQuitOnLastWindowClosed(False)
    session = GlassTranslateApp(cfg, config_path)
    session.control.closed.connect(app.quit)

    autoexit_env = os.environ.get(AUTOEXIT_ENV)
    autoexit_ms = max(0, int(autoexit_env)) if autoexit_env else None
    smoke_path = os.environ.get(SMOKE_LOG_ENV)
    # Connected to aboutToQuit *before* the shutdown hook on purpose: Qt runs those slots in
    # connection order, and the report's grabWindow must not wait behind the pipeline tearing a
    # quality sidecar down (which can take tens of seconds after a real job).
    smoke = SmokeReport(session, Path(smoke_path), autoexit_ms) if smoke_path else None
    app.aboutToQuit.connect(session.shutdown)

    session.show()
    if autoexit_ms is not None:
        QTimer.singleShot(autoexit_ms, app.quit)

    code = app.exec()
    if smoke is not None:
        smoke.write_if_needed()
    session.teardown()
    del smoke, session
    gc.collect()
    app.processEvents()
    log.info("GlassTranslate exiting with code %d", code)
    return code
