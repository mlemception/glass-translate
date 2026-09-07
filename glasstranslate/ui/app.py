"""Application entry point: wires config, control window, glass overlay,
pipeline and global hotkeys together.

``main(config_path=None)`` blocks until the Qt event loop exits.  Environment
variables for automated runs:

* ``GLASSTRANSLATE_AUTOEXIT_MS`` - quit automatically after that many ms.
* ``GLASSTRANSLATE_CONFIG`` - config file path (overridden by ``config_path``).
* ``GLASSTRANSLATE_SMOKE_LOG`` - write the JSON smoke report of
  docs/GLASS_DESIGN.md section 5 to this path (screenshot PNG next to it).
* ``GLASSTRANSLATE_LOGLEVEL`` - root logging level (default INFO).

Frozen (PyInstaller, no console) specifics: logging goes to a rotating file under
``%LOCALAPPDATA%/GlassTranslate/logs`` whenever ``sys.stderr`` is missing or the
app is frozen, and Qt's own messages are routed to the ``qt`` logger.
"""
from __future__ import annotations

import gc
import json
import logging
import logging.handlers
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QCoreApplication, QEvent, QFile, QObject, QRect, QTimer, QtMsgType, Signal, Slot, qInstallMessageHandler
from PySide6.QtGui import QFontDatabase, QSurfaceFormat
from PySide6.QtQml import QQmlExpression, qmlContext
from PySide6.QtQuick import QQuickItem, QQuickView
from PySide6.QtWidgets import QApplication

from .. import __version__
from ..config.settings import AppConfig, Hotkeys, OverlayGeometry, default_config_path, user_data_dir
from ..core.pipeline import Pipeline
from ..core.types import PipelineStats
from .control import ControlWindow, is_frozen
from .glass import win32
from .hotkeys import HotkeyManager
from .overlay import GlassOverlay

__all__ = ["GlassTranslateApp", "SmokeReport", "main"]

log = logging.getLogger(__name__)

AUTOEXIT_ENV = "GLASSTRANSLATE_AUTOEXIT_MS"
CONFIG_ENV = "GLASSTRANSLATE_CONFIG"
SMOKE_LOG_ENV = "GLASSTRANSLATE_SMOKE_LOG"
LOGLEVEL_ENV = "GLASSTRANSLATE_LOGLEVEL"
LOG_FILE_NAME = "glasstranslate.log"
LOG_MAX_BYTES = 2_000_000
LOG_BACKUP_COUNT = 3
SMOKE_GRAB_LEAD_MS = 1500  # screenshot this long before the autoexit
SMOKE_PAGES_LEAD_MS = 3000  # tab walk this long before the autoexit
MANGA_FONT_FAMILY = "Anime Ace 2.0 BB"
PAGE_NAMES = ("TranslatePage", "OverlayPage", "EnginesPage", "HotkeysPage")
_SHADER_STATUS_COMPILED, _SHADER_STATUS_UNCOMPILED, _SHADER_STATUS_ERROR = 0, 1, 2
_RHI_NAMES = {
    "Direct3D11": "D3D11", "Direct3D11Rhi": "D3D11", "Direct3D12": "D3D12", "OpenGL": "OpenGL",
    "OpenGLRhi": "OpenGL", "Vulkan": "Vulkan", "VulkanRhi": "Vulkan", "Metal": "Metal", "MetalRhi": "Metal",
    "Software": "Software", "Null": "Null", "NullRhi": "Null", "Unknown": "Unknown",
}
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
        self.last_stats: Optional[PipelineStats] = None
        self._bound_hotkeys: Optional[Hotkeys] = None
        self._torn_down = False

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
    ``%LOCALAPPDATA%/GlassTranslate/logs`` when frozen or without stderr.  Returns the handlers
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
        log_dir = user_data_dir() / "logs"
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


def _qt_message_handler(mode: QtMsgType, context: Any, message: str) -> None:
    """Route Qt / QML messages to the ``qt`` logger (there is no console in the exe)."""
    category = getattr(context, "category", "") or "default"
    logging.getLogger("qt").log(_QT_LEVELS.get(mode, logging.WARNING), "[%s] %s", category, message)


# --------------------------------------------------------------------------- smoke report
def _qml_number(item: QObject, expression: str) -> Optional[float]:
    """Evaluate a QML expression on ``item`` as a number (enum properties have no Python converter)."""
    context = qmlContext(item)
    if context is None:
        return None
    expr = QQmlExpression(context, item, f"Number({expression})")
    result = expr.evaluate()
    value = result[0] if isinstance(result, tuple) else result
    if expr.hasError() or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quick_items(control: QQuickView) -> List[QQuickItem]:
    root = control.rootObject()
    if root is None:
        return []
    return [root] + list(root.findChildren(QQuickItem))


def _shader_effects(control: QQuickView) -> Dict[str, Any]:
    """Walk every item exposing ``fragmentShader`` + ``status``; ``error`` gates, ``compiled`` informs."""
    total = compiled = error = 0
    logs: List[str] = []
    for item in _quick_items(control):
        mo = item.metaObject()
        if mo.indexOfProperty("fragmentShader") < 0 or mo.indexOfProperty("status") < 0:
            continue
        total += 1
        status = _qml_number(item, "status")
        if status == _SHADER_STATUS_COMPILED:
            compiled += 1
        elif status == _SHADER_STATUS_ERROR:
            error += 1
        text = item.property("log")
        if text:
            logs.append(str(text))
    return {"total": total, "compiled": compiled, "error": error, "logs": logs}


def _rhi_backend(control: QQuickView) -> str:
    try:
        api = control.rendererInterface().graphicsApi()
    except Exception:  # pragma: no cover - scene graph not initialised yet
        return "Unknown"
    name = getattr(api, "name", None) or str(api).rsplit(".", 1)[-1]
    return _RHI_NAMES.get(str(name), str(name))


def _class_matches(class_name: str, base: str) -> bool:
    return class_name == base or class_name.startswith(base + "_QMLTYPE") or class_name.startswith(base + "_QML")


def _qrc_component_names() -> Tuple[List[str], List[str]]:
    """(Glass* components, pages) present in the compiled ``:/qml`` resources."""
    from PySide6.QtCore import QDir

    comps = sorted(f[:-4] for f in QDir(":/qml").entryList(["Glass*.qml"]) if f.endswith(".qml"))
    pages = sorted(f[:-4] for f in QDir(":/qml/pages").entryList(["*.qml"]) if f.endswith(".qml"))
    return comps, pages


def exercise_pages(control: QQuickView) -> Tuple[bool, Dict[str, Any]]:
    """Open every page through the tab bar and check every Glass* component was instantiated.

    The tab bar is found by ``objectName == "tabBar"`` or by class name (``GlassSegmentedBar``);
    its ``currentIndex`` is set to 0..3 (then back to 0).  Returns ``(pages_ok, details)``.
    """
    details: Dict[str, Any] = {"opened": [], "missing": [], "tab_bar": False}
    root = control.rootObject()
    if root is None:
        details["error"] = "no root object"
        return False, details
    bar = root.findChild(QQuickItem, "tabBar")
    if bar is None:
        for item in _quick_items(control):
            if _class_matches(item.metaObject().className(), "GlassSegmentedBar"):
                bar = item
                break
    if bar is not None and bar.metaObject().indexOfProperty("currentIndex") >= 0:
        details["tab_bar"] = True
        app = QCoreApplication.instance()
        for index in range(len(PAGE_NAMES)):
            bar.setProperty("currentIndex", index)
            if app is not None:
                app.processEvents()
            if int(bar.property("currentIndex") or 0) == index:
                details["opened"].append(index)
        bar.setProperty("currentIndex", 0)
        if app is not None:
            app.processEvents()
    class_names = [item.metaObject().className() for item in _quick_items(control)]
    comps, pages = _qrc_component_names()
    required = list(comps) + (pages or list(PAGE_NAMES))
    for base in required:
        if not any(_class_matches(name, base) for name in class_names):
            details["missing"].append(base)
    details["required"] = required
    ok = details["tab_bar"] and len(details["opened"]) == len(PAGE_NAMES) and not details["missing"]
    return ok, details


class SmokeReport(QObject):
    """Writes the JSON report of docs/GLASS_DESIGN.md section 5 for automated runs.

    With an autoexit, the tab walk runs ``SMOKE_PAGES_LEAD_MS`` and the ``grabWindow()``
    screenshot ``SMOKE_GRAB_LEAD_MS`` before the quit, the latter only after at least one
    ``frameSwapped`` (a grab from ``aboutToQuit`` finds an unexposed window).  Without an
    autoexit - or if the timers never fired - the report is written from ``aboutToQuit``
    without a screenshot, so the file always exists.
    """

    def __init__(self, session: GlassTranslateApp, path: Path, autoexit_ms: Optional[int],
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.session = session
        self.path = Path(path)
        self.written = False
        self._frames = 0
        self._pending_grab = False
        self._pages: Optional[Tuple[bool, Dict[str, Any]]] = None
        session.control.frameSwapped.connect(self._on_frame)
        if autoexit_ms is not None:
            QTimer.singleShot(max(100, autoexit_ms - SMOKE_PAGES_LEAD_MS), self._walk_pages)
            QTimer.singleShot(max(200, autoexit_ms - SMOKE_GRAB_LEAD_MS), self._grab_and_write)
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.write_if_needed)

    @Slot()
    def _on_frame(self) -> None:
        self._frames += 1
        if self._pending_grab:
            self._pending_grab = False
            self._grab_and_write()

    @Slot()
    def _walk_pages(self) -> None:
        if self._pages is None:
            try:
                self._pages = exercise_pages(self.session.control)
            except Exception as exc:  # pragma: no cover - report the failure instead of crashing
                log.exception("smoke: page walk failed")
                self._pages = (False, {"error": repr(exc)})

    @Slot()
    def _grab_and_write(self) -> None:
        if self.written:
            return
        if self._frames == 0:
            self._pending_grab = True  # wait for the first swapped frame
            return
        if self._pages is None:
            self._walk_pages()
        screenshot: Optional[Path] = None
        try:
            image = self.session.control.grabWindow()
            if not image.isNull():
                screenshot = self.path.with_suffix(".png")
                image.save(str(screenshot))
        except Exception:  # pragma: no cover
            log.exception("smoke: grabWindow failed")
        self.write(screenshot)

    @Slot()
    def write_if_needed(self) -> None:
        if not self.written:
            if self._pages is None and self.session.control.rootObject() is not None:
                self._walk_pages()
            self.write(None)

    def build(self, screenshot: Optional[Path]) -> Dict[str, Any]:
        control = self.session.control
        bridge = control.bridge
        appearance = control.appearance.signals
        pages_ok, pages_detail = self._pages if self._pages is not None else (False, {"error": "not run"})
        geom = control.geometry()
        dpr = float(control.devicePixelRatio() or 1.0)
        return {
            "frozen": is_frozen(),
            "meipass": getattr(sys, "_MEIPASS", None),
            "exe": sys.executable,
            "version": __version__,
            "qml_status": control.status().name if hasattr(control.status(), "name") else str(control.status()),
            "qml_errors": [e.toString() for e in control.errors()],
            "qml_warnings": list(control.qml_warnings),
            "shader_effects": _shader_effects(control),
            "rhi_backend": _rhi_backend(control),
            "appearance": {
                "mode": appearance.mode,
                "transparency": appearance.transparency,
                "reduceMotion": appearance.reduce_motion,
                "highContrast": appearance.high_contrast,
                "darkMode": appearance.dark_mode,
                "textScale": appearance.text_scale,
            },
            "backdrop_serial": int(bridge.backdropSerial),
            "backdrop_luma": float(bridge.backdropLuma),
            "ink_polarity": int(bridge.inkPolarity),
            "fonts_ok": MANGA_FONT_FAMILY in QFontDatabase.families(),
            "resources_ok": {
                "qml": QFile.exists(":/qml/Main.qml"),
                "shaders": all(QFile.exists(f":/qml/shaders/{n}.frag.qsb") for n in ("glass", "blur", "shadow")),
                "fonts": all(QFile.exists(f":/fonts/animeace2_{n}.ttf") for n in ("reg", "ital")),
                "icon": QFile.exists(":/icons/app.png"),
            },
            "control_exposed": control.isExposed(),
            "overlay_shown": self.session.overlay.isVisible(),
            "pages_ok": bool(pages_ok),
            "pages_detail": pages_detail,
            "dpi_awareness": win32.dpi_awareness(),
            "status_history": list(control.status_history),
            "dpr": dpr,
            "window": {"x": geom.x(), "y": geom.y(), "w": geom.width(), "h": geom.height(), "dpr": dpr},
            "frames_swapped": self._frames,
            "screenshot": str(screenshot.resolve()) if screenshot is not None else None,
        }

    def write(self, screenshot: Optional[Path]) -> None:
        report = self.build(screenshot)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)
        self.written = True
        log.info("smoke report written to %s", self.path)


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
    app.aboutToQuit.connect(session.shutdown)

    autoexit_env = os.environ.get(AUTOEXIT_ENV)
    autoexit_ms = max(0, int(autoexit_env)) if autoexit_env else None
    smoke_path = os.environ.get(SMOKE_LOG_ENV)
    smoke = SmokeReport(session, Path(smoke_path), autoexit_ms) if smoke_path else None

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
