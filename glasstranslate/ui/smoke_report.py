"""The JSON smoke report of docs/GLASS_DESIGN.md section 5 and the actions that drive it.

``GLASSTRANSLATE_SMOKE_LOG=<path>`` makes :func:`glasstranslate.ui.app.main` build a
:class:`SmokeReport` for the session; it walks the pages, grabs the control window and writes
the report before the app quits.  ``GLASSTRANSLATE_SMOKE_ACTIONS`` additionally drives a
whitelist of actions inside the running (usually frozen) app:

* ``downloadMangaOcr`` - the Engines page download, on a clean profile (smoke run D).
* ``probeSidecar``     - start the quality sidecar the app itself would launch, in fake mode,
  on a worker thread and record ``/health`` (the portable run's proof that the frozen sidecar
  under the bundle root is found and starts).
* ``feedPage``         - ``GLASSTRANSLATE_SMOKE_PAGE=<image>`` replaces screen capture by a
  still page so the pipeline OCRs, typesets and - with the real sidecar - upgrades it.

The app reports and quits by itself once the requested actions are done.  Lives apart from
``app.py`` only for size; ``app.py`` re-exports every name the tests and the packaging scripts
use, and imports this module at import time so PyInstaller's analysis follows it.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QCoreApplication, QFile, QObject, QTimer, Signal, Slot
from PySide6.QtGui import QFontDatabase
from PySide6.QtQml import QQmlExpression, qmlContext
from PySide6.QtQuick import QQuickItem, QQuickView

from .. import __version__
from ..config.settings import cache_dir, logs_dir, portable_root, user_data_dir
from .control import is_frozen
from .glass import win32

if TYPE_CHECKING:  # pragma: no cover - annotations only, never imported at run time
    from .app import GlassTranslateApp

__all__ = [
    "MANGA_FONT_FAMILY",
    "PAGE_NAMES",
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
]

log = logging.getLogger(__name__)

SMOKE_LOG_ENV = "GLASSTRANSLATE_SMOKE_LOG"
SMOKE_GRAB_LEAD_MS = 1500  # screenshot this long before the autoexit
SMOKE_PAGES_LEAD_MS = 3000  # tab walk this long before the autoexit
SMOKE_ACTIONS_ENV = "GLASSTRANSLATE_SMOKE_ACTIONS"  # comma list of whitelisted actions to drive
SMOKE_ACTION_DELAY_MS = 1500  # run the actions this long after start (window exposed, pipeline up)
SMOKE_ACTION_SETTLE_MS = 8000  # after the last action: let the pipeline rebuild its engines, then report + quit
SMOKE_PAGE_ENV = "GLASSTRANSLATE_SMOKE_PAGE"  # still image the feedPage action feeds the pipeline
# probeSidecar: how long the fake sidecar may take to print READY.  It must stay well inside the
# portable run's 45 s autoexit, or a hung sidecar would cost the report instead of failing a row.
PROBE_READY_TIMEOUT_S = 30.0
PROBE_WATCHDOG_MS = int(PROBE_READY_TIMEOUT_S * 1000)  # ...and the record says so if it does hang
PAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})  # feedPage accepts these
PAGE_MAX_BYTES = 64 << 20  # ...at this size at most
# The overlay converts patches on a *later* event-loop turn than the result that carried them,
# and a still page may never produce another result to look at, so the counters are re-read a
# few times instead of being trusted once (2 s of grace at the defaults).
FEED_RECHECK_MS = 100
FEED_RECHECK_MAX = 20
_SMOKE_ACTIONS = ("downloadMangaOcr", "probeSidecar", "feedPage")  # the only actions the environment may drive
MANGA_FONT_FAMILY = "Anime Ace 2.0 BB"
PAGE_NAMES = ("TranslatePage", "OverlayPage", "EnginesPage", "HotkeysPage")
_SHADER_STATUS_COMPILED, _SHADER_STATUS_UNCOMPILED, _SHADER_STATUS_ERROR = 0, 1, 2
_RHI_NAMES = {
    "Direct3D11": "D3D11", "Direct3D11Rhi": "D3D11", "Direct3D12": "D3D12", "OpenGL": "OpenGL",
    "OpenGLRhi": "OpenGL", "Vulkan": "Vulkan", "VulkanRhi": "Vulkan", "Metal": "Metal", "MetalRhi": "Metal",
    "Software": "Software", "Null": "Null", "NullRhi": "Null", "Unknown": "Unknown",
}


# ------------------------------------------------------------------------------- QML probes
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


def _ssl_available() -> bool:
    """``import ssl`` works in this tree (the frozen exe must carry CPython's OpenSSL DLLs for any https)."""
    try:
        importlib.import_module("ssl")
    except ImportError:
        return False
    return True


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


def _resolve_page(value: str) -> Path:
    """Validate ``GLASSTRANSLATE_SMOKE_PAGE``: an existing image file of a known type and size.

    The messages name the file, never the directory it came from - the report is a persisted
    artefact and the full path is already in ``paths``.
    """
    if not value:
        raise ValueError(f"{SMOKE_PAGE_ENV} is not set")
    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"{SMOKE_PAGE_ENV} is not a file: {path.name}")
    if path.suffix.lower() not in PAGE_SUFFIXES:
        raise ValueError(f"{SMOKE_PAGE_ENV} is not an image: {path.name}")
    size = path.stat().st_size
    if size > PAGE_MAX_BYTES:
        raise ValueError(f"{SMOKE_PAGE_ENV} is {size} bytes, over the {PAGE_MAX_BYTES} byte cap")
    return path


class SmokeReport(QObject):
    """Writes the JSON report of docs/GLASS_DESIGN.md section 5 for automated runs.

    With an autoexit, the tab walk runs ``SMOKE_PAGES_LEAD_MS`` and the ``grabWindow()``
    screenshot ``SMOKE_GRAB_LEAD_MS`` before the quit, the latter only after at least one
    ``frameSwapped`` (a grab from ``aboutToQuit`` finds an unexposed window).  Without an
    autoexit - or if the timers never fired - the report is written from ``aboutToQuit``
    without a screenshot, so the file always exists.
    """

    _probe_done = Signal(object)  # sidecar probe record, from the probe thread to the GUI thread

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
        # Whitelisted UI actions driven from the environment (smoke run D drives the manga-ocr
        # download this way on a clean %LOCALAPPDATA%); the report carries their outcome.
        self._actions: Dict[str, Dict[str, Any]] = {}
        self._action_t0 = 0.0
        self._feed_t0 = 0.0
        self._feed_rechecks = 0
        self._pending_actions: List[str] = []
        self._running_actions: Set[str] = set()  # the quit waits for *every* requested action
        self._probe_settled = False
        self._probe_done.connect(self._on_probe_done)
        for name in (n.strip() for n in os.environ.get(SMOKE_ACTIONS_ENV, "").split(",")):
            if not name:
                continue
            if name in _SMOKE_ACTIONS:
                if name not in self._pending_actions:
                    self._pending_actions.append(name)
            else:
                log.warning("smoke: ignoring unknown action %r", name)
        if self._pending_actions:
            QTimer.singleShot(SMOKE_ACTION_DELAY_MS, self._run_actions)

    @Slot()
    def _run_actions(self) -> None:
        """Drive the whitelisted actions requested through ``GLASSTRANSLATE_SMOKE_ACTIONS``."""
        runners = {
            "downloadMangaOcr": self._action_download_manga_ocr,
            "probeSidecar": self._action_probe_sidecar,
            "feedPage": self._action_feed_page,
        }
        pending, self._pending_actions = self._pending_actions, []
        self._running_actions = set(pending)
        for name in pending:
            try:
                runners[name]()
            except Exception:  # pragma: no cover - an action must not take the report down
                log.exception("smoke: action %s failed", name)
                self._action_done(name)  # ...nor block the quit

    def _action_done(self, name: str) -> None:
        """One action reached its end, whatever its outcome."""
        self._running_actions.discard(name)
        self._maybe_finish()

    def _maybe_finish(self) -> None:
        """The one place that arms the report + quit: when *every* action has finished.

        Quitting on the first one would cut a second short - a run asking for
        ``probeSidecar,feedPage`` would end with the probe.
        """
        if not self._running_actions:
            QTimer.singleShot(SMOKE_ACTION_SETTLE_MS, self._finish_after_actions)

    # ------------------------------------------------------------------ downloadMangaOcr
    def _action_download_manga_ocr(self) -> None:
        """Engines page: fetch the manga-ocr bundle into the models dir (smoke run D)."""
        from ..ocr.models import models_ready

        bridge = self.session.control.bridge
        models_dir = str(bridge.modelsDir)
        self._actions["downloadMangaOcr"] = {
            "started": True, "finished": False, "seconds": None, "label": "", "progress": -1,
            "models_dir": models_dir, "models_ready_before": models_ready(models_dir),
            "models_ready_after": None,
        }
        self._action_t0 = time.perf_counter()
        bridge.downloadChanged.connect(self._on_action_download_changed)
        bridge.downloadMangaOcr()

    # ---------------------------------------------------------------------- probeSidecar
    def _action_probe_sidecar(self) -> None:
        """Start the sidecar the app itself would launch, in fake mode, and record ``/health``.

        The launch blocks until the sidecar prints ``READY``, so it runs on a worker thread and
        the record comes back through :attr:`_probe_done`; the GUI thread never waits.
        """
        from ..render.quality import find_sidecar_python

        self._actions["probeSidecar"] = {
            "launcher": None, "kind": None, "seconds": None, "health": None, "ok": False, "error": "",
        }
        # Armed before anything can finish: a sidecar that never prints READY would otherwise
        # hold the run until the autoexit and leave the record saying nothing.
        self._probe_settled = False
        QTimer.singleShot(PROBE_WATCHDOG_MS, self._on_probe_timeout)
        launcher = find_sidecar_python(self.session.cfg)
        if launcher is None:
            self._probe_done.emit({"launcher": None, "kind": None, "ok": False, "error": "no sidecar"})
            return
        models_dir = str(self.session.control.bridge.modelsDir)
        threading.Thread(
            target=self._probe_sidecar_worker, args=(Path(launcher), models_dir),
            name="glasstranslate-smoke-probe", daemon=True,
        ).start()

    def _probe_sidecar_worker(self, launcher: Path, models_dir: str) -> None:
        """Worker thread: start the fake sidecar, ask ``/health``, shut it down again."""
        from ..render.quality import QualityClient, sidecar_kind

        record: Dict[str, Any] = {
            "launcher": str(launcher), "kind": sidecar_kind(launcher),
            "seconds": None, "health": None, "ok": False, "error": "",
        }
        client = QualityClient(launcher, models_dir, fake=True, log_path=None,
                               ready_timeout=PROBE_READY_TIMEOUT_S)
        started = time.perf_counter()
        try:
            client.start()
            record["seconds"] = round(time.perf_counter() - started, 2)
            record["health"] = client.health()
            record["ok"] = True
        except Exception as exc:  # noqa: BLE001 - the outcome is the report
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                client.shutdown()
            except Exception:  # pragma: no cover - best effort
                log.exception("smoke: shutting the probed sidecar down failed")
        self._probe_done.emit(record)

    @Slot(object)
    def _on_probe_done(self, record: object) -> None:
        if self._probe_settled:
            return  # the watchdog already gave up on this probe; its verdict stands
        self._probe_settled = True
        if isinstance(record, dict):
            self._actions.setdefault("probeSidecar", {}).update(record)
        self._action_done("probeSidecar")

    @Slot()
    def _on_probe_timeout(self) -> None:
        """The probe thread is still inside ``start()``: record that and let the run finish."""
        if self._probe_settled:
            return
        self._probe_settled = True
        record = self._actions.setdefault("probeSidecar", {})
        record["ok"] = False
        record["error"] = f"probe did not finish within {PROBE_READY_TIMEOUT_S:.0f} s"
        self._action_done("probeSidecar")

    # -------------------------------------------------------------------------- feedPage
    def _action_feed_page(self) -> None:
        """Replace screen capture by the page named in ``GLASSTRANSLATE_SMOKE_PAGE``.

        The pipeline then OCRs, typesets and (with the quality renderer on) upgrades that page;
        :meth:`_on_feed_result` watches every result and finishes the run once an upgraded block
        reached the glass.
        """
        raw = os.environ.get(SMOKE_PAGE_ENV, "")
        record: Dict[str, Any] = {
            # Only the file name: the report is a persisted artefact and the directory it came
            # from says nothing about the run.
            "page": Path(raw).name if raw else "", "started": False, "finished": False,
            "seconds": None, "blocks": 0, "upgraded": 0, "overlay_upgrades": 0,
            "quality_stats": None, "status": "", "error": "",
        }
        self._actions["feedPage"] = record
        try:
            page = _resolve_page(raw)
            from ..capture.static import StaticPageCapture

            StaticPageCapture(page)  # fail here, where it is a record, not an engine error
        except Exception as exc:  # noqa: BLE001 - the outcome is the report
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            self._action_done("feedPage")
            return
        record["page"] = page.name

        def capture_factory(cfg: Any, path: Path = page) -> StaticPageCapture:
            return StaticPageCapture(path)

        session = self.session
        session._result_ready.connect(self._on_feed_result)
        session.capture_factory = capture_factory
        record["started"] = True
        record["status"] = "running"
        self._feed_t0 = time.perf_counter()
        pipeline = session.pipeline
        if pipeline is not None and pipeline.is_alive():
            # Never restart a running worker to change the source: stopping it cannot interrupt
            # an engine load, and a second pipeline loading its own onnxruntime-DirectML session
            # while the first is still loading crashes the process.  It swaps its own capture.
            pipeline.replace_capture(capture_factory)
        else:
            session.start_pipeline()  # picks the factory up at construction

    @Slot(object, object)
    def _on_feed_result(self, segments: object, stats: object) -> None:
        """Count the blocks of every pipeline result until one upgrade reached the overlay."""
        record = self._actions.get("feedPage")
        if record is None or not record["started"] or record["finished"]:
            return
        items = list(segments) if isinstance(segments, (list, tuple)) else []
        record["blocks"] = sum(1 for s in items if getattr(s.style, "is_block", False))
        record["upgraded"] = sum(1 for s in items if int(getattr(s.style, "clean_patch_serial", 0)) >= 1)
        if not self._feed_finished(record) and record["upgraded"] >= 1:
            # The serial advanced but the glass has not re-converted it *yet*: the overlay
            # applies segments one event-loop turn later, and a still page may never emit
            # another result to ask again.
            self._arm_feed_recheck()

    def _feed_finished(self, record: Dict[str, Any]) -> bool:
        """Re-read the overlay's counters and finish the action once an upgrade reached the glass."""
        record["overlay_upgrades"] = int(self.session.overlay.patch_stats["upgrades"])
        if record["upgraded"] < 1 or record["overlay_upgrades"] < 1:
            return False
        pipeline = self.session.pipeline
        record["finished"] = True
        record["seconds"] = round(time.perf_counter() - self._feed_t0, 1)
        record["quality_stats"] = pipeline.quality_snapshot() if pipeline is not None else None
        record["status"] = "upgraded"
        self._action_done("feedPage")
        return True

    def _arm_feed_recheck(self) -> None:
        """Look again shortly, a bounded number of times: a conversion that never happens must
        still end at the autoexit, with the counts of the last look recorded."""
        if self._feed_rechecks >= FEED_RECHECK_MAX:
            return
        self._feed_rechecks += 1
        QTimer.singleShot(FEED_RECHECK_MS, self._recheck_feed_page)

    @Slot()
    def _recheck_feed_page(self) -> None:
        record = self._actions.get("feedPage")
        if record is None or not record["started"] or record["finished"]:
            return
        if not self._feed_finished(record):
            self._arm_feed_recheck()

    @Slot()
    def _on_action_download_changed(self) -> None:
        from ..ocr.models import models_ready

        bridge = self.session.control.bridge
        record = self._actions.get("downloadMangaOcr")
        if record is None or record["finished"]:
            return
        record["label"] = str(bridge.downloadLabel)
        record["progress"] = int(bridge.downloadProgress)
        if not bridge.downloadActive:
            record["finished"] = True
            record["seconds"] = round(time.perf_counter() - self._action_t0, 1)
            record["models_ready_after"] = models_ready(record["models_dir"])
            self._action_done("downloadMangaOcr")

    @Slot()
    def _finish_after_actions(self) -> None:
        """Report and quit as soon as the actions are done instead of waiting for the autoexit."""
        self._grab_and_write()
        app = QCoreApplication.instance()
        if app is not None:
            app.quit()

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

    # --------------------------------------------------------------- always-present blocks
    def _guarded(self, name: str, build: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Never lose the whole report over one of its blocks."""
        try:
            return build()
        except Exception as exc:  # noqa: BLE001 - the failure is the field's value
            log.exception("smoke: building the %s block failed", name)
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _quality_block(self) -> Dict[str, Any]:
        """Quality renderer: the setting, the launcher the app would use, the models and the
        scheduler's live state (``None`` for the three scheduler fields without a pipeline)."""
        from ..render.quality import find_sidecar_python, sidecar_kind
        from ..render.quality_models import models_ready

        session = self.session
        cfg = session.cfg
        launcher = find_sidecar_python(cfg)
        pipeline = session.pipeline
        snapshot = pipeline.quality_snapshot() if pipeline is not None else None
        return {
            "setting": cfg.quality_renderer,
            "launcher": None if launcher is None else str(launcher),
            "kind": None if launcher is None else sidecar_kind(launcher),
            "models_ready": bool(models_ready(cfg.models_dir)),
            "hint": str(session.control.bridge.qualityStatus),
            "scheduler_active": None if snapshot is None else snapshot["active"],
            "scheduler_available": None if snapshot is None else snapshot["available"],
            "scheduler_stats": None if snapshot is None else snapshot["stats"],
        }

    def _overlay_block(self) -> Dict[str, Any]:
        stats = self.session.overlay.patch_stats
        return {"patch_conversions": stats["conversions"], "patch_upgrades": stats["upgrades"]}

    def _paths_block(self) -> Dict[str, Any]:
        """Where this session actually reads and writes (portable root vs. user profile)."""
        root = portable_root()
        return {
            "portable_root": None if root is None else str(root),
            "config": str(self.session.config_path),
            "models_dir": self.session.cfg.models_dir,
            "user_data_dir": str(user_data_dir()),
            "logs": str(logs_dir()),
            "cache": str(cache_dir()),
        }

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
            "ssl_ok": _ssl_available(),
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
            # last pipeline stats as shown on the status strip (proof that passes produced output)
            "stats": {name: str(getattr(bridge.stats, name, "")) for name in
                      ("totalText", "fpsText", "stagesText", "segmentsText", "cacheText", "devicesText")},
            "dpr": dpr,
            "window": {"x": geom.x(), "y": geom.y(), "w": geom.width(), "h": geom.height(), "dpr": dpr},
            "frames_swapped": self._frames,
            "quality": self._guarded("quality", self._quality_block),
            "overlay": self._guarded("overlay", self._overlay_block),
            "paths": self._guarded("paths", self._paths_block),
            "actions": self._actions,
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
