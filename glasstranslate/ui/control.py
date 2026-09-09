"""Settings / status window: the Liquid Glass control panel.

:class:`ControlWindow` is a frameless, per-pixel-translucent ``QQuickView`` that loads
``qrc:/qml/Main.qml`` and exposes two context properties to it: ``bridge``
(:class:`ControlBridge` - every setting, list, state flag and action of the panel) and
``appearance`` (:class:`~glasstranslate.ui.glass.appearance.Appearance` - Windows
transparency / reduced motion / high contrast / dark mode / text scale).  The public Python
contract is unchanged from the old QWidget window so ``app.py`` keeps working: signals
``config_changed(AppConfig)``, ``start_stop_requested(bool)``, ``grab_mode_requested()``,
``toggle_glass_requested()``, ``models_changed()``, ``closed()``; methods ``show()``,
``load_config()``, ``set_running()``, ``update_stats()``, ``show_status()``, ``save_soon()``,
``save_now()`` and the ``config`` property.  Edits made from QML mutate the
:class:`~glasstranslate.config.settings.AppConfig`, emit ``config_changed`` and persist with a
300 ms debounce.

What the glass refracts (docs/GLASS_DESIGN.md section 1.2): while the window is visible, not
minimised and glass is allowed, a :class:`~glasstranslate.ui.glass.backdrop.BackdropGrabber`
thread copies the desktop pixels behind the window at 15 Hz (the window itself is excluded
from capture with ``SetWindowDisplayAffinity``); the GUI thread stores each frame in the
``image://backdrop/<serial>`` provider and updates ``bridge.backdropSerial`` /
``backdropOrigin`` / ``backdropLuma`` / ``inkPolarity``.

Qt resources: this module imports the compiled ``resources_rc``; in a source checkout without
it (the file is generated, see ``tools/build_resources.py``) it builds the resources once.
"""
from __future__ import annotations

import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import (
    Property,
    QObject,
    QPointF,
    QSize,
    QSizeF,
    QThread,
    QTimer,
    QUrl,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QDesktopServices,
    QExposeEvent,
    QFontDatabase,
    QHideEvent,
    QIcon,
    QMoveEvent,
    QResizeEvent,
    QScreen,
    QShowEvent,
    QWindow,
)
from PySide6.QtQuick import QQuickView

from ..config.settings import AppConfig, project_root, user_data_dir
from ..core.types import PipelineStats
from .glass import win32
from .glass.appearance import Appearance
from .glass.backdrop import BackdropFrame, BackdropGrabber, BackdropProvider, InkPolarity, LumaSmoother, WindowGeometry
from .hotkeys import normalize_hotkey

__all__ = [
    "CUDA_FROZEN_HINT",
    "LANGUAGES",
    "MAIN_QML_URL",
    "WINDOW_TITLE",
    "ControlBridge",
    "ControlWindow",
    "ModelDownloadWorker",
    "StatsModel",
    "download_label",
    "download_progress",
    "format_stats",
    "is_frozen",
    "translate_device_items",
]

log = logging.getLogger(__name__)

# ISO-639-1 code -> display name, in menu order.
LANGUAGES: List[Tuple[str, str]] = [
    ("en", "English"),
    ("de", "German"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("ru", "Russian"),
    ("uk", "Ukrainian"),
    ("ja", "Japanese"),
    ("zh", "Chinese"),
    ("ko", "Korean"),
    ("ar", "Arabic"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("vi", "Vietnamese"),
    ("th", "Thai"),
]

_SAVE_DEBOUNCE_MS = 300
_OCR_DEVICES = ("auto", "gpu", "cpu")
_TRANSLATE_DEVICES = ("auto", "cuda", "cpu")
_TRANSLATE_DEVICES_FROZEN = ("auto", "cpu")
_ONLINE_BACKENDS = {"libretranslate"}
CUDA_FROZEN_HINT = "CUDA is not available in the packaged build; translation runs on CPU"

WINDOW_TITLE = "GlassTranslate"
MAIN_QML_URL = "qrc:/qml/Main.qml"
WINDOW_DEFAULT_SIZE = QSize(832, 640)  # slab 760x560 inside the 36/28/36/52 shadow margin; fixed-size, not resizable
WINDOW_FLAGS = (
    Qt.WindowType.Window
    | Qt.WindowType.FramelessWindowHint
    | Qt.WindowType.WindowMinimizeButtonHint
    | Qt.WindowType.WindowSystemMenuHint
)
# setHotkey(which, ...) accepts the short key, the bridge property name or the config field.
_HOTKEY_FIELDS: Dict[str, str] = {
    "grab": "toggle_grab",
    "running": "toggle_running",
    "hidden": "toggle_hidden",
    "hotkeyGrab": "toggle_grab",
    "hotkeyRunning": "toggle_running",
    "hotkeyHidden": "toggle_hidden",
    "toggle_grab": "toggle_grab",
    "toggle_running": "toggle_running",
    "toggle_hidden": "toggle_hidden",
}
_HOTKEY_PROPS: Dict[str, str] = {"toggle_grab": "hotkeyGrab", "toggle_running": "hotkeyRunning", "toggle_hidden": "hotkeyHidden"}


def is_frozen() -> bool:
    """True inside the PyInstaller exe."""
    return bool(getattr(sys, "frozen", False))


# ------------------------------------------------------------------------------- resources
def _ensure_resources() -> bool:
    """Import the compiled Qt resources (``:/qml``, ``:/fonts``, ``:/icons``).

    ``resources_rc.py`` is generated by ``tools/build_resources.py`` and not committed during
    development; in a source checkout it is built once here when missing, and rebuilt when
    present but stale (its digest no longer matches its QML/shader/font inputs) so an outdated
    file left over from before a redesign can never be served silently.  The frozen exe must ship
    a fresh one; there is no ``tools/`` there to rebuild from.
    """
    try:
        module = importlib.import_module("glasstranslate.ui.resources_rc")
        imported = True
    except ImportError:
        module = None
        imported = False

    if is_frozen():
        if imported:
            return True
        log.error("resources_rc is missing from the frozen bundle; QML, fonts and icon are unavailable")
        return False

    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from tools.build_resources import build, check_digest  # noqa: WPS433 - dev-only dependency
    except ImportError:
        # pip-installed layout without tools/: nothing to rebuild from, fall back to the import result.
        return imported

    if imported and check_digest()[0]:
        return True
    if imported:
        log.warning("resources_rc.py is stale (inputs changed since it was generated); rebuilding")
    try:
        out = build()
        importlib.invalidate_caches()
        if imported:
            # Unregister the old byte buffers before reload overwrites the module
            # globals that keep them alive; Qt would otherwise hold dangling data.
            module.qCleanupResources()
            importlib.reload(module)
        else:
            importlib.import_module("glasstranslate.ui.resources_rc")
        log.info("built Qt resources: %s", out)
        return True
    except Exception:
        log.exception("building the Qt resources failed; the control window cannot load its QML")
        return False


RESOURCES_OK = _ensure_resources()


# ------------------------------------------------------------------------------ pure helpers
def format_stats(stats: PipelineStats) -> Dict[str, str]:
    """The six readout strings, formatted exactly as the old ``update_stats`` did."""
    skipped = " (unchanged)" if stats.skipped_unchanged else ""
    blocks = stats.extra.get("blocks")
    blocks_txt = f", {blocks} blocks" if isinstance(blocks, int) else ""
    rate = stats.extra.get("cache_hit_rate")
    rate_txt = f"{rate * 100:.0f}%" if isinstance(rate, (int, float)) else "-"
    return {
        "total": f"{stats.total_ms:.1f} ms",
        "fps": f"{stats.fps:.1f}",
        "stages": (
            f"capture {stats.capture_ms:.1f} | diff {stats.diff_ms:.1f} | ocr {stats.ocr_ms:.1f} | "
            f"style {stats.style_ms:.1f} | translate {stats.translate_ms:.1f}"
        ),
        "segments": f"{stats.segments} live{blocks_txt}, {stats.dirty_regions} dirty{skipped}",
        "cache": f"{stats.cache_hits} hit / {stats.cache_misses} miss, {rate_txt} overall",
        "devices": (
            f"ocr: {stats.extra.get('ocr', '?')}@{stats.ocr_device or '?'}   "
            f"translate: {stats.extra.get('translator', '?')}@{stats.translate_device or '?'}   "
            f"capture: {stats.extra.get('capture', '?')}   src: {stats.extra.get('src_lang', '-')}"
        ),
    }


def download_label(sugoi: bool, src: str, tgt: str) -> str:
    """``"Sugoi v4 ja → en"`` or ``"<src> → <tgt>"``."""
    return "Sugoi v4 ja → en" if sugoi else f"{src} → {tgt}"


def download_progress(label: str, done: int, total: int) -> Tuple[str, int]:
    """Progress row text and percentage (-1 = indeterminate) for a download callback."""
    if total > 0:
        return f"Downloading {label}: {done / 1e6:.1f} / {total / 1e6:.1f} MB", int(done * 100 // total)
    return f"Downloading {label}: {done / 1e6:.1f} MB", -1


def _items(pairs: Sequence[Tuple[str, str]], current: str) -> List[Dict[str, str]]:
    """``{value, text}`` entries; an unknown ``current`` is appended (the old ``_select_data`` rule)."""
    items = [{"value": value, "text": text} for value, text in pairs]
    if current not in {value for value, _ in pairs}:
        items.append({"value": current, "text": current})
    return items


def translate_device_items(current: str, frozen: Optional[bool] = None) -> List[Dict[str, str]]:
    """``auto, cuda, cpu`` in a source checkout; ``auto, cpu`` when frozen (the exe prunes the CUDA
    libraries).  A stored ``cuda`` is still appended by the unknown-value rule, carrying a ``hint``."""
    frozen = is_frozen() if frozen is None else frozen
    base = _TRANSLATE_DEVICES_FROZEN if frozen else _TRANSLATE_DEVICES
    items = _items([(d, d) for d in base], current)
    if frozen and current == "cuda":
        items[-1]["hint"] = CUDA_FROZEN_HINT
    return items


# ------------------------------------------------------------------------- download worker
class ModelDownloadWorker(QThread):
    """Downloads one Argos package in the background.

    Emits :attr:`progress` (done, total_or_-1) while downloading, then either
    :attr:`finished_ok` with the extracted directory or :attr:`failed`.
    """

    progress = Signal(int, int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        models_dir: str,
        from_code: str,
        to_code: str,
        parent: Optional[QObject] = None,
        *,
        sugoi: bool = False,
    ) -> None:
        super().__init__(parent)
        self._models_dir = models_dir
        self._from = from_code
        self._to = to_code
        self._sugoi = sugoi  # install the Sugoi v4 ja->en model instead of an Argos package

    def run(self) -> None:  # noqa: D401 - QThread API
        try:
            from ..translate.argos import ArgosCT2Translator

            translator = ArgosCT2Translator(self._models_dir, device="cpu")

            def cb(done: int, total: Optional[int]) -> None:
                self.progress.emit(int(done), int(total) if total else -1)

            if self._sugoi:
                path = translator.download_sugoi(progress_cb=cb)
            else:
                path = translator.download_package(self._from, self._to, progress_cb=cb)
            self.finished_ok.emit(str(path))
        except Exception as exc:
            log.exception("model download failed")
            self.failed.emit(str(exc))


# ------------------------------------------------------------------------------ stats model
class StatsModel(QObject):
    """Live latency readout for the status strip (``bridge.stats``).

    ``totalText`` / ``fpsText`` feed the strip, the four detail lines the drawer; every value is
    ``"-"`` until the first pipeline pass.
    """

    changed = Signal()

    _KEYS = ("total", "fps", "stages", "segments", "cache", "devices")

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._values: Dict[str, str] = {k: "-" for k in self._KEYS}

    def update(self, stats: PipelineStats) -> None:
        """Refresh every line from a pipeline pass."""
        self._values = format_stats(stats)
        self.changed.emit()

    def reset(self) -> None:
        self._values = {k: "-" for k in self._KEYS}
        self.changed.emit()

    def _text(self, key: str) -> str:
        return self._values[key]

    totalText = Property(str, lambda self: self._text("total"), notify=changed)
    fpsText = Property(str, lambda self: self._text("fps"), notify=changed)
    stagesText = Property(str, lambda self: self._text("stages"), notify=changed)
    segmentsText = Property(str, lambda self: self._text("segments"), notify=changed)
    cacheText = Property(str, lambda self: self._text("cache"), notify=changed)
    devicesText = Property(str, lambda self: self._text("devices"), notify=changed)


# ------------------------------------------------------------------------- config plumbing
def _clamp(lo: float, hi: float) -> Callable[[Any], float]:
    return lambda v: min(hi, max(lo, float(v)))


def _strip(v: Any) -> str:
    return str(v).strip()


@dataclass(frozen=True)
class _Field:
    """One config property: the ``AppConfig`` attribute (dotted for ``hotkeys.x``) and its coercion."""

    attr: str
    coerce: Callable[[Any], Any]


_CONFIG_FIELDS: Dict[str, _Field] = {
    "sourceLang": _Field("source_lang", str),
    "targetLang": _Field("target_lang", str),
    "ocrEngine": _Field("ocr_engine", str),
    "ocrDevice": _Field("ocr_device", str),
    "translationBackend": _Field("translation_backend", str),
    "translateDevice": _Field("translate_device", str),
    "apiUrl": _Field("translation_api_url", _strip),
    "apiKey": _Field("translation_api_key", _strip),
    "modelsDir": _Field("models_dir", _strip),
    "overlayOpacity": _Field("overlay_opacity", _clamp(0.0, 1.0)),
    "fontFamily": _Field("font_family", str),
    "hideOriginal": _Field("hide_original", bool),
    "mangaMode": _Field("manga_mode", bool),
    "uppercase": _Field("uppercase", bool),
    "refreshHz": _Field("refresh_hz", _clamp(0.5, 60.0)),
    "debounceMs": _Field("debounce_ms", lambda v: int(min(2000, max(0, round(float(v)))))),
    "minConfidence": _Field("min_confidence", _clamp(0.0, 1.0)),
    "hotkeyGrab": _Field("hotkeys.toggle_grab", str),
    "hotkeyRunning": _Field("hotkeys.toggle_running", str),
    "hotkeyHidden": _Field("hotkeys.toggle_hidden", str),
}


def _cfg_get(cfg: AppConfig, attr: str) -> Any:
    obj: Any = cfg
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _cfg_set(cfg: AppConfig, attr: str, value: Any) -> None:
    parts = attr.split(".")
    obj: Any = cfg
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _config_property(name: str, qtype: type, notify: Signal) -> Property:
    """A read/write QML property backed by ``AppConfig`` through :meth:`ControlBridge._set_config`."""
    attr = _CONFIG_FIELDS[name].attr

    def fget(self: "ControlBridge") -> Any:
        return _cfg_get(self._cfg, attr)

    def fset(self: "ControlBridge", value: Any) -> None:
        self._set_config(name, value)

    return Property(qtype, fget, fset, notify=notify)


# ------------------------------------------------------------------------------------ bridge
class ControlBridge(QObject):
    """Everything ``Main.qml`` reads and calls (context property ``bridge``).

    Config properties are read/write: a set coerces the value, mutates the ``AppConfig``, emits
    the property's notify signal and ``config_changed(cfg)`` and restarts the 300 ms save timer;
    sets are ignored while :meth:`load_config` runs and when the value is unchanged.  Lists are
    ``{value, text}`` entries (an unknown current value is appended).  State (running, status,
    stats, download row, backdrop, ink polarity) is read-only from QML and driven from Python.
    ``window`` (the :class:`ControlWindow`) is optional so the bridge can be tested headless.
    """

    # -- to the application (relayed by ControlWindow)
    config_changed = Signal(object)  # AppConfig
    start_stop_requested = Signal(bool)  # True = start
    grab_mode_requested = Signal()
    toggle_glass_requested = Signal()
    models_changed = Signal()  # a package was downloaded

    # -- QML notifies
    sourceLangChanged = Signal()
    targetLangChanged = Signal()
    ocrEngineChanged = Signal()
    ocrDeviceChanged = Signal()
    translationBackendChanged = Signal()
    translateDeviceChanged = Signal()
    apiUrlChanged = Signal()
    apiKeyChanged = Signal()
    modelsDirChanged = Signal()
    overlayOpacityChanged = Signal()
    fontFamilyChanged = Signal()
    hideOriginalChanged = Signal()
    mangaModeChanged = Signal()
    uppercaseChanged = Signal()
    refreshHzChanged = Signal()
    debounceMsChanged = Signal()
    minConfidenceChanged = Signal()
    hotkeyGrabChanged = Signal()
    hotkeyRunningChanged = Signal()
    hotkeyHiddenChanged = Signal()
    listsChanged = Signal()
    backendFlagsChanged = Signal()
    loadingChanged = Signal()
    runningChanged = Signal()
    statusMessageChanged = Signal()
    downloadChanged = Signal()
    backdropChanged = Signal()
    inkPolarityChanged = Signal()

    def __init__(
        self,
        cfg: AppConfig,
        config_path: Optional[Path] = None,
        *,
        window: Optional[QWindow] = None,
        appearance: Optional[Appearance] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        from ..ocr.factory import available_engines
        from ..translate.factory import available_backends

        self._cfg = cfg
        self._config_path = config_path
        self._window = window
        self._appearance = appearance
        self._loading = False
        self._running = False
        self._status = "Ready"
        self.status_history: List[str] = []
        self._stats = StatsModel(self)
        self._engines: List[str] = list(available_engines())
        self._backends: List[str] = list(available_backends())
        self._families: Optional[List[str]] = None
        # download row
        self._download_worker: Optional[ModelDownloadWorker] = None
        self._download_visible = False
        self._download_label = ""
        self._download_progress = -1
        self._download_base_label = ""
        # backdrop
        self._backdrop_serial = 0
        self._backdrop_origin = QPointF(0.0, 0.0)
        self._backdrop_size = QSizeF(0.0, 0.0)
        self._luma = LumaSmoother()
        self._polarity = InkPolarity(self._dark_mode())
        # persistence
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(_SAVE_DEBOUNCE_MS)
        self._save_timer.timeout.connect(self.save_now)
        if appearance is not None:
            appearance.changed.connect(self.appearance_changed)

    # ------------------------------------------------------------------ python API
    @property
    def config(self) -> AppConfig:
        return self._cfg

    @property
    def window(self) -> Optional[QWindow]:
        return self._window

    def load_config(self, cfg: AppConfig) -> None:
        """Adopt ``cfg`` and refresh every binding without emitting ``config_changed``."""
        self._cfg = cfg
        self._loading = True
        self.loadingChanged.emit()
        try:
            for name in _CONFIG_FIELDS:
                getattr(self, name + "Changed").emit()
            self.listsChanged.emit()
            self.backendFlagsChanged.emit()
        finally:
            self._loading = False
            self.loadingChanged.emit()

    def set_running(self, running: bool) -> None:
        running = bool(running)
        if running != self._running:
            self._running = running
            self.runningChanged.emit()

    def show_status(self, message: str) -> None:
        """Set the status-strip message (kept in ``status_history`` for the smoke report)."""
        message = str(message)
        self.status_history.append(message)
        if message != self._status:
            self._status = message
            self.statusMessageChanged.emit()
        else:
            self.statusMessageChanged.emit()  # the same text again still refreshes the strip

    def save_soon(self) -> None:
        """Schedule a debounced save (changes made outside the window, e.g. the glass geometry)."""
        self._save_timer.start()

    @Slot()
    def save_now(self) -> None:
        """Persist the config immediately (normally driven by the debounce timer)."""
        self._save_timer.stop()
        try:
            self._cfg.save(self._config_path)
        except Exception as exc:
            log.exception("saving config failed")
            self.show_status(f"Could not save config: {exc}")

    def flush_pending_save(self) -> None:
        if self._save_timer.isActive():
            self.save_now()

    def push_backdrop(self, frame: BackdropFrame) -> None:
        """GUI-thread entry for a grabbed frame: serial, origin, smoothed luma, polarity."""
        ox, oy = frame.origin_logical
        self._backdrop_serial += 1
        self._backdrop_origin = QPointF(ox, oy)
        dpr = frame.dpr if frame.dpr > 0 else 1.0
        self._backdrop_size = QSizeF(frame.image.width() / dpr, frame.image.height() / dpr)
        self._luma.update(frame.luma, frame.timestamp)
        self.backdropChanged.emit()
        if self._glass_mode() and self._polarity.update(self._backdrop_luma()):
            self.inkPolarityChanged.emit()

    @Slot()
    def appearance_changed(self) -> None:
        """Outside glass mode (or before the first frame) the polarity follows the system theme."""
        if not self._glass_mode() or self._backdrop_serial == 0:
            if self._polarity.reset(self._dark_mode()):
                self.inkPolarityChanged.emit()

    # ------------------------------------------------------------------- internals
    def _dark_mode(self) -> bool:
        return bool(self._appearance.darkMode) if self._appearance is not None else False

    def _glass_mode(self) -> bool:
        return bool(self._appearance.glassAllowed) if self._appearance is not None else True

    def _backdrop_luma(self) -> float:
        return float(self._luma.value) if self._luma.value is not None else 0.5

    def _config_edited(self) -> None:
        self.config_changed.emit(self._cfg)
        self._save_timer.start()

    def _base_values(self, name: str) -> Optional[Sequence[str]]:
        """Known values of a list-backed property (None for free-form fields)."""
        if name == "sourceLang":
            return ["auto"] + [code for code, _ in LANGUAGES]
        if name == "targetLang":
            return [code for code, _ in LANGUAGES]
        if name == "ocrEngine":
            return self._engines
        if name == "ocrDevice":
            return _OCR_DEVICES
        if name == "translationBackend":
            return self._backends
        if name == "translateDevice":
            return _TRANSLATE_DEVICES_FROZEN if is_frozen() else _TRANSLATE_DEVICES
        if name == "fontFamily":
            return self._font_families_base()
        return None

    def _set_config(self, name: str, value: Any) -> None:
        if self._loading:
            return
        spec = _CONFIG_FIELDS[name]
        try:
            value = spec.coerce(value)
        except (TypeError, ValueError):
            log.warning("ignoring invalid value %r for %s", value, name)
            return
        if name == "modelsDir" and not value:
            value = self._cfg.models_dir  # an empty models dir keeps the previous value
        if _cfg_get(self._cfg, spec.attr) == value:
            return
        _cfg_set(self._cfg, spec.attr, value)
        getattr(self, name + "Changed").emit()
        if name == "translationBackend":
            self.backendFlagsChanged.emit()
        base = self._base_values(name)
        if base is not None and value not in base:
            self.listsChanged.emit()
        self._config_edited()

    def _font_families_base(self) -> List[str]:
        if self._families is None:
            try:
                self._families = [str(f) for f in QFontDatabase.families()]
            except Exception:  # pragma: no cover - no QGuiApplication yet
                self._families = []
        return self._families

    # ------------------------------------------------------------ config properties
    sourceLang = _config_property("sourceLang", str, sourceLangChanged)
    targetLang = _config_property("targetLang", str, targetLangChanged)
    ocrEngine = _config_property("ocrEngine", str, ocrEngineChanged)
    ocrDevice = _config_property("ocrDevice", str, ocrDeviceChanged)
    translationBackend = _config_property("translationBackend", str, translationBackendChanged)
    translateDevice = _config_property("translateDevice", str, translateDeviceChanged)
    apiUrl = _config_property("apiUrl", str, apiUrlChanged)
    apiKey = _config_property("apiKey", str, apiKeyChanged)
    modelsDir = _config_property("modelsDir", str, modelsDirChanged)
    overlayOpacity = _config_property("overlayOpacity", float, overlayOpacityChanged)
    fontFamily = _config_property("fontFamily", str, fontFamilyChanged)
    hideOriginal = _config_property("hideOriginal", bool, hideOriginalChanged)
    mangaMode = _config_property("mangaMode", bool, mangaModeChanged)
    uppercase = _config_property("uppercase", bool, uppercaseChanged)
    refreshHz = _config_property("refreshHz", float, refreshHzChanged)
    debounceMs = _config_property("debounceMs", int, debounceMsChanged)
    minConfidence = _config_property("minConfidence", float, minConfidenceChanged)
    hotkeyGrab = _config_property("hotkeyGrab", str, hotkeyGrabChanged)
    hotkeyRunning = _config_property("hotkeyRunning", str, hotkeyRunningChanged)
    hotkeyHidden = _config_property("hotkeyHidden", str, hotkeyHiddenChanged)

    # ------------------------------------------------------------------------ lists
    def _languages_source(self) -> List[Dict[str, str]]:
        pairs = [("auto", "Auto-detect")] + [(code, f"{name} ({code})") for code, name in LANGUAGES]
        return _items(pairs, self._cfg.source_lang)

    def _languages_target(self) -> List[Dict[str, str]]:
        return _items([(code, f"{name} ({code})") for code, name in LANGUAGES], self._cfg.target_lang)

    def _ocr_engines(self) -> List[Dict[str, str]]:
        return _items([(n, n) for n in self._engines], self._cfg.ocr_engine)

    def _ocr_devices(self) -> List[Dict[str, str]]:
        return _items([(d, d) for d in _OCR_DEVICES], self._cfg.ocr_device)

    def _backend_items(self) -> List[Dict[str, str]]:
        return _items([(n, n) for n in self._backends], self._cfg.translation_backend)

    def _translate_devices(self) -> List[Dict[str, str]]:
        return translate_device_items(self._cfg.translate_device)

    def _font_families(self) -> List[str]:
        families = list(self._font_families_base())
        if self._cfg.font_family not in families:
            families.append(self._cfg.font_family)
        return families

    languagesSource = Property("QVariantList", _languages_source, notify=listsChanged)
    languagesTarget = Property("QVariantList", _languages_target, notify=listsChanged)
    ocrEngines = Property("QVariantList", _ocr_engines, notify=listsChanged)
    ocrDevices = Property("QVariantList", _ocr_devices, notify=listsChanged)
    backends = Property("QVariantList", _backend_items, notify=listsChanged)
    translateDevices = Property("QVariantList", _translate_devices, notify=listsChanged)
    fontFamilies = Property("QVariantList", _font_families, notify=listsChanged)

    # ----------------------------------------------------------------- derived flags
    def _backend_online(self) -> bool:
        return self._cfg.translation_backend in _ONLINE_BACKENDS

    def _backend_argos(self) -> bool:
        return self._cfg.translation_backend == "argos"

    def _translate_device_hint(self) -> str:
        return CUDA_FROZEN_HINT if is_frozen() and self._cfg.translate_device == "cuda" else ""

    backendOnline = Property(bool, _backend_online, notify=backendFlagsChanged)
    backendArgos = Property(bool, _backend_argos, notify=backendFlagsChanged)
    translateDeviceHint = Property(str, _translate_device_hint, notify=translateDeviceChanged)
    frozen = Property(bool, lambda self: is_frozen(), constant=True)
    loading = Property(bool, lambda self: self._loading, notify=loadingChanged)

    # ------------------------------------------------------------------------ state
    running = Property(bool, lambda self: self._running, notify=runningChanged)
    statusMessage = Property(str, lambda self: self._status, notify=statusMessageChanged)
    stats = Property(QObject, lambda self: self._stats, constant=True)
    windowTitle = Property(str, lambda self: WINDOW_TITLE, constant=True)

    def _download_active(self) -> bool:
        return self._download_worker is not None and self._download_worker.isRunning()

    downloadActive = Property(bool, _download_active, notify=downloadChanged)
    downloadVisible = Property(bool, lambda self: self._download_visible, notify=downloadChanged)
    downloadLabel = Property(str, lambda self: self._download_label, notify=downloadChanged)
    downloadProgress = Property(int, lambda self: self._download_progress, notify=downloadChanged)

    backdropSerial = Property(int, lambda self: self._backdrop_serial, notify=backdropChanged)
    backdropOrigin = Property(QPointF, lambda self: QPointF(self._backdrop_origin), notify=backdropChanged)
    # Logical size of the current backdrop image.  Qt Quick ignores QImage.devicePixelRatio for
    # image-provider images (Image.implicitWidth reports the *physical* width), so Main.qml sizes
    # the backdrop layer from this instead - otherwise the texture is magnified by the DPR at 150 %.
    backdropSize = Property(QSizeF, lambda self: QSizeF(self._backdrop_size), notify=backdropChanged)
    backdropLuma = Property(float, _backdrop_luma, notify=backdropChanged)
    inkPolarity = Property(int, lambda self: self._polarity.value, notify=inkPolarityChanged)

    # ------------------------------------------------------------------------ slots
    @Slot(bool)
    def startStop(self, start: bool) -> None:
        self.start_stop_requested.emit(bool(start))

    @Slot()
    def grabMode(self) -> None:
        self.grab_mode_requested.emit()

    @Slot()
    def toggleGlass(self) -> None:
        self.toggle_glass_requested.emit()

    @Slot()
    def browseModelsDir(self) -> None:
        """Native folder picker (no parent window: ours is a QWindow, not a QWidget)."""
        from PySide6.QtWidgets import QFileDialog

        chosen = QFileDialog.getExistingDirectory(None, "Models directory", self._cfg.models_dir)
        if chosen:
            self._set_config("modelsDir", chosen)

    @Slot()
    @Slot(bool)
    def downloadModel(self, sugoi: bool = False) -> None:
        """Download the Argos package for the selected pair (via English pivot when the source is
        "auto": we download en -> target), or the Sugoi ja -> en model when ``sugoi`` is set."""
        if self._download_active():
            self.show_status("A download is already running.")
            return
        if sugoi:
            src, tgt = "ja", "en"
        else:
            src, tgt = self._cfg.source_lang, self._cfg.target_lang
            if src == "auto":
                src = "en"
            if src == tgt:
                self.show_status("Source and target language are the same.")
                return
        label = download_label(bool(sugoi), src, tgt)
        worker = ModelDownloadWorker(self._cfg.models_dir.strip() or self._cfg.models_dir, src, tgt, self, sugoi=bool(sugoi))
        self._download_worker = worker
        self._download_base_label = label
        self._download_label = f"Downloading {label} model…"
        self._download_progress = -1
        self._download_visible = True
        worker.progress.connect(self._on_download_progress)
        worker.finished_ok.connect(self._on_download_ok)
        worker.failed.connect(self._on_download_failed)
        worker.start()
        self.downloadChanged.emit()

    @Slot()
    def hideDownload(self) -> None:
        """Hide the progress row only; the download keeps running and reports to the status strip."""
        if self._download_visible:
            self._download_visible = False
            self.downloadChanged.emit()

    @Slot(int, int)
    def _on_download_progress(self, done: int, total: int) -> None:
        self._download_label, self._download_progress = download_progress(self._download_base_label, done, total)
        self.downloadChanged.emit()

    @Slot(str)
    def _on_download_ok(self, path: str) -> None:
        self._download_visible = False
        self.downloadChanged.emit()
        self.show_status(f"Installed model at {path}")
        self.models_changed.emit()

    @Slot(str)
    def _on_download_failed(self, message: str) -> None:
        self._download_visible = False
        self.downloadChanged.emit()
        self.show_status(f"Download failed: {message}")

    @Slot(str, str, result=bool)
    def setHotkey(self, which: str, text: str) -> bool:
        """Commit a hotkey field; ``which`` is ``"grab"`` | ``"running"`` | ``"hidden"``.

        Returns True when the (normalized) hotkey was accepted - the field should then display
        the property value.  Empty text returns False silently (the field reverts, no status);
        invalid text returns False with ``Invalid hotkey ...`` in the status strip.
        """
        field = _HOTKEY_FIELDS.get(str(which))
        if field is None:
            self.show_status(f"Unknown hotkey {which!r}")
            return False
        prop = _HOTKEY_PROPS[field]
        text = str(text).strip()
        if not text:
            return False
        try:
            normalized = normalize_hotkey(text)
        except Exception as exc:
            self.show_status(f"Invalid hotkey {text!r}: {exc}")
            return False
        if getattr(self._cfg.hotkeys, field) != normalized:
            setattr(self._cfg.hotkeys, field, normalized)
            self._config_edited()
        getattr(self, prop + "Changed").emit()
        return True

    @Slot()
    def startMove(self) -> None:
        """System move of the window; only valid from a QML ``onPressed`` handler."""
        if self._window is not None:
            self._window.startSystemMove()

    @Slot(int)
    def startResize(self, edges: int) -> None:
        """No-op: the window is fixed-size.  Kept so old/cached QML never crashes calling it."""
        return

    @Slot()
    def keyboardMove(self) -> None:
        """Alt+Space menu "Move": DefWindowProc's keyboard move loop (arrows, Enter / Esc)."""
        if self._window is not None:
            win32.keyboard_move(self._window)

    @Slot()
    def keyboardSize(self) -> None:
        """No-op: the window is fixed-size, so there is no keyboard size loop to start."""
        return

    @Slot()
    def minimize(self) -> None:
        if self._window is not None:
            self._window.showMinimized()

    @Slot()
    def close(self) -> None:
        """Close the window (``closeEvent`` flushes the pending save and emits ``closed``)."""
        if self._window is not None:
            self._window.close()

    @Slot()
    def openLogs(self) -> None:
        """Open the log folder of the packaged build in Explorer."""
        folder = user_data_dir() / "logs"
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            self.show_status(f"No log folder yet ({folder})")


# ------------------------------------------------------------------------------------ window
class ControlWindow(QQuickView):
    """Main settings window; see the module docstring."""

    config_changed = Signal(object)  # AppConfig
    start_stop_requested = Signal(bool)  # True = start
    grab_mode_requested = Signal()
    toggle_glass_requested = Signal()
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
                     "models_changed"):
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
        win32.exclude_from_capture(self)
        self._refresh_geometry()
        self._update_grabber()

    def hideEvent(self, event: QHideEvent) -> None:
        super().hideEvent(event)
        self._update_grabber()

    def exposeEvent(self, event: QExposeEvent) -> None:
        super().exposeEvent(event)
        if self.isExposed():
            self._refresh_geometry()
            self._poke()

    def moveEvent(self, event: QMoveEvent) -> None:
        super().moveEvent(event)
        self._refresh_geometry()
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

    def _update_grabber(self) -> None:
        minimized = bool(self.windowStates() & Qt.WindowState.WindowMinimized)
        active = self.isVisible() and not minimized and self.appearance.glassAllowed
        if active:
            if self._grabber is None:
                self._grabber = BackdropGrabber(self._geometry, self._frame_ready.emit)
                self._grabber.start()
            self._grabber.resume()
        elif self._grabber is not None:
            self._grabber.pause()

    def _poke(self) -> None:
        if self._grabber is not None and not self._grabber.paused:
            self._grabber.poke()

    @Slot(object)
    def _on_backdrop_frame(self, frame: object) -> None:
        if not isinstance(frame, BackdropFrame):
            return
        self._provider.set_image(frame.image)
        self.bridge.push_backdrop(frame)

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
