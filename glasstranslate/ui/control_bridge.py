"""``ControlBridge`` - the QML-facing state object of the control window.

Split out of ``ui/control.py`` (which still re-exports every name here); see that module's
docstring for the window/bridge/backdrop design.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Property, QObject, QPointF, QSizeF, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QFontDatabase, QWindow

from ..config.settings import AppConfig, user_data_dir
from .bridge_fields import (
    _CONFIG_FIELDS,
    _ONLINE_BACKENDS,
    _OCR_DEVICES,
    _TRANSLATE_DEVICES,
    _TRANSLATE_DEVICES_FROZEN,
    CUDA_FROZEN_HINT,
    LANGUAGES,
    QUALITY_RENDERER_ITEMS,
    _cfg_get,
    _cfg_set,
    _config_property,
    _items,
    download_label,
    download_progress,
    is_frozen,
    translate_device_items,
)
from .download_workers import MangaOcrDownloadWorker, ModelDownloadWorker, QualityModelsDownloadWorker
from .glass import win32
from .glass.appearance import Appearance
from .glass.backdrop import BackdropFrame, InkPolarity, LumaSmoother
from .hotkeys import normalize_hotkey
from .stats_model import StatsModel

__all__ = ["ControlBridge", "WINDOW_TITLE"]

log = logging.getLogger(__name__)

_SAVE_DEBOUNCE_MS = 300
WINDOW_TITLE = "GlassTranslate"
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
    capture_mode_requested = Signal(bool)  # capture mode on/off (runtime only, F1; the overlay and the control window)
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
    qualityRendererChanged = Signal()
    qualityChanged = Signal()  # the quality card's derived hint / models-ready flag
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
    geminiModelChanged = Signal()
    geminiApiFormatChanged = Signal()
    geminiBaseUrlChanged = Signal()
    geminiTimeoutSChanged = Signal()
    geminiMaxRetriesChanged = Signal()
    geminiApiKeyChanged = Signal()  # set/not-set only; the key itself never crosses the bridge
    seriesNameChanged = Signal()
    seriesPromptTemplateChanged = Signal()
    listsChanged = Signal()
    backendFlagsChanged = Signal()
    loadingChanged = Signal()
    runningChanged = Signal()
    overlayCaptureModeChanged = Signal()
    statusMessageChanged = Signal()
    downloadChanged = Signal()
    backdropChanged = Signal()
    backdropShiftChanged = Signal()
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
        self._capture_mode = False  # runtime only: never read from / written to AppConfig
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
        # F2-1: physical origin of the window the current frame was grabbed for, and of the
        # window now (None until the first moveEvent).  Their difference is backdropShift.
        self._frame_window_origin: Tuple[int, int] = (0, 0)
        self._window_origin: Optional[Tuple[int, int]] = None
        self._backdrop_dpr = 1.0
        self._backdrop_shift = QPointF(0.0, 0.0)
        self._luma = LumaSmoother()
        self._polarity = InkPolarity(self._dark_mode())
        # persistence
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(_SAVE_DEBOUNCE_MS)
        self._save_timer.timeout.connect(self.save_now)
        # The quality card's hint follows the download row (the models may have just landed).
        self.downloadChanged.connect(self.qualityChanged)
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
            self.qualityChanged.emit()
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
        self._backdrop_dpr = dpr
        self._backdrop_size = QSizeF(frame.image.width() / dpr, frame.image.height() / dpr)
        self._frame_window_origin = (frame.window_rect.x, frame.window_rect.y)
        self._luma.update(frame.luma, frame.timestamp)
        self.backdropChanged.emit()
        self._update_backdrop_shift()
        if self._glass_mode() and self._polarity.update(self._backdrop_luma()):
            self.inkPolarityChanged.emit()

    def set_window_origin(self, x: int, y: int) -> None:
        """GUI-thread entry from ``moveEvent``: physical top-left of the window right now."""
        self._window_origin = (int(x), int(y))
        self._update_backdrop_shift()

    def _update_backdrop_shift(self) -> None:
        """Logical offset that keeps a frame grabbed for an older window rect desktop-aligned."""
        if self._window_origin is None or self._backdrop_serial == 0:
            shift = QPointF(0.0, 0.0)
        else:
            fx, fy = self._frame_window_origin
            wx, wy = self._window_origin
            shift = QPointF((fx - wx) / self._backdrop_dpr, (fy - wy) / self._backdrop_dpr)
        if shift != self._backdrop_shift:
            self._backdrop_shift = shift
            self.backdropShiftChanged.emit()

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
        if name == "qualityRenderer":
            return [value for value, _ in QUALITY_RENDERER_ITEMS]
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
        if name in ("qualityRenderer", "modelsDir"):
            self.qualityChanged.emit()
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
    qualityRenderer = _config_property("qualityRenderer", str, qualityRendererChanged)
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
    geminiModel = _config_property("geminiModel", str, geminiModelChanged)
    geminiApiFormat = _config_property("geminiApiFormat", str, geminiApiFormatChanged)
    geminiBaseUrl = _config_property("geminiBaseUrl", str, geminiBaseUrlChanged)
    geminiTimeoutS = _config_property("geminiTimeoutS", float, geminiTimeoutSChanged)
    geminiMaxRetries = _config_property("geminiMaxRetries", int, geminiMaxRetriesChanged)
    seriesName = _config_property("seriesName", str, seriesNameChanged)
    seriesPromptTemplate = _config_property("seriesPromptTemplate", str, seriesPromptTemplateChanged)

    # ------------------------------------------------------------ Gemini API key (secret store)
    def _gemini_api_key_set(self) -> bool:
        from ..config.secrets import has_gemini_api_key

        return has_gemini_api_key()

    def _gemini_api_key_hint(self) -> str:
        from ..config.secrets import GEMINI_API_KEY_ENV, get_gemini_api_key, redact

        if os.environ.get(GEMINI_API_KEY_ENV, "").strip():
            return f"Using the {GEMINI_API_KEY_ENV} environment variable"
        key = get_gemini_api_key()
        return f"Key stored ({redact(key)})" if key else "No key stored"

    geminiApiKeySet = Property(bool, _gemini_api_key_set, notify=geminiApiKeyChanged)
    geminiApiKeyHint = Property(str, _gemini_api_key_hint, notify=geminiApiKeyChanged)

    @Slot(str, result=bool)
    def setGeminiApiKey(self, text: str) -> bool:
        """Store (or, with empty text, clear) the Gemini API key in the user secret store.

        The key never enters ``AppConfig``: it is written to ``secrets.json`` under the user data
        directory and only ``geminiApiKeySet`` / a redacted hint are readable from QML.
        """
        from ..config.secrets import set_gemini_api_key

        try:
            set_gemini_api_key(str(text))
        except OSError as exc:
            log.warning("storing the Gemini API key failed: %s", exc.__class__.__name__)
            self.show_status("Could not store the Gemini API key (see log)")
            return False
        self.geminiApiKeyChanged.emit()
        self.show_status("Gemini API key stored" if str(text).strip() else "Gemini API key cleared")
        # The key lives outside AppConfig, so a config diff can never notice it changed: ask the
        # pipeline to rebuild the translator explicitly (app.py wires this to refresh_models()).
        self.models_changed.emit()
        return True

    @Slot(result=str)
    def defaultPromptTemplate(self) -> str:
        from ..translate.context import DEFAULT_TEMPLATE

        return DEFAULT_TEMPLATE

    @Slot()
    def resetPromptTemplate(self) -> None:
        self._set_config("seriesPromptTemplate", "")

    # ------------------------------------------------------------------------ lists
    def _languages_source(self) -> List[Dict[str, str]]:
        pairs = [("auto", "Auto-detect")] + [(code, f"{name} ({code})") for code, name in LANGUAGES]
        return _items(pairs, self._cfg.source_lang)

    def _languages_target(self) -> List[Dict[str, str]]:
        return _items([(code, f"{name} ({code})") for code, name in LANGUAGES], self._cfg.target_lang)

    def _ocr_engines(self) -> List[Dict[str, str]]:
        from ..ocr.factory import engine_label

        return _items([(n, engine_label(n)) for n in self._engines], self._cfg.ocr_engine)

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

    def _backend_gemini(self) -> bool:
        return self._cfg.translation_backend == "gemini"

    def _backend_libre(self) -> bool:
        return self._cfg.translation_backend == "libretranslate"

    def _gemini_models(self) -> List[Dict[str, str]]:
        from ..translate.gemini import MODEL_PRESETS

        return _items([(m, m) for m in MODEL_PRESETS], self._cfg.gemini_model)

    def _gemini_api_formats(self) -> List[Dict[str, str]]:
        return _items([("native", "Gemini native (x-goog-api-key)"), ("openai", "OpenAI-compatible (Bearer)")],
                      self._cfg.gemini_api_format)

    # Kept for a future preset picker; since 2026-09-10 the Engines page shows a free-text Model field
    # whose hint comes from geminiModelPresets instead.
    geminiModels = Property("QVariantList", _gemini_models, notify=listsChanged)
    # Free-text model id on the Engines page: any id the endpoint serves; presets are only a hint.
    def _gemini_model_default(self) -> str:
        from ..translate.gemini import DEFAULT_MODEL  # lazy: keep requests out of window construction

        return DEFAULT_MODEL

    def _gemini_model_presets(self) -> str:
        from ..translate.gemini import MODEL_PRESETS

        return ", ".join(MODEL_PRESETS)

    geminiModelDefault = Property(str, _gemini_model_default, constant=True)
    geminiModelPresets = Property(str, _gemini_model_presets, constant=True)
    geminiApiFormats = Property("QVariantList", _gemini_api_formats, notify=listsChanged)

    def _translate_device_hint(self) -> str:
        return CUDA_FROZEN_HINT if is_frozen() and self._cfg.translate_device == "cuda" else ""

    backendOnline = Property(bool, _backend_online, notify=backendFlagsChanged)
    backendArgos = Property(bool, _backend_argos, notify=backendFlagsChanged)
    backendGemini = Property(bool, _backend_gemini, notify=backendFlagsChanged)
    backendLibre = Property(bool, _backend_libre, notify=backendFlagsChanged)
    translateDeviceHint = Property(str, _translate_device_hint, notify=translateDeviceChanged)
    frozen = Property(bool, lambda self: is_frozen(), constant=True)
    loading = Property(bool, lambda self: self._loading, notify=loadingChanged)

    # ------------------------------------------------------------------------ state
    def _set_capture_mode(self, on: bool) -> None:
        on = bool(on)
        if on == self._capture_mode:
            return
        self._capture_mode = on
        self.overlayCaptureModeChanged.emit()
        self.capture_mode_requested.emit(on)

    running = Property(bool, lambda self: self._running, notify=runningChanged)
    overlayCaptureMode = Property(
        bool, lambda self: self._capture_mode, _set_capture_mode, notify=overlayCaptureModeChanged
    )
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
    # F2-1: (frame window origin - current window origin) / dpr.  Main.qml adds it to backdropOrigin
    # so a frame grabbed before the last move is drawn where that desktop content still is.
    backdropShift = Property(QPointF, lambda self: QPointF(self._backdrop_shift), notify=backdropShiftChanged)
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
    def downloadMangaOcr(self) -> None:
        """Fetch the manga-ocr ONNX bundle into the models dir (F3); shares the progress row."""
        if self._download_active():
            self.show_status("A download is already running.")
            return
        worker = MangaOcrDownloadWorker(self._cfg.models_dir.strip() or self._cfg.models_dir, self)
        self._download_worker = worker
        self._download_base_label = "manga-ocr models"
        self._download_label = "Downloading manga-ocr models…"
        self._download_progress = -1
        self._download_visible = True
        worker.progress.connect(self._on_download_progress)
        worker.finished_ok.connect(self._on_manga_ocr_ok)
        worker.failed.connect(self._on_download_failed)
        worker.start()
        self.downloadChanged.emit()

    def _manga_ocr_models_ready(self) -> bool:
        from ..ocr.models import models_ready

        return models_ready(self._cfg.models_dir)

    mangaOcrModelsReady = Property(bool, _manga_ocr_models_ready, notify=downloadChanged)

    # --------------------------------------------------------------- quality renderer
    def _quality_renderer_options(self) -> List[Dict[str, str]]:
        return _items(QUALITY_RENDERER_ITEMS, self._cfg.quality_renderer)

    def _quality_models_ready(self) -> bool:
        from ..render.quality_models import models_ready

        return bool(models_ready(self._cfg.models_dir))

    def _quality_status(self) -> str:
        """The card's hint: Off / no sidecar / no models (with the size) / Ready."""
        from ..render.quality import find_sidecar_python
        from ..render.quality_models import size_label

        if self._cfg.quality_renderer != "auto":
            return "Off"
        if find_sidecar_python(self._cfg) is None:
            return "Sidecar not installed - run renderer\\install.bat"
        if not self._quality_models_ready():
            return f"Models not downloaded ({size_label()})"
        return "Ready"

    qualityRendererOptions = Property("QVariantList", _quality_renderer_options, notify=listsChanged)
    qualityModelsReady = Property(bool, _quality_models_ready, notify=qualityChanged)
    qualityStatus = Property(str, _quality_status, notify=qualityChanged)

    @Slot()
    def downloadQualityModels(self) -> None:
        """Fetch the quality renderer's models into the models dir; shares the progress row."""
        if self._download_active():
            self.show_status("A download is already running.")
            return
        worker = QualityModelsDownloadWorker(self._cfg.models_dir.strip() or self._cfg.models_dir, self)
        self._download_worker = worker
        self._download_base_label = "quality renderer models"
        self._download_label = "Downloading quality renderer models…"
        self._download_progress = -1
        self._download_visible = True
        worker.progress.connect(self._on_download_progress)
        worker.finished_ok.connect(self._on_quality_models_ok)
        worker.failed.connect(self._on_download_failed)
        worker.start()
        self.downloadChanged.emit()

    @Slot(str)
    def _on_quality_models_ok(self, path: str) -> None:
        self._download_visible = False
        self.downloadChanged.emit()
        self.show_status(f"Quality renderer models installed at {path}")
        self.models_changed.emit()

    @Slot(str)
    def _on_manga_ocr_ok(self, path: str) -> None:
        self._download_visible = False
        self.downloadChanged.emit()
        self.show_status(f"manga-ocr models installed at {path}")
        self.models_changed.emit()

    @Slot()
    def hideDownload(self) -> None:
        """Hide the progress row only; the download keeps running and reports to the status strip."""
        if self._download_visible:
            self._download_visible = False
            self.downloadChanged.emit()

    @Slot(int, int)
    @Slot("qlonglong", "qlonglong")  # the quality bundle is several GB: an int would overflow
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
