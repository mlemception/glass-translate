"""Settings / status window.

:class:`ControlWindow` edits an :class:`~glasstranslate.config.AppConfig`,
emits :attr:`ControlWindow.config_changed` on every edit and persists the
config with a 300 ms debounce.  It also shows the live latency readout fed by
:meth:`ControlWindow.update_stats` and a status-bar line fed by
:meth:`ControlWindow.show_status`.  Buttons whose effect lives elsewhere
(Start/Stop, grab mode, show/hide glass) only emit signals; ``app.py`` wires
them to the pipeline and the overlay.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config.settings import AppConfig
from ..core.types import PipelineStats
from .hotkeys import normalize_hotkey

__all__ = ["ControlWindow", "LANGUAGES", "ModelDownloadWorker"]

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
_ONLINE_BACKENDS = {"libretranslate"}


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
        parent: Optional[QWidget] = None,
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


class ControlWindow(QMainWindow):
    """Main settings window; see the module docstring."""

    config_changed = Signal(object)  # AppConfig
    start_stop_requested = Signal(bool)  # True = start
    grab_mode_requested = Signal()
    toggle_glass_requested = Signal()
    models_changed = Signal()  # a package was downloaded
    closed = Signal()  # the user closed the window

    def __init__(
        self,
        cfg: AppConfig,
        config_path: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("GlassTranslate")
        self._cfg = cfg
        self._config_path = config_path
        self._loading = False
        self._download_worker: Optional[ModelDownloadWorker] = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(_SAVE_DEBOUNCE_MS)
        self._save_timer.timeout.connect(self.save_now)

        self._build_ui()
        self.load_config(cfg)

    # ---------------------------------------------------------------- public
    @property
    def config(self) -> AppConfig:
        return self._cfg

    def load_config(self, cfg: AppConfig) -> None:
        """Populate every control from ``cfg`` without emitting changes."""
        self._cfg = cfg
        self._loading = True
        try:
            self._select_data(self.source_combo, cfg.source_lang)
            self._select_data(self.target_combo, cfg.target_lang)
            self._select_data(self.ocr_engine_combo, cfg.ocr_engine)
            self._select_data(self.ocr_device_combo, cfg.ocr_device)
            self._select_data(self.backend_combo, cfg.translation_backend)
            self._select_data(self.translate_device_combo, cfg.translate_device)
            self.api_url_edit.setText(cfg.translation_api_url)
            self.api_key_edit.setText(cfg.translation_api_key)
            self.models_dir_edit.setText(cfg.models_dir)
            self.opacity_slider.setValue(int(round(cfg.overlay_opacity * 100)))
            self.refresh_spin.setValue(cfg.refresh_hz)
            self.debounce_spin.setValue(cfg.debounce_ms)
            self.confidence_spin.setValue(cfg.min_confidence)
            self.font_combo.setCurrentFont(QFont(cfg.font_family))
            self.hide_original_check.setChecked(cfg.hide_original)
            self.hotkey_grab_edit.setText(cfg.hotkeys.toggle_grab)
            self.hotkey_running_edit.setText(cfg.hotkeys.toggle_running)
            self.hotkey_hidden_edit.setText(cfg.hotkeys.toggle_hidden)
            self._update_backend_enabled()
        finally:
            self._loading = False

    def set_running(self, running: bool) -> None:
        """Reflect the pipeline state on the Start/Stop button."""
        self.start_button.setText("Stop" if running else "Start")
        self.start_button.setChecked(running)

    @Slot(object)
    def update_stats(self, stats: PipelineStats) -> None:
        """Refresh the latency readout from a pipeline pass."""
        self.total_label.setText(f"{stats.total_ms:.1f} ms")
        self.fps_label.setText(f"{stats.fps:.1f}")
        self.stage_label.setText(
            f"capture {stats.capture_ms:.1f} | diff {stats.diff_ms:.1f} | ocr {stats.ocr_ms:.1f} | "
            f"style {stats.style_ms:.1f} | translate {stats.translate_ms:.1f}"
        )
        skipped = " (unchanged)" if stats.skipped_unchanged else ""
        self.segments_label.setText(f"{stats.segments} live, {stats.dirty_regions} dirty{skipped}")
        rate = stats.extra.get("cache_hit_rate")
        rate_txt = f"{rate * 100:.0f}%" if isinstance(rate, (int, float)) else "-"
        self.cache_label.setText(f"{stats.cache_hits} hit / {stats.cache_misses} miss, {rate_txt} overall")
        self.devices_label.setText(
            f"ocr: {stats.extra.get('ocr', '?')}@{stats.ocr_device or '?'}   "
            f"translate: {stats.extra.get('translator', '?')}@{stats.translate_device or '?'}   "
            f"capture: {stats.extra.get('capture', '?')}   src: {stats.extra.get('src_lang', '-')}"
        )

    @Slot(str)
    def show_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def save_soon(self) -> None:
        """Schedule a debounced save (for changes made outside this window,
        e.g. the glass geometry)."""
        self._save_timer.start()

    @Slot()
    def save_now(self) -> None:
        """Persist the config immediately (normally driven by the debounce timer)."""
        try:
            self._cfg.save(self._config_path)
        except Exception as exc:
            log.exception("saving config failed")
            self.show_status(f"Could not save config: {exc}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
            self.save_now()
        super().closeEvent(event)
        self.closed.emit()

    # ------------------------------------------------------------- building
    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.addWidget(self._build_language_group())
        root.addWidget(self._build_engine_group())
        root.addWidget(self._build_overlay_group())
        root.addWidget(self._build_pipeline_group())
        root.addWidget(self._build_hotkey_group())
        root.addLayout(self._build_action_row())
        root.addWidget(self._build_stats_group())
        root.addStretch(1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_language_group(self) -> QGroupBox:
        box = QGroupBox("Languages")
        form = QFormLayout(box)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Auto-detect", "auto")
        self.target_combo = QComboBox()
        for code, name in LANGUAGES:
            self.source_combo.addItem(f"{name} ({code})", code)
            self.target_combo.addItem(f"{name} ({code})", code)
        form.addRow("Source", self.source_combo)
        form.addRow("Target", self.target_combo)
        self.source_combo.currentIndexChanged.connect(self._on_edit)
        self.target_combo.currentIndexChanged.connect(self._on_edit)
        return box

    def _build_engine_group(self) -> QGroupBox:
        from ..ocr.factory import available_engines
        from ..translate.factory import available_backends

        box = QGroupBox("Engines")
        form = QFormLayout(box)
        self.ocr_engine_combo = QComboBox()
        for name in available_engines():
            self.ocr_engine_combo.addItem(name, name)
        self.ocr_device_combo = QComboBox()
        for name in _OCR_DEVICES:
            self.ocr_device_combo.addItem(name, name)
        self.backend_combo = QComboBox()
        for name in available_backends():
            self.backend_combo.addItem(name, name)
        self.translate_device_combo = QComboBox()
        for name in _TRANSLATE_DEVICES:
            self.translate_device_combo.addItem(name, name)
        self.api_url_edit = QLineEdit()
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.models_dir_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_models_dir)
        self.download_button = QPushButton("Download model…")
        self.download_button.clicked.connect(self._download_model)
        self.sugoi_button = QPushButton("Get Sugoi (ja→en)…")
        self.sugoi_button.setToolTip(
            "Install the Sugoi v4 Japanese→English model (~700 MB). Trained on game and "
            "visual-novel dialogue, it handles short manga lines much better than the "
            "generic Argos ja→en package and takes priority for that pair."
        )
        self.sugoi_button.clicked.connect(lambda: self._download_model(sugoi=True))
        models_row = QHBoxLayout()
        models_row.addWidget(self.models_dir_edit, 1)
        models_row.addWidget(browse)
        models_row.addWidget(self.download_button)
        models_row.addWidget(self.sugoi_button)

        form.addRow("OCR engine", self.ocr_engine_combo)
        form.addRow("OCR device", self.ocr_device_combo)
        form.addRow("Translation", self.backend_combo)
        form.addRow("Translate device", self.translate_device_combo)
        form.addRow("API URL", self.api_url_edit)
        form.addRow("API key", self.api_key_edit)
        form.addRow("Models dir", models_row)

        for combo in (self.ocr_engine_combo, self.ocr_device_combo, self.translate_device_combo):
            combo.currentIndexChanged.connect(self._on_edit)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        self.api_url_edit.editingFinished.connect(self._on_edit)
        self.api_key_edit.editingFinished.connect(self._on_edit)
        self.models_dir_edit.editingFinished.connect(self._on_edit)
        return box

    def _build_overlay_group(self) -> QGroupBox:
        box = QGroupBox("Glass")
        form = QFormLayout(box)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_value = QLabel("10%")
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_value)
        self.font_combo = QFontComboBox()
        self.hide_original_check = QCheckBox("Paint background-coloured box under the translation")
        form.addRow("Background opacity", opacity_row)
        form.addRow("Font", self.font_combo)
        form.addRow("Hide original", self.hide_original_check)
        self.opacity_slider.valueChanged.connect(self._on_edit)
        self.font_combo.currentFontChanged.connect(self._on_edit)
        self.hide_original_check.toggled.connect(self._on_edit)
        return box

    def _build_pipeline_group(self) -> QGroupBox:
        box = QGroupBox("Pipeline")
        form = QFormLayout(box)
        self.refresh_spin = QDoubleSpinBox()
        self.refresh_spin.setRange(0.5, 60.0)
        self.refresh_spin.setSingleStep(0.5)
        self.refresh_spin.setSuffix(" Hz")
        self.debounce_spin = QSpinBox()
        self.debounce_spin.setRange(0, 2000)
        self.debounce_spin.setSingleStep(10)
        self.debounce_spin.setSuffix(" ms")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        form.addRow("Refresh rate", self.refresh_spin)
        form.addRow("Debounce", self.debounce_spin)
        form.addRow("Min OCR confidence", self.confidence_spin)
        self.refresh_spin.valueChanged.connect(self._on_edit)
        self.debounce_spin.valueChanged.connect(self._on_edit)
        self.confidence_spin.valueChanged.connect(self._on_edit)
        return box

    def _build_hotkey_group(self) -> QGroupBox:
        box = QGroupBox("Hotkeys (pynput syntax, e.g. <ctrl>+<alt>+g)")
        form = QFormLayout(box)
        self.hotkey_grab_edit = QLineEdit()
        self.hotkey_running_edit = QLineEdit()
        self.hotkey_hidden_edit = QLineEdit()
        form.addRow("Toggle grab mode", self.hotkey_grab_edit)
        form.addRow("Start / stop", self.hotkey_running_edit)
        form.addRow("Show / hide glass", self.hotkey_hidden_edit)
        for edit in (self.hotkey_grab_edit, self.hotkey_running_edit, self.hotkey_hidden_edit):
            edit.editingFinished.connect(self._on_edit)
        return box

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setCheckable(True)
        self.start_button.clicked.connect(self._on_start_clicked)
        self.grab_button = QPushButton("Grab mode")
        self.grab_button.clicked.connect(self.grab_mode_requested)
        self.glass_button = QPushButton("Show/Hide glass")
        self.glass_button.clicked.connect(self.toggle_glass_requested)
        row.addWidget(self.start_button)
        row.addWidget(self.grab_button)
        row.addWidget(self.glass_button)
        return row

    def _build_stats_group(self) -> QGroupBox:
        box = QGroupBox("Live")
        grid = QGridLayout(box)
        self.total_label = QLabel("-")
        self.fps_label = QLabel("-")
        self.stage_label = QLabel("-")
        self.segments_label = QLabel("-")
        self.cache_label = QLabel("-")
        self.devices_label = QLabel("-")
        rows = [
            ("Total", self.total_label),
            ("FPS", self.fps_label),
            ("Stages (ms)", self.stage_label),
            ("Segments", self.segments_label),
            ("Cache", self.cache_label),
            ("Devices", self.devices_label),
        ]
        for i, (title, label) in enumerate(rows):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(QLabel(title), i, 0)
            grid.addWidget(label, i, 1)
        grid.setColumnStretch(1, 1)
        return box

    # ------------------------------------------------------------ handlers
    @staticmethod
    def _select_data(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx < 0:
            combo.addItem(value, value)
            idx = combo.count() - 1
        combo.setCurrentIndex(idx)

    def _update_backend_enabled(self) -> None:
        online = self.backend_combo.currentData() in _ONLINE_BACKENDS
        self.api_url_edit.setEnabled(online)
        self.api_key_edit.setEnabled(online)
        argos = self.backend_combo.currentData() == "argos"
        self.download_button.setEnabled(argos)
        self.translate_device_combo.setEnabled(argos)

    def _on_backend_changed(self) -> None:
        self._update_backend_enabled()
        self._on_edit()

    def _read_hotkey(self, edit: QLineEdit, current: str) -> str:
        text = edit.text().strip()
        if not text:
            return current
        try:
            return normalize_hotkey(text)
        except Exception as exc:
            self.show_status(f"Invalid hotkey {text!r}: {exc}")
            edit.setText(current)
            return current

    def _on_edit(self, *_args: object) -> None:
        """Read every control into the config, emit and schedule a save."""
        if self._loading:
            return
        cfg = self._cfg
        cfg.source_lang = str(self.source_combo.currentData())
        cfg.target_lang = str(self.target_combo.currentData())
        cfg.ocr_engine = str(self.ocr_engine_combo.currentData())
        cfg.ocr_device = str(self.ocr_device_combo.currentData())
        cfg.translation_backend = str(self.backend_combo.currentData())
        cfg.translate_device = str(self.translate_device_combo.currentData())
        cfg.translation_api_url = self.api_url_edit.text().strip()
        cfg.translation_api_key = self.api_key_edit.text().strip()
        cfg.models_dir = self.models_dir_edit.text().strip() or cfg.models_dir
        cfg.overlay_opacity = self.opacity_slider.value() / 100.0
        self.opacity_value.setText(f"{self.opacity_slider.value()}%")
        cfg.refresh_hz = float(self.refresh_spin.value())
        cfg.debounce_ms = int(self.debounce_spin.value())
        cfg.min_confidence = float(self.confidence_spin.value())
        cfg.font_family = self.font_combo.currentFont().family()
        cfg.hide_original = self.hide_original_check.isChecked()
        cfg.hotkeys.toggle_grab = self._read_hotkey(self.hotkey_grab_edit, cfg.hotkeys.toggle_grab)
        cfg.hotkeys.toggle_running = self._read_hotkey(self.hotkey_running_edit, cfg.hotkeys.toggle_running)
        cfg.hotkeys.toggle_hidden = self._read_hotkey(self.hotkey_hidden_edit, cfg.hotkeys.toggle_hidden)
        self.config_changed.emit(cfg)
        self._save_timer.start()

    def _on_start_clicked(self, checked: bool) -> None:
        self.start_stop_requested.emit(bool(checked))

    def _browse_models_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Models directory", self.models_dir_edit.text())
        if chosen:
            self.models_dir_edit.setText(chosen)
            self._on_edit()

    def _download_model(self, sugoi: bool = False) -> None:
        """Download the Argos package for the selected pair (via English pivot
        when the source is "auto": we download en -> target), or the Sugoi
        ja -> en model when ``sugoi`` is set."""
        if self._download_worker is not None and self._download_worker.isRunning():
            QMessageBox.information(self, "Download", "A download is already running.")
            return
        if sugoi:
            src, tgt = "ja", "en"
        else:
            src = str(self.source_combo.currentData())
            tgt = str(self.target_combo.currentData())
            if src == "auto":
                src = "en"
            if src == tgt:
                QMessageBox.information(self, "Download", "Source and target language are the same.")
                return
        label = "Sugoi v4 ja → en" if sugoi else f"{src} → {tgt}"
        dialog = QProgressDialog(f"Downloading {label} model…", "Hide", 0, 0, self)
        dialog.setWindowTitle("Download model")
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        worker = ModelDownloadWorker(
            self.models_dir_edit.text().strip() or self._cfg.models_dir, src, tgt, self, sugoi=sugoi
        )
        self._download_worker = worker

        def on_progress(done: int, total: int) -> None:
            if total > 0:
                dialog.setRange(0, 100)
                dialog.setValue(int(done * 100 / total))
                dialog.setLabelText(f"Downloading {label}: {done / 1e6:.1f} / {total / 1e6:.1f} MB")
            else:
                dialog.setLabelText(f"Downloading {label}: {done / 1e6:.1f} MB")

        def on_ok(path: str) -> None:
            dialog.close()
            self.show_status(f"Installed model at {path}")
            self.models_changed.emit()

        def on_failed(message: str) -> None:
            dialog.close()
            QMessageBox.warning(self, "Download failed", message)
            self.show_status(f"Download failed: {message}")

        worker.progress.connect(on_progress)
        worker.finished_ok.connect(on_ok)
        worker.failed.connect(on_failed)
        # urllib downloads cannot be interrupted cleanly; "Hide" lets it finish
        # in the background and the result is reported in the status bar.
        dialog.canceled.connect(dialog.hide)
        worker.start()
        dialog.show()
