"""Bridge round-trip, hotkeys, backend flags, stats / download strings, polarity, frozen rule
(docs/GLASS_DESIGN.md section 7).  Headless: no window is created."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from glasstranslate.config.settings import AppConfig
from glasstranslate.core.types import PipelineStats, Rect
from glasstranslate.ui import control as C
from glasstranslate.ui.glass.appearance import Appearance, AppearanceSignals
from glasstranslate.ui.glass.backdrop import POLARITY_HIGH, POLARITY_LOW, BackdropFrame, InkPolarity

pytestmark = pytest.mark.usefixtures("qapp")


def _bridge(tmp_path: Path, cfg: AppConfig | None = None, *, dark: bool = True):
    appearance = Appearance(watch=False, overrides="")
    appearance.apply(AppearanceSignals(dark_mode=dark))
    cfg = cfg or AppConfig()
    bridge = C.ControlBridge(cfg, tmp_path / "config.json", appearance=appearance)
    fired: List[AppConfig] = []
    bridge.config_changed.connect(fired.append)
    return bridge, cfg, fired


def _frame(luma: float, t: float, dpr: float = 1.0) -> BackdropFrame:
    # a real grab: window (832x640 logical) + MARGIN 32 *physical* px on every side
    img = QImage(int(832 * dpr) + 64, int(640 * dpr) + 64, QImage.Format.Format_RGB32)
    return BackdropFrame(img, (68, 68), Rect(100, 100, int(832 * dpr), int(640 * dpr)), dpr, luma, t)


# ------------------------------------------------------------------------ round trip / save
@pytest.mark.parametrize(
    "prop, value, attr",
    [
        ("sourceLang", "ja", "source_lang"),
        ("targetLang", "de", "target_lang"),
        ("ocrEngine", "other-ocr", "ocr_engine"),
        ("ocrDevice", "cpu", "ocr_device"),
        ("translationBackend", "identity", "translation_backend"),
        ("translateDevice", "cpu", "translate_device"),
        ("apiUrl", "https://example.org", "translation_api_url"),
        ("apiKey", "secret", "translation_api_key"),
        ("modelsDir", "C:/models", "models_dir"),
        ("overlayOpacity", 0.42, "overlay_opacity"),
        ("fontFamily", "Arial", "font_family"),
        ("hideOriginal", False, "hide_original"),
        ("mangaMode", False, "manga_mode"),
        ("uppercase", False, "uppercase"),
        ("refreshHz", 4.5, "refresh_hz"),
        ("debounceMs", 250, "debounce_ms"),
        ("minConfidence", 0.7, "min_confidence"),
    ],
)
def test_config_property_round_trip_and_save(tmp_path: Path, prop: str, value, attr: str) -> None:
    bridge, cfg, fired = _bridge(tmp_path)
    assert getattr(bridge, prop) == getattr(cfg, attr)
    setattr(bridge, prop, value)
    assert getattr(cfg, attr) == value
    assert getattr(bridge, prop) == value
    assert len(fired) == 1 and fired[0] is cfg
    setattr(bridge, prop, value)  # unchanged value: no second emit
    assert len(fired) == 1
    bridge.save_now()
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))[attr] == value


def test_setters_strip_and_keep_models_dir(tmp_path: Path) -> None:
    bridge, cfg, fired = _bridge(tmp_path)
    previous = cfg.models_dir
    bridge.apiUrl = "  https://x.y  "
    bridge.apiKey = " k "
    assert cfg.translation_api_url == "https://x.y" and cfg.translation_api_key == "k"
    bridge.modelsDir = "   "
    assert cfg.models_dir == previous
    assert len(fired) == 2  # the empty models dir was a no-op


def test_setting_while_loading_is_ignored(tmp_path: Path) -> None:
    bridge, cfg, fired = _bridge(tmp_path)
    new = AppConfig(target_lang="fr", overlay_opacity=0.3)
    notified: List[str] = []
    bridge.targetLangChanged.connect(lambda: notified.append("target"))
    bridge.load_config(new)
    assert bridge.config is new and bridge.targetLang == "fr" and not fired and "target" in notified
    bridge._loading = True
    bridge.targetLang = "es"
    assert new.target_lang == "fr" and not fired


def test_save_failure_reports_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bridge, cfg, _ = _bridge(tmp_path)

    def boom(self, path=None):
        raise OSError("disk full")

    monkeypatch.setattr(AppConfig, "save", boom)
    bridge.save_now()
    assert bridge.statusMessage == "Could not save config: disk full"


def test_debounced_save_timer(tmp_path: Path) -> None:
    bridge, cfg, _ = _bridge(tmp_path)
    bridge.targetLang = "it"
    assert bridge._save_timer.isActive() and bridge._save_timer.interval() == 300
    bridge.save_soon()
    bridge.flush_pending_save()
    assert not bridge._save_timer.isActive()
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))["target_lang"] == "it"


# ------------------------------------------------------------------------------- hotkeys
def test_hotkey_empty_reverts_silently(tmp_path: Path) -> None:
    bridge, cfg, fired = _bridge(tmp_path)
    before = cfg.hotkeys.toggle_grab
    assert bridge.setHotkey("grab", "   ") is False
    assert cfg.hotkeys.toggle_grab == before and bridge.statusMessage == "Ready" and not fired


def test_hotkey_invalid_sets_status_and_returns_false(tmp_path: Path) -> None:
    bridge, cfg, fired = _bridge(tmp_path)
    before = cfg.hotkeys.toggle_running
    assert bridge.setHotkey("running", "ctrl+") is False
    assert cfg.hotkeys.toggle_running == before and not fired
    assert bridge.statusMessage.startswith("Invalid hotkey 'ctrl+':")


def test_hotkey_normalized_form_is_displayed(tmp_path: Path) -> None:
    bridge, cfg, fired = _bridge(tmp_path)
    assert bridge.setHotkey("hidden", "Ctrl+Alt+Q") is True
    assert cfg.hotkeys.toggle_hidden == "<ctrl>+<alt>+q" and bridge.hotkeyHidden == "<ctrl>+<alt>+q"
    assert len(fired) == 1
    assert bridge.setHotkey("hotkeyHidden", "<ctrl>+<alt>+q") is True  # same value: no new edit
    assert len(fired) == 1


def test_hotkey_property_setter_stores_verbatim(tmp_path: Path) -> None:
    bridge, cfg, fired = _bridge(tmp_path)
    bridge.hotkeyGrab = "<ctrl>+<alt>+z"
    assert cfg.hotkeys.toggle_grab == "<ctrl>+<alt>+z" and len(fired) == 1


# ------------------------------------------------------------------------ backend flags (section 3)
def test_backend_flags(tmp_path: Path) -> None:
    """Deliberate changes 1-2: Download and Sugoi share the argos rule; models dir is always on."""
    bridge, cfg, _ = _bridge(tmp_path, AppConfig(translation_backend="argos"))
    assert bridge.backendArgos is True and bridge.backendOnline is False
    bridge.translationBackend = "libretranslate"
    assert bridge.backendArgos is False and bridge.backendOnline is True
    bridge.translationBackend = "identity"
    assert bridge.backendArgos is False and bridge.backendOnline is False
    # The models dir stays editable for every backend: the setter is never gated.
    bridge.modelsDir = "D:/m"
    assert cfg.models_dir == "D:/m"


# ---------------------------------------------------------------------------------- lists
def test_lists_and_unknown_value_rule(tmp_path: Path) -> None:
    bridge, cfg, _ = _bridge(tmp_path, AppConfig(source_lang="xx", ocr_device="npu", font_family="Nope Sans"))
    src = bridge.languagesSource
    assert src[0] == {"value": "auto", "text": "Auto-detect"} and src[1]["value"] == "en"
    assert src[-1] == {"value": "xx", "text": "xx"}
    assert [i["value"] for i in bridge.languagesTarget][:3] == ["en", "de", "es"]
    assert bridge.ocrDevices[-1] == {"value": "npu", "text": "npu"}
    assert [i["value"] for i in bridge.ocrDevices][:3] == ["auto", "gpu", "cpu"]
    assert "argos" in [i["value"] for i in bridge.backends]
    assert "rapidocr" in [i["value"] for i in bridge.ocrEngines]
    assert bridge.fontFamilies[-1] == "Nope Sans"
    notified: List[str] = []
    bridge.listsChanged.connect(lambda: notified.append("lists"))
    bridge.targetLang = "zz"  # unknown -> appended, lists notify
    assert bridge.languagesTarget[-1]["value"] == "zz" and notified


def test_translate_devices_frozen_rule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert [i["value"] for i in C.translate_device_items("auto", frozen=False)] == ["auto", "cuda", "cpu"]
    assert [i["value"] for i in C.translate_device_items("auto", frozen=True)] == ["auto", "cpu"]
    frozen_cuda = C.translate_device_items("cuda", frozen=True)
    assert [i["value"] for i in frozen_cuda] == ["auto", "cpu", "cuda"]
    assert frozen_cuda[-1]["hint"] == C.CUDA_FROZEN_HINT
    monkeypatch.setattr(C.sys, "frozen", True, raising=False)
    bridge, cfg, _ = _bridge(tmp_path, AppConfig(translate_device="cuda"))
    assert [i["value"] for i in bridge.translateDevices] == ["auto", "cpu", "cuda"]
    assert bridge.translateDeviceHint == C.CUDA_FROZEN_HINT and bridge.frozen is True


# ---------------------------------------------------------------------------------- stats
def test_stats_formatting(tmp_path: Path) -> None:
    bridge, _, _ = _bridge(tmp_path)
    stats = bridge.stats
    assert (stats.totalText, stats.fpsText, stats.stagesText, stats.segmentsText, stats.cacheText,
            stats.devicesText) == ("-",) * 6
    s = PipelineStats(capture_ms=1.24, diff_ms=0.5, ocr_ms=30.06, style_ms=2.0, translate_ms=8.44,
                      total_ms=12.34, fps=8.06, segments=3, dirty_regions=2, cache_hits=5, cache_misses=1,
                      skipped_unchanged=True, ocr_device="gpu", translate_device="cpu",
                      extra={"blocks": 2, "cache_hit_rate": 0.833, "ocr": "rapidocr", "translator": "argos",
                             "capture": "dxcam", "src_lang": "ja"})
    stats.update(s)
    assert stats.totalText == "12.3 ms" and stats.fpsText == "8.1"
    assert stats.stagesText == "capture 1.2 | diff 0.5 | ocr 30.1 | style 2.0 | translate 8.4"
    assert stats.segmentsText == "3 live, 2 blocks, 2 dirty (unchanged)"
    assert stats.cacheText == "5 hit / 1 miss, 83% overall"
    assert stats.devicesText == "ocr: rapidocr@gpu   translate: argos@cpu   capture: dxcam   src: ja"
    plain = C.format_stats(PipelineStats())
    assert plain["segments"] == "0 live, 0 dirty" and plain["cache"] == "0 hit / 0 miss, - overall"
    assert plain["devices"] == "ocr: ?@?   translate: ?@?   capture: ?   src: -"


# ------------------------------------------------------------------------------- download
def test_download_strings() -> None:
    assert C.download_label(True, "x", "y") == "Sugoi v4 ja → en"
    assert C.download_label(False, "en", "de") == "en → de"
    assert C.download_progress("en → de", 12_345_678, 100_000_000) == ("Downloading en → de: 12.3 / 100.0 MB", 12)
    assert C.download_progress("en → de", 5_000_000, -1) == ("Downloading en → de: 5.0 MB", -1)


class _FakeWorker(QObject):
    """Stands in for ModelDownloadWorker: never touches the network."""

    progress = Signal(int, int)
    finished_ok = Signal(str)
    failed = Signal(str)
    instances: List["_FakeWorker"] = []

    def __init__(self, models_dir, from_code, to_code, parent=None, *, sugoi=False):
        super().__init__(parent)
        self.args = (models_dir, from_code, to_code, sugoi)
        self._running = False
        _FakeWorker.instances.append(self)

    def start(self):
        self._running = True

    def isRunning(self):  # noqa: N802 - QThread API
        return self._running

    def finish(self, path: str):
        self._running = False
        self.finished_ok.emit(path)

    def fail(self, message: str):
        self._running = False
        self.failed.emit(message)


def test_download_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C, "ModelDownloadWorker", _FakeWorker)
    _FakeWorker.instances.clear()
    bridge, cfg, _ = _bridge(tmp_path, AppConfig(source_lang="auto", target_lang="en"))
    models: List[int] = []
    bridge.models_changed.connect(lambda: models.append(1))
    bridge.downloadModel(False)  # auto -> en pivots to en -> en: same language
    assert bridge.statusMessage == "Source and target language are the same." and not _FakeWorker.instances
    bridge.targetLang = "de"
    bridge.downloadModel(False)
    worker = _FakeWorker.instances[-1]
    assert worker.args == (cfg.models_dir, "en", "de", False)
    assert bridge.downloadActive and bridge.downloadVisible
    assert bridge.downloadLabel == "Downloading en → de model…" and bridge.downloadProgress == -1
    bridge.downloadModel(True)
    assert bridge.statusMessage == "A download is already running." and len(_FakeWorker.instances) == 1
    worker.progress.emit(2_500_000, -1)
    assert bridge.downloadLabel == "Downloading en → de: 2.5 MB" and bridge.downloadProgress == -1
    worker.progress.emit(50_000_000, 200_000_000)
    assert bridge.downloadLabel == "Downloading en → de: 50.0 / 200.0 MB" and bridge.downloadProgress == 25
    bridge.hideDownload()
    assert not bridge.downloadVisible and bridge.downloadActive  # hidden, still running, still refused
    bridge.downloadModel(False)
    assert bridge.statusMessage == "A download is already running."
    worker.finish("C:/models/en_de")
    assert bridge.statusMessage == "Installed model at C:/models/en_de" and models == [1]
    assert not bridge.downloadActive and not bridge.downloadVisible
    bridge.downloadModel(True)
    sugoi = _FakeWorker.instances[-1]
    assert sugoi.args == (cfg.models_dir, "ja", "en", True) and bridge.downloadLabel == "Downloading Sugoi v4 ja → en model…"
    sugoi.fail("404")
    assert bridge.statusMessage == "Download failed: 404" and not bridge.downloadVisible


# ------------------------------------------------------------------------------- polarity
def test_ink_polarity_state_machine() -> None:
    p = InkPolarity(dark_mode=True)
    assert p.value == 0 and InkPolarity(dark_mode=False).value == 1
    assert not p.update(0.5) and p.value == 0  # inside the band: unchanged
    assert not p.update(POLARITY_HIGH) and p.value == 0  # boundary is not enough
    assert p.update(POLARITY_HIGH + 0.01) and p.value == 1
    assert not p.update(0.5) and p.value == 1  # hysteresis keeps the light polarity
    assert not p.update(POLARITY_LOW) and p.value == 1
    assert p.update(POLARITY_LOW - 0.01) and p.value == 0
    assert p.reset(dark_mode=False) and p.value == 1


def test_bridge_polarity_from_frames(tmp_path: Path) -> None:
    bridge, cfg, _ = _bridge(tmp_path, dark=True)
    assert bridge.inkPolarity == 0 and bridge.backdropSerial == 0
    flips: List[int] = []
    bridge.inkPolarityChanged.connect(lambda: flips.append(bridge.inkPolarity))
    t = time.perf_counter()
    bridge.push_backdrop(_frame(0.95, t))
    assert bridge.backdropSerial == 1 and bridge.backdropLuma == pytest.approx(0.95)
    assert bridge.backdropOrigin.x() == -32 and bridge.backdropOrigin.y() == -32
    assert bridge.backdropSize.width() == 896 and bridge.backdropSize.height() == 704
    assert flips == [1]
    # The EMA (tau 250 ms) needs time to fall below .38: one dark frame 50 ms later is not enough.
    bridge.push_backdrop(_frame(0.0, t + 0.05))
    assert bridge.inkPolarity == 1 and 0.38 < bridge.backdropLuma < 0.95
    bridge.push_backdrop(_frame(0.0, t + 2.0))
    assert bridge.inkPolarity == 0 and flips == [1, 0]


def test_bridge_polarity_initial_from_light_mode(tmp_path: Path) -> None:
    bridge, _, _ = _bridge(tmp_path, dark=False)
    assert bridge.inkPolarity == 1


def test_appearance_change_resets_polarity_outside_glass(tmp_path: Path) -> None:
    bridge, _, _ = _bridge(tmp_path, dark=True)
    appearance = bridge._appearance
    appearance.apply(AppearanceSignals(dark_mode=False, transparency=False))  # solid mode, light theme
    assert bridge.inkPolarity == 1
    bridge.push_backdrop(_frame(0.0, time.perf_counter()))  # grabber frames are ignored outside glass
    assert bridge.inkPolarity == 1


# --------------------------------------------------------------------------------- state
def test_running_status_and_slots(tmp_path: Path) -> None:
    bridge, _, _ = _bridge(tmp_path)
    got: List[object] = []
    bridge.start_stop_requested.connect(got.append)
    bridge.grab_mode_requested.connect(lambda: got.append("grab"))
    bridge.toggle_glass_requested.connect(lambda: got.append("glass"))
    bridge.startStop(True)
    bridge.grabMode()
    bridge.toggleGlass()
    assert got == [True, "grab", "glass"]
    bridge.set_running(True)
    assert bridge.running is True
    bridge.show_status("Paused")
    assert bridge.statusMessage == "Paused" and bridge.status_history == ["Paused"]
    assert bridge.windowTitle == "GlassTranslate" and bridge.loading is False
    # Window slots are safe without a window.
    bridge.startMove()
    bridge.startResize(3)
    bridge.minimize()
    bridge.close()
    bridge.keyboardMove()
    bridge.keyboardSize()


def test_bridge_backdrop_geometry_is_logical_at_dpr_1_5(tmp_path: Path) -> None:
    """Origin and size are logical (physical / dpr): Qt ignores the QImage DPR for provider images."""
    bridge, cfg, _ = _bridge(tmp_path, dark=True)
    bridge.push_backdrop(_frame(0.5, time.perf_counter(), dpr=1.5))
    assert bridge.backdropOrigin.x() == pytest.approx(-32 / 1.5)
    assert bridge.backdropSize.width() == pytest.approx(1312 / 1.5)
    assert bridge.backdropSize.height() == pytest.approx(1024 / 1.5)
