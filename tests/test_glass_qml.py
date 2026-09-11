"""QML under the offscreen platform (docs/GLASS_DESIGN.md section 7).

The offscreen scene graph is software-only: shaders never compile there, so nothing
render-dependent is asserted.  Everything else of the loaded control window is: status Ready,
no errors, no engine warnings, every Glass* / page component instantiable, accessibility roles
and names, the four Theme modes, bridge round trips through the real window, and ``closed``.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List

import pytest
from PySide6.QtCore import Qt, QObject, QUrl
from PySide6.QtGui import QImage
from PySide6.QtQml import QQmlComponent, QQmlExpression, qmlContext
from PySide6.QtQuick import QQuickItem, QQuickView

from glasstranslate.config.settings import AppConfig
from glasstranslate.core.types import Rect
from glasstranslate.ui import control as C
from glasstranslate.ui.glass.appearance import AppearanceSignals
from glasstranslate.ui.glass.backdrop import BackdropFrame

ROOT = Path(__file__).resolve().parents[1]
QML_DIR = ROOT / "glasstranslate" / "ui" / "qml"
pytestmark = pytest.mark.usefixtures("qapp")

if not (QML_DIR / "Main.qml").is_file():
    pytest.skip("qml/Main.qml not delivered yet", allow_module_level=True)


def _component_files() -> List[Path]:
    return sorted(QML_DIR.glob("Glass*.qml")) + sorted((QML_DIR / "pages").glob("*.qml"))


def _declared_props(qml: Path) -> Dict[str, str]:
    text = qml.read_text(encoding="utf-8")
    return {m.group(2): m.group(1) for m in re.finditer(r"^\s*(?:readonly\s+|required\s+)?property\s+([\w.]+)\s+(\w+)", text, re.M)}


def _qml_value(item: QObject, expression: str):
    ctx = qmlContext(item)
    assert ctx is not None
    expr = QQmlExpression(ctx, item, expression)
    result = expr.evaluate()
    value = result[0] if isinstance(result, tuple) else result
    assert not expr.hasError(), expr.error().toString()
    return value


@pytest.fixture()
def window(qapp, tmp_path: Path):
    w = C.ControlWindow(AppConfig(), tmp_path / "c.json")
    yield w
    w.shutdown()
    w.deleteLater()
    qapp.processEvents()


def test_window_loads_clean(window: C.ControlWindow) -> None:
    assert window.status() == QQuickView.Status.Ready
    assert window.errors() == []
    assert window.qml_warnings == []
    assert window.rootObject() is not None
    assert window.flags() & Qt.WindowType.FramelessWindowHint
    assert window.minimumSize().width() == 832 and window.minimumSize().height() == 640
    assert window.size().width() == 832 and window.size().height() == 640
    assert window.rootContext().contextProperty("bridge") is window.bridge
    assert window.rootContext().contextProperty("appearance") is window.appearance
    assert window.engine().imageProvider("backdrop") is not None


def test_backdrop_layer_applies_frame_offset(window: C.ControlWindow, qapp) -> None:
    """F2-1: backdropLayer (and the blur chain anchored to it) follows bridge.backdropShift."""
    root = window.rootObject()
    image = QImage(896, 704, QImage.Format.Format_RGB32)
    frame = BackdropFrame(image, (68, 68), Rect(100, 100, 832, 640), 1.0, 0.5, time.perf_counter())
    window.bridge.set_window_origin(100, 100)
    window._on_backdrop_frame(frame)
    qapp.processEvents()
    assert _qml_value(root, "backdropLayer.x") == -32 and _qml_value(root, "backdropLayer.y") == -32
    window.bridge.set_window_origin(140, 110)
    qapp.processEvents()
    assert _qml_value(root, "backdropLayer.x") == -72 and _qml_value(root, "backdropLayer.y") == -42
    assert _qml_value(root, "blurH1.x") == -72 and _qml_value(root, "blurV2.y") == -42


def test_segmented_bar_springs_settle_within_a_quarter_pixel(window: C.ControlWindow) -> None:
    """F2-3: SpringAnimation's default epsilon (0.01 px) keeps the render loop alive ~1.2 s after
    every tab switch; a quarter logical px is invisible and settles inside the 180 ms budget."""
    comp = QQmlComponent(window.engine(), QUrl("qrc:/qml/GlassSegmentedBar.qml"))
    obj = comp.create()
    assert obj is not None, [e.toString() for e in comp.errors()]
    try:
        for spring in ("springL", "springR"):
            assert _qml_value(obj, f"{spring}.epsilon") == pytest.approx(0.25)
    finally:
        obj.deleteLater()


def test_window_is_fixed_size(window: C.ControlWindow) -> None:
    """min == max == default: no affordance can resize the window (see docs/GLASS_DESIGN.md §1.1)."""
    assert window.minimumSize() == window.maximumSize() == C.WINDOW_DEFAULT_SIZE


def test_start_resize_does_not_start_a_system_resize(window: C.ControlWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []
    monkeypatch.setattr(window, "startSystemResize", lambda edges: calls.append(edges))
    window.bridge.startResize(int(Qt.Edge.LeftEdge.value))
    assert calls == []


def _walk_items(item: QQuickItem):
    yield item
    for child in item.childItems():
        yield from _walk_items(child)


def test_no_resize_edge_items_in_tree(window: C.ControlWindow) -> None:
    """No item in the loaded QML tree exposes the ``edges`` property of the old ``ResizeEdge``."""
    root = window.rootObject()
    assert root is not None
    edge_items = [it for it in _walk_items(root) if it.property("edges") is not None]
    assert edge_items == []


@pytest.mark.parametrize("qml", _component_files(), ids=lambda p: p.stem)
def test_component_instantiable(window: C.ControlWindow, qml: Path) -> None:
    comp = QQmlComponent(window.engine(), QUrl(f"qrc:/qml/{qml.relative_to(QML_DIR).as_posix()}"))
    assert comp.status() == QQmlComponent.Status.Ready, [e.toString() for e in comp.errors()]
    props = _declared_props(qml)
    init = {k: "x" for k in ("label", "text", "title", "placeholder", "hint") if k in props}
    obj = comp.createWithInitialProperties(init) if init else comp.create()
    assert obj is not None, [e.toString() for e in comp.errors()]
    try:
        assert not window.qml_warnings, window.qml_warnings
        if isinstance(obj, QQuickItem) and _qml_value(obj, "typeof Accessible !== 'undefined'"):
            role = _qml_value(obj, "Number(Accessible.role)")
            name = _qml_value(obj, "Accessible.name")
            # Item.focusPolicy exists on every Item since Qt 6.7; a *control* sets it to something
            # other than NoFocus (pure visuals such as GlassSurface/GlassShadow keep the default).
            is_control = bool(
                _qml_value(obj, "typeof focusPolicy !== 'undefined' && focusPolicy !== Qt.NoFocus")
            ) or qml.stem in {"GlassTextField", "GlassStepper"}
            if is_control:
                assert role not in (None, 0), f"{qml.stem}: Accessible.role missing"
                assert isinstance(name, str) and name, f"{qml.stem}: Accessible.name empty"
    finally:
        obj.deleteLater()


def test_theme_modes(window: C.ControlWindow) -> None:
    root = window.rootObject()
    app = window.appearance
    for sig, mode in [
        (AppearanceSignals(), "glass"),
        (AppearanceSignals(transparency=False), "solid"),
        (AppearanceSignals(composition=False), "solid"),
        (AppearanceSignals(high_contrast=True), "hc"),
        (AppearanceSignals(high_contrast=True, reduce_motion=True), "hc"),
        (AppearanceSignals(reduce_motion=True), "glass"),
    ]:
        app.apply(sig)
        assert app.mode == mode
        assert _qml_value(root, "Theme.mode") == mode
        assert bool(_qml_value(root, "Theme.reduceMotion")) is sig.reduce_motion
    app.apply(AppearanceSignals())


@pytest.mark.parametrize(
    "prop, value, attr",
    [
        ("sourceLang", "ja", "source_lang"),
        ("targetLang", "de", "target_lang"),
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
        ("hotkeyGrab", "<ctrl>+<alt>+z", "hotkeys.toggle_grab"),
    ],
)
def test_bridge_property_through_window(window: C.ControlWindow, tmp_path: Path, prop: str, value, attr: str) -> None:
    fired: List[AppConfig] = []
    window.config_changed.connect(fired.append)
    setattr(window.bridge, prop, value)
    assert len(fired) == 1 and fired[0] is window.config
    window.save_now()
    data = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
    for part in attr.split("."):
        data = data[part]
    assert data == value
    assert _qml_value(window.rootObject(), f"bridge.{prop}") == value


def test_overlay_page_has_capture_mode_toggle(window: C.ControlWindow) -> None:
    """F1: card "Misc" on the Overlay page hosts the runtime-only capture-mode toggle."""
    toggle = window.rootObject().findChild(QQuickItem, "captureModeToggle")
    assert toggle is not None
    assert _qml_value(toggle, "checked") is False
    assert _qml_value(toggle, "Accessible.name")
    assert "capture" in str(_qml_value(toggle, "Accessible.description")).lower()
    assert "window" in str(_qml_value(toggle, "text")).lower()  # the control window is captured too


def test_capture_mode_toggle_round_trip(window: C.ControlWindow, qapp) -> None:
    root = window.rootObject()
    toggle = root.findChild(QQuickItem, "captureModeToggle")
    assert toggle is not None
    root.findChild(QQuickItem, "tabBar").setProperty("currentIndex", 1)  # inactive pages are disabled
    qapp.processEvents()
    assert _qml_value(toggle, "enabled") is True
    requests: List[bool] = []
    window.capture_mode_requested.connect(requests.append)
    fired: List[AppConfig] = []
    window.config_changed.connect(fired.append)
    window.bridge.overlayCaptureMode = True
    assert _qml_value(toggle, "checked") is True
    _qml_value(toggle, "click()")  # the user clicks it off (click() emits toggled; toggle() does not)
    assert window.bridge.overlayCaptureMode is False
    assert requests == [True, False]
    assert fired == []  # never a config change


def test_engines_page_gemini_card_follows_backend(window: C.ControlWindow, qapp) -> None:
    """F4: the Gemini card is shown iff the gemini backend is selected; the key field is password."""
    root = window.rootObject()
    root.findChild(QQuickItem, "tabBar").setProperty("currentIndex", 2)  # inactive pages are disabled/hidden
    qapp.processEvents()
    card = root.findChild(QQuickItem, "geminiCard")
    key = root.findChild(QQuickItem, "geminiApiKeyField")
    assert card is not None and key is not None
    # ``visible`` is effective (ancestors count) and the page crossfade is animated, so assert the
    # card's own binding source plus its layout height instead of the effective flag.
    assert _qml_value(card, "bridge.backendGemini") is False
    window.bridge.translationBackend = "gemini"
    qapp.processEvents()
    assert _qml_value(card, "bridge.backendGemini") is True
    assert float(_qml_value(card, "height")) > 0
    assert _qml_value(key, "password") is True
    assert _qml_value(key, "echoMode") == 2  # TextInput.Password
    assert _qml_value(key, "text") == ""  # the stored key is never redisplayed
    model_field = root.findChild(QQuickItem, "geminiModelField")
    assert _qml_value(model_field, "text") == "gemini-3.8-flash"  # free text: any id the endpoint serves
    assert _qml_value(model_field, "placeholder") == "gemini-3.8-flash"


def test_engines_page_has_the_quality_renderer_card(window: C.ControlWindow, qapp) -> None:
    """The quality-renderer card: mode combo, status hint and the download button."""
    root = window.rootObject()
    root.findChild(QQuickItem, "tabBar").setProperty("currentIndex", 2)
    qapp.processEvents()
    card = root.findChild(QQuickItem, "qualityCard")
    combo = root.findChild(QQuickItem, "qualityRendererCombo")
    button = root.findChild(QQuickItem, "qualityDownloadButton")
    assert card is not None and combo is not None and button is not None
    assert _qml_value(combo, "value") == "off"
    assert _qml_value(card, "bridge.qualityStatus") == "Off"
    assert _qml_value(button, "enabled") is True  # no download running
    assert "Download quality models" in str(_qml_value(button, "text"))
    window.bridge.qualityRenderer = "auto"
    qapp.processEvents()
    assert _qml_value(combo, "value") == "auto"
    assert window.config.quality_renderer == "auto"


def test_translate_page_has_series_field_and_engines_page_template_editor(window: C.ControlWindow, qapp) -> None:
    """F5: quick series name on the main tab, the full template editor on the Engines tab."""
    root = window.rootObject()
    series = root.findChild(QQuickItem, "seriesNameField")
    area = root.findChild(QQuickItem, "templateArea")
    reset = root.findChild(QQuickItem, "resetTemplateButton")
    assert series is not None and area is not None and reset is not None
    assert _class_prefix(area) == "GlassTextArea"
    root.findChild(QQuickItem, "tabBar").setProperty("currentIndex", 2)
    qapp.processEvents()
    assert _qml_value(reset, "enabled") is False  # built-in template in use
    window.bridge.seriesName = "Jujutsu Kaisen"
    window.bridge.seriesPromptTemplate = "Custom [Series Name] prompt"
    qapp.processEvents()
    assert _qml_value(series, "text") == "Jujutsu Kaisen"
    assert _qml_value(area, "text") == "Custom [Series Name] prompt"
    assert _qml_value(reset, "enabled") is True
    _qml_value(reset, "click()")
    assert window.config.series_prompt_template == ""
    assert "[Series Name]" in str(_qml_value(area, "placeholder"))


def _class_prefix(item: QObject) -> str:
    return item.metaObject().className().split("_QML")[0]


def test_public_contract_and_close(window: C.ControlWindow) -> None:
    from glasstranslate.core.types import PipelineStats

    window.set_running(True)
    assert window.bridge.running is True
    window.update_stats(PipelineStats(total_ms=12.34, fps=8.06))
    assert window.bridge.stats.totalText == "12.3 ms"
    window.show_status("hello")
    assert window.bridge.statusMessage == "hello" and window.status_history == ["hello"]
    new = AppConfig(target_lang="fr")
    window.load_config(new)
    assert window.config is new and window.bridge.targetLang == "fr"
    closed: List[int] = []
    window.closed.connect(lambda: closed.append(1))
    window.bridge.targetLang = "es"
    assert window.close() is True
    assert closed == [1]
    assert not window.bridge._save_timer.isActive()  # the pending save was flushed


def test_theme_tab_springs_are_the_measured_2026_09_10_values(window: C.ControlWindow) -> None:
    """F2 follow-up: the indicator springs settle in ~0.55 s (8.0/0.50 lead, 6.0/0.45 trail); the
    original 4.6/0.36 + 3.0/0.30 kept the scene rendering ~0.95 s per switch (docs/perf baseline)."""
    root = window.rootObject()
    assert float(_qml_value(root, "Theme.springLead.spring")) == pytest.approx(8.0)
    assert float(_qml_value(root, "Theme.springLead.damping")) == pytest.approx(0.50)
    assert float(_qml_value(root, "Theme.springTrail.spring")) == pytest.approx(6.0)
    assert float(_qml_value(root, "Theme.springTrail.damping")) == pytest.approx(0.45)
