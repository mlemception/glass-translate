"""QML under the offscreen platform (docs/GLASS_DESIGN.md section 7).

The offscreen scene graph is software-only: shaders never compile there, so nothing
render-dependent is asserted.  Everything else of the loaded control window is: status Ready,
no errors, no engine warnings, every Glass* / page component instantiable, accessibility roles
and names, the four Theme modes, bridge round trips through the real window, and ``closed``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pytest
from PySide6.QtCore import QObject, QUrl
from PySide6.QtQml import QQmlComponent, QQmlExpression, qmlContext
from PySide6.QtQuick import QQuickItem, QQuickView

from glasstranslate.config.settings import AppConfig
from glasstranslate.ui import control as C
from glasstranslate.ui.glass.appearance import AppearanceSignals

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
    assert window.flags() & C.Qt.WindowType.FramelessWindowHint
    assert window.minimumSize().width() == 752 and window.minimumSize().height() == 560
    assert window.size().width() == 832 and window.size().height() == 640
    assert window.rootContext().contextProperty("bridge") is window.bridge
    assert window.rootContext().contextProperty("appearance") is window.appearance
    assert window.engine().imageProvider("backdrop") is not None


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
