"""Tests for glasstranslate.config.settings (no display, GPU, models or network)."""
from __future__ import annotations

import json
from pathlib import Path

from glasstranslate.config.settings import (
    AppConfig,
    Hotkeys,
    OverlayGeometry,
    default_config_path,
    default_models_dir,
    project_root,
)


def test_defaults_are_sane():
    cfg = AppConfig()
    assert cfg.source_lang == "auto"
    assert cfg.target_lang == "en"
    assert 0.0 <= cfg.overlay_opacity <= 1.0
    assert cfg.refresh_hz > 0
    assert Path(cfg.models_dir) == default_models_dir()
    assert isinstance(cfg.hotkeys, Hotkeys)
    assert isinstance(cfg.overlay, OverlayGeometry)


def test_quality_renderer_defaults_auto_and_is_normalised(tmp_path: Path):
    """The default is "auto", and an unknown mode from a hand-edited config must
    still never launch the sidecar.

    "auto" is safe as a default because it degrades to the quick fill by itself:
    ``core/engines.build_quality_scheduler`` returns None when there is no sidecar
    interpreter and again when the models are absent, and nothing downloads them
    unasked.  Garbage still normalises to "off", which is what this pins.
    """
    from glasstranslate.config.settings import normalize_quality_renderer

    cfg = AppConfig()
    assert cfg.quality_renderer == "auto" and cfg.quality_sidecar_python == ""
    assert normalize_quality_renderer("Auto") == "auto"
    assert normalize_quality_renderer(" off ") == "off"
    for bad in ("cuda", "", None, 3, "on"):
        assert normalize_quality_renderer(bad) == "off"
    assert AppConfig.from_dict({"quality_renderer": "auto"}).quality_renderer == "auto"
    assert AppConfig.from_dict({"quality_renderer": "nonsense"}).quality_renderer == "off"
    assert AppConfig.from_dict({"quality_renderer": 7}).quality_renderer == "off"
    assert AppConfig.from_dict({"quality_sidecar_python": "  C:/py.exe "}).quality_sidecar_python == "C:/py.exe"
    path = tmp_path / "config.json"
    AppConfig(quality_renderer="auto", quality_sidecar_python="C:/py.exe").save(path)
    loaded = AppConfig.load(path)
    assert loaded.quality_renderer == "auto" and loaded.quality_sidecar_python == "C:/py.exe"


def test_roundtrip_save_load(tmp_path: Path):
    path = tmp_path / "sub" / "config.json"
    cfg = AppConfig(
        source_lang="ja",
        target_lang="en",
        ocr_device="cpu",
        translation_backend="identity",
        translation_api_key="secret",
        overlay_opacity=0.42,
        hide_original=False,
        font_family="Arial",
        overlay=OverlayGeometry(10, 20, 300, 400),
        refresh_hz=4.5,
        debounce_ms=250,
        min_confidence=0.7,
        change_threshold=0.05,
        tile_size=32,
        hotkeys=Hotkeys(toggle_grab="<ctrl>+g", toggle_running="<ctrl>+t", toggle_hidden="<ctrl>+h"),
        running_on_start=True,
    )
    saved = cfg.save(path)
    assert saved == path and path.exists()
    assert not path.with_suffix(".json.tmp").exists()  # atomic replace left no temp file

    loaded = AppConfig.load(path)
    assert loaded == cfg
    assert loaded.to_dict() == cfg.to_dict()


def test_saved_file_is_readable_json(tmp_path: Path):
    path = tmp_path / "config.json"
    AppConfig(target_lang="fr").save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["target_lang"] == "fr"
    assert data["overlay"] == {"x": 200, "y": 200, "w": 900, "h": 500}
    assert set(data["hotkeys"]) == {"toggle_grab", "toggle_running", "toggle_hidden"}


def test_missing_file_gives_defaults(tmp_path: Path):
    assert AppConfig.load(tmp_path / "nope.json") == AppConfig()


def test_corrupt_file_gives_defaults(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert AppConfig.load(path) == AppConfig()


def test_wrong_shape_gives_defaults(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"overlay": {"x": "not-an-int"}}), encoding="utf-8")
    assert AppConfig.load(path) == AppConfig()


def test_partial_file_keeps_defaults_for_missing_keys(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"target_lang": "es", "hotkeys": {"toggle_grab": "<ctrl>+q"}}), encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg.target_lang == "es"
    assert cfg.hotkeys.toggle_grab == "<ctrl>+q"
    assert cfg.hotkeys.toggle_running == Hotkeys().toggle_running
    assert cfg.overlay == OverlayGeometry()


def test_from_dict_ignores_unknown_keys_and_bad_nested_types():
    cfg = AppConfig.from_dict({"unknown": 1, "hotkeys": "bogus", "overlay": ["x"], "refresh_hz": 2.0})
    assert cfg.refresh_hz == 2.0
    assert cfg.hotkeys == Hotkeys()
    assert cfg.overlay == OverlayGeometry()


def test_from_dict_overlay_coerces_ints():
    cfg = AppConfig.from_dict({"overlay": {"x": "1", "y": 2.0, "w": 3, "h": 4, "extra": 9}})
    assert cfg.overlay == OverlayGeometry(1, 2, 3, 4)


def test_paths():
    assert default_config_path().name == "config.json"
    assert default_models_dir() == project_root() / "models"
    assert (project_root() / "glasstranslate").is_dir()
