"""Persistent settings.  Loaded once at startup, saved whenever the control
window changes something.  Location: %APPDATA%/GlassTranslate/config.json on
Windows, ~/.config/glasstranslate/config.json elsewhere.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


def default_config_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "GlassTranslate" / "config.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "GlassTranslate" / "config.json"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "glasstranslate" / "config.json"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def user_data_dir() -> Path:
    """Per-user writable data directory (downloaded models, logs).

    ``%LOCALAPPDATA%/GlassTranslate`` on Windows; next to the config file
    elsewhere.  Used by the packaged build, whose ``project_root()`` is the
    PyInstaller extraction directory and vanishes at exit.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "GlassTranslate"
    return default_config_path().parent


def default_models_dir() -> Path:
    """Where Argos packages are installed by default.

    A source checkout uses ``<project>/models``; the frozen exe
    (``sys.frozen``) uses :func:`user_data_dir` because its project root is
    a temporary ``_MEI*`` directory.
    """
    if getattr(sys, "frozen", False):
        return user_data_dir() / "models"
    return project_root() / "models"


@dataclass
class Hotkeys:
    toggle_grab: str = "<ctrl>+<alt>+g"
    toggle_running: str = "<ctrl>+<alt>+t"
    toggle_hidden: str = "<ctrl>+<alt>+h"


@dataclass
class OverlayGeometry:
    x: int = 200
    y: int = 200
    w: int = 900
    h: int = 500


@dataclass
class AppConfig:
    # languages
    source_lang: str = "auto"  # "auto" or ISO-639-1
    target_lang: str = "en"
    # engines
    # "mangaocr" (manga-ocr ONNX recogniser, falls back to PaddleOCR when its
    # models are missing) or "paddleocr" (rapidocr running PP-OCRv5 via ONNX;
    # "rapidocr" is accepted as an alias).  See ocr/factory.py.
    ocr_engine: str = "mangaocr"
    ocr_device: str = "auto"  # auto | gpu | cpu
    translation_backend: str = "argos"  # argos | libretranslate | gemini | identity
    translation_api_key: str = ""
    translation_api_url: str = "https://libretranslate.com"
    translate_device: str = "auto"  # auto | cuda | cpu
    models_dir: str = field(default_factory=lambda: str(default_models_dir()))
    # Gemini provider (the API key lives in config/secrets.py, never here)
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_format: str = "native"  # native (x-goog-api-key, generateContent) | openai (Bearer, chat/completions)
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_timeout_s: float = 30.0
    gemini_max_retries: int = 3
    # Series context (Gemini only): the quick series name from the main tab and the editable
    # system-prompt template with the [Series Name] placeholder; "" = the built-in template.
    series_name: str = ""
    series_prompt_template: str = ""
    # overlay
    overlay_opacity: float = 0.10  # background alpha, 0..1
    hide_original: bool = True  # paint bg-colored box under translation
    font_family: str = "Segoe UI"
    # Manga typesetting: group OCR lines into bubbles/blocks, translate the
    # whole utterance and letter it in the bundled comic font (see
    # render/layout.py).  Off = one translation per OCR line.
    manga_mode: bool = True
    # Letter typeset blocks in capitals, as printed English comics do.
    uppercase: bool = True
    overlay: OverlayGeometry = field(default_factory=OverlayGeometry)
    # pipeline
    refresh_hz: float = 10.0  # capture polling rate
    debounce_ms: int = 120  # wait for the screen to settle before OCR
    min_confidence: float = 0.5
    change_threshold: float = 0.02  # fraction of tile pixels that must change
    tile_size: int = 64
    hotkeys: Hotkeys = field(default_factory=Hotkeys)
    running_on_start: bool = False

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        path = path or default_config_path()
        cfg = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                cfg = cls.from_dict(data)
            except Exception:
                # Corrupt config: fall back to defaults rather than crash.
                cfg = cls()
        return cfg

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        cfg = cls()
        for f in fields(cls):
            if f.name not in data:
                continue
            v = data[f.name]
            if f.name == "hotkeys":
                if isinstance(v, dict):
                    cfg.hotkeys = Hotkeys(
                        **{k: str(v[k]) for k in v if k in Hotkeys.__dataclass_fields__}
                    )
                # Non-dict values are malformed: keep the defaults.
            elif f.name == "overlay":
                if isinstance(v, dict):
                    try:
                        cfg.overlay = OverlayGeometry(
                            **{k: int(v[k]) for k in v if k in OverlayGeometry.__dataclass_fields__}
                        )
                    except (TypeError, ValueError):
                        pass
            elif isinstance(v, type(getattr(cfg, f.name))) or (
                isinstance(v, (int, float)) and isinstance(getattr(cfg, f.name), (int, float))
                and not isinstance(v, bool)
            ):
                setattr(cfg, f.name, v)
            # Values of the wrong type are ignored so a hand-edited config
            # cannot put the app into an inconsistent state.
        return cfg
