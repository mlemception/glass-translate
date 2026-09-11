"""Persistent settings.  Loaded once at startup, saved whenever the control
window changes something.  Location: %APPDATA%/GlassTranslate/config.json on
Windows, ~/.config/glasstranslate/config.json elsewhere; downloaded models and
logs sit in %LOCALAPPDATA%/GlassTranslate (:func:`user_data_dir`).

**Portable bundle**: a file named :data:`PORTABLE_MARKER` next to the running
executable (the checkout's root in a source tree) makes that folder the single
place the app reads and writes - ``models/``, ``config/config.json``, ``logs/``
and the secrets file in ``config/`` - so an extracted zip needs no installer and
leaves nothing in the user profile.  ``models_dir`` is then *stored* relative to
that root (``"models"``) and :func:`resolve_models_dir` anchors it again on load,
so moving the folder needs no config edit; a portable config whose ``models_dir``
is an absolute path that no longer exists falls back to the default for the same
reason.  Without the marker every path below - that fallback included - is
exactly what it was before.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

# Presence of this file (its content is informational) makes a bundle portable.
PORTABLE_MARKER = "portable.txt"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def app_dir() -> Path:
    """The folder the application runs from: the exe's directory when frozen,
    the checkout otherwise.  Never the PyInstaller ``_MEI*`` extraction
    directory, which vanishes at exit."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return project_root()


def portable_root() -> Optional[Path]:
    """The portable bundle's root - :func:`app_dir` when it holds
    :data:`PORTABLE_MARKER` - or None for a normal install."""
    root = app_dir()
    return root if (root / PORTABLE_MARKER).is_file() else None


def is_portable() -> bool:
    """True when the app runs from a portable bundle (see :func:`portable_root`)."""
    return portable_root() is not None


def default_config_path() -> Path:
    root = portable_root()
    if root is not None:
        return root / "config" / "config.json"
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / "GlassTranslate" / "config.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "GlassTranslate" / "config.json"
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "glasstranslate" / "config.json"


def user_data_dir() -> Path:
    """Writable data directory (downloaded models, logs).

    The portable root when there is one; otherwise per user:
    ``%LOCALAPPDATA%/GlassTranslate`` on Windows and next to the config file
    elsewhere.  Used by the packaged build, whose ``project_root()`` is the
    PyInstaller extraction directory and vanishes at exit.
    """
    root = portable_root()
    if root is not None:
        return root
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "GlassTranslate"
    return default_config_path().parent


def default_models_dir() -> Path:
    """Where Argos packages and the other model stores are installed by default.

    ``<portable root>/models`` in a portable bundle; a source checkout uses
    ``<project>/models``; the frozen exe (``sys.frozen``) uses
    :func:`user_data_dir` because its project root is a temporary ``_MEI*``
    directory.
    """
    root = portable_root()
    if root is not None:
        return root / "models"
    if getattr(sys, "frozen", False):
        return user_data_dir() / "models"
    return project_root() / "models"


def logs_dir() -> Path:
    """Where the app's and the sidecar's log files are written."""
    return user_data_dir() / "logs"


def cache_dir() -> Path:
    """Where disposable caches go: ``<root>\\cache`` in portable mode, else the user data dir.

    Qt would otherwise put its QML and pipeline caches under ``QStandardPaths::CacheLocation``
    (``%LOCALAPPDATA%``), which a portable bundle may not leave behind; see
    ``ui/app.py apply_portable_qt_environment``.
    """
    return user_data_dir() / "cache"


def secrets_dir() -> Path:
    """Where ``secrets.json`` lives: next to the portable config, else the user data dir."""
    root = portable_root()
    return root / "config" if root is not None else user_data_dir()


def _is_inside(candidate: Path, root: Path) -> bool:
    """True when ``candidate`` resolves to ``root`` or something under it."""
    try:
        return candidate.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):  # pragma: no cover - an unresolvable path is not inside
        return False


def resolve_models_dir(value: str) -> str:
    """The usable ``models_dir`` for a stored ``value`` (see :meth:`AppConfig.load`).

    Empty becomes :func:`default_models_dir`.  **Outside portable mode the
    function is the identity for every other value**: a normal install's store
    may simply sit on a drive that is not plugged in right now, and silently
    rewriting it would make the next :meth:`AppConfig.save` persist the change
    and lose the user's setting.

    In a portable bundle two rewrites apply, both of them contained by
    :func:`portable_root` - a config that travels inside the bundle must not be
    able to point the app at an arbitrary folder:

    * a relative path (only a portable config ever stores one) is anchored to
      :func:`user_data_dir`, which is how a moved folder finds its store again;
      one that escapes the root (``"../../x"``) falls back to the default;
    * an absolute path that is no longer a directory while the default is one
      falls back to the default too - the bundle was moved and an older config
      kept an absolute path.  An absolute path that *does* exist is kept, so a
      deliberately relocated store outside the bundle still works.
    """
    text = str(value or "").strip()
    if not text:
        return str(default_models_dir())
    root = portable_root()
    if root is None:
        return text
    default = default_models_dir()
    path = Path(text)
    if not path.is_absolute():
        candidate = (user_data_dir() / path).resolve()
        return str(candidate) if _is_inside(candidate, root) else str(default)
    if not path.is_dir() and default.is_dir():
        return str(default)
    return text


def _stored_models_dir(value: str) -> str:
    """``models_dir`` as it is written to disk: relative to the portable root when
    it *resolves* to somewhere inside it, so the folder can be moved; unchanged
    otherwise.  Resolving first matters: ``<root>/models/../../x`` starts with the
    root textually but leaves it, and must never be persisted as a relative value
    that :func:`resolve_models_dir` would then anchor somewhere else."""
    root = portable_root()
    text = str(value or "").strip()
    if root is None or not text:
        return text
    path = Path(text)
    if not _is_inside(path, root):
        return text
    relative = path.resolve().relative_to(root.resolve())
    return relative.as_posix() if relative.parts else text


# Quality renderer (render/quality.py): "off" = never launch the sidecar,
# "auto" = launch it when an interpreter for it can be found.
QUALITY_RENDERER_MODES = ("off", "auto")


def normalize_quality_renderer(value: object) -> str:
    """Coerce a stored / typed quality-renderer mode to a known one ("off" otherwise)."""
    text = str(value or "").strip().lower()
    return text if text in QUALITY_RENDERER_MODES else QUALITY_RENDERER_MODES[0]


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
    # Quality renderer (generative fill of free text over artwork, run by the
    # torch-using sidecar in ``renderer/``): "off" | "auto".  The optional
    # interpreter path overrides the lookup in ``render/quality.find_sidecar_python``.
    quality_renderer: str = "off"
    quality_sidecar_python: str = ""
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
        # Paths are resolved here and never in from_dict, which stays a pure parser:
        # a relative models_dir follows the portable root and a stale absolute one
        # falls back to the default.
        cfg.models_dir = resolve_models_dir(cfg.models_dir)
        return cfg

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["models_dir"] = _stored_models_dir(self.models_dir)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
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
        # Enum-valued fields are normalised rather than trusted: an unknown
        # quality-renderer mode must never launch a sidecar.
        cfg.quality_renderer = normalize_quality_renderer(cfg.quality_renderer)
        cfg.quality_sidecar_python = str(cfg.quality_sidecar_python or "").strip()
        return cfg
