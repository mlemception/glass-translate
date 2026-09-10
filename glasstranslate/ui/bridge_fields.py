"""Config-field plumbing and pure formatting helpers of the control-window bridge.

Split out of :mod:`glasstranslate.ui.control` (which re-exports every name here) so the
bridge/window module stays readable: nothing in this file touches Qt objects beyond building
``Property`` descriptors.  ``_CONFIG_FIELDS`` is the single table that maps a QML property name
to the ``AppConfig`` attribute it edits (dotted for ``hotkeys.x``) and the coercion applied to a
value coming from QML; :func:`_config_property` turns one entry into a read/write Qt property
that routes writes through ``ControlBridge._set_config``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Property, Signal

from ..config.settings import AppConfig
from ..core.types import PipelineStats

__all__ = [
    "CUDA_FROZEN_HINT",
    "LANGUAGES",
    "MAX_SERIES_NAME_CHARS",
    "MAX_TEMPLATE_CHARS",
    "download_label",
    "download_progress",
    "format_stats",
    "is_frozen",
    "translate_device_items",
]

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

_OCR_DEVICES = ("auto", "gpu", "cpu")
_TRANSLATE_DEVICES = ("auto", "cuda", "cpu")
_TRANSLATE_DEVICES_FROZEN = ("auto", "cpu")
_ONLINE_BACKENDS = {"libretranslate", "gemini"}
CUDA_FROZEN_HINT = "CUDA is not available in the packaged build; translation runs on CPU"
# Series context (F5): caps mirror glasstranslate.translate.context (imported lazily by the coercers).
MAX_SERIES_NAME_CHARS = 80
MAX_TEMPLATE_CHARS = 4000


def is_frozen() -> bool:
    """True inside the PyInstaller exe."""
    return bool(getattr(sys, "frozen", False))


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


# ------------------------------------------------------------------------- config plumbing
def _clamp(lo: float, hi: float) -> Callable[[Any], float]:
    return lambda v: min(hi, max(lo, float(v)))


def _strip(v: Any) -> str:
    return str(v).strip()


def _series_name(v: Any) -> str:
    """The same normalisation the prompt engine applies (trim, collapse, strip control chars, cap)."""
    from ..translate.context import normalize_series_name

    return normalize_series_name(v)


def _capped_text(limit: int) -> Callable[[Any], str]:
    """Multi-line text: normalise line endings, strip, cut to ``limit`` characters."""

    def coerce(v: Any) -> str:
        text = str(v).replace("\r\n", "\n").replace("\r", "\n").strip()
        return text[:limit].rstrip()

    return coerce


def _timeout(v: Any) -> float:
    return min(300.0, max(1.0, float(v)))


def _retries(v: Any) -> int:
    return int(min(10, max(0, round(float(v)))))


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
    "geminiModel": _Field("gemini_model", _strip),
    "geminiApiFormat": _Field("gemini_api_format", str),
    "geminiBaseUrl": _Field("gemini_base_url", _strip),
    "geminiTimeoutS": _Field("gemini_timeout_s", _timeout),
    "geminiMaxRetries": _Field("gemini_max_retries", _retries),
    "seriesName": _Field("series_name", _series_name),
    "seriesPromptTemplate": _Field("series_prompt_template", _capped_text(MAX_TEMPLATE_CHARS)),
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
    """A read/write QML property backed by ``AppConfig`` through ``ControlBridge._set_config``."""
    attr = _CONFIG_FIELDS[name].attr

    def fget(self: Any) -> Any:
        return _cfg_get(self._cfg, attr)

    def fset(self: Any, value: Any) -> None:
        self._set_config(name, value)

    return Property(qtype, fget, fset, notify=notify)
