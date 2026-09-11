"""Wire format of the sidecar protocol (``renderer/PROTOCOL.md``, version 1).

Everything that crosses the loopback socket is described here and nowhere
else: the parameter defaults, the base64-PNG image codec, the request schema
with its error strings, and the response envelope.  The module imports numpy
and Pillow only, so both sides of the contract - the torch-free app and the
sidecar - can use it, and the whole schema is testable without a GPU.

Images are PNG: ``image`` is 8-bit RGB, ``mask`` is 8-bit greyscale where
**255 = fill** and 0 = keep.  Every validation failure raises
:class:`ProtocolError` with a ``"<field>: <what>"`` message that the server
returns verbatim as ``400 {"error": ...}``; it never leaks a traceback.
"""
from __future__ import annotations

import base64
import binascii
import io
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

VERSION = "1"

#: The contract's defaults.  Read-only: :func:`validate_params` copies it.
DEFAULT_PARAMS: Mapping[str, Any] = MappingProxyType(
    {
        "lama": True,
        "sdxl": True,
        "strength": 0.4,
        "steps": 24,
        "controlnet_scale": 0.8,
        "guidance": 4.0,
        "seed": 0,
        "feather_px": 2,
        "prompt": "",
        "negative_prompt": "",
    }
)

MAX_BODY_BYTES = 64 * 1024 * 1024  # the server's Content-Length cap
MAX_JOB_ID_CHARS = 128
MAX_PROMPT_CHARS = 2000
#: PNG level 1: the panels are large and the link is a loopback socket, so a
#: millisecond of latency costs more than a kilobyte of pipe.
PNG_COMPRESS_LEVEL = 1

_BOOL_FIELDS: Tuple[str, ...] = ("lama", "sdxl")
_INT_RANGES: Mapping[str, Tuple[int, int]] = {"steps": (1, 100), "feather_px": (0, 32)}
_FLOAT_RANGES: Mapping[str, Tuple[float, float]] = {
    "strength": (0.0, 1.0),
    "controlnet_scale": (0.0, 2.0),
    "guidance": (0.0, 20.0),
}
_STR_FIELDS: Tuple[str, ...] = ("prompt", "negative_prompt")


class ProtocolError(ValueError):
    """A request does not satisfy the contract; answered with ``400``."""


# --- images ------------------------------------------------------------------


def _encode(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_png_rgb(image: np.ndarray) -> str:
    """Encode an ``HxWx3`` uint8 RGB array as a base64 PNG string."""
    if image.dtype != np.uint8:
        raise ProtocolError(f"image: must be uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ProtocolError(f"image: must be HxWx3, got shape {image.shape}")
    if image.size == 0:
        raise ProtocolError("image: must not be empty")
    return _encode(Image.fromarray(np.ascontiguousarray(image), "RGB"))


def encode_png_mask(mask: np.ndarray) -> str:
    """Encode an ``HxW`` uint8 mask (255 = fill) as a base64 PNG string."""
    if mask.dtype != np.uint8:
        raise ProtocolError(f"mask: must be uint8, got {mask.dtype}")
    if mask.ndim != 2:
        raise ProtocolError(f"mask: must be HxW, got shape {mask.shape}")
    if mask.size == 0:
        raise ProtocolError("mask: must not be empty")
    return _encode(Image.fromarray(np.ascontiguousarray(mask), "L"))


#: Largest image accepted by ``/inpaint`` before decoding (the biggest bucket is 1024 px on
#: the long side; a whole page at print resolution is ~6 MP).
MAX_IMAGE_PIXELS = 32_000_000


def _open_png(data: Any, field_name: str) -> Image.Image:
    if not isinstance(data, str):
        raise ProtocolError(f"{field_name}: must be a base64 string")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(f"{field_name}: not valid base64 ({exc})") from None
    try:
        image = Image.open(io.BytesIO(raw))
        # The body cap bounds the compressed bytes, not the pixels a PNG header can
        # declare: refuse anything beyond the largest panel before it is decoded.
        width, height = image.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ProtocolError(f"{field_name}: {width}x{height} exceeds {MAX_IMAGE_PIXELS} pixels")
        image.load()
    except ProtocolError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise ProtocolError(f"{field_name}: not a PNG image") from None
    if image.format != "PNG":
        raise ProtocolError(f"{field_name}: not a PNG image (got {image.format})")
    return image


def decode_png_rgb(data: Any, *, field: str = "image") -> np.ndarray:
    """Decode a base64 PNG string into an ``HxWx3`` uint8 RGB array."""
    image = _open_png(data, field)
    if image.mode != "RGB":
        raise ProtocolError(f"{field}: expected an 8-bit RGB PNG, got mode {image.mode}")
    return np.asarray(image, dtype=np.uint8)


def decode_png_mask(data: Any, *, field: str = "mask") -> np.ndarray:
    """Decode a base64 PNG string into an ``HxW`` uint8 mask."""
    image = _open_png(data, field)
    if image.mode != "L":
        raise ProtocolError(
            f"{field}: expected an 8-bit greyscale PNG, got mode {image.mode}"
        )
    return np.asarray(image, dtype=np.uint8)


# --- params ------------------------------------------------------------------


def _check_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError(f"params.{name}: must be a boolean")
    return value


def _check_int(name: str, value: Any, low: int | None, high: int | None) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"params.{name}: must be an integer")
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise ProtocolError(f"params.{name}: must be an integer")
    if low is not None and high is not None and not low <= value <= high:
        raise ProtocolError(f"params.{name}: must be between {low} and {high}")
    return value


def _check_float(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"params.{name}: must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError(f"params.{name}: must be a number")
    if not low <= number <= high:
        raise ProtocolError(f"params.{name}: must be between {low} and {high}")
    return number


def _check_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ProtocolError(f"params.{name}: must be a string")
    if len(value) > MAX_PROMPT_CHARS:
        raise ProtocolError(f"params.{name}: at most {MAX_PROMPT_CHARS} characters")
    return value


def validate_params(raw: Any) -> Dict[str, Any]:
    """Merge ``raw`` over :data:`DEFAULT_PARAMS` and validate every field.

    ``None`` and ``{}`` both yield the defaults.  Unknown keys are refused so
    a typo in the client never silently loses a setting.
    """
    if raw is None:
        return dict(DEFAULT_PARAMS)
    if not isinstance(raw, dict):
        raise ProtocolError("params: expected a JSON object")
    for key in raw:
        if key not in DEFAULT_PARAMS:
            raise ProtocolError(f"params: unknown key {key!r}")
    params = dict(DEFAULT_PARAMS)
    for name in _BOOL_FIELDS:
        if name in raw:
            params[name] = _check_bool(name, raw[name])
    for name, (low_i, high_i) in _INT_RANGES.items():
        if name in raw:
            params[name] = _check_int(name, raw[name], low_i, high_i)
    if "seed" in raw:
        params["seed"] = _check_int("seed", raw["seed"], None, None)
    for name, (low_f, high_f) in _FLOAT_RANGES.items():
        if name in raw:
            params[name] = _check_float(name, raw[name], low_f, high_f)
    for name in _STR_FIELDS:
        if name in raw:
            params[name] = _check_str(name, raw[name])
    return params


# --- messages ----------------------------------------------------------------


def _require(payload: Mapping[str, Any], name: str) -> Any:
    if name not in payload:
        raise ProtocolError(f"{name}: required")
    return payload[name]


@dataclass(frozen=True)
class InpaintRequest:
    """A validated ``POST /inpaint`` body."""

    job_id: str
    image: np.ndarray
    mask: np.ndarray
    params: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))

    @classmethod
    def from_payload(cls, payload: Any) -> "InpaintRequest":
        """Validate a decoded JSON body, or raise :class:`ProtocolError`."""
        if not isinstance(payload, dict):
            raise ProtocolError("body: expected a JSON object")
        job_id = _require(payload, "job_id")
        if not isinstance(job_id, str):
            raise ProtocolError("job_id: must be a string")
        if not job_id or len(job_id) > MAX_JOB_ID_CHARS:
            raise ProtocolError(f"job_id: at most {MAX_JOB_ID_CHARS} characters")
        image = decode_png_rgb(_require(payload, "image"), field="image")
        mask = decode_png_mask(_require(payload, "mask"), field="mask")
        if mask.shape != image.shape[:2]:
            raise ProtocolError(
                f"mask: {mask.shape[1]}x{mask.shape[0]} does not match image "
                f"{image.shape[1]}x{image.shape[0]}"
            )
        return cls(job_id, image, mask, validate_params(payload.get("params")))


@dataclass(frozen=True)
class InpaintResponse:
    """A ``POST /inpaint`` result, ready to be serialised."""

    job_id: str
    image: np.ndarray
    timings_ms: Mapping[str, float]
    mode: str
    stages: Sequence[str]

    def to_payload(self) -> Dict[str, Any]:
        """The JSON body of the contract, with the image as a base64 PNG."""
        height, width = self.image.shape[:2]
        return {
            "job_id": self.job_id,
            "image": encode_png_rgb(self.image),
            "timings_ms": {k: round(float(v), 1) for k, v in self.timings_ms.items()},
            "mode": self.mode,
            "stages": list(self.stages),
            "size": [int(width), int(height)],
        }


__all__: List[str] = [
    "VERSION",
    "DEFAULT_PARAMS",
    "MAX_BODY_BYTES",
    "MAX_JOB_ID_CHARS",
    "MAX_PROMPT_CHARS",
    "ProtocolError",
    "InpaintRequest",
    "InpaintResponse",
    "encode_png_rgb",
    "encode_png_mask",
    "decode_png_rgb",
    "decode_png_mask",
    "validate_params",
]
