"""Pick a screen-capture backend."""
from __future__ import annotations

import logging

from glasstranslate.core.interfaces import ScreenCapture

from .backends import DXCamCapture, MSSCapture

__all__ = ["create_capture", "available_backends"]

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, type[ScreenCapture]] = {
    DXCamCapture.name: DXCamCapture,
    MSSCapture.name: MSSCapture,
}


def available_backends() -> list[str]:
    """Names accepted by :func:`create_capture` besides ``"auto"``."""
    return list(_BACKENDS)


def create_capture(preferred: str = "auto") -> ScreenCapture:
    """Create a capture backend.

    Args:
        preferred: ``"auto"`` tries dxcam first and falls back to mss.  A
            backend name (``"dxcam"`` / ``"mss"``) tries that backend first
            and still falls back to mss when it cannot be initialised.

    Raises:
        ValueError: unknown backend name.
        RuntimeError: no backend could be initialised (mss import failure).
    """
    key = preferred.lower().strip()
    if key == "auto":
        order = [DXCamCapture.name, MSSCapture.name]
    elif key in _BACKENDS:
        order = [key] + [n for n in _BACKENDS if n != key]
    else:
        raise ValueError(f"unknown capture backend {preferred!r}; choose from auto, {', '.join(_BACKENDS)}")

    errors: list[str] = []
    for name in order:
        try:
            backend = _BACKENDS[name]()
        except Exception as exc:
            logger.info("capture backend %s unavailable: %s", name, exc)
            errors.append(f"{name}: {exc}")
            continue
        logger.info("using capture backend %s", name)
        return backend
    raise RuntimeError("no screen capture backend available: " + "; ".join(errors))
