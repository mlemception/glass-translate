"""Generative stages of the quality renderer and the factory the server uses.

``renderer/PROTOCOL.md`` fixes two entry points:

``build_stages(models_dir, device, *, fake) -> (lama, sdxl)``
    The pair :mod:`glassrenderer.pipeline` runs in order.  The real stages are
    returned **constructed but not loaded**: importing torch is cheap compared
    with 9.3 GB of weights, and the protocol wants ``READY <port>`` on stdout
    *before* the models are resident so ``/health`` can report ``"loading"``.
    Call ``stage.load()`` from the server's loading thread (or just call
    ``run``, which loads on first use).

``model_status(models_dir) -> {"lama": ..., "sdxl": ..., "controlnet": ...}``
    ``"ready"`` or ``"missing"`` per group, straight off the pinned table, for
    the ``/health`` body.  ``"loading"`` is the server's state, not the store's.

Nothing here imports torch at module level, so ``--fake`` really does run
without it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union

# The sidecar never talks to the Hub at run time: every file is pinned and verified locally,
# and diffusers must not fetch or execute remote code (belt and braces with the app launcher).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_DISABLE_REMOTE_CODE", "true")

from .. import models as M  # noqa: E402
from . import lineart  # noqa: E402
from .lama import LamaStage  # noqa: E402
from .sdxl import SdxlStage  # noqa: E402

__all__ = [
    "LamaStage",
    "SdxlStage",
    "build_stages",
    "model_status",
    "lineart",
]


def build_stages(
    models_dir: Union[str, Path], device: str, *, fake: bool
) -> Tuple[Any, Any]:
    """The ``(lama, sdxl)`` stage pair for ``models_dir`` on ``device``.

    ``fake=True`` returns the torch-free numpy stand-ins from
    :mod:`glassrenderer.fake` so the whole protocol is testable without a GPU.
    """
    if fake:
        return _build_fake_stages()
    return LamaStage(models_dir, device), SdxlStage(models_dir, device)


def _build_fake_stages() -> Tuple[Any, Any]:
    """Delegate to :mod:`glassrenderer.fake` (imported lazily: no torch there)."""
    from .. import fake

    builder = getattr(fake, "build_fake_stages", None)
    if builder is not None:
        return builder()
    return fake.FakeLamaStage(), fake.FakeSdxlStage()  # type: ignore[attr-defined]


def model_status(models_dir: Union[str, Path], *, fake: bool = False) -> Dict[str, str]:
    """``{"lama": "ready" | "missing", "sdxl": ..., "controlnet": ...}``.

    In fake mode there are no files to miss, so every group reports ``"ready"``.
    """
    if fake:
        return {group: "ready" for group in M.GROUPS}
    return M.group_status(models_dir)
