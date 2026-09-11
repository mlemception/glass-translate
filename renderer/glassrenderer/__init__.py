"""GlassTranslate quality renderer sidecar.

A separate local process that redraws the erased regions of a manga panel:
LaMa for structure, then SDXL img2img with a line-art ControlNet, composited
back under a feathered mask.  It lives outside the app because it needs torch,
CUDA and diffusers, and the app must stay torch-free and buildable as one
PyInstaller exe.

The app talks to it over JSON on ``127.0.0.1`` with a per-session token; the
contract is ``renderer/PROTOCOL.md`` and :mod:`glassrenderer.protocol` is its
executable half.  ``--fake`` mode swaps the two GPU stages for numpy
stand-ins, so the protocol, the pipeline and the compositing run - and are
tested - in the app's own venv without a GPU.

Layout::

    protocol.py     the wire format: params, base64 PNG codec, validation
    compositing.py  the mask composite (the byte-identity guarantee) and the
                    512 / 768 / 1024 panel buckets
    fake.py         numpy stand-ins for the two stages
    pipeline.py     Renderer: health, and one job from bucket to composite
    server.py       the loopback HTTP front end
    stages/         the real torch stages (sidecar venv only, imported lazily)
"""
from __future__ import annotations

from typing import List

#: Package version; the *protocol* version is :data:`glassrenderer.protocol.VERSION`.
__version__ = "0.1.0"

__all__: List[str] = ["__version__"]
