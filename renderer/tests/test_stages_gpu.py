"""Opt-in GPU integration test for the real LaMa and SDXL stages.

Skipped unless ``GT_GPU_TESTS=1`` **and** the interpreter is the sidecar venv
with CUDA **and** the models are in ``--models-dir`` (default ``models``)::

    set GT_GPU_TESTS=1
    renderer\\.venv\\Scripts\\python -m pytest renderer/tests/test_stages_gpu.py -s

It runs one synthetic 512x512 panel through ``lama`` and then ``sdxl``, asserts
the sizes and the protocol's hard guarantee — after
:func:`glassrenderer.compositing.composite` every pixel where ``mask == 0`` is
byte-identical to the input — and prints the timings.

The bucketing and prompt helpers below need no GPU (this module imports torch
nowhere at import time), so they always run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest

RENDERER_ROOT = Path(__file__).resolve().parents[1]
if str(RENDERER_ROOT) not in sys.path:
    sys.path.insert(0, str(RENDERER_ROOT))

from glassrenderer import models as M  # noqa: E402
from glassrenderer.compositing import composite  # noqa: E402
from glassrenderer.stages import build_stages, model_status  # noqa: E402
from glassrenderer.stages.sdxl import DEFAULTS, NEGATIVE_PROMPT, POSITIVE_PROMPT, bucket_size  # noqa: E402

MODELS_DIR = os.environ.get("GT_QUALITY_MODELS", "models")
FEATHER_PX = 2
PANEL = 512


def _skip_reason() -> str:
    if os.environ.get("GT_GPU_TESTS") != "1":
        return "set GT_GPU_TESTS=1 to run the GPU integration test"
    try:
        import torch
    except ImportError:
        return "torch is not installed (run this with renderer/.venv)"
    if not torch.cuda.is_available():
        return "no CUDA device"
    if not M.models_ready(MODELS_DIR):
        return f"quality models are not in {M.quality_dir(MODELS_DIR)}"
    return ""


gpu_only = pytest.mark.skipif(bool(_skip_reason()), reason=_skip_reason() or "gpu")


# ------------------------------------------------------------------- CPU
def test_bucket_size_rounds_the_long_side_onto_a_bucket() -> None:
    assert bucket_size(300, 250) == (512, 432)
    assert bucket_size(700, 700) == (768, 768)
    assert bucket_size(1600, 1067) == (1024, 688)
    assert bucket_size(200, 200, 1024) == (1024, 1024)
    for height, width in ((300, 250), (700, 700), (1600, 1067), (17, 999)):
        out_h, out_w = bucket_size(height, width)
        assert out_h % 16 == 0 and out_w % 16 == 0


def test_the_built_in_prompt_does_not_fight_a_manga_clean_up() -> None:
    """Stock anime negatives contain "monochrome, greyscale, comic" - ours must not."""
    for tag in ("monochrome", "greyscale", "comic", "halftone"):
        assert tag in POSITIVE_PROMPT
        assert tag not in NEGATIVE_PROMPT
    for tag in ("text", "speech bubble", "watermark", "spot color"):
        assert tag in NEGATIVE_PROMPT
    assert DEFAULTS["strength"] == 0.4 and DEFAULTS["steps"] == 24
    assert DEFAULTS["controlnet_scale"] == 0.8 and DEFAULTS["guidance"] == 4.0


def test_model_status_reports_the_three_groups() -> None:
    assert set(model_status(MODELS_DIR)) == {"lama", "sdxl", "controlnet"}
    assert model_status(MODELS_DIR, fake=True) == {
        "lama": "ready",
        "sdxl": "ready",
        "controlnet": "ready",
    }


# ------------------------------------------------------------------- GPU
def _synthetic_panel() -> Tuple[np.ndarray, np.ndarray]:
    """A white panel with ruled ink lines, a tone field and a 'text' block."""
    panel = np.full((PANEL, PANEL, 3), 245, np.uint8)
    panel[::16, :] = 25  # horizontal ruling
    panel[:, ::24] = 70  # vertical ruling
    panel[320:480, 32:480] = 140  # a flat tone field
    panel[120:260, 140:300] = 15  # the block we are about to erase
    mask = np.zeros((PANEL, PANEL), np.uint8)
    mask[120:260, 140:300] = 255
    return panel, mask


@pytest.fixture(scope="module")
def stages() -> Tuple[Any, Any]:
    lama, sdxl = build_stages(MODELS_DIR, "cuda", fake=False)
    lama.load()
    print(f"\nlama load {lama.load_ms:.0f} ms, jit warm-up {lama.warm_up():.0f} ms")
    sdxl.load()
    print(f"sdxl load {sdxl.load_ms:.0f} ms")
    return lama, sdxl


@gpu_only
def test_lama_fills_the_mask_and_keeps_everything_else(stages: Tuple[Any, Any]) -> None:
    lama, _ = stages
    panel, mask = _synthetic_panel()

    filled = lama.run(panel, mask, {})
    print(f"lama {PANEL}px: {lama.last_ms:.1f} ms")

    assert filled.shape == panel.shape and filled.dtype == np.uint8
    assert filled[mask > 0].mean() > 120, "the ink block should be gone"
    out = composite(panel, filled, mask, FEATHER_PX)
    keep = mask == 0
    assert (out[keep] == panel[keep]).all(), "compositing must not touch unmasked pixels"


@gpu_only
def test_sdxl_returns_the_input_size_and_composites_byte_exactly(stages: Tuple[Any, Any]) -> None:
    lama, sdxl = stages
    panel, mask = _synthetic_panel()

    filled = lama.run(panel, mask, {})
    rendered = sdxl.run(filled, mask, {"steps": 12, "strength": 0.4, "seed": 11})
    print(f"sdxl {PANEL}px 12 steps: {sdxl.last_ms:.1f} ms")

    assert rendered.shape == panel.shape and rendered.dtype == np.uint8
    out = composite(panel, rendered, mask, FEATHER_PX)
    keep = mask == 0
    assert (out[keep] == panel[keep]).all()
    assert not (out[mask > 0] == panel[mask > 0]).all(), "the masked region must change"


@gpu_only
def test_a_non_square_panel_comes_back_at_its_own_size(stages: Tuple[Any, Any]) -> None:
    lama, sdxl = stages
    panel = np.full((300, 220, 3), 240, np.uint8)
    panel[::12, :] = 30
    mask = np.zeros((300, 220), np.uint8)
    mask[80:180, 60:160] = 255

    filled = lama.run(panel, mask, {})
    rendered = sdxl.run(filled, mask, {"steps": 8, "strength": 0.35})

    assert filled.shape == (300, 220, 3)
    assert rendered.shape == (300, 220, 3)


@gpu_only
def test_the_same_seed_reproduces_the_same_render(stages: Tuple[Any, Any]) -> None:
    _, sdxl = stages
    panel, mask = _synthetic_panel()
    params = {"steps": 8, "strength": 0.4, "seed": 1234}

    first = sdxl.run(panel, mask, params)
    second = sdxl.run(panel, mask, params)

    assert np.array_equal(first, second)


@gpu_only
def test_peak_vram_stays_under_the_budget(stages: Tuple[Any, Any]) -> None:
    """The plan budgets < 12 GB; VAE tiling is what keeps 1024 inside it."""
    import torch

    lama, sdxl = stages
    panel = np.full((1024, 1024, 3), 240, np.uint8)
    panel[::9, :] = 40
    mask = np.zeros((1024, 1024), np.uint8)
    mask[300:600, 300:700] = 255

    sdxl.run(panel, mask, {"steps": 8, "strength": 0.4})  # settle the allocator
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    filled = lama.run(panel, mask, {})
    sdxl.run(filled, mask, {"steps": 24, "strength": 0.4})
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"lama 1024px {lama.last_ms:.1f} ms, sdxl 1024px {sdxl.last_ms:.1f} ms, peak {peak_gb:.2f} GB")

    assert peak_gb < 12.0
