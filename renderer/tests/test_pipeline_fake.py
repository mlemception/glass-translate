"""The fake pipeline end to end: no GPU, no torch, only numpy and OpenCV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glassrenderer import protocol as P
from glassrenderer.fake import FakeLama, FakeSdxl
from glassrenderer.pipeline import Renderer, RendererError

PAPER = 235
INK = 20
TIMING_KEYS = {"decode", "lama", "sdxl", "composite", "total"}


def _page(h: int = 64, w: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A paper panel with a black blob and the mask that covers it."""
    image = np.full((h, w, 3), PAPER, np.uint8)
    image[20:40, 24:44] = INK
    mask = np.zeros((h, w), np.uint8)
    mask[20:40, 24:44] = 255
    return image, mask


# --- the fake stages ---------------------------------------------------------


def test_fake_lama_fills_with_the_surrounding_ring_median() -> None:
    image, mask = _page()
    out = FakeLama().run(image, mask, P.DEFAULT_PARAMS)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert np.all(out[mask > 0] == PAPER)
    assert np.array_equal(out[mask == 0], image[mask == 0])


def test_fake_lama_fills_each_component_from_its_own_ring() -> None:
    image = np.full((64, 96, 3), PAPER, np.uint8)
    image[:, 48:] = 60  # a dark half
    mask = np.zeros((64, 96), np.uint8)
    mask[24:40, 8:24] = 255  # a blob on the light half
    mask[24:40, 64:80] = 255  # and one on the dark half
    out = FakeLama().run(image, mask, P.DEFAULT_PARAMS)
    assert np.all(out[24:40, 8:24] == PAPER)
    assert np.all(out[24:40, 64:80] == 60)


def test_fake_lama_leaves_an_empty_mask_alone() -> None:
    image, _ = _page()
    mask = np.zeros(image.shape[:2], np.uint8)
    assert np.array_equal(FakeLama().run(image, mask, P.DEFAULT_PARAMS), image)


def test_fake_sdxl_is_the_identity() -> None:
    image, mask = _page()
    out = FakeSdxl().run(image, mask, P.DEFAULT_PARAMS)
    assert np.array_equal(out, image)


def test_the_fake_stages_carry_the_contract_names() -> None:
    assert FakeLama().name == "lama" and FakeSdxl().name == "sdxl"


# --- health ------------------------------------------------------------------


def test_fake_health_matches_the_contract(models_dir: Path) -> None:
    health = Renderer(models_dir, fake=True).health()
    assert health["ok"] is True
    assert health["version"] == P.VERSION
    assert health["mode"] == "fake"
    assert health["device"] == "cpu"
    assert health["models"] == {"lama": "ready", "sdxl": "ready", "controlnet": "ready"}
    assert health["warm"] is False
    assert health["vram_total_mb"] is None and health["vram_used_mb"] is None
    assert isinstance(health["uptime_s"], float) and health["uptime_s"] >= 0.0


def test_real_mode_without_the_stages_module_reports_missing(models_dir: Path) -> None:
    # glassrenderer.stages is owned by another slice and may not exist yet; a
    # failed import must degrade, never crash.
    health = Renderer(models_dir, device="cpu", fake=False).health()
    assert health["mode"] == "real"
    assert set(health["models"].values()) <= {"missing", "loading", "ready"}


# --- inpaint -----------------------------------------------------------------


def test_fake_inpaint_fills_the_mask_and_keeps_the_rest(models_dir: Path) -> None:
    image, mask = _page()
    result, timings, stages = Renderer(models_dir, fake=True).inpaint(
        image, mask, P.DEFAULT_PARAMS
    )
    assert result.shape == image.shape and result.dtype == np.uint8
    assert np.array_equal(result[mask == 0], image[mask == 0])  # the hard guarantee
    assert not np.array_equal(result[mask > 0], image[mask > 0])  # something happened
    assert np.all(result[26:34, 30:38] == PAPER)  # the blob is gone
    assert stages == ["lama", "sdxl"]
    assert TIMING_KEYS <= set(timings)
    assert all(isinstance(v, float) and v >= 0.0 for v in timings.values())
    assert timings["total"] >= timings["lama"]


def test_inpaint_reports_the_decode_time_it_is_given(models_dir: Path) -> None:
    image, mask = _page()
    _, timings, _ = Renderer(models_dir, fake=True).inpaint(
        image, mask, P.DEFAULT_PARAMS, decode_ms=12.5
    )
    assert timings["decode"] == pytest.approx(12.5)
    assert timings["total"] >= 12.5


@pytest.mark.parametrize(
    "flags, expected",
    [
        ({"lama": False}, ["sdxl"]),
        ({"sdxl": False}, ["lama"]),
        ({"lama": False, "sdxl": False}, []),
    ],
)
def test_stages_can_be_skipped(
    models_dir: Path, flags: dict, expected: list[str]
) -> None:
    image, mask = _page()
    params = P.validate_params(flags)
    result, timings, stages = Renderer(models_dir, fake=True).inpaint(image, mask, params)
    assert stages == expected
    if not expected:
        assert np.array_equal(result, image)
    assert TIMING_KEYS <= set(timings)


def test_a_bucketed_panel_comes_back_at_the_source_size(models_dir: Path) -> None:
    image = np.full((400, 600, 3), PAPER, np.uint8)
    image[150:250, 200:400] = INK
    mask = np.zeros((400, 600), np.uint8)
    mask[150:250, 200:400] = 255
    result, timings, _ = Renderer(models_dir, fake=True).inpaint(
        image, mask, P.validate_params({"feather_px": 4})
    )
    assert result.shape == image.shape
    assert np.array_equal(result[mask == 0], image[mask == 0])
    assert abs(int(result[200, 300, 0]) - PAPER) <= 4
    assert timings["total"] > 0.0


def test_feather_zero_still_keeps_the_outside(models_dir: Path) -> None:
    image, mask = _page()
    result, _, _ = Renderer(models_dir, fake=True).inpaint(
        image, mask, P.validate_params({"feather_px": 0})
    )
    assert np.array_equal(result[mask == 0], image[mask == 0])


def test_inpaint_rejects_a_mask_of_another_size(models_dir: Path) -> None:
    image, _ = _page()
    with pytest.raises(RendererError, match="same size"):
        Renderer(models_dir, fake=True).inpaint(
            image, np.zeros((8, 8), np.uint8), P.DEFAULT_PARAMS
        )


def test_real_mode_inpaint_without_the_stages_module_raises(models_dir: Path) -> None:
    image, mask = _page()
    with pytest.raises(RendererError):
        Renderer(models_dir, device="cpu", fake=False).inpaint(image, mask, P.DEFAULT_PARAMS)


def test_a_broken_composite_is_caught_and_not_returned(
    models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from glassrenderer import pipeline

    image, mask = _page()
    monkeypatch.setattr(
        pipeline, "composite", lambda src, out, m, feather: np.zeros_like(src)
    )
    with pytest.raises(RendererError, match="outside the mask"):
        Renderer(models_dir, fake=True).inpaint(image, mask, P.DEFAULT_PARAMS)
