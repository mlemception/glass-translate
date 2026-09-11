"""``Renderer.preload``: the server loads and warms the real stages off the request
path; fake mode is instant, and a real mode without torch or models fails softly."""
from __future__ import annotations

import numpy as np

from glassrenderer.pipeline import MODEL_KEYS, Renderer


def test_preload_in_fake_mode_builds_the_stages_and_health_is_ready(tmp_path):
    renderer = Renderer(tmp_path, fake=True)
    renderer.preload()
    health = renderer.health()
    assert health["models"] == {key: "ready" for key in MODEL_KEYS}
    assert health["load_error"] is None and health["warm"] is False  # nothing to compile in fake mode
    image = np.full((64, 64, 3), 200, np.uint8)
    mask = np.zeros((64, 64), np.uint8)
    mask[20:40, 20:40] = 255
    out, timings, stages = renderer.inpaint(image, mask, {})
    assert out.shape == image.shape and stages == ["lama", "sdxl"]
    assert np.array_equal(out[mask == 0], image[mask == 0])


def test_preload_in_real_mode_without_torch_or_models_fails_softly(tmp_path):
    renderer = Renderer(tmp_path, fake=False, device="cpu")
    renderer.preload()  # no torch in the app venv and no model files: no exception either way
    health = renderer.health()
    assert health["ok"] is True
    assert renderer._load_state in ("failed", "ready")
    if renderer._load_state == "failed":
        assert health["load_error"]
        assert all(value in ("missing", "ready") for value in health["models"].values())


def test_loading_state_is_visible_in_health_for_present_models(tmp_path, monkeypatch):
    renderer = Renderer(tmp_path, fake=False, device="cpu")
    import glassrenderer.stages as stages

    monkeypatch.setattr(stages, "model_status", lambda models_dir: {key: "ready" for key in MODEL_KEYS})
    renderer._load_state = "loading"
    assert renderer.health()["models"] == {key: "loading" for key in MODEL_KEYS}
    renderer._load_state = "ready"
    assert renderer.health()["models"] == {key: "ready" for key in MODEL_KEYS}
