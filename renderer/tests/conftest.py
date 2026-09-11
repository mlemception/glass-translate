"""Pytest fixtures for the sidecar tests.

The sidecar is never installed into a venv for the tests: this module puts
``renderer/`` (the directory holding the ``glassrenderer`` package) at the
front of ``sys.path`` so ``import glassrenderer`` works from the project root,
and it exports the same path as :data:`RENDERER_DIR` for the tests that spawn
the server as a subprocess (they pass it through ``PYTHONPATH``).

Everything here must run under the *main* venv: numpy, OpenCV, Pillow and the
standard library only, never torch.
"""
from __future__ import annotations

import secrets
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

RENDERER_DIR = Path(__file__).resolve().parents[1]
if str(RENDERER_DIR) not in sys.path:
    sys.path.insert(0, str(RENDERER_DIR))


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so a failure is always reproducible."""
    return np.random.default_rng(20260910)


@pytest.fixture
def token() -> str:
    """A per-test session token in the shape the app generates."""
    return secrets.token_hex(32)


@pytest.fixture
def models_dir(tmp_path: Path) -> Iterator[Path]:
    """An empty models directory (fake mode never reads it)."""
    path = tmp_path / "quality"
    path.mkdir()
    yield path
