"""Shared fixtures (docs/GLASS_DESIGN.md section 7).

* Qt runs on the ``offscreen`` platform (set before any ``QApplication`` exists; the
  scene graph there is software-only, so shaders never compile and nothing
  render-dependent is asserted).
* The Qt resources are compiled once per session with ``tools.build_resources.build()``
  - at conftest import time, so ``glasstranslate.ui.control`` (which imports the
  generated ``resources_rc``) never sees a stale bundle even when a test module imports
  it at collection; the ``resources_built`` fixture hands out the result and ``qapp``
  depends on it and returns the shared ``QApplication``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RC: Optional[Path] = None


def _build_resources_once() -> Path:
    global _RC
    if _RC is None:
        from tools.build_resources import build

        _RC = build()
    return _RC


_build_resources_once()


@pytest.fixture(scope="session")
def resources_built() -> Path:
    """Path of the freshly generated ``glasstranslate/ui/resources_rc.py``."""
    return _build_resources_once()


@pytest.fixture(scope="session")
def qapp(resources_built: Path):
    """The session-wide offscreen ``QApplication`` (resources compiled first)."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
