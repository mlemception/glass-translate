"""The frozen sidecar's entry shim (``packaging/glassrenderer_entry.py``).

``glassrenderer/__main__.py`` cannot be frozen directly - its relative imports
need the package context - so PyInstaller is pointed at a shim instead.  This
suite guards the two things that shim owns and that a build failure would only
surface as a broken exe: the argv it forwards, and the offline environment it
installs *before* ``huggingface_hub`` is ever imported.

Runs under the **app** venv like the rest of ``renderer/tests`` (no torch): the
shim only imports ``glassrenderer.__main__``, which is torch-free.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))

import glassrenderer_entry  # noqa: E402
from glassrenderer import __main__ as cli  # noqa: E402


class _Recorder:
    """Stand-in for ``main`` / ``_exit`` that remembers what it was handed."""

    def __init__(self, code: int = 0) -> None:
        self.code = code
        self.argv: Optional[Sequence[str]] = None
        self.exit_code: Optional[int] = None

    def main(self, argv: Sequence[str]) -> int:
        self.argv = list(argv)
        return self.code

    def exit(self, code: int) -> None:
        self.exit_code = code


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """``glassrenderer.__main__.main`` / ``_exit`` replaced, looked up at call time."""
    rec = _Recorder(code=3)
    monkeypatch.setattr(cli, "main", rec.main)
    monkeypatch.setattr(cli, "_exit", rec.exit)
    return rec


def test_run_forwards_its_argv_and_exits_with_the_return_code(recorder: _Recorder):
    glassrenderer_entry.run(["serve", "--fake", "--models-dir", "D:/store"])
    assert recorder.argv == ["serve", "--fake", "--models-dir", "D:/store"]
    assert recorder.exit_code == 3


def test_run_defaults_to_sys_argv_without_the_program_name(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(sys, "argv", ["glassrenderer.exe", "selftest", "--fake"])
    glassrenderer_entry.run()
    assert recorder.argv == ["selftest", "--fake"]
    assert recorder.exit_code == 3


def test_run_resolves_main_at_call_time_not_import_time(recorder: _Recorder):
    """A module-level ``from ... import main`` would freeze the wrong binding."""
    calls: List[Sequence[str]] = []

    def replacement(argv: Sequence[str]) -> int:
        calls.append(list(argv))
        return 0

    cli.main = replacement  # the fixture's monkeypatch restores the original
    glassrenderer_entry.run(["selftest"])
    assert calls == [["selftest"]]
    assert recorder.argv is None


def test_the_offline_environment_is_installed_at_import_time():
    import os

    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_XET",
        "DO_NOT_TRACK",
    ):
        assert os.environ.get(name) == "1", name
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"


def test_offline_defaults_never_override_the_parent(monkeypatch: pytest.MonkeyPatch):
    """The app sets these too; ``setdefault`` must not fight it."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    glassrenderer_entry.apply_offline_environment()
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "0"
