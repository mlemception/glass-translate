"""``LamaStage.load`` hands torch a file object, never a path string.

Found by the portable acceptance run (2026-09-11): ``torch.jit.load(str)`` opens the
file through a C++ ``fopen`` that treats the UTF-8 path bytes as the ANSI code page
on Windows, so a models folder such as ``…\\moved ünïcode\\models`` fails with
errno 2 although the file exists.  Python's ``open`` uses the wide-character API.
The test runs without torch: a stand-in module records what ``jit.load`` received.
"""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from glassrenderer import models as M
from glassrenderer.stages.lama import LamaStage


class _Module:
    """The minimum a loaded TorchScript module must offer to ``load``."""

    def eval(self) -> None:
        pass

    def parameters(self):
        return []


def _fake_torch(record: Dict[str, Any]) -> types.ModuleType:
    torch = types.ModuleType("torch")
    jit = types.ModuleType("torch.jit")
    c_api = types.SimpleNamespace(
        _jit_set_profiling_executor=lambda flag: None,
        _jit_set_profiling_mode=lambda flag: None,
    )

    def load(source: Any, map_location: Any = None) -> _Module:
        record["source"] = source
        record["is_file_object"] = isinstance(source, io.IOBase)
        record["readable"] = getattr(source, "readable", lambda: False)()
        record["map_location"] = map_location
        return _Module()

    jit.load = load
    torch.jit = jit
    torch._C = c_api
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    return torch


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    torch = _fake_torch(record)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.jit", torch.jit)
    return record


def test_load_passes_an_open_file_object_from_a_non_ascii_path(
    tmp_path: Path, fake_torch: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = tmp_path / "moved ünïcode" / "models"
    weights = models_dir / "quality" / "lama" / M.LAMA_WEIGHTS
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not a real torchscript file")
    # The digest gate is exercised elsewhere; here it must let the stand-in file through.
    monkeypatch.setattr(M, "verify_file", lambda root, spec: M.file_path(root, spec))

    stage = LamaStage(models_dir, "cpu")
    stage.load()

    assert stage.ready
    assert fake_torch["is_file_object"] and fake_torch["readable"]
    assert Path(fake_torch["source"].name) == weights
    assert fake_torch["map_location"] == "cpu"
    assert fake_torch["source"].closed  # the context manager released the handle


def test_load_is_idempotent(tmp_path: Path, fake_torch: Dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "quality" / "lama" / M.LAMA_WEIGHTS
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    monkeypatch.setattr(M, "verify_file", lambda root, spec: M.file_path(root, spec))
    stage = LamaStage(tmp_path, "cpu")
    stage.load()
    first = fake_torch["source"]
    stage.load()
    assert fake_torch["source"] is first  # no second open once the module is resident
