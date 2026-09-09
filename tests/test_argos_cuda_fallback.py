"""Regression: a CUDA device that is detected but unusable at translation time
(the frozen exe ships no cuBLAS, so ``ctranslate2`` raises ``Library
cublas64_12.dll is not found or cannot be loaded`` on the first batch) must
fall back to CPU instead of silently returning the input unchanged forever.

Models are never really loaded: ``_load_models`` is patched to hand back fakes
whose behaviour depends on the device the translator is currently on.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from glasstranslate.translate.argos import ArgosCT2Translator, _LoadedPair

_CUBLAS_ERROR = "Library cublas64_12.dll is not found or cannot be loaded"


def _sugoi_layout(directory: Path) -> None:
    (directory / "spm").mkdir(parents=True, exist_ok=True)
    (directory / "config.json").touch()
    (directory / "model.bin").touch()
    (directory / "spm" / "spm.ja.nopretok.model").touch()
    (directory / "spm" / "spm.en.nopretok.model").touch()


class _FakeTokenizer:
    def encode(self, text: str, out_type: type = str) -> List[str]:
        return [text]

    def decode(self, tokens: List[str]) -> str:
        return " ".join(tokens)


class _FakeResult:
    def __init__(self, tokens: List[str]) -> None:
        self.hypotheses = [tokens]


class _FakeCT2Translator:
    """Raises the frozen-exe cuBLAS error on cuda; translates on cpu."""

    def __init__(self, device: str) -> None:
        self.device = device

    def translate_batch(self, batch: List[List[str]], **kwargs: object) -> List[_FakeResult]:
        if self.device == "cuda":
            raise RuntimeError(_CUBLAS_ERROR)
        return [_FakeResult(["TRANSLATED"]) for _ in batch]

    def unload_model(self) -> None:
        pass


@pytest.fixture()
def translator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ArgosCT2Translator:
    _sugoi_layout(tmp_path)
    loads: List[str] = []

    def fake_load_models(self: ArgosCT2Translator, pkg: object) -> _LoadedPair:
        loads.append(self.device)
        tok = _FakeTokenizer()
        return _LoadedPair(_FakeCT2Translator(self.device), tok, tok)

    monkeypatch.setattr(ArgosCT2Translator, "_load_models", fake_load_models)
    tr = ArgosCT2Translator(tmp_path, device="cuda")
    tr.load_devices = loads  # type: ignore[attr-defined] - test bookkeeping
    return tr


def test_cuda_translation_failure_falls_back_to_cpu(translator: ArgosCT2Translator) -> None:
    # Act
    out = translator.translate_batch(["こんにちは"], "ja", "en")

    # Assert: the text is translated (not returned unchanged) on the CPU.
    assert out == ["TRANSLATED"]
    assert translator.device == "cpu"


def test_fallback_sticks_for_later_batches(translator: ArgosCT2Translator) -> None:
    translator.translate_batch(["こんにちは"], "ja", "en")

    out = translator.translate_batch(["さようなら"], "ja", "en")

    assert out == ["TRANSLATED"]
    # cuda attempt once, cpu reload once, then the cached cpu model is reused.
    assert translator.load_devices == ["cuda", "cpu"]  # type: ignore[attr-defined]


def test_cpu_translation_failure_still_returns_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On CPU there is nothing to fall back to: the public API keeps its
    return-input-unchanged contract instead of retrying forever."""
    _sugoi_layout(tmp_path)

    class _AlwaysFailing(_FakeCT2Translator):
        def translate_batch(self, batch: List[List[str]], **kwargs: object) -> List[_FakeResult]:
            raise RuntimeError(_CUBLAS_ERROR)

    def fake_load_models(self: ArgosCT2Translator, pkg: object) -> _LoadedPair:
        tok = _FakeTokenizer()
        return _LoadedPair(_AlwaysFailing(self.device), tok, tok)

    monkeypatch.setattr(ArgosCT2Translator, "_load_models", fake_load_models)
    tr = ArgosCT2Translator(tmp_path, device="cpu")

    out = tr.translate_batch(["こんにちは"], "ja", "en")

    assert out == ["こんにちは"]
    assert tr.device == "cpu"


def test_transient_gpu_error_does_not_downgrade_to_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only GPU-stack-unusable errors (missing cuBLAS/cuDNN, no driver) may
    permanently downgrade the session; a one-off batch error must not."""
    _sugoi_layout(tmp_path)

    class _TransientError(_FakeCT2Translator):
        def translate_batch(self, batch: List[List[str]], **kwargs: object) -> List[_FakeResult]:
            raise RuntimeError("CUDA failed with error out of memory")

    def fake_load_models(self: ArgosCT2Translator, pkg: object) -> _LoadedPair:
        tok = _FakeTokenizer()
        return _LoadedPair(_TransientError(self.device), tok, tok)

    monkeypatch.setattr(ArgosCT2Translator, "_load_models", fake_load_models)
    tr = ArgosCT2Translator(tmp_path, device="cuda")

    out = tr.translate_batch(["こんにちは"], "ja", "en")

    assert out == ["こんにちは"]  # degraded for this batch only
    assert tr.device == "cuda"  # next batch tries the GPU again


def test_fallback_survives_unload_model_raising(translator: ArgosCT2Translator, monkeypatch: pytest.MonkeyPatch) -> None:
    """``unload_model`` on the just-failed CUDA translator may itself throw;
    the CPU fallback must still complete."""
    monkeypatch.setattr(
        _FakeCT2Translator, "unload_model", lambda self: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    out = translator.translate_batch(["こんにちは"], "ja", "en")

    assert out == ["TRANSLATED"]
    assert translator.device == "cpu"
