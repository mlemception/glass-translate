"""OCR engine registry (``glasstranslate.ocr.factory``): names, alias,
labels, and the manga-ocr fallback chain wiring.  ``RapidOCREngine`` and the
``mangaocr`` module are replaced by fakes so no model is ever loaded."""
from __future__ import annotations

import sys
import types
from typing import Any, List

import numpy as np
import pytest

from glasstranslate.config.settings import AppConfig, default_models_dir
from glasstranslate.core.interfaces import OCREngine
from glasstranslate.core.types import Segment
from glasstranslate.ocr import factory as F
from glasstranslate.ocr.chain import FallbackOCREngine


class FakeRapid(OCREngine):
    name = "paddleocr"

    def __init__(self, device: str = "auto", min_confidence: float = 0.5) -> None:
        self.device = device
        self.min_confidence = min_confidence

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        return []

    def warmup(self) -> None:
        pass


class FakeMangaOcr(OCREngine):
    name = "mangaocr"
    instances: List["FakeMangaOcr"] = []

    def __init__(self, models_dir: Any, detector: OCREngine, *, device: str, min_confidence: float) -> None:
        self.models_dir = models_dir
        self.detector = detector
        self.device = device
        self.min_confidence = min_confidence
        FakeMangaOcr.instances.append(self)

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        return []

    def warmup(self) -> None:
        pass


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(F, "RapidOCREngine", FakeRapid)
    module = types.ModuleType("glasstranslate.ocr.mangaocr")
    module.MangaOcrEngine = FakeMangaOcr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "glasstranslate.ocr.mangaocr", module)
    FakeMangaOcr.instances = []


# --------------------------------------------------------------- registry
def test_available_engines_lists_registered_names() -> None:
    assert F.available_engines() == ["mangaocr", "paddleocr"]


def test_registry_is_not_mutated_by_callers() -> None:
    names = F.available_engines()
    names.append("bogus")
    names.remove("mangaocr")
    assert F.available_engines() == ["mangaocr", "paddleocr"]
    assert isinstance(F._ENGINES, tuple) and F._ENGINES == ("mangaocr", "paddleocr")


@pytest.mark.parametrize(
    "raw, expected",
    [("mangaocr", "mangaocr"), ("paddleocr", "paddleocr"), ("rapidocr", "paddleocr"), (" PaddleOCR ", "paddleocr")],
)
def test_normalize_engine_name(raw: str, expected: str) -> None:
    assert F.normalize_engine_name(raw) == expected


def test_normalize_unknown_engine_raises_with_the_available_list() -> None:
    with pytest.raises(ValueError, match="mangaocr, paddleocr"):
        F.normalize_engine_name("tesseract")


def test_create_unknown_engine_raises_with_the_available_list() -> None:
    with pytest.raises(ValueError, match="'tesseract'.*mangaocr, paddleocr"):
        F.create_ocr("tesseract", "cpu", 0.5)


# --------------------------------------------------------------- labels
def test_engine_label_values() -> None:
    assert F.ENGINE_LABELS == {"mangaocr": "manga-ocr (ONNX)", "paddleocr": "PaddleOCR (PP-OCRv5 via ONNX)"}
    assert F.engine_label("mangaocr") == "manga-ocr (ONNX)"
    assert F.engine_label("paddleocr") == "PaddleOCR (PP-OCRv5 via ONNX)"
    assert F.engine_label("rapidocr") == "PaddleOCR (PP-OCRv5 via ONNX)"
    assert F.engine_label("something-else") == "something-else"  # unknown values stay listable in the UI


# --------------------------------------------------------------- paddleocr
def test_rapidocr_alias_creates_the_same_class_as_paddleocr(fakes: None) -> None:
    a = F.create_ocr("paddleocr", "cpu", 0.4)
    b = F.create_ocr("rapidocr", "cpu", 0.4)
    assert type(a) is type(b) is FakeRapid
    assert (a.device, a.min_confidence) == ("cpu", 0.4)
    assert not isinstance(a, FallbackOCREngine)


# --------------------------------------------------------------- mangaocr
def test_mangaocr_returns_a_fallback_chain_over_rapidocr(fakes: None) -> None:
    statuses: List[str] = []
    eng = F.create_ocr("mangaocr", "cpu", 0.5, models_dir="D:/models", status=statuses.append)
    assert isinstance(eng, FallbackOCREngine)
    assert isinstance(eng.fallback, FakeRapid) and eng.primary_name == "mangaocr"
    assert FakeMangaOcr.instances == []  # primary is built lazily
    eng.warmup()
    assert len(FakeMangaOcr.instances) == 1
    built = FakeMangaOcr.instances[0]
    assert built.models_dir == "D:/models"
    assert built.detector is eng.fallback  # the rapidocr detector is shared, not built twice
    assert (built.device, built.min_confidence) == ("cpu", 0.5)
    assert eng.name == "mangaocr" and statuses == []


def test_mangaocr_models_dir_defaults_to_default_models_dir(fakes: None) -> None:
    eng = F.create_ocr("mangaocr", "cpu", 0.5)
    eng.warmup()
    assert FakeMangaOcr.instances[0].models_dir == str(default_models_dir())


def test_mangaocr_import_failure_falls_back_to_paddleocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(F, "RapidOCREngine", FakeRapid)
    monkeypatch.setitem(sys.modules, "glasstranslate.ocr.mangaocr", None)  # import raises
    statuses: List[str] = []
    eng = F.create_ocr("mangaocr", "cpu", 0.5, status=statuses.append)
    eng.warmup()
    assert eng.name == "paddleocr"
    assert len(statuses) == 1 and statuses[0].startswith("OCR: mangaocr failed (") and "using paddleocr" in statuses[0]


def test_mangaocr_missing_models_fall_back_to_paddleocr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(F, "RapidOCREngine", FakeRapid)

    class Missing(FakeMangaOcr):
        def __init__(self, *a: Any, **kw: Any) -> None:
            raise FileNotFoundError("encoder.onnx")

    module = types.ModuleType("glasstranslate.ocr.mangaocr")
    module.MangaOcrEngine = Missing  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "glasstranslate.ocr.mangaocr", module)
    statuses: List[str] = []
    eng = F.create_ocr("mangaocr", "cpu", 0.5, status=statuses.append)
    eng.warmup()
    assert eng.name == "paddleocr" and statuses == ["OCR: mangaocr failed (FileNotFoundError); using paddleocr"]


# --------------------------------------------------------------- config default
def test_default_config_engine_is_mangaocr() -> None:
    assert AppConfig().ocr_engine == "mangaocr"
    assert AppConfig().ocr_engine in F.available_engines()


def test_package_exports() -> None:
    import glasstranslate.ocr as pkg

    for name in ("FallbackOCREngine", "ENGINE_LABELS", "engine_label", "normalize_engine_name",
                 "strip_furigana", "furigana_indices", "is_kana_text", "available_engines", "create_ocr"):
        assert name in pkg.__all__ and hasattr(pkg, name)
