"""Pipeline wiring for the OCR registry: the default factory passes
``models_dir``/``status`` to ``create_ocr``, ``_ensure_engines`` hands the
status callback to factories that take one, a ``models_dir`` change rebuilds
the OCR engine, and plain mode strips furigana only on the PaddleOCR path.
Fake engines only."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
import pytest

from glasstranslate.config.settings import AppConfig
from glasstranslate.core import pipeline as P
from glasstranslate.core.interfaces import LanguageDetector, OCREngine, ScreenCapture, Translator
from glasstranslate.core.types import Frame, Rect, Segment

W, H = 400, 300


def rect_quad(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


class FakeCapture(ScreenCapture):
    name = "fake"

    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def grab(self, region: Rect) -> Optional[Frame]:
        return Frame(self.image.copy(), region.x, region.y, 0.0)


class NamedOCR(OCREngine):
    device = "cpu"

    def __init__(self, name: str, segments: Sequence[Segment]) -> None:
        self.name = name
        self.segments = list(segments)

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        return [Segment(s.text, s.quad.copy(), s.confidence) for s in self.segments]

    def warmup(self) -> None:
        pass


class EchoTranslator(Translator):
    name = "echo"
    device = "cpu"

    def supported_pairs(self):
        return [("ja", "en")]

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        return [f"EN:{t}" for t in texts]


class JaDetector(LanguageDetector):
    def detect(self, text: str) -> Optional[str]:
        return "ja"


def page_with_ruby() -> tuple[np.ndarray, List[Segment]]:
    """A kanji column with a furigana column on its right, plus a free column."""
    img = np.full((H, W, 3), 255, np.uint8)
    segs = [
        Segment("漢字の縦書き", rect_quad(100, 50, 30, 200), 0.9),
        Segment("かんじ", rect_quad(132, 60, 12, 80), 0.9),
        Segment("外の文", rect_quad(320, 100, 20, 80), 0.9),
    ]
    return img, segs


def plain_pipeline(ocr: OCREngine, img: np.ndarray, **cfg_kw: Any) -> tuple[P.Pipeline, list, List[str]]:
    cfg = AppConfig(translation_backend="identity", source_lang="ja", target_lang="en", manga_mode=False, **cfg_kw)
    results: list = []
    statuses: List[str] = []
    pipe = P.Pipeline(
        cfg,
        region_provider=lambda: Rect(0, 0, W, H),
        on_result=lambda segments, stats: results.append((segments, stats)),
        on_status=statuses.append,
        capture_factory=lambda c: FakeCapture(img),
        ocr_factory=lambda c: ocr,
        translator_factory=lambda c: EchoTranslator(),
        detector_factory=JaDetector,
    )
    return pipe, results, statuses


# ------------------------------------------------------------ default factory
def test_default_ocr_factory_passes_models_dir_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[tuple] = []

    def fake_create_ocr(name: str, device: str, min_confidence: float, **kw: Any) -> OCREngine:
        calls.append((name, device, min_confidence, kw))
        return NamedOCR(name, [])

    import glasstranslate.ocr as ocr_pkg

    monkeypatch.setattr(ocr_pkg, "create_ocr", fake_create_ocr)
    cfg = AppConfig(ocr_engine="paddleocr", ocr_device="cpu", min_confidence=0.3, models_dir="D:/m")
    status_cb = lambda msg: None  # noqa: E731
    P._default_ocr_factory(cfg, status_cb)
    assert calls == [("paddleocr", "cpu", 0.3, {"models_dir": "D:/m", "status": status_cb})]
    P._default_ocr_factory(cfg)
    assert calls[1][3] == {"models_dir": "D:/m", "status": None}


def test_ensure_engines_hands_status_to_a_factory_that_takes_it() -> None:
    seen: List[Any] = []

    def factory(cfg: AppConfig, status: Any = None) -> OCREngine:
        seen.append(status)
        return NamedOCR("paddleocr", [])

    statuses: List[str] = []
    pipe = P.Pipeline(
        AppConfig(translation_backend="identity"),
        region_provider=lambda: Rect(0, 0, W, H),
        on_result=lambda s, st: None,
        on_status=statuses.append,
        capture_factory=lambda c: FakeCapture(np.zeros((H, W, 3), np.uint8)),
        ocr_factory=factory,
        translator_factory=lambda c: EchoTranslator(),
        detector_factory=JaDetector,
    )
    assert pipe._ensure_engines()
    assert len(seen) == 1 and callable(seen[0])
    seen[0]("OCR: mangaocr failed (FileNotFoundError); using paddleocr")
    assert statuses[-1] == "OCR: mangaocr failed (FileNotFoundError); using paddleocr"


def test_ensure_engines_still_accepts_a_cfg_only_factory() -> None:
    img = np.zeros((H, W, 3), np.uint8)
    pipe, _, statuses = plain_pipeline(NamedOCR("fake", []), img)
    assert pipe._ensure_engines()
    assert any(s.startswith("OCR: fake on cpu") for s in statuses)


def test_accepts_status_signature_probe() -> None:
    assert P._accepts_status(lambda cfg, status=None: None)
    assert P._accepts_status(lambda cfg, status: None)
    assert P._accepts_status(lambda cfg, *, status=None: None)
    assert P._accepts_status(P._default_ocr_factory)
    assert not P._accepts_status(lambda cfg: None)
    assert not P._accepts_status(lambda cfg, callback=None: None)
    assert not P._accepts_status(lambda: None)


class FlippingOCR(NamedOCR):
    """Reports a different engine name on every call, like a fallback chain
    that switches engines between the crops of one tick."""

    def __init__(self, names: Sequence[str], segments: Sequence[Segment]) -> None:
        super().__init__(names[0], segments)
        self._names = list(names)

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        self.name = self._names.pop(0)
        return super().recognize(image_bgr)


def test_furigana_filter_follows_the_engine_of_each_crop() -> None:
    img, segs = page_with_ruby()
    ocr = FlippingOCR(["paddleocr", "mangaocr"], segs)
    pipe, _, _ = plain_pipeline(ocr, img)
    assert pipe._ensure_engines()
    first = pipe._recognize_crop(img)
    second = pipe._recognize_crop(img)
    assert [s.text for s in first] == ["漢字の縦書き", "外の文"]  # PaddleOCR crop: ruby stripped
    assert len(second) == 3  # manga-ocr crop: every line kept


# ------------------------------------------------------------ rebuild rule
def test_changing_models_dir_rebuilds_the_ocr_engine(tmp_path: Any) -> None:
    assert "models_dir" in P._OCR_FIELDS
    img = np.zeros((H, W, 3), np.uint8)
    pipe, _, _ = plain_pipeline(NamedOCR("fake", []), img)
    assert pipe._ensure_engines() and not pipe._rebuild_ocr
    cfg = AppConfig.from_dict(pipe._cfg.to_dict())
    cfg.models_dir = str(tmp_path / "elsewhere")
    pipe.set_config(cfg)
    pipe._apply_pending_config()
    assert pipe._rebuild_ocr


def test_refresh_models_rebuilds_the_ocr_chain_as_well_as_the_translator() -> None:
    """After a model download (``models_changed`` -> ``refresh_models``) the OCR chain must be rebuilt
    right away so manga-ocr takes over without waiting for the fallback chain's next timed re-probe."""
    img = np.zeros((H, W, 3), np.uint8)
    pipe, _, _ = plain_pipeline(NamedOCR("fake", []), img)
    assert pipe._ensure_engines() and not pipe._rebuild_ocr and not pipe._rebuild_translator
    pipe.refresh_models()
    assert pipe._rebuild_ocr and pipe._rebuild_translator
    assert pipe._ensure_engines() and not pipe._rebuild_ocr and not pipe._rebuild_translator


# ------------------------------------------------------------ furigana in plain mode
def test_plain_mode_strips_furigana_on_the_paddleocr_path() -> None:
    img, segs = page_with_ruby()
    pipe, results, _ = plain_pipeline(NamedOCR("paddleocr", segs), img)
    pipe.step()
    texts = sorted(s.source_text for s in results[-1][0])
    assert texts == ["外の文", "漢字の縦書き"]


def test_plain_mode_keeps_every_line_for_other_engines() -> None:
    img, segs = page_with_ruby()
    pipe, results, _ = plain_pipeline(NamedOCR("mangaocr", segs), img)
    pipe.step()
    assert len(results[-1][0]) == 3


def test_manga_mode_grouping_is_unchanged_by_the_plain_filter() -> None:
    """Manga mode keeps its own furigana rule in render/layout.py: the block
    still absorbs the ruby line instead of translating it separately."""
    img, segs = page_with_ruby()
    cfg = AppConfig(translation_backend="identity", source_lang="ja", target_lang="en", manga_mode=True)
    results: list = []
    pipe = P.Pipeline(
        cfg,
        region_provider=lambda: Rect(0, 0, W, H),
        on_result=lambda segments, stats: results.append((segments, stats)),
        on_status=lambda m: None,
        capture_factory=lambda c: FakeCapture(img),
        ocr_factory=lambda c: NamedOCR("paddleocr", segs),
        translator_factory=lambda c: EchoTranslator(),
        detector_factory=JaDetector,
    )
    pipe.step()
    texts = [s.source_text for s in results[-1][0]]
    assert "かんじ" not in texts and all(s.style.is_block for s in results[-1][0])
