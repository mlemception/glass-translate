"""Pipeline manga-mode logic: block emission, dirty-rect growth, unknown-token
cleanup and config persistence.  Uses fake engines only (no GPU, models,
network or display)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import pytest

from glasstranslate.config.settings import AppConfig
from glasstranslate.core.interfaces import LanguageDetector, OCREngine, ScreenCapture, Translator
from glasstranslate.core.pipeline import Pipeline, _footprint, _offset_translated, clean_translation
from glasstranslate.core.types import (
    Frame,
    PipelineStats,
    Rect,
    Segment,
    SegmentStyle,
    StyledSegment,
    TranslatedSegment,
)

W, H = 400, 300


# ----------------------------------------------------------------- fakes
class FakeCapture(ScreenCapture):
    name = "fake"

    def __init__(self, image: np.ndarray) -> None:
        self.image = image

    def grab(self, region: Rect) -> Optional[Frame]:
        return Frame(self.image.copy(), region.x, region.y, 0.0)


class FakeOCR(OCREngine):
    """Returns the configured segments that fall inside the crop, in crop
    coordinates, and records every crop it was asked to read."""

    name = "fake"
    device = "cpu"

    def __init__(self, segments: Sequence[Segment]) -> None:
        self.segments = list(segments)
        self.crops: List[tuple[int, int]] = []

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        self.crops.append(image_bgr.shape[:2])
        return [Segment(s.text, s.quad.copy(), s.confidence) for s in self.segments]

    def warmup(self) -> None:
        pass


class RecordingTranslator(Translator):
    name = "recording"
    device = "cpu"

    def __init__(self, outputs: Optional[dict] = None) -> None:
        self.calls: List[List[str]] = []
        self.outputs = outputs or {}

    def supported_pairs(self):
        return [("ja", "en")]

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        self.calls.append(list(texts))
        return [self.outputs.get(t, f"EN:{t}") for t in texts]


class FixedDetector(LanguageDetector):
    def detect(self, text: str) -> Optional[str]:
        return "ja"


def bubble_page() -> tuple[np.ndarray, List[Segment]]:
    """White page with a grey backdrop, one white oval bubble holding two
    vertical text columns, and one free-standing column on the backdrop."""
    img = np.full((H, W, 3), 150, np.uint8)
    cv2.ellipse(img, (120, 120), (90, 100), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (120, 120), (90, 100), 0, 0, 360, (0, 0, 0), 2)

    def column(x: int, y: int, w: int, h: int) -> np.ndarray:
        for cy in range(y + 4, y + h - 4, 12):
            cv2.rectangle(img, (x + 2, cy), (x + w - 2, cy + 7), (0, 0, 0), -1)
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

    segs = [
        Segment("右の列", column(130, 60, 20, 110), 0.9),
        Segment("左の列", column(100, 60, 20, 110), 0.9),
        Segment("外の文", column(320, 200, 20, 80), 0.9),
    ]
    return img, segs


def make_pipeline(
    img: np.ndarray, segs: Sequence[Segment], cfg: Optional[AppConfig] = None
) -> tuple[Pipeline, FakeOCR, RecordingTranslator, list]:
    cfg = cfg or AppConfig(translation_backend="identity", source_lang="ja", target_lang="en")
    results: list = []
    ocr = FakeOCR(segs)
    translator = RecordingTranslator()
    pipe = Pipeline(
        cfg,
        region_provider=lambda: Rect(0, 0, W, H),
        on_result=lambda segments, stats: results.append((segments, stats)),
        on_status=lambda msg: None,
        capture_factory=lambda c: FakeCapture(img),
        ocr_factory=lambda c: ocr,
        translator_factory=lambda c: translator,
        detector_factory=FixedDetector,
    )
    return pipe, ocr, translator, results


# ----------------------------------------------------------------- tests
def test_clean_translation_strips_unknown_tokens() -> None:
    assert clean_translation("This man's ⁇ sword") == "This man's sword"
    assert clean_translation("a <unk> b  <unk>") == "a b"
    assert clean_translation("⁇") == ""
    assert clean_translation(" ⁇ ... ") == ""
    assert clean_translation("  keep   spaces  ") == "keep spaces"


def test_config_roundtrip_keeps_manga_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    AppConfig(manga_mode=False, uppercase=False).save(path)
    loaded = AppConfig.load(path)
    assert loaded.manga_mode is False and loaded.uppercase is False
    assert AppConfig().manga_mode is True and AppConfig().uppercase is True
    # Wrong types are ignored, not coerced.
    assert AppConfig.from_dict({"manga_mode": "yes"}).manga_mode is True


def _translated(quad: np.ndarray, style: SegmentStyle) -> TranslatedSegment:
    seg = Segment("x", quad, 0.9)
    return TranslatedSegment(StyledSegment(seg, style, "ja"), "y", "en")


def test_footprint_includes_clean_rect_and_offset_moves_style() -> None:
    quad = np.array([[10, 10], [30, 10], [30, 50], [10, 50]], dtype=np.float32)
    plain = SegmentStyle((0, 0, 0), (255, 255, 255), 0.0, 20.0)
    assert _footprint(_translated(quad, plain)) == Rect(10, 10, 20, 40)
    block = SegmentStyle(
        (0, 0, 0), (255, 255, 255), 0.0, 20.0, layout_box=Rect(0, 0, 80, 80), clean_rect=Rect(5, 5, 40, 60)
    )
    assert _footprint(_translated(quad, block)) == Rect(0, 0, 80, 80)
    moved = _offset_translated(_translated(quad, block), 7, -3)
    assert moved.style.layout_box == Rect(7, -3, 80, 80)
    assert moved.style.clean_rect == Rect(12, 2, 40, 60)
    assert moved.quad[0].tolist() == [17.0, 7.0]


def test_manga_mode_emits_blocks_and_translates_whole_utterance() -> None:
    img, segs = bubble_page()
    pipe, ocr, translator, results = make_pipeline(img, segs)
    stats = pipe.step()
    assert isinstance(stats, PipelineStats)
    segments, _ = results[-1]
    assert stats.extra["blocks"] == 2 and len(segments) == 2
    assert all(s.style.is_block for s in segments)
    by_text = {s.source_text: s for s in segments}
    # Right column is read first in vertical Japanese, columns join without a space.
    assert "右の列左の列" in by_text and "外の文" in by_text
    assert translator.calls == [["右の列左の列", "外の文"]] or translator.calls == [["外の文", "右の列左の列"]]
    assert by_text["右の列左の列"].translation == "EN:右の列左の列"
    bubble = by_text["右の列左の列"].style
    assert bubble.layout_mask is not None and bubble.clean_patch is not None
    assert bubble.layout_box is not None and bubble.layout_box.w > 100  # the bubble, not the columns
    # Whole-frame crop on the first pass.
    assert ocr.crops == [(H, W)]


def test_plain_mode_emits_one_segment_per_line() -> None:
    img, segs = bubble_page()
    cfg = AppConfig(translation_backend="identity", source_lang="ja", target_lang="en", manga_mode=False)
    pipe, _, translator, results = make_pipeline(img, segs, cfg)
    stats = pipe.step()
    assert stats is not None and stats.extra["blocks"] == 0
    segments, _ = results[-1]
    assert len(segments) == 3 and not any(s.style.is_block for s in segments)
    assert sorted(translator.calls[0]) == sorted(s.text for s in segs)


def test_dirty_rect_touching_block_regrows_crop_to_bubble() -> None:
    img, segs = bubble_page()
    pipe, ocr, translator, results = make_pipeline(img, segs)
    pipe.step()
    first = {s.source_text: s for s in results[-1][0]}
    bubble_box = first["右の列左の列"].style.layout_box
    assert bubble_box is not None

    # Scribble on the bubble text: the block must be dropped and re-read from
    # a crop that covers the whole bubble plus a margin.
    dirty = img.copy()
    dirty[60:80, 130:150] = 0
    pipe._capture = FakeCapture(dirty)
    ocr.crops.clear()
    pipe.step()
    segments, stats = results[-1]
    assert not stats.skipped_unchanged and stats.dirty_regions >= 1
    assert len(ocr.crops) == 1
    crop_h, crop_w = ocr.crops[0]
    assert crop_w >= min(W, bubble_box.w + 2 * 64) - 1 and crop_h >= min(H, bubble_box.h + 2 * 64) - 1
    # The untouched free column survived; the bubble block was replaced (same text, fresh object).
    texts = sorted(s.source_text for s in segments)
    assert texts == ["右の列左の列", "外の文"]
    assert len(segments) == 2
    # The translation of the re-read block came from the cache.
    assert translator.calls == [translator.calls[0]] or len(translator.calls) == 1


def test_unusable_translation_falls_back_to_source_and_is_not_cached() -> None:
    img, segs = bubble_page()
    pipe, _, translator, results = make_pipeline(img, segs)
    translator.outputs = {"外の文": "⁇", "右の列左の列": "the ⁇ words"}
    pipe.step()
    by_text = {s.source_text: s.translation for s in results[-1][0]}
    assert by_text["外の文"] == "外の文"
    assert by_text["右の列左の列"] == "the words"
    assert pipe.cache.get("外の文", "ja", "en") is None
    assert pipe.cache.get("右の列左の列", "ja", "en") == "the words"


def test_unchanged_frame_is_skipped_and_keeps_blocks() -> None:
    img, segs = bubble_page()
    pipe, ocr, _, results = make_pipeline(img, segs)
    pipe.step()
    stats = pipe.step()
    assert stats is not None and stats.skipped_unchanged
    assert len(results[-1][0]) == 2 and len(ocr.crops) == 1


@pytest.mark.parametrize("manga", [True, False])
def test_switching_manga_mode_invalidates(manga: bool) -> None:
    img, segs = bubble_page()
    cfg = AppConfig(translation_backend="identity", source_lang="ja", target_lang="en", manga_mode=manga)
    pipe, ocr, _, results = make_pipeline(img, segs, cfg)
    pipe.step()
    cfg.manga_mode = not manga
    pipe.set_config(cfg)
    stats = pipe.step()
    assert stats is not None and not stats.skipped_unchanged
    assert len(ocr.crops) == 2
    assert len(results[-1][0]) == (3 if manga else 2)
