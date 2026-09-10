"""``MangaOcrEngine`` with fake onnxruntime sessions and a fake detector.

No models, no network, no GPU: the "sessions" are scripted so the greedy
decoder produces a known token sequence from a tiny ``vocab.txt`` written to
``tmp_path`` together with fake ``config.json`` / ``generation_config.json``.
The engine must only depend on the file names published by
``glasstranslate.ocr.models``.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytest

from glasstranslate.core.interfaces import OCREngine
from glasstranslate.core.types import Segment
from glasstranslate.ocr import mangaocr as MO
from glasstranslate.ocr import models as M

# id: 0 PAD, 1 UNK, 2 CLS, 3 SEP, 4 MASK, then real tokens
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "こ", "ん", "に", "ち", "は", "a", "b", "1", "…", "・", "##ば", "た"]
TOK = {t: i for i, t in enumerate(VOCAB)}
HIDDEN = 8  # fake encoder width, keeps the fake fast


def make_model_dir(root: Path, *, vocab: Sequence[str] = VOCAB, max_length: int = 40) -> Path:
    """Create every file ``models.MANGA_OCR_FILES`` expects (fake content)."""
    target = M.manga_ocr_dir(root)
    target.mkdir(parents=True)
    (target / M.VOCAB_FILE).write_text("\n".join(vocab) + "\n", encoding="utf-8")
    (target / M.GENERATION_CONFIG_FILE).write_text(
        json.dumps({"decoder_start_token_id": 2, "eos_token_id": 3, "pad_token_id": 0, "max_length": max_length}),
        encoding="utf-8",
    )
    (target / M.CONFIG_FILE).write_text(json.dumps({"model_type": "vision-encoder-decoder"}), encoding="utf-8")
    for spec in M.MANGA_OCR_FILES:
        path = target / spec.name
        if not path.exists():
            path.write_bytes(b"fake")
    return target


# ----------------------------------------------------------------- fakes
class _Meta:
    def __init__(self, name: str, type_: str, shape: List[object]) -> None:
        self.name = name
        self.type = type_
        self.shape = shape


class FakeEncoder:
    def __init__(self, providers: Sequence[str]) -> None:
        self.providers = list(providers)
        self.calls: List[np.ndarray] = []

    def get_inputs(self) -> List[_Meta]:
        return [_Meta("pixel_values", "tensor(float)", ["batch_size", 3, 224, 224])]

    def get_outputs(self) -> List[_Meta]:
        return [_Meta("last_hidden_state", "tensor(float)", ["batch_size", 197, HIDDEN])]

    def get_providers(self) -> List[str]:
        return self.providers

    def run(self, names: Optional[List[str]], feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        x = feed["pixel_values"]
        self.calls.append(x)
        assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
        return [np.zeros((1, 197, HIDDEN), np.float32)]


class FakeDecoder:
    """Emits ``script`` (token ids) one per step, then eos forever."""

    def __init__(self, providers: Sequence[str], script: Sequence[int]) -> None:
        self.providers = list(providers)
        self.script = list(script)
        self.calls = 0

    def get_inputs(self) -> List[_Meta]:
        return [
            _Meta("input_ids", "tensor(int64)", ["batch_size", "decoder_sequence_length"]),
            _Meta("encoder_hidden_states", "tensor(float)", ["batch_size", "encoder_sequence_length", HIDDEN]),
        ]

    def get_outputs(self) -> List[_Meta]:
        return [_Meta("logits", "tensor(float)", ["batch_size", "decoder_sequence_length", len(VOCAB)])]

    def get_providers(self) -> List[str]:
        return self.providers

    def run(self, names: Optional[List[str]], feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        self.calls += 1
        ids = feed["input_ids"]
        assert ids.dtype == np.int64 and ids.shape[0] == 1 and int(ids[0, 0]) == 2
        step = ids.shape[1] - 1
        logits = np.full((1, ids.shape[1], len(VOCAB)), -10.0, np.float32)
        nxt = self.script[step] if step < len(self.script) else 3
        logits[0, -1, nxt] = 10.0
        return [logits]


class SessionFactory:
    def __init__(self, script: Sequence[int]) -> None:
        self.script = list(script)
        self.encoder: Optional[FakeEncoder] = None
        self.decoder: Optional[FakeDecoder] = None
        self.providers: Dict[str, List[str]] = {}

    def __call__(self, path: str, providers: Sequence[str]) -> object:
        name = Path(path).name
        self.providers[name] = list(providers)
        if name == M.ENCODER_FILE:
            self.encoder = FakeEncoder(providers)
            return self.encoder
        if name == M.DECODER_FILE:
            self.decoder = FakeDecoder(providers, self.script)
            return self.decoder
        raise AssertionError(f"unexpected session path {path}")


class FakeDetector(OCREngine):
    name = "fakedet"
    device = "cpu"

    def __init__(self, segments: Sequence[Segment]) -> None:
        self.segments = list(segments)
        self.calls = 0

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        self.calls += 1
        return [Segment(s.text, s.quad.copy(), s.confidence) for s in self.segments]


def quad(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def page_with_text(boxes: Sequence[Tuple[int, int, int, int]], size: Tuple[int, int] = (300, 400)) -> np.ndarray:
    """White page with dense black strokes inside each box (so ink checks pass)."""
    img = np.full((size[0], size[1], 3), 255, np.uint8)
    for x, y, w, h in boxes:
        for k in range(x + 2, x + w - 2, 4):
            cv2.line(img, (k, y + 2), (k, y + h - 3), (0, 0, 0), 2)
    return img


@pytest.fixture()
def model_root(tmp_path: Path) -> Path:
    make_model_dir(tmp_path)
    return tmp_path


def build(
    root: Path,
    detector: OCREngine,
    script: Sequence[int],
    *,
    device: str = "cpu",
    dml: bool = False,
    monkeypatch: Optional[pytest.MonkeyPatch] = None,
) -> Tuple[MO.MangaOcrEngine, SessionFactory]:
    if monkeypatch is not None:
        monkeypatch.setattr(MO, "_dml_available", lambda: dml)
    factory = SessionFactory(script)
    engine = MO.MangaOcrEngine(root, detector, device=device, session_factory=factory)
    return engine, factory


# ------------------------------------------------------------ recognise
def test_returns_one_segment_per_detected_line(model_root: Path) -> None:
    boxes = [(10, 10, 120, 30), (10, 60, 120, 30), (150, 10, 40, 120)]
    det = FakeDetector([Segment("x", quad(*b), 0.9) for b in boxes])
    engine, factory = build(model_root, det, [TOK["こ"], TOK["ん"], TOK["に"]])

    segs = engine.recognize(page_with_text(boxes))

    assert len(segs) == 3
    assert [s.text for s in segs] == ["こんに"] * 3
    assert all(s.lang_hint == "ja" for s in segs)
    assert det.calls == 1
    assert factory.encoder is not None and len(factory.encoder.calls) == 3


def test_quads_are_in_source_image_coordinates(model_root: Path) -> None:
    boxes = [(30, 40, 100, 25), (200, 100, 60, 90)]
    det = FakeDetector([Segment("x", quad(*b), 0.8) for b in boxes])
    engine, _ = build(model_root, det, [TOK["は"]])

    segs = engine.recognize(page_with_text(boxes))

    for seg, b in zip(segs, boxes):
        assert seg.quad.dtype == np.float32 and seg.quad.shape == (4, 2)
        np.testing.assert_array_equal(seg.quad, quad(*b))


def test_detector_box_score_becomes_segment_confidence(model_root: Path) -> None:
    boxes = [(10, 10, 100, 30), (10, 60, 100, 30)]
    det = FakeDetector([Segment("x", quad(*boxes[0]), 0.91), Segment("x", quad(*boxes[1]), 0.66)])
    engine, _ = build(model_root, det, [TOK["は"]])

    segs = engine.recognize(page_with_text(boxes))

    assert [round(s.confidence, 2) for s in segs] == [0.91, 0.66]


def test_low_detector_confidence_is_dropped_by_min_confidence(model_root: Path) -> None:
    boxes = [(10, 10, 100, 30), (10, 60, 100, 30)]
    det = FakeDetector([Segment("x", quad(*boxes[0]), 0.95), Segment("x", quad(*boxes[1]), 0.3)])
    factory = SessionFactory([TOK["は"]])
    engine = MO.MangaOcrEngine(model_root, det, device="cpu", min_confidence=0.5, session_factory=factory)

    segs = engine.recognize(page_with_text(boxes))

    assert len(segs) == 1 and segs[0].confidence == pytest.approx(0.95)


def test_blank_crop_produces_no_segment(model_root: Path) -> None:
    det = FakeDetector([Segment("x", quad(10, 10, 100, 30), 0.9)])
    engine, factory = build(model_root, det, [TOK["は"]])

    segs = engine.recognize(np.full((100, 200, 3), 255, np.uint8))  # pure white page

    assert segs == []
    assert factory.encoder is not None and factory.encoder.calls == []  # model never ran
    assert factory.decoder is not None and factory.decoder.calls == 0


def test_degenerate_repeated_output_is_dropped(model_root: Path) -> None:
    boxes = [(10, 10, 100, 30)]
    det = FakeDetector([Segment("x", quad(*boxes[0]), 0.9)])
    engine, _ = build(model_root, det, [TOK["は"]] * 12)
    assert engine.recognize(page_with_text(boxes)) == []

    engine2, _ = build(model_root, det, [TOK["こ"], TOK["ん"]] * 9)
    assert engine2.recognize(page_with_text(boxes)) == []

    engine3, _ = build(model_root, det, [TOK["は"]] * 3)  # short repeat is legitimate ("ははは")
    assert [s.text for s in engine3.recognize(page_with_text(boxes))] == ["ははは"]


def test_empty_decode_produces_no_segment(model_root: Path) -> None:
    boxes = [(10, 10, 100, 30)]
    det = FakeDetector([Segment("x", quad(*boxes[0]), 0.9)])
    engine, _ = build(model_root, det, [])  # eos immediately
    assert engine.recognize(page_with_text(boxes)) == []


def test_crop_outside_the_image_is_skipped(model_root: Path) -> None:
    det = FakeDetector([Segment("x", quad(500, 500, 20, 20), 0.9), Segment("x", quad(10, 10, 100, 30), 0.9)])
    engine, _ = build(model_root, det, [TOK["は"]])
    segs = engine.recognize(page_with_text([(10, 10, 100, 30)]))
    assert len(segs) == 1


def test_recognize_stops_at_eos_and_strips_specials(model_root: Path) -> None:
    boxes = [(10, 10, 100, 30)]
    det = FakeDetector([Segment("x", quad(*boxes[0]), 0.9)])
    script = [TOK["こ"], TOK["[UNK]"], TOK["##ば"], TOK["た"], 3, TOK["は"], TOK["は"]]
    engine, factory = build(model_root, det, script)

    segs = engine.recognize(page_with_text(boxes))

    assert [s.text for s in segs] == ["こばた"]
    assert factory.decoder is not None and factory.decoder.calls == 5  # 4 tokens + eos


def test_max_length_bounds_the_decoder_loop(tmp_path: Path) -> None:
    make_model_dir(tmp_path, max_length=6)
    boxes = [(10, 10, 100, 30)]
    det = FakeDetector([Segment("x", quad(*boxes[0]), 0.9)])
    script = [TOK["こ"], TOK["ん"], TOK["に"], TOK["ち"], TOK["は"], TOK["た"], TOK["a"], TOK["b"]]
    engine, factory = build(tmp_path, det, script)

    segs = engine.recognize(page_with_text(boxes))

    assert factory.decoder is not None and factory.decoder.calls <= 6
    assert segs and len(segs[0].text) <= 6


def test_per_crop_failure_is_skipped_but_session_errors_propagate(model_root: Path) -> None:
    boxes = [(10, 10, 100, 30), (10, 60, 100, 30)]
    det = FakeDetector([Segment("x", quad(*b), 0.9) for b in boxes])
    engine, factory = build(model_root, det, [TOK["は"]])
    assert factory.encoder is not None
    original = factory.encoder.run
    calls = {"n": 0}

    def flaky(names: Optional[List[str]], feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("bad crop")
        return original(names, feed)

    factory.encoder.run = flaky  # type: ignore[method-assign]
    assert len(engine.recognize(page_with_text(boxes))) == 1

    class BrokenDetector(OCREngine):
        def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
            raise RuntimeError("detector down")

    engine2, _ = build(model_root, BrokenDetector(), [TOK["は"]])
    with pytest.raises(RuntimeError, match="detector down"):
        engine2.recognize(page_with_text(boxes))


def test_per_crop_failure_is_logged_as_warning_then_debug(
    model_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    boxes = [(10, 10, 100, 30)] * (MO.CROP_FAILURE_WARNINGS + 2)
    det = FakeDetector([Segment("x", quad(*b), 0.9) for b in boxes])
    engine, factory = build(model_root, det, [TOK["は"]])
    assert factory.encoder is not None

    def broken(names: Optional[List[str]], feed: Dict[str, np.ndarray]) -> List[np.ndarray]:
        raise ValueError("bad crop")

    factory.encoder.run = broken  # type: ignore[method-assign]
    with caplog.at_level(logging.DEBUG, logger="glasstranslate.ocr.mangaocr"):
        assert engine.recognize(page_with_text(boxes)) == []

    records = [r for r in caplog.records if "skipped a crop" in r.getMessage()]
    assert len(records) == len(boxes)
    assert [r.levelno for r in records[: MO.CROP_FAILURE_WARNINGS]] == [logging.WARNING] * MO.CROP_FAILURE_WARNINGS
    assert all(r.levelno == logging.DEBUG for r in records[MO.CROP_FAILURE_WARNINGS :])
    assert "ValueError: bad crop" in records[0].getMessage()


def test_recognize_handles_none_and_empty_images(model_root: Path) -> None:
    det = FakeDetector([Segment("x", quad(10, 10, 100, 30), 0.9)])
    engine, _ = build(model_root, det, [TOK["は"]])
    assert engine.recognize(None) == []  # type: ignore[arg-type]
    assert engine.recognize(np.zeros((0, 0, 3), np.uint8)) == []
    assert det.calls == 0


def test_empty_vocab_raises_value_error(tmp_path: Path) -> None:
    target = make_model_dir(tmp_path)
    # Blank lines only: the file is present (a 0-byte file would count as
    # missing and raise FileNotFoundError instead) but holds no tokens.
    (target / M.VOCAB_FILE).write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        MO.load_vocab(target)
    with pytest.raises(ValueError):
        MO.MangaOcrEngine(tmp_path, FakeDetector([]), device="cpu", session_factory=SessionFactory([]))


def test_warmup_runs_the_model_once(model_root: Path) -> None:
    det = FakeDetector([])
    engine, factory = build(model_root, det, [TOK["は"]])
    engine.warmup()
    assert factory.encoder is not None and len(factory.encoder.calls) == 1
    assert factory.decoder is not None and factory.decoder.calls >= 1
    assert det.calls == 0  # warmup does not need the detector


# ------------------------------------------------------------- devices
def test_cpu_provider_is_used_when_dml_is_unavailable(model_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, factory = build(model_root, FakeDetector([]), [], device="auto", dml=False, monkeypatch=monkeypatch)
    assert factory.providers[M.ENCODER_FILE] == ["CPUExecutionProvider"]
    assert factory.providers[M.DECODER_FILE] == ["CPUExecutionProvider"]
    assert engine.device == "cpu"


def test_device_attribute_reports_gpu_or_cpu(model_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, factory = build(model_root, FakeDetector([]), [], device="auto", dml=True, monkeypatch=monkeypatch)
    assert engine.device == "gpu"
    assert factory.providers[M.ENCODER_FILE] == ["DmlExecutionProvider", "CPUExecutionProvider"]
    # The 2-layer decoder is faster on the CPU than through DirectML (probe
    # report); it always stays on the CPU provider.
    assert factory.providers[M.DECODER_FILE] == ["CPUExecutionProvider"]

    engine_cpu, _ = build(model_root, FakeDetector([]), [], device="cpu", dml=True, monkeypatch=monkeypatch)
    assert engine_cpu.device == "cpu"
    assert engine_cpu.name == "mangaocr"


def test_gpu_device_without_dml_raises(model_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="DmlExecutionProvider"):
        build(model_root, FakeDetector([]), [], device="gpu", dml=False, monkeypatch=monkeypatch)


def test_auto_falls_back_to_cpu_when_dml_session_fails(model_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MO, "_dml_available", lambda: True)
    inner = SessionFactory([])
    seen: List[List[str]] = []

    def factory(path: str, providers: Sequence[str]) -> object:
        seen.append(list(providers))
        if "DmlExecutionProvider" in providers:
            raise RuntimeError("no adapter")
        return inner(path, providers)

    engine = MO.MangaOcrEngine(model_root, FakeDetector([]), device="auto", session_factory=factory)
    assert engine.device == "cpu"
    assert seen[0] == ["DmlExecutionProvider", "CPUExecutionProvider"]


def test_unknown_device_is_rejected(model_root: Path) -> None:
    with pytest.raises(ValueError):
        MO.MangaOcrEngine(model_root, FakeDetector([]), device="tpu", session_factory=SessionFactory([]))


# -------------------------------------------------------------- models
def test_missing_models_raise_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        MO.MangaOcrEngine(tmp_path, FakeDetector([]), device="cpu", session_factory=SessionFactory([]))
    msg = str(exc.value)
    assert M.ENCODER_FILE in msg and str(M.manga_ocr_dir(tmp_path)) in msg


def test_generation_params_fall_back_to_defaults(tmp_path: Path) -> None:
    target = make_model_dir(tmp_path)
    (target / M.GENERATION_CONFIG_FILE).write_text("{}", encoding="utf-8")
    params = MO.load_generation_params(target)
    assert (params.decoder_start_token_id, params.eos_token_id, params.pad_token_id) == (2, 3, 0)
    assert params.max_length == MO.DEFAULT_MAX_LENGTH


# --------------------------------------------------------- pure helpers
def _reference_post_process(text: str) -> str:
    """Verbatim manga_ocr.ocr.post_process (kha-white/manga-ocr, Apache-2.0)."""
    import jaconv

    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text


@pytest.mark.parametrize(
    "raw",
    ["こん にちは", "abc 123", "待って…", "え・・・", "ｼﾞｮｼﾞｮ", "はい。。", "Ｇｏ！ go!", "  ", "・"],
)
def test_post_process_matches_manga_ocr(raw: str) -> None:
    assert MO.post_process(raw) == _reference_post_process(raw)


def test_post_process_literal_expectations() -> None:
    assert MO.post_process("こん にちは") == "こんにちは"
    assert MO.post_process("abc") == "ａｂｃ"
    assert MO.post_process("1 2") == "１２"
    assert MO.post_process("え…") == "え．．．"
    assert MO.post_process("え・・・") == "え．．．"


def test_decode_tokens_strips_wordpiece_and_specials() -> None:
    ids = [2, TOK["こ"], TOK["##ば"], 1, TOK["た"], 3, 0, 0]
    assert MO.decode_tokens(ids, VOCAB, {0, 1, 2, 3, 4}) == "こばた"


def test_is_degenerate_repeat() -> None:
    assert MO.is_degenerate("あ" * 8)
    assert MO.is_degenerate("あい" * 8)
    assert not MO.is_degenerate("あ" * 7)
    assert not MO.is_degenerate("あいうえおかきくけこ")
    assert not MO.is_degenerate("")


def test_preprocess_crop_shape_and_range() -> None:
    crop = np.zeros((30, 120, 3), np.uint8)
    crop[:, :60] = 255
    x = MO.preprocess_crop(crop, np.float32)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
    assert x.min() == pytest.approx(-1.0) and x.max() == pytest.approx(1.0)
    np.testing.assert_array_equal(x[0, 0], x[0, 1])  # grayscale replicated to RGB
    assert MO.preprocess_crop(crop, np.float16).dtype == np.float16


def test_ink_fraction() -> None:
    white = np.full((20, 20, 3), 255, np.uint8)
    assert MO.ink_fraction(white) == 0.0
    half = white.copy()
    half[:10] = 0
    assert MO.ink_fraction(half) == pytest.approx(0.5)
    assert MO.is_blank(white) and MO.is_blank(np.zeros((20, 20, 3), np.uint8)) and not MO.is_blank(half)
