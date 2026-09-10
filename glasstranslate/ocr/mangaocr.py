"""OCR engine that recognises detected text lines with manga-ocr (ONNX).

Layout:

    detector (any :class:`OCREngine`, in practice rapidocr) -> line quads
    manga-ocr ViT encoder (onnxruntime, DirectML when available) -> hidden states
    manga-ocr BERT decoder (onnxruntime, CPU) -> greedy token ids -> text

manga-ocr (kha-white, Apache-2.0) is a VisionEncoderDecoder trained on manga:
it reads vertical and horizontal Japanese, ignores furigana and copes with
text drawn over artwork, but emits one string per image with no boxes, so the
detector supplies the geometry and its box score becomes the segment
confidence.  The model files come from :mod:`glasstranslate.ocr.models`.

Provider choice (docs/perf/2026-09-09-mangaocr-probe.md): the 86 M-parameter
encoder is ~8x faster on DirectML than on the CPU, while the tiny 2-layer
decoder is ~10x *slower* through DirectML (dispatch overhead dominates), so
the decoder always runs on the CPU provider.  Sequences are short (typically
< 15 tokens) so the plain decoder is re-run on the whole prefix each step
instead of carrying a KV cache.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike

from glasstranslate.core.interfaces import OCREngine
from glasstranslate.core.types import Segment

from . import models as M

log = logging.getLogger(__name__)

_DML_PROVIDER = "DmlExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"

# generation_config.json values of onnx-community/manga-ocr-base-ONNX, used
# when the file is missing a key.
DEFAULT_DECODER_START = 2  # [CLS]
DEFAULT_EOS = 3  # [SEP]
DEFAULT_PAD = 0  # [PAD]
DEFAULT_MAX_LENGTH = 300
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

INPUT_SIZE = 224  # ViT input, preprocessor_config.json
CROP_PAD_PX = 2
# Hallucination guards.  A crop with almost no dark pixels is not text (the
# model still answers "それは、" on a white image); a decode that repeats the
# same character or 2-gram this many times is a runaway.
MIN_INK_FRACTION = 0.01
INK_THRESHOLD = 128
DEGENERATE_REPEATS = 8
# Stop decoding as soon as the tail repeats this many times: a runaway
# otherwise burns ~1.7 s per crop reaching max_length (probe report).
REPEAT_STOP = DEGENERATE_REPEATS
# Per-crop failures are logged as warnings this many times per engine, then
# demoted to debug so a persistently odd page cannot flood the log.
CROP_FAILURE_WARNINGS = 3

SessionFactory = Callable[[str, Sequence[str]], object]


def _dml_available() -> bool:
    """True if the installed onnxruntime exposes the DirectML provider."""
    import onnxruntime as ort

    return _DML_PROVIDER in ort.get_available_providers()


def _default_session_factory(path: str, providers: Sequence[str]) -> object:
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=list(providers))


# ------------------------------------------------------------ pure helpers
def post_process(text: str) -> str:
    """manga-ocr's ``post_process``: drop whitespace, normalise ellipses,
    convert half-width ASCII / digits to full-width (``jaconv.h2z``)."""
    import jaconv

    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    return jaconv.h2z(text, ascii=True, digit=True)


def decode_tokens(ids: Sequence[int], vocab: Sequence[str], special_ids: Set[int]) -> str:
    """WordPiece ids -> text (strip ``##`` continuations, skip specials)."""
    parts: List[str] = []
    for tid in ids:
        if tid in special_ids or tid < 0 or tid >= len(vocab):
            continue
        tok = vocab[tid]
        parts.append(tok[2:] if tok.startswith("##") else tok)
    return "".join(parts)


def is_degenerate(text: str, repeats: int = DEGENERATE_REPEATS) -> bool:
    """True when ``text`` is one character or one 2-gram repeated ``repeats``
    or more times and nothing else."""
    for unit in (1, 2):
        if len(text) >= unit * repeats and len(text) % unit == 0:
            head = text[:unit]
            if text == head * (len(text) // unit):
                return True
    return False


def _tail_repeats(ids: Sequence[int], repeats: int = REPEAT_STOP) -> bool:
    """True when the last ``repeats`` tokens are one token or one 2-gram."""
    for unit in (1, 2):
        n = unit * repeats
        if len(ids) < n:
            continue
        tail = list(ids[-n:])
        if tail == tail[:unit] * repeats:
            return True
    return False


def ink_fraction(crop_bgr: np.ndarray, threshold: int = INK_THRESHOLD) -> float:
    """Fraction of pixels darker than ``threshold`` (grey)."""
    import cv2

    if crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    return float(np.count_nonzero(gray < threshold)) / float(gray.size)


def is_blank(crop_bgr: np.ndarray, min_ink: float = MIN_INK_FRACTION) -> bool:
    """True when the crop is (almost) uniformly light or uniformly dark, i.e.
    carries no glyphs for the recogniser to read."""
    ink = ink_fraction(crop_bgr)
    return ink < min_ink or ink > 1.0 - min_ink


def preprocess_crop(crop_bgr: np.ndarray, dtype: DTypeLike = np.float32) -> np.ndarray:
    """BGR crop -> (1, 3, 224, 224) normalised tensor as manga-ocr does it:
    grey, back to 3 identical channels, resize, ``(x / 255 - 0.5) / 0.5``."""
    import cv2

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    chw = np.repeat(norm[None, None, :, :], 3, axis=1)
    return np.ascontiguousarray(chw.astype(dtype, copy=False))


def _crop_quad(image: np.ndarray, quad: np.ndarray, pad: int = CROP_PAD_PX) -> Optional[np.ndarray]:
    """Axis-aligned crop of ``quad`` padded by ``pad`` px and clamped; None if empty."""
    h, w = image.shape[:2]
    x0 = max(0, int(np.floor(quad[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(quad[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(quad[:, 0].max())) + pad)
    y1 = min(h, int(np.ceil(quad[:, 1].max())) + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return image[y0:y1, x0:x1]


# ------------------------------------------------------------- model files
@dataclass(frozen=True)
class GenerationParams:
    decoder_start_token_id: int = DEFAULT_DECODER_START
    eos_token_id: int = DEFAULT_EOS
    pad_token_id: int = DEFAULT_PAD
    max_length: int = DEFAULT_MAX_LENGTH


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s (%s); using defaults", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def load_generation_params(model_dir: Path) -> GenerationParams:
    """Token ids / max_length from ``generation_config.json`` with
    ``config.json`` as the second source and the known constants last."""
    gen = _read_json(model_dir / M.GENERATION_CONFIG_FILE)
    cfg = _read_json(model_dir / M.CONFIG_FILE)

    def pick(key: str, default: int) -> int:
        for source in (gen, cfg):
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return default

    return GenerationParams(
        decoder_start_token_id=pick("decoder_start_token_id", DEFAULT_DECODER_START),
        eos_token_id=pick("eos_token_id", DEFAULT_EOS),
        pad_token_id=pick("pad_token_id", DEFAULT_PAD),
        max_length=max(1, pick("max_length", DEFAULT_MAX_LENGTH)),
    )


def load_vocab(model_dir: Path) -> List[str]:
    """``vocab.txt`` as an id -> token list (one token per line)."""
    lines = (model_dir / M.VOCAB_FILE).read_text(encoding="utf-8").split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError(f"{model_dir / M.VOCAB_FILE} is empty")
    return lines


def _input_dtype(session: object, name: str) -> DTypeLike:
    for meta in session.get_inputs():  # type: ignore[attr-defined]
        if meta.name == name:
            return np.float16 if "float16" in meta.type else np.float32
    return np.float32


# --------------------------------------------------------------- decoding
class GreedyDecoder:
    """Encoder + decoder sessions plus the tokenizer tables; recognises one
    preprocessed crop at a time."""

    def __init__(
        self,
        encoder: object,
        decoder: object,
        vocab: Sequence[str],
        params: GenerationParams,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = list(vocab)
        self.params = params
        self.special_ids: Set[int] = {i for i, t in enumerate(self.vocab) if t in SPECIAL_TOKENS}
        self.special_ids.update({params.pad_token_id, params.eos_token_id, params.decoder_start_token_id})
        self.encoder_input = encoder.get_inputs()[0].name  # type: ignore[attr-defined]
        self.encoder_dtype = _input_dtype(encoder, self.encoder_input)
        self.hidden_dtype = _input_dtype(decoder, "encoder_hidden_states")

    @classmethod
    def from_dir(cls, model_dir: Path, encoder: object, decoder: object) -> "GreedyDecoder":
        return cls(encoder, decoder, load_vocab(model_dir), load_generation_params(model_dir))

    def generate(self, pixel_values: np.ndarray) -> Tuple[List[int], bool]:
        """Greedy token ids for one crop and whether the decode ran away
        (tail repeats or max_length hit without eos)."""
        hidden = self.encoder.run(None, {self.encoder_input: pixel_values})[0]  # type: ignore[attr-defined]
        hidden = np.ascontiguousarray(hidden.astype(self.hidden_dtype, copy=False))
        ids: List[int] = [self.params.decoder_start_token_id]
        eos = self.params.eos_token_id
        for _ in range(self.params.max_length - 1):
            input_ids = np.asarray([ids], dtype=np.int64)
            logits = self.decoder.run(None, {"input_ids": input_ids, "encoder_hidden_states": hidden})[0]  # type: ignore[attr-defined]
            nxt = int(np.argmax(logits[0, -1]))
            if nxt == eos:
                return ids[1:], False
            ids.append(nxt)
            if _tail_repeats(ids[1:]):
                return ids[1:], True
        return ids[1:], True

    def recognize(self, pixel_values: np.ndarray) -> str:
        """Post-processed text for one crop; empty for runaway decodes."""
        ids, runaway = self.generate(pixel_values)
        text = post_process(decode_tokens(ids, self.vocab, self.special_ids))
        if runaway and (is_degenerate(text) or _tail_repeats(ids)):
            return ""
        return text


# ----------------------------------------------------------------- engine
class MangaOcrEngine(OCREngine):
    """Line detector + manga-ocr recogniser.

    Args:
        models_dir: the app models directory (``manga-ocr/`` is appended).
        detector: engine whose ``recognize`` supplies line quads and scores;
            its text is ignored.
        device: ``"auto"`` uses DirectML for the encoder when available and
            falls back to CPU; ``"gpu"`` requires DirectML; ``"cpu"`` forces
            the CPU provider.
        min_confidence: detector segments scoring below this are dropped.
        session_factory: ``(path, providers) -> session``; tests inject fakes.

    Raises ``FileNotFoundError`` when the model files are not downloaded yet
    (see :func:`glasstranslate.ocr.models.ensure_models`).
    """

    name = "mangaocr"

    def __init__(
        self,
        models_dir: Union[str, Path],
        detector: OCREngine,
        *,
        device: str = "auto",
        min_confidence: float = 0.5,
        session_factory: Optional[SessionFactory] = None,
    ) -> None:
        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"unknown OCR device {device!r}; expected auto, gpu or cpu")
        self.detector = detector
        self.min_confidence = float(min_confidence)
        self.model_dir = M.manga_ocr_dir(models_dir)
        self._factory: SessionFactory = session_factory or _default_session_factory
        self.device = "cpu"
        self._crop_failures = 0

        missing = M.missing_files(models_dir)
        if missing:
            names = ", ".join(f.name for f in missing)
            raise FileNotFoundError(f"manga-ocr model files missing in {self.model_dir}: {names}")

        encoder = self._build_encoder(device)
        decoder = self._factory(str(self.model_dir / M.DECODER_FILE), [_CPU_PROVIDER])
        self._decoder = GreedyDecoder.from_dir(self.model_dir, encoder, decoder)
        log.info("manga-ocr ready on %s (detector=%s)", self.device, getattr(detector, "name", "?"))

    def _build_encoder(self, device: str) -> object:
        """Encoder session on DirectML (device auto/gpu) or the CPU."""
        path = str(self.model_dir / M.ENCODER_FILE)
        if device in ("auto", "gpu"):
            try:
                if not _dml_available():
                    raise RuntimeError(f"{_DML_PROVIDER} not available in onnxruntime")
                session = self._factory(path, [_DML_PROVIDER, _CPU_PROVIDER])
                self.device = "gpu"
                return session
            except Exception as exc:  # noqa: BLE001 - any DML failure means fall back
                if device == "gpu":
                    raise
                log.warning("manga-ocr DirectML init failed (%s); falling back to CPU", exc)
        self.device = "cpu"
        return self._factory(path, [_CPU_PROVIDER])

    # ------------------------------------------------------------ OCREngine
    def warmup(self) -> None:
        """Run one crop through encoder + decoder (initialises the providers)."""
        crop = np.full((32, 96, 3), 255, np.uint8)
        crop[8:24, 16:80] = 0
        self._decoder.recognize(preprocess_crop(crop, self._decoder.encoder_dtype))

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        """Detect lines with ``detector`` and read each with manga-ocr."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        image = np.ascontiguousarray(image_bgr)
        segments: List[Segment] = []
        for box in self.detector.recognize(image):
            if box.confidence < self.min_confidence:
                continue
            text = self._read_line(image, box)
            if text:
                quad = np.ascontiguousarray(np.asarray(box.quad, dtype=np.float32).reshape(4, 2))
                segments.append(Segment(text=text, quad=quad, confidence=float(box.confidence), lang_hint="ja"))
        return segments

    def _read_line(self, image: np.ndarray, box: Segment) -> str:
        """Text of one detected line, or "" when it should be dropped."""
        try:
            crop = _crop_quad(image, np.asarray(box.quad, dtype=np.float32).reshape(4, 2))
            if crop is None or is_blank(crop):
                return ""
            text = self._decoder.recognize(preprocess_crop(crop, self._decoder.encoder_dtype))
        except Exception as exc:  # noqa: BLE001 - one bad crop must not kill the frame
            self._report_crop_failure(exc)
            return ""
        if not text or is_degenerate(text):
            return ""
        return text

    def _report_crop_failure(self, exc: Exception) -> None:
        """Warn for the first few per-crop failures, then log at debug."""
        self._crop_failures += 1
        if self._crop_failures <= CROP_FAILURE_WARNINGS:
            log.warning("manga-ocr skipped a crop (%s: %s)", type(exc).__name__, exc)
        else:
            log.debug("manga-ocr skipped a crop (%s: %s)", type(exc).__name__, exc)
