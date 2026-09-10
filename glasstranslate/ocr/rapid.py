"""OCR engine backed by ``rapidocr`` (PP-OCR det/cls/rec through onnxruntime).

DirectML is used when available so the models run on the GPU; otherwise the
same package runs on the CPU execution provider.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from glasstranslate.core.interfaces import OCREngine
from glasstranslate.core.types import Segment

log = logging.getLogger(__name__)

_DML_PROVIDER = "DmlExecutionProvider"

# RapidOCR logs a line per inference at INFO; keep only warnings and above.
logging.getLogger("RapidOCR").setLevel(logging.WARNING)


def _dml_available() -> bool:
    """True if the installed onnxruntime exposes the DirectML provider."""
    import onnxruntime as ort

    return _DML_PROVIDER in ort.get_available_providers()


class RapidOCREngine(OCREngine):
    """Text detection + recognition with rotated quads and confidences.

    Args:
        device: ``"auto"`` tries DirectML first and falls back to CPU;
            ``"gpu"`` requires DirectML; ``"cpu"`` forces the CPU provider.
        min_confidence: segments whose recognition score is below this are
            dropped.
        use_cls: run the text-angle classifier (fixes upside-down lines at
            the cost of one extra model pass per segment).
    """

    # Config name: this is the "PaddleOCR (PP-OCRv5 via ONNX)" engine of the
    # registry (``ocr/factory.py``); "rapidocr" is only an accepted alias.
    name = "paddleocr"

    def __init__(self, device: str = "auto", min_confidence: float = 0.5, use_cls: bool = True) -> None:
        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"unknown OCR device {device!r}; expected auto, gpu or cpu")
        self.min_confidence = float(min_confidence)
        self.use_cls = bool(use_cls)
        self._engine: Any = None
        self.device = "cpu"

        if device in ("auto", "gpu"):
            try:
                if not _dml_available():
                    raise RuntimeError(f"{_DML_PROVIDER} not available in onnxruntime")
                self._engine = self._build(use_dml=True)
                self.device = "gpu"
            except Exception as exc:  # noqa: BLE001 - any DML failure means fall back
                if device == "gpu":
                    raise
                log.warning("RapidOCR DirectML init failed (%s); falling back to CPU", exc)
        if self._engine is None:
            self._engine = self._build(use_dml=False)
        log.info("RapidOCR ready on %s (use_cls=%s)", self.device, self.use_cls)

    def _build(self, use_dml: bool) -> Any:
        from rapidocr import RapidOCR

        params: Dict[str, Any] = {
            "EngineConfig.onnxruntime.use_dml": use_dml,
            "Global.use_cls": self.use_cls,
            "Global.text_score": self.min_confidence,
            "Global.log_level": "warning",
        }
        return RapidOCR(params=params)

    def warmup(self) -> None:
        """Run one inference containing real text so the detector, angle
        classifier and recogniser are all initialised (a blank image would
        only exercise the detector and make RapidOCR log a warning)."""
        import cv2

        img = np.full((64, 320, 3), 255, np.uint8)
        cv2.putText(img, "warmup 123", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
        self.recognize(img)

    def recognize(self, image_bgr: np.ndarray) -> List[Segment]:
        """OCR ``image_bgr`` (HxWx3 uint8) into segments in image coordinates."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        img = np.ascontiguousarray(image_bgr)
        result = self._engine(img)
        if result.boxes is None or result.txts is None or result.scores is None:
            return []

        boxes = np.asarray(result.boxes, dtype=np.float32).reshape(-1, 4, 2)
        segments: List[Segment] = []
        for quad, text, score in zip(boxes, result.txts, result.scores):
            conf = float(score)
            text = text.strip()
            # Low-confidence lines are dropped here (and by RapidOCR's own
            # ``Global.text_score``); see the TODO in ``Pipeline._group_blocks``
            # for threading them to the eraser as ``all_segments``.
            if not text or conf < self.min_confidence:
                continue
            segments.append(Segment(text=text, quad=np.ascontiguousarray(quad), confidence=conf))
        return segments
