"""Dev-only probe for the manga-ocr ONNX recogniser (docs/perf/*-mangaocr-probe.md).

Detects text lines on a page with rapidocr, then runs each crop through the
manga-ocr encoder/decoder on the requested onnxruntime providers and prints
per-crop latency plus the recognised text next to rapidocr's own reading.

    PYTHONUTF8=1 .venv/Scripts/python.exe tools/ocr_probe.py Examples/before.jpg --providers dml,cpu
    PYTHONUTF8=1 .venv/Scripts/python.exe tools/ocr_probe.py Examples/before.jpg --engine

``--engine`` runs the shipped :class:`glasstranslate.ocr.mangaocr.MangaOcrEngine`
end to end instead of the raw sessions.  Printing is intentional: this is a
command-line report, not library code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glasstranslate.config.settings import default_models_dir  # noqa: E402
from glasstranslate.core.types import Segment  # noqa: E402
from glasstranslate.ocr import models as M  # noqa: E402
from glasstranslate.ocr.rapid import RapidOCREngine  # noqa: E402

_PROVIDERS: Dict[str, List[str]] = {
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


def _describe_session(label: str, sess) -> None:  # type: ignore[no-untyped-def]
    print(f"  [{label}] providers={sess.get_providers()}")
    for i in sess.get_inputs():
        print(f"    in  {i.name:32} {i.type:18} {i.shape}")
    for o in sess.get_outputs():
        print(f"    out {o.name:32} {o.type:18} {o.shape}")


def _load_sessions(model_dir: Path, providers: Sequence[str]):  # type: ignore[no-untyped-def]
    import onnxruntime as ort

    t0 = time.perf_counter()
    enc = ort.InferenceSession(str(model_dir / M.ENCODER_FILE), providers=list(providers))
    t1 = time.perf_counter()
    dec = ort.InferenceSession(str(model_dir / M.DECODER_FILE), providers=list(providers))
    t2 = time.perf_counter()
    print(f"  session load: encoder {1000 * (t1 - t0):.0f} ms, decoder {1000 * (t2 - t1):.0f} ms")
    return enc, dec


def _crops(image: np.ndarray, segments: Sequence[Segment], pad: int = 2) -> List[np.ndarray]:
    h, w = image.shape[:2]
    out: List[np.ndarray] = []
    for seg in segments:
        r = seg.bbox
        x0, y0 = max(0, r.x - pad), max(0, r.y - pad)
        x1, y1 = min(w, r.x2 + pad), min(h, r.y2 + pad)
        out.append(image[y0:y1, x0:x1])
    return out


def _run_raw(model_dir: Path, crops: Sequence[np.ndarray], provider_key: str) -> List[Tuple[str, float]]:
    """Recognise ``crops`` with raw sessions; returns (text, ms) per crop."""
    from glasstranslate.ocr.mangaocr import GreedyDecoder, preprocess_crop

    print(f"provider set {provider_key!r}:")
    enc, dec = _load_sessions(model_dir, _PROVIDERS[provider_key])
    _describe_session("encoder", enc)
    _describe_session("decoder", dec)
    decoder = GreedyDecoder.from_dir(model_dir, enc, dec)
    results: List[Tuple[str, float]] = []
    for crop in crops:
        t0 = time.perf_counter()
        text = decoder.recognize(preprocess_crop(crop, decoder.encoder_dtype))
        results.append((text, 1000 * (time.perf_counter() - t0)))
    return results


def _run_engine(model_dir: Path, image: np.ndarray, detector: RapidOCREngine, device: str) -> None:
    from glasstranslate.ocr.mangaocr import MangaOcrEngine

    t0 = time.perf_counter()
    engine = MangaOcrEngine(model_dir.parent, detector, device=device)
    print(f"engine built on {engine.device} in {1000 * (time.perf_counter() - t0):.0f} ms")
    engine.warmup()
    t0 = time.perf_counter()
    segs = engine.recognize(image)
    print(f"recognize: {len(segs)} segments in {1000 * (time.perf_counter() - t0):.0f} ms")
    for s in segs:
        print(f"  {s.confidence:.2f} {s.text}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("--models-dir", type=Path, default=default_models_dir())
    ap.add_argument("--providers", default="dml,cpu", help="comma list of dml,cpu")
    ap.add_argument("--max-lines", type=int, default=12)
    ap.add_argument("--engine", action="store_true", help="run MangaOcrEngine end to end")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", type=Path, help="write per-line results here")
    args = ap.parse_args(argv)

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"cannot read {args.image}")
        return 2
    model_dir = M.manga_ocr_dir(args.models_dir)
    missing = [f.name for f in M.missing_files(args.models_dir)]
    if missing:
        print(f"missing model files in {model_dir}: {missing}")
        return 2

    detector = RapidOCREngine(device="auto", min_confidence=0.5)
    if args.engine:
        _run_engine(model_dir, image, detector, args.device)
        return 0

    t0 = time.perf_counter()
    rapid = detector.recognize(image)
    print(f"rapidocr ({detector.device}): {len(rapid)} lines in {1000 * (time.perf_counter() - t0):.0f} ms")
    rapid = rapid[: args.max_lines]
    crops = _crops(image, rapid)
    report: Dict[str, List[Dict[str, object]]] = {}
    for key in [k.strip() for k in args.providers.split(",") if k.strip()]:
        results = _run_raw(model_dir, crops, key)
        lat = [ms for _, ms in results]
        print(f"  per-crop ms: first {lat[0]:.0f}, median {np.median(lat[1:] or lat):.0f}, max {max(lat):.0f}")
        rows = []
        for seg, (text, ms) in zip(rapid, results):
            print(f"  {ms:6.0f} ms  manga-ocr: {text!r:40}  rapidocr: {seg.text!r}")
            rows.append({"ms": ms, "mangaocr": text, "rapidocr": seg.text})
        report[key] = rows
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
