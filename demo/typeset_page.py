"""Typeset one manga page: OCR -> block grouping -> translation -> rendering.

Usage::

    .venv\\Scripts\\python demo\\typeset_page.py examples\\before.jpg --out demo\\output

Writes ``<name>_typeset.png`` (the translated page) and ``<name>_blocks.png``
(source lines in green, furigana in grey, bubble regions in blue, layout
boxes in red, block text index) next to it.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The report prints Japanese source text; a cp1252 console (Windows default)
# must not abort the run on it.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - closed or exotic stream
            pass

from glasstranslate.core.pipeline import clean_translation  # noqa: E402
from glasstranslate.core.types import StyledSegment, TranslatedSegment  # noqa: E402
from glasstranslate.ocr import RapidOCREngine, ScriptLanguageDetector  # noqa: E402
from glasstranslate.render import build_blocks, compose  # noqa: E402
from glasstranslate.translate import ArgosCT2Translator, IdentityTranslator  # noqa: E402

log = logging.getLogger("typeset")


def draw_blocks(img: np.ndarray, blocks) -> np.ndarray:
    out = img.copy()
    for i, b in enumerate(blocks):
        st = b.style
        if st.layout_mask is not None and st.layout_box is not None:
            # The mask is aligned with the layout box (not the bubble rect).
            overlay = out.copy()
            ys, xs = np.nonzero(st.layout_mask)
            ys = np.clip(ys + st.layout_box.y, 0, out.shape[0] - 1)
            xs = np.clip(xs + st.layout_box.x, 0, out.shape[1] - 1)
            overlay[ys, xs] = (255, 200, 120)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        for m in b.members:
            cv2.polylines(out, [m.quad.astype(np.int32)], True, (0, 180, 0), 1)
        for f in b.furigana:
            cv2.polylines(out, [f.quad.astype(np.int32)], True, (150, 150, 150), 1)
        lb = st.layout_box
        if lb is not None:
            cv2.rectangle(out, (lb.x, lb.y), (lb.x2, lb.y2), (0, 0, 230), 1)
        bb = b.segment.bbox
        label = f"{i}{'B' if b.bubble else ''}{'O' if st.outline else ''}"
        cv2.putText(out, label, (bb.x, max(12, bb.y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 230), 1, cv2.LINE_AA)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo" / "output")
    parser.add_argument("--src", default=None)
    parser.add_argument("--tgt", default="en")
    parser.add_argument("--backend", choices=("argos", "identity"), default="argos")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--no-upper", action="store_true", help="keep the translation's case")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    ocr = RapidOCREngine(device=args.device, min_confidence=args.min_confidence)
    ocr.warmup()
    t1 = time.perf_counter()
    segments = ocr.recognize(img)
    t2 = time.perf_counter()
    blocks = build_blocks(img, segments)
    t3 = time.perf_counter()

    detector = ScriptLanguageDetector()
    src = args.src or detector.detect_dominant([b.segment.text for b in blocks]) or "ja"
    if args.backend == "argos":
        translator = ArgosCT2Translator(args.models, device={"auto": "auto", "gpu": "cuda", "cpu": "cpu"}[args.device])
        if not translator.supports(src, args.tgt):
            log.warning("no model for %s->%s in %s; using identity", src, args.tgt, args.models)
            translator = IdentityTranslator()
    else:
        translator = IdentityTranslator()
    texts = [b.segment.text for b in blocks]
    translations = translator.translate_batch(texts, src, args.tgt)
    t4 = time.perf_counter()

    translated: List[TranslatedSegment] = []
    for b, tr in zip(blocks, translations):
        # Same rule as the live pipeline: drop <unk> markers, and show the
        # source untouched when nothing usable came back.
        tr = clean_translation(tr) or b.segment.text
        styled = StyledSegment(segment=b.segment, style=b.style, src_lang=src)
        translated.append(TranslatedSegment(styled=styled, translation=tr, tgt_lang=args.tgt))
        kind = "bubble" if b.bubble else ("art" if b.style.outline else "flat")
        print(f"[{kind:6}] {b.segment.text!r}\n         -> {tr!r}")
    out = compose(img, translated, hide_original=True, uppercase=not args.no_upper)
    t5 = time.perf_counter()

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    cv2.imwrite(str(args.out / f"{stem}_typeset.png"), out)
    cv2.imwrite(str(args.out / f"{stem}_blocks.png"), draw_blocks(img, blocks))
    print(
        f"\n{len(segments)} lines -> {len(blocks)} blocks | ocr {1000*(t2-t1):.0f} ms, "
        f"layout {1000*(t3-t2):.0f} ms, translate {1000*(t4-t3):.0f} ms, render {1000*(t5-t4):.0f} ms "
        f"(engine load {1000*(t1-t0):.0f} ms)"
    )
    print(f"wrote {args.out / (stem + '_typeset.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
