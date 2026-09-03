"""End-to-end GlassTranslate demo on static images (no Qt, no screen capture).

For every PNG/JPG in ``--images``: OCR with :class:`RapidOCREngine`, measure
each segment's style, detect the source language (unless ``--src`` is given),
translate (Argos via CTranslate2 when a model for the pair exists in
``--models``, otherwise the identity translator with a warning), compose the
translated overlay with PIL and write two files to ``--out``:

* ``<name>_overlay.png`` - the original with every segment replaced by its
  translation in the measured colours and orientation.
* ``<name>_debug.png``   - the original with each quad outlined in its measured
  foreground colour and the angle plus fg/bg hex codes printed beside it.

A per-image latency table is printed at the end.

Usage::

    .venv\\Scripts\\python demo\\run_demo.py --src en --tgt de --backend argos --device auto
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glasstranslate.core.interfaces import OCREngine, Translator  # noqa: E402
from glasstranslate.core.types import RGB, Segment, StyledSegment, TranslatedSegment  # noqa: E402
from glasstranslate.ocr import RapidOCREngine, ScriptLanguageDetector  # noqa: E402
from glasstranslate.render import compose, measure_style  # noqa: E402
from glasstranslate.translate import ArgosCT2Translator, IdentityTranslator  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES = DEMO_DIR / "images"
DEFAULT_OUTPUT = DEMO_DIR / "output"
DEFAULT_MODELS = PROJECT_ROOT / "models"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

# OCR device names (RapidOCREngine) -> ctranslate2 device names (ArgosCT2Translator).
_TRANSLATE_DEVICE = {"auto": "auto", "gpu": "cuda", "cpu": "cpu"}

log = logging.getLogger("demo")


@dataclass
class ImageResult:
    """Timings and counts for one processed image (all times in ms)."""

    name: str
    segments: int
    src_lang: str
    ocr_ms: float
    style_ms: float
    translate_ms: float
    compose_ms: float

    @property
    def total_ms(self) -> float:
        return self.ocr_ms + self.style_ms + self.translate_ms + self.compose_ms


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def list_images(directory: Path) -> List[Path]:
    """Image files in ``directory`` sorted by name."""
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def build_translator(backend: str, models_dir: Path, device: str, src: Optional[str], tgt: str) -> Translator:
    """Pick the translator for the run.

    ``argos`` is used when the models directory holds a package (or English
    pivot) for ``src -> tgt``; otherwise, or when ``--backend identity`` is
    given, the identity translator is returned.  When ``src`` is None (auto
    detection) the Argos translator is returned as long as any package exists
    and the per-image pair check happens in :func:`translate_segments`.
    """
    if backend == "identity":
        return IdentityTranslator()
    argos = ArgosCT2Translator(models_dir, device=_TRANSLATE_DEVICE[device])
    pairs = set(argos.supported_pairs())
    if not pairs:
        print(f"WARNING: no Argos packages in {models_dir}; using identity translator", file=sys.stderr)
        return IdentityTranslator()
    if src is not None and (src, tgt) not in pairs:
        print(
            f"WARNING: no Argos package for {src}->{tgt} in {models_dir} "
            f"(have {sorted(pairs)}); using identity translator",
            file=sys.stderr,
        )
        return IdentityTranslator()
    return argos


def translate_segments(
    translator: Translator, styled: Sequence[StyledSegment], src: str, tgt: str
) -> List[TranslatedSegment]:
    """Translate all segments in one batch; unsupported pairs pass through."""
    texts = [s.segment.text for s in styled]
    if src == tgt or not translator.supports(src, tgt):
        if src != tgt:
            print(f"WARNING: {translator.name} cannot translate {src}->{tgt}; leaving text unchanged", file=sys.stderr)
        translations = list(texts)
    else:
        translations = translator.translate_batch(texts, src, tgt)
    return [TranslatedSegment(styled=s, translation=t, tgt_lang=tgt) for s, t in zip(styled, translations)]


def draw_debug(img_bgr: np.ndarray, styled: Sequence[StyledSegment]) -> np.ndarray:
    """Outline each quad in its measured fg colour (BGR for OpenCV) and print
    ``angle  fg  bg`` beside it in the same colour on a small dark tag."""
    out = img_bgr.copy()
    for s in styled:
        fg_bgr = tuple(int(c) for c in reversed(s.style.fg))
        pts = np.round(s.segment.quad).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=fg_bgr, thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(out, tuple(int(v) for v in pts[0, 0]), 5, fg_bgr, -1, lineType=cv2.LINE_AA)  # p0 marker

        label = f"{s.style.angle_deg:+.1f}deg fg {_hex(s.style.fg)} bg {_hex(s.style.bg)}"
        if s.style.vertical:
            label += " vertical"
        bbox = s.segment.bbox
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x = min(max(bbox.x, 0), out.shape[1] - tw - 4)
        y = bbox.y - 6 if bbox.y - th - 10 >= 0 else bbox.y2 + th + 6
        cv2.rectangle(out, (x - 2, y - th - 3), (x + tw + 2, y + baseline), (40, 40, 40), -1)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, fg_bgr, 1, cv2.LINE_AA)
    return out


def process_image(
    path: Path,
    out_dir: Path,
    ocr: OCREngine,
    detector: ScriptLanguageDetector,
    translator: Translator,
    src: Optional[str],
    tgt: str,
    font_path: Optional[str],
) -> ImageResult:
    """Run the full pipeline on one image and write overlay + debug PNGs."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")

    t = time.perf_counter()
    segments: List[Segment] = ocr.recognize(img)
    ocr_ms = _ms(t)

    t = time.perf_counter()
    detected = src or detector.detect_dominant([s.text for s in segments]) or "en"
    styled = [StyledSegment(segment=s, style=measure_style(img, s), src_lang=detected) for s in segments]
    style_ms = _ms(t)

    t = time.perf_counter()
    translated = translate_segments(translator, styled, detected, tgt) if styled else []
    translate_ms = _ms(t)

    t = time.perf_counter()
    overlay = compose(img, translated, font_path=font_path, hide_original=True)
    compose_ms = _ms(t)

    cv2.imwrite(str(out_dir / f"{path.stem}_overlay.png"), overlay)
    cv2.imwrite(str(out_dir / f"{path.stem}_debug.png"), draw_debug(img, styled))

    for seg in translated:
        st = seg.style
        print(
            f"  [{st.angle_deg:+6.1f}deg fg {_hex(st.fg)} bg {_hex(st.bg)}"
            f"{' V' if st.vertical else '  '}] {seg.source_text!r} -> {seg.translation!r}"
        )
    return ImageResult(path.stem, len(segments), detected, ocr_ms, style_ms, translate_ms, compose_ms)


def print_table(results: Iterable[ImageResult]) -> None:
    header = f"{'image':<14}{'segs':>5}{'lang':>6}{'ocr ms':>10}{'style ms':>10}{'transl ms':>11}{'compose ms':>12}{'total ms':>10}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<14}{r.segments:>5}{r.src_lang:>6}{r.ocr_ms:>10.1f}{r.style_ms:>10.1f}"
            f"{r.translate_ms:>11.1f}{r.compose_ms:>12.1f}{r.total_ms:>10.1f}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default=None, help="source language (ISO-639-1); default: auto-detect per image")
    parser.add_argument("--tgt", default="de", help="target language (default: de)")
    parser.add_argument("--backend", choices=("argos", "identity"), default="argos", help="translation backend")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "gpu"), default="auto", help="OCR device; gpu also means CUDA for Argos"
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="input directory (default: demo/images)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="output directory (default: demo/output)")
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS, help="Argos models directory (default: models/)")
    parser.add_argument("--font", default=None, help="TTF/TTC font for the overlay (default: system font with CJK)")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="drop OCR segments below this score")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    images = list_images(args.images) if args.images.is_dir() else []
    if not images:
        print(f"no images in {args.images}; run demo/make_images.py first", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    ocr = RapidOCREngine(device=args.device, min_confidence=args.min_confidence)
    init_ms = _ms(t)
    t = time.perf_counter()
    ocr.warmup()
    warmup_ms = _ms(t)
    print(f"OCR: {ocr.name} on {ocr.device} (init {init_ms:.0f} ms, warmup {warmup_ms:.0f} ms)")

    translator = build_translator(args.backend, args.models, args.device, args.src, args.tgt)
    print(f"translator: {translator.name} on {translator.device}")
    detector = ScriptLanguageDetector()
    detector.detect("load the language model now")  # ~300 ms one-off, kept out of the table

    results: List[ImageResult] = []
    for path in images:
        print(f"\n{path.name}")
        results.append(process_image(path, args.out, ocr, detector, translator, args.src, args.tgt, args.font))
    print_table(results)
    print(f"\noutputs in {args.out}")
    translator.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
