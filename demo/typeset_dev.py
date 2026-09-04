"""Fast typesetting dev loop for one manga page (default ``Examples/before.jpg``).

OCR results and machine translations are cached under ``demo/cache`` so a
re-render after a code change takes about a second instead of loading the
engines.  ``--ref-text`` letters every block with the English from the
professionally typeset reference (``demo/reference_text.json``) so layout
and lettering can be compared against ``Examples/after.webp`` with identical
wording.

Outputs (in ``--out``, default ``demo/output/dev``), all prefixed by ``--tag``:

* ``<tag>_typeset.png``   the translated page at full resolution
* ``<tag>_erased.png``    the page with the original text erased, no lettering
* ``<tag>_blocks.png``    debug view (source lines green, furigana grey,
  bubble mask blue, layout box red, seed box orange)
* ``<tag>_small.jpg``     the translated page scaled to 720 px tall (greyscale)
* ``<tag>_cmp_<region>.jpg``  side-by-side crops  original | ours | reference
  for the regions in ``REGIONS`` (or ``--regions``)

Keep every image you *view* small: the ``_small`` and ``_cmp_`` files are
greyscale JPEGs of well under 100 KB made for that.  Never open the full-size
PNGs in a viewer that feeds a language model (a 1 MB image costs hundreds of
thousands of tokens).

The pipeline under test is exactly the library's: ``build_blocks`` (grouping,
sizing, ``erase.apply`` and ``place.prepare``) followed by ``compose``, whose
``typeset_block`` is the shared placement geometry (``render.place``).
``build_blocks`` gets the OCR lines of *every* confidence so the eraser can use
the low-confidence boxes as evidence of glyphs.  The flags ``--new-erase`` /
``--new-place`` are kept for old command lines and do nothing (a note is
printed).

Usage::

    set PYTHONUTF8=1
    .venv/Scripts/python.exe demo/typeset_dev.py --ref-text --tag base
    .venv/Scripts/python.exe demo/typeset_dev.py --ref-text --tag c --regions C_panel3_right --quiet
    .venv/Scripts/python.exe demo/typeset_dev.py --ocr --translate   # refresh the caches
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glasstranslate.core.pipeline import clean_translation  # noqa: E402
from glasstranslate.core.types import Segment, StyledSegment, TranslatedSegment  # noqa: E402
compose_mod = importlib.import_module("glasstranslate.render.compose")  # the package re-exports the function under this name
from glasstranslate.render import build_blocks  # noqa: E402
from glasstranslate.render.compose import block_font_path_for, has_cjk, pil_measurer  # noqa: E402

log = logging.getLogger("typeset_dev")

CACHE_DIR = PROJECT_ROOT / "demo" / "cache"
REFERENCE_IMAGE = PROJECT_ROOT / "Examples" / "after.webp"
REFERENCE_TEXT = PROJECT_ROOT / "demo" / "reference_text.json"

# Crop regions of Examples/before.jpg (x1, y1, x2, y2) used for comparison sheets.
REGIONS: Dict[str, Tuple[int, int, int, int]] = {
    "A_leftmid": (0, 380, 230, 720),  # free text over artwork, no box
    "B_panel2_right": (620, 820, 1096, 1240),  # two dialogue blocks beside the face
    "C_panel3_right": (600, 1260, 1096, 1600),  # dialogue over hatched art
    "D_panel3_left": (0, 1260, 380, 1600),  # dialogue on flat paper
    "E_topleft": (60, 0, 420, 380),  # two joined caption boxes
    "F_topright": (840, 0, 1096, 330),  # rectangular caption box
    "G_caption": (560, 730, 1096, 900),  # chapter title: horizontal caption in the gutter, no box
}
TILE_W = 240  # width of each tile in a comparison sheet
SHEET_QUALITY = 65  # JPEG quality; sheets are greyscale and must stay well under 60 KB for model viewing


# ----------------------------------------------------------------- caches
def load_segments(img: np.ndarray, refresh: bool, device: str) -> List[Segment]:
    path = CACHE_DIR / "segments_all.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as fh:
            return pickle.load(fh)
    from glasstranslate.ocr import RapidOCREngine

    ocr = RapidOCREngine(device=device, min_confidence=0.0)
    ocr.warmup()
    segs = ocr.recognize(img)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(segs, fh)
    log.info("OCR: %d lines cached to %s", len(segs), path)
    return segs


def load_translations(texts: Sequence[str], src: str, tgt: str, refresh: bool, device: str, models: Path) -> Dict[str, str]:
    path = CACHE_DIR / f"translations_{src}_{tgt}.json"
    cache: Dict[str, str] = {}
    if path.exists() and not refresh:
        cache = json.loads(path.read_text(encoding="utf-8"))
    missing = [t for t in texts if t not in cache]
    if missing:
        from glasstranslate.translate import ArgosCT2Translator, IdentityTranslator

        translator = ArgosCT2Translator(models, device={"auto": "auto", "gpu": "cuda", "cpu": "cpu"}[device])
        if not translator.supports(src, tgt):
            log.warning("no %s->%s model in %s; identity translation", src, tgt, models)
            translator = IdentityTranslator()
        for text, tr in zip(missing, translator.translate_batch(missing, src, tgt)):
            cache[text] = clean_translation(tr) or text
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info("translated %d blocks, cache %s", len(missing), path)
    return cache


# ------------------------------------------------------------------ views
def draw_blocks(img: np.ndarray, blocks) -> np.ndarray:
    out = img.copy()
    for i, b in enumerate(blocks):
        st = b.style
        if st.layout_mask is not None and st.layout_box is not None:
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
        sd = st.layout_seed
        if sd is not None:
            cv2.rectangle(out, (sd.x, sd.y), (sd.x2, sd.y2), (0, 140, 255), 1)
        bb = b.segment.bbox
        label = f"{i}{'B' if b.bubble else ''}{'O' if st.outline else ''}"
        cv2.putText(out, label, (bb.x, max(12, bb.y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 230), 1, cv2.LINE_AA)
    return out


def small(img_bgr: np.ndarray, long_side: int = 720) -> Image.Image:
    im = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1])).convert("L")
    scale = long_side / max(im.size)
    return im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)


def comparison_sheet(
    orig: Image.Image, ours: Image.Image, ref: Optional[Image.Image], region: Tuple[int, int, int, int]
) -> Image.Image:
    """``original | ours | reference`` crops of one region, each ``TILE_W`` wide.
    The reference has a different pixel size and a slightly different crop,
    so it is mapped by independent x/y scale factors (close enough for a
    visual comparison)."""
    x1, y1, x2, y2 = region
    a = orig.crop(region)
    b = ours.crop(region)
    tiles = [a, b]
    if ref is not None:
        sx, sy = ref.width / orig.width, ref.height / orig.height
        c = ref.crop((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))).resize(a.size, Image.LANCZOS)
        tiles.append(c)
    scale = TILE_W / a.width
    h = max(1, int(a.height * scale))
    tiles = [t.resize((TILE_W, h), Image.LANCZOS) for t in tiles]
    sheet = Image.new("L", (TILE_W * len(tiles) + 6 * (len(tiles) - 1), h), 110)
    for i, t in enumerate(tiles):
        sheet.paste(t.convert("L"), (i * (TILE_W + 6), 0))
    return sheet


# ------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, default=PROJECT_ROOT / "Examples" / "before.jpg")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo" / "output" / "dev")
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--ocr", action="store_true", help="re-run OCR instead of using the cache")
    parser.add_argument("--translate", action="store_true", help="re-run translation instead of using the cache")
    parser.add_argument("--ref-text", action="store_true", help="letter blocks with the reference English text")
    parser.add_argument("--src", default="ja")
    parser.add_argument("--tgt", default="en")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--no-upper", action="store_true")
    parser.add_argument("--new-erase", action="store_true", help="no-op (erase.apply runs inside build_blocks)")
    parser.add_argument("--new-place", action="store_true", help="no-op (place is the default geometry)")
    parser.add_argument("--regions", default="all", help="comma-separated region names, or 'all'/'none'")
    parser.add_argument("--no-ref-image", action="store_true", help="omit the reference tile from the sheets")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"cannot read {args.image}", file=sys.stderr)
        return 2

    all_segments = load_segments(img, args.ocr, args.device)
    segments = [s for s in all_segments if s.confidence >= args.min_confidence]

    if args.new_erase or args.new_place:
        print("note: --new-erase/--new-place are no-ops: build_blocks runs erase.apply and place.prepare, "
              "and compose.typeset_block is the placement geometry")
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    blocks = build_blocks(img, segments, all_segments, timings=timings)
    t1 = time.perf_counter()

    texts = [b.segment.text for b in blocks]
    translations = load_translations(texts, args.src, args.tgt, args.translate, args.device, args.models)
    reference: Dict[str, str] = {}
    if args.ref_text and REFERENCE_TEXT.exists():
        reference = {k: v for k, v in json.loads(REFERENCE_TEXT.read_text(encoding="utf-8")).items() if not k.startswith("_")}

    translated: List[TranslatedSegment] = []
    for b in blocks:
        text = b.segment.text
        tr = reference.get(text) or translations.get(text) or text
        styled = StyledSegment(segment=b.segment, style=b.style, src_lang=args.src)
        translated.append(TranslatedSegment(styled=styled, translation=tr, tgt_lang=args.tgt))

    t4 = time.perf_counter()
    out = compose_mod.compose(img, translated, hide_original=True, uppercase=not args.no_upper)
    t5 = time.perf_counter()
    erased_only = [TranslatedSegment(s.styled, "", s.tgt_lang) for s in translated if not s.untranslated]
    erased = compose_mod.compose(img, erased_only, hide_original=True)

    # Per-block report: what the typesetter chose.
    print(f"{'#':>2} {'kind':6} {'glyph':>5} {'ceil':>5} {'size':>5} {'ln':>2} {'box':>18}  text")
    for i, (b, seg) in enumerate(zip(blocks, translated)):
        st = b.style
        kind = "bubble" if st.in_bubble else ("art" if st.outline else "flat")
        text = seg.translation.strip()
        if not args.no_upper and not has_cjk(text):
            text = text.upper()
        size_s, n_s, box_s = "-", "-", "-"
        if not seg.untranslated and text and st.layout_box is not None:
            bpath = block_font_path_for(st, text, compose_mod.default_font_path())
            try:
                ts = compose_mod.typeset_block(text, st, pil_measurer(bpath), lang=args.tgt)
                bb = ts.bbox
                size_s = f"{ts.size:5.1f}"
                n_s = f"{len(ts.lines):2d}"
                box_s = f"{bb.w}x{bb.h}@{bb.x},{bb.y}" if bb else "-"
                if not ts.fitted:
                    size_s += "!"
            except Exception as exc:  # experimental stages may raise; keep the report going
                size_s = f"ERR {type(exc).__name__}"
        ceil = f"{st.max_font_px:5.1f}" if st.max_font_px else "-"
        print(f"{i:>2} {kind:6} {st.text_height_px:5.1f} {ceil:>5} {size_s:>6} {n_s:>2} {box_s:>18}  {text[:60]!r}")
    print(
        f"{len(segments)} lines -> {len(blocks)} blocks | layout {1000*(t1-t0):.0f} ms"
        f" (group {timings.get('group', 0):.0f}, erase {timings.get('erase', 0):.0f},"
        f" place-prep {timings.get('place', 0):.0f}), render {1000*(t5-t4):.0f} ms"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    cv2.imwrite(str(args.out / f"{tag}_typeset.png"), out)
    cv2.imwrite(str(args.out / f"{tag}_erased.png"), erased)
    cv2.imwrite(str(args.out / f"{tag}_blocks.png"), draw_blocks(img, blocks))
    small(out).save(args.out / f"{tag}_small.jpg", quality=SHEET_QUALITY)

    if args.regions != "none":
        names = list(REGIONS) if args.regions == "all" else [r.strip() for r in args.regions.split(",") if r.strip()]
        orig_pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        ours_pil = Image.fromarray(np.ascontiguousarray(out[:, :, ::-1]))
        erased_pil = Image.fromarray(np.ascontiguousarray(erased[:, :, ::-1]))
        ref_pil = None
        if not args.no_ref_image and REFERENCE_IMAGE.exists() and args.image.resolve() == (PROJECT_ROOT / "Examples" / "before.jpg").resolve():
            ref_pil = Image.open(REFERENCE_IMAGE).convert("RGB")
        for name in names:
            region = REGIONS.get(name)
            if region is None:
                print(f"unknown region {name!r}; known: {', '.join(REGIONS)}", file=sys.stderr)
                continue
            comparison_sheet(orig_pil, ours_pil, ref_pil, region).save(args.out / f"{tag}_cmp_{name}.jpg", quality=SHEET_QUALITY)
            comparison_sheet(orig_pil, erased_pil, None, region).save(args.out / f"{tag}_erasecmp_{name}.jpg", quality=SHEET_QUALITY)
    print(f"wrote {args.out / (tag + '_typeset.png')} (+ _erased, _blocks, _small, _cmp_*, _erasecmp_*)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
