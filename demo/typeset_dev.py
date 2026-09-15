"""Fast typesetting dev loop for one manga page, or for every reference pair.

Works on any page image.  Everything derived from a page lives under a
per-page *stem* (the file name without its suffix):

* OCR results and machine translations are cached under ``demo/cache/<stem>/``
  so a re-render after a code change takes about a second instead of loading
  the engines;
* ``--ref-text`` letters every block with the English from the professionally
  typeset reference, read from ``demo/reference_text_<stem>.json`` (keyed by
  the OCR block text; ``demo/reference_text.json`` is still accepted for
  ``Examples/before.jpg``), so layout and lettering can be compared against
  the reference page with identical wording;
* the reference image is resolved automatically: ``Examples/before.jpg`` ->
  ``Examples/after.webp``; ``<n>ja.jpg`` / ``<n>jp.jpg`` -> ``<n>eng.jpg`` next
  to it (the ``more_comparisons/`` pairs).  When ``demo/typeset_reference.py``
  has aligned the reference into the page's pixel grid
  (``demo/reference/<stem>/eng_aligned.png``) that aligned image is used.
* comparison regions: the hand-picked ``REGIONS_BY_STEM["before"]`` for the
  example page, otherwise one region per block (``b<i>_<kind>``) grown around
  the block (``--regions`` accepts names, ``all`` or ``none``).

``--batch`` runs every pair (:func:`all_pairs`: the example page plus every
``more_comparisons`` pair) into ``--out/<stem>/``.

Outputs (in ``--out``, default ``demo/output/dev``), all prefixed by ``--tag``:

* ``<tag>_typeset.png``   the translated page at full resolution
* ``<tag>_erased.png``    the page with the original text erased, no lettering
* ``<tag>_blocks.png``    debug view (source lines green, furigana grey,
  bubble mask blue, layout box red, seed box orange)
* ``<tag>_blocks.json``   per-block record (kind, text, translation, boxes,
  chosen size and lines) consumed by ``demo/typeset_metrics.py``
* ``<tag>_small.jpg``     the translated page scaled to 720 px tall (greyscale)
* ``<tag>_cmp_<region>.jpg``  side-by-side crops  original | ours | reference
* ``<tag>_erasecmp_<region>.jpg``  original | erased only

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
    .venv/Scripts/python.exe demo/typeset_dev.py more_comparisons/3jp.jpg --ref-text --tag base
    .venv/Scripts/python.exe demo/typeset_dev.py --batch --ref-text --tag base --quiet
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glasstranslate.core.pipeline import clean_translation  # noqa: E402
from glasstranslate.core.types import Rect, Segment, StyledSegment, TranslatedSegment  # noqa: E402
compose_mod = importlib.import_module("glasstranslate.render.compose")  # the package re-exports the function under this name
from glasstranslate.render import build_blocks  # noqa: E402
from glasstranslate.render.compose import block_font_path_for, has_cjk, pil_measurer  # noqa: E402
from glasstranslate.render.layout import TextBlock  # noqa: E402

log = logging.getLogger("typeset_dev")

DEMO_DIR = PROJECT_ROOT / "demo"
CACHE_ROOT = DEMO_DIR / "cache"
REFERENCE_ROOT = DEMO_DIR / "reference"
EXAMPLES_DIR = PROJECT_ROOT / "Examples"
COMPARISONS_DIR = PROJECT_ROOT / "more_comparisons"
DEFAULT_IMAGE = EXAMPLES_DIR / "before.jpg"
DEFAULT_REFERENCE = EXAMPLES_DIR / "after.webp"
ALIGNED_REFERENCE_NAME = "eng_aligned.png"  # written by demo/typeset_reference.py
LEGACY_REFERENCE_TEXT = "reference_text.json"  # the pre-2026-09-10 name of reference_text_before.json
_PAIR_SUFFIXES = ("ja", "jp")  # <n>ja.jpg / <n>jp.jpg pair with <n>eng.jpg
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

# Crop regions (x1, y1, x2, y2) used for comparison sheets, per page stem.
# Pages without an entry get one automatic region per block (``block_regions``).
REGIONS_BY_STEM: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    "before": {
        "A_leftmid": (0, 380, 230, 720),  # free text over artwork, no box
        "B_panel2_right": (620, 820, 1096, 1240),  # two dialogue blocks beside the face
        "C_panel3_right": (600, 1260, 1096, 1600),  # dialogue over hatched art
        "D_panel3_left": (0, 1260, 380, 1600),  # dialogue on flat paper
        "E_topleft": (60, 0, 420, 380),  # two joined caption boxes
        "F_topright": (840, 0, 1096, 330),  # rectangular caption box
        "G_caption": (560, 730, 1096, 900),  # chapter title: horizontal caption in the gutter, no box
    }
}
REGIONS = REGIONS_BY_STEM["before"]  # kept for old imports
REGION_PAD_EM = 1.5  # automatic regions reach this many source ems around the block
REGION_MIN_PX = 160  # ...and are at least this big on each side
TILE_W = 240  # width of each tile in a comparison sheet
SHEET_QUALITY = 65  # JPEG quality; sheets are greyscale and must stay well under 60 KB for model viewing


# ------------------------------------------------------------ page resolution
def page_stem(image: Path) -> str:
    """The per-page key: the file name without its suffix (``3jp``, ``before``)."""
    return Path(image).stem


def cache_dir(stem: str) -> Path:
    return CACHE_ROOT / stem


def reference_dir(stem: str) -> Path:
    """Ground-truth directory written by ``demo/typeset_reference.py``."""
    return REFERENCE_ROOT / stem


def reference_text_path(stem: str, root: Optional[Path] = None) -> Path:
    """``demo/reference_text_<stem>.json``; the example page also accepts the
    legacy ``demo/reference_text.json`` when the new name is absent.

    ``root`` redirects the file elsewhere.  The corpus harness passes one, because
    ``demo/`` is tracked and the harness bootstraps one of these per corpus page."""
    base = Path(root) if root is not None else DEMO_DIR
    path = base / f"reference_text_{stem}.json"
    if stem == "before" and not path.exists():
        legacy = DEMO_DIR / LEGACY_REFERENCE_TEXT
        if legacy.exists():
            return legacy
    return path


def load_reference_text(stem: str, root: Optional[Path] = None) -> Dict[str, str]:
    """Reference English per OCR block text; keys starting with ``_`` are
    notes (``_comment``, ``_uncertain``) and never match a block."""
    path = reference_text_path(stem, root)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k and not k.startswith("_") and isinstance(v, str)}


def reference_image_for(image: Path) -> Optional[Path]:
    """The professionally lettered page for ``image``, or None: the example
    page's ``after.webp``, else ``<n>eng.*`` next to a ``<n>ja`` / ``<n>jp`` page."""
    image = Path(image)
    if image.resolve() == DEFAULT_IMAGE.resolve():
        return DEFAULT_REFERENCE if DEFAULT_REFERENCE.exists() else None
    stem = image.stem
    for suffix in _PAIR_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            base = stem[: -len(suffix)]
            for cand in sorted(image.parent.glob(f"{base}eng.*")):
                if cand.is_file() and cand.suffix.lower() in _IMAGE_SUFFIXES:
                    return cand
    return None


def aligned_reference_for(stem: str) -> Optional[Path]:
    """The reference page warped into this page's pixel grid, when derived."""
    path = reference_dir(stem) / ALIGNED_REFERENCE_NAME
    return path if path.exists() else None


def all_pairs() -> List[Path]:
    """Every page that has a reference: the example page first, then the
    ``more_comparisons`` pages in name order."""
    pages: List[Path] = [DEFAULT_IMAGE] if DEFAULT_IMAGE.exists() else []
    if COMPARISONS_DIR.is_dir():
        for p in sorted(COMPARISONS_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES and reference_image_for(p) is not None:
                pages.append(p)
    return pages


def block_kind(block: TextBlock) -> str:
    st = block.style
    return "bubble" if st.in_bubble else ("art" if st.outline else "flat")


def block_regions(blocks: Sequence[TextBlock], shape: Tuple[int, ...]) -> Dict[str, Tuple[int, int, int, int]]:
    """One comparison region per block, ``b<i>_<kind>``: the bubble (or the
    source box plus the erased patch) grown by ``REGION_PAD_EM`` ems, at least
    ``REGION_MIN_PX`` on each side, clipped to the page (``shape`` = (h, w))."""
    h, w = int(shape[0]), int(shape[1])
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for i, b in enumerate(blocks):
        st = b.style
        base = b.bubble if (st.in_bubble and b.bubble is not None) else b.segment.bbox
        if st.clean_rect is not None:
            base = base.union(st.clean_rect)
        em = b.em_px or st.text_height_px
        pad = int(round(REGION_PAD_EM * em))
        x1, y1, x2, y2 = base.x - pad, base.y - pad, base.x2 + pad, base.y2 + pad
        if x2 - x1 < REGION_MIN_PX:
            grow = (REGION_MIN_PX - (x2 - x1) + 1) // 2
            x1, x2 = x1 - grow, x2 + grow
        if y2 - y1 < REGION_MIN_PX:
            grow = (REGION_MIN_PX - (y2 - y1) + 1) // 2
            y1, y2 = y1 - grow, y2 + grow
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            out[f"b{i}_{block_kind(b)}"] = (x1, y1, x2, y2)
    return out


def page_regions(stem: str, blocks: Sequence[TextBlock], shape: Tuple[int, ...]) -> Dict[str, Tuple[int, int, int, int]]:
    fixed = REGIONS_BY_STEM.get(stem)
    return dict(fixed) if fixed else block_regions(blocks, shape)


# ----------------------------------------------------------------- caches
def load_segments(img: np.ndarray, stem: str, refresh: bool, device: str) -> List[Segment]:
    path = cache_dir(stem) / "segments_all.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as fh:
            return pickle.load(fh)
    from glasstranslate.ocr import RapidOCREngine

    ocr = RapidOCREngine(device=device, min_confidence=0.0)
    ocr.warmup()
    segs = ocr.recognize(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(segs, fh)
    log.info("OCR: %d lines cached to %s", len(segs), path)
    return segs


def load_translations(
    texts: Sequence[str], stem: str, src: str, tgt: str, refresh: bool, device: str, models: Path
) -> Dict[str, str]:
    path = cache_dir(stem) / f"translations_{src}_{tgt}.json"
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
        path.parent.mkdir(parents=True, exist_ok=True)
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
    A reference of the page's own size (the aligned reference) is cropped
    directly; one of another size is mapped by independent x/y scale factors
    (close enough for a visual comparison)."""
    x1, y1, x2, y2 = region
    a = orig.crop(region)
    b = ours.crop(region)
    tiles = [a, b]
    if ref is not None:
        if ref.size == orig.size:
            c = ref.crop(region)
        else:
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


def _rect_list(r: Optional[Rect]) -> Optional[List[int]]:
    return None if r is None else [int(r.x), int(r.y), int(r.w), int(r.h)]


def block_record(index: int, block: TextBlock, seg: TranslatedSegment, text: str, ts, from_reference: bool) -> Dict[str, Any]:
    """The per-block JSON record of ``<tag>_blocks.json`` (``ts`` is the
    :class:`Typeset` chosen for the block, or None when it was not lettered)."""
    st = block.style
    bb = ts.bbox if ts is not None and ts.lines else None
    return {
        "index": index,
        "kind": block_kind(block),
        "in_bubble": bool(st.in_bubble),
        "outline": bool(st.outline),
        "vertical": bool(st.vertical),
        "text": block.segment.text,
        "translation": text,
        "from_reference": bool(from_reference),
        "untranslated": bool(seg.untranslated),
        "glyph_px": float(st.text_height_px),
        "em_px": float(block.em_px),
        "max_font_px": float(st.max_font_px) if st.max_font_px else None,
        "fg": list(st.fg),
        "bg": list(st.bg),
        "source_bbox": _rect_list(block.segment.bbox),
        "layout_box": _rect_list(st.layout_box),
        "bubble": _rect_list(block.bubble),
        "clean_rect": _rect_list(st.clean_rect),
        "size": float(ts.size) if ts is not None and ts.lines else None,
        "lines": [l.text for l in ts.lines] if ts is not None else [],
        "line_count": len(ts.lines) if ts is not None else 0,
        "bbox": _rect_list(bb),
        "fitted": bool(ts.fitted) if ts is not None else None,
    }


# --------------------------------------------------------------- quality renderer
def run_quality(args: argparse.Namespace, img: np.ndarray, blocks: Sequence[TextBlock]) -> int:
    """Run the sidecar synchronously over the page's panel jobs and swap the patches.

    ``--quality`` uses the sidecar venv found by ``render.quality.find_sidecar_python``;
    ``--quality-fake`` runs the protocol's numpy stand-ins in the *main* venv, so the
    whole path can be exercised without a GPU.  Returns the number of blocks upgraded;
    a missing or failing sidecar only prints a line (the quick fill stays).
    """
    from types import SimpleNamespace

    from glasstranslate.render import quality as Q

    if args.quality_fake:
        python: Optional[Path] = Path(sys.executable)
    else:
        python = Q.find_sidecar_python(SimpleNamespace(quality_sidecar_python=""))
        if python is None:
            print("quality: no sidecar interpreter (run renderer\\install.bat)", file=sys.stderr)
            return 0
    jobs = Q.panel_jobs(img, list(blocks))
    if not jobs:
        print("quality: no free text over artwork on this page")
        return 0
    params = parse_quality_params(args.quality_params)
    client = getattr(args, "_quality_client", None)
    if client is None:
        # One sidecar for the whole run (a batch pays the cold start once); main() shuts it down.
        client = Q.QualityClient(python, args.models, fake=bool(args.quality_fake), log_path=None)
        args._quality_client = client
    upgraded = 0
    try:
        client.start()
        waited = time.perf_counter()
        health = client.wait_warm()  # the first job must not race the warm-up against its own timeout
        waited = time.perf_counter() - waited
        print(f"quality: sidecar {health.get('mode', '?')} on {health.get('device', '?')}, {len(jobs)} panel(s)"
              + (f", params {params}" if params else "") + (f", warm after {waited:.0f} s" if waited > 1 else ""))
        for job in jobs:
            started = time.perf_counter()
            crop = client.inpaint(job.image, job.mask, params or None)
            elapsed = 1000.0 * (time.perf_counter() - started)
            stages = ", ".join(f"{k} {v:.0f}" for k, v in sorted(client.last_timings.items()))
            print(f"quality: panel {job.rect.w}x{job.rect.h} at {job.rect.x},{job.rect.y}"
                  f" in {elapsed:.0f} ms [{stages}]")
            for block in blocks:
                if Q.block_key(block.segment.text, block.style) not in job.members:
                    continue
                block.style.clean_patch = Q.composite_block_patch(img, job, crop, block.style)
                block.style.clean_patch_serial += 1
                upgraded += 1
    except Q.QualityUnavailable as exc:
        print(f"quality: unavailable ({exc}); using the quick fill", file=sys.stderr)
        if getattr(exc, "transport", True):
            # The process or its socket is gone: drop it (the next page pays a cold start).
            # A job the running sidecar refused keeps the warm sidecar for the next page.
            client.shutdown()
            args._quality_client = None
    return upgraded


def parse_quality_params(text: str) -> Dict[str, Any]:
    """``strength=0.4,steps=24,controlnet_scale=0.8,guidance=4,lama=true`` -> params dict
    (numbers become numbers, ``true`` / ``false`` booleans; unknown keys are rejected by the sidecar)."""
    out: Dict[str, Any] = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        elif key in ("steps", "seed", "feather_px"):
            out[key] = int(float(value))
        elif key in ("prompt", "negative_prompt"):
            out[key] = value
        else:
            out[key] = float(value)
    return out


# ------------------------------------------------------------------- pages
def run_page(args: argparse.Namespace, image: Path, out_dir: Path) -> int:
    stem = page_stem(image)
    img = cv2.imread(str(image))
    if img is None:
        print(f"cannot read {image}", file=sys.stderr)
        return 2

    all_segments = load_segments(img, stem, args.ocr, args.device)
    segments = [s for s in all_segments if s.confidence >= args.min_confidence]

    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    blocks = build_blocks(img, segments, all_segments, timings=timings)
    t1 = time.perf_counter()
    if args.quality or args.quality_fake:
        run_quality(args, img, blocks)

    texts = [b.segment.text for b in blocks]
    translations = load_translations(texts, stem, args.src, args.tgt, args.translate, args.device, args.models)
    reference = (load_reference_text(stem, getattr(args, "reference_text_root", None))
                 if args.ref_text else {})

    translated: List[TranslatedSegment] = []
    from_ref: List[bool] = []
    for b in blocks:
        text = b.segment.text
        ref = reference.get(text)
        tr = ref or translations.get(text) or text
        from_ref.append(ref is not None)
        styled = StyledSegment(segment=b.segment, style=b.style, src_lang=args.src)
        translated.append(TranslatedSegment(styled=styled, translation=tr, tgt_lang=args.tgt))

    t4 = time.perf_counter()
    out = compose_mod.compose(img, translated, hide_original=True, uppercase=not args.no_upper)
    t5 = time.perf_counter()
    erased_only = [TranslatedSegment(s.styled, "", s.tgt_lang) for s in translated if not s.untranslated]
    erased = compose_mod.compose(img, erased_only, hide_original=True)

    # Per-block report: what the typesetter chose.
    records: List[Dict[str, Any]] = []
    print(f"== {image} ({img.shape[1]}x{img.shape[0]})")
    print(f"{'#':>2} {'kind':6} {'glyph':>5} {'ceil':>5} {'size':>5} {'ln':>2} {'box':>18}  text")
    for i, (b, seg) in enumerate(zip(blocks, translated)):
        st = b.style
        kind = block_kind(b)
        text = seg.translation.strip()
        if not args.no_upper and not has_cjk(text):
            text = text.upper()
        size_s, n_s, box_s = "-", "-", "-"
        ts = None
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
                ts = None
        ceil = f"{st.max_font_px:5.1f}" if st.max_font_px else "-"
        mark = "*" if from_ref[i] else " "
        print(f"{i:>2} {kind:6} {st.text_height_px:5.1f} {ceil:>5} {size_s:>6} {n_s:>2} {box_s:>18} {mark}{text[:60]!r}")
        records.append(block_record(i, b, seg, text, ts, from_ref[i]))
    print(
        f"{len(segments)} lines -> {len(blocks)} blocks | layout {1000*(t1-t0):.0f} ms"
        f" (group {timings.get('group', 0):.0f}, erase {timings.get('erase', 0):.0f},"
        f" place-prep {timings.get('place', 0):.0f}), render {1000*(t5-t4):.0f} ms"
        + (f" | {sum(from_ref)}/{len(blocks)} blocks with reference text" if args.ref_text else "")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag
    cv2.imwrite(str(out_dir / f"{tag}_typeset.png"), out)
    cv2.imwrite(str(out_dir / f"{tag}_erased.png"), erased)
    cv2.imwrite(str(out_dir / f"{tag}_blocks.png"), draw_blocks(img, blocks))
    small(out).save(out_dir / f"{tag}_small.jpg", quality=SHEET_QUALITY)
    (out_dir / f"{tag}_blocks.json").write_text(
        json.dumps(
            {
                "image": str(image),
                "stem": stem,
                "tag": tag,
                "ref_text": bool(args.ref_text),
                "timings_ms": {"layout": 1000 * (t1 - t0), "render": 1000 * (t5 - t4), **{k: float(v) for k, v in timings.items()}},
                "blocks": records,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    if args.regions != "none":
        regions = page_regions(stem, blocks, img.shape)
        names = list(regions) if args.regions in ("all", "auto") else [r.strip() for r in args.regions.split(",") if r.strip()]
        orig_pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        ours_pil = Image.fromarray(np.ascontiguousarray(out[:, :, ::-1]))
        erased_pil = Image.fromarray(np.ascontiguousarray(erased[:, :, ::-1]))
        ref_pil = None
        if not args.no_ref_image:
            ref_path = aligned_reference_for(stem) or reference_image_for(image)
            if ref_path is not None:
                ref_pil = Image.open(ref_path).convert("RGB")
        for name in names:
            region = regions.get(name)
            if region is None:
                print(f"unknown region {name!r}; known: {', '.join(regions)}", file=sys.stderr)
                continue
            comparison_sheet(orig_pil, ours_pil, ref_pil, region).save(out_dir / f"{tag}_cmp_{name}.jpg", quality=SHEET_QUALITY)
            comparison_sheet(orig_pil, erased_pil, None, region).save(out_dir / f"{tag}_erasecmp_{name}.jpg", quality=SHEET_QUALITY)
    print(f"wrote {out_dir / (tag + '_typeset.png')} (+ _erased, _blocks.png/.json, _small, _cmp_*, _erasecmp_*)")
    return 0


# ------------------------------------------------------------------- main
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--batch", action="store_true", help="run every reference pair (all_pairs) into --out/<stem>/")
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
    parser.add_argument("--quality", action="store_true",
                        help="run the quality-renderer sidecar over the page's panels before compose")
    parser.add_argument("--quality-fake", action="store_true",
                        help="as --quality, but with the sidecar's numpy stand-ins in the main venv (no GPU)")
    parser.add_argument("--quality-params", default="",
                        help="sidecar params, e.g. strength=0.4,steps=24,controlnet_scale=0.8,guidance=4.0,sdxl=true")
    parser.add_argument("--new-erase", action="store_true", help="no-op (erase.apply runs inside build_blocks)")
    parser.add_argument("--new-place", action="store_true", help="no-op (place is the default geometry)")
    parser.add_argument("--regions", default="all", help="comma-separated region names, or 'all'/'auto'/'none'")
    parser.add_argument("--no-ref-image", action="store_true", help="omit the reference tile from the sheets")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    if args.new_erase or args.new_place:
        print("note: --new-erase/--new-place are no-ops: build_blocks runs erase.apply and place.prepare, "
              "and compose.typeset_block is the placement geometry")

    try:
        if args.batch:
            pages = all_pairs()
            if not pages:
                print("no reference pairs found", file=sys.stderr)
                return 2
            worst = 0
            for image in pages:
                worst = max(worst, run_page(args, image, args.out / page_stem(image)))
            return worst
        return run_page(args, args.image, args.out)
    finally:
        client = getattr(args, "_quality_client", None)
        if client is not None:
            client.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
