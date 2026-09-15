"""Art-preservation LPIPS for corpus pages - runs in the SIDECAR venv only.

    renderer\\.venv\\Scripts\\python.exe demo/corpus_lpips.py --tag base

The app and the exe stay torch-free, so this is the one corpus module that imports torch
and it is never imported by the others: ``corpus_metrics.page_extras`` only reads the JSON
this writes.  Run it after ``demo/corpus_eval.py`` has produced the renders, then re-run
``corpus_eval.py`` to fold ``c_art`` into the headline.

Two numbers per page, both comparing OUR erased render against the aligned official page
with the English lettering painted out on both sides (so the score reflects the artwork,
not the missing letters):

* ``mean_art`` - LPIPS over the free-text block windows, where the erase actually had to
  reconstruct artwork.
* ``floor`` - LPIPS over windows containing NO text at all.  Two scans of the same page
  differ in JPEG quality, screentone moire and gamma, and LPIPS is sensitive to all three;
  without subtracting this per-page floor the metric reports damage we did not cause and is
  not comparable across volumes.  ``corpus_metrics.lpips_excess`` does the subtraction.

Nothing here reads the corpus: it works from the derived files under demo/output/corpus/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import corpus_metrics as CM  # noqa: E402

OUT_ROOT = DEMO_DIR / "output" / "corpus"
REFERENCE_DIR = OUT_ROOT / "reference"
RENDER_DIR = OUT_ROOT / "renders"

PAD = 16  # px around a block window, as in tools/lpips_score.py
MIN_SIDE = 64  # AlexNet features need a few pooling levels
INK_DILATE = 5


def _read_gray(path: Path) -> Optional[np.ndarray]:
    """Bytes through Python, never a ``str`` path into OpenCV - see corpus_index."""
    path = Path(path)
    if not path.exists():
        return None
    buf = np.frombuffer(path.read_bytes(), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if buf.size else None


def blank_lettering(gray: np.ndarray, ink: np.ndarray, paper: int) -> np.ndarray:
    out = gray.copy()
    out[ink] = paper
    return out


def crop(arr: np.ndarray, window: Sequence[int]) -> np.ndarray:
    x, y, w, h = (int(v) for v in window)
    height, width = arr.shape[:2]
    if w < MIN_SIDE:
        x, w = x - (MIN_SIDE - w) // 2, MIN_SIDE
    if h < MIN_SIDE:
        y, h = y - (MIN_SIDE - h) // 2, MIN_SIDE
    x0, y0 = max(0, x - PAD), max(0, y - PAD)
    x1, y1 = min(width, x + w + PAD), min(height, y + h + PAD)
    return arr[y0:y1, x0:x1]


def _score(model, device: str, a: np.ndarray, b: np.ndarray) -> Optional[float]:
    import torch

    if a.size == 0 or a.shape != b.shape or min(a.shape[:2]) < 8:
        return None
    ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0)[None, None].repeat(1, 3, 1, 1).to(device)
    tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0)[None, None].repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        return float(model(ta, tb).item())


def lpips_for_page(model, device: str, page_id: str, tag: str) -> Optional[Dict[str, Any]]:
    ref_dir = REFERENCE_DIR / page_id
    render_dir = RENDER_DIR / page_id
    render = render_dir / f"{tag}_erased.png"
    blocks_json = ref_dir / "blocks.json"
    if not render.exists() or not blocks_json.exists():
        return None

    ours = _read_gray(render)
    eng = _read_gray(ref_dir / "eng_aligned.png")
    ink_img = _read_gray(ref_dir / "english_ink.png")
    if ours is None or eng is None or ink_img is None or ours.shape != eng.shape:
        return None
    ink = cv2.dilate((ink_img > 0).astype(np.uint8),
                     np.ones((INK_DILATE, INK_DILATE), np.uint8)).astype(bool)
    records: List[Dict[str, Any]] = json.loads(
        blocks_json.read_text(encoding="utf-8"))["blocks"]

    rows: List[Dict[str, Any]] = []
    for record in records:
        window = record.get("window")
        if not window:
            continue
        x, y, w, h = (int(v) for v in window)
        paper = int(np.median(eng[y:y + h, x:x + w])) if w > 0 and h > 0 else 255
        value = _score(model, device,
                       crop(blank_lettering(ours, ink, paper), window),
                       crop(blank_lettering(eng, ink, paper), window))
        if value is not None:
            rows.append({"index": record.get("index"), "kind": record.get("kind"),
                         "text": record.get("text"), "lpips": value})

    # The floor: the same comparison where there is no text at all.
    boxes = [r["window"] for r in records if r.get("window")]
    paper = int(np.median(eng))
    floors: List[float] = []
    for window in CM.text_free_windows(ours.shape[:2], boxes):
        value = _score(model, device,
                       crop(blank_lettering(ours, ink, paper), window),
                       crop(blank_lettering(eng, ink, paper), window))
        if value is not None:
            floors.append(value)

    art = [r["lpips"] for r in rows if r["kind"] == "art"]
    result = {
        "stem": page_id,
        "tag": tag,
        "blocks": rows,
        "mean": float(np.mean([r["lpips"] for r in rows])) if rows else None,
        "mean_art": float(np.mean(art)) if art else None,
        "floor": float(np.median(floors)) if floors else None,
        "floor_windows": len(floors),
    }
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / f"{tag}_lpips.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.4f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="base")
    parser.add_argument("--net", default="alex", choices=("alex", "vgg"))
    parser.add_argument("--page", action="append", default=[],
                        help="page id (repeatable); default every rendered page")
    args = parser.parse_args(argv)

    try:
        import lpips
        import torch
    except ImportError:
        print("demo/corpus_lpips.py needs the sidecar venv: "
              "renderer\\.venv\\Scripts\\python.exe demo/corpus_lpips.py", file=sys.stderr)
        return 2

    pages = args.page or sorted(
        p.parent.name for p in RENDER_DIR.glob(f"*/{args.tag}_erased.png"))
    if not pages:
        print(f"no renders for tag '{args.tag}' under {RENDER_DIR}", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net=args.net, verbose=False).to(device).eval()
    print(f"{len(pages)} page(s) on {device}")

    excess: List[float] = []
    for i, page_id in enumerate(pages, 1):
        result = lpips_for_page(model, device, page_id, args.tag)
        if result is None:
            print(f"[{i}/{len(pages)}] {page_id}: skipped")
            continue
        gap = CM.lpips_excess(result["mean_art"], result["floor"])
        if gap is not None:
            excess.append(gap)
        print(f"[{i}/{len(pages)}] {page_id}: art={_fmt(result['mean_art'])} "
              f"floor={_fmt(result['floor'])} excess={_fmt(gap)}", flush=True)

    if excess:
        print(f"\nmean lpips_excess over {len(excess)} page(s): {np.mean(excess):.4f}")
    print("re-run demo/corpus_eval.py to fold c_art into the headline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
