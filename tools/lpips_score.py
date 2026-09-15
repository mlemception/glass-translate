"""LPIPS of a typeset render against the aligned reference, per block (sidecar venv).

The main venv has no torch, so ``demo/typeset_metrics.py`` computes SSIM / MAE and this
script adds LPIPS (Zhang et al., AlexNet backbone, the ``lpips`` package, BSD-2) from the
sidecar venv, reading only files: the erased render ``<dir>/<stem>/<tag>_erased.png``, the
ground truth under ``demo/reference/<stem>/`` (``eng_aligned.png``, ``blocks.json`` for the
block windows and ``english_ink.png`` to blank the reference's lettering in both crops) and
the metrics JSON for the block kinds.  Per block the crop is the block's window grown by
``PAD`` px; the English lettering is painted with the local paper colour on both sides so the
score reflects the erased artwork, not the missing letters.  Lower is better.

Usage (from the project root)::

    renderer/.venv/Scripts/python tools/lpips_score.py --batch --tag base --tag q_s40 --tag q_lama

Writes ``demo/output/metrics/<stem>_<tag>_lpips.json`` and prints a per-tag table over the
free-text (``art``) blocks.  The first run downloads the AlexNet weights through torchvision
(dev machine only; the sidecar itself never touches the network).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "demo" / "output" / "metrics"
REFERENCE_ROOT = PROJECT_ROOT / "demo" / "reference"
RENDER_ROOT = PROJECT_ROOT / "demo" / "output" / "dev"
PAD = 16  # px around the block window
MIN_SIDE = 64  # crops are grown to at least this (AlexNet features need a few pooling levels)


def _pairs(batch: bool, image: Optional[Path]) -> List[str]:
    if batch:
        stems = [p.name for p in sorted(REFERENCE_ROOT.iterdir()) if (p / "blocks.json").exists()]
        return ["before"] + [s for s in stems if s != "before"] if "before" in stems else stems
    return [Path(image).stem] if image else ["before"]


def _blank_lettering(gray: np.ndarray, ink: np.ndarray, paper: int) -> np.ndarray:
    out = gray.copy()
    out[ink] = paper
    return out


def _crop(arr: np.ndarray, window: List[int]) -> np.ndarray:
    x, y, w, h = window
    hh, ww = arr.shape[:2]
    if w < MIN_SIDE:
        x, w = x - (MIN_SIDE - w) // 2, MIN_SIDE
    if h < MIN_SIDE:
        y, h = y - (MIN_SIDE - h) // 2, MIN_SIDE
    x0, y0 = max(0, x - PAD), max(0, y - PAD)
    x1, y1 = min(ww, x + w + PAD), min(hh, y + h + PAD)
    return arr[y0:y1, x0:x1]


def score_page(model, stem: str, tag: str, device: str, *,
               reference_root: Path = REFERENCE_ROOT,
               render_root: Path = RENDER_ROOT,
               metrics_dir: Path = METRICS_DIR) -> Optional[Dict]:
    """The roots are parameters so the corpus harness can point them at
    ``demo/output/corpus/`` instead of the single-page ``demo/reference`` tree."""
    import torch

    ref_dir = reference_root / stem
    render = render_root / stem / f"{tag}_erased.png"
    if not render.exists() or not (ref_dir / "blocks.json").exists():
        print(f"{stem}/{tag}: missing render or ground truth", file=sys.stderr)
        return None
    ours = cv2.imread(str(render), cv2.IMREAD_GRAYSCALE)
    eng = cv2.imread(str(ref_dir / "eng_aligned.png"), cv2.IMREAD_GRAYSCALE)
    ink = cv2.imread(str(ref_dir / "english_ink.png"), cv2.IMREAD_GRAYSCALE) > 0
    ink = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    records = json.loads((ref_dir / "blocks.json").read_text(encoding="utf-8"))["blocks"]
    rows = []
    for rec in records:
        x, y, w, h = rec["window"]
        paper = int(np.median(eng[y:y + h, x:x + w])) if w > 0 and h > 0 else 255
        a = _crop(_blank_lettering(ours, ink, paper), rec["window"])
        b = _crop(_blank_lettering(eng, ink, paper), rec["window"])
        if a.size == 0 or a.shape != b.shape:
            continue
        ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0)[None, None].repeat(1, 3, 1, 1).to(device)
        tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0)[None, None].repeat(1, 3, 1, 1).to(device)
        with torch.no_grad():
            value = float(model(ta, tb).item())
        rows.append({"index": rec["index"], "kind": rec["kind"], "text": rec["text"], "lpips": value})
    art = [r["lpips"] for r in rows if r["kind"] == "art"]
    result = {"stem": stem, "tag": tag, "blocks": rows, "mean": float(np.mean([r["lpips"] for r in rows])) if rows else None,
              "mean_art": float(np.mean(art)) if art else None}
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{stem}_{tag}_lpips.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--tag", action="append", default=[], help="render tag(s) to score (repeatable)")
    parser.add_argument("--net", default="alex", choices=("alex", "vgg"))
    args = parser.parse_args(argv)
    import lpips
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = lpips.LPIPS(net=args.net, verbose=False).to(device).eval()
    stems = _pairs(args.batch, args.image)
    tags = args.tag or ["base"]
    print(f"{'tag':10} " + " ".join(f"{s:>8}" for s in stems) + f" {'all art':>8}")
    for tag in tags:
        cells = []
        all_art: List[float] = []
        for stem in stems:
            result = score_page(model, stem, tag, device)
            if result is None:
                cells.append(f"{'-':>8}")
                continue
            art = [r["lpips"] for r in result["blocks"] if r["kind"] == "art"]
            all_art.extend(art)
            cells.append(f"{(np.mean(art) if art else float('nan')):8.3f}")
        print(f"{tag:10} " + " ".join(cells) + f" {np.mean(all_art) if all_art else float('nan'):8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
