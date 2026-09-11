"""Measure and eyeball the quality renderer's generative stages.

Run it with the **sidecar** venv (the main app never imports torch)::

    renderer\\.venv\\Scripts\\python tools/quality_probe.py \\
        --image more_comparisons/2ja.jpg --crop 150,100,250,300 \\
        --mask-boxes "197,188,100,111" --strengths 0.3,0.4,0.5 --cn 0.6,0.8,1.0

It loads a page, crops one panel, builds an erase mask from page-coordinate
boxes (or a mask PNG), runs LaMa once and then SDXL over a strength x
ControlNet-scale grid, composites every result through
:func:`glassrenderer.compositing.composite`, and writes to
``demo/output/quality/``:

* ``<tag>-lama.png`` and ``<tag>-s<strength>-c<scale>.png`` — full-colour results;
* ``<tag>-sheet.jpg`` — a greyscale contact sheet, under 1000 px and 100 KB,
  which is the only image a reviewer should open;
* ``<tag>-probe.json`` — cold start, per-stage timings and VRAM.

The tool never displays an image; it only writes them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "renderer") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "renderer"))

DEFAULT_IMAGE = "more_comparisons/2ja.jpg"
DEFAULT_OUT = "demo/output/quality"
SHEET_MAX_PX = 1000
SHEET_MAX_BYTES = 100_000
FEATHER_PX = 2
LABEL_H = 16


# ------------------------------------------------------------------ inputs
@dataclass
class Box:
    """An axis-aligned box in page coordinates."""

    x: int
    y: int
    w: int
    h: int

    def shifted(self, dx: int, dy: int) -> "Box":
        return Box(self.x - dx, self.y - dy, self.w, self.h)


def parse_box(text: str) -> Box:
    """``"x,y,w,h"`` -> :class:`Box` (raises ``ValueError`` on anything else)."""
    parts = [int(p) for p in text.replace(" ", "").split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError(f"expected x,y,w,h with positive w/h, got {text!r}")
    return Box(*parts)


def parse_boxes(text: str) -> List[Box]:
    """``"x,y,w,h;x,y,w,h"`` -> boxes."""
    return [parse_box(chunk) for chunk in text.split(";") if chunk.strip()]


def parse_floats(text: str) -> List[float]:
    return [float(p) for p in text.replace(" ", "").split(",") if p]


def load_rgb(path: Path) -> np.ndarray:
    """Read ``path`` as uint8 RGB (never shown, only measured)."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_mask(shape: Tuple[int, int], boxes: Sequence[Box], dilate_px: int) -> np.ndarray:
    """Erase mask (uint8 HxW, 255 = fill) from crop-relative ``boxes``."""
    mask = np.zeros(shape, np.uint8)
    height, width = shape
    for box in boxes:
        x0, y0 = max(0, box.x), max(0, box.y)
        x1, y1 = min(width, box.x + box.w), min(height, box.y + box.h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    if dilate_px > 0:
        size = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    return mask


# ------------------------------------------------------------ contact sheet
def _label(cell: np.ndarray, text: str) -> np.ndarray:
    """Stack a caption strip under a greyscale cell."""
    strip = np.full((LABEL_H, cell.shape[1]), 255, np.uint8)
    cv2.putText(strip, text[:32], (2, LABEL_H - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, 0, 1, cv2.LINE_AA)
    return np.vstack([cell, strip])


def _grid(cells: List[np.ndarray], columns: int) -> np.ndarray:
    """Tile equally sized cells into ``columns`` columns, padding the last row."""
    blank = np.full_like(cells[0], 255)
    padded = cells + [blank] * ((-len(cells)) % columns)
    rows = [np.hstack(padded[i : i + columns]) for i in range(0, len(padded), columns)]
    return np.vstack(rows)


def contact_sheet(named: List[Tuple[str, np.ndarray]], out_path: Path) -> int:
    """Write a greyscale JPEG sheet under 1000 px / 100 KB; return its size."""
    columns = min(4, len(named))
    rows = (len(named) + columns - 1) // columns
    cell_w = max(64, SHEET_MAX_PX // columns)
    cell_h = max(64, int(round(cell_w * named[0][1].shape[0] / named[0][1].shape[1])))
    if (cell_h + LABEL_H) * rows > SHEET_MAX_PX:
        cell_h = max(48, SHEET_MAX_PX // rows - LABEL_H)
        cell_w = max(48, int(round(cell_h * named[0][1].shape[1] / named[0][1].shape[0])))
    cells = [
        _label(
            cv2.resize(
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (cell_w, cell_h), interpolation=cv2.INTER_AREA
            ),
            name,
        )
        for name, img in named
    ]
    sheet = _grid(cells, columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_within_budget(sheet, out_path)


def _write_within_budget(sheet: np.ndarray, out_path: Path) -> int:
    """Write ``sheet`` as JPEG, dropping quality then size until it fits.

    Halftone pages barely compress, so quality alone is not enough: a reviewer
    must be able to open the file, which means <= 1000 px and <= 100 KB.
    """
    current = sheet
    for _ in range(6):
        for quality in (85, 70, 55, 40, 30, 20):
            cv2.imwrite(str(out_path), current, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if out_path.stat().st_size <= SHEET_MAX_BYTES:
                return out_path.stat().st_size
        height, width = current.shape[:2]
        current = cv2.resize(
            current, (int(width * 0.8), int(height * 0.8)), interpolation=cv2.INTER_AREA
        )
    return out_path.stat().st_size


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# ----------------------------------------------------------------- the run
@dataclass
class ProbeResult:
    """Everything the probe measured, serialised next to the PNGs."""

    image: str
    crop: List[int]
    size: List[int]
    cold: Dict[str, float] = field(default_factory=dict)
    lama_ms: List[float] = field(default_factory=list)
    grid: List[Dict[str, Any]] = field(default_factory=list)
    vram: Dict[str, float] = field(default_factory=dict)


def _vram_gb() -> Dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "device_used_gb": round((total - free) / 1e9, 2),
        "device_total_gb": round(total / 1e9, 2),
    }


def _load_stages(models_dir: str, device: str) -> Tuple[Any, Any, Dict[str, float]]:
    """Build and load both stages; returns them with the cold-start breakdown."""
    started = time.perf_counter()
    from glassrenderer.stages import build_stages

    imported = time.perf_counter()
    lama, sdxl = build_stages(models_dir, device, fake=False)
    lama.load()
    lama_done = time.perf_counter()
    sdxl.load()
    done = time.perf_counter()
    cold = {
        "import_s": round(imported - started, 2),
        "lama_load_s": round(lama_done - imported, 2),
        "sdxl_load_s": round(done - lama_done, 2),
        "total_s": round(done - started, 2),
    }
    return lama, sdxl, cold


def _run_grid(
    sdxl: Any,
    base: np.ndarray,
    source: np.ndarray,
    mask: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
    sheet: List[Tuple[str, np.ndarray]],
    result: ProbeResult,
) -> None:
    """Sweep strength x ControlNet scale, saving and timing every cell."""
    from glassrenderer.compositing import composite

    for strength in parse_floats(args.strengths):
        for scale in parse_floats(args.cn):
            params = {
                "strength": strength,
                "steps": args.steps,
                "controlnet_scale": scale,
                "guidance": args.guidance,
                "seed": args.seed,
            }
            rendered = sdxl.run(base, mask, params)
            out = composite(source, rendered, mask, FEATHER_PX)
            name = f"s{strength:g}-c{scale:g}"
            save_rgb(out_dir / f"{args.tag}-{name}.png", out)
            sheet.append((name, out))
            result.grid.append({"name": name, "strength": strength, "cn": scale, "ms": round(sdxl.last_ms, 1)})
            print(f"  sdxl {name}: {sdxl.last_ms:8.1f} ms", flush=True)


def probe(args: argparse.Namespace) -> ProbeResult:
    """Run the whole measurement and write every artefact."""
    from glassrenderer.compositing import composite

    image_path = PROJECT_ROOT / args.image
    page = load_rgb(image_path)
    crop = parse_box(args.crop) if args.crop else Box(0, 0, page.shape[1], page.shape[0])
    source = np.ascontiguousarray(page[crop.y : crop.y + crop.h, crop.x : crop.x + crop.w])
    mask = _resolve_mask(args, crop, source.shape[:2])

    lama, sdxl, cold = _load_stages(args.models_dir, args.device)
    print(f"cold start: {cold}", flush=True)
    result = ProbeResult(image=args.image, crop=[crop.x, crop.y, crop.w, crop.h],
                         size=[source.shape[1], source.shape[0]], cold=cold)

    out_dir = PROJECT_ROOT / args.out
    filled = source
    for run in range(args.repeats):
        filled = lama.run(source, mask, {})
        result.lama_ms.append(round(lama.last_ms, 1))
        print(f"  lama run {run}: {lama.last_ms:8.1f} ms", flush=True)
    lama_out = composite(source, filled, mask, FEATHER_PX)
    save_rgb(out_dir / f"{args.tag}-lama.png", lama_out)

    sheet: List[Tuple[str, np.ndarray]] = [("source", source), ("lama", lama_out)]
    _run_grid(sdxl, filled, source, mask, args, out_dir, sheet, result)

    result.vram = _vram_gb()
    sheet_path = out_dir / f"{args.tag}-sheet.jpg"
    size = contact_sheet(sheet, sheet_path)
    print(f"sheet: {sheet_path} ({size} bytes)", flush=True)
    print(f"vram: {result.vram}", flush=True)
    (out_dir / f"{args.tag}-probe.json").write_text(
        json.dumps(result.__dict__, indent=2), encoding="utf-8"
    )
    return result


def _resolve_mask(args: argparse.Namespace, crop: Box, shape: Tuple[int, int]) -> np.ndarray:
    """The erase mask for the crop, from ``--mask-png`` or ``--mask-boxes``."""
    if args.mask_png:
        raw = cv2.imread(str(PROJECT_ROOT / args.mask_png), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"cannot read {args.mask_png}")
        if raw.shape != shape:
            raw = raw[crop.y : crop.y + crop.h, crop.x : crop.x + crop.w]
        return (raw >= 128).astype(np.uint8) * 255
    if not args.mask_boxes:
        raise SystemExit("one of --mask-boxes / --mask-png is required")
    boxes = [b.shifted(crop.x, crop.y) for b in parse_boxes(args.mask_boxes)]
    return build_mask(shape, boxes, args.dilate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="page, relative to the project root")
    parser.add_argument("--crop", default="", help="panel as x,y,w,h in page coordinates")
    parser.add_argument("--mask-boxes", default="", help='erase boxes "x,y,w,h;..." in page coordinates')
    parser.add_argument("--mask-png", default="", help="greyscale mask PNG instead of --mask-boxes")
    parser.add_argument("--dilate", type=int, default=2, help="mask dilation in px (default 2)")
    parser.add_argument("--strengths", default="0.3,0.4,0.5")
    parser.add_argument("--cn", default="0.6,0.8,1.0", help="ControlNet conditioning scales")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3, help="LaMa runs (the first is cold)")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--tag", default="probe", help="filename prefix under --out")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    probe(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
