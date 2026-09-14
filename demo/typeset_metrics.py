"""Score a typeset render against the ground truth of ``demo/typeset_reference.py``.

Inputs are a render directory written by ``demo/typeset_dev.py``
(``<tag>_erased.png``, ``<tag>_typeset.png``, ``<tag>_blocks.json``) and
``demo/reference/<stem>/`` (``eng_aligned.png``, ``erase_gt.png``,
``kept_art.png``, ``english_ink.png``, ``blocks.json``).  Every block is scored
inside the window the reference derivation used:

* **erase_iou / erase_recall / erase_precision** - the ink we removed (page
  ink that is no longer ink in our erased render) against the letterer's erase
  mask; recall < 1 means Japanese residue, precision < 1 means we destroyed
  something the letterer kept;
* **art_kept** - the share of the art ink the letterer kept that survives in
  our erased render;
* **ssim / mae** - structural similarity (Gaussian window, numpy/OpenCV
  implementation) and mean absolute grey difference between our erased render
  and the aligned reference, in the erased neighbourhood (the letterer's erase
  mask dilated by a few pixels, the English lettering excluded);
* lettering (bubble blocks, and free text when the reference has lettering
  there): **cap_ratio** (our cap height / the reference's), **inset_em** ours
  and the reference's (distance from the lettering to the bubble outline, in
  source ems), **centre_dx/dy_em** (offset of the lettering from the bubble
  centroid), **lines** ours / the reference's, **overflow_px** (our lettering
  ink outside the bubble interior) and **collision_px** (our lettering ink on
  another block's text / bubble or on a panel border line);
* geometry of the block against the balloon it belongs in - **containment**
  (the fraction of our glyph pixels outside the interior eroded by
  ``INTERIOR_MARGIN_EM`` source ems; 0 is the only passing value, and the page
  row also carries the worst single block), **centre_offset_em** (our ink's
  centroid to the interior's inscribed-circle centre, in source ems),
  **leftover_px** (source ink still inside the interior after the erase pass,
  the outline band excluded - this is what catches furigana the eraser never
  saw) and **blocks_per_bubble** (blocks lettered into this one's balloon
  interior, once :mod:`render.bubbles` has cut a joined component into one
  interior per block; more than one means two blocks letter into each other's
  half).  All four are reported only for bubble blocks: free text has no
  interior, scores ``-`` and is left out of the bubble means.

Tables (markdown) and JSON go to ``demo/output/metrics/<stem>_<tag>.{md,json}``;
``--batch`` adds ``<tag>_summary.md`` with one row per page.  LPIPS is
deliberately not required: the main venv has no torch.  ``--lpips`` computes it
when the ``lpips`` package is importable (the sidecar venv).

Usage::

    set PYTHONUTF8=1
    .venv/Scripts/python.exe demo/typeset_metrics.py more_comparisons/3jp.jpg --tag base
    .venv/Scripts/python.exe demo/typeset_metrics.py --batch --tag base
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEMO_DIR = PROJECT_ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from glasstranslate.core.types import RGB  # noqa: E402
from glasstranslate.render import build_blocks  # noqa: E402
from glasstranslate.render.place import panel_borders  # noqa: E402
import typeset_dev as TD  # noqa: E402
import typeset_reference as TR  # noqa: E402

log = logging.getLogger("typeset_metrics")

METRICS_DIR = PROJECT_ROOT / "demo" / "output" / "metrics"
SSIM_SIGMA = 1.5
SSIM_K1, SSIM_K2 = 0.01, 0.03
ERASE_HALO = 4  # px: the SSIM / MAE neighbourhood reaches this far beyond the letterer's erase mask
ENGLISH_HALO = 4  # px around the English lettering excluded from SSIM / MAE
OWN_PAD = 3  # px around our placed lettering box that counts as the block's own lettering (free text)
COLLISION_MARGIN = 1  # px: panel border lines are grown this much before counting collisions
FONT_CAP_EM = 0.84  # Anime Ace 2.0 BB cap height as a fraction of the font size (render/typeset.py metrics)
MIN_REF_CAP_EM = 0.3  # a reference "cap height" below this many source ems is an OCR fragment, not a line
# Geometry metrics (see the "interior" helpers below).
INTERIOR_MARGIN_EM = 0.18  # the air the lettering keeps from the outline (render/layout._BUBBLE_MARGIN_EM)
OUTLINE_BAND_EM = 0.25  # leftover ink: the outline and its anti-aliased fringe are not the eraser's work
PAIR_MIN_IOU = 0.3  # source-box overlap below which a rendered block is not the ground truth's block


def _luma(rgb: RGB) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


# --------------------------------------------------------------- similarity
def ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = SSIM_SIGMA) -> np.ndarray:
    """Per-pixel SSIM of two float grey images (0..255), Gaussian window."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (SSIM_K1 * 255.0) ** 2, (SSIM_K2 * 255.0) ** 2

    def g(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)

    mu_a, mu_b = g(a), g(b)
    var_a = g(a * a) - mu_a * mu_a
    var_b = g(b * b) - mu_b * mu_b
    cov = g(a * b) - mu_a * mu_b
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return num / den


def masked_mean(values: np.ndarray, mask: np.ndarray) -> Optional[float]:
    mask = mask.astype(bool)
    return float(values[mask].mean()) if mask.any() else None


def _window_mask(shape: Sequence[int], window: Sequence[int]) -> np.ndarray:
    x, y, w, h = (int(v) for v in window)
    m = np.zeros((int(shape[0]), int(shape[1])), bool)
    m[max(0, y): y + h, max(0, x): x + w] = True
    return m


def _ratio(num: int, den: int) -> Optional[float]:
    return float(num) / den if den > 0 else None


# --------------------------------------------------------------- erase
def score_erase(ja_gray: np.ndarray, ours_gray: np.ndarray, eng_gray: np.ndarray, gt: TR.EraseGroundTruth, record: Dict[str, Any],
                region: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Erase scores of one block (see the module docstring); ``ours_gray`` is
    our erased render, ``record`` the block's ground-truth record (window,
    dark flag), ``region`` the judged area (``typeset_reference.gt_region_mask``;
    the record's window when None)."""
    window = region.astype(bool) if region is not None else _window_mask(ja_gray.shape, record["window"])
    dark = bool(record.get("dark"))
    ja_ink = TR.ink_mask(ja_gray, dark)
    ours_ink = TR.ink_mask(ours_gray, dark)
    english_zone = TR._dilate(gt.english, ENGLISH_HALO)
    judged = window & ~english_zone
    # The same rule as typeset_reference.erase_ground_truth: ink is "removed" when no ink survives within ENG_TOL px.
    ours_removed = ja_ink & ~TR._dilate(ours_ink, TR.ENG_TOL) & judged
    gt_removed = gt.erase & judged
    inter = int((ours_removed & gt_removed).sum())
    union = int((ours_removed | gt_removed).sum())
    kept = gt.kept_art & window
    region = TR._dilate(gt_removed, ERASE_HALO) & judged
    ssim = masked_mean(ssim_map(ours_gray, eng_gray), region)
    mae = masked_mean(np.abs(ours_gray.astype(np.float64) - eng_gray.astype(np.float64)), region)
    # Bubbles: ink of any kind left in the judged interior core (the letterer leaves paper there).
    residue = int((ours_ink & judged).sum()) if record.get("kind") == "bubble" else None
    return {
        "erase_iou": _ratio(inter, union),
        "erase_recall": _ratio(inter, int(gt_removed.sum())),
        "erase_precision": _ratio(inter, int(ours_removed.sum())),
        "art_kept": _ratio(int((kept & ours_ink).sum()), int(kept.sum())),
        "ssim": ssim,
        "mae": mae,
        "residue_px": residue,
        "gt_erase_px": int(gt_removed.sum()),
        "our_erase_px": int(ours_removed.sum()),
    }


# --------------------------------------------------------------- lettering
def score_lettering(ours_ink: np.ndarray, interior: Optional[np.ndarray], em: float, ref_stats: Dict[str, Any], blocked: np.ndarray,
                    region: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Compare our lettering ink with the reference's statistics (from
    ``typeset_reference.lettering_stats``) with the same procedure; ``region``
    is the block's own part of a shared bubble (centring reference)."""
    ours = TR.lettering_stats(ours_ink, interior, em, region)
    cap_ratio = None
    if ours["cap_height_px"] and ref_stats.get("cap_height_px"):
        cap_ratio = ours["cap_height_px"] / ref_stats["cap_height_px"]
    return {
        "cap_ratio": cap_ratio,
        "cap_em": ours["cap_em"],
        "cap_ref_em": ref_stats.get("cap_em"),
        "inset_em": ours["inset_min_em"],
        "inset_ref_em": ref_stats.get("inset_min_em"),
        "centre_dx_em": ours["centre_dx_em"],
        "centre_dy_em": ours["centre_dy_em"],
        "centre_dx_ref_em": ref_stats.get("centre_dx_em"),
        "centre_dy_ref_em": ref_stats.get("centre_dy_em"),
        "lines": ours["line_count"],
        "lines_ref": ref_stats.get("line_count"),
        "overflow_px": int(ours["overflow_px"]),
        "collision_px": int((ours_ink.astype(bool) & blocked.astype(bool)).sum()),
        "ink_px": ours["ink_px"],
    }


def our_lettering(typeset_gray: np.ndarray, erased_gray: np.ndarray, own: np.ndarray, dark: bool) -> np.ndarray:
    """Our lettering ink: pixels the lettering pass changed that read as ink, inside ``own``."""
    changed = np.abs(typeset_gray.astype(np.int16) - erased_gray.astype(np.int16)) > 8
    return changed & TR.ink_mask(typeset_gray, dark) & own.astype(bool)


# --------------------------------------------------------------- geometry
def interior_core(interior: np.ndarray, em: float, margin_em: float = INTERIOR_MARGIN_EM) -> np.ndarray:
    """The *interior eroded by a margin*: the bubble's paper region (as
    ``typeset_reference.block_interior`` finds it - the light/dark paper
    component holding the block's text, the outline excluded) shrunk by
    ``margin_em`` source ems.  ``render/layout.py`` insets the layout region
    from the outline by the same ``_BUBBLE_MARGIN_EM``, so a correct render
    touches this core's edge at worst and never crosses it.  The un-eroded
    interior is returned when the erosion empties the mask (a caption box
    barely taller than its text).

    ``cv2.erode``'s default border treats the outside of the image as maximal,
    so a panel running off the page is *not* eroded at that edge.  That is
    deliberate: there is no outline there to keep air from, and eroding it
    would fail a render for lettering the letterer would have set just as
    close."""
    radius = max(1, int(round(margin_em * max(float(em), 1e-6))))
    core = cv2.erode(interior.astype(np.uint8), TR._kernel(radius)).astype(bool)
    return core if core.any() else interior.astype(bool)


def interior_centre(region: np.ndarray) -> Optional[Tuple[float, float]]:
    """Optical centre of a bubble region: the peak of its distance transform,
    i.e. the centre of the largest circle inscribed in it (the mean of the
    peak pixels when several tie).  Balloons have tails and flat sides, so
    neither the bounding-box centre nor the area centroid is where a letterer
    puts the middle of the block; the inscribed circle is.

    The region is padded with one row/column of background first.  A panel that
    runs to the edge of the page has no zeros beyond it, so an unpadded
    transform keeps rising all the way out and puts the peak *on the border*
    (4ja block 8, a dark panel at ``46,734,348,466`` on a 1200-row page, scored
    its centre at ``(241, 1199)`` instead of ``(171, 933)``).  A letterer
    centres the block in the part of the panel that is on the page."""
    region = region.astype(bool)
    if not region.any():
        return None
    padded = np.pad(region, 1).view(np.uint8)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dt >= dt.max() - 1e-6)
    return float(xs.mean()) - 1.0, float(ys.mean()) - 1.0


def score_geometry(ours_ink: np.ndarray, interior: Optional[np.ndarray], em: float,
                   region: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Where our lettering sits in the bubble: ``containment`` is the fraction
    of our glyph pixels *outside* :func:`interior_core` (0 = every glyph inside,
    the only passing value) with ``outside_px`` the count; ``centre_offset_em``
    is the distance from our ink's centroid to :func:`interior_centre` of
    ``region`` (the block's own part of a shared bubble; the whole interior
    when None), in source ems.

    Free text has no interior: both scores are None, and :func:`summarize`
    leaves such blocks out of the bubble means instead of folding a zero in.
    """
    out: Dict[str, Any] = {"containment": None, "outside_px": None, "centre_offset_em": None}
    ink = ours_ink.astype(bool)
    if interior is None or not interior.any() or not ink.any():
        return out
    core = interior_core(interior, em)
    outside = int((ink & ~core).sum())
    out["outside_px"] = outside
    out["containment"] = outside / float(int(ink.sum()))
    centre = interior_centre(region if region is not None else interior)
    m = cv2.moments(ink.view(np.uint8), binaryImage=True)
    if centre is not None and m["m00"] > 0:
        dx = m["m10"] / m["m00"] - centre[0]
        dy = m["m01"] / m["m00"] - centre[1]
        out["centre_offset_em"] = float(np.hypot(dx, dy)) / max(float(em), 1e-6)
    return out


def score_leftover(erased_gray: np.ndarray, interior: Optional[np.ndarray], em: float, dark: bool) -> Dict[str, Any]:
    """``leftover_px``: source ink still readable inside the bubble after the
    erase pass.  Judged on our *erased* render (no lettering on it yet) inside
    the interior eroded by ``OUTLINE_BAND_EM`` ems, so the outline and its
    anti-aliased fringe - which the eraser deliberately keeps - never count.
    This is what catches furigana the eraser never saw.  None for free text."""
    if interior is None or not interior.any():
        return {"leftover_px": None}
    band = interior_core(interior, em, OUTLINE_BAND_EM)
    return {"leftover_px": int((TR.ink_mask(erased_gray, dark) & band).sum())}


def _rect_iou(gt_box: Sequence[int], bbox: Any) -> float:
    """Overlap of a ground-truth ``[x, y, w, h]`` with a block's source bbox."""
    ax, ay, aw, ah = (int(v) for v in gt_box)
    x1, y1 = max(ax, bbox.x), max(ay, bbox.y)
    x2, y2 = min(ax + aw, bbox.x2), min(ay + ah, bbox.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bbox.w * bbox.h - inter
    return inter / float(union) if union > 0 else 0.0


def pair_blocks(records: Sequence[Dict[str, Any]], blocks: Sequence[Any],
                min_iou: float = PAIR_MIN_IOU) -> Tuple[List[Optional[int]], List[int]]:
    """Match each ground-truth record to the rendered block covering the same
    source text, by source-box overlap, best pair first.

    Deliberately not by position.  The layout may split one utterance into two
    blocks, or stop dropping a line it used to read as furigana; from that
    point on a positional pairing scores every block against its neighbour's
    ground truth, and says nothing (3jp gained a block at index 14 and its last
    three rows were judged against the wrong records, at overlaps of 0.25, 0.00
    and 0.00).  Returns the block index per record - None where we rendered
    nothing for it - and the indices of the blocks no record claims.
    """
    scored = sorted(
        (_rect_iou(rec["source_bbox"], blocks[j].segment.bbox), k, j)
        for k, rec in enumerate(records) if rec.get("source_bbox")
        for j in range(len(blocks))
    )
    out: List[Optional[int]] = [None] * len(records)
    taken: set = set()
    for score, k, j in reversed(scored):
        if score < min_iou or out[k] is not None or j in taken:
            continue
        out[k] = j
        taken.add(j)
    return out, [j for j in range(len(blocks)) if j not in taken]


def _interior_of(block: Any) -> Optional[Tuple[Any, np.ndarray]]:
    """The patch of paper a block is *lettered into*, as ``(box, mask)`` in
    page coordinates, or None for free text (which has no interior)."""
    style = block.style
    box = getattr(style, "layout_box", None)
    mask = getattr(style, "layout_mask", None)
    if not getattr(style, "in_bubble", False) or box is None or mask is None:
        return None
    if mask.shape[:2] != (box.h, box.w):
        return None
    return box, mask.astype(bool)


def _interiors_overlap(a: Tuple[Any, np.ndarray], b: Tuple[Any, np.ndarray]) -> bool:
    """True when two lettering regions share a pixel."""
    ra, ma = a
    rb, mb = b
    x1, y1 = max(ra.x, rb.x), max(ra.y, rb.y)
    x2, y2 = min(ra.x2, rb.x2), min(ra.y2, rb.y2)
    if x2 <= x1 or y2 <= y1:
        return False
    sa = ma[y1 - ra.y: y2 - ra.y, x1 - ra.x: x2 - ra.x]
    sb = mb[y1 - rb.y: y2 - rb.y, x1 - rb.x: x2 - rb.x]
    return bool(np.any(sa & sb))


def blocks_per_bubble(blocks: Sequence[Any]) -> List[Optional[int]]:
    """Per block, how many blocks are lettered into the same balloon interior.
    One balloon holds one lettered block, so 1 is the only right answer; 2 or
    more means two blocks letter into each other's half.

    A block the layout put in a bubble is lettered into its own interior
    (``style.layout_box`` / ``layout_mask``), so the count is how many
    interiors overlap this one: :mod:`render.bubbles` cuts a joined component
    into one interior per block, and after that cut a shared patch of *paper*
    is no longer a shared *lettering region*.  Free text has no interior and
    scores ``-``, like the other three geometry metrics - several free-text
    blocks on one panel background are ordinary, not the defect."""
    interiors = [_interior_of(b) for b in blocks]
    out: List[Optional[int]] = []
    for own in interiors:
        if own is None:
            out.append(None)
            continue
        out.append(sum(1 for other in interiors if other is not None and _interiors_overlap(own, other)))
    return out


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.{digits}f}"


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Means over the blocks that have a value, plus counts.  Untranslated
    blocks (logos, author names, watermarks the renderer leaves alone by
    design) are listed but not averaged."""
    skipped = sum(1 for r in rows if r.get("untranslated"))
    rows = [r for r in rows if not r.get("untranslated")]
    out: Dict[str, Any] = {"blocks": len(rows), "untranslated": skipped}
    for key in ("erase_iou", "erase_recall", "erase_precision", "art_kept", "ssim", "mae", "cap_ratio", "inset_em", "inset_ref_em"):
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    dx = [abs(float(r["centre_dx_em"])) for r in rows if r.get("centre_dx_em") is not None]
    dy = [abs(float(r["centre_dy_em"])) for r in rows if r.get("centre_dy_em") is not None]
    out["centre_abs_em"] = float(np.mean(dx + dy)) if dx or dy else None
    out["line_matches"] = sum(1 for r in rows if r.get("lines_ref") is not None and r.get("lines") == r.get("lines_ref"))
    out["line_compared"] = sum(1 for r in rows if r.get("lines_ref") is not None)
    out["overflows"] = sum(1 for r in rows if (r.get("overflow_px") or 0) > 0)
    out["collisions"] = sum(1 for r in rows if (r.get("collision_px") or 0) > 0)
    residue = [int(r["residue_px"]) for r in rows if r.get("residue_px") is not None]
    out["residue_px"] = int(np.sum(residue)) if residue else None
    out["residue_bubbles"] = sum(1 for v in residue if v > 0)
    out["bubbles"] = len(residue)
    # Geometry: bubble blocks only - free text has no interior and is never
    # folded into these means (its rows carry None and drop out here).
    contain = [float(r["containment"]) for r in rows if r.get("containment") is not None]
    out["containment"] = float(np.mean(contain)) if contain else None
    out["containment_worst"] = float(np.max(contain)) if contain else None
    out["uncontained"] = sum(1 for v in contain if v > 0)
    offsets = [float(r["centre_offset_em"]) for r in rows if r.get("centre_offset_em") is not None]
    out["centre_offset_em"] = float(np.mean(offsets)) if offsets else None
    out["centre_offset_worst_em"] = float(np.max(offsets)) if offsets else None
    leftover = [int(r["leftover_px"]) for r in rows if r.get("leftover_px") is not None]
    out["leftover_px"] = int(np.sum(leftover)) if leftover else None
    out["leftover_blocks"] = sum(1 for v in leftover if v > 0)
    shared = [int(r["blocks_per_bubble"]) for r in rows if r.get("blocks_per_bubble") is not None]
    out["blocks_per_bubble_max"] = int(np.max(shared)) if shared else None
    out["shared_blocks"] = sum(1 for v in shared if v > 1)
    return out


def page_table(stem: str, tag: str, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    head = ("| # | kind | IoU | recall | prec | art kept | SSIM | MAE | residue px | cap ratio | inset ours/ref (em) | centre dx/dy (em) | "
            "lines ours/ref | overflow px | collision px | contain | centre off (em) | leftover px | blk/bubble |")
    lines = [f"### {stem} ({tag})", "", head, "|" + "---|" * 19]
    for r in rows:
        lines.append(
            f"| {r['index']} | {r['kind']} | {_fmt(r.get('erase_iou'))} | {_fmt(r.get('erase_recall'))} | {_fmt(r.get('erase_precision'))} | "
            f"{_fmt(r.get('art_kept'))} | {_fmt(r.get('ssim'), 3)} | {_fmt(r.get('mae'), 1)} | {_fmt(r.get('residue_px'))} | {_fmt(r.get('cap_ratio'))} | "
            f"{_fmt(r.get('inset_em'))}/{_fmt(r.get('inset_ref_em'))} | {_fmt(r.get('centre_dx_em'))}/{_fmt(r.get('centre_dy_em'))} | "
            f"{_fmt(r.get('lines'))}/{_fmt(r.get('lines_ref'))} | {_fmt(r.get('overflow_px'))} | {_fmt(r.get('collision_px'))} | "
            f"{_fmt(r.get('containment'), 3)} | {_fmt(r.get('centre_offset_em'))} | {_fmt(r.get('leftover_px'))} | "
            f"{_fmt(r.get('blocks_per_bubble'))} |"
        )
    lines.append(
        f"| **mean** | {summary['blocks']} blocks | {_fmt(summary['erase_iou'])} | {_fmt(summary['erase_recall'])} | "
        f"{_fmt(summary['erase_precision'])} | {_fmt(summary['art_kept'])} | {_fmt(summary['ssim'], 3)} | {_fmt(summary['mae'], 1)} | "
        f"{_fmt(summary['residue_px'])} in {summary['residue_bubbles']}/{summary['bubbles']} | "
        f"{_fmt(summary['cap_ratio'])} | {_fmt(summary['inset_em'])}/{_fmt(summary['inset_ref_em'])} | abs {_fmt(summary['centre_abs_em'])} | "
        f"{summary['line_matches']}/{summary['line_compared']} same | {summary['overflows']} blocks | {summary['collisions']} blocks | "
        f"{_fmt(summary['containment'], 3)} worst {_fmt(summary['containment_worst'], 3)} | "
        f"{_fmt(summary['centre_offset_em'])} worst {_fmt(summary['centre_offset_worst_em'])} | "
        f"{_fmt(summary['leftover_px'])} in {summary['leftover_blocks']} | "
        f"max {_fmt(summary['blocks_per_bubble_max'])} in {summary['shared_blocks']} |"
    )
    return "\n".join(lines) + "\n"


def summary_table(tag: str, pages: Sequence[Dict[str, Any]]) -> str:
    lines = [f"## Summary ({tag})", "",
             "| page | blocks | IoU | recall | prec | art kept | SSIM | MAE | bubble residue px (bubbles) | cap ratio | inset ours/ref | centre abs (em) | lines same | overflows | collisions | contain mean/worst | centre off mean/worst (em) | leftover px (blocks) | blk/bubble max (blocks sharing) |",
             "|" + "---|" * 19]

    def row(name: str, s: Dict[str, Any]) -> str:
        return (f"| {name} | {s['blocks']} | {_fmt(s['erase_iou'])} | {_fmt(s['erase_recall'])} | {_fmt(s['erase_precision'])} | {_fmt(s['art_kept'])} | "
                f"{_fmt(s['ssim'], 3)} | {_fmt(s['mae'], 1)} | {_fmt(s['residue_px'])} ({s['residue_bubbles']}/{s['bubbles']}) | "
                f"{_fmt(s['cap_ratio'])} | {_fmt(s['inset_em'])}/{_fmt(s['inset_ref_em'])} | "
                f"{_fmt(s['centre_abs_em'])} | {s['line_matches']}/{s['line_compared']} | {s['overflows']} | {s['collisions']} | "
                f"{_fmt(s['containment'], 3)}/{_fmt(s['containment_worst'], 3)} | "
                f"{_fmt(s['centre_offset_em'])}/{_fmt(s['centre_offset_worst_em'])} | "
                f"{_fmt(s['leftover_px'])} ({s['leftover_blocks']}) | {_fmt(s['blocks_per_bubble_max'])} ({s['shared_blocks']}) |")

    for p in pages:
        lines.append(row(p["stem"], p["summary"]))
    if pages:
        lines.append(row("**all**", summarize([r for p in pages for r in p["blocks"]])))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- pages
def load_ground_truth(stem: str) -> Optional[Dict[str, Any]]:
    d = TD.reference_dir(stem)
    needed = [d / "erase_gt.png", d / "kept_art.png", d / "english_ink.png", d / TD.ALIGNED_REFERENCE_NAME, d / "blocks.json"]
    if not all(p.exists() for p in needed):
        return None
    masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in needed[:3]]
    aligned = cv2.imread(str(needed[3]))
    return {
        "gt": TR.EraseGroundTruth(*masks),
        "eng_gray": cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY),
        "records": json.loads(needed[4].read_text(encoding="utf-8"))["blocks"],
    }


def score_page(image: Path, render_dir: Path, tag: str, *, device: str = "auto", min_confidence: float = 0.5) -> Optional[Dict[str, Any]]:
    stem = TD.page_stem(image)
    truth = load_ground_truth(stem)
    if truth is None:
        print(f"{stem}: no ground truth under {TD.reference_dir(stem)} (run demo/typeset_reference.py)", file=sys.stderr)
        return None
    ja = cv2.imread(str(image))
    erased = cv2.imread(str(render_dir / f"{tag}_erased.png"))
    typeset = cv2.imread(str(render_dir / f"{tag}_typeset.png"))
    blocks_path = render_dir / f"{tag}_blocks.json"
    if ja is None or erased is None or typeset is None or not blocks_path.exists():
        print(f"{stem}: render {tag} not found in {render_dir} (run demo/typeset_dev.py)", file=sys.stderr)
        return None
    ours_records = json.loads(blocks_path.read_text(encoding="utf-8"))["blocks"]
    ja_gray = cv2.cvtColor(ja, cv2.COLOR_BGR2GRAY)
    erased_gray = cv2.cvtColor(erased, cv2.COLOR_BGR2GRAY)
    typeset_gray = cv2.cvtColor(typeset, cv2.COLOR_BGR2GRAY)
    gt: TR.EraseGroundTruth = truth["gt"]
    eng_gray: np.ndarray = truth["eng_gray"]
    records: List[Dict[str, Any]] = truth["records"]

    all_segments = TD.load_segments(ja, stem, False, device)
    segments = [s for s in all_segments if s.confidence >= min_confidence]
    blocks = build_blocks(ja, segments, all_segments)
    interiors = [TR.block_interior(ja_gray, b) for b in blocks]
    ownership = TR.bubble_ownership(blocks, interiors)
    _, border_lines = panel_borders(ja_gray, [b.segment.bbox for b in blocks])
    borders = TR._dilate(border_lines, COLLISION_MARGIN)
    shared = blocks_per_bubble(blocks)
    h, w = ja_gray.shape

    pairing, unpaired = pair_blocks(records, blocks)
    if unpaired:
        log.warning("%s: %d rendered block(s) match no ground-truth block and are unscored: %s",
                    stem, len(unpaired), [blocks[j].segment.text[:12] for j in unpaired])
    missed = [rec["index"] for k, rec in enumerate(records) if pairing[k] is None]
    if missed:
        log.warning("%s: %d ground-truth block(s) we produced nothing for: %s", stem, len(missed), missed)

    rows: List[Dict[str, Any]] = []
    for k, rec in enumerate(records):
        i = pairing[k]
        if i is None:
            rows.append({"index": rec["index"], "kind": rec["kind"], "text": rec["text"], "missed": True})
            continue
        ours_rec = ours_records[i] if i < len(ours_records) else None
        row: Dict[str, Any] = {"index": rec["index"], "kind": rec["kind"], "text": rec["text"]}
        interior = interiors[i] if i < len(interiors) else None
        region = TR.gt_region_mask(blocks[i], interior, (h, w)) if i < len(blocks) else None
        row.update(score_erase(ja_gray, erased_gray, eng_gray, gt, rec, region))
        dark = bool(rec.get("dark"))
        em = float(rec.get("em_px") or 1.0)
        # Where our own lettering may be: our part of the bubble interior, else our placed box.
        own_part = ownership[i] if i < len(ownership) else None
        placed = None
        if ours_rec is not None and ours_rec.get("bbox"):
            x, y, bw, bh = ours_rec["bbox"]
            placed = _window_mask((h, w), [x - OWN_PAD, y - OWN_PAD, bw + 2 * OWN_PAD, bh + 2 * OWN_PAD])
        if own_part is not None:
            own = TR._dilate(own_part, OWN_PAD)
            if placed is not None:
                own &= placed  # a free-text neighbour lettered inside our bubble is not our lettering
        elif placed is not None:
            own = placed
        else:
            own = np.zeros((h, w), bool)
        # Blocked for collisions: other blocks' text / bubbles (a co-tenant of our bubble: its own part
        # of it) and the panel border lines.
        blocked = borders.copy()
        same_bubble = blocks[i].bubble if i < len(blocks) else None
        for j, other in enumerate(blocks):
            if j == i:
                continue
            if same_bubble is not None and other.bubble == same_bubble and ownership[j] is not None:
                blocked |= ownership[j]
                continue
            if interiors[j] is not None:
                blocked |= interiors[j]
            s = other.segment.bbox
            blocked |= _window_mask((h, w), [s.x, s.y, s.w, s.h])
        ink = our_lettering(typeset_gray, erased_gray, own, dark)
        row.update(score_lettering(ink, interior, em, rec.get("stats") or {}, blocked, own_part))
        row.update(score_geometry(ink, interior, em, own_part))
        row.update(score_leftover(erased_gray, interior, em, dark))
        row["blocks_per_bubble"] = shared[i] if i < len(shared) else None
        if ours_rec is not None:
            row["translation"] = ours_rec.get("translation")
            row["size_px"] = ours_rec.get("size")
            row["untranslated"] = bool(ours_rec.get("untranslated"))
            # Our size and line count are known exactly from the typeset record; pixel clustering of our
            # lettering can merge with a neighbour's.  The reference cap must be a real line (>= MIN_REF_CAP_EM).
            ref_cap = (rec.get("stats") or {}).get("cap_height_px")
            if ours_rec.get("size") and ref_cap and ref_cap >= MIN_REF_CAP_EM * em:
                row["cap_ratio"] = FONT_CAP_EM * float(ours_rec["size"]) / float(ref_cap)
            else:
                row["cap_ratio"] = None
            row["lines"] = int(ours_rec.get("line_count") or 0)
        rows.append(row)
    summary = summarize(rows)
    summary["unpaired_blocks"] = len(unpaired)
    summary["missed_blocks"] = len(missed)
    return {"stem": stem, "tag": tag, "image": str(image), "render_dir": str(render_dir), "blocks": rows, "summary": summary}


def write_page(result: Dict[str, Any], out_dir: Path = METRICS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{result['stem']}_{result['tag']}"
    base.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    md = page_table(result["stem"], result["tag"], result["blocks"], result["summary"])
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    return base.with_suffix(".md")


def _lpips_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("lpips") is not None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, default=TD.DEFAULT_IMAGE)
    parser.add_argument("--batch", action="store_true", help="score every reference pair (renders under --dir/<stem>/)")
    parser.add_argument("--tag", default="dev")
    parser.add_argument("--dir", type=Path, default=PROJECT_ROOT / "demo" / "output" / "dev",
                        help="render directory (typeset_dev --out); in --batch mode the per-stem subdirectories")
    parser.add_argument("--out", type=Path, default=METRICS_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--lpips", action="store_true", help="also compute LPIPS when the lpips package is importable")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.lpips:
        print("lpips available" if _lpips_available() else "note: --lpips needs the sidecar venv (torch); ignored here", file=sys.stderr)

    if args.batch:
        pages = []
        for image in TD.all_pairs():
            result = score_page(image, args.dir / TD.page_stem(image), args.tag, device=args.device, min_confidence=args.min_confidence)
            if result is None:
                continue
            write_page(result, args.out)
            pages.append(result)
            print(page_table(result["stem"], result["tag"], result["blocks"], result["summary"]))
        if not pages:
            return 1
        md = summary_table(args.tag, pages)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / f"{args.tag}_summary.md").write_text(md, encoding="utf-8")
        print(md)
        return 0
    result = score_page(args.image, args.dir, args.tag, device=args.device, min_confidence=args.min_confidence)
    if result is None:
        return 1
    path = write_page(result, args.out)
    print(page_table(result["stem"], result["tag"], result["blocks"], result["summary"]))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
