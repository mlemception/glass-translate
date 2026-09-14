"""Ground truth from a professionally lettered reference page.

For every page that has a reference (see ``typeset_dev.reference_image_for``:
``Examples/before.jpg`` -> ``after.webp``, ``more_comparisons/<n>ja.jpg`` ->
``<n>eng.jpg``) this script

1. **aligns** the reference to the page: an ORB + RANSAC homography for the
   coarse fit, refined with ECC (``MOTION_HOMOGRAPHY``) on blurred, downscaled
   greyscale; hand-picked point pairs (``--pairs "xj,yj=xe,ye;..."``) are the
   fallback.  The reference is warped into the page's pixel grid
   (``eng_aligned.png``, ``align.json``);
2. **OCRs** the reference (rapidocr on the full-resolution scan, boxes mapped
   through the homography; cached in ``demo/cache/<stem>/eng_segments.pkl``)
   to find the English lettering;
3. **derives** the letterer's work near every block of the page:
   ``erase_gt.png`` (Japanese ink the letterer removed, outside the English
   lettering), ``kept_art.png`` (ink present in both pages), ``english_ink.png``
   (the English lettering's ink), and per block the reference lettering's
   statistics - cap height against the source em, line count and pitch, inset
   from the bubble outline, centring - into ``blocks.json``;
4. **bootstraps** ``demo/reference_text_<stem>.json`` from the English OCR
   when that file does not exist yet (it is never overwritten: correct it by
   hand from the ``--crops`` review sheets).

Outputs live under ``demo/reference/<stem>/``.  ``demo/typeset_metrics.py``
scores a render against them with the same ``lettering_stats`` procedure, so
"our" numbers and the reference's are directly comparable.

Usage::

    set PYTHONUTF8=1
    .venv/Scripts/python.exe demo/typeset_reference.py more_comparisons/3jp.jpg --crops
    .venv/Scripts/python.exe demo/typeset_reference.py --batch --crops
    .venv/Scripts/python.exe demo/typeset_reference.py more_comparisons/1ja.jpg --pairs "10,20=14,27;700,30=933,40;..."
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEMO_DIR = PROJECT_ROOT / "demo"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from glasstranslate.core.types import Rect, Segment  # noqa: E402
from glasstranslate.render import build_blocks  # noqa: E402
from glasstranslate.render.layout import TextBlock  # noqa: E402
import typeset_dev as TD  # noqa: E402

log = logging.getLogger("typeset_reference")

Box = Rect  # the same axis-aligned integer rectangle as the pipeline's

# --- tunables --------------------------------------------------------------
INK_THR = 128  # grey level below which a pixel is ink on light paper (above 255 - INK_THR on dark paper)
NEAR_EM = 1.5  # free-text window: the source box grown by this many source ems
BUBBLE_PAD = 3  # px the bubble window reaches beyond the bubble rect
ASSIGN_EM = 2.0  # English lines whose centre is within this many ems of a free-text source belong to it
ENGLISH_PAD = 3  # px around the English lettering that never count as erased or kept
ENGLISH_BOX_PAD = 2  # px the English OCR boxes are grown before masking their ink
ENG_TOL = 2  # px: page ink with reference ink within this distance counts as kept (scan / alignment tolerance)
GT_BOX_EM = 0.5  # free text: the letterer's erase is judged inside the block's own OCR boxes grown by this many ems
CORE_ERODE = 3  # bubbles: judged inside the interior eroded by this many px (the outline's inner edge is scan noise)
ECC_MAX_SIDE = 700  # ECC runs on images no larger than this (pixels, long side)
ECC_ITERATIONS = 200
ECC_EPS = 1e-6
ECC_BLUR = 2.0  # sigma of the Gaussian blur before ECC (line art needs smooth gradients)
ORB_FEATURES = 8000
ORB_RATIO = 0.75
ORB_MIN_MATCHES = 12
ORB_RANSAC_PX = 4.0
LINE_MIN_GAP = 3  # rows without ink that separate two lettering lines
LINE_MIN_HEIGHT = 3  # rows: shorter ink runs are specks, not lines
PAPER_LIGHT = 200  # as render/layout.py: grey above this is light paper...
PAPER_DARK = 60  # ...and below this is dark paper
TEXT_PAD = 2  # px around a text box painted as paper when finding the bubble interior
ENG_MIN_CONFIDENCE = 0.3  # rapidocr score below which an English line is ignored
CROP_WIDTH = 300  # review sheet tile width
SHEET_MAX_HEIGHT = 950  # review sheets stay under 1000 px so a model may view them


# --------------------------------------------------------------- alignment
@dataclass
class AlignResult:
    """``homography`` maps reference (eng) pixels to page (ja) pixels."""

    homography: np.ndarray
    method: str
    score: float
    note: str = ""


def _normalize(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h


def _blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)


def alignment_score(ja_gray: np.ndarray, eng_aligned_gray: np.ndarray) -> float:
    """Pearson correlation of the two blurred pages (1 = identical)."""
    a = _blur(ja_gray, ECC_BLUR).ravel()
    b = _blur(eng_aligned_gray, ECC_BLUR).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def _prescale(eng_gray: np.ndarray, ja_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Resize the reference to the page's height; returns ``(scaled, S)`` with
    ``S`` mapping reference pixels to scaled pixels."""
    h = ja_shape[0]
    s = h / eng_gray.shape[0]
    w = max(1, int(round(eng_gray.shape[1] * s)))
    scaled = cv2.resize(eng_gray, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return scaled, np.diag([s, s, 1.0])


def _sane(h: np.ndarray) -> bool:
    if not np.all(np.isfinite(h)):
        return False
    a = h[:2, :2]
    det = float(np.linalg.det(a))
    if det <= 0:
        return False
    scale = float(np.sqrt(det))
    return 0.25 <= scale <= 4.0 and abs(h[2, 0]) < 1e-2 and abs(h[2, 1]) < 1e-2


def _orb_homography(ja_gray: np.ndarray, eng_gray: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    """Coarse ``eng -> ja`` homography from ORB matches; ``(None, matches)`` when too few agree."""
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp_j, des_j = orb.detectAndCompute(ja_gray, None)
    kp_e, des_e = orb.detectAndCompute(eng_gray, None)
    if des_j is None or des_e is None or len(kp_j) < ORB_MIN_MATCHES or len(kp_e) < ORB_MIN_MATCHES:
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_e, des_j, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ORB_RATIO * n.distance]
    if len(good) < ORB_MIN_MATCHES:
        return None, len(good)
    src = np.float32([kp_e[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_j[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ORB_RANSAC_PX)
    n_in = int(inliers.sum()) if inliers is not None else 0
    if h is None or n_in < ORB_MIN_MATCHES or not _sane(h):
        return None, n_in
    return _normalize(h), n_in


def _ecc_refine(ja_gray: np.ndarray, eng_gray: np.ndarray, h_init: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """Refine an ``eng -> ja`` homography with ECC on downscaled, blurred pages."""
    f = min(1.0, ECC_MAX_SIDE / max(ja_gray.shape))
    s = np.diag([f, f, 1.0])
    s_inv = np.diag([1.0 / f, 1.0 / f, 1.0])

    def small(gray: np.ndarray) -> np.ndarray:
        if f < 1.0:
            gray = cv2.resize(gray, (max(1, int(round(gray.shape[1] * f))), max(1, int(round(gray.shape[0] * f)))), interpolation=cv2.INTER_AREA)
        return _blur(gray, ECC_BLUR) / 255.0

    template, inp = small(ja_gray), small(eng_gray)
    # ECC's warp maps template (ja) coordinates to input (eng) coordinates.
    w_init = s @ np.linalg.inv(h_init) @ s_inv
    warp = np.ascontiguousarray(_normalize(w_init), dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPS)
    try:
        cc, warp = cv2.findTransformECC(template, inp, warp, cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
    except cv2.error as exc:
        log.warning("ECC did not converge (%s); keeping the coarse homography", str(exc).splitlines()[-1][:80])
        return None
    w = s_inv @ np.asarray(warp, dtype=np.float64) @ s
    h = _normalize(np.linalg.inv(w))
    if not _sane(h):
        return None
    return h, float(cc)


def align_pages(
    ja_gray: np.ndarray,
    eng_gray: np.ndarray,
    pairs: Optional[Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
) -> AlignResult:
    """Homography mapping the reference page onto the page (``eng -> ja``).

    ``pairs`` are hand-picked ``((xj, yj), (xe, ye))`` correspondences (at
    least four): they replace the automatic ORB + ECC estimate.
    """
    if pairs:
        if len(pairs) < 4:
            raise ValueError("at least four point pairs are needed")
        src = np.float64([p[1] for p in pairs]).reshape(-1, 1, 2)
        dst = np.float64([p[0] for p in pairs]).reshape(-1, 1, 2)
        h, _ = cv2.findHomography(src, dst, 0)
        if h is None:
            raise ValueError("degenerate point pairs")
        h = _normalize(h)
        score = alignment_score(ja_gray, warp_reference(eng_gray, h, ja_gray.shape))
        return AlignResult(h, "pairs", score)

    scaled, s_pre = _prescale(eng_gray, ja_gray.shape[:2])
    h_s, n_in = _orb_homography(ja_gray, scaled)
    method, note = "orb", f"{n_in} ORB inliers"
    if h_s is None:
        h_s, method, note = np.eye(3), "scale", f"ORB failed ({n_in} matches); pure scale"
    refined = _ecc_refine(ja_gray, scaled, h_s)
    if refined is not None:
        h_s, cc = refined
        method, note = "ecc", f"{note}; ECC cc {cc:.4f}"
    h = _normalize(h_s @ s_pre)
    score = alignment_score(ja_gray, warp_reference(eng_gray, h, ja_gray.shape))
    return AlignResult(h, method, score, note)


def warp_reference(eng: np.ndarray, homography: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Warp the reference (grey or BGR) into the page's pixel grid.  A reference
    larger than the page is area-downsampled first so it is not aliased."""
    h, w = int(shape[0]), int(shape[1])
    hom = _normalize(homography)
    scale = float(np.sqrt(abs(np.linalg.det(hom[:2, :2]))))
    src = eng
    if scale < 0.95:
        src = cv2.resize(eng, (max(1, int(round(eng.shape[1] * scale))), max(1, int(round(eng.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        hom = hom @ np.diag([1.0 / scale, 1.0 / scale, 1.0])
    return cv2.warpPerspective(src, hom, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def parse_pairs(text: str) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """``"xj,yj=xe,ye;xj,yj=xe,ye"`` -> ``[((xj, yj), (xe, ye)), ...]``."""
    out = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        left, right = item.split("=")
        xj, yj = (float(v) for v in left.split(","))
        xe, ye = (float(v) for v in right.split(","))
        out.append(((xj, yj), (xe, ye)))
    return out


# --------------------------------------------------------------- masks
def ink_mask(gray: np.ndarray, dark_paper: bool = False) -> np.ndarray:
    """Bool ink mask: dark pixels on light paper (or light pixels on dark paper)."""
    return (gray > 255 - INK_THR) if dark_paper else (gray < INK_THR)


def _kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), _kernel(radius)).astype(bool)


def boxes_mask(shape: Tuple[int, int], boxes: Sequence[Rect], pad: int = 0) -> np.ndarray:
    mask = np.zeros(shape[:2], bool)
    h, w = mask.shape
    for b in boxes:
        r = Rect(b.x - pad, b.y - pad, b.w + 2 * pad, b.h + 2 * pad).clamp(w, h)
        if r.w > 0 and r.h > 0:
            mask[r.y: r.y2, r.x: r.x2] = True
    return mask


@dataclass
class EraseGroundTruth:
    erase: np.ndarray  # bool: Japanese ink the letterer removed
    kept_art: np.ndarray  # bool: ink present in both pages near the text
    english: np.ndarray  # bool: the English lettering's ink


def erase_ground_truth(
    ja_gray: np.ndarray,
    eng_gray: np.ndarray,
    near_text: np.ndarray,
    english_boxes: Sequence[Rect],
    dark: Optional[np.ndarray] = None,
) -> EraseGroundTruth:
    """Derive the letterer's erase mask from the page and its aligned reference.

    ``near_text`` bounds the comparison to the neighbourhood of the page's text
    blocks; ``english_boxes`` are the OCR boxes of the English lettering on
    the aligned reference; ``dark`` (optional bool mask) flags dark-paper areas
    where the ink sense is inverted.
    """
    ja_ink = ink_mask(ja_gray)
    eng_ink = ink_mask(eng_gray)
    if dark is not None and dark.any():
        ja_ink = np.where(dark, ink_mask(ja_gray, True), ja_ink)
        eng_ink = np.where(dark, ink_mask(eng_gray, True), eng_ink)
    english = eng_ink & boxes_mask(ja_gray.shape, english_boxes, ENGLISH_BOX_PAD)
    zone = _dilate(english, ENGLISH_PAD)
    keep_free = near_text & ~zone
    erase = ja_ink & ~_dilate(eng_ink, ENG_TOL) & keep_free
    kept_art = ja_ink & eng_ink & keep_free
    return EraseGroundTruth(erase, kept_art, english)


# --------------------------------------------------------------- lettering
def line_clusters(profile: np.ndarray, min_gap: int = LINE_MIN_GAP, min_height: int = LINE_MIN_HEIGHT) -> List[Tuple[int, int]]:
    """``(top, bottom)`` row ranges of the lettering lines in a row ink profile:
    runs of inked rows, joined across gaps shorter than ``min_gap``, dropped
    when shorter than ``min_height``."""
    rows = np.flatnonzero(np.asarray(profile) > 0)
    if rows.size == 0:
        return []
    runs: List[List[int]] = [[int(rows[0]), int(rows[0]) + 1]]
    for r in rows[1:]:
        r = int(r)
        if r - runs[-1][1] < min_gap:
            runs[-1][1] = r + 1
        else:
            runs.append([r, r + 1])
    return [(a, b) for a, b in runs if b - a >= min_height]


def lettering_stats(ink: np.ndarray, interior: Optional[np.ndarray], em: float, region: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Measure a block of lettering (bool ``ink``): cap height and line pitch
    from the row profile, line count, bounding box, inset from the outline of
    ``interior`` (the bubble's paper, bool; None for free text), centring
    against the centroid of ``region`` (the block's own part of a shared
    bubble; the interior when None), and ink outside the interior.  Sizes are
    also given in units of ``em`` (the source text's em) so pages compare."""
    ink = ink.astype(bool)
    em = max(1e-6, float(em))
    out: Dict[str, Any] = {
        "ink_px": int(ink.sum()), "bbox": None, "line_count": 0, "cap_height_px": None, "cap_em": None,
        "line_pitch_px": None, "pitch_em": None, "inset_min_px": None, "inset_p05_px": None, "inset_min_em": None,
        "centre_dx_px": None, "centre_dy_px": None, "centre_dx_em": None, "centre_dy_em": None, "overflow_px": 0,
    }
    if not ink.any():
        return out
    ys, xs = np.nonzero(ink)
    x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    out["bbox"] = [x1, y1, x2 - x1, y2 - y1]
    clusters = line_clusters(ink.sum(axis=1))
    out["line_count"] = len(clusters)
    if clusters:
        heights = np.array([b - a for a, b in clusters], dtype=np.float64)
        cap = float(np.median(heights))
        out["cap_height_px"] = cap
        out["cap_em"] = cap / em
        if len(clusters) >= 2:
            centres = np.array([(a + b) / 2.0 for a, b in clusters])
            pitch = float(np.median(np.diff(centres)))
            out["line_pitch_px"] = pitch
            out["pitch_em"] = pitch / em
    if interior is not None and interior.any():
        interior = interior.astype(bool)
        inside = ink & interior
        out["overflow_px"] = int((ink & ~interior).sum())
        if inside.any():
            dt = cv2.distanceTransform(interior.astype(np.uint8), cv2.DIST_L2, 5)
            d = dt[inside]
            out["inset_min_px"] = float(d.min())
            out["inset_p05_px"] = float(np.percentile(d, 5))
            out["inset_min_em"] = float(d.min()) / em
        centre_of = interior if region is None else region.astype(bool)
        m = cv2.moments(centre_of.astype(np.uint8), binaryImage=True)
        if m["m00"] > 0:
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            dx, dy = (x1 + x2) / 2.0 - cx, (y1 + y2) / 2.0 - cy
            out["centre_dx_px"], out["centre_dy_px"] = float(dx), float(dy)
            out["centre_dx_em"], out["centre_dy_em"] = float(dx) / em, float(dy) / em
    return out


def paper_component(gray: np.ndarray, text_boxes: Sequence[Rect], seed: Rect, window: Rect, dark: bool) -> Optional[np.ndarray]:
    """The bubble interior as ``render/layout.py`` sees it: the paper component
    (light or dark paper with every text box painted over) holding ``seed``'s
    centre, inside ``window``.  Page-sized bool mask, or None."""
    h, w = gray.shape[:2]
    win = window.clamp(w, h)
    if win.w <= 0 or win.h <= 0:
        return None
    sub = gray[win.y: win.y2, win.x: win.x2]
    paper = (sub < PAPER_DARK) if dark else (sub > PAPER_LIGHT)
    paper = paper.astype(np.uint8)
    for b in text_boxes:
        r = Rect(b.x - TEXT_PAD - win.x, b.y - TEXT_PAD - win.y, b.w + 2 * TEXT_PAD, b.h + 2 * TEXT_PAD).clamp(win.w, win.h)
        if r.w > 0 and r.h > 0:
            paper[r.y: r.y2, r.x: r.x2] = 1
    _, labels = cv2.connectedComponents(paper, connectivity=4)
    cy = min(win.h - 1, max(0, seed.y + seed.h // 2 - win.y))
    cx = min(win.w - 1, max(0, seed.x + seed.w // 2 - win.x))
    label = int(labels[cy, cx])
    if label == 0:
        return None
    out = np.zeros((h, w), bool)
    out[win.y: win.y2, win.x: win.x2] = labels == label
    return out


# --------------------------------------------------------------- per page
def _luma(rgb: Sequence[int]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def block_window(block: TextBlock, shape: Tuple[int, ...]) -> Rect:
    """The neighbourhood the letterer worked in: the bubble (plus a margin) or
    the source box grown by ``NEAR_EM`` ems."""
    h, w = int(shape[0]), int(shape[1])
    st = block.style
    if st.in_bubble and block.bubble is not None:
        b = block.bubble
        return Rect(b.x - BUBBLE_PAD, b.y - BUBBLE_PAD, b.w + 2 * BUBBLE_PAD, b.h + 2 * BUBBLE_PAD).clamp(w, h)
    em = block.em_px or st.text_height_px
    pad = int(round(NEAR_EM * em))
    s = block.segment.bbox
    return Rect(s.x - pad, s.y - pad, s.w + 2 * pad, s.h + 2 * pad).clamp(w, h)


def gt_region_mask(block: TextBlock, interior: Optional[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    """Where the letterer's erase is judged: the bubble's paper interior (the
    outline is not the letterer's work), else the block's own OCR boxes grown
    by ``GT_BOX_EM`` ems (free text: the letterer only erased the glyphs).
    Shared with ``demo/typeset_metrics.py`` so both sides judge one region."""
    h, w = int(shape[0]), int(shape[1])
    if interior is not None:
        # The core of the interior: the outline's inner edge differs between two scans by a pixel
        # or two and would otherwise dominate the small mask left beside the English lettering.
        core = cv2.erode(interior.astype(np.uint8), _kernel(CORE_ERODE)).astype(bool)
        return core if core.any() else interior.astype(bool)
    em = block.em_px or block.style.text_height_px
    boxes = [m.bbox for m in block.members] + [f.bbox for f in block.furigana]
    return boxes_mask((h, w), boxes, int(round(GT_BOX_EM * em)))


def block_interior(gray: np.ndarray, block: TextBlock) -> Optional[np.ndarray]:
    """Bubble interior (page-sized bool) for a bubble block, else None."""
    st = block.style
    if not (st.in_bubble and block.bubble is not None):
        return None
    boxes = [m.bbox for m in block.members] + [f.bbox for f in block.furigana]
    return paper_component(gray, boxes, block.members[0].bbox, block_window(block, gray.shape), _luma(st.bg) < 128.0)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """``(x1, y1, x2, y2)`` of a mask's True pixels, or None when it is empty."""
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return None
    ys, xs = np.flatnonzero(rows), np.flatnonzero(cols)
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _cotenant_groups(interiors: Sequence[Optional[np.ndarray]]) -> List[List[int]]:
    """Block indices grouped by the balloon they share: two blocks are
    co-tenants when their reference interiors overlap.

    Overlap, not rect equality.  ``render/layout._split_shared_bubbles`` used
    to give co-tenants an identical ``TextBlock.bubble``, so keying on that
    rect worked; ``render/bubbles`` replaced it and every block now carries
    its own rect, which no other block's ever equals.  Keying on the rect
    after that change silently stopped detecting co-tenancy and scored a
    block's half of a balloon against the whole balloon's centre.
    """
    idx = [i for i, m in enumerate(interiors) if m is not None and m.any()]
    boxes = {i: _mask_bbox(interiors[i]) for i in idx}
    parent = {i: i for i in idx}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pos, a in enumerate(idx):
        for b in idx[pos + 1:]:
            if find(a) == find(b) or not _boxes_touch(boxes[a], boxes[b]):
                continue
            if np.any(interiors[a] & interiors[b]):
                parent[find(b)] = find(a)
    groups: Dict[int, List[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _boxes_touch(a: Optional[Tuple[int, int, int, int]], b: Optional[Tuple[int, int, int, int]]) -> bool:
    """Cheap gate before the full mask intersection."""
    if a is None or b is None:
        return False
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def bubble_ownership(blocks: Sequence[TextBlock], interiors: Sequence[Optional[np.ndarray]]) -> List[Optional[np.ndarray]]:
    """Per block, the part of its bubble interior that is its own: the whole
    interior, or - for blocks sharing one bubble (joined bubbles, one
    utterance read as two blocks) - the pixels nearer to its source box than
    to any co-tenant's.  ``render/bubbles`` now cuts a joined component into
    one interior per block up front, so co-tenancy is the exception and this
    is a no-op on pages where the cut succeeded.  None for free text."""
    out: List[Optional[np.ndarray]] = [None] * len(blocks)
    for idxs in _cotenant_groups(interiors):
        if len(idxs) == 1:
            out[idxs[0]] = interiors[idxs[0]]
            continue
        union = np.zeros_like(interiors[idxs[0]])
        for i in idxs:
            union |= interiors[i]
        h, w = union.shape
        dists = []
        for i in idxs:
            src = blocks[i].segment.bbox.clamp(w, h)
            inv = np.full((h, w), 255, np.uint8)
            if src.w > 0 and src.h > 0:
                inv[src.y: src.y2, src.x: src.x2] = 0
            dists.append(cv2.distanceTransform(inv, cv2.DIST_L2, 3))
        owner = np.argmin(np.stack(dists), axis=0)
        for k, i in enumerate(idxs):
            out[i] = union & (owner == k)
    return out


def _transform_segments(segments: Sequence[Segment], homography: np.ndarray) -> List[Segment]:
    out = []
    for s in segments:
        quad = cv2.perspectiveTransform(np.asarray(s.quad, dtype=np.float64).reshape(-1, 1, 2), homography).reshape(-1, 2)
        out.append(Segment(s.text, quad.astype(np.float32), s.confidence, s.lang_hint))
    return out


def load_english_segments(eng_bgr: np.ndarray, stem: str, refresh: bool, device: str) -> List[Segment]:
    """OCR lines of the reference scan (its own pixel grid), cached per stem."""
    path = TD.cache_dir(stem) / "eng_segments.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as fh:
            return pickle.load(fh)
    from glasstranslate.ocr import RapidOCREngine

    ocr = RapidOCREngine(device=device, min_confidence=ENG_MIN_CONFIDENCE)
    ocr.warmup()
    segs = ocr.recognize(eng_bgr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(segs, fh)
    log.info("English OCR: %d lines cached to %s", len(segs), path)
    return segs


def assign_english(blocks: Sequence[TextBlock], interiors: Sequence[Optional[np.ndarray]], eng_segments: Sequence[Segment], shape: Tuple[int, ...]) -> Tuple[List[List[Segment]], List[Segment]]:
    """Give every English line to the block whose (own part of the) bubble
    holds its centre, else to the nearest free-text block within
    ``ASSIGN_EM`` ems of its source box.  ``interiors`` may already be the
    per-block ownership masks of :func:`bubble_ownership`."""
    h, w = int(shape[0]), int(shape[1])
    owned: List[List[Segment]] = [[] for _ in blocks]
    unassigned: List[Segment] = []
    for seg in eng_segments:
        bb = seg.bbox
        cx, cy = bb.x + bb.w / 2.0, bb.y + bb.h / 2.0
        ix, iy = min(w - 1, max(0, int(cx))), min(h - 1, max(0, int(cy)))
        chosen = None
        for i, interior in enumerate(interiors):
            if interior is not None and interior[iy, ix]:
                chosen = i
                break
        if chosen is None:
            best = None
            for i, b in enumerate(blocks):
                if b.style.in_bubble:
                    continue
                em = b.em_px or b.style.text_height_px
                s = b.segment.bbox
                dx = max(s.x - cx, cx - s.x2, 0.0)
                dy = max(s.y - cy, cy - s.y2, 0.0)
                d = float(np.hypot(dx, dy)) / max(em, 1e-6)
                if d <= ASSIGN_EM and (best is None or d < best[0]):
                    best = (d, i)
            chosen = best[1] if best is not None else None
        if chosen is None:
            unassigned.append(seg)
        else:
            owned[chosen].append(seg)
    for lines in owned:
        lines.sort(key=lambda s: (s.bbox.y + s.bbox.h / 2.0, s.bbox.x))
    return owned, unassigned


def _seg_record(seg: Segment) -> Dict[str, Any]:
    b = seg.bbox
    return {"text": seg.text, "bbox": [b.x, b.y, b.w, b.h], "confidence": round(float(seg.confidence), 3)}


def derive(image: Path, *, out_root: Path = TD.REFERENCE_ROOT, device: str = "auto", refresh_ocr: bool = False,
           pairs: Optional[Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]] = None, crops: bool = False,
           min_confidence: float = 0.5) -> Optional[Path]:
    """Align, derive and write the ground truth of one page; returns its directory."""
    ref = TD.reference_image_for(image)
    if ref is None:
        print(f"{image}: no reference image", file=sys.stderr)
        return None
    stem = TD.page_stem(image)
    ja = cv2.imread(str(image))
    eng = cv2.imread(str(ref))
    if ja is None or eng is None:
        print(f"cannot read {image} or {ref}", file=sys.stderr)
        return None
    ja_gray = cv2.cvtColor(ja, cv2.COLOR_BGR2GRAY)
    eng_gray = cv2.cvtColor(eng, cv2.COLOR_BGR2GRAY)
    out_dir = out_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    align = align_pages(ja_gray, eng_gray, pairs)
    aligned = warp_reference(eng, align.homography, ja.shape)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out_dir / TD.ALIGNED_REFERENCE_NAME), aligned)
    (out_dir / "align.json").write_text(json.dumps({
        "method": align.method, "score": round(align.score, 4), "note": align.note,
        "homography": align.homography.tolist(), "ja_shape": list(ja.shape[:2]), "eng_shape": list(eng.shape[:2]),
        "pairs": [list(map(list, p)) for p in pairs] if pairs else [],
    }, indent=1), encoding="utf-8")
    print(f"== {stem}: aligned by {align.method} (score {align.score:.3f}; {align.note})")

    all_segments = TD.load_segments(ja, stem, False, device)
    segments = [s for s in all_segments if s.confidence >= min_confidence]
    blocks = build_blocks(ja, segments, all_segments)
    eng_segments = _transform_segments(load_english_segments(eng, stem, refresh_ocr, device), align.homography)

    h, w = ja_gray.shape
    near = np.zeros((h, w), bool)
    dark = np.zeros((h, w), bool)
    windows = [block_window(b, ja.shape) for b in blocks]
    interiors = [block_interior(ja_gray, b) for b in blocks]
    regions = [gt_region_mask(b, interior, ja.shape) for b, interior in zip(blocks, interiors)]
    for b, win, region in zip(blocks, windows, regions):
        near |= region
        if _luma(b.style.bg) < 128.0:
            dark[win.y: win.y2, win.x: win.x2] = True
    gt = erase_ground_truth(ja_gray, aligned_gray, near, [s.bbox for s in eng_segments], dark)
    cv2.imwrite(str(out_dir / "erase_gt.png"), gt.erase.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "kept_art.png"), gt.kept_art.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "english_ink.png"), gt.english.astype(np.uint8) * 255)

    ownership = bubble_ownership(blocks, interiors)
    owned, unassigned = assign_english(blocks, ownership, eng_segments, ja.shape)
    records: List[Dict[str, Any]] = []
    uncertain: List[str] = []
    for i, (b, win, interior, lines, region) in enumerate(zip(blocks, windows, interiors, owned, regions)):
        window_mask = np.zeros((h, w), bool)
        window_mask[win.y: win.y2, win.x: win.x2] = True
        english_ink = gt.english & window_mask
        if lines:
            # Only the ink of this block's own lines: neighbours' lettering may share the window.
            english_ink &= boxes_mask((h, w), [s.bbox for s in lines], ENGLISH_BOX_PAD)
        else:
            english_ink[:] = False
        stats = lettering_stats(english_ink, interior, b.em_px or b.style.text_height_px, ownership[i])
        text = " ".join(s.text.strip() for s in lines if s.text.strip())
        if not lines or min(s.confidence for s in lines) < 0.6:
            uncertain.append(b.segment.text)
        gt_removed = int((gt.erase & region).sum())
        kept = int((gt.kept_art & region).sum())
        records.append({
            "index": i, "kind": TD.block_kind(b), "text": b.segment.text, "glyph_px": float(b.style.text_height_px),
            "em_px": float(b.em_px), "dark": bool(_luma(b.style.bg) < 128.0),
            "source_bbox": [b.segment.bbox.x, b.segment.bbox.y, b.segment.bbox.w, b.segment.bbox.h],
            "bubble": None if b.bubble is None else [b.bubble.x, b.bubble.y, b.bubble.w, b.bubble.h],
            "window": [win.x, win.y, win.w, win.h], "interior_px": int(interior.sum()) if interior is not None else 0,
            "shared_bubble": bool(ownership[i] is not None and interior is not None and ownership[i].sum() < interior.sum()),
            "gt_region_px": int(region.sum()), "erase_gt_px": gt_removed, "kept_art_px": kept,
            "english_text": text, "english_lines": [_seg_record(s) for s in lines], "stats": stats,
        })
        cap = stats["cap_em"]
        inset = stats["inset_min_em"]
        print(f"  {i:2d} {TD.block_kind(b):6} lines {len(lines):2d} cap {'-' if cap is None else round(cap, 2)!s:>5} "
              f"inset {'-' if inset is None else round(inset, 2)!s:>5}  {text[:60]!r}")
    (out_dir / "blocks.json").write_text(json.dumps({
        "stem": stem, "image": str(image), "reference": str(ref), "align": {"method": align.method, "score": round(align.score, 4)},
        "blocks": records, "unassigned_english": [_seg_record(s) for s in unassigned],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    ref_text = TD.reference_text_path(stem)
    if not ref_text.exists():
        data: Dict[str, Any] = {
            "_comment": f"English lettering of {ref.name} keyed by the OCR block text of {image.name}; bootstrapped by "
                        "demo/typeset_reference.py from OCR of the reference page - correct by hand from the --crops sheets. "
                        "Blocks left empty keep the machine translation.",
            "_uncertain": uncertain,
        }
        for r in records:
            data[r["text"]] = r["english_text"]
        ref_text.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  bootstrapped {ref_text.name} ({len(records)} blocks, {len(uncertain)} uncertain)")
    if crops:
        write_review_sheets(stem, aligned_gray, records, TD.PROJECT_ROOT / "demo" / "output" / "reference" / stem)
    print(f"  wrote {out_dir} (eng_aligned, erase_gt, kept_art, english_ink, blocks.json); {len(unassigned)} English lines unassigned")
    return out_dir


def write_review_sheets(stem: str, aligned_gray: np.ndarray, records: Sequence[Dict[str, Any]], out_dir: Path) -> List[Path]:
    """Greyscale contact sheets of every block's English lettering region (from
    the aligned reference), block index drawn top-left, at most
    ``SHEET_MAX_HEIGHT`` tall each, so the transcription can be checked."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles: List[Image.Image] = []
    for r in records:
        x, y, w, h = r["window"]
        crop = aligned_gray[y: y + h, x: x + w]
        if crop.size == 0:
            continue
        scale = CROP_WIDTH / max(1, crop.shape[1])
        tile = cv2.resize(crop, (CROP_WIDTH, max(8, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(tile, 14, 2, 0, 0, cv2.BORDER_CONSTANT, value=90)
        cv2.putText(tile, f"{r['index']} {r['kind']}", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA)
        tiles.append(Image.fromarray(tile))
    paths: List[Path] = []
    sheet: List[Image.Image] = []
    height = 0
    n = 0

    def flush() -> None:
        nonlocal sheet, height, n
        if not sheet:
            return
        img = Image.new("L", (CROP_WIDTH, height), 90)
        y = 0
        for t in sheet:
            img.paste(t, (0, y))
            y += t.height
        path = out_dir / f"{stem}_english_{n}.jpg"
        img.save(path, quality=70)
        paths.append(path)
        sheet, height, n = [], 0, n + 1

    for t in tiles:
        if height + t.height > SHEET_MAX_HEIGHT and sheet:
            flush()
        sheet.append(t)
        height += t.height
    flush()
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", type=Path, default=TD.DEFAULT_IMAGE)
    parser.add_argument("--batch", action="store_true", help="every reference pair (typeset_dev.all_pairs)")
    parser.add_argument("--out", type=Path, default=TD.REFERENCE_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--ocr", action="store_true", help="re-run the English OCR instead of using the cache")
    parser.add_argument("--pairs", default="", help='hand-picked correspondences "xj,yj=xe,ye;..." (page=reference)')
    parser.add_argument("--crops", action="store_true", help="write greyscale review sheets of the English lettering")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    pairs = parse_pairs(args.pairs) if args.pairs else None
    images = TD.all_pairs() if args.batch else [args.image]
    if args.batch and pairs:
        print("--pairs applies to one page only", file=sys.stderr)
        return 2
    failed = 0
    for image in images:
        if derive(image, out_root=args.out, device=args.device, refresh_ocr=args.ocr, pairs=pairs, crops=args.crops,
                  min_confidence=args.min_confidence) is None:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
