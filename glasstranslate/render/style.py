"""Measure the visual style (orientation, size, colours) of an OCR segment.

Colour-space note: every image entering this module is **BGR** (OpenCV
convention, as produced by the capture backends).  Every colour leaving this
module (``SegmentStyle.fg`` / ``SegmentStyle.bg``) is an **RGB** tuple.

Angle sign convention
---------------------
``quad_angle_deg`` is *counter-clockwise positive as seen on screen*.  Screen
space has ``y`` growing downward, so a baseline whose right end is higher on
the screen than its left end has a **positive** angle::

    angle = -degrees(atan2(p1.y - p0.y, p1.x - p0.x))

normalised into ``(-90, 90]``.
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np

from ..core.types import RGB, Rect, Segment, SegmentStyle

_BLACK: RGB = (0, 0, 0)
_WHITE: RGB = (255, 255, 255)

# Upper bound on the number of pixels handed to k-means.
_KMEANS_MAX_SAMPLES = 4000
# How much of the foreground cluster, measured from the end furthest from the
# paper, is the ink rather than its anti-aliased fringe.  See :func:`_ink_core`.
_INK_CORE_PERCENTILE = 25.0
# A quad whose p0->p3 edge is more than this many times longer than its p0->p1
# edge is treated as a vertical (top-to-bottom) text column.
_VERTICAL_ASPECT = 1.5


def _as_quad(quad: np.ndarray) -> np.ndarray:
    """Return ``quad`` as a (4, 2) float64 array (validates the shape)."""
    return np.asarray(quad, dtype=np.float64).reshape(4, 2)


def quad_angle_deg(quad: np.ndarray) -> float:
    """Angle of the text baseline (edge p0 -> p1) in degrees.

    CCW-positive in screen space (y down): a line whose right end is higher on
    screen than its left end returns a positive angle.  The result is
    normalised to ``(-90, 90]`` so a quad and its 180-degree flip agree.
    """
    q = _as_quad(quad)
    dx = float(q[1, 0] - q[0, 0])
    dy = float(q[1, 1] - q[0, 1])
    angle = -math.degrees(math.atan2(dy, dx))
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return angle or 0.0  # collapse -0.0 to 0.0


def quad_text_height(quad: np.ndarray) -> float:
    """Length of the edge p0 -> p3, i.e. the quad's extent across the baseline."""
    q = _as_quad(quad)
    return float(np.hypot(*(q[3] - q[0])))


def quad_text_width(quad: np.ndarray) -> float:
    """Length of the edge p0 -> p1, i.e. the quad's extent along the baseline."""
    q = _as_quad(quad)
    return float(np.hypot(*(q[1] - q[0])))


def is_vertical(quad: np.ndarray, text: str) -> bool:
    """True when the quad is a tall column (``> 1.5x`` taller than wide along
    its own edges) holding at least two characters, i.e. vertical script."""
    if len(text.strip()) < 2:
        return False
    width = quad_text_width(quad)
    height = quad_text_height(quad)
    return width > 0.0 and height > _VERTICAL_ASPECT * width


def _contrast_for(rgb: RGB) -> RGB:
    """Black or white, whichever contrasts more with ``rgb``."""
    r, g, b = rgb
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return _BLACK if luma > 127.5 else _WHITE


def _subsample(pixels: np.ndarray, limit: int) -> np.ndarray:
    """Deterministically pick at most ``limit`` rows (evenly strided)."""
    n = pixels.shape[0]
    if n <= limit:
        return pixels
    idx = np.linspace(0, n - 1, limit).astype(np.intp)
    return pixels[idx]


def _to_rgb_tuple(v: np.ndarray) -> RGB:
    r, g, b = (int(round(float(c))) for c in v[:3])
    return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def extract_colors(img_bgr: np.ndarray, quad: np.ndarray, pad: int = 2) -> Tuple[RGB, RGB]:
    """Estimate ``(fg, bg)`` RGB colours of the text inside ``quad``.

    The axis-aligned bbox of the quad (grown by ``pad``) is cropped and
    clustered into two colours with ``cv2.kmeans`` on an evenly strided
    subsample of at most 4000 pixels.  The cluster that owns more of the
    1-pixel border ring of the crop is the background; the other is the
    foreground.  Each returned colour is the median of its cluster, which is
    robust against anti-aliased edge pixels pulling the mean.

    Fallbacks: an empty crop returns ``(black, white)``; a crop with a single
    colour returns that colour as background and black/white as foreground.
    """
    h, w = img_bgr.shape[:2]
    bbox = Rect.from_quad(_as_quad(quad))
    crop_rect = Rect(bbox.x - pad, bbox.y - pad, bbox.w + 2 * pad, bbox.h + 2 * pad).clamp(w, h)
    if crop_rect.w <= 0 or crop_rect.h <= 0:
        return _BLACK, _WHITE

    crop_bgr = img_bgr[crop_rect.y : crop_rect.y2, crop_rect.x : crop_rect.x2]
    if crop_bgr.ndim == 2:
        crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
    crop = crop_bgr[:, :, :3][:, :, ::-1]  # -> RGB view
    pixels = crop.reshape(-1, 3)

    border_mask = np.zeros(crop.shape[:2], dtype=bool)
    border_mask[0, :] = border_mask[-1, :] = True
    border_mask[:, 0] = border_mask[:, -1] = True
    border_pixels = pixels[border_mask.reshape(-1)]

    samples = _subsample(pixels, _KMEANS_MAX_SAMPLES).astype(np.float32)
    if samples.shape[0] < 2 or np.all(samples == samples[0]):
        bg = _to_rgb_tuple(pixels[0])
        return _contrast_for(bg), bg

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    _, labels, centers = cv2.kmeans(samples, 2, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    # Which cluster does the border ring belong to?  Assign border pixels to
    # the nearest centre and count.
    border_f = border_pixels.astype(np.float32)
    d = np.linalg.norm(border_f[:, None, :] - centers[None, :, :], axis=2)
    border_labels = np.argmin(d, axis=1)
    bg_label = 1 if np.count_nonzero(border_labels == 1) > np.count_nonzero(border_labels == 0) else 0
    fg_label = 1 - bg_label

    fg_members = samples[labels == fg_label]
    bg_members = samples[labels == bg_label]
    bg = _to_rgb_tuple(np.median(bg_members, axis=0) if len(bg_members) else centers[bg_label])
    if len(fg_members) == 0:
        return _contrast_for(bg), bg
    fg = _to_rgb_tuple(np.median(_ink_core(fg_members, bg), axis=0))
    return fg, bg


def _ink_core(fg_members: np.ndarray, bg: RGB) -> np.ndarray:
    """The part of the foreground cluster that is the ink itself.

    A glyph's edge pixels are part ink and part paper, and the smaller the
    type the more of them there are; on fine CJK lettering they outnumber the
    solid centres, so the *median* of the foreground cluster is a blend, not a
    colour anyone printed.  Over the corpus that put 47 of 316 blocks at a
    foreground luma of 90 - 126 - grey lettering where the page is printed in
    black, and just the wrong side of the ``_FG_SNAP_LUMA`` floor that exists
    to catch exactly this.

    Anti-aliasing only ever pulls a pixel *toward* the paper and never past
    it, so the ink's own colour is the end of the distribution furthest from
    the paper.  Taking the quarter of the cluster that lies that way leaves
    the solid centres of the strokes and drops the fringe.  Flat synthetic
    text is unaffected: every member has the same colour, so any quarter of it
    is that colour.
    """
    fg_luma = _luma_of(fg_members)
    if float(np.median(fg_luma)) < float(_luma_of(np.asarray(bg, np.float32))):
        keep = fg_luma <= np.percentile(fg_luma, _INK_CORE_PERCENTILE)
    else:
        keep = fg_luma >= np.percentile(fg_luma, 100.0 - _INK_CORE_PERCENTILE)
    core = fg_members[keep]
    return core if len(core) else fg_members


def _luma_of(rgb: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an RGB array (``..., 3``)."""
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def measure_style(img_bgr: np.ndarray, segment: Segment) -> SegmentStyle:
    """Measure angle, text height, colours and orientation of ``segment``."""
    fg, bg = extract_colors(img_bgr, segment.quad)
    return SegmentStyle(
        fg=fg,
        bg=bg,
        angle_deg=quad_angle_deg(segment.quad),
        text_height_px=quad_text_height(segment.quad),
        vertical=is_vertical(segment.quad, segment.text),
    )
