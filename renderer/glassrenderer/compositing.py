"""Compositing and panel bucketing for the sidecar.

:func:`composite` is the safety net of the whole feature: whatever a stage
produces, **every pixel where the mask is 0 comes back byte-identical to the
source**.  Inside the mask the new content is blended under an alpha that
ramps up from the mask edge *inwards* over ``feather_px`` pixels (the distance
transform of the mask), so a fill never shows a hard seam and never bleeds a
single pixel outside the erase set.  ``feather_px = 0`` is a hard composite.

The bucketing helpers map a panel onto a long side of 512 / 768 / 1024 px
(never upscaling the source) so the diffusion stages see a handful of shapes
instead of one per panel, and map the result back to the source size.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

#: The long sides the panels are bucketed onto.
BUCKETS: Tuple[int, ...] = (512, 768, 1024)
#: Every bucketed side is a multiple of this (the UNet's stride).
SIZE_MULTIPLE = 8


# --- compositing -------------------------------------------------------------


def feather_alpha(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Alpha for ``mask`` (255 = fill), ramping inwards over ``feather_px``.

    The ramp lives *inside* the mask: alpha is exactly 0 wherever the mask is
    0, so the feather can never touch a kept pixel.  A mask that runs off the
    image is not feathered against the image border (there is no zero pixel
    outside to ramp from), which is what a panel-edge erase set wants.
    """
    binary = (mask > 0).astype(np.uint8)
    if feather_px <= 0 or not binary.any():
        return binary.astype(np.float32)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    alpha = np.clip(distance / float(feather_px), 0.0, 1.0)
    return alpha.astype(np.float32, copy=False)


def composite(
    src_rgb: np.ndarray, out_rgb: np.ndarray, mask: np.ndarray, feather_px: int
) -> np.ndarray:
    """Blend ``out_rgb`` into ``src_rgb`` under the feathered ``mask``.

    Returns a fresh uint8 ``HxWx3`` array.  Pixels with ``mask == 0`` are
    copied from ``src_rgb`` byte for byte, whatever the arithmetic below
    rounds to.
    """
    if src_rgb.shape != out_rgb.shape or mask.shape != src_rgb.shape[:2]:
        raise ValueError(
            "image, output and mask must be the same size "
            f"({src_rgb.shape}, {out_rgb.shape}, {mask.shape})"
        )
    keep = mask == 0
    if keep.all():
        return src_rgb.copy()
    if feather_px <= 0:  # a hard composite needs no arithmetic at all
        result = src_rgb.copy()
        result[~keep] = out_rgb[~keep]
        return result
    alpha = feather_alpha(mask, feather_px)[:, :, None]
    blended = src_rgb.astype(np.float32) * (1.0 - alpha) + out_rgb.astype(np.float32) * alpha
    result = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    result[keep] = src_rgb[keep]  # the hard guarantee, independent of rounding
    return result


# --- bucketing ---------------------------------------------------------------


def _floor_multiple(value: float) -> int:
    """Largest multiple of :data:`SIZE_MULTIPLE` at or below ``value`` (min 8).

    The epsilon absorbs the float error of ``side * (bucket / side)``, which
    lands a hair under an exact multiple and would otherwise drop 8 px.
    """
    return max(SIZE_MULTIPLE, int(value + 1e-6) // SIZE_MULTIPLE * SIZE_MULTIPLE)


def bucket_size(height: int, width: int) -> Tuple[int, int]:
    """Bucketed ``(height, width)`` for a panel, never larger than the source.

    The long side lands on the largest bucket that fits it; a panel shorter
    than the smallest bucket keeps its own long side.  Both sides are floored
    to a multiple of :data:`SIZE_MULTIPLE` (at least one multiple), so the
    aspect ratio moves by less than 8 px and nothing is ever upscaled.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"panel must not be empty, got {height}x{width}")
    source_long = max(height, width)
    fitting = [bucket for bucket in BUCKETS if bucket <= source_long]
    target_long = max(fitting) if fitting else _floor_multiple(source_long)
    scale = target_long / float(source_long)
    if height >= width:
        return target_long, _floor_multiple(width * scale)
    return _floor_multiple(height * scale), target_long


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resample an RGB panel to ``size`` = ``(height, width)``."""
    height, width = int(size[0]), int(size[1])
    if image.shape[:2] == (height, width):
        return image
    shrinking = height * width < image.shape[0] * image.shape[1]
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LANCZOS4
    return cv2.resize(image, (width, height), interpolation=interpolation)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resample a mask to ``size``, keeping it strictly 0 / 255."""
    height, width = int(size[0]), int(size[1])
    if mask.shape[:2] == (height, width):
        return mask
    resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(resized > 0, 255, 0).astype(np.uint8)


def pad_to_multiple(
    image: np.ndarray, multiple: int = SIZE_MULTIPLE
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Edge-pad ``image`` right and bottom to a multiple of ``multiple``.

    Returns the padded array and the source ``(height, width)`` to hand to
    :func:`crop_to`.  The array is returned untouched when it already fits.
    """
    height, width = image.shape[:2]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if not pad_h and not pad_w:
        return image, (height, width)
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    return padded, (height, width)


def crop_to(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`pad_to_multiple`: crop back to ``(height, width)``."""
    height, width = int(size[0]), int(size[1])
    if image.shape[0] < height or image.shape[1] < width:
        raise ValueError(f"cannot crop {image.shape[:2]} up to {(height, width)}")
    return image[:height, :width]


__all__: List[str] = [
    "BUCKETS",
    "SIZE_MULTIPLE",
    "composite",
    "feather_alpha",
    "bucket_size",
    "resize_image",
    "resize_mask",
    "pad_to_multiple",
    "crop_to",
]
