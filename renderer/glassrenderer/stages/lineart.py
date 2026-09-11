"""Line-art control image for the union ControlNet's thin-line channel.

Monochrome manga is already black ink on white paper, so a learned line
detector would only hallucinate over strokes we can read directly: greyscale,
invert, one light contrast cleanup.  The ecosystem convention (and what
``xinsir/controlnet-union-sdxl-1.0`` was trained on) is **white lines on a
black ground**, which is exactly what inverting black ink gives.

The masked region is blanked to pure black *after* extraction and with a few
pixels of dilation, so nothing about the Japanese text — not a stroke, not the
halo around it — conditions the generation.  The control map is always built
from the *original* panel, never from the LaMa result, or the flat fill would
be read as "no lines here" while its seam would be read as one.

Pure OpenCV/NumPy: no torch, so this module is unit-testable on the CPU.
"""
from __future__ import annotations

import cv2
import numpy as np

#: Pixels the erase mask grows by before it is blanked out of the control map.
DILATE_PX = 3
#: Inverted intensities below this are paper grain / light screentone, not ink.
BLACK_LEVEL = 96
#: 255 = fill, in line with the protocol's mask convention.
MASK_ON = 128


def dilate_mask(mask: np.ndarray, dilate_px: int = DILATE_PX) -> np.ndarray:
    """``mask`` grown by ``dilate_px`` with an elliptical kernel (uint8 0/255)."""
    binary = (mask >= MASK_ON).astype(np.uint8) * 255
    if dilate_px <= 0:
        return binary
    size = 2 * int(dilate_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(binary, kernel)


def extract_lines(panel_rgb: np.ndarray, black_level: int = BLACK_LEVEL) -> np.ndarray:
    """White-lines-on-black greyscale map of ``panel_rgb`` (uint8 HxW).

    Inverts the panel so ink becomes bright, then stretches everything above
    ``black_level`` back over the full range.  The stretch is linear rather
    than a hard threshold so anti-aliased stroke edges survive as grey — a
    binarised map makes the ControlNet re-etch hard, jagged outlines.
    """
    grey = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2GRAY) if panel_rgb.ndim == 3 else panel_rgb
    inverted = 255.0 - grey.astype(np.float32)
    level = float(np.clip(black_level, 0, 254))
    stretched = (inverted - level) * (255.0 / (255.0 - level))
    return np.clip(stretched, 0.0, 255.0).astype(np.uint8)


def build_control_image(
    panel_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    dilate_px: int = DILATE_PX,
    black_level: int = BLACK_LEVEL,
) -> np.ndarray:
    """Control image for the thin-line channel: uint8 HxWx3, same size as input.

    ``panel_rgb`` is the untouched panel (uint8 HxWx3 RGB), ``mask`` the erase
    mask (uint8 HxW, 255 = fill).  The result is greyscale replicated over
    three channels — white strokes on black, with the dilated mask region
    blanked so the generator sees a hole rather than the old text.
    """
    if panel_rgb.ndim != 3 or panel_rgb.shape[2] != 3:
        raise ValueError(f"panel must be HxWx3 RGB, got {panel_rgb.shape}")
    if mask.shape[:2] != panel_rgb.shape[:2]:
        raise ValueError(f"mask {mask.shape[:2]} does not match panel {panel_rgb.shape[:2]}")
    lines = extract_lines(panel_rgb, black_level)
    lines[dilate_mask(mask, dilate_px) > 0] = 0
    return cv2.cvtColor(lines, cv2.COLOR_GRAY2RGB)
