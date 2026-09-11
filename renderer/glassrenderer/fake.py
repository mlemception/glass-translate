"""Numpy stand-ins for the two GPU stages.

``--fake`` mode exists so the protocol, the pipeline and the compositing can
be exercised on any machine - CI, a laptop, the main torch-free venv - with
the same code path the real stages take.  Nothing here imports torch.

* :class:`FakeLama` fills each connected component of the mask with the median
  colour of the 6 px ring around it, which is roughly what a structural
  inpainter does to a small erase set on flat paper and is enough to tell
  "the mask was filled" from "the mask was not".
* :class:`FakeSdxl` returns its input: the refinement pass is a no-op.

Both match the ``Stage`` protocol of ``renderer/PROTOCOL.md``: ``name`` plus
``run(image_rgb, mask, params) -> image_rgb``.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Tuple

import cv2
import numpy as np

#: Width of the ring the fill colour is sampled from.
RING_PX = 6
#: Fallback when a component has no unmasked neighbourhood at all.
NEUTRAL_GREY = 127


def _ring_colour(
    image_rgb: np.ndarray, component: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """Median colour of the ``RING_PX`` ring around ``component``, on kept pixels."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1)
    )
    grown = cv2.dilate(component, kernel)
    ring = np.logical_and(grown > 0, keep)
    if not ring.any():
        ring = keep  # the mask swallowed its own neighbourhood
    if not ring.any():
        return np.full(3, NEUTRAL_GREY, np.uint8)
    return np.median(image_rgb[ring], axis=0).astype(np.uint8)


class FakeLama:
    """Structural-inpainter stand-in: a flat ring-median fill per component."""

    name = "lama"

    def run(
        self, image_rgb: np.ndarray, mask: np.ndarray, params: Mapping[str, Any]
    ) -> np.ndarray:
        """Fill every connected mask component with its surrounding median."""
        del params  # the stand-in has nothing to tune
        binary = (mask > 0).astype(np.uint8)
        if not binary.any():
            return image_rgb
        keep = binary == 0
        count, labels = cv2.connectedComponents(binary, connectivity=8)
        result = image_rgb.copy()
        for label in range(1, count):
            component = (labels == label).astype(np.uint8)
            result[component > 0] = _ring_colour(image_rgb, component, keep)
        return result


class FakeSdxl:
    """Refinement-pass stand-in: the identity."""

    name = "sdxl"

    def run(
        self, image_rgb: np.ndarray, mask: np.ndarray, params: Mapping[str, Any]
    ) -> np.ndarray:
        """Return the input unchanged."""
        del mask, params
        return image_rgb


def build_fake_stages() -> Tuple[FakeLama, FakeSdxl]:
    """The ``(lama, sdxl)`` pair :mod:`glassrenderer.pipeline` uses in fake mode."""
    return FakeLama(), FakeSdxl()


__all__: List[str] = ["RING_PX", "FakeLama", "FakeSdxl", "build_fake_stages"]
