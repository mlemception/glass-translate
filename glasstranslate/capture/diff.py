"""Tile-based frame change detection.

The pipeline grabs the glass region every tick; :class:`ChangeDetector` tells
it which parts of the frame actually changed so OCR can be restricted to those
rectangles (or skipped entirely).

Algorithm
---------
1. Convert the frame to grayscale and downsample it 2x with ``INTER_AREA``
   (cheap, and averages out single-pixel noise).
2. ``absdiff`` against the previous downsampled frame and threshold at
   ``min_pixel_delta`` to get a binary "changed pixel" mask.
3. Average the mask over the tile grid with ``cv2.resize(INTER_AREA)``; each
   cell is the fraction of changed pixels in that tile.  Partial tiles on the
   right/bottom edges are re-weighted so the fraction is over their real area.
4. Tiles whose fraction exceeds ``threshold`` are dirty.  Connected groups of
   dirty tiles (8-connectivity) are merged into their bounding rectangles.

Everything is done with OpenCV / numpy primitives on the whole image at once;
a 1920x1080 frame takes ~1.5 ms.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from glasstranslate.core.types import Rect

__all__ = ["ChangeDetector"]

_DOWNSAMPLE = 2  # grayscale is compared at 1/2 resolution


class ChangeDetector:
    """Detects which tiles of a video-like frame stream changed between calls.

    Attributes:
        fraction_changed: fraction of frame pixels (at the compared resolution)
            whose grayscale value moved by more than ``min_pixel_delta`` on the
            last :meth:`update` call.  ``1.0`` after a first frame / size change,
            ``0.0`` before the first call or after :meth:`reset`.  The pipeline
            uses it to detect scroll/video storms.
    """

    def __init__(self, tile: int = 64, threshold: float = 0.02, min_pixel_delta: int = 24) -> None:
        """
        Args:
            tile: tile edge length in *frame* pixels (rounded down to an even
                number, minimum 2, so tiles map exactly onto the 2x-downsampled
                comparison image).
            threshold: a tile is dirty when more than this fraction of its
                pixels changed.
            min_pixel_delta: minimum grayscale difference (0-255) for a pixel to
                count as changed.
        """
        if tile < _DOWNSAMPLE:
            raise ValueError(f"tile must be >= {_DOWNSAMPLE}, got {tile}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if not 0 <= min_pixel_delta <= 255:
            raise ValueError(f"min_pixel_delta must be in [0, 255], got {min_pixel_delta}")
        self.tile = (int(tile) // _DOWNSAMPLE) * _DOWNSAMPLE
        self.threshold = float(threshold)
        self.min_pixel_delta = int(min_pixel_delta)
        self.fraction_changed: float = 0.0
        self._prev: Optional[np.ndarray] = None  # downsampled grayscale of the previous frame
        self._prev_shape: Optional[tuple[int, int]] = None  # (h, w) of the previous full frame
        self._weights: Optional[np.ndarray] = None  # per-tile area correction, (gy, gx) float32

    # ------------------------------------------------------------------ api
    def reset(self) -> None:
        """Forget the previous frame; the next :meth:`update` marks everything dirty."""
        self._prev = None
        self._prev_shape = None
        self._weights = None
        self.fraction_changed = 0.0

    def update(self, frame_bgr: np.ndarray) -> List[Rect]:
        """Compare ``frame_bgr`` with the previous frame and return dirty rectangles.

        Args:
            frame_bgr: ``HxWx3`` uint8 BGR image (an ``HxW`` grayscale image is
                accepted too).

        Returns:
            Merged dirty rectangles in frame pixel coordinates, or an empty list
            when nothing changed.  The whole frame on the first call and
            whenever the frame size changes.
        """
        gray = _to_gray(frame_bgr)
        h, w = gray.shape
        small = cv2.resize(
            gray,
            (_ceil_div(w, _DOWNSAMPLE), _ceil_div(h, _DOWNSAMPLE)),
            interpolation=cv2.INTER_AREA,
        )

        if self._prev is None or self._prev_shape != (h, w):
            self._prev = small
            self._prev_shape = (h, w)
            self._weights = self._tile_weights(small.shape[0], small.shape[1])
            self.fraction_changed = 1.0
            return [Rect(0, 0, w, h)]

        diff = cv2.absdiff(small, self._prev)
        self._prev = small
        _, changed = cv2.threshold(diff, self.min_pixel_delta, 255, cv2.THRESH_BINARY)
        self.fraction_changed = cv2.countNonZero(changed) / changed.size

        grid_frac = self._tile_fractions(changed)
        dirty = (grid_frac > self.threshold).astype(np.uint8)
        if not dirty.any():
            return []
        return self._merge_tiles(dirty, w, h)

    # ------------------------------------------------------------- internals
    @property
    def _small_tile(self) -> int:
        return self.tile // _DOWNSAMPLE

    def _grid_size(self, small_h: int, small_w: int) -> tuple[int, int]:
        """(gy, gx) number of tiles covering a downsampled ``small_h x small_w`` image."""
        t = self._small_tile
        return _ceil_div(small_h, t), _ceil_div(small_w, t)

    def _tile_weights(self, small_h: int, small_w: int) -> np.ndarray:
        """Correction factors so partial edge tiles report the fraction over their
        real pixel area rather than over a full ``tile x tile`` block."""
        t = self._small_tile
        gy, gx = self._grid_size(small_h, small_w)
        rows = np.full(gy, t, np.float32)
        cols = np.full(gx, t, np.float32)
        if small_h % t:
            rows[-1] = small_h % t
        if small_w % t:
            cols[-1] = small_w % t
        return (t * t) / (rows[:, None] * cols[None, :])

    def _tile_fractions(self, changed: np.ndarray) -> np.ndarray:
        """Per-tile fraction of changed pixels, shape (gy, gx), float32 in [0, 1]."""
        t = self._small_tile
        sh, sw = changed.shape
        gy, gx = self._grid_size(sh, sw)
        pad_b, pad_r = gy * t - sh, gx * t - sw
        if pad_b or pad_r:
            changed = cv2.copyMakeBorder(changed, 0, pad_b, 0, pad_r, cv2.BORDER_CONSTANT, value=0)
        # Integer-factor INTER_AREA is an exact box mean: cell = 255 * fraction.
        means = cv2.resize(changed, (gx, gy), interpolation=cv2.INTER_AREA)
        assert self._weights is not None
        return means.astype(np.float32) * (self._weights / 255.0)

    def _merge_tiles(self, dirty: np.ndarray, w: int, h: int) -> List[Rect]:
        """Bounding rectangles (frame pixels) of 8-connected groups of dirty tiles."""
        n, _, stats, _ = cv2.connectedComponentsWithStats(dirty, connectivity=8)
        rects: List[Rect] = []
        for i in range(1, n):  # label 0 is background
            gx0, gy0, gw, gh = (int(v) for v in stats[i, :4])
            x0, y0 = gx0 * self.tile, gy0 * self.tile
            x1, y1 = min((gx0 + gw) * self.tile, w), min((gy0 + gh) * self.tile, h)
            rects.append(Rect(x0, y0, x1 - x0, y1 - y0))
        return rects


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("frame must be a uint8 numpy array")
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"expected HxWx3 BGR or HxW gray image, got shape {frame.shape}")
