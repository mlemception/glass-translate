"""A still image standing in for the screen (the ``feedPage`` smoke action).

The pipeline is unchanged: it asks its capture backend for the glass region
once per tick and gets the page back.  The image is served at its **native**
resolution, never scaled to the region, so OCR, typesetting and the quality
renderer see exactly the pixels the page file holds; only the frame origin
follows the region, so the overlay's coordinates line up with the glass.

Reading the file is the constructor's job: an unreadable path must fail where
the caller can report it, not inside a pipeline pass.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from ..core.interfaces import ScreenCapture
from ..core.types import Frame, Rect

__all__ = ["StaticPageCapture"]


class StaticPageCapture(ScreenCapture):
    """Serves one image file as every frame.

    Args:
        path: image readable by OpenCV (BGR).

    Raises:
        ValueError: the file is missing or not a decodable image.
    """

    name = "static"

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        # Decoded from memory, not with ``cv2.imread``: OpenCV opens the file through the
        # ANSI code page on Windows and fails on any non-ASCII character in the path.
        try:
            data = np.frombuffer(self.path.read_bytes(), np.uint8)
        except OSError as exc:
            raise ValueError(f"cannot read the static page image {self.path}: {exc}") from exc
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if image is None or image.size == 0:
            raise ValueError(f"cannot decode the static page image {self.path}")
        self._image = np.ascontiguousarray(image, dtype=np.uint8)

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the page in pixels."""
        return int(self._image.shape[1]), int(self._image.shape[0])

    def grab(self, region: Rect) -> Optional[Frame]:
        """The whole page, placed at ``region``'s origin.

        A copy per tick: a stage that wrote into the frame would otherwise
        corrupt every later pass, and the change detector still sees identical
        pixels, so the page is OCRed once and the ticks after it are cheap.
        """
        return Frame(self._image.copy(), region.x, region.y, time.perf_counter())
