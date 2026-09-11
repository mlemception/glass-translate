"""Offscreen Qt checks for the glass overlay's caches: layouts, lettering
layers and patch images survive a pipeline pass that re-emits the same
segment objects, are dropped for new objects, and blitting a cached layer
draws the same pixels as stroking the glyph paths directly."""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from glasstranslate.core.types import Rect, Segment, SegmentStyle, StyledSegment, TranslatedSegment  # noqa: E402
from glasstranslate.render.typeset import PlacedLine, Typeset  # noqa: E402
from glasstranslate.ui import overlay as O  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication(["test"])


def _block_segment(text: str, x: int) -> TranslatedSegment:
    quad = np.array([[x, 10], [x + 30, 10], [x + 30, 120], [x, 120]], dtype=np.float32)
    patch = np.full((120, 60, 3), 255, np.uint8)
    style = SegmentStyle(
        fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, text_height_px=30.0, vertical=True,
        layout_box=Rect(x - 40, 0, 120, 140), layout_seed=Rect(x, 10, 30, 110), max_font_px=20.0,
        clean_patch=patch, clean_rect=Rect(x - 5, 5, 60, 120), outline=True,
        source_quads=quad[None], search_box=Rect(x - 40, 0, 120, 140),
        ink_map=np.zeros((140, 120), np.uint8), blocked_map=np.zeros((140, 120), bool),
    )
    return TranslatedSegment(StyledSegment(Segment("テスト", quad, 0.9), style, "ja"), text, "en")


def _image_of(paint) -> np.ndarray:
    img = QImage(400, 300, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(128, 128, 128))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    paint(p)
    p.end()
    ptr = img.constBits()
    return np.frombuffer(ptr, np.uint8).reshape(300, img.bytesPerLine() // 4, 4)[:, :400, :3].copy()


def test_caches_survive_identical_segments_and_drop_new_ones(app):
    glass = O.GlassOverlay()
    a, b = _block_segment("HELLO THERE FRIEND", 60), _block_segment("ANOTHER ONE", 200)
    glass._apply_segments([a, b])
    assert set(glass._patches) == {0, 1}
    patch_a = glass._patches[0]
    # Painting fills the typeset / layer caches.
    _image_of(lambda p: [glass._paint_block(p, seg, 1.0, i) for i, seg in enumerate(glass._segments)])
    assert set(glass._typesets) == {0, 1} and set(glass._layers) == {0, 1}
    ts_a, layer_a = glass._typesets[0], glass._layers[0]
    # The same objects re-emitted (a "nothing changed" pass): everything is kept, re-indexed.
    glass._apply_segments([b, a])
    assert glass._typesets[1] is ts_a and glass._layers[1] is layer_a and glass._patches[1] is patch_a
    assert 0 in glass._typesets and 0 in glass._layers  # b's entries moved to index 0
    # A new object for the same block is laid out afresh.
    a2 = _block_segment("HELLO THERE FRIEND", 60)
    glass._apply_segments([a2, b])
    assert 0 not in glass._typesets and 0 not in glass._layers
    assert glass._patches[0] is not patch_a
    assert glass._typesets[1] is not None and glass._layers[1] is not None
    # Case / font changes invalidate layouts and layers but not the patches.
    glass.set_uppercase(False)
    assert glass._typesets == {} and glass._layers == {} and set(glass._patches) == {0, 1}


def test_an_upgraded_patch_is_reconverted_and_the_lettering_is_kept(app):
    """The quality renderer swaps ``clean_patch`` in place and bumps ``clean_patch_serial``;
    the patch image must be rebuilt while the typeset and its layer survive."""
    glass = O.GlassOverlay()
    seg = _block_segment("HELLO THERE FRIEND", 60)
    glass._apply_segments([seg])
    _image_of(lambda p: glass._paint_block(p, glass._segments[0], 1.0, 0))
    patch, ts, layer = glass._patches[0], glass._typesets[0], glass._layers[0]
    # Same object, same serial: everything is reused.
    glass._apply_segments([seg])
    assert glass._patches[0] is patch and glass._typesets[0] is ts and glass._layers[0] is layer
    # A sidecar result: the patch pixels changed in place and the serial went up.
    seg.style.clean_patch = np.full_like(seg.style.clean_patch, 17)
    seg.style.clean_patch_serial += 1
    glass._apply_segments([seg])
    assert glass._patches[0] is not patch
    assert glass._typesets[0] is ts and glass._layers[0] is layer  # the lettering is untouched
    assert QColor(glass._patches[0].pixel(5, 5)).blue() == 17


def test_patch_counters_count_conversions_and_upgrades(app):
    """``patch_stats`` feeds the smoke report: how often a clean patch was converted to a
    QImage and how often that replaced a cached image of the same block (a sidecar upgrade)."""
    glass = O.GlassOverlay()
    assert glass.patch_stats == {"conversions": 0, "upgrades": 0}
    a, b = _block_segment("HELLO THERE FRIEND", 60), _block_segment("ANOTHER ONE", 200)
    glass._apply_segments([a, b])
    assert glass.patch_stats == {"conversions": 2, "upgrades": 0}
    # A pass that changed nothing re-uses both images.
    glass._apply_segments([a, b])
    assert glass.patch_stats == {"conversions": 2, "upgrades": 0}
    # A sidecar result for one block: one re-conversion of an already cached segment.
    a.style.clean_patch = np.full_like(a.style.clean_patch, 9)
    a.style.clean_patch_serial += 1
    glass._apply_segments([a, b])
    assert glass.patch_stats == {"conversions": 3, "upgrades": 1}
    # A brand new segment object is converted, but replacing nothing is not an upgrade.
    glass._apply_segments([_block_segment("HELLO THERE FRIEND", 60), b])
    assert glass.patch_stats == {"conversions": 4, "upgrades": 1}


def test_layer_blit_matches_direct_stroking(app):
    glass = O.GlassOverlay()
    seg = _block_segment("HELLO THERE FRIEND", 60)
    style = seg.style
    family = glass._block_family
    font = O.make_font(family, 20.0, italic=True)
    ts = Typeset(20.0, 17.0, [PlacedLine("HELLO", 100.0, 40.0, 60.0), PlacedLine("THERE", 100.0, 57.0, 62.0)], True, 0, 23.0)
    direct = _image_of(lambda p: glass._draw_typeset(p, ts, font, style))
    image, x0, y0 = glass._render_layer(ts, font, style)
    blit = _image_of(lambda p: p.drawImage(QPointF(x0, y0), image))
    # Same pixels (both are the same paths rasterised at the same sub-pixel offsets).
    assert np.abs(direct.astype(int) - blit.astype(int)).max() <= 1
    # The layer is small and covers the lettering with its halo.
    assert image.width() < 200 and image.height() < 100
    ys, xs = np.nonzero(np.any(direct != 128, axis=2))
    assert x0 <= xs.min() and y0 <= ys.min() and x0 + image.width() > xs.max() and y0 + image.height() > ys.max()
