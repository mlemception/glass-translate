"""Synthetic tests for the glyph eraser (``glasstranslate.render.erase``)."""
from __future__ import annotations

import cv2
import numpy as np

from glasstranslate.core.types import Rect, Segment, SegmentStyle
from glasstranslate.render import erase
from glasstranslate.render.layout import TextBlock

PAPER = 250
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _quad(r: Rect) -> np.ndarray:
    return np.array([[r.x, r.y], [r.x2, r.y], [r.x2, r.y2], [r.x, r.y2]], np.float32)


def _block(box: Rect, glyph: float, *, bubble: Rect | None = None, furigana: list[Rect] = ()) -> TextBlock:
    seg = Segment("テスト", _quad(box), 0.9)
    style = SegmentStyle(fg=(0, 0, 0), bg=(PAPER, PAPER, PAPER), angle_deg=0.0, text_height_px=glyph, vertical=True,
                         in_bubble=bubble is not None)
    furi = [Segment("ふ", _quad(f), 0.9) for f in furigana]
    return TextBlock(seg, style, [seg], furi, bubble)


def _page(h: int = 400, w: int = 400) -> np.ndarray:
    return np.full((h, w, 3), PAPER, np.uint8)


def _draw_column(img: np.ndarray, x: int, y0: int, n: int, glyph: int, gap: int = 6) -> Rect:
    """``n`` fake glyphs (rectangular strokes) of ``glyph`` px down a column at ``x``; returns the column box."""
    for i in range(n):
        y = y0 + i * (glyph + gap)
        cv2.rectangle(img, (x, y), (x + glyph - 1, y + 4), (0, 0, 0), -1)  # top bar
        cv2.rectangle(img, (x + glyph // 2 - 2, y), (x + glyph // 2 + 2, y + glyph - 1), (0, 0, 0), -1)  # stem
        cv2.rectangle(img, (x, y + glyph - 5), (x + glyph - 1, y + glyph - 1), (0, 0, 0), -1)  # bottom bar
    return Rect(x, y0, glyph, n * (glyph + gap) - gap)


def _gray(patch: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)


def test_glyphs_on_paper_leave_exact_paper_and_no_outline():
    img = _page()
    box = _draw_column(img, 180, 100, 4, 30)
    block = _block(box, 30.0)
    patch, rect, outline = erase.erase_block(img, block)
    assert not outline
    g = _gray(patch)
    # Every pixel of the patch is the paper colour: nothing dark, nothing tinted.
    assert g.min() == PAPER and g.max() == PAPER
    # The patch covers the glyphs with a small margin only.
    assert rect.x <= box.x and rect.y <= box.y and rect.x2 >= box.x2 and rect.y2 >= box.y2
    assert rect.w <= box.w + 2 * (erase.ERASE_DILATE + erase.RECT_PAD + erase.RING) + 2
    assert rect.h <= box.h + 2 * (erase.ERASE_DILATE + erase.RECT_PAD + erase.RING) + 2


def test_art_line_crossing_glyphs_survives_and_continues():
    img = _page()
    box = _draw_column(img, 180, 100, 4, 30)
    # A 2 px diagonal art line running through the whole column.
    cv2.line(img, (100, 60), (300, 300), (0, 0, 0), 2, cv2.LINE_8)
    block = _block(box, 30.0)
    patch, rect, outline = erase.erase_block(img, block)
    assert outline  # ink stays under the text
    g = _gray(patch)
    # Sample the line where it crosses the column: dark pixels must remain
    # close to the ideal line at most steps (the line continues through).
    hits = 0
    total = 0
    for x in range(box.x + 2, box.x2 - 2, 2):
        y = int(round(60 + (x - 100) * 240 / 200))
        if not (box.y < y < box.y2):
            continue
        total += 1
        lx, ly = x - rect.x, y - rect.y
        if g[max(0, ly - 2) : ly + 3, max(0, lx - 2) : lx + 3].min() < 128:
            hits += 1
    assert total > 5 and hits >= 0.8 * total
    # Outside the column (but inside the patch) the line is untouched.
    for x in range(rect.x + 3, box.x - 6, 3):
        y = int(round(60 + (x - 100) * 240 / 200))
        if rect.y + 2 < y < rect.y2 - 2:
            assert g[y - rect.y, x - rect.x] < 128
    # The glyph strokes themselves are gone: away from the line the box is paper.
    dark = (g < 128)
    ys, xs = np.nonzero(dark)
    dist = np.abs((ys + rect.y) - (60 + ((xs + rect.x) - 100) * 240 / 200))
    assert (dist > 6).sum() < 0.02 * dark.size


def test_column_completion_takes_glyph_sized_blob_not_large_one():
    img = _page(500, 400)
    glyph = 30
    box = _draw_column(img, 180, 160, 4, glyph)
    # A missed glyph just above the column (OCR box too short)...
    cv2.rectangle(img, (182, 125), (208, 152), (0, 0, 0), -1)
    # ...and a large art blob further up on the same axis (not text).
    cv2.rectangle(img, (150, 20), (240, 90), (0, 0, 0), -1)
    block = _block(box, float(glyph))
    patch, rect, _ = erase.erase_block(img, block)
    g = _gray(patch)
    assert rect.y <= 125  # the erase reached the missed glyph
    assert g[125 - rect.y : 153 - rect.y, 182 - rect.x : 209 - rect.x].min() == PAPER
    # The large blob is neither erased nor inside the clean rect.
    assert img[20:90, 150:240].max() == 0
    assert rect.y > 90 or g[max(0, 20 - rect.y) : max(0, 90 - rect.y), 150 - rect.x : 240 - rect.x].size == 0


def test_bubble_interior_painted_and_outline_kept():
    img = _page()
    centre, axes = (200, 200), (90, 120)
    cv2.ellipse(img, centre, axes, 0, 0, 360, (0, 0, 0), 3)  # bubble outline
    box = _draw_column(img, 185, 130, 4, 30)
    bubble = Rect(centre[0] - axes[0], centre[1] - axes[1], 2 * axes[0], 2 * axes[1])
    block = _block(box, 30.0, bubble=bubble)
    patch, rect, outline = erase.erase_block(img, block)
    assert not outline
    g = _gray(patch)
    # The text is gone and the interior is paper.
    inner = g[box.y - rect.y : box.y2 - rect.y, box.x - rect.x : box.x2 - rect.x]
    assert inner.min() == PAPER
    # The bubble outline pixels that fall inside the patch are untouched.
    ring = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(ring, centre, axes, 0, 0, 360, 255, 3)
    sub = ring[rect.y : rect.y2, rect.x : rect.x2] > 0
    if sub.any():
        assert g[sub].max() < 128


def test_apply_fills_style_fields_for_every_block():
    img = _page()
    a = _draw_column(img, 100, 60, 3, 24)
    b = _draw_column(img, 260, 60, 3, 24)
    blocks = [_block(a, 24.0), _block(b, 24.0)]
    erase.apply(img, blocks)
    for blk in blocks:
        assert blk.style.clean_patch is not None and blk.style.clean_rect is not None
        assert _gray(blk.style.clean_patch).min() == PAPER
        assert not blk.style.outline
    # One block's patch never covers the other block's glyphs.
    assert not blocks[0].style.clean_rect.intersects(b)
    assert not blocks[1].style.clean_rect.intersects(a)
