"""Synthetic tests for the glyph eraser (``glasstranslate.render.erase``)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from glasstranslate.core.types import Rect, Segment, SegmentStyle
from glasstranslate.render import erase
from glasstranslate.render.layout import TextBlock

PAPER = 250
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _quad(r: Rect) -> np.ndarray:
    return np.array([[r.x, r.y], [r.x2, r.y], [r.x2, r.y2], [r.x, r.y2]], np.float32)


def _block(box: Rect, glyph: float, *, bubble: Rect | None = None, furigana: list[Rect] = (),
           text: str = "テスト") -> TextBlock:
    seg = Segment(text, _quad(box), 0.9)
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


# ----------------------------------------------------------------- bubble redraw (2026-09-10)
HATCH = (60, 60, 60)


def _hatched_page(h: int = 400, w: int = 400, paper: int = PAPER) -> np.ndarray:
    """A page whose background is hatched artwork, so anything outside a bubble must survive."""
    img = np.full((h, w, 3), paper, np.uint8)
    for x in range(-h, w, 9):
        cv2.line(img, (x, 0), (x + h, h), HATCH, 1)
    return img


def _bubble_page(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Rect, Rect]:
    """``(page, interior, outline, tail_interior, bubble_rect, text_box)`` for one synthetic
    bubble: the interior is paper (dark paper for ``dark``), the outline is 3 px, a text column
    plus stray furigana-like marks (outside the OCR box) sit inside, the rest of the page is
    hatched art.  ``interior`` and ``outline`` are bool masks of the page."""
    dark = kind == "dark"
    paper, ink = (20, 235) if dark else (PAPER, 0)
    img = _hatched_page(paper=PAPER)
    shape = np.zeros((400, 400), np.uint8)
    c = (200, 200)
    if kind in ("round", "dark"):
        cv2.circle(shape, c, 110, 1, -1)
    elif kind == "oval":
        cv2.ellipse(shape, c, (130, 95), 0, 0, 360, 1, -1)
    elif kind == "joined":
        cv2.ellipse(shape, (150, 200), (90, 80), 0, 0, 360, 1, -1)
        cv2.ellipse(shape, (255, 200), (90, 80), 0, 0, 360, 1, -1)
    elif kind == "jagged":
        pts = []
        for i in range(24):
            r = 120 if i % 2 == 0 else 90
            a = 2 * np.pi * i / 24
            pts.append((int(c[0] + r * np.cos(a)), int(c[1] + r * np.sin(a))))
        cv2.fillPoly(shape, [np.array(pts, np.int32)], 1)
    elif kind == "thought":
        for i in range(14):
            a = 2 * np.pi * i / 14
            cv2.circle(shape, (int(c[0] + 95 * np.cos(a)), int(c[1] + 85 * np.sin(a))), 34, 1, -1)
        cv2.ellipse(shape, c, (95, 85), 0, 0, 360, 1, -1)
    elif kind == "tailed":
        cv2.ellipse(shape, c, (110, 90), 0, 0, 360, 1, -1)
        cv2.fillPoly(shape, [np.array([(240, 275), (285, 370), (205, 288)], np.int32)], 1)
    else:
        raise ValueError(kind)
    interior = shape.astype(bool)
    outline = cv2.dilate(shape, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool) & ~interior
    img[interior] = (paper, paper, paper)
    img[outline] = (ink, ink, ink)
    tail = np.zeros((400, 400), bool)
    if kind == "tailed":
        cv2.fillPoly(tail.view(np.uint8), [np.array([(245, 300), (275, 355), (220, 300)], np.int32)], 1)
    # Text column inside, plus stray marks (furigana, a stray dot) outside its OCR box.
    glyph = 24
    x0, y0 = c[0] - glyph // 2, c[1] - 60
    for i in range(4):
        y = y0 + i * (glyph + 6)
        cv2.rectangle(img, (x0, y), (x0 + glyph - 1, y + 4), (ink, ink, ink), -1)
        cv2.rectangle(img, (x0 + glyph // 2 - 2, y), (x0 + glyph // 2 + 2, y + glyph - 1), (ink, ink, ink), -1)
    for i in range(3):
        cv2.rectangle(img, (x0 + glyph + 6, y0 + 10 + i * 30), (x0 + glyph + 12, y0 + 22 + i * 30), (ink, ink, ink), -1)  # furigana
    cv2.circle(img, (x0 - 30, y0 + 40), 3, (ink, ink, ink), -1)  # stray dot
    ys, xs = np.nonzero(interior)
    bubble = Rect(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    box = Rect(x0, y0, glyph, 4 * (glyph + 6) - 6)
    return img, interior, outline, tail, bubble, box


def _redraw(kind: str):
    img, interior, outline, tail, bubble, box = _bubble_page(kind)
    dark = kind == "dark"
    seg = Segment("テスト", _quad(box), 0.9)
    style = SegmentStyle(fg=(235, 235, 235) if dark else (0, 0, 0), bg=(20, 20, 20) if dark else (PAPER, PAPER, PAPER),
                         angle_deg=0.0, text_height_px=24.0, vertical=True, in_bubble=True)
    block = TextBlock(seg, style, [seg], [], bubble)
    patch, rect, flag = erase.erase_block(img, block)
    result = img.copy()
    result[rect.y: rect.y2, rect.x: rect.x2] = patch
    return img, result, interior, outline, tail, bubble, rect, flag


@pytest.mark.parametrize("kind", ["round", "oval", "joined", "jagged", "thought", "tailed", "dark"])
def test_bubble_redraw_fills_interior_keeps_outline_and_touches_nothing_outside(kind: str):
    img, result, interior, outline, tail, bubble, rect, flag = _redraw(kind)
    paper = 20 if kind == "dark" else PAPER
    assert not flag
    inner = interior & ~cv2.dilate(outline.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    # Every interior pixel away from the outline is the paper colour: glyphs, furigana and strays are gone.
    assert (_gray(result)[inner] == paper).all()
    # The outline is byte-identical to the source (its own thickness, its anti-aliased edge).
    assert (result[outline] == img[outline]).all()
    # Nothing outside the bubble changed.
    outside = ~cv2.dilate((interior | outline).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert (result[outside] == img[outside]).all()
    # The patch stays inside the bubble's rectangle (plus a small margin).
    assert rect.x >= bubble.x - 3 and rect.y >= bubble.y - 3 and rect.x2 <= bubble.x2 + 3 and rect.y2 <= bubble.y2 + 3


def test_tailed_bubble_keeps_the_tail_as_paper_inside_its_outline():
    img, result, interior, outline, tail, bubble, rect, flag = _redraw("tailed")
    assert (_gray(result)[tail] == PAPER).all()
    tail_outline = outline & (np.arange(400)[:, None] > 285)
    assert tail_outline.any() and (result[tail_outline] == img[tail_outline]).all()


def test_joined_bubbles_fill_both_lobes_from_either_block():
    img, result, interior, outline, tail, bubble, rect, flag = _redraw("joined")
    left = interior & (np.arange(400)[None, :] < 90)
    right = interior & (np.arange(400)[None, :] > 310)
    assert left.any() and right.any()
    assert (_gray(result)[left] == PAPER).all() and (_gray(result)[right] == PAPER).all()


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


def test_apply_exports_the_glyph_mask_aligned_with_the_clean_rect():
    """The mask the quality renderer regenerates: the pixels the eraser painted over."""
    img = _page()
    box = _draw_column(img, 100, 60, 3, 24)
    blocks = [_block(box, 24.0)]
    erase.apply(img, blocks)
    style = blocks[0].style
    rect, mask = style.clean_rect, style.erase_mask
    assert mask is not None and mask.dtype == bool
    assert mask.shape == (rect.h, rect.w) == style.clean_patch.shape[:2]
    assert mask.any() and not mask.all()  # the glyphs, not the whole patch
    # Every erased pixel was ink in the source and is paper in the patch.
    source = _gray(img[rect.y : rect.y2, rect.x : rect.x2])
    assert source[mask].min() < PAPER
    assert (_gray(style.clean_patch)[mask] == PAPER).all()
    # The mask covers the glyph column and nothing far outside it.
    ys, xs = np.nonzero(mask)
    assert rect.x + xs.min() >= box.x - erase.ERASE_DILATE - 1
    assert rect.x + xs.max() <= box.x2 + erase.ERASE_DILATE + 1


def test_erase_block_masked_and_erase_block_agree():
    img = _page()
    box = _draw_column(img, 180, 100, 4, 30)
    block = _block(box, 30.0)
    patch, rect, outline = erase.erase_block(img, block)
    patch2, rect2, outline2, mask = erase.erase_block_masked(img, block)
    assert rect == rect2 and outline == outline2 and np.array_equal(patch, patch2)
    assert mask.shape == (rect.h, rect.w)


# ------------------------------------------------------- glyphs nobody reported
def _redrawn(img: np.ndarray, block: TextBlock) -> np.ndarray:
    """The page with ``block``'s clean patch painted back on."""
    patch, rect, _flag = erase.erase_block(img, block)
    out = img.copy()
    out[rect.y : rect.y2, rect.x : rect.x2] = patch
    return out


def _ink(page: np.ndarray, rect: Rect) -> int:
    return int((_gray(page)[rect.y : rect.y2, rect.x : rect.x2] < PAPER - 20).sum())


def test_page_lettering_scale_bounds_a_giant_stylised_block():
    """One stylised character must not set the page's erase scale.

    Every reach in the eraser is a multiple of the block's glyph size, so a
    block measured many times the page's lettering sweeps - and deletes -
    artwork hundreds of pixels from the text.  ``GLYPH_CAP`` exists for this,
    but it was skipped entirely on pages of fewer than three blocks and took a
    *plain* median, which a one-character sound effect sets as readily as a line
    of dialogue.  Both holes are on this page: two blocks, and a plain median of
    (24, 180) that leaves the sound effect uncapped either way.
    """
    lettering, stylised = 24.0, 180.0
    capped = erase.GLYPH_CAP * lettering
    img = _page(760, 620)
    dialogue = _draw_column(img, 60, 60, 6, int(lettering))
    sfx = _draw_column(img, 300, 120, 1, int(stylised))
    # An art blob on the sound effect's column axis, below its foot and big enough to
    # extend a column.  The geometry is *derived* from the tunables, not hard-coded:
    # a test that hard-codes it silently stops discriminating when one of them moves.
    gap = round(0.72 * erase.SWEEP_GAP * stylised)
    blob = Rect(345, sfx.y2 + gap, 90, 90)
    assert gap < erase.SWEEP_GAP * stylised  # uncapped: the sweep bridges the gap...
    assert blob.y2 - sfx.y2 < erase.SWEEP_MAX * stylised  # ...and reaches that far...
    assert erase.SWEEP_BODY * stylised <= blob.w <= erase.GLYPH_MAX * stylised  # ...over a glyph-sized body
    assert gap > erase.SWEEP_GAP * capped  # capped: the gap is out of reach...
    assert blob.w > erase.GLYPH_MAX * capped  # ...and the blob is not glyph-sized either
    cv2.rectangle(img, (blob.x, blob.y), (blob.x2 - 1, blob.y2 - 1), (0, 0, 0), -1)
    before = _ink(img, blob)
    assert before == blob.w * blob.h

    blocks = [_block(dialogue, lettering, text="ダイアローグです"), _block(sfx, stylised, text="ド")]
    # The page's lettering scale is the dialogue's, not the median of the two.
    assert erase.page_glyph(blocks) == pytest.approx(lettering)

    erase.apply(img, blocks)
    page = _compose(img, blocks)
    assert _ink(page, blob) == before  # the art blob is untouched
    assert _ink(page, sfx) == 0  # ...and the sound effect is still erased
    assert _ink(page, dialogue) == 0


def test_page_glyph_ignores_blocks_with_no_measured_glyph():
    box = Rect(180, 100, 30, 138)
    assert erase.page_glyph([]) == 0.0
    # A block with no measured glyph does not vote...
    assert erase.page_glyph([_block(box, 30.0), _block(box, 0.0, text="x")]) == pytest.approx(30.0)
    # ...and a block with no text still votes once, so it cannot vanish from the scale.
    assert erase.page_glyph([_block(box, 50.0, text="")]) == pytest.approx(50.0)


def test_a_column_the_detector_never_reported_is_erased_inside_the_balloon(monkeypatch):
    """Whole columns go missing from the OCR - 1ja returns one of the five in
    one balloon - and a column that far out is well past the zone the reported
    boxes grow (``BUBBLE_ZONE_EM``).  Inside a balloon there is nothing but
    paper and lettering, so ``BUBBLE_STRAY_INSIDE`` lets the fill follow
    glyph-sized ink across the interior to reach it."""
    img, _interior, _outline, _tail, bubble, box = _bubble_page("round")
    missed = _draw_column(img, 252, 150, 3, 24)
    assert missed.x - box.x2 > erase.BUBBLE_ZONE_EM * 24  # out of reach of the boxes' own zone
    assert _ink(img, missed) > 0
    block = _block(box, 24.0, bubble=bubble)
    assert _ink(_redrawn(img, block), missed) == 0
    # ...and it is that rule doing it: with nothing able to qualify, it stays.
    monkeypatch.setattr(erase, "BUBBLE_STRAY_INSIDE", 2.0)
    assert _ink(_redrawn(img, block), missed) > 0


def test_art_at_the_same_reach_is_too_big_for_the_stray_rule():
    """The escape that keeps the rule selective rather than "erase everything
    on the interior": what it collects has to be glyph sized.  This shape sits
    exactly as far out as the missed column above - 4ja's chibi, whose head
    reaches into a balloon - and is bigger than a glyph, so the fill never
    follows it and the art stays."""
    img, _interior, _outline, _tail, bubble, box = _bubble_page("round")
    art = Rect(250, 145, 57, 57)
    cv2.circle(img, (278, 173), 27, (0, 0, 0), 2)
    assert art.x - box.x2 > erase.BUBBLE_ZONE_EM * 24  # the same reach as the missed column
    assert art.w > erase.GLYPH_MAX * 24  # ...but not a glyph
    assert _ink(_redrawn(img, _block(box, 24.0, bubble=bubble)), art) > 0


# ------------------------------------------------------- overlapping patches
def _compose(img: np.ndarray, blocks) -> np.ndarray:
    """The page with every block's clean patch painted on, in the order given."""
    out = img.copy()
    for blk in blocks:
        r = blk.style.clean_rect
        out[r.y : r.y2, r.x : r.x2] = blk.style.clean_patch
    return out


def test_overlapping_patches_do_not_repaint_each_others_glyphs(monkeypatch):
    """Two columns close enough that their clean rects overlap.  Neither block
    erases the other's glyphs (they are ``foreign``), so each patch still
    carries them: painted in the wrong order the later patch puts the erased
    ink back (2ja: the ``E`` of ``SASUKE`` came back once the balloon beside it
    grew).  ``_share_erased`` hands every patch what the others erased inside
    it, so the painting order stops mattering."""
    img = _page()
    a = _draw_column(img, 150, 120, 3, 24)
    b = _draw_column(img, 178, 120, 3, 24)  # four pixels of paper between the columns
    blocks = [_block(a, 24.0), _block(b, 24.0)]
    erase.apply(img, blocks)
    assert blocks[0].style.clean_rect.intersects(blocks[1].style.clean_rect)
    page = _compose(img, blocks)
    assert np.array_equal(page, _compose(img, blocks[::-1]))
    assert _ink(page, Rect(a.x, a.y, b.x2 - a.x, a.h)) == 0  # both columns gone, either way round

    # ...and without the step the two orders really do differ.
    monkeypatch.setattr(erase, "_share_erased", lambda blocks: None)
    raw = [_block(a, 24.0), _block(b, 24.0)]
    erase.apply(img, raw)
    assert not np.array_equal(_compose(img, raw), _compose(img, raw[::-1]))
