"""A quality patch must not repaint a neighbouring balloon's paper.

The sidecar is only ever scheduled on free text over artwork - ``quality.py``'s
``_is_free_text_over_art`` returns False for any ``in_bubble`` block - so nothing
aims a job at a balloon.  What reaches one is the *edge* of a neighbour's patch:
a free-text block's ``clean_rect`` can abut a balloon, and ``QUALITY_MASK_DILATE``
plus ``_feather`` carry the blend a few px further.

Measured over the 50-page corpus, comparing the plain and quality erased renders
and counting only pixels that had been clean balloon paper (>= 225 grey):

    v12_p140_28a413    7,057 changed /  3,061 paper   43.4 %
    v16_p143_8b9f24   20,388 changed /  3,622 paper   17.8 %
    v13_p004h1_6b176b  3,477 changed /    418 paper   12.0 %
    v04_p093_732007   21,687 changed /    970 paper    4.5 %
    v23_p011_0e160b   26,305 changed /     22 paper    0.1 %

That is the whole of the c_contain regression the quality renderer showed,
because ``typeset_metrics.our_lettering`` defines our ink as
``|typeset - erased| > 8``: anything the sidecar changed inside a block's own
region is scored as our lettering.

Balloons are redrawn by ``render/erase.py``.  The fix cuts their interiors out
of the blend at the swap, which constrains where a result lands and never what
is generated - so the job key is deliberately unaffected.

Synthetic frames only; no corpus imagery enters the repository.
"""
from __future__ import annotations

import numpy as np

from glasstranslate.core.types import Rect, SegmentStyle
from glasstranslate.render import quality as Q

FILL = 17  # a value neither the frame nor the patch uses, so writes are visible


def _free_text(clean: Rect, panel: Rect) -> SegmentStyle:
    """A free-text-over-artwork block: the only kind that gets a job."""
    mask = np.zeros((clean.h, clean.w), bool)
    mask[8:-8, 6:-6] = True
    patch = np.full((clean.h, clean.w, 3), 235, np.uint8)
    patch[::3, :, :] = 0  # hatching through the text: what the eraser keeps
    return SegmentStyle(
        fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, text_height_px=20.0, vertical=True,
        layout_box=Rect(clean.x, clean.y, clean.w, clean.h), clean_rect=clean,
        clean_patch=patch, outline=True, in_bubble=False, erase_mask=mask,
        panel_box=panel, max_font_px=16.0,
    )


def _balloon(box: Rect) -> SegmentStyle:
    """A balloon block, with the interior mask ``layout_mask`` carries."""
    interior = np.ones((box.h, box.w), bool)
    return SegmentStyle(
        fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, text_height_px=20.0, vertical=True,
        layout_box=box, layout_mask=interior, clean_rect=box,
        clean_patch=np.full((box.h, box.w, 3), 235, np.uint8),
        outline=False, in_bubble=True, erase_mask=np.zeros((box.h, box.w), bool),
        panel_box=None, max_font_px=16.0,
    )


def _scene():
    """A free-text block whose mask reaches into a balloon sharing its edge."""
    frame = np.full((200, 300, 3), 180, np.uint8)
    clean = Rect(10, 10, 30, 40)
    panel = Rect(0, 0, 150, 200)
    text = _free_text(clean, panel)
    # Overlaps the right-hand columns of the free-text block, which is exactly
    # where the dilated mask and the feathered edge land.
    bubble = _balloon(Rect(30, 10, 20, 40))
    return frame, clean, text, bubble


def test_the_fixture_really_would_have_been_written_without_the_guard() -> None:
    """The control that stops the next test passing for the wrong reason.

    If the job's mask never reached the balloon columns, the assertion below
    would hold no matter what the code did.  This pins that it does reach them.
    """
    frame, clean, text, bubble = _scene()
    job = Q.panel_jobs(frame, [("t", text), ("b", bubble)])[0]
    box = bubble.layout_box
    x0, x1 = max(clean.x, box.x), min(clean.x2, box.x2)
    cols = job.mask[:, x0 - job.rect.x : x1 - job.rect.x]
    assert (cols > 0).any(), "the job mask does not reach the balloon; the fixture is wrong"


def test_a_patch_never_repaints_a_neighbouring_balloon() -> None:
    """The defect, end to end.

    Fails against the old code, where the balloon's columns were blended over
    like any other masked pixel.
    """
    frame, clean, text, bubble = _scene()
    job = Q.panel_jobs(frame, [("t", text), ("b", bubble)])[0]
    result = np.full((job.rect.h, job.rect.w, 3), FILL, np.uint8)
    patch = Q.composite_block_patch(frame, job, result, text)

    box = bubble.layout_box
    x0 = max(clean.x, box.x) - clean.x
    x1 = min(clean.x2, box.x2) - clean.x
    assert x1 > x0, "the fixture must overlap, or this test proves nothing"
    assert np.array_equal(patch[:, x0:x1], text.clean_patch[:, x0:x1]), (
        "the patch repainted balloon paper")


def test_pixels_outside_the_balloon_are_still_painted() -> None:
    """The guard must not turn the patch off altogether."""
    frame, clean, text, bubble = _scene()
    job = Q.panel_jobs(frame, [("t", text), ("b", bubble)])[0]
    result = np.full((job.rect.h, job.rect.w, 3), FILL, np.uint8)
    patch = Q.composite_block_patch(frame, job, result, text)
    assert (patch == FILL).any(), "nothing was painted at all"


def test_a_page_with_no_balloons_carries_no_protect_mask() -> None:
    """Pages without balloons are untouched by this change."""
    frame, _clean, text, _bubble = _scene()
    job = Q.panel_jobs(frame, [("t", text)])[0]
    assert job.protect is None


def test_protect_does_not_change_the_job_key() -> None:
    """The mask constrains where a result lands, never what is generated, so a
    cached panel must not be invalidated by a balloon appearing beside it."""
    frame, _clean, text, bubble = _scene()
    without = Q.panel_jobs(frame, [("t", text)])[0]
    with_balloon = Q.panel_jobs(frame, [("t", text), ("b", bubble)])[0]
    assert with_balloon.protect is not None
    assert with_balloon.key == without.key


def test_balloon_interiors_ignores_free_text_layout_masks() -> None:
    """Free text also carries a ``layout_mask`` - its open-layout region - and
    protecting that would switch the sidecar off everywhere."""
    _frame, _clean, text, bubble = _scene()
    text.layout_mask = np.ones((text.layout_box.h, text.layout_box.w), bool)
    assert Q._balloon_interiors([text], 300, 200) is None
    both = Q._balloon_interiors([text, bubble], 300, 200)
    assert both is not None
    box = bubble.layout_box
    assert both[box.y : box.y2, box.x : box.x2].all()
    assert not both[text.layout_box.y, text.layout_box.x]
