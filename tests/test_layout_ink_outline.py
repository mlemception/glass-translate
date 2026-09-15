"""Balloons found by tracing the page's ink rather than flood-filling its paper.

``_PaperMaps`` flood-fills *paper*, and paper leaks.  Where a balloon's interior
is continuous with the page behind it, the joined component is far too wide and
not convex, so ``_is_bubble`` rejects it; the tight patch cut out of it can be
rejected too, when most of its edge is the cut rather than a wall
(``walled`` below ``_BUBBLE_MIN_WALLED``).  The block then falls through to the
free-text branch: its interior is never redrawn, so the source lettering
survives under ours, and it is set in sampled artwork colour with a halo.

The measured case is a real page, ``v01_p020_36f9fe`` block 11: the paper pass
offers a 1011 px-wide leak (rejected on width and solidity) and a tight patch at
``walled`` 0.432 against a 0.5 floor.  A closed ink contour bounds the same
balloon at solidity 0.987.  That page cannot be a fixture here - no corpus
imagery enters the repository - so these tests cover the mechanism on synthetic
pages and the page itself is checked through the corpus harness.

``bubbles.ink_rooms`` is the trace; a room bounded by ink is walled by
construction, which is the one measurement the paper pass cannot make.  Every
other ``_is_bubble`` guard still has to pass, which is what keeps a panel frame
or the counter of a glyph from being adopted as a balloon.
"""
from __future__ import annotations

import cv2
import numpy as np

from glasstranslate.core.types import Rect, Segment
from glasstranslate.render import build_blocks, bubbles
from glasstranslate.render import layout as L

BLACK, WHITE = (0, 0, 0), (255, 255, 255)


def quad(r: Rect) -> np.ndarray:
    return np.array([[r.x, r.y], [r.x2, r.y], [r.x2, r.y2], [r.x, r.y2]], np.float32)


def draw_column(img: np.ndarray, x: int, y: int, n: int, glyph: int = 20, pitch: int = 24,
                colour: tuple[int, int, int] = BLACK) -> Rect:
    """``n`` square glyphs stacked into a vertical column; returns its OCR box."""
    for k in range(n):
        cv2.rectangle(img, (x + 1, y + pitch * k + 1),
                      (x + glyph - 1, y + pitch * k + glyph - 1), colour, -1)
    return Rect(x - 2, y - 2, glyph + 4, pitch * (n - 1) + glyph + 4)


def balloon_page() -> tuple[np.ndarray, list[Segment]]:
    """A drawn balloon on page white, two columns in it."""
    img = np.full((460, 460, 3), WHITE, np.uint8)
    cv2.ellipse(img, (150, 210), (96, 130), 0, 0, 360, BLACK, 3)
    boxes = [draw_column(img, 104 + 34 * k, 150, 4) for k in range(2)]
    texts = ("ふきだしの", "ぶんです")
    return img, [Segment(t, quad(b), 0.95) for t, b in zip(texts, boxes)]


def unenclosed_text() -> tuple[np.ndarray, list[Segment]]:
    """Free text over artwork: no outline encloses it, so no room can."""
    img = np.full((420, 420, 3), WHITE, np.uint8)
    cv2.rectangle(img, (20, 100), (400, 320), (90, 90, 90), -1)
    box = draw_column(img, 180, 140, 4, colour=WHITE)
    return img, [Segment("あーとのうえ", quad(box), 0.95)]


def framed_panel() -> tuple[np.ndarray, list[Segment]]:
    """A panel border is a closed ink contour too, and is not a balloon."""
    img = np.full((520, 520, 3), WHITE, np.uint8)
    cv2.rectangle(img, (20, 20), (500, 500), BLACK, 4)
    box = draw_column(img, 250, 250, 3)
    return img, [Segment("ぱねるのなか", quad(box), 0.95)]


def spec_for(img: np.ndarray, segs: list[Segment]) -> tuple[L._PaperMaps, L._Spec]:
    """The maps and the first block's spec, as ``build_blocks`` would build them."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lines = [L._line(s, w, h) for s in segs]
    L._mark_furigana(lines)
    groups = L._group(lines)
    maps = L._PaperMaps(gray, [l.bbox for l in lines])
    return maps, L._spec(lines, groups[0], [])


# --------------------------------------------------------- bubbles.ink_rooms

def test_ink_rooms_finds_the_room_a_balloon_encloses() -> None:
    img, _ = balloon_page()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count, labels, stats = bubbles.ink_rooms(gray)
    assert count > 1, "no enclosed room found for a drawn balloon"
    room = int(labels[210, 150])
    assert room > 0, "the balloon's own centre is in no room"
    x, y, w, h, area = (int(v) for v in stats[room])
    # The ellipse is 192 x 260 about (150, 210); the room is its interior.
    assert 150 <= w <= 200 and 220 <= h <= 270, f"room is not the balloon: {(x, y, w, h)}"
    assert area > 20_000


def test_ink_rooms_encloses_nothing_where_no_outline_closes() -> None:
    img, _ = unenclosed_text()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    count, labels, stats = bubbles.ink_rooms(gray)
    # The artwork slab is solid ink with no hole in it, so the text on it is
    # in no room at all.
    assert int(labels[210, 210]) == 0, "free text over artwork was given a room"


def test_ink_rooms_is_stable_on_a_blank_page() -> None:
    gray = np.full((80, 80), 255, np.uint8)
    count, labels, stats = bubbles.ink_rooms(gray)
    assert count >= 1 and labels.shape == gray.shape
    assert not (labels > 0).any(), "a blank page enclosed something"


# ------------------------------------------------------- the rescue decision

def test_the_rescue_offers_the_balloon_the_ink_encloses() -> None:
    img, segs = balloon_page()
    maps, spec = spec_for(img, segs)
    interior = L._ink_interior(maps, spec)
    assert interior is not None, "the drawn balloon was not offered"
    assert interior.walled == 1.0, "a room bounded by ink is walled by construction"
    assert L._is_bubble(interior, spec), "the offered balloon failed the bubble gate"


def test_the_rescue_offers_nothing_for_free_text_over_artwork() -> None:
    img, segs = unenclosed_text()
    maps, spec = spec_for(img, segs)
    assert L._ink_interior(maps, spec) is None


def test_the_rescue_offers_a_panel_frame_but_the_gate_rejects_it() -> None:
    """A closed contour is necessary, not sufficient."""
    img, segs = framed_panel()
    maps, spec = spec_for(img, segs)
    interior = L._ink_interior(maps, spec)
    if interior is not None:
        assert not L._is_bubble(interior, spec), "a panel frame passed the bubble gate"


# ---------------------------------------------------------- end to end, safe

def test_free_text_over_artwork_is_still_free_text() -> None:
    img, segs = unenclosed_text()
    for b in build_blocks(img, segs):
        assert b.bubble is None, f"free text was given a bubble: {b.bubble}"


def test_a_panel_frame_is_not_adopted_as_a_balloon() -> None:
    img, segs = framed_panel()
    for b in build_blocks(img, segs):
        assert b.bubble is None, f"the panel frame became a bubble: {b.bubble}"


def test_a_drawn_balloon_is_still_a_balloon() -> None:
    """The paper pass already places this one; the rescue must not disturb it."""
    img, segs = balloon_page()
    blocks = build_blocks(img, segs)
    assert blocks and any(b.bubble is not None for b in blocks)
    for b in blocks:
        if b.bubble is not None:
            assert b.bubble.w <= 260 and b.bubble.h <= 320, f"bubble too big: {b.bubble}"
