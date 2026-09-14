"""Defect A - connected balloons are read as one patch of paper.

When two speech balloons touch, the artist erases the arc they share, so their
white interiors are a single connected region.  ``render/layout._make_block``
decides "bubble or not" from that whole region: the joined paper is several
times the area of either block's source text and is not convex, so
``_BUBBLE_MAX_AREA_RATIO`` / ``_BUBBLE_MAX_DIM_RATIO`` / the solidity test all
reject it and both blocks fall through to the free-text branch.  The old
``_split_shared_bubbles`` never got a chance: it only ran over blocks that had
already passed as bubbles.

Every balloon must end up with its own interior, so each block is lettered
inside the balloon its source text sits in and nowhere else.  The fixtures are
synthetic pages (joined balloons, a balloon roomier than the OCR read of it,
the page's own background) and crops of real pages (see
``tests/fixtures/README.md``).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from glasstranslate.core.types import Rect, Segment
from glasstranslate.render import build_blocks

FIXTURES = Path(__file__).resolve().parent / "fixtures"
BLACK, WHITE = (0, 0, 0), (255, 255, 255)

# ``2ja_joined_balloons.png``: the OCR lines of that crop (text, x, y, w, h, confidence).
LINES_2JA = [
    ("本気か伝わ", 21, 45, 21, 72, 0.973),
    ("挑んで……", 106, 26, 28, 79, 0.963),
    ("オマエにカ勝負を", 124, 24, 35, 128, 0.925),
    ("短距離選手の", 136, 21, 49, 118, 0.947),
    ("せんしゅ", 168, 73, 10, 31, 0.827),
    ("すまんな", 171, 27, 24, 69, 0.999),
]
# ``4ja_joined_balloons.png``: three balloons sharing one patch of paper.
LINES_4JA = [
    ("くんないと", 38, 36, 21, 76, 0.999),
    ("だよね", 126, 30, 18, 41, 0.930),
    ("気に入ってん", 136, 26, 27, 81, 0.996),
    ("俺は別に", 184, 27, 30, 58, 0.831),
]


def quad(r: Rect) -> np.ndarray:
    return np.array([[r.x, r.y], [r.x2, r.y], [r.x2, r.y2], [r.x, r.y2]], np.float32)


def fixture(name: str) -> np.ndarray:
    """A committed greyscale crop as a BGR page."""
    gray = cv2.imread(str(FIXTURES / name), cv2.IMREAD_GRAYSCALE)
    assert gray is not None, f"missing fixture {name}"
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def segments(rows) -> list[Segment]:
    return [Segment(t, quad(Rect(x, y, w, h)), c) for t, x, y, w, h, c in rows]


def draw_column(img: np.ndarray, x: int, y: int, n: int, glyph: int = 20, pitch: int = 24,
                colour: tuple[int, int, int] = BLACK, thickness: int = -1) -> Rect:
    """``n`` square glyphs stacked into a vertical column; returns its OCR box."""
    for k in range(n):
        cv2.rectangle(img, (x + 1, y + pitch * k + 1), (x + glyph - 1, y + pitch * k + glyph - 1), colour, thickness)
    return Rect(x - 2, y - 2, glyph + 4, pitch * (n - 1) + glyph + 4)


def joined_balloons(stacked: bool = False, dark: bool = False) -> tuple[np.ndarray, list[Segment]]:
    """Two balloons drawn the way a manga artist joins them: both interiors
    filled, both outlines drawn, then each interior re-filled so the arc the
    balloons share disappears and the paper is one region.  Three columns of
    text in each.  Stands for 2ja blocks 2/11/12 and 4ja blocks 9/10/11.
    ``dark`` inverts paper and ink: ``_PaperMaps`` has a branch for each and
    the split has to work on both."""
    paper, ink, backdrop = (BLACK, WHITE, 128) if dark else (WHITE, BLACK, 40)
    if stacked:
        img = np.full((520, 320, 3), backdrop, np.uint8)
        shape = [((160, 150), (120, 105)), ((160, 350), (120, 105))]
        starts = [(105, 95), (105, 295)]
    else:
        img = np.full((400, 540, 3), backdrop, np.uint8)
        shape = [((170, 200), (130, 115)), ((370, 200), (130, 115))]
        starts = [(105, 145), (305, 145)]
    for centre, axes in shape:
        cv2.ellipse(img, centre, axes, 0, 0, 360, paper, -1)
    for centre, axes in shape:
        cv2.ellipse(img, centre, axes, 0, 0, 360, ink, 3)
    for centre, axes in shape:
        cv2.ellipse(img, centre, (axes[0] - 2, axes[1] - 2), 0, 0, 360, paper, -1)
    texts = ["ひだりの", "ふきだしの", "ぶんです", "みぎの", "ふきだしの", "ぶんです"]
    boxes = [draw_column(img, x + 34 * k, y, 4, colour=ink) for x, y in starts for k in range(3)]
    return img, [Segment(t, quad(b), 0.95) for t, b in zip(texts, boxes)]


def roomy_balloon(dark: bool = False) -> tuple[np.ndarray, list[Segment]]:
    """One balloon with five columns in it, of which the OCR reports the last
    only - the 1ja 13/14 geometry, where the page yields 25 raw lines and
    four of a balloon's five columns are simply absent.  The four the OCR
    missed are drawn: their strokes are still ink, so the paper has holes in
    it exactly as the real page does."""
    paper, ink, backdrop = (BLACK, WHITE, 128) if dark else (WHITE, BLACK, 40)
    img = np.full((360, 360, 3), backdrop, np.uint8)
    for axes, colour, thick in (((66, 90), paper, -1), ((66, 90), ink, 3), ((64, 88), paper, -1)):
        cv2.ellipse(img, (172, 176), axes, 0, 0, 360, colour, thick)
    boxes = [draw_column(img, 112 + 24 * k, 138, 4, glyph=16, colour=ink, thickness=1) for k in range(5)]
    return img, [Segment("ぶんです", quad(boxes[-1]), 0.95)]


def page_background() -> tuple[np.ndarray, list[Segment]]:
    """Two blocks on the white of the page itself, which wraps round a panel
    of artwork: 2ja's component ``(False, 1)`` is the whole 764x1200 page at
    a solidity of 0.39.  Cutting components into rooms must not turn the page
    into a set of fake balloons."""
    img = np.full((420, 420, 3), WHITE, np.uint8)
    cv2.rectangle(img, (30, 110), (390, 300), BLACK, -1)
    boxes = [draw_column(img, 60, 20, 3), draw_column(img, 300, 330, 3)]
    return img, [Segment(t, quad(b), 0.95) for t, b in zip(("うえのぶん", "したのぶん"), boxes)]


def region_of(block, shape) -> np.ndarray:
    """The block's layout region as a page-sized mask."""
    st = block.style
    out = np.zeros(shape[:2], bool)
    if st.layout_box is None:
        return out
    box = st.layout_box
    out[box.y: box.y2, box.x: box.x2] = True if st.layout_mask is None else st.layout_mask
    return out


# ------------------------------------------------------------ synthetic pair
def test_joined_balloons_each_become_their_own_bubble() -> None:
    img, segs = joined_balloons()
    blocks = build_blocks(img, segs, segs)
    assert len(blocks) == 2, "one block per balloon"
    assert [b.style.in_bubble for b in blocks] == [True, True]
    assert all(b.bubble is not None for b in blocks)
    # Neither block's bubble may be the joined paper: a balloon is about as
    # wide as its own ellipse, not as wide as both.
    assert all(b.bubble.w < 0.75 * img.shape[1] for b in blocks)


def test_joined_balloons_do_not_share_one_layout_region() -> None:
    img, segs = joined_balloons()
    blocks = build_blocks(img, segs, segs)
    left, right = (region_of(b, img.shape) for b in blocks)
    assert int((left & right).sum()) == 0, "the two balloons' layout regions overlap"
    for block, region in zip(blocks, (left, right)):
        other = blocks[1 - blocks.index(block)].segment.bbox
        covered = int(region[other.y: other.y2, other.x: other.x2].sum())
        assert covered == 0, f"a block's layout region covers the other balloon's text ({covered} px)"


def test_lines_of_two_balloons_are_never_grouped_into_one_block() -> None:
    """The other way the join can go wrong: the columns of both balloons share
    ``_Line.comp``, so ``_group``/``_compatible`` may merge them into one run
    of text.  Two stacked balloons whose columns are a glyph apart."""
    img, segs = joined_balloons(stacked=True)
    blocks = build_blocks(img, segs, segs)
    assert len(blocks) == 2, "the two balloons' lines were merged into one block"
    for block in blocks:
        ys = [m.bbox.y + m.bbox.h / 2.0 for m in block.members]
        assert max(ys) - min(ys) < 150, "one block spans both balloons"


# ------------------------------------------------------------ real page crop
def test_a_bubbles_layout_region_never_covers_another_blocks_text() -> None:
    """``more_comparisons/2ja.jpg`` cropped to 470, 780, 220, 175: two joined
    balloons, four columns in the right one and one in the left (page blocks
    2, 11 and 12).  Their interiors are one region, so the right block's
    lettering area swallows the left balloon and its text."""
    img = fixture("2ja_joined_balloons.png")
    blocks = build_blocks(img, segments(LINES_2JA), segments(LINES_2JA))
    assert len(blocks) == 2
    assert all(b.style.in_bubble for b in blocks), "both columns sit in a balloon"
    for i, block in enumerate(blocks):
        region = region_of(block, img.shape)
        for j, other in enumerate(blocks):
            if i == j:
                continue
            src = other.segment.bbox
            covered = int(region[src.y: src.y2, src.x: src.x2].sum())
            assert covered == 0, f"block {i}'s region covers block {j}'s source text ({covered} px)"


def test_three_balloons_on_one_patch_of_paper_are_all_still_balloons() -> None:
    """``more_comparisons/4ja.jpg`` cropped to 500, 725, 230, 160 (page blocks
    9, 10 and 11): three balloons whose interiors run into each other.  The
    joined paper is 3.5 x the area of any one block's text and not convex, so
    every one of them is demoted to free text and lettered over the artwork
    instead of inside its balloon."""
    img = fixture("4ja_joined_balloons.png")
    segs = segments(LINES_4JA)
    blocks = build_blocks(img, segs, segs)
    # Pinned, not taken from the result: four lines in three balloons, and a
    # grouping change that merged any two of them would otherwise pass here.
    assert len(blocks) == 3
    assert [b.style.in_bubble for b in blocks] == [True, True, True]


# ------------------------------------------------------- the other three cases
def test_joined_balloons_on_dark_paper_each_become_their_own_bubble() -> None:
    """The same pair with paper and ink swapped (4ja block 8 is a dark panel):
    ``_PaperMaps.get`` has a branch for each and the split runs on both."""
    img, segs = joined_balloons(dark=True)
    blocks = build_blocks(img, segs, segs)
    assert len(blocks) == 2
    assert [b.style.in_bubble for b in blocks] == [True, True]
    left, right = (region_of(b, img.shape) for b in blocks)
    assert int((left & right).sum()) == 0, "the two balloons' layout regions overlap"


def test_a_balloon_roomier_than_the_ocr_read_is_still_a_balloon() -> None:
    """1ja blocks 13 and 14: a clean convex balloon that fails only because
    it is wider than 2.5 x the *found* text - and the OCR found one of its
    five columns.  Measuring the balloon against the text the OCR happened to
    report is the wrong premise, not the wrong constant."""
    for dark in (False, True):
        img, segs = roomy_balloon(dark=dark)
        blocks = build_blocks(img, segs, segs)
        assert len(blocks) == 1
        block = blocks[0]
        assert block.style.in_bubble, f"dark={dark}: the balloon was read as panel background"
        assert block.bubble is not None and block.bubble.w >= 120, "the whole balloon, not the column"


def test_the_page_background_is_never_cut_into_balloons() -> None:
    """The paper the page itself is printed on holds text too (2ja blocks 2,
    11 and 12 sit on a component that is the whole page).  Whatever the
    rooms pass does, that paper must still read as panel background."""
    img, segs = page_background()
    blocks = build_blocks(img, segs, segs)
    assert len(blocks) == 2
    assert [b.style.in_bubble for b in blocks] == [False, False]
