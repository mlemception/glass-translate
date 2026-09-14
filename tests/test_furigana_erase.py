"""Defect C - ruby ink survives the erase and shows through the English.

Furigana is the smallest, faintest thing on the page, so the detector returns
it well below the pipeline's confidence floor when it returns it at all: on
``more_comparisons/4ja.jpg`` the ruby ``めぐみ`` comes back at 0.26 while its
kanji comes back at 0.56.  Only the confident lines reach ``build_blocks`` as
``segments``, so ``_mark_furigana`` never sees the ruby, it never lands in
``TextBlock.furigana``, and the eraser - which works from
``block.members + block.furigana`` - never paints over it.

``build_blocks`` takes the low-confidence lines as ``all_segments`` and its
docstring promises they are "used by the eraser as evidence of glyphs the
confident boxes missed"; ``render/erase.apply`` used to open with
``del all_segments`` and throw the evidence away unread.  It now hands each
dropped line to the block it belongs to (``erase._evidence``), which erases it
and never translates it - and these tests hold it to both halves.

The scenes are synthetic (squares for glyphs, a small column of squares for
the ruby beside them) and stand for the 4ja caption above and for the ruby in
the balloon of 1ja block 13, where ``かいしゅう``/``かえ``/``だめ`` survive.
Two of them, because a balloon hides the defect: the bubble branch paints its
whole paper interior and takes ruby beside a column with it either way.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pytest

from glasstranslate.core.types import Rect, Segment
from glasstranslate.render import build_blocks
from glasstranslate.render.erase import _evidence
from glasstranslate.render.layout import _line, _mark_furigana
from glasstranslate.render.place import _source_rect

BLACK, WHITE = (0, 0, 0), (255, 255, 255)
# The confidence floor of the manga pipeline (``core/pipeline.py``): ruby lands under it.
LOW_CONFIDENCE = 0.26


def quad(r: Rect) -> np.ndarray:
    return np.array([[r.x, r.y], [r.x2, r.y], [r.x2, r.y2], [r.x, r.y2]], np.float32)


def draw_column(img: np.ndarray, x: int, y: int, n: int, glyph: int, pitch: int) -> Rect:
    for k in range(n):
        cv2.rectangle(img, (x + 1, y + pitch * k + 1), (x + glyph - 1, y + pitch * k + glyph - 1), BLACK, -1)
    return Rect(x - 2, y - 2, glyph + 4, pitch * (n - 1) + glyph + 4)


def ruby_page(gap: int = 12) -> Tuple[np.ndarray, List[Segment], List[Segment], List[Rect]]:
    """A balloon with two kanji columns, each with a ruby column ``gap`` px to
    its right.  Returns ``(page, confident lines, ruby lines, ruby boxes)``."""
    img = np.full((420, 420, 3), 40, np.uint8)
    cv2.ellipse(img, (210, 200), (130, 140), 0, 0, 360, WHITE, -1)
    cv2.ellipse(img, (210, 200), (130, 140), 0, 0, 360, BLACK, 3)
    cv2.ellipse(img, (210, 200), (128, 138), 0, 0, 360, WHITE, -1)
    main, ruby = [], []
    for k in range(2):
        box = draw_column(img, 120 + 58 * k, 110, 5, 26, 32)
        main.append(box)
        ruby.append(draw_column(img, box.x2 + gap, 114, 8, 9, 12))
    texts = ["漢字の文章", "読み方です"]
    kana = ["かんじ", "よみかた"]
    confident = [Segment(t, quad(b), 0.95) for t, b in zip(texts, main)]
    rubies = [Segment(t, quad(b), LOW_CONFIDENCE) for t, b in zip(kana, ruby)]
    return img, confident, rubies, ruby


def open_ruby_page(gap: int = 12) -> Tuple[np.ndarray, List[Segment], List[Segment], List[Rect]]:
    """The same two annotated columns, over artwork instead of inside a
    balloon.  A balloon hides the defect: the bubble branch of the eraser
    paints the whole paper interior out to ``BUBBLE_ZONE_EM`` glyphs past the
    text boxes, which happens to swallow ruby sitting beside a column whether
    anyone knew it was there or not.  Free text has no interior to paint, so
    only ink the eraser was told about goes."""
    img = np.full((420, 420, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (419, 56), (40, 40, 40), -1)  # a scrap of art, well clear of the text
    main, ruby = [], []
    for k in range(2):
        box = draw_column(img, 120 + 58 * k, 110, 5, 26, 32)
        main.append(box)
        ruby.append(draw_column(img, box.x2 + gap, 114, 8, 9, 12))
    confident = [Segment(t, quad(b), 0.95) for t, b in zip(["漢字の文章", "読み方です"], main)]
    rubies = [Segment(t, quad(b), LOW_CONFIDENCE) for t, b in zip(["かんじ", "よみかた"], ruby)]
    return img, confident, rubies, ruby


def erased_page(img: np.ndarray, blocks) -> np.ndarray:
    """The page with every block's ``clean_patch`` pasted back, in grey."""
    out = img.copy()
    for block in blocks:
        st = block.style
        if st.clean_patch is not None and st.clean_rect is not None:
            r = st.clean_rect
            out[r.y: r.y2, r.x: r.x2] = st.clean_patch
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)


def ink_in(gray: np.ndarray, boxes: Sequence[Rect]) -> List[int]:
    return [int((gray[b.y: b.y2, b.x: b.x2] < 128).sum()) for b in boxes]


# ------------------------------------------------------------ the defect
@pytest.mark.parametrize("scene", [ruby_page, open_ruby_page], ids=["balloon", "free-text"])
def test_ruby_the_detector_only_half_saw_is_still_erased(scene) -> None:
    img, confident, rubies, ruby_boxes = scene()
    source = ink_in(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), ruby_boxes)
    assert all(px > 0 for px in source), "the fixture must actually draw ruby"
    blocks = build_blocks(img, confident, list(confident) + list(rubies))
    left = ink_in(erased_page(img, blocks), ruby_boxes)
    assert left == [0, 0], (
        f"ruby ink survives the erase: {left} px left of {source} px, although the ruby quads "
        "were handed to build_blocks as all_segments"
    )
    # ...and the other half: what was erased as evidence is not translated.
    text = "".join(b.segment.text for b in blocks)
    assert not any(r.text in text for r in rubies), f"the ruby reading leaked into the block text {text!r}"


def test_free_text_ruby_needs_the_low_confidence_evidence() -> None:
    """The guard on the guard: with the ruby quads withheld, the free-text
    scene must still show the ink.  Without it the parametrised case above
    could pass on a balloon's paper fill alone and prove nothing."""
    img, confident, rubies, ruby_boxes = open_ruby_page()
    blocks = build_blocks(img, confident, list(confident))
    assert all(b.bubble is None for b in blocks), "the free-text scene must not read as a balloon"
    assert ink_in(erased_page(img, blocks), ruby_boxes) != [0, 0]


def test_a_blocks_source_footprint_covers_its_furigana() -> None:
    """``render/place._source_rect`` anchors the lettering on
    ``SegmentStyle.source_quads``, which ``layout._make_block`` used to fill
    from the member lines only - the furigana quads were dropped, so the
    footprint the placement (and any consumer of the style alone) saw was
    narrower than the ink that was erased.  Either the ruby quads join
    ``source_quads`` or the style grows a field for them; this asserts the
    footprint, not the field."""
    img, confident, rubies, ruby_boxes = ruby_page()
    blocks = build_blocks(img, list(confident) + list(rubies), list(confident) + list(rubies))
    with_ruby = [b for b in blocks if b.furigana]
    assert with_ruby, "the confident ruby lines must be recognised as furigana"
    for block in with_ruby:
        footprint = _source_rect(block.style)
        for ruby in block.furigana:
            box = ruby.bbox
            assert footprint.x <= box.x and footprint.x2 >= box.x2, (
                f"the source footprint {footprint} leaves the furigana {box} out"
            )
            assert footprint.y <= box.y and footprint.y2 >= box.y2, (
                f"the source footprint {footprint} leaves the furigana {box} out"
            )


# ------------------------------------------------------------ guard
def test_confident_ruby_is_erased() -> None:
    """The path that already works, kept so a fix cannot trade it away."""
    img, confident, rubies, ruby_boxes = ruby_page()
    promoted = [Segment(s.text, s.quad, 0.9) for s in rubies]
    blocks = build_blocks(img, list(confident) + promoted, list(confident) + promoted)
    assert ink_in(erased_page(img, blocks), ruby_boxes) == [0, 0]


# ------------------------------------------------------------ the other half
# ``layout._mark_furigana`` drops whatever it flags from the translation, so a
# rule loose enough to catch ruby by size alone eats dialogue: a kana column
# beside a slightly bigger one clears a 0.7 glyph-box ratio easily.  The pairs
# below are measured off the five reference pages - (ruby text, ruby bbox,
# parent text, parent bbox) - and the two lists are what the old rule got
# wrong and right.  Spoken lines first: every one of these was being eaten.
DIALOGUE = [
    ("すまんな", (641, 807, 24, 69), "短距離選手の", (606, 801, 49, 118)),
    ("ところで", (170, 130, 18, 58), "消毒くせえ", (181, 128, 28, 73)),
    ("ねー", (138, 131, 19, 32), "消毒くせえ", (181, 128, 28, 73)),
    ("じゃねえ", (605, 96, 26, 69), "花とかも", (660, 96, 35, 72)),
    ("こねーよ", (287, 413, 25, 68), "いちいち", (319, 411, 36, 73)),
    ("なくても", (232, 970, 25, 75), "迷っても", (272, 963, 46, 88)),
    ("あまり", (355, 106, 25, 60), "悪いが", (372, 103, 42, 66)),
    ("これだ", (673, 435, 29, 62), "待ってる", (635, 429, 53, 90)),
    ("だろ", (629, 437, 26, 45), "待ってる", (635, 429, 53, 90)),
    ("だよね", (626, 755, 18, 41), "俺は別に", (684, 752, 30, 58)),
    ("とらん", (85, 570, 21, 53), "まあこの際", (134, 566, 33, 86)),
    ("ねえ", (299, 82, 23, 36), "って", (274, 76, 39, 73)),
]
# ...and real ruby, which must keep being read as ruby.  ``はら`` is the
# largest measured on the reference pages (0.71 of its parent's em).
RUBY = [
    ("はら", (669, 1023, 21, 42), "細心の注意を払った……", (638, 873, 43, 297)),
    ("せんしゅ", (638, 853, 10, 31), "短距離選手の", (606, 801, 49, 118)),
    ("じゅじゅつこうせん", (718, 106, 17, 73), "呪術高専の", (687, 98, 46, 103)),
    ("ひろ", (88, 438, 15, 23), "拾ったわ", (65, 433, 31, 86)),
    ("たす", (167, 786, 16, 25), "人を助けろ", (140, 736, 34, 123)),
    ("こじょうさとる", (945, 931, 24, 102), "俺も五条悟との", (915, 872, 43, 198)),
    ("ちゅうだん", (168, 210, 17, 53), "中断と", (141, 207, 32, 83)),
    ("ねんがっ", (384, 460, 66, 20), "2018年6月", (332, 471, 115, 30)),
    ("みや、きけんせんだいし", (486, 460, 136, 20), "宮城県仙台市", (481, 464, 151, 44)),
]


def _is_ruby(case) -> bool:
    """Run ``_mark_furigana`` over one measured (ruby, parent) pair."""
    text, box, parent_text, parent_box = case
    lines = [_line(Segment(t, quad(Rect(*b)), 0.9), 4000, 4000) for t, b in ((text, box), (parent_text, parent_box))]
    for l in lines:
        l.comp = (False, 1)  # one patch of paper, as they are on the page
    _mark_furigana(lines)
    return lines[0].furigana_of is not None


@pytest.mark.parametrize("case", DIALOGUE, ids=[c[0] for c in DIALOGUE])
def test_dialogue_is_not_read_as_furigana(case) -> None:
    assert not _is_ruby(case), f"{case[0]!r} beside {case[2]!r} is speech, and furigana is dropped from the translation"


@pytest.mark.parametrize("case", RUBY, ids=[c[0] for c in RUBY])
def test_real_ruby_is_still_read_as_furigana(case) -> None:
    assert _is_ruby(case), f"{case[0]!r} beside {case[2]!r} is ruby and would be translated as noise"


# ------------------------------------------------- what evidence is *not*
# ``erase._evidence`` erases what it takes and never translates it, so the
# same rule applies to it as to ``_mark_furigana``: what it waves through is
# lost speech.  A Japanese column is one em *wide* however long it runs, so
# the box's smaller side is the same 26 px for a column of dialogue as for the
# block beside it - only the character em separates them.  3jp hands the
# eraser ``っーか`` at 0.41, one full-size kana column next to the block that
# should have had it; the box measure calls it ruby.
def _evidence_scene(gap: int, glyph: int, pitch: int, n: int, text: str) -> Tuple[Rect, Rect, List[Rect]]:
    """One confident column with one low-confidence line ``gap`` px to its
    right; returns ``(block box, candidate box, boxes taken as evidence)``."""
    img = np.full((420, 420, 3), 255, np.uint8)
    cv2.rectangle(img, (0, 0), (419, 40), (40, 40, 40), -1)  # a scrap of art, clear of the text
    main = draw_column(img, 150, 100, 5, 26, 26)
    other = draw_column(img, main.x2 + gap, 100, n, glyph, pitch)
    confident = [Segment("漢字文章読", quad(main), 0.95)]
    every = confident + [Segment(text, quad(other), LOW_CONFIDENCE)]
    blocks = build_blocks(img, confident, every)
    return main, other, [r for boxes in _evidence(blocks, every, 420, 420) for r in boxes]


EVIDENCE_CASES = [
    ("ruby at half the pitch, 6 px away", 6, 11, 13, 8, "かんじのよみかた", True),
    ("a full-size column 12 px away", 12, 26, 26, 5, "短距離選手の", False),
    ("a full-size column 38 px away", 38, 26, 26, 5, "短距離選手の", False),
    ("a full-size column 45 px away", 45, 26, 26, 5, "短距離選手の", False),
    ("a 3x-em sound effect 20 px away", 20, 78, 78, 2, "ドン", False),
]


@pytest.mark.parametrize("case", EVIDENCE_CASES, ids=[c[0] for c in EVIDENCE_CASES])
def test_only_ruby_sized_ink_is_taken_as_erase_evidence(case) -> None:
    _name, gap, glyph, pitch, n, text, absorbed = case
    _main, other, taken = _evidence_scene(gap, glyph, pitch, n, text)
    assert any(r == other for r in taken) is absorbed, (
        f"{_name}: {'should' if absorbed else 'must not'} be erased as the block's own ink "
        f"({'' if absorbed else 'it is a line of speech the grouping did not get'})"
    )


def test_the_box_measure_cannot_tell_speech_from_ruby() -> None:
    """Why the guard is on the em: the candidate column and the block's own
    column have the *same* smaller side, so any bound on the box admits both."""
    main, column, _ = _evidence_scene(12, 26, 26, 5, "短距離選手の")
    _m, ruby, _ = _evidence_scene(6, 11, 13, 8, "かんじのよみかた")
    assert min(column.w, column.h) == min(main.w, main.h)
    assert min(ruby.w, ruby.h) < min(main.w, main.h)
