"""Typesetting metrics (render/typeset.py) and letterer-style placement
(render/place.py) with a fake linear measurer: no fonts, no images."""
from __future__ import annotations

import importlib
import time

import numpy as np
import pytest

from glasstranslate.core.types import Rect, SegmentStyle
from glasstranslate.render import place

# The package re-exports a *function* named ``typeset``; import the module.
T = importlib.import_module("glasstranslate.render.typeset")

TEXT = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AGAIN AND AGAIN TONIGHT"


def fake_measure(text: str, size: float) -> tuple[float, float]:
    """Every glyph is half an em wide; the line box is 1.15 em (Comic Neue)."""
    return 0.5 * size * len(text), 1.15 * size


def free_style(search: Rect, source: Rect, *, ink=None, blocked=None, max_font_px: float = 20.0, vertical: bool = True) -> SegmentStyle:
    quad = np.array([[source.x, source.y], [source.x2, source.y], [source.x2, source.y2], [source.x, source.y2]], dtype=np.float32)
    return SegmentStyle(
        fg=(0, 0, 0),
        bg=(255, 255, 255),
        angle_deg=0.0,
        text_height_px=32.0,
        vertical=vertical,
        layout_box=search,
        layout_seed=source,
        source_quads=quad[None],
        max_font_px=max_font_px,
        outline=True,
        search_box=search,
        ink_map=np.zeros((search.h, search.w), np.uint8) if ink is None else ink,
        blocked_map=np.zeros((search.h, search.w), bool) if blocked is None else blocked,
    )


# ------------------------------------------------------------ metrics
def test_caps_leading_is_tight_and_descenders_clear():
    ts = T.typeset(TEXT, T.rect_spans(Rect(0, 0, 400, 400)), 0.0, fake_measure, max_size=20)
    glyph_h = 1.15 * 20
    assert ts.glyph_h == pytest.approx(glyph_h)
    assert ts.line_h == pytest.approx(glyph_h * T.CAPS_PITCH_RATIO)
    assert ts.line_h < glyph_h  # tighter than ascent + descent
    tops = [l.top for l in ts.lines]
    assert len(tops) >= 2
    assert np.allclose(np.diff(tops), ts.line_h)
    # The deepest all-caps descender of one line ends above the next line's cap top.
    assert T.DESCENT_RATIO < T.CAPS_PITCH_RATIO + T.CAP_TOP_RATIO
    # bbox spans the inked rows: cap top of the first line to the descender of the last.
    bb = ts.bbox
    assert bb.y == int(np.floor(tops[0] + T.CAP_TOP_RATIO * glyph_h))
    assert bb.y2 >= int(tops[-1] + T.DESCENT_RATIO * glyph_h) - 1


def test_mixed_case_keeps_full_pitch():
    ts = T.typeset("The quick brown fox jumps over the lazy dog", T.rect_spans(Rect(0, 0, 400, 400)), 0.0, fake_measure, max_size=20)
    assert ts.line_h == pytest.approx(1.15 * 20)
    assert T.line_pitch("Mixed", 23.0) == pytest.approx(23.0)
    assert T.line_pitch("CAPS", 23.0) == pytest.approx(23.0 * T.CAPS_PITCH_RATIO)


def test_memoize_measure_counts_calls():
    calls = []

    def counting(text, size):
        calls.append(text)
        return fake_measure(text, size)

    m = T.memoize_measure(counting)
    for _ in range(5):
        m("HELLO", 10.0)
        m("HELLO", 12.0)
    assert len(calls) == 2
    assert T.memoize_measure(m) is m  # idempotent


def test_line_breaking_dp_and_forced_breaks():
    words = TEXT.split()
    width = T.span_widths(words, 10.0, fake_measure)
    w1 = T.min_max_width(len(words), 1, width)
    w3 = T.min_max_width(len(words), 3, width)
    w6 = T.min_max_width(len(words), 6, width)
    assert w1 > w3 > w6
    starts = T.balanced_breaks(len(words), 3, w3 + 0.01, width)
    assert starts[0] == 0 and starts[-1] == len(words) and len(starts) == 4
    for a, b in zip(starts, starts[1:]):
        assert width(a, b) <= w3 + 0.01
    # A forced break after word 2 ends a line there in both DPs.
    wf = T.min_max_width(len(words), 4, width, forced=(2,))
    assert wf >= w3 * 0  # exists
    starts_f = T.balanced_breaks(len(words), 4, wf + 0.01, width, forced=(2,))
    assert 3 in starts_f


# ------------------------------------------------------------ placement
def test_ceiling_reached_on_open_paper():
    search = Rect(0, 0, 600, 400)
    source = Rect(260, 120, 80, 160)  # a two-column Japanese block
    ts = place.typeset_block(TEXT, free_style(search, source), fake_measure)
    assert ts.fitted and ts.lines
    assert ts.size == pytest.approx(20.0)  # the ceiling, unscaled
    bb = ts.bbox
    assert bb.x >= search.x and bb.y >= search.y and bb.x2 <= search.x2 and bb.y2 <= search.y2
    # Compact block near the source: not a one-line banner, not one word per line.
    assert 2 <= len(ts.lines) <= 8
    assert 0.5 <= bb.w / bb.h <= 3.0
    assert bb.x <= 300 <= bb.x2 and bb.y <= 200 <= bb.y2  # covers the source centre
    assert ts.line_h == pytest.approx(ts.glyph_h * T.CAPS_PITCH_RATIO)


def test_block_moves_off_dark_stripe():
    search = Rect(0, 0, 700, 400)
    source = Rect(300, 100, 200, 120)
    ink = np.zeros((search.h, search.w), np.uint8)
    ink[:, :300] = 1  # solid ink left of the source
    ts = place.typeset_block(TEXT, free_style(search, source, ink=ink), fake_measure)
    assert ts.fitted and ts.lines
    assert ts.bbox.x >= 300  # nothing lettered over the stripe
    assert ts.size == pytest.approx(20.0)  # the ceiling, unscaled


def test_block_avoids_obstacles():
    search = Rect(0, 0, 700, 400)
    source = Rect(300, 100, 100, 120)
    blocked = np.zeros((search.h, search.w), bool)
    obstacle = Rect(430, 0, 270, 400)  # another block's zone: everything right of x=430
    blocked[:, obstacle.x :] = True
    ts = place.typeset_block(TEXT, free_style(search, source, blocked=blocked), fake_measure)
    assert ts.fitted and ts.lines
    assert not ts.bbox.intersects(obstacle)
    for line in ts.lines:
        assert line.cx + line.width / 2.0 <= obstacle.x


def test_caption_keeps_covering_its_source():
    search = Rect(0, 0, 800, 200)
    source = Rect(200, 80, 400, 30)  # a horizontal caption line
    ts = place.typeset_block("CHAPTER 247: INHUMAN MAKYO SHINJUKU SHOWDOWN, PART 19", free_style(search, source, vertical=False), fake_measure)
    bb = ts.bbox
    assert bb.x <= 400 <= bb.x2 and bb.y <= 95 <= bb.y2
    assert len(ts.lines) <= 3 and bb.w / bb.h >= 2.0  # horizontal captions stay wide


def test_tiny_region_terminates_and_returns_lines():
    search = Rect(0, 0, 60, 40)
    source = Rect(20, 5, 20, 30)
    t0 = time.perf_counter()
    ts = place.typeset_block(TEXT, free_style(search, source), fake_measure)
    assert time.perf_counter() - t0 < 2.0
    assert ts.lines and ts.text.replace("-", "").replace(" ", "") == TEXT.replace(" ", "")


def test_bubble_uses_mask_and_stays_inside():
    box = Rect(100, 100, 300, 200)
    yy, xx = np.mgrid[0 : box.h, 0 : box.w]
    mask = ((xx - box.w / 2) / (box.w / 2)) ** 2 + ((yy - box.h / 2) / (box.h / 2)) ** 2 <= 1.0
    style = SegmentStyle(fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, text_height_px=30.0, vertical=True,
                         layout_box=box, layout_mask=mask, max_font_px=18.0, in_bubble=True)
    ts = place.typeset_block(TEXT, style, fake_measure)
    assert ts.fitted and len(ts.lines) >= 3
    bb = ts.bbox
    assert bb.x >= box.x and bb.x2 <= box.x2 and bb.y >= box.y and bb.y2 <= box.y2
    assert ts.size == pytest.approx(18.0)  # the ceiling, unscaled


def test_without_prepare_data_falls_back_to_layout_box():
    box = Rect(0, 0, 400, 300)
    style = free_style(box, Rect(150, 50, 40, 200))
    style.search_box = None
    style.ink_map = None
    style.blocked_map = None
    ts = place.typeset_block(TEXT, style, fake_measure)
    bb = ts.bbox
    assert ts.lines and bb.x >= 0 and bb.x2 <= 400 and bb.y >= 0 and bb.y2 <= 300


def test_hyphenate_widest_forces_a_break():
    words = "WHEN I ACTIVATED AMPLIFICATION DURING MY FIGHT".split()
    split = place.hyphenate_widest(words, 20.0, fake_measure, "en")
    if split is None:  # no pyphen dictionary in this environment
        pytest.skip("no hyphenation dictionary")
    new_words, head = split
    assert new_words[head].endswith("-") and new_words[head].startswith("AMPLI")
    assert "".join(new_words[head : head + 2]).replace("-", "") == "AMPLIFICATION"
    width = T.span_widths(new_words, 20.0, fake_measure)
    starts = T.balanced_breaks(len(new_words), 4, T.min_max_width(len(new_words), 4, width, (head,)) + 0.01, width, forced=(head,))
    assert head + 1 in starts  # the tail starts a new line


def test_shifted_moves_search_box_and_keeps_maps():
    style = free_style(Rect(10, 20, 100, 80), Rect(30, 30, 20, 40))
    moved = style.shifted(5, -7)
    assert moved.search_box == Rect(15, 13, 100, 80)
    assert moved.layout_box == Rect(15, 13, 100, 80) and moved.layout_seed == Rect(35, 23, 20, 40)
    assert moved.ink_map is style.ink_map and moved.blocked_map is style.blocked_map
    assert np.allclose(moved.source_quads, style.source_quads + np.array([5, -7], np.float32))
    # No field may be dropped by shifted(): whatever the original set, the copy has too.
    import dataclasses
    for f in dataclasses.fields(SegmentStyle):
        original = getattr(style, f.name)
        if original is not None and original is not False:
            assert getattr(moved, f.name) is not None, f.name


def test_size_ceiling_is_max_font_px_or_box_fraction():
    style = free_style(Rect(0, 0, 100, 100), Rect(10, 10, 20, 40), max_font_px=28.4)
    assert place.size_ceiling(style) == pytest.approx(28.4)
    style.max_font_px = None
    assert place.size_ceiling(style) == pytest.approx(32.0 * place._FALLBACK_EM_RATIO)
    assert place.size_ceiling(style, min_size=40.0) == 40.0


def test_compose_typeset_block_is_the_placement_geometry():
    """The PIL renderer and the Qt overlay both call compose.typeset_block;
    it must produce the same geometry as place.typeset_block."""
    compose = importlib.import_module("glasstranslate.render.compose")  # the package re-exports a function of that name

    style = free_style(Rect(0, 0, 600, 400), Rect(260, 120, 80, 160))
    a = compose.typeset_block(TEXT, style, fake_measure)
    b = place.typeset_block(TEXT, style, fake_measure)
    assert a == b
    assert a.size == pytest.approx(20.0)


def test_prepare_builds_maps_from_an_image():
    """A synthetic page: two panels split by a black gutter line; the block's
    own glyphs count as paper; the other panel is blocked."""
    from glasstranslate.core.types import Segment
    from glasstranslate.render.layout import TextBlock

    img = np.full((400, 600, 3), 255, np.uint8)
    img[195:200, :] = 0  # horizontal panel border
    img[300:330, 100:400] = 0  # solid black artwork in the lower panel
    src = Rect(250, 40, 30, 120)
    img[src.y : src.y2, src.x : src.x2] = 0  # the source glyphs
    quad = np.array([[src.x, src.y], [src.x2, src.y], [src.x2, src.y2], [src.x, src.y2]], np.float32)
    seg = Segment("テスト", quad, 0.9)
    style = SegmentStyle(fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, text_height_px=30.0, vertical=True,
                         layout_box=Rect(200, 20, 130, 160), layout_seed=src, source_quads=quad[None], max_font_px=18.0)
    block = TextBlock(seg, style, [seg])
    t0 = time.perf_counter()
    place.prepare(img, [block])
    assert time.perf_counter() - t0 < 1.0
    sb = style.search_box
    assert sb is not None and style.ink_map.shape == (sb.h, sb.w) == style.blocked_map.shape
    # Own glyphs are paper in the ink map; the black artwork is ink where visible.
    assert style.ink_map[src.y - sb.y : src.y2 - sb.y, src.x - sb.x : src.x2 - sb.x].max() == 0
    # The border line and everything below it (the other panel) are blocked.
    assert style.blocked_map[197 - sb.y, 250 - sb.x]
    if sb.y2 > 250:
        assert style.blocked_map[250 - sb.y, 250 - sb.x]
    assert not style.blocked_map[src.y - sb.y + 5, src.x - sb.x + 5]


# ------------------------------------------------------------ hyphenation and balance
def test_existing_hyphen_is_a_break_point_and_split_at_does_not_double_it():
    points = T.hyphenation_points("RE-CREATING")
    assert 3 in points  # right after the hyphen the word already has (no dictionary needed)
    assert T.split_at("RE-CREATING", 3) == ("RE-", "CREATING")
    assert T.split_at("DOMAIN", 2) == ("DO-", "MAIN")
    if 6 in points:  # syllable point inside the second part (needs the dictionary)
        assert T.split_at("RE-CREATING", 6) == ("RE-CRE-", "ATING")
    assert T.hyphenation_points("247") == [] and T.hyphenation_points("AS") == []


def test_word_is_never_split_twice_and_arbitrary_splits_wait_for_the_shrink():
    # "XXXXXXXXXXXX" has no dictionary break; at 20 px it is 120 px wide in an 80 px region.
    ts = T.typeset("XXXXXXXXXXXX IS LONG", T.rect_spans(Rect(0, 0, 80, 400)), 0.0, fake_measure, max_size=20, min_size=6)
    assert ts.fitted
    pieces = [l.text for l in ts.lines if "X" in l.text]
    assert len(pieces) <= 2  # one arbitrary split at most
    if len(pieces) == 2:  # the split was only allowed once the size had fallen to _FORCE_SPLIT_SCALE
        assert ts.size <= T._FORCE_SPLIT_SCALE * 20 + 1e-6
    else:
        assert 0.5 * ts.size * 12 <= 80  # or it shrank until the word fitted intact


def test_bubble_lines_are_balanced_no_orphan_last_word():
    # Greedy filling would give "AAAA BBBB CCCC DDDD" (95 px) + "EE" alone in a 100 px wide region.
    ts = T.typeset("AAAA BBBB CCCC DDDD EE", T.rect_spans(Rect(0, 0, 100, 200)), 0.0, fake_measure, max_size=10, min_size=6)
    assert ts.fitted and len(ts.lines) == 2
    assert all(len(l.text.split()) >= 2 for l in ts.lines)
    widths = [l.width for l in ts.lines]
    assert max(widths) / min(widths) < 1.6


def test_balanced_breaks_accepts_per_line_limits():
    words = "AA BB CC DD EE FF".split()
    width = T.span_widths(words, 10.0, fake_measure)  # 10 px per word, 5 px per space
    starts = T.balanced_breaks(len(words), 2, [60.0, 100.0], width, [60.0, 100.0])
    assert starts is not None and len(starts) == 3
    assert width(starts[0], starts[1]) <= 60.0 + 1e-6 and width(starts[1], starts[2]) <= 100.0 + 1e-6
    # Fewer lines are allowed when they suffice (the whole text fits the second line's limit)...
    assert T.balanced_breaks(len(words), 2, [9.0, 100.0], width) == [0, 6]
    # ...but a word wider than every line's limit has no solution.
    assert T.balanced_breaks(len(words), 2, [9.0, 50.0], width) is None


def _two_boxes():
    spans = np.zeros((200, 2), dtype=np.float64)
    spans[:100] = (200.0, 400.0)  # upper box, 200 px wide
    spans[100:] = (0.0, 150.0)  # lower box, 150 px wide, joined along row 100
    return spans


def test_lobe_split_row_and_typeset_lobes():
    spans = _two_boxes()
    assert T.lobe_split_row(spans) == 100
    assert T.lobe_split_row(T.rect_spans(Rect(0, 0, 300, 100))) is None
    text = "ONE TWO THREE FOUR... ...FIVE SIX SEVEN"
    ts = T.typeset_lobes(text, spans, 0.0, fake_measure, max_size=20)
    upper = [l for l in ts.lines if l.cx == pytest.approx(300.0)]
    lower = [l for l in ts.lines if l.cx == pytest.approx(75.0)]
    assert upper and lower and len(upper) + len(lower) == len(ts.lines)
    assert " ".join(l.text for l in upper) == "ONE TWO THREE FOUR..."
    assert " ".join(l.text for l in lower) == "...FIVE SIX SEVEN"
    # Inked rows (cap top .. descender) of each part stay inside its own box.
    assert max(l.top for l in upper) + T.DESCENT_RATIO * ts.glyph_h <= 100 + 1e-6
    assert min(l.top for l in lower) + T.CAP_TOP_RATIO * ts.glyph_h >= 100 - 1e-6
    # Without an ellipsis boundary the continuous flow is used unchanged.
    plain = "ONE TWO THREE FOUR FIVE SIX SEVEN"
    assert T.typeset_lobes(plain, spans, 0.0, fake_measure, max_size=20) == T.typeset(plain, spans, 0.0, fake_measure, max_size=20)


# ------------------------------------------------------------ placement costs and zones
def test_ragged_shape_loses_to_balanced_one():
    words = "IN OTHER WORDS, HE WIELDS HIS CURSED TECHNIQUE ALMOST AS EFFECTIVELY AS I WIELD MINE.".split()
    width = T.span_widths(words, 20.0, fake_measure)
    ragged = place._raggedness(len(words), 7, T.min_max_width(len(words), 7, width), width, ())
    even = place._raggedness(len(words), 9, T.min_max_width(len(words), 9, width), width, ())
    assert 0.0 <= even < ragged < 0.2
    assert place._raggedness(3, 1, 100.0, width, ()) == 0.0


def test_widest_word_is_hyphenated_at_the_ceiling_before_shrinking():
    if not T.hyphenation_points("EXTRAORDINARY"):
        pytest.skip("no hyphenation dictionary")
    search = Rect(0, 0, 130, 400)  # "EXTRAORDINARY" is 130 px at 20 px: it never fits with its halo
    source = Rect(40, 100, 50, 150)
    ts = place.typeset_block("AN EXTRAORDINARY MAN", free_style(search, source), fake_measure)
    assert ts.fitted and ts.size == pytest.approx(20.0)
    assert any(l.text.endswith("-") for l in ts.lines)
    assert ts.text.replace("-", "").replace(" ", "") == "ANEXTRAORDINARYMAN"


def test_zone_split_favours_the_block_against_the_border():
    """Two free-text sources side by side; a panel border right of the right
    one.  The right block gets the corridor between the sources (its zone
    reaches the left source's edge plus the gap); the left block yields."""
    from glasstranslate.core.types import Segment
    from glasstranslate.render.layout import TextBlock

    img = np.full((300, 600, 3), 255, np.uint8)
    img[:, 300:303] = 0  # vertical panel border
    a_src, b_src = Rect(100, 80, 60, 140), Rect(200, 80, 60, 140)
    blocks = []
    for src in (a_src, b_src):
        img[src.y : src.y2, src.x : src.x2] = 0
        quad = np.array([[src.x, src.y], [src.x2, src.y], [src.x2, src.y2], [src.x, src.y2]], np.float32)
        seg = Segment("テスト", quad, 0.9)
        style = SegmentStyle(fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, text_height_px=30.0, vertical=True,
                             layout_box=Rect(src.x - 40, src.y - 20, src.w + 80, src.h + 40), layout_seed=src,
                             source_quads=quad[None], max_font_px=20.0)
        blocks.append(TextBlock(seg, style, [seg]))
    place.prepare(img, blocks)
    a, b = blocks[0].style, blocks[1].style
    gap = place._ZONE_GAP_EM * 20.0
    row = 150

    def blocked(style, x):
        sb = style.search_box
        return bool(style.blocked_map[row - sb.y, x - sb.x])

    assert not blocked(b, int(a_src.x2 + gap) + 2)  # B may come right up to A's edge plus the gap...
    assert blocked(b, a_src.x2 + 2)  # ...but not into A's source
    assert blocked(a, a_src.x2 + 3) and not blocked(a, a_src.x2 - 5)  # A stops at its own source edge
    assert blocked(b, 296) and not blocked(b, 280)  # the border line and its margin stay off limits
