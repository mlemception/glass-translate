"""Tests for the corpus index walker, the structural descriptors and the pairing.

Everything runs on synthetic pages built with numpy/cv2 and a synthetic ``.cbz`` in
``tmp_path``: no corpus page, no archive on disk, no OCR, no GPU and no network.  The
real corpus is licensed third-party material and must never be needed by a test.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import corpus_descriptor as CD  # noqa: E402
import corpus_index as CI  # noqa: E402
import corpus_pair as CP  # noqa: E402


# --------------------------------------------------------------------- fixtures
def _page(seed: int, w: int = 300, h: int = 450, colour: bool = False) -> np.ndarray:
    """A synthetic manga-ish page: paper, a border, and a few panel rectangles."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(img, (12, 12), (w - 12, h - 12), (0, 0, 0), 2)
    for _ in range(4):
        x0 = int(rng.integers(20, w - 90))
        y0 = int(rng.integers(20, h - 120))
        x1 = x0 + int(rng.integers(50, 80))
        y1 = y0 + int(rng.integers(60, 110))
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 3)
        shade = int(rng.integers(90, 200))
        cv2.rectangle(img, (x0 + 5, y0 + 5), (x1 - 5, y1 - 5), (shade,) * 3, -1)
    if colour:
        img[:, :, 2] = np.clip(img[:, :, 2].astype(np.int16) + 70, 0, 255).astype(np.uint8)
    return img


def _lettered(img: np.ndarray, seed: int) -> np.ndarray:
    """The same artwork with different small ink on it - what a translation changes."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    for _ in range(14):
        x = int(rng.integers(25, out.shape[1] - 40))
        y = int(rng.integers(25, out.shape[0] - 20))
        cv2.rectangle(out, (x, y), (x + 12, y + 6), (0, 0, 0), -1)
    return out


def _encode(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


def _en_name(chapter: int, volume: int, page: int, page2: Optional[int] = None) -> str:
    span = f"p{page:03d}" if page2 is None else f"p{page:03d}-p{page2:03d}"
    return (f"Jujutsu Kaisen - c{chapter:03d} (v{volume:02d}) - {span}"
            " [VIZ Media] [Digital] [1r0n].png")


@pytest.fixture()
def en_cbz(tmp_path: Path) -> Path:
    """A four-entry EN volume: a colour cover, two singles and one joined spread."""
    path = tmp_path / "Jujutsu Kaisen v01 (2019) (Digital) (1r0n).cbz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(_en_name(1, 1, 0), _encode(_page(1, colour=True)))
        z.writestr(_en_name(1, 1, 1), _encode(_page(2)))
        z.writestr(_en_name(1, 1, 2), _encode(_page(3)))
        spread = np.concatenate([_page(4), _page(5)], axis=1)
        z.writestr(_en_name(1, 1, 6, 7), _encode(spread))
    return path


# --------------------------------------------------------------------- index walker
def test_index_walker_reads_names_sizes_and_identity(en_cbz: Path) -> None:
    archive = CI.open_archive(en_cbz)
    try:
        refs = CI.enumerate_archive(archive, "en", workers=1)
    finally:
        archive.close()
    assert len(refs) == 4
    by_page = {r.page: r for r in refs}
    assert sorted(by_page) == [0, 1, 2, 6]
    cover = by_page[0]
    assert cover.volume == 1 and cover.chapter == "001" and cover.side == "en"
    assert cover.is_colour and cover.colour_metric > CI.COLOUR_METRIC_THRESHOLD
    assert (cover.width, cover.height) == (300, 450) and cover.fmt == "png"
    # crc + size are the cache identity and come free from the directory listing.
    assert all(r.size > 0 for r in refs)
    assert len({r.crc for r in refs}) == 4
    assert not by_page[1].is_colour


def test_index_walker_marks_the_named_spread(en_cbz: Path) -> None:
    archive = CI.open_archive(en_cbz)
    try:
        refs = CI.enumerate_archive(archive, "en", workers=1)
    finally:
        archive.close()
    spread = next(r for r in refs if r.page == 6)
    assert spread.is_spread and spread.page2 == 7
    assert all(not r.is_spread for r in refs if r.page != 6)


def test_parse_en_name_handles_singles_spreads_and_the_v30_fallback() -> None:
    single = CI.parse_en_name(_en_name(7, 2, 42))
    assert single == {"volume": 2, "chapter": "007", "page": 42, "page2": None,
                      "named_spread": False}
    spread = CI.parse_en_name(_en_name(7, 2, 42, 43))
    assert spread["page2"] == 43 and spread["named_spread"] is True
    # v30 is a different scanlator: flat NNNN.ext names, volume only in the file name.
    flat = CI.parse_en_name("0012.jpg", "Jujutsu Kaisen v30 (2026) (Digital) (Rillant).cbz")
    assert flat["volume"] == 30 and flat["page"] == 12 and flat["chapter"] is None


def test_page_id_is_ascii_safe_even_for_japanese_entries() -> None:
    ref = CI.PageRef(side="ja", archive="JA.rar", entry="第08巻/ページ.jpg",
                     group="第08巻", volume=8, chapter=None, page=12, page2=None,
                     is_spread=False, width=764, height=1200, fmt="jpeg",
                     colour_metric=3.0, is_colour=False, crc=0xABCDEF12, size=1234)
    assert ref.page_id.isascii()
    assert ref.page_id == "ja_v08_p012_abcdef12"


def test_canonical_ja_and_exclusions() -> None:
    assert CI.ja_volume("x/DLRAW.TO_Jujutsu Kaisen v08") == 8
    assert CI.ja_volume("x/DLRAW.TO_[..] ..第14巻") == 14
    assert CI.ja_volume("x/DLRAW.TO_Jujutsu Kaisen v99") is None
    assert CI.is_excluded_ja("a/DLRAW.TO_Jujutsu Kaisen vOfficial Fanbook")
    assert CI.is_excluded_ja("a/DLRAW.TO_Jujutsu Kaisen v15-16s/b")
    assert not CI.is_excluded_ja("a/DLRAW.TO_Jujutsu Kaisen v08")


def test_colourfulness_separates_grey_from_colour() -> None:
    grey = np.full((60, 40, 3), 128, np.uint8)
    assert CI.colourfulness(grey) == pytest.approx(0.0, abs=0.01)
    colour = grey.copy()
    colour[:, :, 0] = 20
    assert CI.colourfulness(colour) > CI.COLOUR_METRIC_THRESHOLD
    assert CI.colourfulness(None) == 0.0


def test_split_spread_returns_the_right_half_first() -> None:
    left = np.zeros((10, 6), np.uint8)
    right = np.full((10, 6), 255, np.uint8)
    joined = np.concatenate([left, right], axis=1)
    first, second = CI.split_spread(joined)
    assert first.mean() == 255 and second.mean() == 0


def test_is_spread_uses_the_name_or_the_aspect() -> None:
    assert CI.is_spread(1500, 2250, named_spread=True)
    assert CI.is_spread(3000, 2250)
    assert not CI.is_spread(1500, 2250)


# --------------------------------------------------------------------- descriptors
def test_descriptor_is_near_one_for_the_same_art_and_low_for_another_page() -> None:
    art = cv2.cvtColor(_page(11), cv2.COLOR_BGR2GRAY)
    same = CD.describe(art)
    relettered = CD.describe(cv2.cvtColor(_lettered(_page(11), 99), cv2.COLOR_BGR2GRAY))
    other = CD.describe(cv2.cvtColor(_page(12), cv2.COLOR_BGR2GRAY))
    assert CD.similarity(same, same) == pytest.approx(1.0, abs=1e-5)
    # Re-lettering the same artwork must barely move the descriptor...
    assert CD.similarity(same, relettered) > 0.9
    # ...while a different page must score far below it.
    assert CD.similarity(same, other) < CD.similarity(same, relettered) - 0.2


def test_descriptor_survives_a_large_resolution_difference() -> None:
    art = cv2.cvtColor(_page(21), cv2.COLOR_BGR2GRAY)
    big = cv2.resize(art, (art.shape[1] * 3, art.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    assert CD.similarity(CD.describe(art), CD.describe(big)) > 0.9


def test_mirrored_page_scores_below_the_unflipped_one() -> None:
    art = cv2.cvtColor(_page(31), cv2.COLOR_BGR2GRAY)
    a, b = CD.describe(art), CD.describe(art.copy())
    assert CD.mirror_similarity(a, b) < CD.similarity(a, b)


def test_content_box_strips_a_blank_border() -> None:
    page = np.full((200, 160), 250, np.uint8)
    cv2.rectangle(page, (40, 50), (120, 150), 0, -1)
    x, y, w, h = CD.content_box(page)
    assert 35 <= x <= 45 and 45 <= y <= 55
    assert 75 <= w <= 90 and 95 <= h <= 110


def test_similarity_matrix_matches_pairwise_similarity() -> None:
    rows = [CD.describe(cv2.cvtColor(_page(s), cv2.COLOR_BGR2GRAY)) for s in (41, 42)]
    cols = [CD.describe(cv2.cvtColor(_page(s), cv2.COLOR_BGR2GRAY)) for s in (42, 43, 41)]
    matrix = CD.similarity_matrix(rows, cols)
    assert matrix.shape == (2, 3)
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            assert matrix[i, j] == pytest.approx(CD.similarity(a, b), abs=1e-5)


def test_cache_key_changes_with_every_input_that_changes_the_numbers() -> None:
    base = CD.cache_key("a.rar", "p1.jpg", 123, 456)
    assert base == CD.cache_key("a.rar", "p1.jpg", 123, 456)
    assert base != CD.cache_key("b.rar", "p1.jpg", 123, 456)
    assert base != CD.cache_key("a.rar", "p2.jpg", 123, 456)
    assert base != CD.cache_key("a.rar", "p1.jpg", 124, 456)
    assert base != CD.cache_key("a.rar", "p1.jpg", 123, 457)


def test_descriptor_cache_round_trip_preserves_similarity(tmp_path: Path) -> None:
    desc = CD.describe(cv2.cvtColor(_page(51), cv2.COLOR_BGR2GRAY))
    cache = CD.DescriptorCache(tmp_path)
    cache.put("a.rar", "grp", "k1", desc)
    assert cache.flush("a.rar", "grp") is not None
    reloaded = CD.DescriptorCache(tmp_path).get("a.rar", "grp", "k1")
    assert reloaded is not None
    # int8 quantisation must not move the score materially.
    assert CD.similarity(desc, reloaded) > 0.999
    assert (reloaded.width, reloaded.height) == (desc.width, desc.height)


# --------------------------------------------------------------------- alignment
def _sim_from(ja_seeds, en_seeds) -> np.ndarray:
    ja = [CD.describe(cv2.cvtColor(_page(s), cv2.COLOR_BGR2GRAY)) for s in ja_seeds]
    en = [CD.describe(cv2.cvtColor(_lettered(_page(s), s + 500), cv2.COLOR_BGR2GRAY))
          for s in en_seeds]
    return CD.similarity_matrix(ja, en)


def test_needleman_wunsch_recovers_the_pairing_when_en_has_extra_pages() -> None:
    # EN carries two extra front-matter pages (seeds 90, 91) the JA raw does not.
    sim = _sim_from([61, 62, 63], [90, 61, 62, 91, 63])
    pairs, ja_gaps, en_gaps = CP.needleman_wunsch(sim)
    assert pairs == [(0, 1), (1, 2), (2, 4)]
    assert ja_gaps == []
    assert sorted(en_gaps) == [0, 3]


def test_needleman_wunsch_skips_a_ja_only_page() -> None:
    sim = _sim_from([71, 72, 73], [71, 73])
    pairs, ja_gaps, en_gaps = CP.needleman_wunsch(sim)
    assert pairs == [(0, 0), (2, 1)]
    assert ja_gaps == [1] and en_gaps == []


def test_needleman_wunsch_is_monotone_and_deterministic() -> None:
    sim = _sim_from([81, 82, 83, 84], [81, 82, 83, 84])
    first = CP.needleman_wunsch(sim)
    assert CP.needleman_wunsch(sim) == first
    pairs = first[0]
    assert [i for i, _ in pairs] == sorted(i for i, _ in pairs)
    assert [j for _, j in pairs] == sorted(j for _, j in pairs)


def test_needleman_wunsch_handles_empty_sides() -> None:
    pairs, ja_gaps, en_gaps = CP.needleman_wunsch(np.zeros((0, 3), np.float32))
    assert pairs == [] and ja_gaps == [] and en_gaps == [0, 1, 2]


def test_row_margin_is_the_gap_to_the_next_best_column() -> None:
    sim = np.array([[0.2, 0.9, 0.3, 0.5]], np.float32)
    assert CP.row_margin(sim, 0, 1) == pytest.approx(0.9 - 0.5, abs=1e-6)


def test_row_margin_sees_the_adjacent_page() -> None:
    """The neighbour must NOT be masked out of the margin.

    An off-by-one against the next page is the exact failure the alignment exists to
    prevent, and ~92 % of pairs are accepted on this margin without ever reaching ORB.
    A page whose neighbour scores almost as well has to fall through to the slow gate.
    """
    sim = np.array([[0.20, 0.95, 0.94, 0.20]], np.float32)
    margin = CP.row_margin(sim, 0, 1)
    assert margin == pytest.approx(0.95 - 0.94, abs=1e-6)
    assert margin < CP.FAST_MARGIN  # so the fast gate cannot fire

    decision = CP.decide_pair(_slot(side="ja"), _slot(), combined=0.95, margin=margin,
                              mirrored=False, volume_mirrored=False,
                              verify=lambda: (900, 0.9))
    assert decision.status == CP.STATUS_VERIFIED  # ORB had to confirm it


# --------------------------------------------------------------------- decision rule
def _slot(width: int = 1500, height: int = 2250, half: Optional[int] = None, page: int = 5,
          side: str = "en", spread: bool = False) -> CP.PageSlot:
    ref = CI.PageRef(side=side, archive="a", entry=f"{side}{page}.png", group="g",
                     volume=1, chapter="001", page=page, page2=None, is_spread=spread,
                     width=width, height=height, fmt="png", colour_metric=0.0,
                     is_colour=False, crc=page, size=100)
    return CP.PageSlot(ref, half)


def test_decide_pair_fast_accepts_a_confident_match() -> None:
    def _never() -> None:
        raise AssertionError("the fast gate must not call ORB")

    d = CP.decide_pair(_slot(side="ja"), _slot(), combined=0.80, margin=0.40,
                       mirrored=False, volume_mirrored=False, verify=_never)
    assert d.status == CP.STATUS_FAST and d.inliers == 0
    assert "0.800" in d.reason


def test_decide_pair_rejects_a_mismatch_that_orb_cannot_confirm() -> None:
    d = CP.decide_pair(_slot(side="ja"), _slot(), combined=0.40, margin=0.05,
                       mirrored=False, volume_mirrored=False, verify=lambda: (3, 0.68))
    assert d.status == CP.STATUS_UNVERIFIABLE and d.inliers == 3
    assert "no shared structure" in d.reason
    # A high Pearson score must not rescue it: true/false overlap on that statistic.
    assert d.align_score == pytest.approx(0.68)


def test_decide_pair_verifies_a_weak_descriptor_score_through_orb() -> None:
    strong = CP.decide_pair(_slot(side="ja"), _slot(), combined=0.37, margin=0.10,
                            mirrored=False, volume_mirrored=False,
                            verify=lambda: (1009, 0.92))
    assert strong.status == CP.STATUS_VERIFIED and strong.inliers == 1009
    weak = CP.decide_pair(_slot(side="ja"), _slot(), combined=0.37, margin=0.10,
                          mirrored=False, volume_mirrored=False, verify=lambda: (90, 0.7))
    assert weak.status == CP.STATUS_WEAK and "weak" in weak.reason


def test_decide_pair_marks_an_aspect_mismatch_unverifiable() -> None:
    joined_spread = _slot(width=3000, height=2250)  # not split: twice as wide
    d = CP.decide_pair(_slot(side="ja"), joined_spread, combined=0.9, margin=0.5,
                       mirrored=False, volume_mirrored=False, verify=lambda: (900, 0.9))
    assert d.status == CP.STATUS_UNVERIFIABLE and "aspect mismatch" in d.reason


def test_decide_pair_splitting_the_spread_makes_the_aspect_agree() -> None:
    half = _slot(width=3000, height=2250, half=0, spread=True)
    assert half.aspect == pytest.approx(1500 / 2250)
    d = CP.decide_pair(_slot(side="ja"), half, combined=0.9, margin=0.5,
                       mirrored=False, volume_mirrored=False, verify=lambda: (900, 0.9))
    assert d.status == CP.STATUS_FAST


def test_decide_pair_flags_an_unexpectedly_mirrored_page() -> None:
    d = CP.decide_pair(_slot(side="ja"), _slot(), combined=0.9, margin=0.5,
                       mirrored=True, volume_mirrored=False, verify=lambda: (900, 0.9))
    assert d.status == CP.STATUS_UNVERIFIABLE and "flipped" in d.reason
    # ...unless the whole volume is mirrored, which is a per-volume property.
    ok = CP.decide_pair(_slot(side="ja"), _slot(), combined=0.9, margin=0.5,
                        mirrored=True, volume_mirrored=True, verify=lambda: (900, 0.9))
    assert ok.status == CP.STATUS_FAST


def test_build_slots_expands_only_spreads() -> None:
    refs = [_slot(page=1).ref, _slot(page=2, spread=True).ref, _slot(page=3).ref]
    slots = CP.build_slots(refs)
    assert [s.half for s in slots] == [None, 0, 1, None]
    assert slots[1].slot_id.endswith("h0") and slots[2].slot_id.endswith("h1")


def test_verified_pairs_filters_by_status() -> None:
    payload = {"volumes": {"8": {"volume": 8, "ja_group": "g", "en_archive": "a", "pairs": [
        {"page_id": "a", "status": CP.STATUS_FAST},
        {"page_id": "b", "status": CP.STATUS_WEAK},
        {"page_id": "c", "status": CP.STATUS_UNVERIFIABLE},
        {"page_id": "d", "status": CP.STATUS_VERIFIED},
    ]}}}
    assert [p["page_id"] for p in CP.verified_pairs(payload)] == ["a", "b", "d"]
    assert [p["page_id"] for p in CP.verified_pairs(payload, include_weak=False)] == ["a", "d"]
