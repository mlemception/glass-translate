"""Tests for the page-resolution helpers of ``demo/typeset_dev.py`` (no OCR engine,
models, GPU or display needed): per-stem caches and reference files, automatic
reference-image lookup, the batch page list and the automatic per-block regions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import typeset_dev as TD  # noqa: E402
from glasstranslate.core.types import Rect, Segment, SegmentStyle  # noqa: E402
from glasstranslate.render.layout import TextBlock  # noqa: E402


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_reference_image_for_known_pairs(tmp_path: Path) -> None:
    ja = _touch(tmp_path / "3jp.jpg")
    eng = _touch(tmp_path / "3eng.jpg")
    assert TD.reference_image_for(ja) == eng
    ja1 = _touch(tmp_path / "1ja.jpg")
    eng1 = _touch(tmp_path / "1eng.jpg")
    assert TD.reference_image_for(ja1) == eng1
    before = TD.PROJECT_ROOT / "Examples" / "before.jpg"
    assert TD.reference_image_for(before) == TD.PROJECT_ROOT / "Examples" / "after.webp"
    assert TD.reference_image_for(_touch(tmp_path / "other.png")) is None
    assert TD.reference_image_for(_touch(tmp_path / "7ja.jpg")) is None  # no 7eng.jpg


def test_page_paths_are_per_stem() -> None:
    assert TD.page_stem(Path("more_comparisons/3jp.jpg")) == "3jp"
    assert TD.cache_dir("3jp") == TD.CACHE_ROOT / "3jp"
    assert TD.reference_dir("3jp") == TD.REFERENCE_ROOT / "3jp"
    assert TD.reference_text_path("3jp") == TD.DEMO_DIR / "reference_text_3jp.json"


def test_reference_text_falls_back_to_legacy_file_for_before(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TD, "DEMO_DIR", tmp_path)
    legacy = tmp_path / "reference_text.json"
    legacy.write_text('{"_comment": "x", "領域": "Domain"}', encoding="utf-8")
    assert TD.reference_text_path("before") == legacy
    assert TD.reference_text_path("3jp") == tmp_path / "reference_text_3jp.json"  # never the legacy file
    new = tmp_path / "reference_text_before.json"
    new.write_text("{}", encoding="utf-8")
    assert TD.reference_text_path("before") == new


def test_load_reference_text_skips_meta_keys_and_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TD, "DEMO_DIR", tmp_path)
    assert TD.load_reference_text("nothing") == {}
    (tmp_path / "reference_text_p.json").write_text(
        json.dumps({"_comment": "meta", "_uncertain": ["a"], "宿儺": "SUKUNA", "": "blank"}), encoding="utf-8"
    )
    assert TD.load_reference_text("p") == {"宿儺": "SUKUNA"}


def test_all_pairs_lists_before_first_then_the_comparison_pages() -> None:
    pairs = TD.all_pairs()
    names = [p.name for p in pairs]
    assert names[0] == "before.jpg"
    assert {"1ja.jpg", "2ja.jpg", "3jp.jpg", "4ja.jpg"} <= set(names)
    assert not any("eng" in n for n in names)
    assert len(pairs) == 5


def _block(text: str, rect: Rect, *, bubble: Rect | None = None, outline: bool = False, em: float = 24.0) -> TextBlock:
    quad = np.array([[rect.x, rect.y], [rect.x2, rect.y], [rect.x2, rect.y2], [rect.x, rect.y2]], np.float32)
    seg = Segment(text, quad, 0.9)
    style = SegmentStyle(fg=(0, 0, 0), bg=(255, 255, 255), angle_deg=0.0, text_height_px=em, vertical=True,
                         layout_box=bubble or rect, in_bubble=bubble is not None, outline=outline)
    return TextBlock(seg, style, [seg], [], bubble, em)


def test_block_regions_one_per_block_grown_and_clipped() -> None:
    blocks = [
        _block("一", Rect(100, 100, 30, 120), bubble=Rect(80, 80, 70, 160)),
        _block("二", Rect(300, 350, 30, 40), outline=True),
        _block("三", Rect(5, 5, 20, 60)),
    ]
    regions = TD.block_regions(blocks, (400, 360))  # (height, width)
    assert list(regions) == ["b0_bubble", "b1_art", "b2_flat"]
    for name, (x1, y1, x2, y2) in regions.items():
        assert 0 <= x1 < x2 <= 360 and 0 <= y1 < y2 <= 400, name
    bx1, by1, bx2, by2 = regions["b0_bubble"]
    assert bx1 <= 80 and by1 <= 80 and bx2 >= 150 and by2 >= 240  # covers the bubble
    ax1, ay1, ax2, ay2 = regions["b1_art"]
    assert ax1 < 300 and ay1 < 350 and ax2 > 330 and ay2 > 390  # grown around the source
    assert ax2 - ax1 >= TD.REGION_MIN_PX or ax2 == 360
    assert regions["b2_flat"][0] == 0 and regions["b2_flat"][1] == 0  # clipped at the page edge


def test_page_regions_keeps_the_hand_picked_before_regions() -> None:
    blocks = [_block("一", Rect(100, 100, 30, 120))]
    assert TD.page_regions("before", blocks, (1600, 1096)) == TD.REGIONS_BY_STEM["before"]
    auto = TD.page_regions("3jp", blocks, (1200, 764))
    assert list(auto) == ["b0_flat"]


def test_comparison_sheet_uses_aligned_reference_pixels_directly() -> None:
    orig = Image.new("L", (200, 100), 255)
    ours = Image.new("L", (200, 100), 200)
    ref = Image.new("L", (200, 100), 0)  # same size as the page: an aligned reference
    sheet = TD.comparison_sheet(orig, ours, ref, (0, 0, 100, 50))
    arr = np.asarray(sheet)
    tile = TD.TILE_W
    assert sheet.width == 3 * tile + 12
    assert arr[:, :tile].mean() > 250 and 190 < arr[:, tile + 6: 2 * tile + 6].mean() < 210
    assert arr[:, 2 * tile + 12:].mean() < 5
    # A reference of another size is mapped by scale (the legacy behaviour).
    big = Image.new("L", (400, 200), 0)
    assert np.asarray(TD.comparison_sheet(orig, ours, big, (0, 0, 100, 50)))[:, 2 * tile + 12:].mean() < 5
