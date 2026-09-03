"""Tests for the demo scripts (no OCR engine, models, GPU or display needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

import make_images  # noqa: E402
import run_demo  # noqa: E402
from glasstranslate.core.types import Segment, SegmentStyle, StyledSegment  # noqa: E402
from glasstranslate.translate import IdentityTranslator  # noqa: E402


def _styled(text: str, x: int, y: int, w: int, h: int) -> StyledSegment:
    quad = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    style = SegmentStyle(fg=(200, 30, 30), bg=(255, 255, 255), angle_deg=0.0, text_height_px=float(h))
    return StyledSegment(segment=Segment(text=text, quad=quad, confidence=0.99), style=style, src_lang="en")


def test_make_images_writes_three_pngs(tmp_path: Path) -> None:
    sys.argv = ["make_images.py", "--out", str(tmp_path)]
    make_images.main()
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == ["dark_ui.png", "manga.png", "rotated.png", "webpage.png"]
    for gen in make_images.GENERATORS.values():
        img = gen()
        assert img.mode == "RGB" and img.width >= 900


def test_draw_rotated_text_places_ink_at_center() -> None:
    from PIL import Image

    canvas = Image.new("RGBA", (400, 300), (255, 255, 255, 255))
    make_images.draw_rotated_text(canvas, (200, 150), "Hello", make_images.font("arial", 40), (255, 0, 0), 20.0)
    arr = np.asarray(canvas)
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80)
    ys, xs = np.nonzero(red)
    assert red.any()
    assert abs(xs.mean() - 200) < 15 and abs(ys.mean() - 150) < 15


def test_list_images_filters_and_sorts(tmp_path: Path) -> None:
    for name in ("b.png", "a.jpg", "notes.txt", "c.PNG"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in run_demo.list_images(tmp_path)] == ["a.jpg", "b.png", "c.PNG"]


def test_translate_segments_identity_and_same_language() -> None:
    styled = [_styled("Hello", 10, 10, 80, 20), _styled("World", 10, 40, 80, 20)]
    out = run_demo.translate_segments(IdentityTranslator(), styled, "en", "de")
    assert [t.translation for t in out] == ["Hello", "World"]
    assert all(t.tgt_lang == "de" for t in out)
    out = run_demo.translate_segments(IdentityTranslator(), styled, "de", "de")
    assert [t.translation for t in out] == ["Hello", "World"]


def test_build_translator_falls_back_to_identity(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_demo.build_translator("identity", tmp_path, "cpu", "en", "de").name == "identity"
    tr = run_demo.build_translator("argos", tmp_path, "cpu", "en", "de")  # empty models dir
    assert tr.name == "identity"
    assert "no Argos packages" in capsys.readouterr().err


def test_draw_debug_outlines_quad_in_fg_colour() -> None:
    img = np.full((200, 300, 3), 255, np.uint8)
    styled = [_styled("Hello", 50, 80, 120, 30)]
    out = run_demo.draw_debug(img, styled)
    assert out.shape == img.shape and out is not img
    # Outline pixel on the top edge must be the fg colour (200,30,30) in BGR.
    assert tuple(int(c) for c in out[80, 110]) == (30, 30, 200)
    # Label was drawn somewhere above the quad.
    assert (out[40:78, 40:300] != 255).any()


def test_hex_formatting() -> None:
    assert run_demo._hex((255, 0, 16)) == "#ff0010"
