"""The blind-panel builder's three guards (``demo/corpus_panel.py``).

This tool decides nothing by itself, but every verdict in
``docs/perf/2026-09-15-typeset-corpus.md`` rests on it, so the things it refuses
to do are worth pinning.  Each guard exists because the tool once did the thing:

* it produced a panel of 4 strips across 2 pages without comment, which is below
  the protocol floor and therefore decides nothing;
* it left 117 KB and 106 KB strips on disk while appearing to enforce a 100 KB
  limit, because it resized *after* its final save;
* it showed a judge an ``eng_aligned.png`` that did not register with our page,
  and the judge confidently reported a defect that was an artifact.

``demo`` is not a package, so the module is loaded by path.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SRC = Path(__file__).resolve().parents[1] / "demo" / "corpus_panel.py"
_spec = importlib.util.spec_from_file_location("corpus_panel", _SRC)
panel = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(panel)


def _rec(page: str, index: int, diff: int = 100, role: str = "affected") -> dict:
    return {"page": page, "index": index, "rect": (0, 0, 40, 40), "diff_px": diff,
            "from_reference": True, "old_from_reference": True, "kind": "bubble", "role": role}


# --------------------------------------------------------- the repo guard
def test_a_destination_inside_the_repo_is_refused():
    """The corpus is licensed; no crop of it may enter the tree."""
    assert panel.check_out_dir(panel.ROOT / "demo" / "output" / "panel") is not None
    assert panel.check_out_dir(panel.ROOT) is not None


def test_a_destination_outside_the_repo_is_accepted(tmp_path):
    assert panel.check_out_dir(tmp_path / "panel") is None


def test_main_refuses_before_doing_any_work(tmp_path, monkeypatch):
    """The guard runs first, so a bad destination costs nothing."""
    def explode(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("changed() ran despite a bad destination")

    monkeypatch.setattr(panel, "changed", explode)
    inside = panel.ROOT / "demo" / "output" / "nope"
    assert panel.main(["old", "new", str(inside)]) == 2


# ------------------------------------------------------- the protocol floor
def test_a_panel_over_too_few_pages_is_not_judgeable():
    """The failure this run actually hit: 5 blocks, but only 2 pages."""
    chosen = [_rec("p1", i) for i in range(4)] + [_rec("p2", i) for i in range(2)]
    chosen += [_rec("p1", 9, 0, "control"), _rec("p2", 9, 0, "control")]
    why = panel.judgeable(chosen, {"p1", "p2"})
    assert why is not None and "page(s)" in why


def test_a_panel_with_too_few_blocks_is_not_judgeable():
    chosen = [_rec(f"p{i}", 0) for i in range(3)] + [_rec("p0", 9, 0, "control")] * 2
    assert panel.judgeable(chosen, {"p0", "p1", "p2"}) is not None


def test_a_panel_without_its_controls_is_not_judgeable():
    """Controls catch a judge that confabulates a preference between identical images."""
    chosen = [_rec(f"p{i}", 0) for i in range(6)]
    assert panel.judgeable(chosen, {f"p{i}" for i in range(6)}) is not None


def test_a_panel_meeting_the_protocol_is_judgeable():
    chosen = [_rec(f"p{i}", 0) for i in range(6)]
    chosen += [_rec("p0", 9, 0, "control"), _rec("p1", 9, 0, "control")]
    assert panel.judgeable(chosen, {f"p{i}" for i in range(6)}) is None


def test_selection_spreads_over_pages_before_taking_seconds():
    """Otherwise the six loudest blocks can all come from one page."""
    hot = [_rec("loud", i, diff=1000 - i) for i in range(10)]
    hot += [_rec(f"quiet{i}", 0, diff=5) for i in range(5)]
    chosen, seen = panel.select(hot, [], random.Random(0))
    assert len(seen) >= panel.MIN_PAGES
    assert sum(1 for c in chosen if c["page"] == "loud") < len(chosen)


def test_a_block_whose_reference_lookup_moved_is_not_eligible():
    """Block grouping wobbles ~2 in 316; that text change is not the slice."""
    wobbled = {**_rec("p1", 0), "from_reference": True, "old_from_reference": False}
    chosen, _ = panel.select([wobbled], [], random.Random(0))
    assert chosen == []


# ------------------------------------------------------------- the floor test
def _write_page(root: Path, page: str, kept: np.ndarray, orig_ink: np.ndarray) -> None:
    (root / "reference" / page).mkdir(parents=True, exist_ok=True)
    (root / "pages").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(kept, 255, 0).astype(np.uint8)).save(
        root / "reference" / page / "kept_art.png")
    Image.fromarray(np.where(orig_ink, 0, 255).astype(np.uint8)).save(
        root / "pages" / f"{page}.png")


def test_a_page_whose_reference_does_not_register_is_dropped(tmp_path, monkeypatch):
    """96.3 % of one real page's kept-art is not ink in our own original."""
    monkeypatch.setattr(panel, "CORPUS", tmp_path)
    kept = np.zeros((200, 200), bool)
    kept[:60, :100] = True  # 6000 px of "kept art"
    _write_page(tmp_path, "bad", kept, np.zeros((200, 200), bool))  # none of it is our ink
    assert panel.registers("bad") is False


def test_a_page_whose_reference_registers_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "CORPUS", tmp_path)
    kept = np.zeros((200, 200), bool)
    kept[:60, :100] = True
    ink = kept.copy()
    ink[:2, :10] = False  # a sliver missing is fine
    _write_page(tmp_path, "good", kept, ink)
    assert panel.registers("good") is True


def test_a_page_with_almost_no_kept_art_is_kept(tmp_path, monkeypatch):
    """A tiny denominator makes the floor meaningless, not damning.

    Real pages divide by as few as 4 pixels; dropping them on a ratio computed
    from that would discard good pages for noise.
    """
    monkeypatch.setattr(panel, "CORPUS", tmp_path)
    kept = np.zeros((200, 200), bool)
    kept[0, :20] = True  # 20 px, far below FLOOR_MIN_KEPT
    _write_page(tmp_path, "tiny", kept, np.zeros((200, 200), bool))
    assert panel.registers("tiny") is True


def test_a_page_with_no_kept_art_mask_at_all_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "CORPUS", tmp_path)
    assert panel.registers("missing") is True


# --------------------------------------------------------------- the size cap
def test_every_strip_lands_under_the_hard_byte_cap(tmp_path, monkeypatch):
    """Noise does not compress, so this is the case that used to breach it."""
    monkeypatch.setattr(panel, "CORPUS", tmp_path)
    rng = np.random.default_rng(0)
    page = "pg"
    (tmp_path / "renders" / page).mkdir(parents=True, exist_ok=True)
    (tmp_path / "reference" / page).mkdir(parents=True, exist_ok=True)
    for name in ("old_typeset.png", "new_typeset.png"):
        Image.fromarray(rng.integers(0, 256, (1400, 1400), dtype=np.uint8)).save(
            tmp_path / "renders" / page / name)
    Image.fromarray(rng.integers(0, 256, (1400, 1400), dtype=np.uint8)).save(
        tmp_path / "reference" / page / "eng_aligned.png")

    rec = {**_rec(page, 0), "rect": (0, 0, 1400, 1400), "pair": 1}
    line = panel.strip(rec, "old", "new", tmp_path, random.Random(0))
    assert (tmp_path / "pair01.png").stat().st_size < panel.MAX_BYTES
    assert "A=" in line and "B=" in line


def test_the_key_line_names_which_side_is_which(tmp_path, monkeypatch):
    """The key is the only record of the randomisation; it must be unambiguous."""
    monkeypatch.setattr(panel, "CORPUS", tmp_path)
    page = "pg"
    (tmp_path / "renders" / page).mkdir(parents=True, exist_ok=True)
    for name in ("old_typeset.png", "new_typeset.png"):
        Image.fromarray(np.full((40, 40), 200, np.uint8)).save(tmp_path / "renders" / page / name)
    rec = {**_rec(page, 3), "pair": 7}
    line = panel.strip(rec, "old", "new", tmp_path, random.Random(1))
    assert line.startswith("pair07.png")
    assert ("A=old B=new" in line) or ("A=new B=old" in line)


@pytest.mark.parametrize("seed", range(8))
def test_the_order_is_reseeded_per_strip_not_once_per_panel(tmp_path, monkeypatch, seed):
    """Both orders must be reachable, or the panel is not blind."""
    monkeypatch.setattr(panel, "CORPUS", tmp_path)
    page = "pg"
    (tmp_path / "renders" / page).mkdir(parents=True, exist_ok=True)
    for name in ("old_typeset.png", "new_typeset.png"):
        Image.fromarray(np.full((40, 40), 200, np.uint8)).save(tmp_path / "renders" / page / name)
    rng = random.Random(seed)
    orders = {panel.strip({**_rec(page, i), "pair": i + 1}, "old", "new", tmp_path, rng)
              .split("  ")[-2] for i in range(12)}
    assert orders == {"A=old B=new", "A=new B=old"}
