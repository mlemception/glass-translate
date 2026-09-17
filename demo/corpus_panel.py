"""Build a blind A/B panel for one change.

Per chosen block this emits one greyscale strip holding three crops of the SAME
rectangle: the aligned reference, and the two renders **in a randomised order**,
reseeded per strip.  The key naming which side is which goes to a separate file
that must not be read until the verdict is recorded.

The protocol it enforces rather than assumes:

* 6-8 affected blocks across **at least 4 pages**, plus **2 controls** - blocks
  the change was not expected to touch, which catch a judge that confabulates a
  preference between two identical images;
* every strip greyscale, at most :data:`MAX_SIDE` px on the long side and
  **under 100 KB**;
* the key read only afterwards, and the bar (wins exceed losses by 2) fixed
  before it is read.

Three things it refuses to do, each because it once did them:

* **write into the repository.** The corpus is licensed third-party material
  and no crop of it may enter the tree; :func:`check_out_dir` rejects any
  destination inside the repo.
* **emit a strip over 100 KB.** An earlier version resized *after* its final
  save and gave up after a fixed number of attempts, so it left 117 KB and
  106 KB strips on disk while appearing to enforce the limit.  The loop now
  asserts.
* **build a panel that decides nothing.** It once produced 4 strips across 2
  pages without comment, which is below the protocol floor and therefore
  decides nothing; :func:`main` now exits non-zero and says so.

It also drops pages whose reference does not register with our page - see
:func:`registers`, and the cycle-9 section of
``docs/perf/2026-09-15-typeset-corpus.md``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "demo" / "output" / "corpus"

MAX_SIDE = 1000  # px on the long side of a strip
MAX_BYTES = 100_000  # hard ceiling on a strip, asserted rather than assumed
GAP = 8  # px of white between the three panels
DIFF_THR = 8  # grey levels: below this a pixel counts as unchanged

# The protocol floor.  A panel that misses it decides nothing.
MIN_BLOCKS = 6
MIN_PAGES = 4
CONTROLS = 2

# A page whose kept-art mask is this far from our own ink is not comparable.
FLOOR = 0.10
FLOOR_MIN_KEPT = 2000


def blocks(page: str, tag: str) -> dict:
    return json.loads((CORPUS / "renders" / page / f"{tag}_blocks.json").read_text(encoding="utf-8"))


def render(page: str, tag: str) -> Image.Image:
    return Image.open(CORPUS / "renders" / page / f"{tag}_typeset.png").convert("L")


def registers(page: str) -> bool:
    """Whether this page's reference actually corresponds to our page.

    Each strip carries ``eng_aligned.png`` beside it as the ground truth.  On a
    page whose reference does not register, the reviewer compares our render
    against *different content* and reports artifacts with confidence.  It did
    exactly that on a 1200x764 half-page whose reference is a 2250x1500 page
    scaled 0.534 onto it: **96.3 %** of that page's kept-art is not ink in our
    own original.

    The stored alignment score does **not** catch this - that page scores 0.894,
    better than pages that register fine - so the floor is measured directly,
    measured directly.
    """
    try:
        kept = np.asarray(Image.open(CORPUS / "reference" / page / "kept_art.png").convert("L")) > 127
        orig = np.asarray(Image.open(CORPUS / "pages" / f"{page}.png").convert("L")) < 128
    except OSError:
        return True  # nothing to test the floor with; leave the page in
    n = int(kept.sum())
    if kept.shape != orig.shape or n < FLOOR_MIN_KEPT:
        return True  # too little kept art for the floor to mean anything
    return float((kept & ~orig).sum()) / n < FLOOR


def pages(tag: str) -> List[str]:
    """Every page rendered under ``tag`` whose reference registers."""
    found = sorted(p.parent.name for p in (CORPUS / "renders").glob(f"*/{tag}_blocks.json"))
    return [p for p in found if registers(p)]


def changed(old_tag: str, new_tag: str, pad: int) -> Tuple[List[dict], List[dict]]:
    """Per-block change magnitude, split into ``(affected, untouched)``."""
    hot: List[dict] = []
    cold: List[dict] = []
    for page in sorted(set(pages(old_tag)) & set(pages(new_tag))):
        try:
            a, b = render(page, old_tag), render(page, new_tag)
            ba, bb = blocks(page, old_tag), blocks(page, new_tag)
        except (OSError, ValueError):
            continue
        if a.size != b.size:
            continue
        na, nb = np.asarray(a, dtype=np.int16), np.asarray(b, dtype=np.int16)
        for i, blk in enumerate(bb["blocks"]):
            box = blk.get("source_bbox")
            if not box or i >= len(ba["blocks"]):
                continue
            x, y, w, h = (int(v) for v in box)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(na.shape[1], x + w + pad), min(na.shape[0], y + h + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            d = np.abs(na[y0:y1, x0:x1] - nb[y0:y1, x0:x1])
            rec = {
                "page": page, "index": i, "rect": (x0, y0, x1, y1),
                "diff_px": int((d > DIFF_THR).sum()),
                "from_reference": bool(blk.get("from_reference")),
                "old_from_reference": bool(ba["blocks"][i].get("from_reference")),
                "kind": blk.get("kind"),
            }
            (hot if rec["diff_px"] > 0 else cold).append(rec)
    hot.sort(key=lambda r: -r["diff_px"])
    return hot, cold


def select(hot: Sequence[dict], cold: Sequence[dict], rng: random.Random) -> Tuple[List[dict], set]:
    """Choose the affected blocks and their controls, spreading over pages.

    Only blocks whose reference lookup was stable on **both** sides are
    eligible: block grouping wobbles between runs (roughly 2 blocks in 316), and
    a text change from that wobble is not the slice under test.
    """
    stable = [r for r in hot if r["from_reference"] == r["old_from_reference"]]
    chosen: List[dict] = []
    seen: set = set()
    for r in stable:
        if r["page"] in seen and len(seen) < MIN_PAGES:
            continue  # spread over pages first, then take seconds
        chosen.append({**r, "role": "affected"})
        seen.add(r["page"])
        if len(chosen) == MIN_BLOCKS:
            break
    controls = [c for c in cold if c["page"] in seen] or list(cold)
    for c in rng.sample(controls, min(CONTROLS, len(controls))):
        chosen.append({**c, "role": "control"})
    return chosen, seen


def strip(rec: dict, old_tag: str, new_tag: str, out: Path, rng: random.Random) -> str:
    """Write one strip and return its line for the key."""
    x0, y0, x1, y1 = rec["rect"]
    ref_path = CORPUS / "reference" / rec["page"] / "eng_aligned.png"
    ref = Image.open(ref_path).convert("L").crop((x0, y0, x1, y1)) if ref_path.is_file() else None
    a = render(rec["page"], old_tag).crop((x0, y0, x1, y1))
    b = render(rec["page"], new_tag).crop((x0, y0, x1, y1))

    first_is_old = rng.random() < 0.5  # reseeded per strip, never once per panel
    left, right = (a, b) if first_is_old else (b, a)
    panels = [p for p in (ref, left, right) if p is not None]

    width = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    sheet = Image.new("L", (width, max(p.height for p in panels)), 255)
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + GAP
    if max(sheet.size) > MAX_SIDE:
        s = MAX_SIDE / max(sheet.size)
        sheet = sheet.resize((max(1, int(sheet.width * s)), max(1, int(sheet.height * s))),
                             Image.Resampling.LANCZOS)

    path = out / f"pair{rec['pair']:02d}.png"
    sheet.save(path, optimize=True)
    while path.stat().st_size >= MAX_BYTES and sheet.width > 200:
        sheet = sheet.resize((int(sheet.width * 0.85), int(sheet.height * 0.85)),
                             Image.Resampling.LANCZOS)
        sheet.save(path, optimize=True)
    assert path.stat().st_size < MAX_BYTES, f"{path.name} is {path.stat().st_size} bytes"

    label = "A=%s B=%s" % ((old_tag, new_tag) if first_is_old else (new_tag, old_tag))
    return "%s  %s b%d  rect=%s  role=%s  %s  diff_px=%d" % (
        path.name, rec["page"], rec["index"], (x0, y0, x1, y1), rec["role"], label, rec["diff_px"])


def check_out_dir(out: Path) -> Optional[str]:
    """Reason the destination is unusable, or None.

    The corpus is licensed third-party material: no page, crop or derived image
    may enter the repository.  A scratchpad is the right destination.
    """
    try:
        out.resolve().relative_to(ROOT)
    except ValueError:
        return None
    return (f"{out} is inside the repository ({ROOT}). Corpus crops must never be written "
            "into the tree - use a scratchpad directory.")


def judgeable(chosen: Sequence[dict], seen_pages: set) -> Optional[str]:
    """Reason this panel cannot decide anything, or None if it can."""
    affected = [c for c in chosen if c["role"] == "affected"]
    controls = [c for c in chosen if c["role"] == "control"]
    if len(affected) < MIN_BLOCKS:
        return f"only {len(affected)} affected block(s); the protocol needs {MIN_BLOCKS}"
    if len(seen_pages) < MIN_PAGES:
        return (f"only {len(seen_pages)} page(s) affected; the protocol needs {MIN_PAGES}. "
                "A change this narrow cannot be validated - record it as unjudgeable "
                "rather than judging it anyway.")
    if len(controls) < CONTROLS:
        return f"only {len(controls)} control(s); the protocol needs {CONTROLS}"
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old_tag", help="the baseline render tag")
    ap.add_argument("new_tag", help="the tag under test")
    ap.add_argument("out", type=Path, help="destination directory, OUTSIDE the repo")
    ap.add_argument("--seed", type=int, default=1, help="randomisation seed (default 1)")
    ap.add_argument("--pad", type=int, default=24,
                    help="px of context around each block; raise it when the change is in the "
                         "artwork AROUND the text rather than the lettering (default 24)")
    ap.add_argument("--force", action="store_true",
                    help="write the strips even when the panel misses the protocol floor. The "
                         "panel still decides nothing; this is for inspection only.")
    args = ap.parse_args(argv)

    bad = check_out_dir(args.out)
    if bad:
        print(bad, file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    hot, cold = changed(args.old_tag, args.new_tag, args.pad)
    chosen, seen = select(hot, cold, rng)
    print("affected blocks: %d  untouched: %d" % (len(hot), len(cold)))
    print("chosen: %d affected across %d page(s), %d control(s)"
          % (sum(c["role"] == "affected" for c in chosen), len(seen),
             sum(c["role"] == "control" for c in chosen)))

    why = judgeable(chosen, seen)
    if why:
        print("\nNOT JUDGEABLE: " + why, file=sys.stderr)
        if not args.force:
            print("no strips written; pass --force to build them anyway for inspection",
                  file=sys.stderr)
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    rng.shuffle(chosen)
    lines = []
    for n, rec in enumerate(chosen, 1):
        rec["pair"] = n
        lines.append(strip(rec, args.old_tag, args.new_tag, args.out, rng))
    (args.out / "KEY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote %d strips to %s" % (len(chosen), args.out))
    print("KEY.txt written - DO NOT READ until the verdict is recorded")
    return 1 if why else 0


if __name__ == "__main__":
    raise SystemExit(main())
