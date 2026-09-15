"""The fixed eyeball set: 24 page pairs a human looks at every cycle.

Some of what makes the official release look right is hand-lettering judgement that no
metric in ``demo/corpus_score.py`` captures - where the line breaks fall relative to
sense units, emphasis, tail-aware placement, SFX treatment.  Those need a person, so a
small fixed set of pages is reviewed by eye alongside the numbers.

Two rules make the set honest:

* It is **chosen once and never re-chosen**, and the choice is committed in
  ``demo/corpus_eyeball.json``.  If the set could be resampled, the score could be
  improved by resampling it instead of by improving the typesetter.
* Eyeball scores are **never folded into R**.  They are a separate column.  Merging a
  24-page human judgement into a 5,000-page automatic number would destroy both.

The committed JSON holds page identifiers, volumes and categories only - never pixels, and
deliberately not the archive entry names, which carry the raw-dump site and the scanlation
release-group tags of an unlicensed archive.  A page id resolves back to its entries
through the gitignored ``pairs.json``.
Review sheets are written under ``demo/output/corpus/eyeball/`` (gitignored) by
``typeset_reference.write_review_sheets``, which already bounds them below 1000 px.

Usage::

    .venv/Scripts/python.exe demo/corpus_eyeball.py --choose      # once, then commit
    .venv/Scripts/python.exe demo/corpus_eyeball.py --sheets --tag base
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import corpus_index as CI  # noqa: E402
import corpus_pair as CP  # noqa: E402

EYEBALL_PATH = DEMO_DIR / "corpus_eyeball.json"
SHEETS_ROOT = CI.OUT_ROOT / "eyeball"
SET_VERSION = 1

# 12 bubble-heavy, 6 free-text/SFX, 3 spreads, 3 colour or chapter-title pages.
QUOTA: Dict[str, int] = {"bubble": 12, "free_text": 6, "spread": 3, "colour": 3}


def categorise(pair: Dict[str, Any], ref_by_entry: Dict[str, CI.PageRef]) -> str:
    """Bucket a pair from index metadata alone - no render and no ground truth needed."""
    en = ref_by_entry.get(str(pair["en_entry"]))
    if en is not None and (en.is_colour or (en.page is not None and en.page <= 1)):
        return "colour"
    if pair.get("en_half") is not None or pair.get("ja_half") is not None:
        return "spread"
    # Chapter openers carry large free text and SFX far more often than mid-chapter
    # pages; without a render that is the best signal available at selection time.
    if en is not None and en.page is not None and en.page % 20 in (2, 3):
        return "free_text"
    return "bubble"


def choose_set(pairs: Sequence[Dict[str, Any]], ref_by_entry: Dict[str, CI.PageRef],
               seed: int = 20260915) -> List[Dict[str, Any]]:
    """Pick the 24 pages, spread across volumes, deterministically for a given seed."""
    buckets: Dict[str, List[Dict[str, Any]]] = {k: [] for k in QUOTA}
    for pair in sorted(pairs, key=lambda p: str(p["page_id"])):
        buckets[categorise(pair, ref_by_entry)].append(pair)

    rng = random.Random(seed)
    chosen: List[Dict[str, Any]] = []
    for category, want in QUOTA.items():
        pool = buckets[category]
        # One page per volume first, so the set is not dominated by a single volume.
        by_volume: Dict[Any, List[Dict[str, Any]]] = {}
        for pair in pool:
            by_volume.setdefault(pair.get("volume"), []).append(pair)
        spread_first: List[Dict[str, Any]] = []
        for volume in sorted(by_volume, key=lambda v: (v is None, v)):
            spread_first.append(rng.choice(by_volume[volume]))
        rng.shuffle(spread_first)
        picked = spread_first[:want]
        if len(picked) < want:
            chosen_ids = {p["page_id"] for p in picked}
            rest = [p for p in pool if p["page_id"] not in chosen_ids]
            rng.shuffle(rest)
            picked += rest[: want - len(picked)]
        for pair in picked:
            # page_id ONLY.  The archive entry names carry the raw-dump site and the
            # scanlation release-group tags of an unlicensed archive, and this file is
            # committed; page_id is the key `corpus_eval --page` takes, and the entries
            # resolve from the gitignored pairs.json whenever they are actually needed.
            chosen.append({
                "page_id": pair["page_id"],
                "volume": pair.get("volume"),
                "category": category,
            })
    return sorted(chosen, key=lambda p: (str(p["category"]), str(p["page_id"])))


def save_set(chosen: Sequence[Dict[str, Any]], seed: int,
             path: Path = EYEBALL_PATH) -> Path:
    path.write_text(json.dumps(
        {"version": SET_VERSION, "chosen_with_seed": seed,
         "note": "Chosen once. Never re-choose: resampling this set would let the "
                 "eyeball review improve without the typesetter improving.",
         "pairs": list(chosen)}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_set(path: Path = EYEBALL_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["pairs"]


def write_sheets(tag: str, out_root: Path = SHEETS_ROOT) -> List[Path]:
    """Review sheets for the eyeball set, from the ground truth already derived."""
    import typeset_metrics as TM
    import typeset_reference as TR

    import corpus_eval as CE

    written: List[Path] = []
    out_root.mkdir(parents=True, exist_ok=True)
    for entry in load_set():
        page_id = str(entry["page_id"])
        truth = TM.load_ground_truth(page_id, CE.REFERENCE_DIR)
        if truth is None:
            print(f"{page_id}: no ground truth yet (run demo/corpus_eval.py first)",
                  file=sys.stderr)
            continue
        out_dir = out_root / tag / page_id
        out_dir.mkdir(parents=True, exist_ok=True)
        written += TR.write_review_sheets(page_id, truth["eng_gray"], truth["records"],
                                          out_dir)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--choose", action="store_true",
                        help="pick the set once and write demo/corpus_eyeball.json")
    parser.add_argument("--force", action="store_true",
                        help="allow --choose to overwrite an existing set (do not)")
    parser.add_argument("--sheets", action="store_true", help="write the review sheets")
    parser.add_argument("--tag", default="base")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--corpus", type=Path, default=CI.DEFAULT_CORPUS)
    args = parser.parse_args(argv)

    if args.choose:
        if EYEBALL_PATH.exists() and not args.force:
            print(f"{EYEBALL_PATH.name} already exists - the set is chosen once and "
                  "never re-chosen. Pass --force only if you mean to break that.",
                  file=sys.stderr)
            return 2
        pairs = CP.verified_pairs(CP.load_pairs())
        if not pairs:
            print("no verified pairs; run demo/corpus_pair.py first", file=sys.stderr)
            return 2
        corpus = Path(args.corpus)
        ref_by_entry: Dict[str, CI.PageRef] = {}
        for cbz in sorted((corpus / CI.EN_DIR_NAME).glob("*.cbz")):
            archive = CI.open_archive(cbz)
            try:
                for ref in CI.enumerate_archive(archive, "en", decode=True, workers=8):
                    ref_by_entry[ref.entry] = ref
            finally:
                archive.close()
        chosen = choose_set(pairs, ref_by_entry, args.seed)
        print(f"wrote {save_set(chosen, args.seed)} ({len(chosen)} pages)")
        for entry in chosen:
            volume = entry["volume"]
            label = f"v{volume:02d}" if isinstance(volume, int) else "v--"
            print(f"  {entry['category']:10s} {label} {entry['page_id']}")
        return 0

    if args.sheets:
        sheets = write_sheets(args.tag)
        print(f"{len(sheets)} sheet(s) under {SHEETS_ROOT / args.tag}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
