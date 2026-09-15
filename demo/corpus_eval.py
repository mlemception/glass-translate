"""Evaluate the typesetter against the official English release, over a page sample.

One command answers three questions: how close the render is to the release, which
pages fail and why, and whether a change made things better or worse.

    .venv/Scripts/python.exe demo/corpus_eval.py --sample 50 --seed 1 --tag base
    .venv/Scripts/python.exe demo/corpus_eval.py --sample 50 --seed 1 --tag try1 --baseline base

Everything written lands under ``demo/output/corpus/`` (gitignored).  The corpus itself
is licensed third-party material: it is opened read-only, no page is ever copied into the
repository, and the committed report carries numbers, page identifiers and paths only.

The pipeline per page is: materialise the JA page and its verified EN counterpart ->
``typeset_reference.derive`` for the ground truth (align, warp, OCR the EN lettering) ->
``typeset_dev.run_page`` for our render -> ``typeset_metrics.score_page`` for the
per-block scores -> ``corpus_metrics`` for the four new ones -> ``corpus_score`` for the
headline.  Nothing here reimplements a metric that already exists.

Blocks are lettered with ``--ref-text``: the official English that ``derive`` read off the
reference page, so both sides carry the same words and the text metrics compare
TYPESETTING rather than translation length.  Blocks whose English the OCR could not read
fall back to the machine translation, and the page records how many were uncertain.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import corpus_index as CI  # noqa: E402
import corpus_metrics as CM  # noqa: E402
import corpus_pair as CP  # noqa: E402
import corpus_score as CS  # noqa: E402

OUT_ROOT = CI.OUT_ROOT
PAGES_DIR = OUT_ROOT / "pages"
REFERENCE_DIR = OUT_ROOT / "reference"
RENDER_DIR = OUT_ROOT / "renders"
METRICS_DIR = OUT_ROOT / "metrics"
CROPS_DIR = OUT_ROOT / "crops"

REFTEXT_DIR = OUT_ROOT / "reftext"

CROP_MAX_SIDE = 1000  # never write anything larger; full pages are never viewed
WORST_N = 20

# Gate 5, the art-content gate: the share of this page's Japanese ink that also exists in
# the aligned English page.  A page whose text sits on no shared artwork - author notes,
# bonus text, ads - is the same page of the book but is pure noise to score.
#
# This is measured over the WHOLE PAGE, not over the text regions.  The obvious version,
# kept_art / gt_region_px, does not work: a balloon interior is blank paper in BOTH
# releases, so kept_art is legitimately ~0 exactly there, and that test rejected ordinary
# bubble pages (measured 0.036 and 0.000 on two normal v08 pages).  Whole-page shared ink
# on those same two pages is 0.956 and 0.915, so the threshold has a wide margin.
SHARED_INK_MIN = 0.55
INK_TOLERANCE_PX = 5  # dilation before intersecting, for scan/registration slack

# How many baseline pages may vanish from a gated run before it fails outright.
MISSING_PAGE_FLOOR = 2
MISSING_PAGE_RATIO = 0.10


# --------------------------------------------------------------------- sampling
def sample_pairs(pairs: Sequence[Dict[str, Any]], count: int, seed: int,
                 ) -> List[Dict[str, Any]]:
    """A reproducible random sample.  The same seed always gives the same pages.

    The candidate list is sorted by ``page_id`` first, so the sample does not depend on
    the order ``pairs.json`` happened to be written in.
    """
    ordered = sorted(pairs, key=lambda p: str(p["page_id"]))
    if count >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, count),
                  key=lambda p: str(p["page_id"]))


def select_pairs(pairs: Sequence[Dict[str, Any]], *, volumes: Sequence[int] = (),
                 chapters: Sequence[str] = (), pages: Sequence[str] = (),
                 pages_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Narrow the candidate pool so one failing page can be rerun on its own."""
    wanted_ids = {p.strip() for p in pages if p.strip()}
    if pages_file:
        wanted_ids |= {line.strip() for line in
                       Path(pages_file).read_text(encoding="utf-8").splitlines()
                       if line.strip() and not line.startswith("#")}
    out = list(pairs)
    if volumes:
        out = [p for p in out if p.get("volume") in set(volumes)]
    if chapters:
        keep = {str(c) for c in chapters}
        out = [p for p in out if chapter_of(p) in keep]
    if wanted_ids:
        out = [p for p in out if str(p["page_id"]) in wanted_ids]
    return out


def chapter_of(pair: Dict[str, Any]) -> str:
    meta = CI.parse_en_name(str(pair["en_entry"]))
    return str(meta.get("chapter") or "")


# --------------------------------------------------------------------- materialise
def slot_gray_from(archive: CI.Archive, entry: str, half: Optional[int]
                   ) -> Optional[np.ndarray]:
    gray = CI.decode_gray(archive.read(entry))
    if gray is None or half is None:
        return gray
    right, left = CI.split_spread(gray)
    return right if half == 0 else left


def materialise_pair(pair: Dict[str, Any], ja_archive: CI.Archive,
                     en_archive: CI.Archive, work_dir: Path = PAGES_DIR,
                     ) -> Optional[Tuple[Path, Path]]:
    """Write the two sides of one pair as greyscale PNGs under ``demo/output/``.

    The stem is the pair's ASCII-safe ``page_id``: ``cv2.imread`` is used downstream by
    ``derive`` and ``score_page`` and it mangles non-ASCII paths, while JA entry names
    are Japanese.
    """
    page_id = str(pair["page_id"])
    ja = slot_gray_from(ja_archive, str(pair["ja_entry"]), pair.get("ja_half"))
    en = slot_gray_from(en_archive, str(pair["en_entry"]), pair.get("en_half"))
    if ja is None or en is None:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    ja_path = work_dir / f"{page_id}.png"
    en_path = work_dir / f"{page_id}_eng.png"
    for path, arr in ((ja_path, ja), (en_path, en)):
        if not path.exists():
            ok, buf = cv2.imencode(".png", arr)
            if not ok:
                return None
            path.write_bytes(buf.tobytes())
    return ja_path, en_path


def to_components(raw: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Map ``corpus_metrics``' raw measurements onto the score's ``c_*`` components.

    The two modules deliberately speak different languages: ``corpus_metrics`` reports
    what was measured (``containment_mean``, ``erase_iou``, ...) and ``corpus_score``
    owns every curve and weight.  This is the only place the two meet, so the weights
    still live in exactly one file.
    """
    return {
        "c_contain": CS.c_contain(raw.get("containment_mean")),
        "c_erase": CS.c_erase(raw.get("erase_iou"), raw.get("art_kept")),
        "c_art": CS.c_art(raw.get("lpips_excess")),
        "c_centre": CS.c_centre(raw.get("centre_offset_em_mean")),
        "c_leftover": CS.c_leftover(raw.get("leftover_em2_mean")),
        "c_size": CS.c_size(raw.get("size_logratio_rms")),
        "c_group": CS.c_group(raw.get("group_f1")),
        "c_lines": CS.c_lines(raw.get("line_closeness")),
        "c_textiou": CS.c_textiou(raw.get("text_iou_mean")),
    }


def shared_ink_ratio(ja_gray: np.ndarray, eng_gray: np.ndarray) -> float:
    """Share of this page's Japanese ink that also exists in the aligned English page.

    High on any real page (the artwork is identical between releases); near zero on a
    page whose every mark was replaced, which is what gate 5 is looking for.
    """
    import typeset_reference as TR

    ja_ink = TR.ink_mask(ja_gray)
    total = int(np.count_nonzero(ja_ink))
    if total == 0:
        return 1.0
    eng_ink = TR.ink_mask(eng_gray).astype(np.uint8)
    grown = cv2.dilate(eng_ink, np.ones((INK_TOLERANCE_PX, INK_TOLERANCE_PX), np.uint8))
    return float(np.count_nonzero(ja_ink & grown.astype(bool))) / total


def render_args(tag: str, device: str, models: Path, quiet: bool = True) -> Namespace:
    """The ``typeset_dev.run_page`` namespace for a corpus page."""
    # ref_text=True letters our blocks with the official English that ``derive`` read off
    # the reference page, so both sides carry the SAME words.  Without it c_textiou and
    # c_lines punish a correct render whenever the machine translation differs in length
    # from VIZ's.  reference_text_root keeps those bootstrapped files out of the tracked
    # demo/ root.
    return Namespace(tag=tag, ocr=False, translate=False, ref_text=True,
                     reference_text_root=REFTEXT_DIR,
                     src="ja", tgt="en", device=device, models=models,
                     min_confidence=0.5, no_upper=False, quality=False,
                     quality_fake=False, quality_params="", new_erase=False,
                     new_place=False, regions="none", no_ref_image=True, quiet=quiet,
                     batch=False, out=RENDER_DIR)


# --------------------------------------------------------------------- one page
def evaluate_page(pair: Dict[str, Any], ja_archive: CI.Archive, en_archive: CI.Archive,
                  *, tag: str, device: str, models: Path,
                  refresh: bool = False, refresh_render: bool = False) -> Dict[str, Any]:
    """Derive the ground truth, render, score, and fold into the headline components."""
    import typeset_dev as TD
    import typeset_metrics as TM
    import typeset_reference as TR

    page_id = str(pair["page_id"])
    paths = materialise_pair(pair, ja_archive, en_archive)
    if paths is None:
        return {"page_id": page_id, "status": "skipped", "reason": "cannot decode page"}
    ja_path, en_path = paths

    ref_dir = REFERENCE_DIR / page_id
    if refresh or not (ref_dir / "blocks.json").exists():
        derived = TR.derive(ja_path, out_root=REFERENCE_DIR, device=device,
                            reference=en_path, reference_text_root=REFTEXT_DIR)
        if derived is None:
            return {"page_id": page_id, "status": "skipped", "reason": "derive failed"}

    truth = TM.load_ground_truth(page_id, REFERENCE_DIR)
    if truth is None:
        return {"page_id": page_id, "status": "skipped", "reason": "no ground truth"}
    records: List[Dict[str, Any]] = truth["records"]

    # Gate 5 of the pairing decision rule - see SHARED_INK_MIN for why it is measured over
    # the whole page rather than over the text regions.
    ja_gray = CI.imread_gray(ja_path)
    shared_ratio = shared_ink_ratio(ja_gray, truth["eng_gray"]) if ja_gray is not None else 1.0
    if shared_ratio < SHARED_INK_MIN:
        return {"page_id": page_id, "status": "unverifiable",
                "reason": f"no shared art (shared ink {shared_ratio:.3f} "
                          f"< {SHARED_INK_MIN})",
                "shared_ink": round(shared_ratio, 4)}
    if not records:
        return {"page_id": page_id, "status": "no_text", "reason": "no ground-truth block"}

    render_dir = RENDER_DIR / page_id
    if refresh or refresh_render or not (render_dir / f"{tag}_blocks.json").exists():
        code = TD.run_page(render_args(tag, device, models), ja_path, render_dir)
        if code != 0:
            return {"page_id": page_id, "status": "skipped",
                    "reason": f"render exit {code}"}

    scored = TM.score_page(ja_path, render_dir, tag, device=device,
                           reference_root=REFERENCE_DIR)
    if scored is None:
        return {"page_id": page_id, "status": "skipped",
                "reason": "score_page returned None"}

    extras = CM.page_extras(scored, truth, render_dir=render_dir, tag=tag,
                            ja_path=ja_path)
    if not extras.get("scored_blocks"):
        # Everything on this page was out of scope (stylised SFX, or untranslated logos).
        return {"page_id": page_id, "status": "no_text",
                "reason": "no in-scope block",
                "sfx_excluded": int(extras.get("sfx_excluded") or 0)}

    raw = dict(CM.page_components(scored, extras))
    answered = raw.pop("answered", 1.0)
    answered = 1.0 if answered is None else float(answered)
    result = CS.page_score(to_components(raw), answered)

    row = {
        "page_id": page_id,
        "status": "scored",
        "volume": pair.get("volume"),
        "pair_status": pair.get("status"),
        "verified_weak": pair.get("status") == CP.STATUS_WEAK,
        "reference_weak": bool(extras.get("reference_weak")),
        "bubble_heavy": bool(extras.get("bubble_heavy")),
        "free_text_only": bool(extras.get("free_text_only")),
        "R": result["R"],
        "answered": answered,
        "components": result["components"],
        "missing": result["missing"],
        "overflow_px": int(extras.get("overflow_px") or 0),
        "leftover_px": int(extras.get("leftover_px") or 0),
        # The gate's sharper invariants: blocks with any glyph outside the balloon
        # interior, and blocks whose ink crosses a panel border.  Both were measured to
        # move on a real defect that left overflow_px stuck on zero.
        "uncontained": int((scored.get("summary") or {}).get("uncontained") or 0),
        "collisions": int((scored.get("summary") or {}).get("collisions") or 0),
        # Reference-free self-checks (glasstranslate/render/selfcheck.py), read back off
        # the render's own block records.  Unlike everything else in this row they need no
        # English edition, so they are the only invariants here that also hold in real use.
        **_self_check_counts(render_dir, tag),
        "blocks": int(extras.get("scored_blocks") or 0),
        "sfx_excluded": int(extras.get("sfx_excluded") or 0),
        "ja_entry": pair.get("ja_entry"),
        "en_entry": pair.get("en_entry"),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{page_id}_{tag}.json").write_text(
        json.dumps({"row": row, "scored": jsonable(scored), "extras": jsonable(extras)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return row


def jsonable(obj: Any) -> Any:
    """Drop numpy arrays and convert numpy scalars, so a metrics dict can be written."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj if not isinstance(v, np.ndarray)]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# --------------------------------------------------------------------- crops
def write_crops(rows: Sequence[Dict[str, Any]], tag: str, limit: int = WORST_N
                ) -> List[Path]:
    """Small greyscale crops of the worst pages, for a human to look at.

    Never a full page at full size and never larger than ``CROP_MAX_SIDE`` on the long
    side.  These stay under ``demo/output/`` and are never committed.
    """
    out_dir = CROPS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    worst = sorted((r for r in rows if r.get("R") is not None),
                   key=lambda r: r["R"])[:limit]
    for row in worst:
        src = RENDER_DIR / str(row["page_id"]) / f"{tag}_typeset.png"
        if not src.exists():
            continue
        img = CI.imread_gray(src)
        if img is None:
            continue
        scale = min(1.0, CROP_MAX_SIDE / max(img.shape[:2]))
        if scale < 1.0:
            img = cv2.resize(img, (max(1, round(img.shape[1] * scale)),
                                   max(1, round(img.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            continue
        path = out_dir / f"{row['page_id']}_R{row['R']:05.1f}.png"
        path.write_bytes(buf.tobytes())
        written.append(path)
    return written


# --------------------------------------------------------------------- report
_FAILING = ("c_contain", "c_erase", "c_art", "c_centre", "c_leftover",
            "c_size", "c_group", "c_lines", "c_textiou")

_SYMPTOM = {
    "c_contain": "glyph pixels outside the balloon interior",
    "c_erase": "erase missed source ink or ate the artwork",
    "c_art": "artwork damaged outside the text (LPIPS above the page floor)",
    "c_centre": "block not optically centred in the balloon",
    "c_leftover": "Japanese ink surviving inside the balloon",
    "c_size": "type size disagrees with the official lettering",
    "c_group": "block grouping disagrees with the release's text areas",
    "c_lines": "line count differs from the official lettering",
    "c_textiou": "text area does not overlap the official lettering",
}


def write_report(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any], tag: str,
                 *, seed: Optional[int], sample: Optional[int],
                 gate: Optional[Dict[str, Any]] = None,
                 crops: Sequence[Path] = (), notes: Sequence[str] = ()) -> Path:
    """The triage report.  Numbers, page identifiers and paths only - never pixels."""
    scored = [r for r in rows if r.get("R") is not None]
    lines: List[str] = []
    lines.append(f"# Corpus typesetting report - `{tag}`\n")
    lines.append(f"- sample: {sample if sample is not None else len(rows)} page(s), "
                 f"seed {seed}")
    lines.append(f"- scored: {len(scored)}  |  skipped: "
                 f"{sum(1 for r in rows if r.get('status') == 'skipped')}"
                 f"  |  unverifiable: "
                 f"{sum(1 for r in rows if r.get('status') == 'unverifiable')}"
                 f"  |  no in-scope text: "
                 f"{sum(1 for r in rows if r.get('status') == 'no_text')}")
    lines.append(f"- stylised onomatopoeia excluded (out of scope): "
                 f"{sum(int(r.get('sfx_excluded') or 0) for r in rows)} block(s) across "
                 f"{sum(1 for r in rows if r.get('sfx_excluded'))} page(s)")

    headline = summary.get("mean_R")
    lines.append("\n## Headline\n")
    lines.append(f"**release-likeness R = {headline:.2f}**" if headline is not None
                 else "**release-likeness R = n/a**")
    lines.append("")
    lines.append("| statistic | all | bubble-heavy | free-text-only |")
    lines.append("|---|---|---|---|")
    for key, label in (("mean_R", "mean R"), ("p05_R", "5th percentile R"),
                       ("zero_count", "pages at R = 0")):
        cells = [summary.get(key),
                 (summary.get("bubble_heavy") or {}).get(key),
                 (summary.get("free_text_only") or {}).get(key)]
        lines.append(f"| {label} | " + " | ".join(_fmt(v) for v in cells) + " |")

    lines.append("\n## Components\n")
    lines.append(CS.component_table_md(summary))
    lines.append("\n## Distribution of per-page R\n")
    lines.append(CS.distribution_md(scored))

    lines.append("\n## Worst pages by failing component\n")
    for key in _FAILING:
        ranked = [r for r in scored if (r.get("components") or {}).get(key) is not None]
        ranked.sort(key=lambda r: r["components"][key])
        ranked = [r for r in ranked if r["components"][key] < 0.9][:WORST_N]
        if not ranked:
            continue
        lines.append(f"\n### `{key}` - {_SYMPTOM[key]}\n")
        lines.append("| page id | component | R | volume |")
        lines.append("|---|---|---|---|")
        for r in ranked:
            vol = r.get("volume")
            vol_text = f"v{vol:02d}" if isinstance(vol, int) else "-"
            lines.append(f"| `{r['page_id']}` | {r['components'][key]:.3f} | "
                         f"{r['R']:.1f} | {vol_text} |")

    if gate is not None:
        lines.append("\n## Regression gate\n")
        lines.append(f"- result: **{'PASS' if gate.get('passed') else 'FAIL'}**")
        lines.append(f"- {gate.get('message', '')}")
        for reg in gate.get("invariant_regressions") or []:
            lines.append(f"  - `{reg['page_id']}` {reg['invariant']}: "
                         f"{reg['was']} -> {reg['now']}")

    if crops:
        rel = CROPS_DIR.relative_to(PROJECT_ROOT).as_posix()
        lines.append("\n## Crops for human review\n")
        lines.append(f"{len(crops)} greyscale crop(s) under `{rel}/{tag}/` "
                     "(uncommitted; no corpus imagery enters the repository).")

    if notes:
        lines.append("\n## Notes\n")
        lines.extend(f"- {n}" for n in notes)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"report-{tag}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


# --------------------------------------------------------------------- baseline
def _self_check_counts(render_dir: Path, tag: str) -> Dict[str, int]:
    """Roll the render's own per-block self-checks up to the page.

    ``fragments`` - blocks that did not draw every character they were handed;
    ``bad_breaks`` - blocks whose lines do not read back as the text given, i.e. a
    break ate its space; ``bubble_ink_px`` - source ink still readable inside the
    balloons.  All three come from ``glasstranslate/render/selfcheck.py`` by way of
    ``typeset_dev.block_record``, so they need no reference page.  A render written
    before those keys existed reports 0, which the gate reads as "held".
    """
    path = Path(render_dir) / f"{tag}_blocks.json"
    if not path.exists():
        return {"fragments": 0, "bad_breaks": 0, "bubble_ink_px": 0}
    try:
        blocks = json.loads(path.read_text(encoding="utf-8")).get("blocks") or []
    except (OSError, ValueError):
        return {"fragments": 0, "bad_breaks": 0, "bubble_ink_px": 0}
    return {
        "fragments": sum(1 for b in blocks if b.get("text_complete") is False),
        "bad_breaks": sum(1 for b in blocks if b.get("breaks_clean") is False),
        "bubble_ink_px": sum(int(b.get("bubble_ink_px") or 0) for b in blocks),
    }


def _gate_payload(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any],
                  tag: str) -> Dict[str, Any]:
    return {
        "tag": tag,
        "mean_R": summary.get("mean_R"),
        "pages": {str(r["page_id"]): {"overflow_px": int(r.get("overflow_px") or 0),
                                      "leftover_px": int(r.get("leftover_px") or 0),
                                      "uncontained": int(r.get("uncontained") or 0),
                                      "collisions": int(r.get("collisions") or 0),
                                      "fragments": int(r.get("fragments") or 0),
                                      "bad_breaks": int(r.get("bad_breaks") or 0),
                                      "bubble_ink_px": int(r.get("bubble_ink_px") or 0),
                                      "R": r.get("R")}
                  for r in rows if r.get("status") == "scored"},
    }


def save_baseline(rows: Sequence[Dict[str, Any]], summary: Dict[str, Any],
                  tag: str) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"baseline-{tag}.json"
    path.write_text(json.dumps(_gate_payload(rows, summary, tag), ensure_ascii=False,
                               indent=1), encoding="utf-8")
    return path


def load_baseline(tag: str) -> Optional[Dict[str, Any]]:
    path = OUT_ROOT / f"baseline-{tag}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------- driver
def run_corpus(selected: Sequence[Dict[str, Any]], *, tag: str, corpus: Path,
               device: str, models: Path, refresh: bool = False,
               refresh_render: bool = False,
               progress: bool = True) -> List[Dict[str, Any]]:
    ja_archive = CI.open_archive(corpus / CI.JA_ARCHIVE_NAME)
    en_cache: Dict[str, CI.Archive] = {}
    rows: List[Dict[str, Any]] = []
    try:
        for i, pair in enumerate(selected, 1):
            volume = pair.get("volume")
            matches = (sorted((corpus / CI.EN_DIR_NAME).glob(f"*v{volume:02d} *.cbz"))
                       if isinstance(volume, int) else [])
            if not matches:
                rows.append({"page_id": pair["page_id"], "status": "skipped",
                             "reason": f"no EN archive for volume {volume}"})
                continue
            key = matches[0].name
            if key not in en_cache:
                en_cache[key] = CI.open_archive(matches[0])
            t0 = time.time()
            try:
                row = evaluate_page(pair, ja_archive, en_cache[key], tag=tag,
                                    device=device, models=models, refresh=refresh,
                                    refresh_render=refresh_render)
            except Exception as exc:  # report the page, never hide it
                row = {"page_id": pair["page_id"], "status": "skipped",
                       "reason": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            if progress:
                score = row.get("R")
                shown = "-" if score is None else f"{score:.1f}"
                print(f"[{i}/{len(selected)}] {pair['page_id']} "
                      f"{row.get('status', '?')} R={shown} ({time.time() - t0:.1f}s)",
                      flush=True)
    finally:
        ja_archive.close()
        for archive in en_cache.values():
            archive.close()
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CI.DEFAULT_CORPUS)
    parser.add_argument("--pairs", type=Path, default=None,
                        help="pairs.json (default: the cached one)")
    parser.add_argument("--tag", default="base")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--volume", type=int, action="append", default=[])
    parser.add_argument("--chapter", action="append", default=[])
    parser.add_argument("--page", action="append", default=[], help="page id (repeatable)")
    parser.add_argument("--pages-file", type=Path, default=None)
    parser.add_argument("--baseline", default=None,
                        help="tag to compare against; non-zero exit on regression")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--models", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--include-weak", action="store_true",
                        help="also score VERIFIED_WEAK pairs (never in the p5 tail)")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore cached ground truth and renders")
    parser.add_argument("--reuse-renders", action="store_true",
                        help="with --baseline, score the cached renders instead of "
                             "re-rendering (fast, but the gate then judges code that may "
                             "not have produced them)")
    parser.add_argument("--no-crops", action="store_true")
    args = parser.parse_args(argv)

    payload = CP.load_pairs(args.pairs)
    pool = CP.verified_pairs(payload, include_weak=args.include_weak)
    if not pool:
        print("no verified pairs; run demo/corpus_pair.py first", file=sys.stderr)
        return 2
    pool = select_pairs(pool, volumes=args.volume, chapters=args.chapter,
                        pages=args.page, pages_file=args.pages_file)
    if not pool:
        print("no pages matched the selectors", file=sys.stderr)
        return 2
    selected = sample_pairs(pool, args.sample, args.seed) if args.sample else pool

    # A gated run re-renders by default: renders are cached per (page, tag) and carry no
    # record of the code that produced them, so reusing them would let the gate pass on
    # changes it never exercised.  The ground truth is NOT re-derived - it depends on the
    # corpus, not on our code, and re-deriving it would re-run OCR on every page.
    refresh_render = bool(args.baseline) and not args.reuse_renders
    rows = run_corpus(selected, tag=args.tag, corpus=Path(args.corpus),
                      device=args.device, models=args.models, refresh=args.refresh,
                      refresh_render=refresh_render)
    scored = [r for r in rows if r.get("status") == "scored"]
    summary = CS.corpus_summary(scored)

    gate = None
    exit_code = 0
    if args.baseline:
        base = load_baseline(args.baseline)
        if base is None:
            print(f"no baseline '{args.baseline}' under {OUT_ROOT}", file=sys.stderr)
            return 2
        gate = CS.compare_baseline(_gate_payload(rows, summary, args.tag), base)
        # A page that stops scoring must not be a free pass.  If a change makes pages
        # crash in render they are simply absent from this run, the mean rises over the
        # surviving easy pages, and the gate would say PASS - the same degenerate
        # strategy `answered` closes within a page, left open across pages.
        vanished = list(gate.get("only_baseline") or [])
        allowed = max(MISSING_PAGE_FLOOR, int(MISSING_PAGE_RATIO * len(base.get("pages") or {})))
        if len(vanished) > allowed:
            gate = dict(gate)
            gate["passed"] = False
            gate["message"] = (f"{gate.get('message', '')} | {len(vanished)} baseline page(s) "
                               f"did not score in this run (allowed {allowed}): "
                               f"{', '.join(sorted(vanished)[:5])}").lstrip(" |")
        exit_code = 0 if gate.get("passed") else 1

    crops = [] if args.no_crops else write_crops(scored, args.tag)
    notes = [
        "Blocks are lettered with the official English that derive() read off the "
        "reference page (--ref-text), so both sides carry the same words and c_textiou / "
        "c_lines compare typesetting rather than translation length. Blocks the OCR could "
        "not read fall back to the machine translation and are flagged `uncertain`.",
        "Stylised onomatopoeia is OUT OF SCOPE and excluded, not penalised: an art-kind "
        "block the release did not re-letter is dropped from the scored set. The release "
        "variously leaves SFX alone, glosses it or redraws it, so scoring it would reward "
        "and punish identical behaviour on different pages.",
        "VERIFIED_WEAK pairs are scored but excluded from the 5th-percentile tail.",
        "Pages whose unassigned English lines exceed 20% are `reference_weak`: their "
        "group / text-IoU / line components are None and renormalise away.",
    ]
    report = write_report(rows, summary, args.tag, seed=args.seed, sample=args.sample,
                          gate=gate, crops=crops, notes=notes)

    if args.save_baseline:
        print(f"baseline: {save_baseline(rows, summary, args.tag)}")

    mean = summary.get("mean_R")
    print(f"\nR = {mean:.2f}" if mean is not None else "\nR = n/a")
    print(json.dumps(summary.get("component_means", {}), ensure_ascii=False, indent=1))
    print(f"report: {report}")
    if gate is not None:
        print(f"gate: {'PASS' if gate['passed'] else 'FAIL'} - {gate['message']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
