"""Pair JA pages with their official EN counterparts, and verify each pair.

Page numbers do NOT line up and a constant offset does not fix it: the EN release
carries a colour frontispiece, front matter, author notes and ad pages the JA raw does
not.  Measured naive index-to-index similarity against the aligned result - v08
0.104 -> 0.696, v12 0.181 -> 0.752, v19 0.181 -> 0.750 - and on v19 the drift
``en - ja`` ranges -7..+2 WITHIN one volume, so the best single constant offset agrees
with the alignment on only 43% of pages.

So pairing is a monotonic Needleman-Wunsch over the whole similarity matrix, free to
skip pages on either side, followed by a decision rule that would rather mark a pair
UNVERIFIABLE than guess.

The verifier is ``typeset_reference.align_pages``, which is needed anyway to produce the
homography, so verification is a byproduct rather than an extra pass.  Its Pearson
``score`` must NOT be thresholded - it overlaps between true and false pairs (true min
0.192 vs false max 0.193).  The ORB inlier count separates totally: true pairs 207-1335,
random non-pairs 2-13.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_descriptor as CD  # noqa: E402
import corpus_index as CI  # noqa: E402

OUT_ROOT = CI.OUT_ROOT

# Alignment parameters.  t0 sits between the true-pair p1 (0.423) and p5 (0.517), so a
# pairing is only rewarded above the level where genuine pairs live; gap is calibrated so
# that skipping beats any pairing below t0 + gap = 0.39, the observed hard-negative
# ceiling (max 0.393).  Both are part of the parameters recorded with a pairing run.
T0 = 0.45
GAP = -0.06

# Fast accept: 0.55 clears the hard-negative max (0.393) by a wide band, and a 0.15
# margin clears the no-true-pair margin max (0.110).  ~85% of pages never reach ORB.
FAST_SIM = 0.55
FAST_MARGIN = 0.15

ORB_VERIFIED = 150  # p1 of the ORB inlier distribution over true pairs was 146
ORB_WEAK = 60  # below this there is no shared structure; false pairs never clear ~20

# |log(aspect_en / aspect_ja)| above this means the two sides are not the same shape of
# page - in practice, an unsplit spread, which is log(2) = 0.693 away.  It must NOT be
# tight: scans are trimmed differently, and the JA v23 source is 1230x1462 (aspect 0.841)
# against EN's 1500x2250 (0.667), a log ratio of 0.233.  At the original 0.15 that
# rejected all 183 of v23's pairs.  0.45 sits between the two and still catches a spread
# with a wide margin.
ASPECT_LOG_TOLERANCE = 0.45

STATUS_FAST = "VERIFIED_FAST"
STATUS_VERIFIED = "VERIFIED"
STATUS_WEAK = "VERIFIED_WEAK"
STATUS_UNVERIFIABLE = "UNVERIFIABLE"


# --------------------------------------------------------------------- page slots
@dataclass(frozen=True)
class PageSlot:
    """One readable page.  A joined spread yields two slots, right half first."""

    ref: CI.PageRef
    half: Optional[int]  # None = whole entry; 0 = right half, 1 = left half

    @property
    def slot_id(self) -> str:
        return self.ref.page_id if self.half is None else f"{self.ref.page_id}h{self.half}"

    @property
    def entry(self) -> str:
        return self.ref.entry

    @property
    def aspect(self) -> float:
        w = self.ref.width / 2 if self.half is not None else self.ref.width
        return w / self.ref.height if self.ref.height else 0.0


def pair_id(ja: PageSlot, en: PageSlot) -> str:
    """The identity of a PAIR, and the stem of every file derived from it.

    Taken from the EN side, because JA entry names carry no page number (they are
    Japanese paths, and ``PageRef.page`` is None for them), so a JA-derived id sorts
    arbitrarily and reads as ``pxxx``.  The JA CRC disambiguates, and the whole thing
    stays ASCII: it becomes a directory name and a ``cv2.imread`` path downstream.
    """
    vol = f"v{en.ref.volume:02d}" if en.ref.volume is not None else "vxx"
    page = f"p{en.ref.page:03d}" if en.ref.page is not None else "pxxx"
    half = "" if en.half is None else f"h{en.half}"
    return f"{vol}_{page}{half}_{ja.ref.crc & 0xFFFFFF:06x}"


def build_slots(refs: Sequence[CI.PageRef]) -> List[PageSlot]:
    """Expand spreads into two slots each, in manga reading order (right half first)."""
    slots: List[PageSlot] = []
    for ref in refs:
        if ref.is_spread:
            slots.append(PageSlot(ref, 0))
            slots.append(PageSlot(ref, 1))
        else:
            slots.append(PageSlot(ref, None))
    return slots


def slot_gray(archive: CI.Archive, slot: PageSlot) -> Optional[np.ndarray]:
    """Greyscale pixels of one slot, splitting a spread when needed."""
    gray = CI.decode_gray(archive.read(slot.entry))
    if gray is None or slot.half is None:
        return gray
    right, left = CI.split_spread(gray)
    return right if slot.half == 0 else left


def slot_descriptors(archive: CI.Archive, slots: Sequence[PageSlot],
                     cache: Optional[CD.DescriptorCache] = None) -> List[CD.Descriptor]:
    """Descriptors for slots, cached per (archive, group) with the half in the key."""
    cache = cache if cache is not None else CD.DescriptorCache()
    out: List[CD.Descriptor] = []
    touched: set = set()
    for slot in slots:
        ref = slot.ref
        entry_key = ref.entry if slot.half is None else f"{ref.entry}#{slot.half}"
        key = CD.cache_key(ref.archive, entry_key, ref.crc, ref.size)
        desc = cache.get(ref.archive, ref.group, key)
        if desc is None:
            gray = slot_gray(archive, slot)
            if gray is None:
                raise ValueError(f"cannot decode {ref.entry}")
            desc = CD.describe(gray)
            cache.put(ref.archive, ref.group, key, desc)
            touched.add((ref.archive, ref.group))
        out.append(desc)
    for archive_name, group in touched:
        cache.flush(archive_name, group)
    return out


# --------------------------------------------------------------------- alignment
def needleman_wunsch(sim: np.ndarray, t0: float = T0, gap: float = GAP,
                     ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Monotonic global alignment of two page sequences.

    ``sim`` is ``n_ja x n_en``.  Pairing (i, j) is rewarded by ``sim[i, j] - t0`` and
    skipping a page on either side costs ``gap``.  Returns ``(pairs, ja_gaps, en_gaps)``;
    monotonicity is structural, so the pairing can never cross itself.
    """
    n, m = sim.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    reward = sim.astype(np.float64) - t0
    f = np.empty((n + 1, m + 1), np.float64)
    ptr = np.zeros((n + 1, m + 1), np.int8)  # 0 = pair, 1 = skip ja, 2 = skip en
    f[0, 0] = 0.0
    f[1:, 0] = gap * np.arange(1, n + 1)
    f[0, 1:] = gap * np.arange(1, m + 1)
    ptr[1:, 0] = 1
    ptr[0, 1:] = 2
    for i in range(1, n + 1):
        prev, cur, row, prow = f[i - 1], f[i], reward[i - 1], ptr[i]
        for j in range(1, m + 1):
            diag = prev[j - 1] + row[j - 1]
            up = prev[j] + gap
            left = cur[j - 1] + gap
            if diag >= up and diag >= left:
                cur[j], prow[j] = diag, 0
            elif up >= left:
                cur[j], prow[j] = up, 1
            else:
                cur[j], prow[j] = left, 2
    pairs: List[Tuple[int, int]] = []
    ja_gaps: List[int] = []
    en_gaps: List[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = ptr[i, j]
        if step == 0 and i > 0 and j > 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == 1 and i > 0:
            ja_gaps.append(i - 1)
            i -= 1
        else:
            en_gaps.append(j - 1)
            j -= 1
    pairs.reverse()
    ja_gaps.reverse()
    en_gaps.reverse()
    return pairs, ja_gaps, en_gaps


def row_margin(sim: np.ndarray, i: int, j: int) -> float:
    """Chosen score in row ``i`` minus the best score at any OTHER column.

    Only column ``j`` is excluded.  Masking the neighbours ``j-1``/``j+1`` as well would
    make the margin blind to the adjacent page - which is precisely the off-by-one the
    alignment exists to prevent, and this margin is what lets ~92 % of pairs skip ORB
    verification.  A page whose neighbour scores almost as well must fall through to the
    slow gate, not be fast-accepted.
    """
    row = sim[i].astype(np.float64).copy()
    chosen = float(row[j])
    row[j] = -np.inf
    other = float(row.max())
    return chosen - other if np.isfinite(other) else chosen


# --------------------------------------------------------------------- decision
@dataclass
class PairDecision:
    page_id: str
    volume: Optional[int]
    ja_entry: str
    ja_half: Optional[int]
    en_entry: str
    en_half: Optional[int]
    combined: float
    margin: float
    mirrored: bool
    status: str
    reason: str
    inliers: int = 0
    align_score: float = 0.0


def decide_pair(ja: PageSlot, en: PageSlot, combined: float, margin: float,
                mirrored: bool, volume_mirrored: bool, verify=None) -> PairDecision:
    """Gates 1-4 of the decision rule.

    Gate 5 (the art-content gate) needs ``erase_ground_truth`` and therefore runs later,
    in ``corpus_eval``, which downgrades the record in place.

    ``verify`` is a callable ``() -> (inliers, score)``; it is only invoked when the fast
    gate does not fire, so ~85% of pages never pay for ORB.
    """
    base = dict(page_id=pair_id(ja, en), volume=en.ref.volume,
                ja_entry=ja.entry, ja_half=ja.half,
                en_entry=en.entry, en_half=en.half,
                combined=round(float(combined), 4), margin=round(float(margin), 4),
                mirrored=mirrored)

    if ja.aspect > 0 and en.aspect > 0:
        ratio = abs(float(np.log(en.aspect / ja.aspect)))
        if ratio > ASPECT_LOG_TOLERANCE:
            return PairDecision(**base, status=STATUS_UNVERIFIABLE,
                                reason=f"aspect mismatch (|log ratio| {ratio:.3f})")

    if mirrored and not volume_mirrored:
        return PairDecision(**base, status=STATUS_UNVERIFIABLE,
                            reason="flipped descriptor scored higher than unflipped")

    if combined >= FAST_SIM and margin >= FAST_MARGIN:
        return PairDecision(**base, status=STATUS_FAST,
                            reason=f"combined {combined:.3f} >= {FAST_SIM}, "
                                   f"margin {margin:.3f} >= {FAST_MARGIN}")

    if verify is None:
        return PairDecision(**base, status=STATUS_UNVERIFIABLE,
                            reason="below the fast gate and no verifier available")

    inliers, score = verify()
    if inliers >= ORB_VERIFIED:
        return PairDecision(**base, status=STATUS_VERIFIED, inliers=inliers,
                            align_score=round(float(score), 4),
                            reason=f"{inliers} ORB inliers")
    if inliers >= ORB_WEAK:
        return PairDecision(**base, status=STATUS_WEAK, inliers=inliers,
                            align_score=round(float(score), 4),
                            reason=f"only {inliers} ORB inliers (weak)")
    return PairDecision(**base, status=STATUS_UNVERIFIABLE, inliers=inliers,
                        align_score=round(float(score), 4),
                        reason=f"no shared structure ({inliers} ORB inliers)")


def _orb_verifier(ja_archive: CI.Archive, en_archive: CI.Archive,
                  ja: PageSlot, en: PageSlot):
    def run() -> Tuple[int, float]:
        import typeset_reference as TR

        ja_gray = slot_gray(ja_archive, ja)
        en_gray = slot_gray(en_archive, en)
        if ja_gray is None or en_gray is None:
            return 0, 0.0
        result = TR.align_pages(ja_gray, en_gray)
        return int(result.inliers), float(result.score)

    return run


# --------------------------------------------------------------------- per volume
def pair_volume(volume: int, ja_archive: CI.Archive, en_archive: CI.Archive,
                ja_refs: Sequence[CI.PageRef], en_refs: Sequence[CI.PageRef],
                *, cache: Optional[CD.DescriptorCache] = None,
                verify: bool = True) -> Dict[str, object]:
    """Align and verify one volume.  Returns the record written into ``pairs.json``."""
    cache = cache if cache is not None else CD.DescriptorCache()
    ja_slots = build_slots(sorted(ja_refs, key=lambda r: r.entry))
    en_slots = build_slots(sorted(en_refs, key=lambda r: r.entry))
    ja_desc = slot_descriptors(ja_archive, ja_slots, cache)
    en_desc = slot_descriptors(en_archive, en_slots, cache)

    sim = CD.similarity_matrix(ja_desc, en_desc)
    flip = CD.similarity_matrix(ja_desc, en_desc, flip=True)
    pairs, ja_gaps, en_gaps = needleman_wunsch(sim)

    flipped_wins = sum(1 for i, j in pairs if flip[i, j] > sim[i, j])
    volume_mirrored = bool(pairs) and flipped_wins > len(pairs) / 2

    decisions: List[PairDecision] = []
    for i, j in pairs:
        verifier = (_orb_verifier(ja_archive, en_archive, ja_slots[i], en_slots[j])
                    if verify else None)
        decisions.append(decide_pair(
            ja_slots[i], en_slots[j], float(sim[i, j]), row_margin(sim, i, j),
            mirrored=bool(flip[i, j] > sim[i, j]), volume_mirrored=volume_mirrored,
            verify=verifier,
        ))

    counts: Dict[str, int] = {}
    for d in decisions:
        counts[d.status] = counts.get(d.status, 0) + 1
    usable = [d for d in decisions if d.status != STATUS_UNVERIFIABLE]
    return {
        "volume": volume,
        "ja_group": ja_refs[0].group if ja_refs else "",
        "en_archive": en_refs[0].archive if en_refs else "",
        "ja_slots": len(ja_slots),
        "en_slots": len(en_slots),
        "stats": {
            "matched": len(pairs),
            "ja_skipped": len(ja_gaps),
            "en_skipped": len(en_gaps),
            "volume_mirrored": volume_mirrored,
            "flipped_wins": flipped_wins,
            "by_status": counts,
            "mean_combined": (round(float(np.mean([d.combined for d in usable])), 4)
                              if usable else None),
        },
        "pairs": [asdict(d) for d in decisions],
    }


def write_pairs(records: Sequence[Dict[str, object]], out_path: Optional[Path] = None) -> Path:
    out_path = Path(out_path) if out_path else OUT_ROOT / "pairs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {"t0": T0, "gap": GAP, "fast_sim": FAST_SIM, "fast_margin": FAST_MARGIN,
                   "orb_verified": ORB_VERIFIED, "orb_weak": ORB_WEAK,
                   "descriptor_version": CD.DESCRIPTOR_VERSION},
        "volumes": {str(r["volume"]): r for r in records},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_path


def load_pairs(path: Optional[Path] = None) -> Dict[str, object]:
    path = Path(path) if path else OUT_ROOT / "pairs.json"
    return json.loads(path.read_text(encoding="utf-8"))


def verified_pairs(payload: Dict[str, object], include_weak: bool = True) -> List[Dict[str, object]]:
    """Flat list of usable pairs across all volumes, in volume/page order."""
    ok = {STATUS_FAST, STATUS_VERIFIED} | ({STATUS_WEAK} if include_weak else set())
    volumes = payload.get("volumes", {})
    out: List[Dict[str, object]] = []
    for key in sorted(volumes, key=lambda k: int(k)):
        record = volumes[key]
        for pair in record["pairs"]:
            if pair["status"] in ok:
                out.append({**pair, "volume": record["volume"],
                            "ja_group": record["ja_group"],
                            "en_archive": record["en_archive"]})
    return out


# --------------------------------------------------------------------- CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=CI.DEFAULT_CORPUS)
    parser.add_argument("--volume", type=int, action="append", default=[],
                        help="volume to pair (repeatable); default all canonical volumes")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-verify", action="store_true", help="skip ORB verification")
    args = parser.parse_args(argv)

    corpus = Path(args.corpus)
    records = []
    # The JA side is one 3.7 GB archive: on Windows an un-closed handle also holds a
    # lock, so it is released even when a volume raises.
    ja_archive = CI.open_archive(corpus / CI.JA_ARCHIVE_NAME)
    try:
        ja_refs_all = CI.enumerate_archive(ja_archive, "ja", decode=False)
        by_volume: Dict[int, List[CI.PageRef]] = {}
        for ref in ja_refs_all:
            if ref.volume is not None and not CI.is_excluded_ja(ref.group):
                by_volume.setdefault(ref.volume, []).append(ref)

        volumes = sorted(args.volume) if args.volume else sorted(by_volume)
        cache = CD.DescriptorCache()
        for volume in volumes:
            matches = sorted((corpus / CI.EN_DIR_NAME).glob(f"*v{volume:02d} *.cbz"))
            if not matches or volume not in by_volume:
                print(f"v{volume:02d}: no JA/EN pair available", file=sys.stderr)
                continue
            en_archive = CI.open_archive(matches[0])
            try:
                en_refs = CI.enumerate_archive(en_archive, "en", decode=False)
                record = pair_volume(volume, ja_archive, en_archive, by_volume[volume],
                                     en_refs, cache=cache, verify=not args.no_verify)
            finally:
                en_archive.close()
            records.append(record)
            print(f"v{volume:02d}: {json.dumps(record['stats'], ensure_ascii=False)}",
                  flush=True)
    finally:
        ja_archive.close()
    out = write_pairs(records, args.out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
