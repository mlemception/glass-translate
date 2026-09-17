# Corpus evaluation harness for typesetting

*2026-09-15*

Measure how close `demo/typeset_dev.py` output on a random Japanese page is to the
official English release, say which pages fail and why, and catch regressions.

The corpus is licensed third-party material. It is opened read-only, never extracted,
and **no page, crop or derived image ever enters the repository**. `/JA vs. EN/`,
`/demo/reference/` and `/demo/cache/` are anchored in `.gitignore`; everything the
harness writes lands under `demo/output/corpus/`. Committed reports carry numbers, page
identifiers and file paths only.

---

## 1. What the corpus actually is

The obvious assumption - a `JA/` folder and an `EN/` folder - is wrong, and the harness
was built against what is really there:

| side | shape |
|---|---|
| EN | `Jujutsu Kaisen ENG/` - 31 `.cbz` (ZIP), volumes 00-30, 5,711 pages. Entry names carry volume/chapter/page, and a two-page spread is **one wide entry** named `... - p006-p007 ...`. Exactly one colour page per volume, always `p000`. v30 is a different scanlator with flat `NNNN.ext` names. |
| JA | one `Jujutsu Kaisen JA.rar` - RAR5, non-solid, every entry **stored**, 10,650 pages in 125 folders. `rarfile` reads it in pure Python (0.17 s for the listing, ~1 ms/page); no external unrar/7z, and the archive is never extracted. |

JA covers v00-v24 only: **v25-v30 are EN-only**. 107 raw leaf folders roll up to 52
source groups, and **24 of 25 volumes have more than one candidate JA source**. The plain
`vNN` folder is often the wrong one - `v16` has ~2x the expected page count, `v17`/`v18`/
`v20` are landscape 1920x1080 / 3840x2160 / 3060x2160 screen dumps rather than scans, and
`v21`/`v22`/`v24` are colour-tinted. `corpus_index.CANONICAL_JA` pins the right folder per
volume; the Official Fanbook and the weekly-magazine group are excluded outright.

**Mirroring and page order: none.** Settled twice, independently: the flipped descriptor
scores 0.17-0.33 against 0.70-0.86 unflipped, and 92-97 % of sampled pages prefer the
unflipped orientation. The flip test is kept anyway because it is nearly free and is a
decisive per-volume sanity flag.

## 2. Pairing

Page numbers do **not** line up, and a constant offset does not fix it. Measured naive
index-to-index similarity against the aligned result:

| volume | naive | aligned | drift `en - ja` |
|---|---|---|---|
| v08 | 0.104 | 0.696 | 0 -> +1 |
| v12 | 0.181 | 0.752 | 0 -> +2 |
| v19 | 0.181 | 0.750 | **-7 -> +2** |

On v19 the drift changes *within one volume*, so the best single constant offset agrees
with the alignment on only 43 % of pages.

**Stage 1 - descriptor** (`demo/corpus_descriptor.py`, version 1). Greyscale, content-box
crop, pooled to 64x64, z-normed so a dot product is an NCC. Two channels,
`combined = 0.5*tone + 0.5*edge`. Measured against the best wrong match in the same row:
tone true 0.858+-0.111 vs false 0.257+-0.075 (min margin +0.126); edge true 0.714+-0.130
vs false 0.216+-0.054 (min margin +0.104). Resolution-robust: two scans of the same volume
at 2479x3508 and 884x1200 score mean 0.976.

Two things that sound sensible and were rejected **because they were measured**:
row/column projection profiles are useless (a wrong page scores 0.715), and down-weighting
text-heavy tiles does not help (margin 0.050 vs 0.104 plain) - lettering is a minority of
a manga page's edge energy and the gutters already dominate.

**Stage 2 - monotonic Needleman-Wunsch** over the whole similarity matrix, free to skip
pages on either side. Pairing `(i, j)` is rewarded by `sim - t0` with `t0 = 0.45`, and
skipping costs `gap = -0.06`. `t0` sits between the true-pair p1 (0.423) and p5 (0.517);
`gap` is calibrated so that skipping beats any pairing below 0.39, the observed
hard-negative ceiling (max 0.393).

**Stage 3 - verification** with `typeset_reference.align_pages` (ORB + RANSAC), which is
needed anyway to produce the homography, so verification is a byproduct. Its Pearson
`score` **must not be thresholded**: true and false pairs overlap (true min 0.192 vs false
max 0.193). The ORB inlier count separates totally - true pairs 207-1335, random non-pairs
2-13. `AlignResult` now carries `inliers` so the verifier reads a field instead of
regexing `note`.

**Decision rule, in order.** 1) aspect gate - `|log(aspect_en/aspect_ja)| > 0.15` means the
EN side is a joined spread: split at the midpoint (**right half = the lower JA index**,
right-to-left reading) and re-enter, else UNVERIFIABLE. 2) mirror gate. 3) fast accept at
`combined >= 0.55` **and** `margin >= 0.15`, which clears the hard-negative max (0.393) and
the no-true-pair margin max (0.110) by a wide band. 4) otherwise ORB: `>= 150` VERIFIED,
`60-150` VERIFIED_WEAK (scored, flagged, excluded from the p5 tail), `< 60` UNVERIFIABLE -
never guessed. 5) art-content gate, applied in `corpus_eval` after `erase_ground_truth`:
`kept_art / gt_region_px < 0.05` means the page has no shared artwork under its text
(author notes, bonus pages, ads) - the same page of the book, but pure noise to score.

Observed on v08: 184 matched, **163 fast-accepted, 19 ORB-verified (191-2902 inliers), 2
unverifiable at 0 inliers** (1.1 %), 14 JA and 8 EN pages correctly skipped. One of the two
unverifiable pairs had a Pearson score of 0.684 - a false accept if that statistic had been
thresholded.

## 3. Metrics

Six of the seven were already in the repo and are **reused, not reimplemented**:
`typeset_metrics.score_geometry` (containment, centring), `score_erase` (erase IoU,
`art_kept`), `score_leftover`, `score_lettering` (cap ratio, line counts),
`blocks_per_bubble`, and `tools/lpips_score.py` for art preservation - which already blanks
the English lettering on both sides and crops per block window, i.e. metric 7 verbatim.

The EN reference side comes from `typeset_reference.derive`: align, warp, OCR the EN page,
transform the boxes through the homography, intersect them with the EN ink. **Not** a naive
JA-EN diff, which would flag every scan/JPEG/registration difference as text.

New work is only `demo/corpus_metrics.py`: `text_iou` (dilate both sides by 0.15 em, because
two correct renders in different fonts share almost no *glyph* pixels but nearly the same
*text area*), `group_f1`, the `leftover_px / em_px**2` normalisation, and `lpips_excess`
(subtract a per-page floor measured on text-free regions - LPIPS is sensitive to JPEG
quality, screentone moire and gamma, so without the floor it is not comparable across
volumes).

A page whose unassigned EN lines exceed 20 % of its total is marked `reference_weak` and
excluded from the grouping, text-IoU and line metrics: rapidocr misses stylised and SFX
lettering, which is exactly where lettering is hardest.

## 4. The headline score

```
R_page = 100 * answered * prod( c_i ** (w_i / sum_of_present_w) )
answered = (GT blocks with EN lettering we rendered non-empty ink for)
         / (GT blocks with EN lettering, excluding `untranslated`)
```

| component | definition | w |
|---|---|---|
| `c_contain`  | `exp(-containment_mean / 0.010)` | 0.20 |
| `c_erase`    | `sqrt(erase_iou * art_kept)` | 0.15 |
| `c_art`      | `clamp(1 - lpips_excess / 0.30, 0, 1)` | 0.15 |
| `c_centre`   | `clamp(1 - centre_offset_em_mean / 1.10, 0, 1)` | 0.12 |
| `c_leftover` | `exp(-leftover_em2_mean / 0.25)` | 0.10 |
| `c_size`     | `clamp(1 - size_logratio_rms / 0.80, 0, 1)` | 0.10 |
| `c_group`    | `group_f1` | 0.08 |
| `c_lines`    | `line_exact` | 0.06 |
| `c_textiou`  | `text_iou_mean` | 0.04 |

The weights live in `demo/corpus_score.py` and nowhere else.

**Two of these kernels were wrong in the first draft, and the correction matters more than
any weight.** `c_size` was a Gaussian with no lower bound; it reached 1.6e-29 on a real
page, so a nominally 0.10-weight component carried **92 % of the variance of `ln R`** while
`c_contain` at 0.20 carried 6 %. It is now a bounded ramp. `c_centre`'s span was 0.50 em,
against which the **official release itself** scored 0.149 - worse than our render - because
a letterer offsets the block toward the balloon's tail; VIZ's own mean offset is 0.425 em
and its p90 is 1.075, which is where the span now sits. The two remaining exponential
kernels (`c_contain`, `c_leftover`) are floored at `COMPONENT_FLOOR = 0.005` so neither can
silently own the score; bounded components reaching a true 0 still zero the page, which is
what the degenerate-output audit requires. A component is only trustworthy if the thing it
is measuring against would pass it.

**Why geometric, not arithmetic.** An arithmetic mean lets a page with text hanging out of
every balloon buy its way back with a clean erase. A geometric mean cannot: any component
near 0 drags `R` to 0. A reader who sees text outside a balloon does not care how good the
inpainting was.

**Why `answered` multiplies** rather than being a component: every other term is a mean
over blocks we rendered, so without the gate, abstaining on the hard blocks raises all of
them. It is the single anti-degenerate term.

**Weights, in noticeability order.** `c_contain` is the largest because text crossing a
balloon outline is the only defect a reader spots at arm's length; the knee at 1 % of glyph
pixels is deliberately brutal, since `interior_core` is already inset by exactly the margin
`layout.py` targets, so a correct render never crosses it. `c_erase` and `c_art` are jointly
0.30 - the "does it look printed" axis - split because they fail differently: erase IoU
catches *not removing enough*, `art_kept`/LPIPS catch *removing too much*, and `c_erase` is
itself a geometric mean so bleaching a whole balloon (which maximises recall) scores 0.
`c_centre` at 0.12 is the most visible piece of a letterer's craft after containment, held
below the erase pair because a modest offset is forgivable and `interior_centre` is
tail-blind. `c_leftover` at 0.10 is unmistakable when present but rare and correlated with
erase IoU, so a higher weight would double-count. `c_size` at 0.10 uses a Gaussian with
sigma 0.15 because +-15 % is within a letterer's own variation. `c_group` at 0.08 is rare
but catastrophic when it fires - merged balloons are a *reading* error - and it usually drags
`c_contain` and `c_centre` with it. `c_lines` (0.06) and `c_textiou` (0.04) are deliberately
lowest: they punish a correct render whose translation differs in length from VIZ's, which
is a real and frequent condition of this harness (see section 7).

**Degenerate-output audit** (asserted in `tests/test_corpus_score.py`): draw nothing ->
`answered = 0` -> R = 0; erase nothing -> `erase_iou = 0` -> R = 0; bleach every balloon ->
`art_kept -> 0` -> R = 0; one dot at the optical centre -> `c_size`/`c_lines`/`c_textiou` ~ 0
-> R ~ 0; copy the JA page unchanged -> R = 0.

A page with 0 ground-truth blocks scores `R = null` and is excluded from every mean -
**never** scored 100. A page with no balloon blocks has `c_contain`, `c_centre` and
`c_leftover` as `None`; the remaining weights renormalise and the page is tagged
`free_text_only`.

**Three numbers, not one.** The report prints the page-weighted mean R, the **5th
percentile** (the bad tail is what a reader remembers, and it is what should drive the
work), and the count of `R = 0` pages - each split bubble-heavy vs free-text-heavy, because
they exercise `render/erase.py` and the `renderer/` sidecar respectively and averaging them
hides regressions in both.

## 5. The loop protocol

This is a **manual, repeatable cycle plus one command**. There is deliberately no
long-running measurement loop.

1. Run the fixed sample:
   `.venv\Scripts\python demo\corpus_eval.py --sample 50 --seed 1 --tag <tag>`
2. Read `demo/output/corpus/report-<tag>.md`: headline, component table, per-page
   distribution, and the worst 20 pages under each failing component.
3. Pick the **largest failure cluster** - the component with the lowest mean that also has
   the most pages under 0.9. Look at its crops in `demo/output/corpus/crops/<tag>/`.
4. Fix it in **one slice**, with a regression test in `tests/`.
5. Rerun the same seed with a new tag and the gate:
   `--sample 50 --seed 1 --tag <new> --baseline <old>`
6. Require both: the headline up, and the gate green (exit 0).
7. Record the tag and the numbers in `docs/perf/`.

The gate fails when the headline drops by more than 0.05 (run-to-run jitter from OCR and
ORB nondeterminism) or when either hard invariant regresses: a page that had zero overflow
now overflows, or a page that had zero leftover ink now has some.

Alongside the numbers, review the 24-page eyeball set
(`demo/corpus_eyeball.py --sheets --tag <tag>`) by eye. **Eyeball scores are never folded
into R.**

## 6. What this score cannot measure

Measurable and measured well: containment, panel-border collisions, leftover ink, erase
IoU/recall/precision, `art_kept`, art damage, optical centring, type size against the
official lettering, line count, block grouping, gross text-area overlap. These are
geometric and photometric facts about ink on paper, with a real reference.

**Not measurable - these need the human eyeball set:**

- **Break points at sense units.** `c_lines` says two renders both used three lines; it
  cannot tell `I'M GONNA / KILL / YOU` from `I'M / GONNA KILL / YOU`.
  `typeset.balanced_breaks` optimises rectangle-ness, not sense. This is probably the single
  largest remaining gap to the release, and the score is blind to it by construction.
  *(Partly measurable after all: `blocks.json` stores VIZ's per-line English boxes and text,
  so on blocks where the reference OCR is confident, comparing our line strings against
  theirs is a string comparison rather than an eyeball judgement. Worth building.)*
- **Word spacing.** A render reading `BEINGAN ADULTISSO CONFUSING!` is invisible to every
  component: the ink is in the right place, the right size, and inside the balloon.
- **The shape of the erase patch.** A white rectangle sitting over artwork registers in
  `erase_iou` as *something*, but nothing in the score notices that it reads as a box.
- **Emphasis.** There is no path from source emphasis to `compose.block_italic` or to font
  choice. The metrics see emphasis only as ink and score a flat render and an emphasised one
  identically.
- **Tail-aware placement.** A letterer nudges the block toward the tail so the balloon
  visibly belongs to its speaker. `interior_centre` is the inscribed-circle peak and is
  deliberately tail-blind, so a render `c_centre` calls perfect can sit wrong.
- **SFX treatment - OUT OF SCOPE, and excluded rather than measured.** The release variously
  leaves Japanese SFX alone, glosses it, or redraws it in English, and which is right is an
  editorial decision. A metric over it would **actively lie**: where VIZ left the SFX
  untouched we would be rewarded for not touching it, and where VIZ redrew it we would be
  punished for the identical behaviour. So an "art"-kind block that the release did not
  re-letter is dropped from the scored set entirely (`corpus_metrics.page_extras.is_sfx`),
  not counted as a false positive. On the first 50-page sample this excluded 25 blocks
  across 20 pages and took one page out of scoring altogether; `c_group` rose from 0.899 to
  0.942 purely because those blocks had been counted against us. The exemption is narrow on
  purpose: a **balloon** we lettered that the release did not is still a false positive.
- Hand-lettering judgement (optical kerning, block silhouette), translation quality, and
  reading order.

## 7. Known limitations of this first cut

**The English we letter with is rapidocr's *reading* of VIZ's lettering, not VIZ's text.**
The harness runs with `--ref-text`, so both sides carry the same words and the text metrics
compare typesetting rather than translation length. But 9 % of reference English lines come
back below 0.90 confidence and 3 % below 0.70; 10 % of blocks contain a sub-0.70 line. That
produces real misreadings in our render - "WHO... ARE IENOX", "LNOO I EVEN KNOW WHAT'S
RIGHT..." - and **R cannot see a wrong word at all**, so the harness can introduce the most
reader-salient defect on a page and still score that page in the 80s. Filtering blocks by
reference OCR confidence is the obvious next guard.

**A gaming hole is open and known.** `containment = outside_px / ink_px` divides by *our own*
ink, so a heavier face raises the denominator faster than the overflow ring: halving
containment from 0.020 to 0.010 moves `c_contain` from 0.135 to 0.368, and the same change
raises `text_iou`, for a render a reader simply calls "the type got heavier". `c_size`
measures cap height and does not object. Not yet closed.

**Shrink-to-fit points the wrong way on paper.** Shrinking type monotonically improves
`c_contain` (0.20), `c_centre` (0.12) and overflow while costing only `c_size` (0.10) and
`c_lines` (0.06) - nominally 0.32 against 0.16 *in favour of* the defect the harness exists
to catch. It is not currently exploitable because today's undersized blocks are also the
misplaced ones (block-level Spearman(cap_ratio, containment) = -0.241), but that is a
coincidence of the present defect, not a property of the metric.

**Two components cannot reach 1**, so R = 100 is not "identical to the release": the best
observed `c_textiou` is 0.581 at page level (it is partly a font-difference floor) and the
best `c_centre` is 0.878.

## 8. Files

| file | responsibility |
|---|---|
| `demo/corpus_index.py` | the only module that touches `JA vs. EN`: archives, `PageRef`, CRC+size identity, spread detect/split, colourfulness, in-memory decode |
| `demo/corpus_descriptor.py` | descriptor v1, content-box crop, tone/edge, mirror test, versioned int8 npz cache |
| `demo/corpus_pair.py` | similarity matrix, Needleman-Wunsch, the decision rule, ORB verification, `pairs.json` |
| `demo/corpus_metrics.py` | the four new metrics and the mapping into score components |
| `demo/corpus_score.py` | **the weights, and nowhere else**; page R, corpus mean/p5/zero-count, markdown tables, the gate comparison |
| `demo/corpus_eval.py` | sampling and selectors, per-page orchestration, triage report, crops, baseline and gate |
| `demo/corpus_eyeball.py` | the fixed 24-page human review set |

Cache layout under `demo/output/corpus/`: `desc/v1/<archive>/<group>.npz` (descriptors,
int8, keyed on `(DESCRIPTOR_VERSION, archive, entry, CRC, size, GRID, EDGE_SIDE,
CONTENT_PCT, channels)` - both `zipfile.ZipInfo` and `rarfile.RarInfo` give CRC and size
from the directory listing, so validating the whole cache costs one `infolist()`),
`pairs.json`, `pages/`, `reference/`, `renders/`, `metrics/`, `crops/<tag>/`,
`report-<tag>.md`, `baseline-<tag>.json`.
