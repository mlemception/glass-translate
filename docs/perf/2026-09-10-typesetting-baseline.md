# Typesetting baseline: the heuristic eraser before the quality renderer — 2026-09-10

Measured with the generalised harness before any render code changed (plan:
`docs/plans/2026-09-10-quality-renderer.md`, slice 1d).  Five pages: `Examples/before.jpg`
(1096x1600, reference `after.webp`) and the four `more_comparisons/` pairs (ja 764x1200, eng
1600x1067 aligned into the ja grid by ORB + ECC homography; alignment scores 0.78–0.97, residual
shift under 1 px on every block window checked with phase correlation).

How to reproduce (all from the project root, `PYTHONUTF8=1`, venv interpreter):

```bat
.venv\Scripts\python demo\typeset_reference.py --batch --crops
.venv\Scripts\python demo\typeset_dev.py --batch --ref-text --tag base --quiet
.venv\Scripts\python demo\typeset_metrics.py --batch --tag base --quiet
```

`demo/typeset_reference.py` derives the ground truth under `demo/reference/<stem>/` (the aligned
reference, the letterer's erase mask, the kept art, the English lettering ink and per-block
lettering statistics); the English wording in `demo/reference_text_<stem>.json` was bootstrapped
from OCR of the reference pages and corrected by hand from greyscale review sheets (4 of 59 new
blocks left `_uncertain`).  `demo/typeset_metrics.py` scores a render inside the region the
letterer worked in: a bubble's paper interior, or the block's own OCR boxes grown by 0.5 em for
free text.  Blocks the renderer leaves untranslated by design (author name, watermark, series
logo) are listed but excluded from the means.

## Metric definitions

| metric | meaning |
|---|---|
| IoU / recall / prec | ink we removed vs the ink the letterer removed (ink counts as removed when no ink survives within 2 px, the scan tolerance); recall < 1 = Japanese residue, precision < 1 = we destroyed something the letterer kept |
| art kept | share of the art ink the letterer kept that survives in our erased render |
| SSIM / MAE | structural similarity and mean absolute grey difference between our erased render and the aligned reference in the erased neighbourhood (erase mask dilated 4 px, English lettering excluded) |
| cap ratio | our cap height (0.84 x the chosen font size) / the reference lettering's cap height (row clusters of its ink) |
| inset ours/ref | distance from the lettering to the bubble outline, in source ems (bubbles only) |
| centre dx/dy | offset of the lettering's box centre from the bubble's centroid, in source ems (bubbles; a shared bubble uses the block's own part) |
| lines | our line count / the reference's |
| overflow / collision | our lettering ink outside the bubble interior / on another block's text or bubble or on a panel border line (pixels) |

## Failure catalogue (current eraser, `render/erase.py` before slice 2)

1. **Bubbles: what the numbers first said, and what they mean.**  The first scoring pass judged
   bubbles inside the whole paper interior and reported recall 0.10–0.35 on a third of them; that
   was outline-edge scan noise (the inner edge of the outline differs by 1–2 px between the two
   scans and the reference's English lettering covers most of the Japanese ink, so the judged
   mask was tiny), not residue.  Judged inside the interior eroded by 3 px, the old eraser leaves
   ink in 2 of 31 bubbles (2ja 5: 76 px at the tail junction, 3jp: 7 px) and the mean bubble
   recall is 0.93.  The bubble redraw (fill the whole paper interior, keep the outline with its
   thickness and tail) is therefore a robustness change on these pages - strays, furigana and
   enclosed marks anywhere in the bubble, joined bubbles, tails, dark paper are covered by the
   synthetic tests in `tests/test_erase.py` - rather than a measured gain; the re-measure at the
   end of this file shows identical scores.
2. **Free text over art: residue and smears.**  Recall 0.35–0.65 on hatched or toned art (2ja 3,
   3jp 5/7/10, 4ja 3/4), SSIM 0.55–0.75 in the erased neighbourhood: local-mean fills read as
   grey smears against the reference's redrawn tone, and glyph fragments stay where the
   classifier called them art.  Precision stays 0.95–1.00 (art is rarely destroyed); the one
   exception is `before` region A/C (precision 0.87, art kept 0.79).  This is the case for the
   generative fill.
3. **Bubble detection misses.**  A balloon whose paper touches the panel background through a gap
   is treated as free text and lettered with a halo against its left edge instead of centred
   (4ja block 1, 1ja 4/5); a caption-sized bubble (4ja 8, 348x466) is lettered at its top (centre
   dy -3.4 em).
4. **Shared bubbles.**  Two blocks in one bubble (3jp 13/15, 4ja 10/11, 1ja 8/9) letter into each
   other's half: overflow / collision pixels on 3 and 5 blocks respectively, and the English page
   merges them into one lettered block.
5. **Size and inset.**  Mean cap ratio 1.06 but spread 0.39–1.40 across bubbles (our uniform
   page size vs the letterer's per-bubble choice); our inset 0.35 em against the letterer's 0.44;
   line counts match on 26 of 66 blocks (we set fewer, wider lines).
6. **Untranslated blocks** (author name, watermark, logo: `before` 8/9, 3jp 0) are left as-is by
   design; the reference pages remove or replace them.  Excluded from the means.

## Results (tag `base`)

The tables below are the metrics script's output, per page (means) and per block.

## Summary (base)

| page | blocks | IoU | recall | prec | art kept | SSIM | MAE | bubble residue px (bubbles) | cap ratio | inset ours/ref | centre abs (em) | lines same | overflows | collisions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 8 | 0.84 | 0.96 | 0.87 | 0.72 | 0.647 | 12.8 | 0 (0/2) | 0.78 | 0.24/0.30 | 0.04 | 2/8 | 0 | 0 |
| 1ja | 16 | 0.94 | 0.94 | 1.00 | 1.00 | 0.663 | 7.3 | 0 (0/7) | 1.14 | 0.29/0.34 | 0.39 | 6/16 | 1 | 1 |
| 2ja | 13 | 0.76 | 0.77 | 1.00 | 0.99 | 0.783 | 8.5 | 74 (1/6) | 1.18 | 0.30/0.26 | 0.20 | 6/13 | 0 | 0 |
| 3jp | 17 | 0.80 | 0.82 | 0.95 | 0.87 | 0.801 | 8.1 | 7 (1/10) | 0.99 | 0.43/0.59 | 0.29 | 9/17 | 1 | 2 |
| 4ja | 12 | 0.77 | 0.79 | 0.97 | 0.81 | 0.774 | 8.9 | 0 (0/6) | 1.11 | 0.35/0.51 | 1.03 | 3/12 | 1 | 2 |
| **all** | 66 | 0.82 | 0.85 | 0.97 | 0.87 | 0.742 | 8.7 | 81 (2/31) | 1.06 | 0.35/0.44 | 0.42 | 26/66 | 3 | 5 |

### before (base)

| # | kind | IoU | recall | prec | art kept | SSIM | MAE | residue px | cap ratio | inset ours/ref (em) | centre dx/dy (em) | lines ours/ref | overflow px | collision px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | bubble | 1.00 | 1.00 | 1.00 | - | 0.977 | 1.0 | 0 | 0.84 | 0.27/0.30 | -0.02/-0.05 | 5/5 | 0 | 0 |
| 1 | bubble | 1.00 | 1.00 | 1.00 | - | 0.953 | 1.0 | 0 | 1.30 | 0.22/0.29 | -0.07/0.05 | 15/11 | 0 | 0 |
| 2 | art | 0.36 | 0.91 | 0.37 | 0.39 | 0.349 | 35.1 | - | 0.63 | -/- | -/- | 6/1 | 0 | 0 |
| 3 | art | 0.81 | 0.95 | 0.84 | 0.78 | 0.663 | 18.3 | - | - | -/- | -/- | 2/0 | 0 | 0 |
| 4 | art | 0.93 | 0.99 | 0.94 | 0.84 | 0.762 | 5.4 | - | 0.38 | -/- | -/- | 11/5 | 0 | 0 |
| 5 | art | 0.86 | 0.90 | 0.95 | 0.84 | 0.521 | 15.4 | - | 0.75 | -/- | -/- | 9/3 | 0 | 0 |
| 6 | flat | 1.00 | 1.00 | 1.00 | 1.00 | 0.786 | 0.3 | - | 0.93 | -/- | -/- | 9/9 | 0 | 0 |
| 7 | art | 0.78 | 0.89 | 0.87 | 0.50 | 0.168 | 26.3 | - | 0.65 | -/- | -/- | 7/4 | 0 | 0 |
| 8 | art | 0.00 | 0.00 | - | 1.00 | 0.089 | 89.4 | - | - | -/- | -/- | 0/0 | 0 | 0 |
| 9 | art | 0.00 | 0.00 | - | 1.00 | 0.056 | 105.3 | - | - | -/- | -/- | 0/0 | 0 | 0 |
| **mean** | 8 blocks | 0.84 | 0.96 | 0.87 | 0.72 | 0.647 | 12.8 | 0 in 0/2 | 0.78 | 0.24/0.30 | abs 0.04 | 2/8 same | 0 blocks | 0 blocks |

### 1ja (base)

| # | kind | IoU | recall | prec | art kept | SSIM | MAE | residue px | cap ratio | inset ours/ref (em) | centre dx/dy (em) | lines ours/ref | overflow px | collision px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | art | - | - | - | 1.00 | - | - | - | 1.18 | -/- | -/- | 8/3 | 0 | 0 |
| 1 | art | 1.00 | 1.00 | 1.00 | 1.00 | 0.327 | 10.9 | - | 1.24 | -/- | -/- | 4/5 | 0 | 0 |
| 2 | bubble | 1.00 | 1.00 | 1.00 | - | 0.946 | 1.0 | 0 | 1.06 | 0.35/0.33 | 0.03/-0.08 | 4/3 | 0 | 0 |
| 3 | bubble | 1.00 | 1.00 | 1.00 | - | 0.905 | 0.9 | 0 | 1.24 | 0.34/0.62 | -0.43/0.46 | 5/6 | 0 | 0 |
| 4 | art | 1.00 | 1.00 | 1.00 | - | 0.339 | 17.2 | - | - | -/- | -/- | 2/0 | 0 | 0 |
| 5 | art | 0.58 | 0.58 | 1.00 | - | 0.044 | 30.5 | - | - | -/- | -/- | 1/0 | 0 | 0 |
| 6 | flat | 1.00 | 1.00 | 1.00 | - | 0.122 | 17.7 | - | 0.91 | -/- | -/- | 1/3 | 0 | 0 |
| 7 | flat | 1.00 | 1.00 | 1.00 | - | 0.330 | 14.7 | - | - | -/- | -/- | 3/0 | 0 | 0 |
| 8 | bubble | 1.00 | 1.00 | 1.00 | - | 0.952 | 1.0 | 0 | 1.24 | 0.06/0.06 | -0.12/-0.02 | 3/3 | 2 | 61 |
| 9 | flat | 1.00 | 1.00 | 1.00 | - | 0.917 | 1.1 | - | - | -/- | -/- | 1/0 | 0 | 0 |
| 10 | bubble | 1.00 | 1.00 | 1.00 | - | 0.960 | 1.0 | 0 | 1.24 | 0.45/0.05 | -0.14/0.62 | 3/3 | 0 | 0 |
| 11 | bubble | 1.00 | 1.00 | 1.00 | - | 0.921 | 0.9 | 0 | 1.24 | 0.41/0.58 | 0.06/-0.83 | 1/1 | 0 | 0 |
| 12 | bubble | 1.00 | 1.00 | 1.00 | - | 0.954 | 1.0 | 0 | 1.24 | 0.12/0.69 | 0.05/-1.53 | 3/3 | 0 | 0 |
| 13 | art | 0.86 | 0.86 | 1.00 | 1.00 | 0.688 | 1.1 | - | 1.14 | -/- | -/- | 5/5 | 0 | 0 |
| 14 | art | 0.63 | 0.63 | 1.00 | - | 0.603 | 9.0 | - | 1.05 | -/- | -/- | 4/4 | 0 | 0 |
| 15 | bubble | 1.00 | 1.00 | 1.00 | - | 0.933 | 1.0 | 0 | 0.93 | 0.30/0.05 | -0.55/-0.51 | 7/4 | 0 | 0 |
| **mean** | 16 blocks | 0.94 | 0.94 | 1.00 | 1.00 | 0.663 | 7.3 | 0 in 0/7 | 1.14 | 0.29/0.34 | abs 0.39 | 6/16 same | 1 blocks | 1 blocks |

### 2ja (base)

| # | kind | IoU | recall | prec | art kept | SSIM | MAE | residue px | cap ratio | inset ours/ref (em) | centre dx/dy (em) | lines ours/ref | overflow px | collision px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | bubble | 1.00 | 1.00 | 1.00 | - | 0.908 | 1.2 | 0 | 1.07 | 0.23/0.06 | -0.25/0.56 | 11/12 | 0 | 0 |
| 1 | bubble | 1.00 | 1.00 | 1.00 | - | 0.935 | 1.0 | 0 | - | 0.24/0.26 | 0.08/-0.61 | 2/2 | 0 | 0 |
| 2 | art | 0.47 | 0.47 | 1.00 | 1.00 | 0.665 | 17.0 | - | 1.98 | -/- | -/- | 6/3 | 0 | 0 |
| 3 | art | 0.35 | 0.35 | 1.00 | 1.00 | 0.730 | 21.7 | - | 0.84 | -/- | -/- | 2/2 | 0 | 0 |
| 4 | art | 0.46 | 0.46 | 1.00 | 0.91 | 0.652 | 21.5 | - | 1.14 | -/- | -/- | 7/6 | 0 | 0 |
| 5 | bubble | 1.00 | 1.00 | 1.00 | 1.00 | 0.850 | 0.9 | 74 | 1.09 | 0.11/0.06 | 0.11/0.04 | 4/5 | 0 | 0 |
| 6 | bubble | 1.00 | 1.00 | 1.00 | - | 0.845 | 0.9 | 0 | 1.09 | 0.55/0.49 | 0.23/-0.11 | 7/7 | 0 | 0 |
| 7 | bubble | 1.00 | 1.00 | 1.00 | - | 0.703 | 4.2 | 0 | 1.07 | 0.22/0.33 | -0.03/0.23 | 5/3 | 0 | 0 |
| 8 | art | 0.68 | 0.68 | 1.00 | 1.00 | 0.757 | 10.3 | - | 1.04 | -/- | -/- | 4/4 | 0 | 0 |
| 9 | art | 0.48 | 0.48 | 1.00 | 1.00 | 0.808 | 12.0 | - | 1.09 | -/- | -/- | 3/3 | 0 | 0 |
| 10 | bubble | 1.00 | 1.00 | 1.00 | - | 0.933 | 1.0 | 0 | 1.40 | 0.46/0.39 | 0.08/-0.04 | 3/2 | 0 | 0 |
| 11 | art | 0.79 | 0.82 | 0.96 | 1.00 | 0.715 | 8.7 | - | 1.09 | -/- | -/- | 6/6 | 0 | 0 |
| 12 | art | 0.69 | 0.69 | 1.00 | - | 0.677 | 10.7 | - | 1.33 | -/- | -/- | 6/5 | 0 | 0 |
| **mean** | 13 blocks | 0.76 | 0.77 | 1.00 | 0.99 | 0.783 | 8.5 | 74 in 1/6 | 1.18 | 0.30/0.26 | abs 0.20 | 6/13 same | 0 blocks | 0 blocks |

### 3jp (base)

| # | kind | IoU | recall | prec | art kept | SSIM | MAE | residue px | cap ratio | inset ours/ref (em) | centre dx/dy (em) | lines ours/ref | overflow px | collision px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | art | 0.95 | 0.95 | 1.00 | 1.00 | 0.897 | 5.3 | - | - | -/- | -/- | 2/0 | 0 | 0 |
| 1 | bubble | 1.00 | 1.00 | 1.00 | - | 0.911 | 1.0 | 0 | 0.79 | 0.22/0.06 | 1.55/-0.69 | 6/6 | 0 | 0 |
| 2 | bubble | 1.00 | 1.00 | 1.00 | - | 0.883 | 1.0 | 0 | 1.14 | 0.43/0.72 | 0.01/-0.11 | 6/5 | 0 | 0 |
| 3 | bubble | 0.94 | 0.94 | 1.00 | - | 0.839 | 2.8 | 7 | 0.39 | 0.07/0.07 | 0.13/-0.06 | 4/2 | 34 | 274 |
| 4 | flat | 0.43 | 0.43 | 1.00 | - | 0.592 | 17.9 | - | 0.77 | -/- | -/- | 3/4 | 0 | 0 |
| 5 | art | 0.57 | 0.63 | 0.86 | 0.57 | 0.293 | 24.9 | - | 1.23 | -/- | -/- | 6/7 | 0 | 0 |
| 6 | art | 0.90 | 0.90 | 1.00 | 1.00 | 0.743 | 6.2 | - | 1.12 | -/- | -/- | 5/5 | 0 | 1 |
| 7 | art | 0.27 | 0.29 | 0.76 | 0.92 | 0.573 | 27.4 | - | 0.44 | -/- | -/- | 1/1 | 0 | 0 |
| 8 | bubble | 1.00 | 1.00 | 1.00 | - | 0.873 | 0.9 | 0 | 1.17 | 0.29/0.65 | -0.10/-0.09 | 6/6 | 0 | 0 |
| 9 | bubble | 1.00 | 1.00 | 1.00 | - | 0.952 | 1.0 | 0 | 0.90 | 0.25/0.35 | 0.09/-0.11 | 1/1 | 0 | 0 |
| 10 | art | 0.10 | 0.10 | 1.00 | 1.00 | 0.631 | 23.8 | - | 1.28 | -/- | -/- | 4/6 | 0 | 0 |
| 11 | bubble | 1.00 | 1.00 | 1.00 | - | 0.934 | 1.0 | 0 | 0.84 | 0.72/0.71 | 0.05/0.27 | 3/3 | 0 | 0 |
| 12 | bubble | 1.00 | 1.00 | 1.00 | - | 0.976 | 1.0 | 0 | 0.90 | 0.24/0.70 | -0.32/-1.12 | 2/2 | 0 | 0 |
| 13 | bubble | 1.00 | 1.00 | 1.00 | - | 0.900 | 1.0 | 0 | 1.23 | 0.36/0.51 | -0.10/-0.19 | 5/5 | 0 | 0 |
| 14 | art | 0.45 | 0.68 | 0.57 | 0.76 | 0.677 | 20.1 | - | 1.23 | -/- | -/- | 4/5 | 0 | 0 |
| 15 | bubble | 1.00 | 1.00 | 1.00 | - | 0.969 | 1.0 | 0 | 1.23 | 0.60/1.02 | 0.11/-0.03 | 3/3 | 0 | 0 |
| 16 | bubble | 1.00 | 1.00 | 1.00 | - | 0.974 | 1.0 | 0 | 1.17 | 1.08/1.10 | 0.06/-0.53 | 3/2 | 0 | 0 |
| **mean** | 17 blocks | 0.80 | 0.82 | 0.95 | 0.87 | 0.801 | 8.1 | 7 in 1/10 | 0.99 | 0.43/0.59 | abs 0.29 | 9/17 same | 1 blocks | 2 blocks |

### 4ja (base)

| # | kind | IoU | recall | prec | art kept | SSIM | MAE | residue px | cap ratio | inset ours/ref (em) | centre dx/dy (em) | lines ours/ref | overflow px | collision px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | bubble | 1.00 | 1.00 | 1.00 | - | 0.849 | 1.0 | 0 | 1.01 | 0.70/1.09 | 0.13/0.00 | 6/7 | 0 | 0 |
| 1 | art | 0.70 | 0.97 | 0.72 | 0.12 | 0.748 | 7.5 | - | 1.29 | -/- | -/- | 3/4 | 0 | 0 |
| 2 | art | 0.79 | 0.79 | 1.00 | - | 0.587 | 10.7 | - | 1.29 | -/- | -/- | 4/5 | 0 | 0 |
| 3 | art | 0.68 | 0.70 | 0.96 | 0.93 | 0.807 | 13.1 | - | 1.42 | -/- | -/- | 3/3 | 0 | 0 |
| 4 | art | 0.14 | 0.14 | 1.00 | 1.00 | 0.731 | 36.5 | - | 1.04 | -/- | -/- | 3/3 | 0 | 0 |
| 5 | bubble | 1.00 | 1.00 | 1.00 | - | 0.971 | 1.0 | 0 | 1.09 | 0.67/0.62 | 0.23/0.22 | 4/3 | 0 | 0 |
| 6 | bubble | 1.00 | 1.00 | 1.00 | - | 0.978 | 1.0 | 0 | 1.18 | 0.26/0.51 | 0.05/-1.24 | 1/1 | 0 | 0 |
| 7 | bubble | 1.00 | 1.00 | 1.00 | - | 0.912 | 0.9 | 0 | 1.18 | 0.18/0.74 | 0.17/0.16 | 4/3 | 0 | 0 |
| 8 | bubble | 1.00 | 1.00 | 1.00 | - | 0.612 | 1.9 | 0 | 0.64 | 0.22/0.04 | -2.04/-4.75 | 6/7 | 0 | 0 |
| 9 | art | 0.91 | 0.91 | 1.00 | 1.00 | 0.726 | 3.3 | - | 0.81 | -/- | -/- | 5/2 | 0 | 0 |
| 10 | art | 0.00 | 0.00 | - | 1.00 | 0.694 | 29.3 | - | 1.31 | -/- | -/- | 7/2 | 0 | 1702 |
| 11 | bubble | 1.00 | 1.00 | 1.00 | - | 0.673 | 1.0 | 0 | 1.09 | 0.07/0.07 | 2.12/-1.25 | 6/4 | 96 | 618 |
| 12 | art | 0.00 | 0.00 | - | 1.00 | 0.338 | 55.7 | - | - | -/- | -/- | 0/1 | 0 | 0 |
| **mean** | 12 blocks | 0.77 | 0.79 | 0.97 | 0.81 | 0.774 | 8.9 | 0 in 0/6 | 1.11 | 0.35/0.51 | abs 1.03 | 3/12 same | 1 blocks | 2 blocks |


## Re-measure after slice 2 (bubble redraw, tag `redraw`)

Same pages, same ground truth, after `render/erase.py` started redrawing bubbles (fill the paper interior, keep the outline).  `bubble residue px` is ink of any kind left in the judged interior core of the erased render (0 = clean paper); `recall` inside bubbles only counts the small part of the Japanese ink that the reference's English lettering does not cover, so it moves little.

### Summary (redraw)

| page | blocks | IoU | recall | prec | art kept | SSIM | MAE | bubble residue px (bubbles) | cap ratio | inset ours/ref | centre abs (em) | lines same | overflows | collisions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 8 | 0.84 | 0.96 | 0.87 | 0.72 | 0.647 | 12.8 | 0 (0/2) | 0.78 | 0.24/0.30 | 0.04 | 2/8 | 0 | 0 |
| 1ja | 16 | 0.94 | 0.94 | 1.00 | 1.00 | 0.663 | 7.3 | 0 (0/7) | 1.14 | 0.29/0.34 | 0.39 | 6/16 | 1 | 1 |
| 2ja | 13 | 0.76 | 0.77 | 1.00 | 0.99 | 0.783 | 8.5 | 74 (1/6) | 1.18 | 0.30/0.26 | 0.20 | 6/13 | 0 | 0 |
| 3jp | 17 | 0.80 | 0.82 | 0.95 | 0.87 | 0.802 | 8.0 | 7 (1/10) | 0.99 | 0.43/0.59 | 0.29 | 9/17 | 1 | 2 |
| 4ja | 12 | 0.79 | 0.88 | 0.90 | 0.61 | 0.732 | 9.0 | 0 (0/6) | 1.11 | 0.35/0.51 | 1.03 | 3/12 | 1 | 2 |
| **all** | 66 | 0.83 | 0.86 | 0.95 | 0.83 | 0.734 | 8.7 | 81 (2/31) | 1.06 | 0.35/0.44 | 0.42 | 26/66 | 3 | 5 |

## 2026-09-14: three defects fixed — connected bubbles, optical centring, furigana ink

Same five pages, same ground truth.  The "before" column is tag **`pre`**: the renderer exactly as
it stands at commit `295225b`, rendered in a detached worktree and scored with *this* file's
current metric definitions, so both columns are read off one ruler.  Scoring an old render in the
working tree does **not** work — `demo/typeset_metrics.py` calls `build_blocks` at scoring time, so
it would judge old pixels against new interiors.  The "after" column is tag **`fixA4`**.

```bat
.venv\Scripts\python demo\typeset_dev.py --batch --ref-text --tag fixA4 --regions all --quiet
.venv\Scripts\python demo\typeset_metrics.py --batch --tag fixA4 --quiet
renderer\.venv\Scripts\python tools\lpips_score.py --batch --tag pre --tag fixA4
```

### A — connected balloons were lettered as one block, or as none

Artists erase the arc two balloons share, so their interiors are a single 4-connected paper
component.  `layout._make_block` took the component under the first member line's centre pixel and
tested *that whole component* against a *single block's* `source_rect`; the joined region failed
`_BUBBLE_MAX_AREA_RATIO`, `_BUBBLE_MAX_DIM_RATIO` and `_BUBBLE_MIN_SOLIDITY`, so every block on it
fell through to the free-text branch.  `_split_shared_bubbles` could not help: it only ran over
blocks that had already passed the gate.

Two sub-cases needed different fixes, and measuring told them apart.  2ja's and 3jp's components
are not "two balloons" at all but **the whole 764x1200 page** (solidity 0.392 and 0.303); no
widening of the text box in the ratios makes that paper convex, so union-of-blocks gating alone
cannot work.  Conversely 1ja 13 and 14 are single clean convex balloons (solidity 0.918 / 0.961)
failing only `_BUBBLE_MAX_DIM_RATIO`, **by 2 px each**, because that gate compares the balloon to
the text the OCR *found* and the OCR found one column of five.  So both were needed:
`render/bubbles.py` seals each block's own room out of a component at necks narrower than
`_NECK_MAX_EM` and cuts a shared room at its waist, *and* `_dim_limits` applies the dimension ratio
only along the reading direction, with a `_BUBBLE_MAX_COLUMNS` floor across it where the OCR only
ever reports a lower bound.

A third gate was needed once the cut existed: a room carved out of panel paper is open along most
of its edge, where a balloon is walled by its own outline.  `_BUBBLE_MIN_WALLED` measures the share
of an interior's edge that is ink, artwork or a neighbour's interior — real balloons score
0.51–1.00, the six false positives 0.03–0.48.  Without it `before` blocks 2, 3 and 5 became
"balloons" and the eraser painted flat paper over a wooden scaffold: `art kept` 0.72 → 0.52 with
1745 px of artwork reported as leftover ink.

`_split_shared_bubbles` is **deleted** — once every component is partitioned up front, no two
blocks can share a `bubble` rect.

### B — lettering was not optically centred

`typeset()` set its anchor to the midpoint of the first and last filled row of the region's spans —
the *bounding box* — so a tail, a flat side or a joined neighbour dragged it off.  Three further
bugs surfaced while fixing it, none of them in the original hypothesis list:

- the widening loop asked for `n` rows, the greedy flow used fewer, and the block was then an
  `m`-line block pinned at an `n`-line base: it *migrated* half a line per iteration instead of
  gaining a line (2.0 em of drift on `tests/fixtures/1ja_tailed_balloon.png`);
- the ladder aimed the block's *box* while the metric scores its *ink centroid*, and a block whose
  last line is longest inks lower than its box centre;
- "the first line count that fits anywhere" beat "the count that fits centred".

`mask_spans` also picked each row's run at the **rect centre column** — the middle of the box the
layout cut, not of the balloon inside it.  On a mask holed by un-OCR'd Japanese that picks a run in
the wrong part of the region: 1ja b13 scored 2.95 em of anchor error, 0.05 em on the optical column.

Two bugs of the same family were found in the *measuring* code and are worth recording, because
each cost a round of chasing a phantom:

- `interior_centre` ran `cv2.distanceTransform` on an **un-padded** region.  A panel reaching the
  edge of the page has no zeros beyond it, so the transform keeps rising and the peak lands *on the
  border*: 4ja block 8 (`46,734,348,466` on a 1200-row page) scored its centre at `(241, 1199)`
  instead of `(171, 933)` and reported **10.26 em** of offset.  Both copies — the metric's and
  `render/anchor.py`'s — now pad by one pixel.
- `_bubble_region` inset the interior with a `MORPH_ELLIPSE` erosion.  Eroding by a *true* disc
  cannot move an inscribed-circle centre (the distance transform simply drops by the radius
  everywhere), but `MORPH_ELLIPSE` only approximates the disc, and 1ja b15's balloon has two nearly
  equal inscribed circles — 26.8 px and ~26.6 px, 28 px apart — so the approximation swapped them.
  `bubbles.inset()` now thresholds the distance transform instead, which is exact with respect to
  the same distance function the metric uses.  Anchor error on that block: **1.43 → 0.00 em**.

### C — furigana ink survived the erase

`erase.apply` opened with `del all_segments`, and its docstring said the extra lines were "accepted
for interface compatibility; glyph completion works from the ink alone".  Both were wrong.  Ruby is
the first thing the detector loses — 4ja `めぐみ` comes back at **0.258** against the 0.5 floor
while its kanji `惠` comes back at 0.561 — and `build_blocks`'s own docstring promises that superset
is "used by the eraser as evidence of glyphs the confident boxes missed".  Reproduced
deterministically: two annotated columns over artwork with ruby 12 px (0.46 glyph) away, handed in
only as low-confidence `all_segments` — **512 px of ruby ink survive per column**; promote the same
quads above the floor and 0 px remain.  The gap matters: at 2 px or less the ruby merges into the
glyph components and is erased incidentally, and from about 0.4 glyph outward it survives.

Found in passing and fixed, because it threatened the same acceptance clause from the other side:
`layout._mark_furigana` was flagging real *dialogue* columns as ruby and dropping them from the
translation — `すまんな`, `ところで`, `じゃねえ`, `こねーよ`, `なくても`, `あまり`, `これだ`,
`だろ` and five more.  The em separates the two cleanly (real ruby tops out at 0.71 of its parent,
the eaten dialogue starts at 0.85), and furigana annotates kanji, so a kana column beside a larger
kana one can never be ruby.  4ja now translates `悪いがあまり時間がない` as "I'm sorry, but we don't
have much time" instead of dropping "much".

The live app does **not** benefit yet: `RapidOCREngine.recognize` drops sub-threshold lines itself
and passes the same value to RapidOCR as `Global.text_score`, so the low lines are never detected.
`core/pipeline.py`'s TODO now states exactly that.  The harness passes `all_segments` already,
which is where the acceptance is measured.

## New metrics

Added to `demo/typeset_metrics.py`, per block and aggregated per page.  "Interior" is the bubble's
paper region as `typeset_reference.block_interior` finds it; all four are reported only for bubble
blocks, and free text scores `-` and is left out of the bubble means.

| metric | meaning |
|---|---|
| `containment` | glyph pixels outside the interior eroded by `INTERIOR_MARGIN_EM` (0.18 source ems, the same inset `render/layout._BUBBLE_MARGIN_EM` lays out with) over total glyph pixels; `outside_px` carries the count and the page row adds the worst block.  0 is the only passing value |
| `centre_offset_em` | distance from the block's ink centroid to `interior_centre` — the mean of the peak pixels of the interior's distance transform, i.e. the centre of the largest inscribed circle — in source ems, measured over the block's own part of a shared bubble |
| `leftover_px` | source ink still readable inside the interior eroded by `OUTLINE_BAND_EM` (0.25 ems, keeping the outline and its anti-aliased fringe out, since the eraser keeps those deliberately) after the erase pass |
| `blocks_per_bubble` | blocks lettered into this one's interior, after `render/bubbles` has cut a joined component into one interior per block.  1 is the only right answer |

`cv2.erode`'s default border treats outside-the-image as maximal, so `interior_core` does not erode
where a panel runs off the page.  That is deliberate: there is no outline there to keep air from.

`typeset_reference.bubble_ownership` grouped co-tenants by `TextBlock.bubble` **rect equality**,
which was right while `_split_shared_bubbles` gave co-tenants an identical rect.  Deleting it
silently broke the grouping, so a block's half-balloon centroid was being compared with the whole
balloon's centre; it now groups by interior overlap.  The change is a **no-op on `pre`** (identical
numbers, 11/31 at ≤ 0.25 em either way) and only bites after the removal — the behaviour a
correctness fix should have.

## Results: `pre` → `final`

| page | blocks | IoU | recall | prec | art kept | SSIM | MAE | cap ratio | lines same | overflows | collisions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| before | 8 | 0.84 → 0.84 | 0.96 → 0.96 | 0.87 → 0.87 | 0.72 → 0.69 | 0.647 → 0.646 | 12.8 → 12.7 | 0.78 → 0.74 | 2/8 → 2/8 | 0 → 0 | 0 → 0 |
| 1ja | 16 | 0.94 → **1.00** | 0.94 → **1.00** | 1.00 → 1.00 | 1.00 → – | 0.663 → **0.695** | 7.3 → **5.5** | 1.14 → 1.04 | 6/16 → 5/15 | 1 → **0** | 1 → 2 |
| 2ja | 13 | 0.76 → **0.96** | 0.77 → **0.96** | 1.00 → 1.00 | 0.99 → 1.00 | 0.783 → **0.868** | 8.5 → **2.4** | 1.18 → 1.04 | 6/13 → 4/13 | 0 → 1 | 0 → 0 |
| 3jp | 17 | 0.78 → **0.94** | 0.80 → **0.94** | 0.95 → 0.99 | 0.87 → **0.98** | 0.788 → **0.851** | 8.4 → **3.3** | 0.99 → 0.95 | 9/17 → 9/17 | 1 → **0** | 2 → 4 |
| 4ja | 12 | 0.78 → **0.89** | 0.88 → 0.89 | 0.90 → **1.00** | 0.61 → **0.98** | 0.722 → **0.837** | 9.1 → **5.4** | 1.11 → 1.05 | 3/12 → **5/12** | 1 → **0** | 2 → **1** |
| **all** | 66 | 0.82 → **0.94** | 0.86 → **0.95** | 0.95 → **0.98** | 0.83 → **0.87** | 0.729 → **0.790** | 8.9 → **5.2** | 1.06 → 0.98 | 26/66 → 25/65 | 3 → **1** | 5 → 7 |

Collisions count *blocks*, not area: collision area is **2656 px → 279 px**, a 90 % reduction
(3jp b3 274 → 5, 4ja b10 1702 → 0, b11 618 → 140, 1ja b8 61 → 0), against a handful of new ones of
1–98 px.  `lines same` is out of 65 rather than 66 because 1ja's stray `……` is now merged into the
block it belongs to — the metric reports that as a ground-truth block we produced nothing for, which
is correct and is the sort of thing the pairing fix below exists to surface.

### Geometry

| page | blocks in a balloon | containment mean/worst | centre_offset mean/worst (em) | leftover px (blocks) | blocks_per_bubble max (sharing) |
|---|---|---|---|---|---|
| before | 2 → **5** | 0.000/0.000 → 0.000/0.000 | 1.38/2.71 → 2.56/5.08 | 0 → 0 | 1 (0) → 1 (0) |
| 1ja | 7 → **14** | 0.004/0.026 → **0.000/0.004** | 1.10/1.67 → **0.40**/1.83 | 0 → 4 | 1 (0) → 1 (0) |
| 2ja | 6 → **13** | 0.000/0.003 → 0.084/1.000 | 0.57/2.01 → **0.29/1.64** | 51 → 146 | 1 (0) → 1 (0) |
| 3jp | 10 → **15** | 0.007/0.067 → **0.000/0.003** | 0.56/2.04 → **0.26/1.47** | 64 → **3** | 1 (0) → 1 (0) |
| 4ja | 6 → **9** | 0.033/0.197 → **0.001/0.003** | 1.21/3.52 → **0.22/0.77** | 0 → 20 | 1 (0) → 1 (0) |
| **all** | **31 → 52** | 0.010/0.197 → 0.020/1.000 | 0.86/3.52 → **0.38**/5.08 | 115 (3) → 173 (7) | 1 (0) → 1 (0) |

`centre_offset_em` at or below 0.25 em: **11/31 (35 %) → 36/52 (69 %)**; above 0.6 em: **15 → 7**.
Total glyph pixels outside an interior, across all 66 blocks: **28**.

Containment and leftover rise in the means, and both are composition: 21 more blocks are lettered
inside a balloon and therefore judged at all.  `blocks_per_bubble` reads 1 on both sides because
defect A manifests as *failing to detect a balloon*, so the broken blocks scored `-` rather than
sharing an interior — the number that shows the fix is the 31 → 52 detection count.

`score_page` used to pair our blocks to the ground truth **by position**.  The moment the layout
gains or loses a block, every block after it is then scored against its neighbour's record, silently
— 3jp gained one at index 14 and its last three rows were judged at source-box overlaps of 0.25,
0.00 and 0.00.  Pairing is now by source-box IoU (`pair_blocks`, `PAIR_MIN_IOU = 0.3`), and blocks
matching no record, or records we rendered nothing for, are logged and counted rather than dropped.

### LPIPS (lower is better; free-text blocks, AlexNet)

| tag | before | 1ja | 2ja | 3jp | 4ja | all art |
|---|---|---|---|---|---|---|
| pre | 0.404 | 0.212 | 0.126 | 0.135 | 0.171 | 0.212 |
| **final** | 0.406 | **0.090** | **0.110** | **0.111** | **0.138** | **0.175** |

Scored per block over the blocks both tags scored, so a block moving from free text to a balloon
cannot masquerade as a gain: **every block that moved, improved; none regressed at any magnitude.**
Largest movers — 1ja b1 0.197 → 0.020, b13 0.219 → 0.056, b14 0.174 → 0.036, b0 0.172 → 0.040,
b5 0.322 → 0.198; 4ja b1 0.158 → 0.047, b9 0.246 → 0.176; 3jp b14 0.144 → 0.066.
`Examples/before.jpg` is flat at 0.404 → 0.406, inside the ±0.07-per-block noise floor.  (The 0.380
figure in `docs/perf/2026-09-11-quality-renderer.md` is the *quality-renderer* run; these are the
heuristic eraser, whose comparable figure is 0.404.)

## Acceptance

| clause | result |
|---|---|
| A: 1ja 13, 1ja 14, 2ja 11, 4ja 4 letter one block per balloon | **met for three**; 4ja 4 is not a balloon — see below |
| A: no block's area exceeds a single balloon's interior | **met** — `blocks_per_bubble` max 1, 0 blocks sharing, on all five pages |
| A: no regression in blocks correctly grouped elsewhere | **met** — 31 → 52 blocks in a balloon, six false positives rejected by `_BUBBLE_MIN_WALLED` |
| B: zero blocks with `containment > 0` | **not met** — 9 blocks, 28 px total; see below |
| B: `centre_offset_em` ≤ 0.25 em for ≥ 90 % of blocks, none above 0.6 | **not met** — 69 % and 7 above; see below |
| B: 4ja 9, 4ja 10/11, 1ja 15 visibly fixed | 4ja 9 (1.64 → 0.11) and 4ja 10 (0.18) fixed; 4ja 11 at 0.77 and 1ja 15 at 1.83 — see below |
| C: `leftover_ink` zero on all five pages | **not met** — 173 px in 7 blocks, every pixel outline or art; see below |
| C: furigana still absent from the translated output | **met**, asserted in `tests/test_furigana_erase.py` |
| no metric regresses beyond ±0.07 LPIPS per block | **met** — every block that moved, improved; none regressed at any magnitude |
| reviewer judges every page at least as good as `base` | **met** — see below |
| full suites green, pyflakes clean | **met** — app 938 passed / 1 skipped, sidecar 180 passed / 5 skipped |

### Where the three numeric targets were not reached, and why

**`containment > 0` on 9 blocks, 28 px in total.**  Eight are **1–4 pixels** out of 280–2072 ink
pixels: antialiasing fringe on the boundary of a mask eroded by `round(0.18 * em)`, which is 3–5 px
and quantised to whole pixels, against a reference scan whose outline edge differs from ours by
about a pixel anyway.  That is below the measurement's own resolution.  The ninth is 2ja b12, where
`containment = 11/11 = 1.000` — the entire block is eleven ink pixels, an OCR artefact rather than
lettering.  The two substantive cases at the start of the session are both gone: 1ja b13's 96 px
(lettering flowing through holes the un-OCR'd Japanese punches in the paper map) closed when
`mask_spans` moved to the balloon's optical column.

**`centre_offset_em` ≤ 0.25 em on 69 % of blocks, 7 above 0.6.**  Three separate causes, none of
them a placement the renderer could improve:

- `before` b1 at 5.08 em is a **two-lobed caption pair**, and `interior_centre` returns the centre
  of one lobe, so a block correctly filling both can never score near it.  The reviewer compared the
  crops and judged this render clearly the better one: `pre` ran one 15-line slab across the border
  between the two boxes and hung `AMPLIFICA-TION.` onto the artwork, while this one splits at the
  `… …` and sets one sentence per box at cap height 0.4699 — exactly the reference letterer's.
- A block that **fills** its balloon has its ink centroid pinned near the region's area centroid,
  while the metric targets the inscribed-circle centre; on 1ja b15's oval those are 2.01 em apart,
  and our ink is 0.19 em from the area centroid.  3jp's merged block is the same effect.
- Five blocks are **entirely** anchor error: we place the block exactly where our own region says
  the middle is, and our interior and `TR.block_interior` are two independent derivations of "the
  balloon" that disagree by 0.17–0.28 em on a broad flat ridge.
- 4ja b11 at 0.77 em is constrained by grouping, not centring: the OCR read one utterance as two
  blocks 60 px apart, so each owns half a balloon and b11 gets 45 px of width for its sentence.

Three interventions were implemented, measured and **rejected** rather than shipped, and are
recorded so they are not retried blind: hole-filling the mask for the anchor (mean 0.131 → 0.345 em,
> 0.6 em count 3 → 9) or for the flow region (5583 glyph px outside the interior across 6 blocks);
dilating the mask to undo `_BUBBLE_MARGIN_EM` (k = 2/3/4/6 px → 0.197 / 0.270 / 0.501 / 0.573
against 0.131); and painting a block's own source footprint into its mask, which buys 16 % of type
on one 3jp block and costs 0.08 em of mean centre offset on all four manga pages plus a rewrite of
`before` b1.

**`leftover_ink` 173 px in 7 blocks.**  Every remaining pixel was identified from the metric's own
leftover mask and checked against greyscale crops: 2ja's 86 px is a 43×2 caption-box rule and its
51 px the balloon's black tail wedge, 4ja's 20 px is balloon outline caught inside the 0.25-em
erosion, 1ja's 4 px is a 4×1 sliver.  None of it is source glyph ink, and none of it is furigana —
which is what the metric was added to catch.  The 1745 px that appeared mid-session (free text
misread as a balloon, the eraser then painting flat paper over a wooden scaffold) is gone.

**4ja 4.**  It is lettered correctly, but it is **not a balloon** — it is a white ID card, and the
`_BUBBLE_MIN_WALLED` gate rejects it at 0.03–0.48 against real balloons' 0.51–1.00.  Treating it as
one cost +0.092 LPIPS, past the noise floor, and painted flat paper over artwork.  So the clause is
satisfied in substance (one block, one region, no merge with its neighbour) but not by the route the
brief anticipated.

### Reviewer's page-level judgement

Better on all five, with one regression named and since fixed.  `before`: the caption pair split
correctly, b7 from 7 narrow lines to 5 wide.  `1ja`: four balloons whose English used to print
outside them, clipped at panel edges over leftover furigana, now contained and clean.  `2ja`: b12
moved from hanging outside its balloon to inside it.  `4ja`: "much better" — `pre` printed two
blocks on top of each other in the big balloon as an illegible tangle while the neighbouring balloon
sat empty with `説明して` still in it; b1's balloon outline is also restored.  `3jp` was judged
**worse** at the time of review — one balloon carrying two unrelated speeches — and that is the
`_compatible` defect below, now fixed.

### Defects the review caught that no metric did

- **One balloon, two speeches (3jp).**  `_compatible` gated size on the **OCR box width**, where a
  column's box is inflated or deflated by detection: `迷っても` (box 46.0, em 22.0) against
  `なくても` (box 25.0, em 18.8) failed `46.0 <= 1.7 * 25.0` and the two halves of one utterance
  became two blocks in one balloon.  On the em it passes at `22.0 <= 1.7 * 18.8`.  Gating on the em
  *alone* breaks 2ja in the mirror image (`って` mid-sentence at an em ratio of 2.20), so the rule
  shipped is **split only when both measures disagree**.  This also merged 1ja's stray `……` and
  removed the last source-box overlap in the contract sweep.
- **Lost speech in the erase evidence.**  `_evidence` gated candidate size on `min(box.w, box.h)`,
  which for a vertical Japanese column is its ~1-em width however long the column runs — so a
  full-size low-confidence *dialogue* column was absorbed as ruby, erased, and never translated.
  Measured over the five pages, three live lines were being taken this way (`っーか` at an em ratio
  of 1.05, `制` 0.98, `n0` 0.89).  The guard now compares `glyph_em` at 0.78, which lands in the
  widest gap of that distribution (0.63 → 0.89); 4ja `めぐみ` at 0.32 is still correctly absorbed.
- **A caption centred in a band it occupies a fraction of.**  1ja's REPORT / date / school strip is
  660×152 while its Japanese occupies 86 rows of that, so centring in the interior dropped the
  caption ~17 px below its own printed rule.  The bubble branch now anchors a **horizontal** block
  on its source line, which is the split the free-text branch fourteen lines below already makes
  (`must_cover = not style.vertical`).  It fires on the only three horizontal bubble blocks in the
  corpus and is pixel-identical on the other four pages.
- **A quadratic cluster window.**  `_clusters` merges transitively, so a screen region whose paper
  is one connected component put every block in one cluster and ran one distance transform per block
  over the whole component: 1716 ms at 3840x2160 with 24 blocks.  Capped at `_CLUSTER_MAX_SPREAD`,
  with a page-wide `_uncross()` sweep restoring the disjointness guarantee the cap drops — 90 ms,
  and a verified no-op on all five harness pages.

### Timing

`build_blocks`'s `group` stage on the largest page (`before.jpg`, 1096x1600, 10 blocks): **42.6 →
108 ms**, of which `_interiors` is ~84 ms, dominated by one 864x674 cluster.  On the 764x1200 pages
it is 36–81 ms.  Taken down from an initial 220 ms by bounding each block's search to the window its
own gate limits allow, clustering only when one block's text reaches into another's window, making
solidity lazy, gating `distanceTransformWithLabels` behind a cheap `connectedComponents`, and
caching seals by neck width.  A single-block component that already reads as a balloon — most
balloons on most pages — takes the fast path and costs nothing extra.  The `erase` stage is flat to
slightly faster (before 53.0 → 45.7 ms, 1ja 25.1 → 20.3, 2ja 13.5 → 15.6, 3jp 21.7 → 21.3, 4ja
21.9 → 20.7).

**The five harness pages are not the worst case, and an earlier draft of this section claimed a
cadence the code does not have.**  For the record, because the wrong version was written down
first: the gate is `period = 1.0 / max(self._cfg.refresh_hz, 0.5)` with `refresh_hz = 10.0`
(`core/pipeline.py:326`, `config/settings.py:253`) — **a 100 ms tick**, not a 120 ms debounce.
`debounce_ms = 120` is read only at `pipeline.py:655` inside `_settle`, which is reached only on the
storm branch (`pipeline.py:389`, `_STORM_FRACTION = 0.4`), and `pipeline.py:392` then sets
`dirty = [Rect(0, 0, frame.width, frame.height)]` — so each debounce fire *guarantees* a full-frame
pass rather than preventing one.  A second full-frame hatch sits at `pipeline.py:698`.  And
`build_blocks` is handed the whole frame regardless of the dirty rect (`pipeline.py:760`), so
`cvtColor` + `connectedComponentsWithStats` cost 4.9 ms at harness size and 17.6 ms at 4K on every
pass.  When a pass overruns, `remaining` goes negative at `pipeline.py:328` and the worker does not
sleep.  There is no re-entrancy and the UI is never blocked; that part of the design is sound.

So the budget is the 100 ms tick, and `_interiors` has to live inside it.  The cost is also not
linear in page size: `_clusters` merges transitively, so a screen region whose paper is one
connected component — a white page, a document viewer, an open manga background — puts every block
in one cluster and `window = _union(...)` re-expands to the whole component, at one distance
transform per block:

| frame | blocks | before the cap | after |
|---|---|---|---|
| 1096x1600 | 2 | 65 ms | **10 ms** |
| 1096x1600 | 24 | 370 ms | **86 ms** |
| 3840x2160 | 2 | 293 ms | **20 ms** |
| 3840x2160 | 24 | **1716 ms** | **90 ms** |

The harness pages hide this because their 8–17 blocks are spread over several components.
`_CLUSTER_MAX_SPREAD = 4.0`: a cluster whose union window exceeds four times its biggest member's
own window is not one room, so its blocks are searched alone.  That drops the in-cluster
disjointness guarantee, so `_uncross()` sweeps the page afterwards and drops the smaller of any two
interiors that share a pixel — a verified no-op on all five harness pages, so the cap costs nothing
in quality.  `_closest` also shares one labelled distance transform across a group's sources when
their boxes are pairwise disjoint (the CCOMP labels of disjoint seeds *are* the nearest-footprint
map), falling back to the per-source loop where two boxes overlap and sharing a label would change
ownership.

Measured `group` per page after all of it: `before.jpg` 126–138 ms, 1ja 39–41, 2ja 48, 3jp 72,
4ja 39 (±10 % run to run on this machine).  `before.jpg` is the one page still over the 100 ms tick,
and the pipeline-side crop below is what fixes it.

Still open, and the next thing to do here: crop the image to the dirty union before calling
`build_blocks`, so the per-pass cost follows the region that actually changed rather than the whole
frame.  That is a coordinate-frame change and was deliberately not taken in this session.
