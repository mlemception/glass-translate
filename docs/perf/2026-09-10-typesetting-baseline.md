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
