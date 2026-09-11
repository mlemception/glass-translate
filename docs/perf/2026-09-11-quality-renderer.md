# Quality renderer — before / after on the five reference pages, 2026-09-11

Companion to [`2026-09-10-typesetting-baseline.md`](2026-09-10-typesetting-baseline.md) (the
eraser before this change, metric definitions, failure catalogue) and
[`2026-09-11-quality-renderer-timings.md`](2026-09-11-quality-renderer-timings.md) (stage timings,
cold start, VRAM).  Everything below was measured on the target machine (RTX 4080 16 GB, Windows 11)
with `demo/typeset_dev.py --batch --ref-text --regions all`, `demo/typeset_metrics.py --batch` and
`tools/lpips_score.py --batch` (sidecar venv).  Render outputs: `demo/output/dev/<stem>/<tag>_*`;
tables: `demo/output/metrics/<stem>_<tag>.md`, `<tag>_summary.md`, `<stem>_<tag>_lpips.json`.

Tags:

| tag | what it is |
|---|---|
| `base` | the eraser before this change (baseline document) |
| `redraw` | bubble redraw in `render/erase.py` (step 2), quick fill for everything else |
| `q_lama` | redraw + sidecar with LaMa only (`sdxl=false`) |
| `q_s40` | redraw + sidecar LaMa → SDXL img2img, strength 0.4, gate at 3 % ring ink measured with the glyph fringe, whole-panel windows |
| `final` | as shipped: defaults of `renderer/PROTOCOL.md` (LaMa → SDXL, strength 0.4, 24 steps, ControlNet 0.8, guidance 4.0, seed 0, feather 2), gate at 5 % ring ink beyond a 2-px fringe, per-cluster context windows |

## 1. Erase and lettering metrics (`<tag>_summary.md`)

`base` and `redraw` differ only on 4ja (the old eraser already left these bubble interiors clean;
see the baseline document's re-measure note).  Lettering columns (cap ratio, inset, centring,
lines, overflows, collisions) are identical across all tags — this change does not touch placement.

| tag | page | IoU | recall | prec | art kept | SSIM | MAE |
|---|---|---|---|---|---|---|---|
| base | before | 0.84 | 0.96 | 0.87 | 0.72 | 0.647 | 12.8 |
| final | before | 0.86 | 0.95 | 0.89 | 0.74 | 0.665 | 13.0 |
| base | 1ja | 0.94 | 0.94 | 1.00 | 1.00 | 0.663 | 7.3 |
| final | 1ja | 0.89 | 0.89 | 1.00 | 1.00 | 0.624 | 9.4 |
| base | 2ja | 0.76 | 0.77 | 1.00 | 0.99 | 0.783 | 8.5 |
| final | 2ja | 0.76 | 0.76 | 1.00 | 0.99 | 0.770 | 9.3 |
| base | 3jp | 0.80 | 0.82 | 0.95 | 0.87 | 0.801 | 8.1 |
| redraw | 3jp | 0.78 | 0.80 | 0.95 | 0.87 | 0.788 | 8.4 |
| final | 3jp | 0.78 | 0.80 | 0.97 | 0.88 | 0.788 | 8.5 |
| base | 4ja | 0.77 | 0.79 | 0.97 | 0.81 | 0.774 | 8.9 |
| redraw | 4ja | 0.78 | 0.88 | 0.90 | 0.61 | 0.722 | 9.1 |
| final | 4ja | 0.78 | 0.87 | 0.90 | 0.60 | 0.701 | 9.7 |
| base | **all** | 0.82 | 0.85 | 0.97 | 0.87 | 0.742 | 8.7 |
| redraw | **all** | 0.82 | 0.86 | 0.95 | 0.83 | 0.729 | 8.9 |
| final | **all** | 0.81 | 0.84 | 0.96 | 0.84 | 0.715 | 9.7 |

Reading: SSIM and MAE inside the erased neighbourhood do **not** favour the generative fill — they
reward flat paper, and a plausible continuation of hatching never matches the English page's
hatching pixel for pixel.  1ja's recall drop (0.94 → 0.89) is real and explained below (strokes
invented inside an undetected bubble).  4ja's recall gain (0.79 → 0.88) is the bubble redraw.
3jp's small drop (0.82 → 0.80) is the price of the redraw's paper fill stopping 1.5 glyphs
beyond the text boxes (`BUBBLE_ZONE_EM`): strays the OCR never boxed farther out now stay,
which is what keeps a character drawn across a balloon intact (4ja b9 — the whole-interior fill
had erased its head; `art kept` never saw it because the head lies outside the judged region).

## 2. LPIPS on the free-text blocks (lower is better)

`tools/lpips_score.py`: AlexNet LPIPS of each `art` block's window (+16 px) against the aligned
English page with the English lettering painted out on both sides.

| tag | before | 1ja | 2ja | 3jp | 4ja | all art blocks |
|---|---|---|---|---|---|---|
| base | 0.404 | 0.212 | 0.126 | 0.135 | 0.192 | 0.216 |
| redraw | 0.404 | 0.212 | 0.126 | 0.135 | 0.171 | 0.212 |
| q_lama | 0.375 | 0.231 | 0.130 | 0.137 | 0.167 | 0.209 |
| q_s40 | 0.381 | 0.224 | 0.121 | 0.125 | 0.166 | 0.205 |
| **final** | **0.380** | 0.226 | 0.138 | 0.135 | 0.175 | 0.213 |

(`q_lama` and `q_s40` were rendered before the `BUBBLE_ZONE_EM` limit; on 4ja that limit costs
~0.01 of page-mean LPIPS and keeps the chibi's head.)

Per block, where anything moved (quick fill → final):

| page | block | quick fill | final | what happened |
|---|---|---|---|---|
| before | b2 そのお能の輝きとは (scaffolding) | 0.363 | **0.301** | the quick fill's white blob is replaced by continued beams and rails — the change this feature exists for |
| before | b3 chapter title over art | 0.311 | **0.235** | same |
| before | b4, b7, b8 | 0.254 / 0.340 / 0.508 | 0.239 / 0.330 / 0.501 | small gains |
| 1ja | b13 駄目だから | 0.219 | 0.287 | **fake kana** generated inside a bubble the layout did not detect (see § 4) |
| 1ja | b14 ますよ…… | 0.174 | 0.189 | same bubble, milder |
| 2ja | b4, b11 | 0.132 / 0.124 | 0.151 / 0.192 | the mask reaches into a face below an undetected bubble (see § 4); the fill regenerates the face |
| 3jp, 2ja | b6, b14, b2 (paper) | 0.227 / 0.144 / 0.123 | unchanged | these were sent under `q_s40` (0.195 / 0.113 / 0.104: the fill cleaned the quick fill's specks) and are now gated out as paper |
| 4ja | b3 | 0.117 | 0.145 | textured backdrop, the fill is busier than the reference |

Noise floor: a block's LPIPS moves by up to ±0.07 when only its context window shifts (2ja b11 under
`q_s40` vs `final`, identical parameters and seed), so differences below ~0.01 in the page means
are not evidence.  The two robust results are `before.jpg` (0.404 → 0.380, and the gains sit on
the two blocks that are visibly broken by the quick fill) and "SDXL after LaMa beats LaMa alone on
the art blocks" (0.375 vs 0.381 on `before` is within noise; 1ja/2ja/3jp favour SDXL).

Parameter sweep (`demo/output/sweep/sweep2.log`, tags `q_s30 q_s50 q_cn06 q_cn10 q_g1 q_st16`):
indistinguishable from `q_s40` on every page within the noise floor, so the protocol's defaults stay
as specified in the plan (strength 0.4, 24 steps, ControlNet 0.8, guidance 4.0).  `guidance 1.0`
would halve the SDXL time at 1024 with no measurable quality change and is the first lever if a
budget is ever missed.

## 3. Timing (warm, per page, `render_final.log`)

Jobs are one per text cluster after gating; every window is ≤ 1024 px and rendered at native
resolution.

| page | jobs | window sizes | LaMa | SDXL | total |
|---|---|---|---|---|---|
| before | 3 | 1024², 1024², 1024×512 | 0.29 / 0.09 / 0.05 s | 3.5 / 3.2 / 1.7 s | **9.4 s** |
| 1ja | 2 | 512², 512² | 0.03 / 0.04 s | 1.1 / 1.0 s | 2.3 s |
| 2ja | 1 | 764×1024 | 0.08 s | 2.3 s | 2.6 s |
| 3jp | 1 | 764×1024 | 0.07 s | 2.3 s | 2.6 s |
| 4ja | 2 | 764×1024, 512² | 0.07 / 0.03 s | 2.4 / 1.0 s | 3.8 s |

Targets: LaMa < 0.2 s per 1024 panel **met** (0.05 – 0.29 s; the 0.29 s is the first job of the
process), SDXL < 3.5 s **met** (3.2 – 3.5 s at 1024²), per page < 10 s **met** (9.4 s worst),
VRAM < 12 GB **met** (11.6 GB reserved after warm-up, 9.2 GB allocated).  Cold start: `READY`
after 1.7 s, models resident after ~19 s (~90 s when the page cache is cold: the 6.9 GB checkpoint
digest is verified on every start), warm after 90 – 270 s (compile 26 s + allocator pool 21 s +
SDXL warm-up 22 s, plus the load; 242 s in the final run).  `QualityClient.wait_warm` holds the
first job until `/health` says warm, so the request timeout covers the job and not the warm-up;
the app renders the quick fill meanwhile and swaps patches when jobs land.

Two stalls found and fixed on the way (details in the timings document and the plan):

* the sidecar answered nothing during its warm-up — the stdin watcher's blocking pipe read stopped
  every new thread on Windows while torch loaded; it polls now;
* the first LaMa run of a page took 6 – 16 s — a fresh `cudaMalloc` with the card near full under
  WDDM; the warm-up now reserves the allocator pool at 1024².

## 4. What limits the result: mask accuracy and bubble detection

The generative fill only ever touches pixels inside the eraser's glyph mask (+2 px, feathered
2 px); everything else is byte-identical (`renderer/tests/test_compositing.py`,
`tests/test_quality_client.py`).  So the fill can only be as good as the mask, and the failures
above are mask failures:

1. **Undetected shared bubbles.**  `layout._make_block` tests the paper component around *one*
   block against that block's own size (`_BUBBLE_MAX_DIM_RATIO`, `_BUBBLE_MAX_AREA_RATIO`).  A
   bubble holding three or four columns fails for each column, the columns become "free text on
   paper", and the open-text sweep of `erase.py` then follows ink beyond the (invisible) bubble:
   on 2ja b11 the mask runs out of the bubble into the face below it — the **old eraser does the
   same** (`base_erasecmp_b11_art.jpg`), the metrics do not see it because the face lies outside
   the judged region, and the generative fill just regenerates the face instead of smearing it
   white.  On 1ja b13/b14 the fill, conditioned on the neighbouring columns, invents kana.
   Fix: evaluate the bubble tests against the union of the blocks inside one paper component
   (then `_split_shared_bubbles` already handles ownership).  This is the top next step.
2. **Recall.**  Precision is 0.96 – 1.00 on every page while recall is 0.76 – 0.95: what stays is
   ink the mask never covered (SFX the OCR did not box, glyphs joined to outlines), not fill
   quality.  Improving the fill cannot move recall; improving the mask can.
3. **Metric blind spot.**  Art damaged *outside* a block's judged region is invisible to
   `art_kept`.  Next: score kept art in a 1-em ring around each block's clean rect as well.

## 5. Gating and windows (why `final` differs from `q_s40`)

* `art_under_text` measures ink in a 4-px ring around the glyph mask of the quick-filled patch.
  The ring used to start at the mask edge and counted the glyphs' anti-aliased fringe as ink: plain
  paper read as 15.6 % "art" on 1ja 記錄.  It now starts 2 px out (4.1 % there) and the threshold
  is 5 %.  On the five pages this drops one harmful block (記錄, +0.011) and two neutral ones, and
  also the three paper blocks whose only gain was the fill cleaning the quick fill's specks.
  The ring cannot tell "text next to other text" from "text over art" (駄目だから reads 8.9 %, the
  scaffolding block 16.5 %) — that case belongs to bubble detection, not to the gate.
* A page-sized "panel" (no borders found, `before.jpg`) used to go to the sidecar whole, i.e.
  downscaled to the 1024 bucket and back.  `panel_jobs` now splits the panel's members into
  clusters that fit one 512 – 1024 px window each (`_cluster_members`), so every window is
  rendered at native resolution; `before.jpg` became three jobs instead of one.
