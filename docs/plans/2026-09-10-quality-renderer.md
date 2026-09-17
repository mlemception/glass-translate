# Quality renderer: bubble redraw + generative free-text fill — 2026-09-10

Staged plan for the request "replace the heuristic text erasure with a generative
quality renderer".  Tier: **large** (new sidecar package with its own venv and external deps, a
localhost protocol, render-pipeline changes in both renderers, an Engines-page setting, model
downloads, and a new measurement harness).  GATE 1 was auto-approved in the request; the only
stop is GATE 2 (commit).  Status: **in progress** (see "Status" at the end).

## Intake (restated)

Make a translated manga page look as if the Japanese was never there, judged against the
professionally lettered English pages in `more_comparisons/` (4 ja/eng pairs, ja 1200x764, eng
1600x1067) plus `Examples/before.jpg` / `after.webp`.  Target: one RTX 4080 (16 GB), Windows 11.

1. Bubbles and caption boxes are **redrawn** (paper fill clipped to the outline, outline kept with
   its own thickness and tails) in the main app (`render/erase.py`), for the PIL renderer and the
   Qt overlay alike.
2. Free text over artwork gets a two-stage generative fill in a **sidecar process** (own venv,
   torch + CUDA): LaMa for structure, then SDXL img2img (denoise 0.3–0.5 of 20–30 steps) with a
   line-art ControlNet, whole panel as context at 1024 px, fp16.  Only the masked region is
   composited back (feathered); every pixel outside the mask stays byte-identical.
3. The mask is the existing eraser's glyph classification (+ completion), dilated 1–2 px.
4. Panels come from `render/place.py`'s panel-border logic; one job per panel with free text on art.

The main app stays torch-free and buildable as the single PyInstaller exe.  The sidecar speaks
JSON over HTTP on 127.0.0.1 with a per-session token and is never exposed on the network.  The app
works without it (the quick fill renders in the normal pass; a sidecar result later swaps the
block's `clean_patch` through the existing patch cache).

## Decisions and assumptions

- **Harness OCR stays rapidocr** (`RapidOCREngine`, `min_confidence=0`) as today, so the
  reference-text keys stay stable; the app's manga-ocr chain uses the same detector boxes.
- **Perceptual score = SSIM** implemented in numpy/OpenCV inside `demo/typeset_metrics.py` (no
  scikit-image, no torch in the main venv).  LPIPS is optional and only computed when the metrics
  script is run with the sidecar venv (`--lpips`), never a hard dependency.
- **Ground truth** is derived once per pair by `demo/typeset_reference.py` (ECC homography, ORB +
  RANSAC fallback, hand-picked crop pairs as a last resort) into `demo/reference/<stem>/`.
- **Reference text** files are `demo/reference_text_<stem>.json`; the old
  `demo/reference_text.json` moves to `reference_text_before.json` (git mv) and the loader keeps
  the old name as a fallback.
- **Caches** move to `demo/cache/<stem>/`; the tracked flat files for `before` are moved there.
- **Bubble redraw**: interior = the paper component of the bubble with the text painted over
  (as `layout.py` finds it), filled with `style.bg` and clipped to a 1-px dilation of the outline
  component; outline pixels (and their anti-aliased fringe) are copied from the source, so the
  outline keeps its own thickness and tails.  Stray glyphs outside the OCR boxes are still collected
  by the column sweep.  Glyphs physically joined to the outline outside any OCR box remain a known
  limitation (documented, not fixed in this pass).
- **Sidecar package** `renderer/` (package `glassrenderer`), own venv `renderer/.venv`
  (`renderer/install.bat`), models under `%LOCALAPPDATA%\GlassTranslate\models\quality\`
  (dev: `<project>/models/quality/`), downloaded through the same Engines-page progress row as
  manga-ocr.  Licences are recorded in `renderer/MODELS.md`; nothing GPL is imported by the app or
  the sidecar package.
- **Protocol**: `GET /health`, `POST /inpaint` (JSON: base64 PNG image + mask + params → base64
  PNG result + timings), `POST /shutdown`; header `X-GT-Token` (constant-time compare), bind
  `127.0.0.1` on an OS-chosen port announced on stdout as `READY <port>`.
- **App-side job flow**: `render/quality.py` — `QualityClient` (launch, health, request thread) +
  `QualityScheduler` (one job per panel with free text over art, keyed by a content hash so an
  unchanged panel is never re-sent; stale results dropped).  Results are queued and applied by the
  pipeline at the start of its next pass (no cross-thread mutation of styles): the block's
  `clean_patch` is replaced, `clean_patch_serial` incremented, and the live list re-emitted; the
  overlay re-converts only patches whose serial changed.
- **No best-of-N, no judge model, no learned stroke segmenter** (out of scope, per request).

## Task list (thin vertical slices)

| # | Slice | Files | Tests |
|---|---|---|---|
| 1a | Generalise the harness: any image, per-stem caches, per-stem reference text, auto reference image, auto per-block regions, `--batch` | `demo/typeset_dev.py`, `demo/cache/<stem>/`, `demo/reference_text_before.json` | `tests/test_typeset_dev.py` (stem/cache/reference resolution, region generation) |
| 1b | Alignment + ground truth per pair (erase mask, kept art, per-bubble inset / cap height / lines / centring, eng OCR bootstrap of the reference text) | `demo/typeset_reference.py`, `demo/reference/<stem>/` | `tests/test_typeset_reference.py` (synthetic pair: homography recovered, mask derived) |
| 1c | Metrics: erase IoU, art-kept ratio, SSIM in erased regions, bubble inset / size ratio / centring / overflow / collisions; per page and per block tables | `demo/typeset_metrics.py` | `tests/test_typeset_metrics.py` (synthetic renders with known scores) |
| 1d | Baseline of the current eraser on all five pages + failure catalogue | `docs/perf/2026-09-10-typesetting-baseline.md` | — |
| 2 | Bubble redraw (tests first: round, oval, joined, jagged, thought, tailed, dark paper) | `render/erase.py`, `tests/test_erase.py` | red → green |
| 3a | Model / licence research | `renderer/MODELS.md` | — |
| 3b | Sidecar package, protocol, fake-model mode | `renderer/glassrenderer/{server,protocol,fake,compositing}.py`, `renderer/requirements.txt`, `renderer/install.bat` | `renderer/tests/` + `tests/test_quality_protocol.py` (fake mode, no GPU) |
| 3c | Real stages: LaMa, SDXL img2img + line-art ControlNet, compositing; one opt-in GPU integration test | `renderer/glassrenderer/stages/*.py` | `renderer/tests/test_gpu_integration.py` (`GT_GPU_TESTS=1`) |
| 4 | App integration: launch + health, per-panel jobs, patch swap (PIL + overlay), Engines setting, model download, graceful fallback | `render/quality.py`, `core/pipeline.py`, `ui/overlay.py`, `config/settings.py`, `ui/bridge_fields.py`, `ui/control_bridge.py`, `ui/download_workers.py`, `ui/qml/pages/EnginesPage.qml`, `packaging/GlassTranslate.spec` | `tests/test_quality_*.py`, bridge / qml tests |
| 5 | Tune denoise, steps, context, ControlNet weight against the metrics on the four pairs; `before.jpg` free text must not regress | `renderer/glassrenderer/params.py`, `docs/perf/2026-09-10-quality-renderer.md` | — |
| 6 | Verify: full suite, harness + metrics on five pages before/after, Qt/PIL parity, timing table, exe build + smoke with the sidecar absent | — | — |
| 7 | Reviews (code, python, security, typesetting quality; one verifier per CRITICAL/HIGH) then fixes | — | — |
| 8 | Docs: this plan, perf tables, README, demo docstrings | — | — |
| 9 | GATE 2 | — | — |

## Timing targets (warm, fp16 CUDA, per panel at 1024 px)

LaMa < 0.2 s, SDXL pass < 3.5 s, per page < 10 s, cold start < 20 s, VRAM < 12 GB.  TensorRT
compilation of the UNet / ControlNet is an optional final slice if these are missed.

## Status

- **1a–1d done** (2026-09-10/11): `demo/typeset_dev.py` generalised (per-stem caches under
  `demo/cache/<stem>/`, `reference_text_<stem>.json`, automatic reference image and per-block
  regions, `--batch`, `<tag>_blocks.json`); `demo/typeset_reference.py` (ORB + ECC alignment,
  scores 0.78–0.97; erase / kept-art / English-ink masks; per-block lettering statistics; review
  sheets; reference-text bootstrap, corrected by hand for the four new pages, 4 keys uncertain);
  `demo/typeset_metrics.py` (erase IoU / recall / precision, art kept, SSIM + MAE in the erased
  neighbourhood, bubble residue, cap ratio, inset, centring, lines, overflow, collisions; per
  page and per block; batch summary).  Baseline + failure catalogue:
  `docs/perf/2026-09-10-typesetting-baseline.md`.  Tests: `tests/test_typeset_{dev,reference,metrics}.py`.
- **2 done**: `render/erase.py` redraws bubbles (paper interior filled, holes included; outline
  = big ink components bordering the interior from outside, kept with a 1-px fringe; tails
  kept); seven synthetic kinds + tail + joined tests in `tests/test_erase.py`; 642 tests green.
  Re-measure: identical scores on the five pages (the old eraser already left the interiors of
  these bubbles clean; the first low "recall" was outline-edge scan noise, now judged out).
- **3a done**: model research in `demo/output/model_research.json`; decisions: manga
  big-LaMa TorchScript (MIT / Apache-2.0 notice), Illustrious-XL-v1.0 (OpenRAIL++-M), xinsir
  controlnet-union promax (Apache-2.0), sdxl-vae-fp16-fix, torch cu130 for CPython 3.13;
  licences recorded in `renderer/MODELS.md`.  Protocol contract: `renderer/PROTOCOL.md`.
- **3b, 3c, 4 done** (three passes on disjoint files, then integration fixes):
  `renderer/glassrenderer/` (protocol, loopback server with token / Host / size guards, fake
  stages, compositing with the byte-identity guarantee, pipeline with preload + warm-up off the
  request path; 166 sidecar tests), real stages (`stages/lama.py` TorchScript manga big-LaMa
  fp32, `stages/sdxl.py` Illustrious-XL + union ControlNet lineart mode, fp16, automatic VAE
  tiling; `models.py` pinned table, `install.bat`, `MODELS.md`, GPU test, `tools/quality_probe.py`,
  `docs/perf/2026-09-11-quality-renderer-timings.md`), app side (`render/quality.py` client /
  panel jobs with a 1024 context window / scheduler, `render/quality_models.py`, `SegmentStyle`
  `erase_mask` / `clean_patch_serial` / `panel_box`, pipeline patch swap, overlay cache serial,
  config `quality_renderer`, Engines card `qualityCard`, download worker, spec; 690 app tests).
  Deviations recorded: unknown `params` keys are a 400; the union ControlNet needs the
  `StableDiffusionXLControlNetUnionImg2ImgPipeline`; `/health` carries `load_error`.
- **5 in progress**: parameter sweep (`demo/output/sweep/sweep2.log`, tags `q_*`) scored with
  `demo/typeset_metrics.py`; defaults to be fixed in `glassrenderer/protocol.py`.
  Found and fixed on the way: the spawned sidecar answered no request for its whole warm-up
  (`/health` and the first `/inpaint` timed out, so the batch renders silently fell back to the
  quick fill).  Cause: the stdin watcher's blocking `read(1)` on the parent's pipe — while torch
  and diffusers loaded in the preload thread, that parked read stopped every new thread from
  starting (the accept loop hung in `Thread.start`, `ExitProcess` hung too).  Bisected with the
  real `serve()` (signals, logging, main-thread `serve_forever` all innocent).  `watch_stdin`
  now polls (`PeekNamedPipe` / `select`, 0.25 s); the contract (exit on stdin EOF) is unchanged;
  `renderer/tests/test_stdin_watch.py`.  `QualityClient` now absolutises `models_dir` (the
  sidecar's cwd is `renderer/`).
- **5 done** (2026-09-11): results in `docs/perf/2026-09-11-quality-renderer.md`.  Defaults stay
  the plan's (strength 0.4, 24 steps, ControlNet 0.8, guidance 4.0, seed 0, feather 2; the sweep
  tags are indistinguishable within the LPIPS noise floor of ±0.07 per block).  Changed on the
  evidence: `art_under_text` measures its ring beyond a 2-px glyph fringe (the fringe alone read
  as 15.6 % "art" on plain paper) with `QUALITY_MIN_ART = 0.05`; `panel_jobs` splits a panel's
  members into clusters that fit one 512–1024 px window (`_cluster_members`) instead of sending
  a page-sized panel downscaled; `LamaStage.warm_up` also runs a 1024² pass so the allocator pool
  holds every block a job needs (a fresh `cudaMalloc` next to the resident SDXL weights cost
  6–16 s per page under WDDM).  Result: `before.jpg` LPIPS 0.404 → 0.380 with the two visibly
  broken blocks fixed (0.363 → 0.301, 0.311 → 0.235); per page ≤ 8.8 s warm; the remaining
  failures (fake kana on 1ja 13/14, a face regenerated under 2ja 11) are undetected shared
  bubbles — a layout gap the old eraser shares, catalogued in the perf document § 4.
- **6 done**: app suite 695, sidecar suite 173 (+5 skipped GPU), GPU test green earlier in the
  session, Qt/PIL parity via the overlay-cache tests, exe built (166.8 MB, 91 s) and smoke-tested
  with the sidecar absent (72/74 under load: the two misses are run C's hard-kill precondition
  racing the app's own autoexit on a fast machine, a harness timing issue re-run below).
- **7 done**: two workflows (code + python + security reviewers with one refuting verifier per
  CRITICAL/HIGH; typesetting-quality reviewer over the `final` sheets).  Confirmed HIGH, fixed:
  (1) no models-ready gate — the scheduler spawned a torch process, failed and respawned every
  30 s forever when the 9.3 GB store was absent (`engines._default_quality_factory` now returns
  None with one status line unless `quality_models.models_ready`; `_ensure_client` also reads
  `/health` `models.*` per PROTOCOL.md and backs off 300 s on `QualityModelsMissing`);
  (2) `QualityScheduler.stop()` could orphan a sidecar started during its bounded join and deliver
  a result after stop (the client is published before `start()`, the worker drops its client in a
  `finally`, `_process` re-checks `_stopping` before and after the job, `_spawn` re-checks
  `_closed` after `Popen`, `_stop_quality` clears queued results).  Also applied from the
  MEDIUM/LOW advisories: `QualityUnavailable.transport` so a job error keeps a warm sidecar,
  (typesetting-quality reviewer, HIGH, fixed) the step-2 bubble redraw painted over art drawn
  across a balloon (4ja: a chibi's head and eyes inside the balloon's paper component were
  filled as if they were stray glyphs; `base` kept them) — the fill is now limited to the text
  boxes grown by `BUBBLE_ZONE_EM = 1.5` glyphs, which still covers furigana and strays next to
  the column; the price is that a column the OCR never boxed and that lies farther away stays
  (4ja `説明して`), which is the pre-existing OCR-coverage limitation, not a regression;
  (typesetting-quality reviewer, HIGH, not fixable in this slice) balloon interiors classified
  as free text (`in_bubble=false`: 1ja 13/14, 2ja 11, 4ja 9/11 — shared / angular balloons
  whose text splits into several column blocks) are routed to the sidecar, which then leaves
  residual or invented strokes on paper (the reviewer's CRITICAL rating of that residue was
  refuted by its verifier down to MEDIUM: the anchor sheet did not show invented glyphs and the
  mechanism claim was wrong) — root cause and fix direction in
  `docs/perf/2026-09-11-quality-renderer.md` § 4, first item of the next steps; two further
  HIGHs are pre-existing placement defects this change exposes rather than causes (4ja 9's line
  set outside its now-empty balloon, overprinted lines on 4ja 10/11 and 1ja 13) — catalogued;
  the reviewer judged `final` better than `base` on all five pages;
  `QualityClient.wait_warm` (the scheduler's worker and the harness poll `/health` until the
  sidecar is warm before the first job, so `REQUEST_TIMEOUT_S` covers the job and not the
  90–270 s warm-up — a re-render under CPU contention had timed out on every page and fallen
  back to the quick fill without saying so loudly),
  `missing_files` compares the pinned size (app and sidecar agreed on "ready" only by luck),
  `Job.image` is an owned copy, `_read_ready` snapshots the deque, `SO_EXCLUSIVEADDRUSE` and a
  30 s socket timeout on the server, `MAX_IMAGE_PIXELS` before PNG decode, no token slice in
  `job_id`.  Refuted: "ControlNet is conditioned on the LaMa output" (the control map has the mask
  blanked, so it is identical either way).  Left as advisories: split `render/quality.py` (935
  lines), a curated sidecar environment, re-hashing all 24 files at SDXL load, the committed
  pickle caches under `demo/cache/`, `demo/reference/` size (6 MB).
