# Lag baseline: the GUI thread while the pipeline runs (2026-09-18)

Measured with `tools/profile_glass.py --with-pipeline` on the development machine: LC49G95T
5120x1440 @ 239.76 Hz (4.17 ms per frame), dpr 1.0, RHI Direct3D 11, `GLASSTRANSLATE_PROFILE=1`,
glass appearance. The question: does the control window stay responsive while OCR, translation and
typesetting run in the same process, and if not, what exactly blocks it?

```
PYTHONUTF8=1 .venv/Scripts/python tools/profile_glass.py --steps 240 --step-px 2 --repeat 5 \
    --with-pipeline --report docs/perf/2026-09-18-m1-lag-baseline.json
```

## Method

The driver plays the same script (240 x 2 px drag, four tab switches, 2 s idle) five times in
three separate processes, one after another:

| run | what is built |
|---|---|
| `baseline` | the bare control window, no config, no pipeline (as in `2026-09-09-glass-baseline.md`) |
| `session_idle` | the whole app session (control window + glass overlay, hotkeys) on a copy of the user's `config.json`, pipeline stopped |
| `pipeline` | the same session with the pipeline running: `Examples/before.jpg` (20 blocks) and `more_comparisons/1ja.jpg` (9 blocks) are served in turn every 2 s in place of the screen, manga mode on, quality renderer off, OCR `mangaocr`, translator `gemini`; both pages are translated once before the measurement starts, so every later translation is a cache hit |

Two signals are judged, both in frame intervals (1.0 = on time, 2.0 = one frame missed):

- **`tick`** - a 4 ms precise `QTimer` on the GUI thread. A gap is time during which the GUI
  thread ran no event at all. It works in every phase, including idle, where nothing is presented.
- **`swap`** - `frameSwapped`, queued to the GUI thread, inside the drag phases and the 200 ms
  after each tab switch. During a drag a swap needs a new backdrop frame, and the GDI grab takes
  5-8 ms, so the drag phase always contains 2-frame gaps and a few long ones at its start
  (17 frames in every baseline): the swap gaps of the drag phase describe the grab cadence, not
  a stalled GUI. The tick is the stall measure.

The overlay's own cost is recorded by four marks that are empty calls unless profiling is on:
`overlay_apply_ms`, `overlay_typeset_ms`, `overlay_layer_ms`, `paint_ms`. A "new page" is an
apply that carried segments the overlay had not laid out before. Every tick gap above two frames
over the whole run is then attributed: `paint` (an overlay `paintEvent` ran inside it), `gc`
(a garbage collection did), `delivery` (a new page was applied right after it), `unattributed`.

## Result

Two full runs on a quiet desktop. The committed JSON is the first one, re-judged with the final
analysis code (`--analyse <pipeline> --against <control>`; its run names are `baseline` / `run`).

| run (start) | mode | ticks | largest tick gap | > 2.0 fr | > 2.5 fr | late ticks by cause |
|---|---|---|---|---|---|---|
| 15:53 | baseline | 6001 | 6.5 ms (1.56 fr) | 0 | 0 | - |
| 15:53 | session_idle | 5987 | 7.5 ms (1.80 fr) | 0 | 0 | - |
| 15:53 | pipeline | 5862 | **146.5 ms (35.1 fr)** | 13 | 11 | paint 13 (<= 146.5 ms), delivery 4 (<= 12.0 ms), unattributed 1 (72.0 ms) |
| 16:27 | baseline | 5958 | 129.0 ms | 1 | 1 | unattributed 1 (129.0 ms) |
| 16:27 | session_idle | 5985 | 6.7 ms (1.60 fr) | 0 | 0 | - |
| 16:27 | pipeline | 5730 | 668.3 ms | 15 | 13 | paint 16 (<= 132.3 ms), delivery 7 (<= 12.6 ms), unattributed 1 (668.3 ms) |

Tick counts are inside the scripted phases; the cause column covers the whole run, so it also
counts pages that landed between two phases.

Swap gaps (drag phases + tab windows), for completeness:

| run | mode | largest | > 2.0 fr | > 2.5 fr | swaps per 200 ms tab window (mean) |
|---|---|---|---|---|---|
| 15:53 | baseline | 71.3 ms | 25 | 3 | 48.2 |
| 15:53 | session_idle | 67.2 ms | 22 | 2 | 48.2 |
| 15:53 | pipeline | 147.1 ms | 39 | 10 | 44.2 |
| 16:27 | baseline | 71.4 ms | 26 | 10 | - |
| 16:27 | pipeline | 128.7 ms | 31 | 14 | - |

Overlay cost per new page (ms, p50 / p90 / max over the pages of the run):

| run | pages | apply | typeset (sum per page) | layer (sum per page) | paint |
|---|---|---|---|---|---|
| 15:53 | 13 | 0.4 / 0.5 / 0.6 | 51.1 / 66.5 / 70.2 | 63.3 / 76.9 / 77.0 | 116.4 / 143.4 / 144.3 |
| 16:27 | 16 | 0.3 / 0.5 / 0.5 | 29.3 / 59.2 / 61.8 | 11.0 / 67.9 / 71.2 | 41.4 / 124.8 / 131.3 |

The distribution is bimodal because the two pages differ (15:53 run): the 9-block page costs
typeset 63.8 + layer 72.4 = **138 ms** per paint (p50; max 144.3), the 20-block page, of which
16 blocks are re-laid each time, typeset 25.4 + layer 9.1 = **35 ms** (max 37.3).

## What blocks the GUI thread

1. **Every new page freezes the GUI thread for one overlay paint.** 13 of 13 pages (15:53) and
   16 of 16 (16:27) produced a late tick whose length equals the `paint_ms` inside it plus
   2-3 ms. `typeset_block` and `_render_layer` run inside `GlassOverlay.paintEvent`
   (`glasstranslate/ui/overlay.py`), once per block, on the first paint after a result arrives:
   31-146 ms = 8-35 missed frames at 240 Hz, in the drag, tab and idle phases alike. The
   "about 80 ms + 80 ms" of the overlay's docstring is confirmed for the 9-block page
   (64-70 ms + 72-77 ms).
2. **Delivering a result costs another 9-13 ms** on the GUI thread just before the apply (the
   queued result slot and the stats update): 4 and 7 late ticks of 2.2-3.0 frames.
3. **The interpreter lock is not a measurable cause of stalls above two frames** in these runs.
   Outside the paints and deliveries the pipeline run has one late tick in ~35 s (72 ms, then
   668 ms in the repeat), and a pipeline-free baseline produced one of 129 ms as well, so that
   class exists without the pipeline. During those gaps the backdrop grabber thread recorded
   nothing either: the whole process stood still. Garbage collection is ruled out by the later
   runs, which recorded it (17-21 collections per run, none above 3 ms, none inside a late tick,
   although those runs had many such freezes). Whether the rest is a C call
   holding the lock or GPU / compositor contention with the DirectML session is **not decided**
   by this data.
4. The session itself (overlay window, hotkey hook, real config) costs nothing measurable:
   `session_idle` equals `baseline`.

Conclusion: the premise "typesetting and lettering run in `paintEvent`" is reproduced and is the
dominant, deterministic stall; "the GUI shares the interpreter lock with the pipeline" is not
shown to cost more than the ~10 ms delivery hiccup. Taking layout and layer rendering off the GUI
thread removes (1); (3) has to be re-measured afterwards.

## Proposed gate for the fix

`tools/profile_glass.py --with-pipeline --assert-gate` (pipeline run against the baseline of the
same session): largest swap and tick gap <= the baseline's + 1 frame; no more gaps above 2.0
frames than the baseline; the run measured at least half of the baseline's samples, finished its
script and dropped no event. Today it fails on all four gap checks (tick: 35.1 frames against a
limit of 2.56; 13 gaps against 0).

## Caveats

- **Online translator.** The session modes use the configured backend; with `gemini` the OCR text
  of the two pages is sent once each (the driver warns). `--translator argos|identity` keeps a
  run offline. Translation is outside the measured stalls either way (cache hits).
- **Disturbed desktop.** Three further full runs after 16:40 were discarded: pipeline-free
  baselines showed up to 115 late ticks and freezes of 0.1-1.2 s that came and went between
  consecutive child processes, and one run had the backdrop `BitBlt` fail with "access is denied"
  (the desktop was not the interactive one). A valid session has a `baseline` and a
  `session_idle` with zero or one late tick; check that first.
- **One native crash.** One pipeline child exited with `0xC0000005` while loading the manga-ocr
  DirectML session, before the measurement started; it did not reproduce.
- **Event capacity.** The recorder used to be a fixed 20,000-event deque that silently evicted
  its oldest events; a 5-repetition run records ~18,000 events per mode. It now counts evictions
  (`dropped`), the driver sizes it (`--max-events`, default 100,000 per repetition, at least
  200,000) and a run that dropped an event fails.
- **Observers that were tried and removed.** A second heartbeat from a plain Python thread (to
  tell a held interpreter lock from a blocked GUI thread) made control-only runs late by itself
  and was dropped. A `gc` callback that recorded through the profiler deadlocked the GUI thread
  (a collection can start inside `Profiler.mark`, under its lock); collections are now buffered
  in a plain list and merged into the dump afterwards.

Raw events: `demo/output/perf/glass-profile-20260918-155305-{control,session,pipeline}.jsonl` and
`...-162742-...` (ignored by git). Re-judge any of them with
`tools/profile_glass.py --analyse <run>.jsonl --against <control>.jsonl`.
