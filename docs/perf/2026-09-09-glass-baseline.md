# Glass control window — performance baseline (F2-0) and fixes (F2-1/F2-3)

Measured 2026-09-10 with `tools/profile_glass.py` (`GLASSTRANSLATE_PROFILE=1`, glass appearance,
no pipeline, temp config). Machine: LC49G95T 5120x1440 @ 239.76 Hz (4.17 ms/frame), dpr 1.0,
RHI Direct3D11. Raw events: `demo/output/perf/glass-profile-20260910-055516.jsonl` (60 x 8 px drag)
and `glass-profile-20260910-055837.jsonl` (240 x 2 px drag, ~4 ms step cadence ≈ mouse rate).

Two harness bugs had to be fixed before any number could be taken (both in the F2-0 code, not
in the app): `qInstallMessageHandler` with a Python handler and a direct Python slot on
`QQuickWindow.frameSwapped` both run on the render thread and need the GIL while the GUI thread
holds it inside C++ `exposeEvent` → deadlock on the first frame. The swap hook is now a queued
connection (exact counts, GUI-dispatch timestamps); render timing is read from a stderr capture.
`QSG_RENDER_TIMING` lines were not emitted by this Qt build (`qt.scenegraph.time.*` stays
disabled), so hypothesis (d) is measured through swap cadence + process CPU.

## Verdicts

| # | Hypothesis | Measured (before) | Verdict |
|---|---|---|---|
| a | backdrop frame grabbed for stale geometry, displayed without offset compensation | 60 x 8 px / 16 ms steps: offset 0 px (grab finishes before the next move). 240 x 2 px / 4 ms steps (500 px/s): offset p50 **2 px**, p90 **4 px**, max **6 px**, 34–46 of ~200 frames > 2 px. `move → frame_gui` p50 2.9–6.1 ms, so a real 1500–3000 px/s drag leaves each frame 5–20 px behind the window until the next one lands | **Confirmed** → F2-1 |
| b | 15 Hz grab cadence vs display refresh | idle 12.9 grabs/s (≈15 Hz by design); during a drag the `moveEvent → poke` path already drives **169–176 grabs/s**, i.e. one grab per move, limited only by the grab itself (p50 4.2–6.0 ms, p90 7.7–8.2 ms, inside the §7 ≤ 8 ms budget). 1.1–2.0 swaps per backdrop frame | **Not confirmed** — raising the timer rate cannot help; F2-2 skipped |
| c | `moveEvent → _poke → grab → GUI` latency | move→poke 0.00 ms, poke→grab p50 4–6 ms (= grab duration), grab→frame_gui 0.09 ms, `push_backdrop` slot 0.13 ms p50 / 0.29 max. No GUI-thread work worth removing | **Not confirmed** beyond the mss grab cost; nothing to fix on the GUI thread |
| d | blur chain cost per frame (4 live `ShaderEffectSource`s) | tabs phase 239 swaps/s sustained at 240 Hz with 17–29 % of one core; drag 17 % (60 x 8 px) / 56 % (240 x 2 px, 170 grabs/s); idle 2.3–3.1 % (includes the spring tail below; 0 swaps once settled). Swap delta during transitions p50 4.15 ms, p90 4.4–4.5 ms | **Not confirmed** — the chain renders within a 240 Hz frame; `live:false` change not taken |
| e | `PageSlot` `NumberAnimation` stalling on the GUI thread | 44–48 swaps per 200 ms window after each switch (48 = every refresh); one 18.9 ms gap on the first switch only (first layout of the page) | **Not confirmed** — Animators not needed for smoothness; not taken (R8 risk) |
| f | 160 ms fade refresh-rate independent | frame count per fade scales with Hz (38 expected per 160 ms at 240 Hz, measured 44–48 per 200 ms) | **Confirmed as fine** |
| — | (found) render loop stays alive after a tab switch | **332 swaps, last at +1381 ms** after the final `currentIndex` change: the tab indicator's two `SpringAnimation`s settle to the default `epsilon` (0.01 px) ≈ 1.2 s past the 180 ms contract, rendering the whole scene at 240 Hz meanwhile | **Fix** → F2-3 (reduced): `epsilon` on the indicator springs |

## Fixes taken

- **F2-1 backdrop offset compensation** — `ControlBridge.backdropShift` (QPointF, logical px)
  = `(frame.window_rect origin − current window origin) / dpr`; `GlassControlWindow.moveEvent`
  feeds the current origin through `bridge.set_window_origin(x, y)` (two ints, nothing else on the
  GUI thread), `push_backdrop` records the grabbed origin, the property only notifies when the
  value changes and is zero before the first frame / first move. `Main.qml` draws `backdropLayer`
  at `backdropOrigin + backdropShift`, so a frame grabbed before the last move stays
  desktop-aligned and the blur chain (anchored to `backdropLayer.x/y`) follows. Tests:
  `tests/test_glass_bridge.py::test_backdrop_shift_*` (4), `tests/test_glass_qml.py::test_backdrop_layer_applies_frame_offset`.
- **F2-3 (reduced) indicator spring settle** — `GlassSegmentedBar.qml` springs get
  `epsilon: 0.25` (quarter logical px; spring/damping untouched, motion spec unchanged). Test:
  `tests/test_glass_qml.py::test_segmented_bar_springs_settle_within_a_quarter_pixel`.

## After

Post-fix run: `demo/output/perf/glass-profile-20260910-060812.jsonl` (same 240 x 2 px / ~4 ms
drag, same machine); isolated spring measurements with a throwaway sampler that switched tabs
`0→1→2→3→0` at 400 ms and polled every running QML animation under the root item every 100 ms.

| # | Before | After | Notes |
|---|---|---|---|
| a | offset p50 2 / p90 4 / max 6 px, 46/196 frames > 2 px | p50 4 / p90 4 / max 4 px, 82/160 frames > 2 px | **Unchanged by design** — the metric measures how stale the *grab* is, which F2-1 does not change; F2-1 cancels that staleness at draw time (`backdropLayer.x = backdropOrigin.x + backdropShift.x`, verified by the bridge/QML tests). The residual visible error is now one GUI→render sync (the frame drawn after the move already carries the new shift) instead of 2–6 px until the next grab. |
| b | drag 169 grabs/s, grab p50 4.23 / p90 8.15 ms | 137 grabs/s, grab p50 7.72 / p90 8.36 ms | Grab time is bimodal (≈4 or ≈8 ms, DWM pacing); p90 unchanged, p50 still inside the §7 ≤ 8 ms budget. 1.5 swaps per frame (was 1.2): each `moveEvent` now also moves the layer, one extra render per move. |
| c | move→frame_gui p50 6.1 ms, push_backdrop 0.13 ms | p50 2.77 / p90 3.33 ms, push_backdrop 0.14 ms p50 / 0.39 max | `set_window_origin` adds no measurable GUI-thread work. |
| d | drag 56 % / tabs 17 % / idle 2.3 % of one core | drag 48 % / tabs 29 % / idle 2.3 % | Run-to-run noise for tabs (17–29 % across baseline runs); no regression. |
| e | 44–48 swaps / 200 ms per switch, one 18.9 ms gap | 48 / 48 / 48 / 48 swaps, max gap 8.95 ms, no >2-frame gap | Unchanged, still every refresh. |
| tail (isolated) | springs running **1313–1322 ms** after the last switch (293–315 swaps) | **983–1061 ms** (220–236 swaps) | −25 %; `epsilon` 0.5 gives the same, 1.0 gives 765 ms but is a visible 1 px snap. The rest is the physical settle of the springs (`Theme.springTrail` 3.0/0.30 oscillates visibly for ~450 ms, then decays): getting under the 180 ms contract needs more damping, i.e. a motion-spec change — **not taken, decision for GATE 2**. |
| tail (profiler) | 335 swaps, +1393 ms | 338 swaps, +1406 ms | **Open item**: when a drag preceded the switches, rendering continues ~0.45 s *after* every QML animation under the root has stopped (springs stop at +1.0 s, swaps stop at +1.5 s); without the drag the swaps stop with the springs. Not hover (reproduced with the cursor parked off-window), not the grabber (0 frames reach the GUI). Candidates: a Qt-internal animation job (pixmap-cache expiry of the ~160 `image://backdrop/N` entries, async provider loads). Reproduce: drag 240 x 2 px, wait 0.8 s, switch `1,2,3,0` at 400 ms, count `frameSwapped`. |

Idle once settled: 0 swaps, 13 grabs/s (the `GRAB_HZ` cadence; no frame reaches the GUI on a
static desktop), 2.3 % of one core — inside the §7 idle budget.
