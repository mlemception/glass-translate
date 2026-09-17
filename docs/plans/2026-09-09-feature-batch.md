# Feature batch plan — 2026-09-09

Five features as thin vertical slices, each independently testable, TDD-driven, ~one commit.
Baseline: 297 tests green (commit 12f3242). Status: **committed 2026-09-10 at GATE 2**
(`19ca771` feat + `7d837b1` docs, 603 tests green; see `the build notes` session 5 for the hand-off).

Features: (F1) overlay capture mode, (F2) liquid-glass performance, (F3) OCR pipeline
(manga-ocr primary / PaddleOCR-family fallback), (F4) Gemini translation provider,
(F5) quick series context. D-1 is docs/packaging.

## Gate 1 decisions (user-approved; these override the draft text below where they differ)

* **OCR**: manga-ocr as ONNX (`onnx-community/manga-ocr-base-ONNX`, **fp16** encoder + decoder +
  `vocab.txt`, ~231 MB) run in-app on `onnxruntime-directml` with CPU fallback. rapidocr
  PP-OCRv5 (already installed) is the fallback engine, labelled **"PaddleOCR (PP-OCRv5 via
  ONNX)"**, config name **`paddleocr`** with **`rapidocr`** accepted as an alias. manga-ocr is
  the **default** engine; models are downloaded on first run into `default_models_dir()`.
  Furigana filter (`ocr/furigana.py`) applies on the PaddleOCR path. **Do not add** torch,
  paddlepaddle, or `manga-ocr-torchless`.
* **Gemini**: plain `requests`; default model **`gemini-3.8-flash`** with `thinkingLevel: "low"`.
  New `gemini_api_format` setting: **`native`** (`x-goog-api-key`,
  `/v1beta/models/{model}:generateContent`) or **`openai`** (`Authorization: Bearer`,
  `/v1beta/openai/chat/completions`); custom `base_url` for both. Retry 408/429/5xx with
  jittered backoff, `Retry-After` clamped to 30 s, timeout `(5, 30)`, cancel via
  `threading.Event`. API key in `%LOCALAPPDATA%\GlassTranslate\secrets.json` with
  `GEMINI_API_KEY` env override; never in `AppConfig`; redacted in logs; password echo in QML.
* **Series context**: `[Series Name]` placeholder, case-insensitive. **Empty name strips the
  placeholder and collapses whitespace** (not the whole line). Malformed template → default +
  status warning. Name trimmed and capped at **80** chars. Series name and template hash join
  the cache key.
* **Capture mode**: overlay only becomes capturable via `win32.set_capture_excluded(window,
  bool)`; `pipeline.pause()` while on; never persisted; always starts off.
* **Glass perf**: profiling slice first (`tools/profile_glass.py`), then offset-compensated
  backdrop motion and `Animator`-based tab transitions.

## Approved slice order

```
F1 capture mode (3)  →  F4 secrets + Gemini client + translator + UI
→  F5 prompt template + wiring + UI  →  F3 OCR fallback/furigana/model fetch/recognizer/default
→  F2 glass profiling + fixes  →  D-1 docs/packaging
```

## Draft order (superseded by the approved order above)

```
F1 (capture mode)  →  F2-0 (profile only)  →  F4 (Gemini)  →  F5 (series context)
                   →  F2-1..3 (perf fixes)  →  F3 (OCR)  →  D-1 (docs/packaging)
```

* **F1 first** — smallest, self-contained, creates the `Misc` card on `OverlayPage.qml`.
* **F2-0 early** — read-only instrumentation; baseline taken before F4/F5 add controls so the
  "trailing while dragging" numbers are not confounded. Fixes land after F5 so re-measurement
  happens once.
* **F4 before F5** — F5's prompt assembly needs the Gemini provider as its consumer.
* **F3 last** — largest dependency/packaging risk, gated on a user decision (Gate 1) and a
  machine measurement (F3-1 spike). Must not block the other four from shipping.

## Research findings that shaped the plan (verified Sept 2026)

* **manga-ocr** 0.1.16 installs on Python 3.13 / Windows, but its torch path is CPU-only here
  (`torch-directml` last released Sept 2024, torch 2.4.1, cp312 max; CUDA wheels ~3 GB) and
  torch+transformers adds 300–500 MB to a onefile exe. It emits one string per crop with no
  boxes, is trained to ignore furigana, and hallucinates on empty crops. Apache-2.0.
* **ONNX ports** of the same model exist (onnx-community/manga-ocr-base-ONNX: fp16 ~231 MB,
  int8 ~117 MB, q4f16 ~74 MB; mayocream/manga-ocr-onnx; pip `manga-ocr-torchless` with DML
  autodetect) and run on the app's existing `onnxruntime-directml`.
* **paddlepaddle** 3.3.1 has cp313 Windows wheels but GPU is CUDA-only and PyInstaller freezing
  of PaddleOCR 3.x/PaddleX is fragile. **rapidocr 3.9.2 (installed) ships PP-OCRv5/v6 — PaddleOCR's
  own models — with Japanese rec and `use_dml`.** "PaddleOCR fallback" = rapidocr running them.
* **Gemini**: plain `requests` against `POST {base_url}/v1beta/models/{model}:generateContent`,
  header `x-goog-api-key` (never `?key=`). Default `gemini-3.1-flash-lite`; presets
  `gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-3.8-flash`. Retry 408/429/5xx with
  exponential backoff + jitter, honour `Retry-After` (clamp 30 s). Safety filters default off on
  2.5/3.x; still send explicit `BLOCK_NONE`. Custom base URL works with Cloudflare AI Gateway,
  LiteLLM pass-through, new-api. `google-genai` SDK rejected (≈10 heavy deps, lazy imports).
* `jaconv` and `huggingface_hub` are **not** in the venv today; `requests` is (transitively).
* Furigana handling **already exists** in `render/layout.py` (`_mark_furigana`) and
  `render/erase.py`, both needing one `Segment` per source line — so manga-ocr must recognise
  per detected line, not per bubble.

---

## F1 — Overlay capture mode

**Semantics.** Runtime-only toggle `overlayCaptureMode`, default off. When on: clear
`WDA_EXCLUDEFROMCAPTURE` on the *overlay* window and `Pipeline.pause()`; the last emitted
segments stay painted; GUI and in-flight translation keep running. When off: re-exclude and
`resume()` only if the pipeline was running when the mode was entered.

**Decisions.**
* **Not persisted** — it defeats the capture-exclusion invariant and freezes the pipeline;
  surviving a restart would leave a frozen pipeline with no memory of why. Lives on the bridge
  as runtime state, not in `AppConfig`.
* **Control window stays excluded** — its `BackdropGrabber` samples the desktop behind it; a
  capturable panel would mirror itself. One-line change if the user disagrees (Gate 1 Q3).
* **App close with mode on:** `teardown()` re-applies exclusion before hiding the overlay.
* **Overlay hide/show:** desired affinity stored on the overlay and re-applied in `showEvent`.

### F1-1 — `win32.set_capture_excluded` + overlay honours a stored affinity
* Files: `glasstranslate/ui/glass/win32.py` (add `WDA_NONE = 0`, `set_capture_excluded(window, excluded) -> bool`; `exclude_from_capture` becomes a wrapper); `glasstranslate/ui/overlay.py` (`set_capture_excluded(bool)` storing `self._capture_excluded`, applied in `_exclude_from_capture` and `showEvent`).
* Tests first (`tests/test_glass_capture_mode.py`): `test_set_capture_excluded_calls_affinity_with_wda_none`, `test_set_capture_excluded_calls_affinity_with_exclude_flag`, `test_exclude_from_capture_delegates_to_set_capture_excluded`, `test_overlay_show_reapplies_stored_affinity`.
* Accept: both directions call `SetWindowDisplayAffinity` with `0x11`/`0x0`; overlay re-applies stored value on every `showEvent`; existing win32 tests pass.
* Security trigger: **yes**. Size: S.

### F1-2 — bridge state + app wiring (headless)
* Files: `glasstranslate/ui/control.py` (`overlayCaptureModeChanged` Signal, plain `Property(bool)` with setter — not `_config_property`; `capture_mode_requested = Signal(bool)` relayed by `ControlWindow` like `grab_mode_requested`); `glasstranslate/ui/app.py` (`_on_capture_mode(bool)`: remember `_capture_mode_prev_running`, `overlay.set_capture_excluded(not on)`, `pipeline.pause()` / conditional `resume()`, status message; never `stop_pipeline()` — it clears overlay segments); `teardown()` re-exclusion.
* Tests first: `test_capture_mode_defaults_off`, `test_capture_mode_is_not_written_to_config_json`, `test_enabling_pauses_pipeline_and_unexcludes_overlay`, `test_disabling_restores_previous_running_state`, `test_disabling_does_not_start_a_stopped_pipeline`, `test_capture_mode_keeps_last_segments`, `test_teardown_reexcludes_overlay_when_mode_on`.
* Accept: toggling twice is a no-op round trip; `config.json` byte-identical after a toggle; status strip shows an explicit warning while on.
* Security trigger: **yes**. Size: M.

### F1-3 — `Misc` card on the Overlay page
* Files: `glasstranslate/ui/qml/pages/OverlayPage.qml` (third `GlassCard { title: "Misc" }` with `GlassToggle` objectName `captureModeToggle` + hint naming the consequences); regenerate resources.
* Tests first: `tests/test_glass_qml.py::test_overlay_page_has_capture_mode_toggle`, `::test_capture_mode_toggle_round_trip`.
* Accept: `tools/build_resources.py --check` passes; page instantiates offscreen; smoke `pages_ok` unaffected.
* Security trigger: no. Size: S.

---

## F2 — Liquid-glass performance

**Outcome (2026-09-10, `docs/perf/2026-09-09-glass-baseline.md`):** F2-0 harness landed and the
baseline was taken at 240 Hz. (a) confirmed → **F2-1 taken** as a single
`ControlBridge.backdropShift` property (simpler than the two properties drafted below; tests
`tests/test_glass_bridge.py::test_backdrop_shift_*` and
`tests/test_glass_qml.py::test_backdrop_layer_applies_frame_offset`). (b)/(c) not confirmed — a
drag already drives one grab per `moveEvent` (137–170 grabs/s), so **F2-2 skipped**. (d)/(e) not
confirmed — the blur chain and `PageSlot` render every refresh at 240 Hz, so Animators and
`live:false` **not taken** (R8 avoided); **F2-3 reduced** to `epsilon: 0.25` on the tab-indicator
springs (render tail after a switch 1.3 s → 1.0 s). Open: the remaining tail is the motion spec
(more damping = GATE 2 decision) and an unexplained ≈ 0.45 s of rendering after a drag + switch
sequence once every QML animation has stopped.

### F2-0 — profiling harness (measurements only, no fix)
* Files: `tools/profile_glass.py` (new; scripts a window drag and a tab-switch burst, writes JSONL); `backdrop.py` + `control.py` (timestamps behind `GLASSTRANSLATE_PROFILE=1`, zero cost when unset); `Main.qml` (`onFrameSwapped` counter behind the same flag).
* Hypotheses and deciding metric:

  | # | Hypothesis | Metric |
  |---|---|---|
  | a | backdrop frame captured for stale geometry, displayed without offset compensation | `frame.window_rect` vs `physical_rect(window)` at paint time during a drag |
  | b | 15 Hz grab cadence vs display refresh | grab-interval histogram; frames drawn per backdrop update at 240 Hz |
  | c | `moveEvent` → `_poke()` → grab wakeup latency | timestamps at `moveEvent`, `poke()`, `_grab_once`, `_on_backdrop_frame` |
  | d | blur-chain cost per frame (4 live `ShaderEffectSource`s) | `QSG_RENDER_TIMING=1` render ms with chain `live:true` vs pinned |
  | e | `PageSlot` `NumberAnimation` stalling on the GUI thread | frame deltas during tab switch; GUI-thread Python slot durations |
  | f | `Theme.fade` fixed 160 ms already refresh-rate independent | frame count per transition at 60 vs 240 Hz |

* Tests first: `tests/test_glass_profile.py::test_profile_disabled_adds_no_timestamps`, `::test_profile_records_move_to_frame_latency`, `::test_profile_report_schema`.
* Accept: committed `docs/perf/<date>-glass-baseline.md` with per-hypothesis verdicts; idle CPU within GLASS_DESIGN §7 budgets.
* Security: no. Size: M.

### F2-1 — backdrop offset compensation (conditional on a)
* Files: `control.py` (`bridge.backdropFrameOrigin` from `BackdropFrame.window_rect`; `bridge.windowOrigin` updated in `moveEvent` — two ints, no other Python work); `Main.qml` (`backdropLayer.x/y += (frameOrigin - windowOrigin) / dpr`).
* Tests first: `tests/test_glass_backdrop.py::test_frame_origin_exposed_in_logical_px`, `::test_offset_is_zero_when_frame_matches_window`, `::test_offset_tracks_a_moved_window`; `tests/test_glass_qml.py::test_backdrop_layer_applies_frame_offset`.
* Accept: scripted 400 px drag stale-geometry error ≤ 2 px; no change while still. Size: M.

### F2-2 — drag-aware grab cadence (conditional on b/c)
* Files: `backdrop.py` (`set_rate(hz)` using the existing `_wake` Event); `control.py` (`startMove()` raises rate, single-shot timer restores it; also restored after a quiet interval).
* Tests first: `tests/test_glass_backdrop.py::test_set_rate_changes_period`, `::test_rate_restored_after_drag_timeout`, `::test_rate_is_default_when_idle`.
* Accept: drag CPU within §7 budget; idle CPU and grab count unchanged. Size: S.

### F2-3 — render-thread tab transitions + fewer redundant redraws (conditional on d/e)
* Files: `Main.qml` (`PageSlot`: `OpacityAnimator`/`XAnimator` on the render thread; **gotcha**: Animators don't write the property until finished, so `visible: opacity > 0.001` must become `visible: active || anim.running`); blur chain `live: false` + `scheduleUpdate()` driven by `bridge.backdropSerial`.
* Tests first: `tests/test_glass_qml.py::test_page_slot_uses_animators`, `::test_inactive_page_becomes_invisible_after_transition`, `::test_blur_chain_sources_are_not_live`, `::test_backdrop_serial_change_schedules_chain_update`.
* Accept: pointer-sweep CPU below F2-0 baseline; 240 Hz transition frame count ≥ 2× the 60 Hz count; no page left invisible-but-enabled. Size: M.

---

## F4 — Gemini translation provider

**Decisions.**
* **Transport**: `requests` (add explicitly to `requirements.txt`). No SDK.
* **API-key storage**: `GEMINI_API_KEY` env var first, else user-scope
  `%LOCALAPPDATA%/GlassTranslate/secrets.json`. **Never in `AppConfig`** → never in
  `config.json`, `to_dict()`, the smoke report, or an Engines-page screenshot.
* **Cancellation**: bounded `(connect=5, read=timeout_s)` + `threading.Event` checked between
  retry attempts, set by `close()`.

### F4-1 — secrets store
* Files: `glasstranslate/config/secrets.py` (new: `get_secret`, `set_secret`, `secrets_path()` under the user data dir, atomic write, `redact`).
* Tests first (`tests/test_secrets.py`): `test_env_var_wins_over_file`, `test_missing_secret_returns_empty_string`, `test_set_secret_writes_atomically_outside_config`, `test_corrupt_secrets_file_returns_empty_and_does_not_raise`, `test_redact_never_returns_the_value`, `test_secrets_path_is_under_user_data_dir`.
* Accept: `AppConfig.to_dict()` contains no key material; file not created until a key is set.
* Security: **yes**. Size: S.

### F4-2 — `GeminiTranslator` (offline unit-tested)
* Files: `glasstranslate/translate/gemini.py` (new, < 300 lines). Request: `system_instruction`, `contents`, `generationConfig{temperature,maxOutputTokens,thinkingConfig}` (`thinkingLevel:"minimal"` for 3.x **or** `thinkingBudget:0` for 2.5, never both), `safetySettings` BLOCK_NONE ×4. Response parsed defensively (`candidates[0].content.parts[].text`, `finishReason`, `promptFeedback.blockReason`, `{error:{code,message}}`). Retry 408/429/5xx, exp backoff 1 s + jitter, `Retry-After` clamped 30 s, `max_retries` 3. Numbered-line batch prompt; per-line soft failure keeps source text; never raises out of `translate_batch`.
* Tests first (`tests/test_gemini_translator.py`): `test_numbered_prompt_round_trip`, `test_returns_source_text_when_response_line_count_mismatches`, `test_returns_source_text_when_prompt_is_blocked`, `test_retries_on_429_and_honours_retry_after`, `test_does_not_retry_on_400`, `test_gives_up_after_max_retries_and_returns_inputs`, `test_cancel_event_aborts_between_attempts`, `test_api_key_is_sent_as_header_not_query`, `test_custom_base_url_is_used_verbatim`, `test_api_key_never_appears_in_log_records`, `test_empty_texts_short_circuits_without_a_request`.
* Accept: all HTTP mocked; zero key material in logs or exception messages.
* Security: **yes**. Size: L.

### F4-3 — config fields + factory registration
* Files: `settings.py` (`gemini_model = "gemini-3.1-flash-lite"`, `gemini_base_url = "https://generativelanguage.googleapis.com"`, `gemini_timeout_s = 30.0`, `gemini_max_retries = 3`); `translate/factory.py` (`_BACKENDS` + `"gemini"`, key from `config.secrets`); `core/pipeline.py` (`_TRANSLATOR_FIELDS` gains the four fields).
* Tests first: `tests/test_config.py::test_gemini_defaults_round_trip_through_from_dict`, `::test_unknown_gemini_model_type_is_ignored`; `tests/test_gemini_translator.py::test_factory_builds_gemini_backend`, `::test_factory_still_builds_argos_libre_identity`, `::test_changing_gemini_model_rebuilds_the_translator`, `::test_missing_api_key_reports_a_clear_status_not_a_traceback`.
* Accept: existing three backends identical; absent key → one actionable status + pass-through.
* Security: **yes**. Size: M.

### F4-4 — Engines page: provider selection + Gemini card
* Files: `pages/EnginesPage.qml` ("Gemini" `GlassCard` shown via `bridge.backendGemini`; model `GlassComboBox` with four presets; base URL / timeout / retries fields; API-key `GlassTextField password: true` with placeholder driven by `bridge.geminiApiKeySet`); `control.py` (`_CONFIG_FIELDS` + `_config_property` for the four fields; `geminiApiKeySet` read-only; `@Slot(str) setGeminiApiKey`; `_ONLINE_BACKENDS` + `"gemini"`).
* Tests first: `tests/test_glass_bridge.py::test_backend_gemini_flag`, `::test_gemini_online_flag`, `::test_set_gemini_api_key_does_not_touch_appconfig`, `::test_gemini_api_key_getter_returns_only_a_boolean`; `tests/test_glass_qml.py::test_engines_page_gemini_card_is_hidden_for_other_backends`, `::test_gemini_key_field_is_password`.
* Accept: `--check` passes; key field never renders stored characters.
* Security: **yes**. Size: M.

### F4-5 — `control.py` extraction (pure move)
* Files: new `glasstranslate/ui/bridge_fields.py` (`_Field`, `_CONFIG_FIELDS`, `_clamp`, `_strip`, `_cfg_get`, `_cfg_set`, `_config_property`, `_items`, `translate_device_items`, `format_stats`, `download_label`, `download_progress`); `control.py` re-exports so import sites are unchanged.
* Tests: existing `tests/test_glass_bridge.py` + `::test_control_module_still_exports_public_names`.
* Accept: 0 behavioural diff; `control.py` back under 800 lines. Size: M.

---

## F5 — Quick series context

**Decisions (as approved at Gate 1; supersede the draft).**
* Placeholder `[Series Name]`, matched **case-insensitively** (inner spaces tolerated).
* **Empty series name → strip the placeholder and collapse the whitespace it leaves** (spaces before punctuation, blank-line runs); the line stays.
* **Malformed template** (empty, no placeholder, > `MAX_TEMPLATE_CHARS = 4000`) → built-in default **plus** one status-strip warning naming the reason. An empty `series_prompt_template` in config means "use the built-in template" and is *not* a warning.
* Series name trimmed, whitespace-collapsed, capped at `MAX_SERIES_CHARS = 80`.
* **Gemini only** via optional `Translator.set_context(system_prompt)`; other providers no-op.
* **Cache key** includes `translator.context_key()` (sha256[:16] of the resolved prompt; `""` for the default) so cached lines cannot leak across series.
* `resolve_prompt` returns a frozen `ResolvedPrompt(prompt, warning, used_default)`.

### F5-1 — template engine (pure functions)
* Files: `glasstranslate/translate/context.py` (new: `DEFAULT_TEMPLATE`, caps, `validate_template(text) -> Optional[str]`, `resolve_prompt(template, series_name) -> Tuple[str, Optional[str]]`, `context_key(template, series_name) -> str`).
* Tests first (`tests/test_series_context.py`): `test_placeholder_substituted`, `test_placeholder_match_is_case_insensitive`, `test_empty_series_name_drops_the_placeholder_line`, `test_empty_series_name_collapses_blank_line_runs`, `test_empty_template_falls_back_to_default_with_reason`, `test_template_without_placeholder_falls_back_with_reason`, `test_oversized_template_falls_back_with_reason`, `test_series_name_is_trimmed_and_capped`, `test_context_key_is_stable_and_empty_when_unset`, `test_context_key_differs_between_series`.
* Accept: pure, no Qt/IO; 100 % branch coverage. Size: S.

### F5-2 — plumbing: config → pipeline → translator → cache key
* Files: `settings.py` (`series_name = ""`, `series_prompt_template = DEFAULT_TEMPLATE`); `core/interfaces.py` (`Translator.set_context`, `context_key()` no-op defaults); `translate/gemini.py`; `translate/cache.py` (`Key` gains `context` defaulting `""`; persisted `version` → 2, v1 loads as `context=""`); `core/pipeline.py` (`_CONTEXT_FIELDS = ("series_name", "series_prompt_template")` → `set_context` + invalidate, **not** in `_TRANSLATOR_FIELDS` so a name edit doesn't rebuild the model; pass `translator.context_key()` to `cache.get/put`).
* Tests first: `tests/test_cache.py::test_context_defaults_to_empty_and_is_backwards_compatible`, `::test_same_text_different_context_is_a_miss`, `::test_v1_file_loads_as_empty_context`; `tests/test_series_context.py::test_pipeline_pushes_context_to_translator_on_config_change`, `::test_context_change_invalidates_but_does_not_rebuild_the_translator`, `::test_non_gemini_backends_ignore_the_context`, `::test_gemini_receives_system_prompt_ocr_text_and_instructions_in_one_request`, `::test_malformed_template_reports_one_status_warning`.
* Accept: switching series yields fresh translations for identical text; Argos/Libre cache behaviour unchanged. Size: L.

### F5-3 — UI: series field on the main tab, template editor in settings
* Files: `glasstranslate/ui/qml/GlassTextArea.qml` (**new**, wraps `T.TextArea` inside a `Flickable`, `committed(text)` on focus loss / Ctrl+Enter); `qmldir`; `pages/TranslatePage.qml` ("Series" `GlassTextField` row in the "Session" card); `pages/EnginesPage.qml` ("Series context" card with `GlassTextArea` + "Reset to default" `GlassButton`); `control.py` (two `_CONFIG_FIELDS` entries with caps in the coercer).
* Tests first: `tests/test_glass_qml.py::test_glass_text_area_instantiates_and_is_accessible`, `::test_translate_page_has_series_field`, `::test_engines_page_has_template_editor`, `::test_reset_button_restores_default_template`; `tests/test_glass_bridge.py::test_series_name_is_trimmed_and_capped_by_the_bridge`, `::test_series_fields_round_trip_to_config_file`.
* Accept: `--check` passes (Templates only); commit path is `config_changed → pipeline.set_config`, no new mechanism. Size: M.

---

## F3 — OCR pipeline: manga-ocr primary, PaddleOCR-family fallback

```
detector:    rapidocr PP-OCRv5 det (ORT + DirectML)   → per-line quads
recogniser:  manga-ocr ONNX encoder/decoder (ORT-DML) → one string per detected LINE
fallback:    manga-ocr unavailable / model missing / inference error / empty output
             → rapidocr full pipeline (PP-OCRv5 / PP-OCRv6 rec = PaddleOCR's own models)
```

Models are never bundled: downloaded on first use into `models_dir` (dev `<project>/models`,
frozen `%LOCALAPPDATA%\GlassTranslate\models`) with a status line, like the Argos flow.

### F3-1 — compatibility spike (go/no-go; report only)
* Files: `tools/ocr_probe.py` (dev-only), `docs/perf/<date>-mangaocr-probe.md`.
* Proves on this machine: ORT-DML runs the manga-ocr ONNX encoder+decoder on 3.13; accuracy vs rapidocr on `Examples/` line crops; per-line latency GPU/CPU; size choice (fp16/int8/q4f16); PyInstaller import surface; licence.
* **No-go path**: keep rapidocr, add the PP-OCRv5 Japanese rec model + furigana filter (F3-6 only). Size: M.

### F3-2 — engine registry accepts more than one engine
* Files: `glasstranslate/ocr/factory.py` (`_ENGINES` → registry name → loader; `available_engines()` shape unchanged).
* Tests first (`tests/test_ocr_factory.py`): `test_available_engines_lists_registered_names`, `test_create_unknown_engine_raises_with_the_available_list`, `test_create_rapidocr_unchanged`, `test_registry_is_not_mutated_by_callers`. Size: S.

### F3-3 — model store + download
* Files: `glasstranslate/ocr/models.py` (new: `manga_ocr_dir`, `ensure_models(..., progress_cb)`, sha256 verification, temp+rename, refuses paths outside `models_dir`).
* Tests first (`tests/test_ocr_models.py`): `test_missing_model_reports_a_clear_status`, `test_download_verifies_digest_and_renames_atomically`, `test_corrupt_download_is_deleted_and_reported`, `test_path_traversal_in_a_model_name_is_rejected`, `test_frozen_build_uses_user_data_dir`.
* Security: **yes**. Size: M.

### F3-4 — `MangaOcrEngine` (primary)
* Files: `glasstranslate/ocr/mangaocr.py` (new: DML with CPU fallback mirroring `rapid.py`; WordPiece decode from `vocab.txt`; `jaconv` post-processing; `warmup()`; detector box score → `Segment.confidence`; hallucination guard: reject low-ink crops and degenerate repeats).
* Tests first (`tests/test_mangaocr_engine.py`): `test_returns_one_segment_per_detected_line`, `test_quads_are_in_source_image_coordinates`, `test_detector_box_score_becomes_segment_confidence`, `test_blank_crop_produces_no_segment`, `test_degenerate_repeated_output_is_dropped`, `test_cpu_provider_is_used_when_dml_is_unavailable`, `test_device_attribute_reports_gpu_or_cpu`.
* Accept: fake ORT sessions in tests; `Segment` contract identical to `RapidOCREngine`. Size: L.

### F3-5 — fallback chain + config
* Files: `glasstranslate/ocr/chain.py` (new `FallbackOCREngine`: triggers = construction failure, missing models, per-call exception, empty output where detector found boxes; one status per *transition*; sticky-with-retry backoff like `_ENGINE_RETRY_S`); `settings.py` (`ocr_fallback = "rapidocr"`); `factory.py` wiring.
* Tests first (`tests/test_ocr_fallback.py`): `test_primary_result_is_used_when_it_succeeds`, `test_construction_failure_falls_back_and_reports_once`, `test_per_call_exception_falls_back_for_that_call`, `test_empty_primary_output_falls_back`, `test_fallback_does_not_spam_status_every_frame`, `test_primary_is_retried_after_the_backoff`, `test_both_engines_failing_returns_empty_and_reports`.
* Accept: pipeline never dies; `stats.extra["ocr"]` names the engine actually used. Size: M.

### F3-6 — furigana handling (both engines)
* Files: `glasstranslate/ocr/furigana.py` (new pure geometric rule over `Segment`s: small boxes adjacent to a taller vertical column, height/width ratio thresholds); `render/layout.py` (`_mark_furigana` delegates, keeps kana text test as a second signal); apply in the non-manga path too.
* Tests first (`tests/test_furigana_filter.py`): `test_small_kana_column_beside_a_tall_column_is_furigana`, `test_same_size_kana_column_is_not_furigana`, `test_left_side_furigana_is_detected`, `test_horizontal_ruby_above_a_line_is_detected`, `test_filter_is_a_no_op_for_latin_text`, `test_manga_mode_grouping_is_unchanged`, `test_furigana_excluded_in_non_manga_mode`.
* Accept: `tests/test_pipeline_blocks.py` and `tests/test_erase.py` stay green. Size: M.

---

## D-1 — Docs, packaging, smoke test

* Files: the build notes (env vars `GEMINI_API_KEY`, `GLASSTRANSLATE_PROFILE`; commands `tools/profile_glass.py`, `tools/ocr_probe.py`); `docs/GLASS_DESIGN.md` (overlay exceptions `set_capture_excluded` + stored-affinity `showEvent`; `Misc` card; `GlassTextArea`; any F2 layer-stack change); `requirements.txt` (`requests`, `jaconv`, ORT runner if F3-1 passes); `packaging/GlassTranslate.spec` (hidden imports; **no** model `collect_data_files`; keep torch/paddle excludes); `packaging/smoke_test.py` (assert Misc card, Gemini card, series field, no key material in report).
* Tests first: `tests/test_glass_resources.py::test_check_digest_is_current`; `tests/test_frozen_paths.py::test_manga_ocr_models_dir_frozen_branch`; `tests/test_glass_qml.py::test_smoke_report_contains_no_secret_values`.
* Security: **yes**. Size: M.

---

## Risks and decisions

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Capture mode disables a capture-protection flag; user leaves it on during a screen share | HIGH | Runtime-only, explicit status warning, re-exclusion at teardown, security review on F1-1/F1-2 |
| R2 | Gemini key leaking into `config.json`, logs, smoke report, screenshots | CRITICAL | Key never enters `AppConfig`; env var or `secrets.json` under `%LOCALAPPDATA%`; `redact()` in log paths; password field shows only set/not-set; caplog + smoke-report tests |
| R3 | Pre-existing LibreTranslate `translation_api_key` **is** stored in `config.json` and shown on the Engines page | MEDIUM | Out of scope; recorded as follow-up (migrate into `secrets.py`) |
| R4 | F3 adds 74–460 MB of models and new imports to a 164 MB exe | HIGH | Runtime download; F3-1 picks size; no-go path degrades F3 to F3-6 |
| R5 | Per-line manga-ocr = N decoder runs per frame | HIGH | Latency measured in F3-1 vs `refresh_hz`; fallback: cache-miss lines only or manga mode only |
| R6 | `context` in cache key changes persisted schema | MEDIUM | Version bump to 2; v1 entries load as `context=""` |
| R7 | F2 fixes speculative until F2-0 | MEDIUM | Each fix conditional on a named hypothesis; clean-measuring hypotheses dropped |
| R8 | `Animator` types don't write back the property → pages permanently invisible | HIGH (silent) | Called out in F2-3 with a dedicated test |
| R9 | `control.py` already over 800 lines | MEDIUM | F4-5 pure-move extraction to `bridge_fields.py` |
| R10 | In-flight Gemini request blocks worker up to read timeout at shutdown | LOW | Bounded timeouts + cancel event; same accepted behaviour as LibreTranslate |
| R11 | New `.qml` files must pass the `qmlimportscanner` allowlist | MEDIUM | `GlassTextArea` wraps `T.TextArea`; `--check` is acceptance on every QML slice |

## Open questions for Gate 1

1. **F3 substitution** — accept ONNX manga-ocr on onnxruntime-directml as "manga-ocr primary" and rapidocr running PP-OCRv5/v6 as "PaddleOCR fallback"? *Recommendation: yes.*
2. **API-key storage** — `GEMINI_API_KEY` env var + `%LOCALAPPDATA%\GlassTranslate\secrets.json` vs Windows Credential Manager. *Recommendation: env var + file.*
3. **Capture mode scope** — overlay only (recommended) or overlay + control window?
4. **manga-ocr ONNX size** — fp16 / int8 / q4f16; defer to F3-1 measurements unless a hard ceiling exists.
5. **Default Gemini model** — `gemini-3.1-flash-lite` (proposed) vs `gemini-2.5-flash-lite` vs `gemini-3.8-flash`.
