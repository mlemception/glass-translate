# Post-batch hardening pass — 2026-09-10

Staged plan for the eight-step request that followed the feature batch (commits `19ca771` /
`7d837b1` / `9eed069`).  Tier: **large** (packaging + secrets trigger + cross-cutting refactor), so the
full pipeline ran: intake → plan (auto-approved by the user in the request) → TDD slices → reviews →
**GATE (commit) — the only stop**.  Status: **implemented, awaiting the commit gate**.

## Plan (as executed)

| # | Step | Outcome |
|---|---|---|
| 1 | Rebuild the exe with `build.bat`, run the smoke test; verify hidden imports, certifi, secret-marker check | Exe 164.6 MB in 81 s, smoke **72/72 PASS**; every new hidden import + `requests`/`certifi`/`jaconv` and `certifi\cacert.pem` present in the PyInstaller TOCs; run B secret-marker check PASS |
| 2 | Live-verify Gemini with the key / base URLs / model ids from `Secret/` (never as env vars; an in-app place to enter them) | `GeminiTranslator` matrix: all **4 relay hosts × native format × 3 model ids translate** (1.4–3.8 s per 3-line batch); the OpenAI-compatible path returns 404 "model not found" on those relays; Google's own endpoint rejects the key (400 INVALID_ARGUMENT — it is a relay key). Context prompt call OK. The key is stored through the app's own store (`%LOCALAPPDATA%\GlassTranslate\secrets.json`), base URL / API format / model are Engines-page fields; the Model row became **free text** (`geminiModelField`) because two of the three ids were not in the preset combo. `Secret/` is now gitignored. |
| 3 | First-run manga-ocr download on a clean `%LOCALAPPDATA%` — dev and frozen | **Dev**: fallback line `OCR: mangaocr failed (FileNotFoundError); using paddleocr`, download 201.6 MB in 142 s with a 0→100 % progress row, `models_changed` emitted, rebuilt chain reports `mangaocr on gpu`. **Frozen**: new smoke run **D** (`GLASSTRANSLATE_SMOKE_ACTIONS=downloadMangaOcr` whitelist hook in `SmokeReport`, clean `LOCALAPPDATA` inside the sandbox) found a real defect — see below. |
| 4 | Split `ui/control.py` (1055) and `core/pipeline.py` (819) preserving behaviour | Pure moves: `ui/control_bridge.py` (700), `ui/control_window.py` (279), `ui/resources_guard.py` (72), `ui/control.py` façade (69) re-exporting the old surface; `core/engines.py` (75) + `core/segments.py` (92), `core/pipeline.py` 695 with re-imports. Three tests now patch/import the concrete modules; 606 tests green, pyflakes clean. |
| 5 | F2 follow-ups: spring damping decision; the ~0.45 s post-drag render tail | Springs: `Theme.springLead/springTrail` → 8.0/0.50 + 6.0/0.45 (settle 0.55 s, ~1 % overshoot; more damping alone did not help). Tail: environmental — ink-polarity flips driven by backdrop frames from an animating desktop behind the test window (≈ 0.33 s of full-rate rendering per flip); over a static desktop the loop is idle once the springs stop. Details in `docs/perf/2026-09-09-glass-baseline.md`. |
| 6 | Regression tests; targeted, smoke and full runs | `tests/test_smoke_actions.py` (3), Gemini model-field test updated; full suite 606; smoke runs A/B/C/D on the final exe |
| 7 | Reviews | 4 reviewers, 0 CRITICAL/HIGH, 3 MEDIUM + 4 LOW advisories — all actionable ones fixed (see Reviews) |
| 8 | Stop before committing | GATE |

## Defect found by run D (frozen exe only)

`ImportError: Can't connect to HTTPS URL because the SSL module is not available` inside the exe.
`packaging/GlassTranslate.spec` dropped `libcrypto-3.dll` / `libssl-3.dll` by basename, assuming CPython
shipped them as `libcrypto-3-x64.dll` / `libssl-3-x64.dll`; this Python 3.13 installs the un-suffixed
names, so `_ssl.pyd` could not load and **every https path in the exe was dead** (manga-ocr download,
Gemini provider, LibreTranslate over https) while dev runs worked.  Nobody noticed because the smoke
runs never made an https request.  Fixes:

- spec: OpenSSL names are dropped only under `PySide6/` (`PYSIDE_DROP`); the generic filter keeps CPython's.
- `MangaOcrDownloadWorker` catches any exception and reports `Download failed: …` (the ImportError was
  raised outside the caught types, the QThread died and the progress row sat at "Downloading…" forever).
- smoke report gains `ssl_ok` (`import ssl` succeeds); run A asserts it, so a future prune cannot
  silently remove https again.  Run D (opt-in) proves the download end to end in the frozen tree.

## F2 follow-ups

- **Spring damping** — grid measured on screen (see the perf doc's "Spring decision" table). Adopted
  `springLead 8.0/0.50`, `springTrail 6.0/0.45`: the render loop goes idle ≈ 0.55 s after a switch
  instead of ≈ 0.95 s, the indicator still visibly settles (≈ 1 % overshoot). Contract §2.3/§2.4 updated;
  `tests/test_glass_qml.py::test_theme_tab_springs_are_the_measured_2026_09_10_values` pins the values.
- **Render tail** — not a Qt-internal job and not drag-specific: the sampler window sat over an
  animating desktop; backdrop frames kept arriving and each ink-polarity flip runs the 180 ms `Theme`
  ink transitions plus a colour Behavior on every control (≈ 0.33 s of full-rate rendering, reproduced
  with synthetic frames and the grabber stopped). Over a static desktop swaps stop with the springs.
  Bisect record: hiding the bar/indicator/content, disabling all Behaviors, the basic render loop and
  `live:false` sources never changed the tail; `qt.quick.dirty` logging showed nothing dirty during
  the extra frames (animation-driver mode), and the flip experiment closed it.

## Reviews

Workflow of four reviewers over the saved diff (code, Python idiom
and security, plus a pure-move equivalence checker that compared every top-level
def/class body of the old `control.py` / `pipeline.py` against the new modules with `ast.dump` and a
comment-stripped text pass, and imported the facades to assert name identity). **No CRITICAL or HIGH
findings**, so the adversarial verification stage had nothing to run. Advisory findings and what was
done:

| Sev | Finding | Action |
|---|---|---|
| MEDIUM | `docs/GLASS_DESIGN.md` §2.3 still said "now ≈ 1.0 s" next to the updated §2.4 | fixed (≈ 0.55 s, points at §2.4) |
| MEDIUM | `control_bridge.py` imported `translate.gemini` (and so `requests`) at module level, against the lazy-backend convention | fixed: `geminiModelDefault/Presets` import lazily inside their getters |
| MEDIUM | free-text model id interpolated unencoded into the native `models/<id>:generateContent` path (same-host path manipulation only; the key stays in a header) | fixed: `urllib.parse.quote(model, safe="")` + `test_model_id_is_url_quoted_into_one_path_segment` |
| LOW | duplicate names in `GLASSTRANSLATE_SMOKE_ACTIONS` would connect the signal twice | fixed: de-duplicated while queuing |
| LOW | one blank line before `_Timer` after the extraction | fixed |
| LOW | `geminiModels` list property no longer used by QML | kept (still tested and documented) with a comment; a preset picker can reuse it |
| LOW | untracked dev probes under `demo/output/dev/` patch `control.MAIN_QML_URL` / `C.win32` through the façade | not changed (not in git); the façade re-exports `MAIN_QML_URL`, `win32` is now imported from `ui.glass` |

