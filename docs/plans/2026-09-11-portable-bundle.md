# Portable offline bundle: app exe + frozen sidecar + every model — 2026-09-11

**Large** tier (cross-cutting: config paths, sidecar launch, a second
PyInstaller build, two zip artefacts, a new acceptance run in the smoke test).  GATE 1 was waived
in the request ("the only stop is GATE 2").  Work split: research, then implementation on disjoint files, then review.

## Intake (restated)

Ship GlassTranslate as two zips that need no installer, no Python and no network on the target
machine:

* **(a)** `GlassTranslate-<version>-portable-win64.zip` → `GlassTranslate.exe` (the existing
  torch-free onefile build), `renderer/` (a PyInstaller **onedir** freeze of
  `renderer/glassrenderer` with torch cu130, diffusers, transformers, safetensors, the LaMa
  TorchScript loader and the CUDA/cuDNN DLLs; entry `renderer\glassrenderer.exe serve
  --models-dir …`, protocol unchanged), `portable.txt`, `README-portable.txt`, licence files.
* **(b)** `GlassTranslate-<version>-models-win64.zip` → `models/` with the complete quality store
  (24 pinned files, 9.31 GiB, verified against `quality_models.json`), the manga-ocr ONNX bundle
  and the Argos packs, laid out exactly as the app's store expects.  Built from the **local**
  stores after verifying every digest; nothing is downloaded during the build.
* **Portable rule**: `portable.txt` next to `GlassTranslate.exe` ⇒ models, config, logs and
  secrets live under that folder (`models/`, `config/`, `logs/`); without the marker nothing
  changes.  One place (`glasstranslate/config/settings.py`) that the app, the download workers and
  the sidecar launch all use.
* **App side**: the sidecar lookup accepts a frozen sidecar (`renderer\glassrenderer.exe` next to
  the running exe, then the config path, then `renderer\.venv` for dev) and launches it with the
  right argv; token, loopback, stdin polling and the offline env stay as they are.  The main exe
  stays torch-free (verified by scanning the built archive).
* **Build**: `build_renderer.bat`, `build_portable.bat`; sizes and timings in
  `docs/perf/2026-09-11-portable-bundle.md`.
* **Acceptance**: `packaging/smoke_test.py --runs portable` (details in § 4).

## 1. Decisions and assumptions

1. **Argos packs = what the local store holds.**  The app offers 18 languages
   (`ui/bridge_fields.py LANGUAGES`, every pair routed through English by Argos); the local store
   holds `ja_en/` (Argos 1.1), `translate-en_de-1_3/` (Argos 1.3) and `sugoi-v4-ja-en/` (priority
   10) plus the `translate-ja_en-1_1.argosmodel` archive of the first.  "Never re-download during
   the build" wins over "every pair": the models zip ships every pack the store has, the manifest
   and the build log list the pairs the app offers that are **not** covered, and the archive is
   skipped when its extracted directory is present (the app prefers directories anyway).
   **Licence facts found while building the index** (2026-09-11): the Sugoi v4 CTranslate2
   conversion (`entai2965/sugoi-v4-ja-en-ctranslate2`) is under NTT's research licence —
   research use only, no commercial use, not for redistribution.  The user confirmed mid-session
   that the zips are for **personal use on the building machine**, so Sugoi is **included by
   default** and its licence is recorded verbatim in `MANIFEST.json` and `LICENSES.md` § 3a
   together with a "do not redistribute" sentence; `build_portable.py --no-research-models`
   leaves it out for a bundle that will be shared (ja→en then falls back to the Argos `ja_en`
   pack, priority 0).  The Argos packs' index (`argosopentech/argospm-index`) states only the
   index repository's MIT/CC0 terms, not the packages'; they are recorded as "unknown - see
   <index URL>" rather than guessed.
2. **Zip layout**: both zips carry one top-level folder `GlassTranslate-<version>/` so "extract
   both here" yields one folder.  The models zip stores the multi-GB safetensors/ONNX/`model.bin`
   entries uncompressed (`ZIP_STORED`; they do not deflate) and deflates the rest; zip64 is on.
   Every zip gets a `<zip>.sha256` sidecar in `sha256sum` format.
3. **Frozen sidecar**: PyInstaller onedir, `console=True` (stdout `READY`, stdin polling need real
   std handles; the app spawns it with `CREATE_NO_WINDOW`), spec `packaging/glassrenderer.spec`,
   entry `packaging/glassrenderer_entry.py` (a package `__main__` cannot be frozen directly because
   of its relative imports).  `glassrenderer/models.json` is a data file next to its module.
   Output `dist/renderer/glassrenderer.exe` + `dist/renderer/_internal/`.
4. **Lookup order** (spec): `<app dir>\renderer\glassrenderer.exe` → `cfg.quality_sidecar_python`
   (an interpreter, or a `glassrenderer.exe`) → `<project>\renderer\.venv` (dev) →
   `<user data>\renderer\.venv`.  `find_sidecar_python` keeps its name and callers (it returns the
   launcher path, exe or interpreter); `QualityClient._spawn` builds the argv from the launcher's
   kind.  `<app dir>` is the frozen exe's directory, `project_root()` in a checkout.
5. **Relocation**: in portable mode `AppConfig.save` writes `models_dir` **relative** to the
   portable root when it lies inside it (`"models"`), and `AppConfig.load` resolves a relative
   `models_dir` against `user_data_dir()`; an absolute `models_dir` that no longer exists while
   the portable default does falls back to the default.  Moving the folder therefore needs no
   config edit.
6. **Secrets** go to `<root>\config\secrets.json` in portable mode (`config/` holds config and
   secrets); `GLASSTRANSLATE_SECRETS_FILE` still overrides.  **Qt's caches** (found by the
   final acceptance pass, 2026-09-11): Qt writes its QML disk cache and RHI pipeline cache to
   the shell-resolved cache location regardless of `%LOCALAPPDATA%`, so portable mode sets
   `QML_DISK_CACHE_PATH=<root>\cache\qmlcache` and `QSG_RHI_DISABLE_DISK_CACHE=1` before the
   QApplication exists (`settings.cache_dir()`, `app.apply_portable_qt_environment()`).
7. **Smoke report additions** (§ 5 of `docs/GLASS_DESIGN.md` is extended, the window is not
   touched): `quality` (setting, launcher, kind, models_ready, hint, scheduler state), `overlay`
   (`patch_conversions`, `patch_upgrades` from the existing patch cache), `paths` (portable root,
   config, models, logs), and two new whitelisted smoke actions: `probeSidecar` (launches the
   sidecar the app found **in fake mode** through the app's own lookup + spawn code and records
   `/health`; seconds, no GPU) and `feedPage` (`GLASSTRANSLATE_SMOKE_PAGE=<image>`: a static page
   replaces screen capture so the pipeline OCRs, typesets and — with the real sidecar — upgrades
   blocks; the app reports and quits once a block's `clean_patch_serial` advanced and the overlay
   re-converted it, or at the autoexit).
8. **Sidecar launch is lazy** (first job), so run (3) proves "sidecar ready" with `probeSidecar`
   (fake mode, seconds) and the `quality` report block; run (4) (opt-in GPU) proves the real
   stages end to end.
9. **PyInstaller's licence**: GPL-2.0 with the bootloader exception (a frozen program may be
   distributed under any licence).  The exe already relies on it; recorded in the licence index.
   The GPL gate applies to every distribution that ends up in a bundle.
10. `gh` is not installed on this machine, so the "GitHub code search first" step uses web search
    and the installed `pyinstaller-hooks-contrib` sources instead.

## 2. Contracts between the workstreams

### 2.1 `glasstranslate/config/settings.py` (I1 provides)

```python
PORTABLE_MARKER = "portable.txt"
def app_dir() -> Path              # Path(sys.executable).parent when frozen, else project_root()
def portable_root() -> Optional[Path]   # app_dir() iff app_dir()/PORTABLE_MARKER is a file
def is_portable() -> bool
def user_data_dir() -> Path        # portable_root() or %LOCALAPPDATA%\GlassTranslate (unchanged)
def default_config_path() -> Path  # <root>\config\config.json when portable, else unchanged
def default_models_dir() -> Path   # <root>\models when portable, else unchanged
def logs_dir() -> Path             # user_data_dir()/"logs"  (app.py, control_bridge, quality use it)
def secrets_dir() -> Path          # <root>\config when portable, else user_data_dir()
def resolve_models_dir(value: str) -> str  # relative → user_data_dir()/value; stale absolute → default
```

### 2.2 `glasstranslate/render/quality.py` (I1 provides)

```python
FROZEN_SIDECAR_EXE = "glassrenderer.exe"
def find_sidecar_python(cfg) -> Optional[Path]   # launcher (exe or interpreter) in the order of § 1.4
def sidecar_kind(launcher) -> str                # "frozen" | "venv"
def sidecar_command(launcher, models_dir, *, fake) -> List[str]
# QualityClient(launcher, models_dir, ...) unchanged signature; _spawn uses sidecar_command and,
# for a frozen launcher, cwd = launcher's directory; env/token/READY/stdin handling unchanged.
```

### 2.3 Smoke actions and report fields (I2 provides)

* `GLASSTRANSLATE_SMOKE_ACTIONS` whitelist: `downloadMangaOcr`, `probeSidecar`, `feedPage`.
* `GLASSTRANSLATE_SMOKE_PAGE=<absolute image path>` (used by `feedPage`).
* Report: `quality = {setting, launcher, kind, models_ready, hint, scheduler_active,
  scheduler_available, scheduler_stats}`, `overlay = {patch_conversions, patch_upgrades}`,
  `paths = {portable_root, config, models_dir, logs}`, `actions.probeSidecar = {launcher, kind,
  seconds, health, ok, error}`, `actions.feedPage = {page, started, finished, seconds, blocks,
  upgraded, overlay_upgrades, quality_stats, status}`.

### 2.4 Bundle layout (I3 provides, I2 consumes)

```
GlassTranslate-<version>-portable-win64.zip
  GlassTranslate-<version>/
    GlassTranslate.exe
    portable.txt                 # "GlassTranslate portable marker … version …"
    README-portable.txt
    licenses/LICENSES.md         # index: app + sidecar distributions, models (links to models/quality/*)
    licenses/<dist>/LICENSE…     # texts copied from the venvs' dist-info
    renderer/glassrenderer.exe
    renderer/_internal/…
GlassTranslate-<version>-models-win64.zip
  GlassTranslate-<version>/
    models/manga-ocr/…  models/quality/…  models/<argos pack dirs>/…  models/MANIFEST.json
dist/<zip>.sha256                # "<hex>  <zip name>"
```

### 2.5 Smoke CLI (I2 provides)

`packaging/smoke_test.py --runs portable --portable-zip <zip> --models-zip <zip>
[--portable-dir <already unpacked root>] [--keep]`; `GT_GPU_TESTS=1` enables checks (2) and (4).

## 3. Task list (thin vertical slices)

| # | Slice | Owner | Files |
|---|---|---|---|
| R1 | Freezing research: PyInstaller 6.22 + torch 2.14 cu130 pitfalls, hooks, metadata, DLL closure, prune list **measured** on a probe build (fake + real `/health`) | research 1 | scratchpad only |
| R2 | Store layouts, digests of the local stores, Argos pairs vs `LANGUAGES`, licence inventory + GPL scan of both venvs, log strings for the zero-download assertion | research 2 | scratchpad only |
| 1 | Portable path rule (tests first) + relocation-safe `models_dir` | I1 | `config/settings.py`, `config/secrets.py`, `tests/test_portable_paths.py`, `tests/test_frozen_paths.py`, `tests/test_config.py`, `tests/test_secrets.py` |
| 2 | Sidecar lookup order + launch argv (tests first); hint text | I1 | `render/quality.py`, `ui/control_bridge.py`, `tests/test_quality_client.py`, `tests/test_glass_bridge.py` |
| 3 | Smoke hooks: `probeSidecar`, `feedPage`, `quality`/`overlay`/`paths` report fields; static page capture; overlay cache counters; pipeline read-only accessor | I2 | `ui/app.py`, `ui/overlay.py`, `core/pipeline.py`, `capture/static.py`, `tests/test_smoke_actions.py`, `tests/test_overlay_cache.py` |
| 4 | Smoke `portable` run: unzip into a space + non-ASCII path, stripped PATH, dead proxies, decoy `LOCALAPPDATA`, checks (1)–(5) | I2 | `packaging/smoke_portable.py`, `packaging/smoke_test.py`, `tests/test_smoke_portable.py` |
| 5 | Sidecar spec + entry + `build_renderer.py/.bat` (measure, prune, closure check, fake self-check) | I3 | `packaging/glassrenderer.spec`, `packaging/glassrenderer_entry.py`, `build_renderer.py`, `build_renderer.bat`, `renderer/tests/test_frozen_entry.py` |
| 6 | Store verification + Argos discovery + manifest; torch-free archive scan; licence index; `build_portable.py/.bat` | I3 | `packaging/portable_store.py`, `packaging/verify_torch_free.py`, `packaging/licenses.py`, `packaging/README-portable.txt`, `build_portable.py`, `build_portable.bat`, `tests/test_portable_build.py` |
| 7 | Suites, exe build, renderer build, portable build, smoke `portable` (fake + GPU), perf doc | main loop | `docs/perf/2026-09-11-portable-bundle.md` |
| 8 | Reviews (code, security, build/packaging); fix CRITICAL/HIGH | — | — |
| 9 | Docs: README, PROTOCOL.md, renderer/README.md, GLASS_DESIGN §5/§6, the build notes | main loop | docs |
| 10 | GATE 2: diff summary + conventional commit messages | main loop | — |

## 4. Acceptance (the `portable` smoke run)

Fresh temp folder whose path contains a space and a non-ASCII character; both zips unpacked
there; `PATH` = System32 family only; no `PYTHON*`/`QT_*`/`VIRTUAL_ENV`; `HTTPS_PROXY` =
`HTTP_PROXY` = `http://127.0.0.1:9`; `LOCALAPPDATA` → an empty decoy folder in the sandbox that
must stay empty; **no** `GLASSTRANSLATE_CONFIG` (the portable config path is under test).

1. `renderer\glassrenderer.exe serve --fake` prints `READY <port>`, answers `/health` and one
   `/inpaint` (byte-identical outside the mask), exits on stdin EOF; cold-start seconds recorded.
2. (`GT_GPU_TESTS=1`) real `serve --models-dir <root>\models`: `/health` reaches all-`ready` with
   no download (stderr has no URL / download / proxy line), one job completes byte-identical
   outside the mask; READY, warm and first-job seconds recorded.
3. `GlassTranslate.exe` from the root with `quality_renderer=auto`, `running_on_start=true`,
   Argos: `static`, `A`, `B`, `C` pass; `actions.probeSidecar.ok`, `quality.kind == "frozen"`,
   `quality.launcher` under the root; `status_history` has `OCR: mangaocr on …`, the log has
   `Argos packages in <root>\models: […]` and no download / URL / proxy line; `paths.*` under
   the root; the decoy `LOCALAPPDATA` stays empty; with `renderer\` renamed away the same run has
   no `Engine error`/`Error:` and `quality.launcher` is null (quick fill).
4. (`GT_GPU_TESTS=1`) `feedPage` with `Examples/before.jpg`: `actions.feedPage.upgraded >= 1`,
   `overlay.patch_upgrades >= 1`, `status_history` has `Quality renderer: ready on cuda`.
5. Move the root to another space + non-ASCII path; rerun (1) and (3).

## 5. Status (end of session 9, 2026-09-11)

Every slice is implemented, built and verified; the work waits at **GATE 2** (nothing committed).

* Tests: app suite 865 passed / 1 skipped (was 696), sidecar suite 180 passed / 5 skipped
  (was 173); pyflakes clean on every touched module.  Final acceptance pass with the GPU
  checks on the final zips: 241 / 245, every functional row green; the four failures are the
  desktop-dependent tab-bar pixel rows.  The pass before it had exposed Qt's QML and pipeline
  caches landing in the profile (now relocated under `<root>\cache`, § 1.6).
* Builds: `dist/GlassTranslate.exe` 166.8 MB (torch-free, 1309 archive entries),
  `dist/renderer/` 2,921 MB (payload 3,253 → 2,840 MB pruned, closure check OK),
  `GlassTranslate-0.2.0-portable-win64.zip` 2,046 MB, `GlassTranslate-0.2.0-models-win64.zip`
  11,588 MB, both with `.sha256` sidecars (`docs/perf/2026-09-11-portable-bundle.md`).
* Acceptance (`packaging/smoke_test.py --runs portable`): the non-GPU pass and the GPU pass
  are recorded in the perf report.  The first GPU pass found two real defects that are now
  fixed and pinned by tests: `torch.jit.load(str)` cannot open a non-ASCII models path
  (`stages/lama.py` now hands torch a file object), and `feedPage` restarting the pipeline
  crashed the exe with two concurrent DirectML sessions (`Pipeline.replace_capture` swaps the
  capture in place).  The review round (three passes: code, security, build/packaging)
  raised 3 HIGH + 15 MEDIUM + 10 LOW; every HIGH and the correctness-relevant MEDIUM/LOW items
  were fixed (non-ASCII page reader, `.gitignore` anchoring, atomic zip writes, portable
  `models_dir` containment, sidecar env hygiene, torch-free gate widening, README template,
  probe watchdog, action bookkeeping, spec closure over delay imports, build preconditions);
  the rest is listed as advisories in `the build notes`.
* Decisions taken while building this: the
  bundle is for personal use, so the Sugoi pack ships by default (`--no-research-models`
  for a shareable build).
