# Portable bundle — sizes, cold start and first job, 2026-09-11

Machine: Windows 11 Pro 26200, RTX 4080 (16 GB), CUDA driver 591.86, NVMe.  Builds from
`build.bat --no-test`, `build_renderer.bat` and `build_portable.bat`; measurements from
`packaging/smoke_test.py --runs portable` (`res.info`) and the build logs under `build/`.
Plan: `docs/plans/2026-09-11-portable-bundle.md`.

_Status: final — every number comes from the builds recorded here
(`build/renderer-build.json`, `build/portable-build.json`) and the acceptance runs
(`build/smoke-portable.json`, `build/smoke-portable-final.json`)._

## 1. What is in the bundle

| Artefact | Content | Size on disk | Zip size |
|---|---|---|---|
| `GlassTranslate.exe` | onefile app (torch-free: 1309 archive entries, 148 top-level modules, verified by `packaging/verify_torch_free.py`) | 166.8 MB | — |
| `renderer\` | frozen sidecar, onedir (`glassrenderer.exe` 81.5 MB + `_internal\`, 5708 files) | 2,921.2 MB | — |
| `licenses\` | index + 34 app and 45 sidecar distribution licence texts | ~1 MB | — |
| `GlassTranslate-0.2.0-portable-win64.zip` | exe + renderer + `portable.txt` + README + licences (5897 entries) | 3,089.3 MB staged | **2,046.4 MB** |
| `models\quality` | 24 pinned files (`quality_models.json`), every sha256 re-verified | 9,993,412,420 B (9.31 GiB) | — |
| `models\manga-ocr` | 7 files (`ocr/models.py MANGA_OCR_FILES`), sha256 verified | 201.6 MB | — |
| Argos / Sugoi packs | `ja_en\` (7 files), `translate-en_de-1_3\` (8), `sugoi-v4-ja-en\` (8); 32 offered pairs uncovered (listed in `MANIFEST.json`) | 1,406 MB | — |
| `GlassTranslate-0.2.0-models-win64.zip` | the store above + `MANIFEST.json` (55 entries; big weights stored, small files deflated) | 11,601.1 MB | **11,588.4 MB** |
| `GlassTranslate-0.2.0-full-win64.zip` (`--single`) | the two halves in one archive, same layout (5953 entries = the union of the pair; 8 big weights stored) | 14,690.5 MB | **13,634.9 MB** |

`build_portable.py` timings: torch-free scan 0.2 s, hashing the store 8.7 s (NVMe, cached), staging
12.9 s, portable zip 119.2 s (deflate), models zip 19.0 s (stored), total 171.5 s
(`build/portable-build.json`).  `--single` (2026-09-11): hashing 8.0 s, staging 13.2 s, the full
zip 150.2 s, total 172.7 s; the pair from the earlier build is left untouched (same timestamps and
digests).  Verified by unpacking the full zip into a space + non-ASCII folder (20.9 s): one
top-level folder, `models/quality` 24 files, `models/manga-ocr` 7 files, every `MANIFEST.json`
row present at its recorded size, sha256 sidecar matching, then the `portable` smoke run with
`--portable-dir` on that root (verdict in § 4).

## 2. Freezing the sidecar (`build_renderer.py`)

| Step | Result |
|---|---|
| PyInstaller analysis + collect + prune + closure check | 239 s |
| Bundle before pruning (binaries + datas) | 3253.3 MB (203 binaries, 5522 datas) |
| Bundle after pruning | 2839.6 MB (193 binaries, 5514 datas; 5708 files, 2,921,178,974 B on disk) |
| Saved | 413.6 MB: `torch/lib` 379.9 (8 DLLs), `opencv_videoio_ffmpeg500_64.dll` 30.9, `torch/bin` 2.8, `torch/utils` 0.1 |
| pefile closure check (torch/lib load-time imports) | OK, 193 binaries |
| `serve --fake` self-check from a stripped env (READY / exit on stdin EOF) | 0.74 s / 0.39 s, no network line |
| `glassrenderer.exe` launcher | 81.5 MB |

What the research probe established before the spec was written (three probe builds, each
verified with a full real job from a space + non-ASCII path with `PATH` = System32 and dead
proxies): `torch/include` and `torch/lib/*.lib` are already dropped by PyInstaller's `hook-torch`;
the eight prunable `torch/lib` DLLs are `cudnn_adv64_9` (106.6 MB), `cusolverMg64_12` (95.4),
`nvrtc64_130_0.alt` (91.0), `curand64_10` (58.9), `nvperf_host` (27.8), `cufftw64_12`,
`libiompstubs5md` and `zlibwapi`, each removed and re-verified with a real job.  Two pieces of
common advice are wrong for this stack and are pinned in the spec's comments: `cupti64_2025.3.0.dll`
is a load-time import of `torch_cpu.dll` (pruning it breaks `import torch`) and
`cudnn_engines_precompiled64_9.dll` (222 MB) is required for every convolution.  Two must-adds
the hooks miss: `collect_dynamic_libs("torchvision")` (torchvision 0.29 renamed its op libraries to
`_C_stable.pyd` / `image_stable.pyd`, so `hook-torchvision` collects nothing and the real run dies
with `operator torchvision::nms does not exist`) and the whole `transformers` package tree
(`transformers/__init__` scans its model directory at import).  `optimize=2` is fatal
(transformers 5.x validates docstrings at import); `optimize=1` is used.  Freezing costs nothing at
run time: the venv sidecar warmed in 85 s with a 4.4 s first job and 1.39 s warm jobs; the frozen
one measured 108–130 s, 4.6–5.3 s and 1.37–1.42 s in the probe.

## 3. Sidecar cold start and first job (from the unpacked, relocated bundle; PATH = System32, dead proxies)

| Measurement | fake mode | real stages (`GT_GPU_TESTS=1`) |
|---|---|---|
| Process start → `READY` (first launch after unzip) | 0.84 s | 5.3 s |
| Process start → `READY` (second launch) | 0.18 s | — |
| Process start → `READY` after the folder was moved (two launches) | 0.19 s / 0.16 s | — |
| `READY` → every model loaded and `warm` (digest verification of 6.9 GB + warm-up, sidecar alone on the GPU) | — | 57.8 s (221 s in the pass where Defender scanned the freshly unpacked store for the first time) |
| First `/inpaint` (256×256 fake / 512×768 real) | < 0.1 s | 1,481 ms (5,500 ms in that first-scan pass) |
| Exit on stdin EOF (fake) / after `/shutdown` (real; CUDA teardown, see § 5) | < 10 s, code 0 | within the 45 s budget, `0xC0000409` |

Final pass (2026-09-11 14:45–14:58, `build/smoke-portable-final.json`): **241 / 245 checks**;
the four failing rows are the desktop-dependent `tab-bar row mean |diff| vs slab >= 6` pixel
check (4.0–5.0, documented in `docs/GLASS_DESIGN.md` § 6 and failing identically for the plain
exe on this desktop).  Every functional row is green, every decoy stayed empty apart from the
OS/driver caches, and the moved root holds `cache\qmlcache`.  The previous pass (242 / 245) had
failed only on the `USERPROFILE` decoy, where Qt's QML and pipeline caches had landed in
`AppData\Local\GlassTranslate\cache` — fixed by `apply_portable_qt_environment` (§ 5).

## 4. The app from the bundle (`portable` smoke run, 2026-09-11)

Sandbox `%TEMP%\gt portable ü_<id>\`, root `GlassTranslate-0.2.0\`, then moved to
`moved ünïcode\GlassTranslate-0.2.0\`; `PATH` = System32 family; `HTTP_PROXY` = `HTTPS_PROXY` =
`http://127.0.0.1:9`; `LOCALAPPDATA` / `APPDATA` / `USERPROFILE` → empty decoys; no
`GLASSTRANSLATE_CONFIG`.

| Check | Result |
|---|---|
| Unzip both zips (space + non-ASCII path) | portable 11.9 s, models 9.4 s; root 14,690 MB on disk |
| `probeSidecar` launch through the app's own lookup (`quality.kind == "frozen"`, launcher under the root) | P1 wall 14.4 s (exit 0); every `paths.*` under the root |
| Zero download attempts (`glasstranslate.log` scan), `Argos packages in <root>\models` present | pass |
| Decoy `%LOCALAPPDATA%` | empty; decoy `%APPDATA%` held only the NVIDIA driver's empty `ComputeCache` (see § 5) |
| static / A / B / C from the root (cold starts 3.0–3.3 s) | 72/74 — the two failing rows are the documented desktop-dependent tab-bar pixel check (4.1) |
| `renderer\` renamed away → quick fill, `quality.launcher` null, no error | P2 wall 12.7 s |
| Relocation: fake sidecar + P1 + static/A/B/C rerun from the moved root | pass (same two pixel rows), `config.json` carries `"models_dir": "models"` |
| `feedPage` end to end (`GT_GPU_TESTS=1`): first upgraded block on the glass (`Examples/before.jpg`, 9 blocks, 2 upgraded, 6 overlay re-conversions; sidecar spawned by the exe, app resident on the same GPU) | 55.9 s; the app reported and quit by itself, exit 73.4 s after launch (402.6 s / 509 s in the first-scan pass) |
| `probeSidecar` through the app's own lookup (fake mode) | 0.17 s, kind `frozen` |
| Total checks, first pass without the GPU checks | 218 / 224, the six failures being the pixel rows and the driver-cache rows above |
| Total checks, final pass with the GPU checks (final zips) | 241 / 245, the four failures being the pixel rows |
| `--single` archive, `--portable-dir` run on the unpacked root (no GPU) | 227 / 229 — the same rows as the pair's non-GPU rows (only the two intake rows differ by design: an existing root instead of two zips unpacked); the two failures are the pixel rows (C2 2.0, M-C2 4.6); fake sidecar READY 0.85 s / 0.19 s, exe cold start 2.9–3.8 s, relocation green |

## 5. Reading the numbers

* **The bundle is dominated by the sidecar's CUDA stack and the weights**, not by our code: of
  the 2.9 GB `renderer\` folder, `torch\lib` alone is about 2.4 GB after pruning (cuDNN 9's split
  engine libraries, cuBLAS/cuBLASLt, cuFFT, cuSPARSE, cuSOLVER, NVRTC and its builtins, nvJitLink);
  the 24 quality files are 9.31 GiB.  The portable zip deflates 3.09 GB to 2.05 GB in two
  minutes; the models zip is stored (the fp16 safetensors, the ONNX and CTranslate2 weights do
  not compress) and takes 19 s.
* **Freezing costs nothing at run time.**  Fake-mode `READY` from a stripped environment is
  0.2 s warm and about 1 s on the first launch after unpacking (Defender's first scan); the
  probe measured the real stages at 108–130 s to warm, 4.6–5.3 s for the first job and
  1.4 s per warm job, the same as the venv sidecar.
* **The first read of a freshly unpacked store is slow, and it is Defender, not the GPU.**  In
  the pass that unpacked the zips for the first time on this machine the real sidecar needed
  221 s to load and warm standalone and `SDXL pipeline ready` took 548–550 s when the exe had
  spawned it (first patch on the glass after 403 s); in the final pass, with byte-identical
  files that Defender had already scanned once, the same steps took 57.8 s standalone and
  55.9 s to the first patch with the app resident on the same 16 GB card, and warm jobs
  1.5–3.6 s.  The 6.9 GB checkpoint is read at roughly 13 MB/s while the real-time scanner
  looks at it for the first time, and `verify_file`'s SHA-256 plus `from_single_file` both read
  it.  For a user this means the first quality-renderer job after unpacking may take up to ten
  minutes; every later launch is a minute.  `README-portable.txt` says so.
* **Exit after a warm sidecar session takes 40–70 s** (autoexit → "exiting" in the log): the
  pipeline stops in ~2.5 s, then the scheduler finishes the job in flight, posts `/shutdown`,
  waits 5 s, kills, and the sidecar's CUDA teardown (14–19 s, `0xC0000409`) plus the client's
  bounded waits add up.  The smoke report is written before that teardown (the report's
  `aboutToQuit` slot is connected first) and the acceptance run now waits for the processes to
  end before it moves the folder.  Listed as a known advisory.
* **Two non-obvious teardown facts** are now pinned in the smoke run and worth knowing when
  reading `renderer.log`: after a real job the sidecar needs 14–19 s to exit on stdin EOF or
  `/shutdown` and leaves with `0xC0000409` (CUDA's `DLL_PROCESS_DETACH` under `os._exit`), so
  `QualityClient.shutdown()`'s 5 s wait ends in a kill of a process that is already exiting —
  harmless, nothing is on disk to flush, and the app's exit stays quick; and `/health`'s per-model
  `ready` only means "file present", so readiness is `warm == true` with `load_error == null`.
* **What the OS and the driver write is not ours, but Qt's cache was.**  The NVIDIA driver
  creates `NVIDIA Corporation\` next to the exe and `NVIDIA\ComputeCache` in `%APPDATA%`;
  Windows and the drivers keep `AppData\Local\D3DSCache`, `NVIDIA\DXCache` and `AMD\DxCache`
  under the profile.  The `USERPROFILE` decoy of the final pass also held
  `AppData\Local\GlassTranslate\cache\qmlcache` and a `qtpipelinecache-*` folder (about 6 MB):
  Qt resolves its cache location through the shell, not through `%LOCALAPPDATA%`, so the
  portable rule had missed it.  Fixed after that pass: in portable mode the app points Qt's QML
  disk cache at `<root>\cache\qmlcache` and turns the automatic RHI pipeline cache off before
  the QApplication exists (`ui/app.py apply_portable_qt_environment`), and the decoy check now
  tolerates exactly the OS/driver cache names and fails on anything else.
* **The three pixel rows** (`tab-bar row mean |diff| vs slab >= 6` at 4.1 in A, C2 and M-A) are
  the desktop-dependent artefact documented in `docs/GLASS_DESIGN.md` § 6 since 2026-09-10; they
  fail identically for the non-portable exe on this desktop.
