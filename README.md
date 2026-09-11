# GlassTranslate

GlassTranslate is a Windows desktop app that puts a translucent "glass" window over any part
of your screen and live-translates the text underneath it. Whatever is below the glass is
captured, OCR'd, translated offline, and redrawn on top of the original text in the same
colour, size and orientation, so a manga page, a foreign web page or a game dialog reads as
if it had been written in your language.

The primary use case is reading Japanese manga in English: vertical speech-bubble text is
detected as vertical, translated, and re-laid horizontally along the bubble so it stays
readable. Any other pair with an offline model works the same way.

Everything runs locally: OCR on the GPU through DirectML, translation with CTranslate2
models (Argos packages, or Sugoi v4 for Japanese to English). No account, no cloud, unless
you opt into a LibreTranslate server.

## Stack and why

| Concern | Choice | Why |
|---|---|---|
| GUI | PySide6 (Qt 6) | Per-pixel alpha (`WA_TranslucentBackground`), always-on-top, frameless. Win32 `WS_EX_TRANSPARENT` makes the glass click-through and `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` hides it from its own screen capture. |
| Capture | `dxcam` (Desktop Duplication, GPU) primary, `mss` fallback | dxcam grabs a region in well under a millisecond once warm. mss is portable. |
| OCR | `rapidocr` (PP-OCR det/cls/rec via `onnxruntime-directml`) | Rotated quads plus a confidence per segment. DirectML runs on any DX12 GPU (about 3x faster than CPU on a small test). CPU fallback is the same package with the CPU provider. |
| Translation (offline) | Argos `.argosmodel` packages executed with `ctranslate2` + `sentencepiece` | Offline, tens of milliseconds per small batch on CPU. The model directory is loaded directly, so the heavy `argostranslate`/`stanza` stack is never imported. CUDA works after installing `requirements-cuda.txt`. |
| Translation (online, optional) | LibreTranslate-compatible HTTP API (`urllib`, no extra deps) | Pluggable, API key optional. |
| Language detection | `py3langid` plus Unicode-script heuristics | Fast and offline. Kana, Hangul, Cyrillic, Arabic, Greek, Thai and Hebrew are decided by script; Latin text goes to py3langid restricted to a whitelist. |
| Hotkeys | `pynput` global listener | Works while the glass is click-through and unfocused. |

See `docs/ARCHITECTURE.md` for the full module contracts.

## Install

Python 3.13 on Windows 10/11. From the project root:

```bat
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Optional, for CUDA-accelerated translation on an NVIDIA GPU (OCR uses DirectML regardless):

```bat
.venv\Scripts\python -m pip install -r requirements-cuda.txt
```

`glasstranslate.core.gpu.prepare_cuda_path()` adds the pip-installed CUDA DLL folders to the
DLL search path before CTranslate2 is imported, so no system-wide CUDA install is needed.

## Portable bundle (no installer, no Python, no network)

Two zips ship the whole thing for a machine that has nothing installed:

| Zip | Unpacks to | Holds |
|---|---|---|
| `GlassTranslate-<version>-portable-win64.zip` | `GlassTranslate-<version>\` | `GlassTranslate.exe` (torch-free onefile build), `renderer\glassrenderer.exe` + `_internal\` (the quality-renderer sidecar frozen with torch cu130, diffusers, transformers and the CUDA/cuDNN DLLs), `portable.txt`, `README-portable.txt`, `licenses\` |
| `GlassTranslate-<version>-models-win64.zip` | the same folder | `models\` — the complete quality store (24 pinned files, 9.3 GB), the manga-ocr ONNX bundle and the translation packs the build machine holds (Argos `ja_en` and `translate-en_de-1_3`, Sugoi `sugoi-v4-ja-en`), laid out as the app's model store; `models\MANIFEST.json` lists every file with its digest, each pack's licence and source, and the offered language pairs that are **not** covered. The bundle is built for personal use on the machine that built it: the Sugoi pack's licence forbids redistribution (see below), so pass `--no-research-models` to `build_portable.py` for a zip that will be shared |

Extract both into the same place and start `GlassTranslate.exe`. Each zip has a
`<zip>.sha256` next to it (`sha256sum -c` format).

**The portable rule.** When a file named `portable.txt` sits next to `GlassTranslate.exe`, the app,
its download workers and the sidecar keep everything under that folder instead of the user
profile: `models\`, `config\config.json`, `config\secrets.json`, `logs\glasstranslate.log`,
`logs\renderer.log`, and Qt's own QML compilation cache under `cache\qmlcache` (Qt's automatic
pipeline cache is turned off in portable mode, so nothing lands in `%LOCALAPPDATA%\GlassTranslate`
either). Mind that `config\secrets.json` then holds the Gemini key in plain text
inside a folder you may copy around: delete it before handing the folder on. `models_dir` is
stored relative to the folder (and confined to it), so the folder can be moved
or put on a USB stick and the settings still point at the right store. Without the marker
nothing changes (`%LOCALAPPDATA%\GlassTranslate` and `%APPDATA%\GlassTranslate\config.json` as
before). The rule lives in one place, `glasstranslate/config/settings.py` (`portable_root()`).
The marker also works next to `run.py` in a source checkout. One thing is outside the app's
control: Windows and the GPU driver keep their own shader caches in the profile
(`AppData\Local\D3DSCache`, `NVIDIA\DXCache`, `AMD\DxCache`, `NVIDIA\ComputeCache`) and the
NVIDIA driver drops an `NVIDIA Corporation\` folder next to the exe when DirectML or CUDA is
used; those are the OS's and the driver's, not GlassTranslate data, and the acceptance run
tolerates exactly those names and nothing else.

**Sidecar lookup order** (`render/quality.py find_sidecar_python`): `renderer\glassrenderer.exe`
next to the running exe, then the *Engines → sidecar interpreter* setting (an interpreter or a
`glassrenderer.exe`), then `renderer\.venv` in a checkout, then `%LOCALAPPDATA%\GlassTranslate\
renderer\.venv`. The frozen sidecar is launched as `glassrenderer.exe serve --models-dir …`;
token, loopback binding, stdin polling and the offline environment are the same as for the venv
form (`renderer/PROTOCOL.md`).

Building the bundle (developer machine, from the local stores, nothing is downloaded):

```bat
build.bat --no-test          :: dist\GlassTranslate.exe
build_renderer.bat           :: dist\renderer\  (PyInstaller onedir freeze of renderer\glassrenderer from renderer\.venv)
build_portable.bat           :: verifies every model digest, assembles and zips both artefacts, writes the .sha256 files
build_portable.bat --single  :: one dist\GlassTranslate-<v>-full-win64.zip (program + models, ~13.6 GB) instead of the pair
.venv\Scripts\python packaging\smoke_test.py --runs portable --portable-zip dist\GlassTranslate-<v>-portable-win64.zip --models-zip dist\GlassTranslate-<v>-models-win64.zip
```

`build_portable.bat --single` writes a single archive, `GlassTranslate-<version>-full-win64.zip`
(about 13.6 GB), with exactly the layout the pair unpacks to, for a one-download bundle; verify it
by unpacking and running the smoke with `--portable-dir <unpacked root>`.

The `portable` smoke run is the acceptance test: it unpacks both zips into a folder whose path
has a space and a non-ASCII character, strips `PATH` to System32, points `HTTP(S)_PROXY` at a
dead port so any network attempt fails, and checks the frozen sidecar (fake mode; the real
stages with `GT_GPU_TESTS=1`), the exe with `quality_renderer=auto` (runs static, A, B, C, the
sidecar probe, zero download attempts in the log, the quick-fill fallback with `renderer\`
removed), the end-to-end sidecar patch on a manga page (`GT_GPU_TESTS=1`) and a relocation of
the folder. Sizes and timings: `docs/perf/2026-09-11-portable-bundle.md`.

## Translation models

Models live in `models/` by default (configurable via `models_dir`). **No model is checked
into the repository**; a fresh clone shows untranslated text until you install one. Three
on-disk layouts are recognised:

* extracted Argos package directories (`metadata.json` + `model/` + `sentencepiece.model`),
* raw `*.argosmodel` zips (extracted on first use),
* plain CTranslate2 directories (`model.bin` + `config.json`) next to a hand-written
  `metadata.json` naming `from_code`, `to_code`, optional `source_spm` / `target_spm`
  (separate SentencePiece models) and an optional `priority`. This is how Hugging Face
  CTranslate2 conversions are installed. When several packages serve the same pair the
  highest priority wins.

### Japanese to English (manga): Sugoi v4

The generic Argos `ja -> en` package is trained on web text and falls apart on the short,
colloquial lines in speech bubbles (it returns `<unk>` for things like うるさい or やめろ).
Licence note: the CTranslate2 conversion the Engines page downloads
(`entai2965/sugoi-v4-ja-en-ctranslate2`) is published under NTT's research licence
(`license: other`, `ntt-license`): redistribution for research only and no commercial use.
The portable models zip therefore includes it only for personal use on the machine that built
the bundle; build a zip meant for others with `build_portable.py --no-research-models`, which
leaves the pack out and records that in `models\MANIFEST.json`.
Sugoi v4 is trained on game and visual-novel dialogue and handles them well. Install it with
the "Get Sugoi (ja→en)…" button in the control window, or from a shell:

```bat
.venv\Scripts\python -c "from glasstranslate.translate.argos import ArgosCT2Translator as A; print(A('models').download_sugoi())"
```

That fetches the CTranslate2 conversion from
`huggingface.co/entai2965/sugoi-v4-ja-en-ctranslate2` (about 700 MB) into
`models/sugoi-v4-ja-en/` with `priority: 10`, so it is preferred over the Argos package for
that pair. Hugging Face rate-limits bursts of downloads; the installer retries with backoff on
HTTP 429, but if it still fails wait a few minutes and run it again (already downloaded files
are skipped).

### Any other pair: Argos packages

To install a pair from the argospm index, either use the "Download model…" button in the
control window, or from a shell:

```bat
.venv\Scripts\python -c "from glasstranslate.translate.argos import ArgosCT2Translator as A; print(A('models').download_package('ja', 'en'))"
```

`download_package` sends a browser-like `User-Agent` (argos-net.com returns 403 otherwise),
tries every mirror listed in the index, downloads to a `.part` file, extracts the package and
rescans the directory. Use `available_packages()` on the same object to list what the index
offers. If `(src, tgt)` is not installed but `(src, "en")` and `("en", tgt)` are, the
translator pivots through English automatically.

## Running

```bat
.venv\Scripts\python run.py
:: or
.venv\Scripts\python -m glasstranslate
```

Both start the Qt application (`glasstranslate.ui.app.main`). Defaults are source
"Auto-detect", target English, Argos/Sugoi backend, GPU where available. Two environment
variables help with scripting: `GLASSTRANSLATE_CONFIG=<path>` overrides the settings file and
`GLASSTRANSLATE_AUTOEXIT_MS=<ms>` quits after that delay (used by the smoke tests).

### The two-window model

* **Control window**: a normal window with all settings (languages, OCR engine and device,
  translation backend, API URL/key for LibreTranslate, models directory and download button,
  overlay opacity, refresh rate, debounce, font, hotkeys, hide-original, manga mode,
  uppercase), Start/Stop, "Grab mode", "Show/Hide glass" and a live latency readout
  (per-stage ms, FPS, segments and blocks, cache hit rate, devices).
* **Glass overlay**: a frameless, always-on-top, click-through window. Whatever is under it is
  translated in place. In *grab mode* it stops being click-through and shows a border with
  handles: drag anywhere to move, drag the bottom-right corner to resize, press `Escape` or
  the hotkey to leave grab mode. Its geometry is persisted.

### Hotkeys (defaults from `config/settings.py`)

| Action | Default |
|---|---|
| Toggle grab mode | `Ctrl+Alt+G` |
| Toggle running (start/stop the pipeline) | `Ctrl+Alt+T` |
| Show/hide the glass | `Ctrl+Alt+H` |

Hotkeys are `pynput` strings such as `<ctrl>+<alt>+g` and can be edited in the control window.

### Settings file

Settings are a JSON file written on every change:

| Platform | Path |
|---|---|
| Windows | `%APPDATA%\GlassTranslate\config.json` |
| macOS | `~/Library/Application Support/GlassTranslate/config.json` |
| Linux | `$XDG_CONFIG_HOME/glasstranslate/config.json` (defaults to `~/.config`) |

A corrupt file is ignored and defaults are used. Notable defaults: source language `auto`,
target `de`, OCR device `auto` (DirectML, then CPU), translation backend `argos`, translate
device `auto` (CUDA if available, else CPU), overlay opacity 0.10, refresh 10 Hz, debounce
120 ms, minimum OCR confidence 0.5.

## Manga mode and typesetting

`manga_mode` (default on; "Manga mode (group bubbles, comic lettering)" in the control
window) switches the pipeline from one translation per OCR line to *blocks*, the way a
professional English edition is lettered. The same code path serves the live glass and the
`demo/typeset_page.py` script, so the demo output is a faithful preview of the overlay.

1. **Blocks** (`render/layout.py`, `build_blocks`). Raw OCR lines are first cleaned of
   furigana (small kana-only lines hugging a larger line are dropped from the text but still
   erased). Lines of the same orientation and glyph size that sit on the same patch of paper
   are grouped; vertical columns are read right to left, horizontal lines top to bottom, so a
   block's text is the whole utterance. That string is what gets translated, which is far
   better than translating each column on its own.
2. **Bubble detection**. The paper around the block is flood-filled with the text painted
   over. When the enclosing light (or dark) region is bubble-sized, the block is typeset
   inside the bubble's outline: the layout region is the bubble eroded by a margin, and the
   flow algorithm (`render/typeset.py`) uses the region's row spans so lines get shorter
   towards the top and bottom of an oval. Otherwise the block is *free text* over artwork or
   a caption on flat paper.
3. **Erase** (`render/erase.py`). A bubble or caption box is *redrawn*, never inpainted: its
   paper interior (the paper component holding the text, with every enclosed glyph, furigana
   mark or stray filled in) is painted with the paper colour, clipped to the outline, and the
   outline is copied from the source with its own thickness and tail. Over artwork only the
   glyph strokes are removed (connected components classified as glyphs, completed along the
   column) and filled with the local background, so screentone and line art between the
   letters survive; no white box is drawn. The optional quality renderer (below) then
   regenerates those free-text patches.
4. **Lettering**. The translation is upper-cased (`uppercase`, default on), flowed at the
   largest size at or below a ceiling derived from the source glyph height (so all blocks on
   a page share a scale), hyphenated only when a word cannot fit a line, and the lines are
   balanced so a bubble reads as a centred oval. Text over artwork gets a halo (a stroke in
   the background colour, 0.11 em wide) for legibility. Dialogue is set in italic;
   flat captions stay upright.
5. **Live glass**. The pipeline emits one segment per block (`style.is_block`), carrying the
   layout region, the erased patch and the halo flag. The overlay paints the patch at its
   frame position, then runs the same `typeset()` with a `QFontMetricsF` measurer of the
   same font, scaled by the device pixel ratio, so placement matches the PIL demo output.
   Change detection works on block footprints: a dirty tile touching a block drops it and
   re-OCRs a crop grown to the whole bubble plus 64 px, so bubbles are never read in halves.
   Unknown-token markers a weak model emits (`<unk>`, `⁇`) are stripped; a translation
   that is nothing but markers shows the source text instead.

With `manga_mode` off, every OCR line is styled and translated on its own and drawn in the
configured font family as before (vertical columns rotated 90 degrees).

### Fonts

Blocks are lettered in **Anime Ace 2.0 BB** (Regular and Italic) by Nate Piekos / Blambot,
bundled in `glasstranslate/render/fonts/animeace/`. It is freeware for independent comic and
non-profit use only and may not be redistributed without the author's permission (see
`font info.txt` alongside the files and https://blambot.com). The demo uses it through PIL,
the overlay registers it with `QFontDatabase` at startup.
The "Font" setting only affects non-block segments.

### Quality renderer (optional sidecar)

Free text over artwork can be regenerated by a **generative sidecar** so the page looks as if
the Japanese was never there: a manga-tuned LaMa pass continues screentone, hatching and lines
under the erased glyphs, then an SDXL img2img pass (Illustrious-XL, line-art ControlNet
conditioned on the surrounding drawing with the text blanked, whole panel as context at up to
1024 px, fp16) sharpens the fill back to ink weight. Only the masked region is composited back,
under a feathered edge; every other pixel stays byte-identical. Bubbles are never sent (they are
redrawn in the app).

The sidecar is a separate process with its own venv (`renderer/`, package `glassrenderer`:
torch + CUDA, diffusers), so the app and the exe stay torch-free. It listens on
`127.0.0.1` only, with a per-session token in its environment, and never touches the network
at run time (models are pinned by revision and sha256, downloaded once by the app into
`%LOCALAPPDATA%\GlassTranslate\models\quality\`, 9.3 GB; licences in `renderer/MODELS.md`).

1. Install: `renderer\install.bat` (needs Python 3.13 and an NVIDIA GPU; ~16 GB VRAM
   recommended, 10.7 GB used with the automatic VAE tiling).
2. Engines tab, card *Quality renderer*: "Download quality models…", then set the mode to
   *On when available*. The app launches the sidecar when the pipeline starts; while its models
   load the quick fill renders as before, and each free-text patch is swapped for the generated
   one a few seconds later (the status strip shows `Quality renderer: ready on cuda:0`).
3. Without the sidecar, the models or a GPU the setting is harmless: the quick fill stays.

Contract and security notes: `renderer/PROTOCOL.md`; timings: `docs/perf/2026-09-11-quality-renderer-timings.md`.

### Typesetting a page from the command line

```bat
.venv\Scripts\python demo\typeset_page.py Examples\before.jpg --out demo\output
```

Writes `demo/output/before_typeset.png` (the lettered page) and `before_blocks.png` (source
lines in green, furigana in grey, bubble regions in blue, layout boxes in red, block index
with `B` for bubble and `O` for outlined free text). `Examples/after.webp` is a professional
English edition of the same page for comparison. Flags: `--backend argos|identity`,
`--device`, `--src`, `--tgt`, `--models`, `--min-confidence`, `--no-upper` (keep the
translation's case).

### Measuring typesetting against lettered references

`more_comparisons/<n>ja.jpg` / `<n>eng.jpg` are Japanese pages with their official English
lettering (plus `Examples/before.jpg` / `after.webp`). Three scripts turn them into numbers:

```bat
.venv\Scripts\python demo\typeset_reference.py --batch --crops
.venv\Scripts\python demo\typeset_dev.py --batch --ref-text --tag base --quiet
.venv\Scripts\python demo\typeset_metrics.py --batch --tag base --quiet
```

`typeset_reference.py` aligns each English page onto its Japanese page (ORB + ECC homography)
and derives the letterer's erase mask, the kept art, the English lettering ink and per-block
lettering statistics under `demo/reference/<stem>/` (the English wording lives in
`demo/reference_text_<stem>.json`, keyed by the OCR block text). `typeset_dev.py` renders any
page with cached OCR and translations (`--ref-text` letters with the reference wording,
`--quality` / `--quality-fake` run the sidecar, `--quality-params` tunes it) and writes
greyscale comparison sheets. `typeset_metrics.py` scores a render: erase IoU / recall /
precision, art kept, SSIM and MAE in the erased regions against the reference, bubble residue,
cap-height ratio, inset, centring, line counts, overflow and collisions, per page and per
block. Baseline and failure catalogue: `docs/perf/2026-09-10-typesetting-baseline.md`.

## Demo (no GUI needed)

The demo runs the whole chain on static images and writes annotated results.

```bat
.venv\Scripts\python demo\make_images.py
.venv\Scripts\python demo\run_demo.py --tgt en --backend argos --device gpu
```

`make_images.py` renders synthetic screenshots into `demo/images/`: a web page (white header
on blue, black body text in two sizes, grey caption), a "rotated" sheet (lines at 0°, +20°,
-15° and a 90° vertical line on light and dark backgrounds) and a dark UI panel with an orange
button. `demo/images/manga.png` is a hand-made manga panel: three speech bubbles with
vertical Japanese in one and two columns, a horizontal bubble, and white and red text on a
dark panel. Run the demo without `--src` to exercise auto-detection (the manga panel detects
as `ja`, the others as `en`).

`run_demo.py` OCRs each image with RapidOCR, measures colours and angle per segment, detects
the language (skipped when `--src` is given), translates with Argos if `models/` holds a
package for the pair (otherwise it warns and uses the identity translator), composes the
overlay with PIL and writes `demo/output/<name>_overlay.png` and `<name>_debug.png` (quads
outlined in their measured foreground colour with angle and fg/bg hex codes). It ends with a
per-image table of OCR, style, translate and compose milliseconds.

Flags: `--src`, `--tgt`, `--backend argos|identity`, `--device auto|cpu|gpu` (gpu selects
DirectML for OCR and CUDA for Argos), `--images DIR`, `--out DIR`, `--models DIR`, `--font`,
`--min-confidence`.

Typical timings on an RTX 4080 machine, warm, per 1000x700 image with 4 to 6 segments:

| Stage | GPU | CPU |
|---|---|---|
| OCR | 130 to 210 ms | 185 to 250 ms |
| Style (k-means per segment) | 3 to 12 ms | 3 to 12 ms |
| Translate (batch) | 40 to 70 ms | 28 to 55 ms |
| Compose (PIL) | 14 to 20 ms | 14 to 20 ms |

The first Argos call includes model load (about 400 ms on CUDA, 250 ms on CPU). For batches
this small the CPU is as fast as CUDA.

## Tests

```bat
.venv\Scripts\python -m pytest -q
```

Tests need no GPU, models, network or display. They cover colour and angle measurement,
text fitting, the translation cache and factory, change detection, language detection, the
manga-mode pipeline (block emission, dirty-rect growth, unknown-token cleanup, config
persistence; `tests/test_pipeline_blocks.py`), the eraser and bubble redraw, the quality
renderer's client, scheduler, patch swap and model store (against an in-test fake sidecar) and
the demo helpers.

The sidecar has its own suite, run separately (both trees have a package named `tests`):

```bat
.venv\Scripts\python -m pytest renderer\tests -q
set GT_GPU_TESTS=1 && renderer\.venv\Scripts\python -m pytest renderer\tests\test_stages_gpu.py -q
```

The first runs in the app venv without torch (protocol, compositing, fake pipeline, the real
HTTP server in `--fake` mode); the second is the opt-in GPU integration test.

## How to add a translation backend

1. Subclass `glasstranslate.core.interfaces.Translator`. Set the class attributes `name`,
   `device` and `offline`, and implement `supported_pairs()` (iterable of `(src, tgt)`
   ISO-639-1 pairs) and `translate_batch(texts, src, tgt)`, which must return one string per
   input and return the inputs unchanged for anything it cannot translate instead of raising.
   `translate_batch` is called from a worker thread, so keep it thread-safe. Override
   `supports()` if your pair set cannot be enumerated (see `translate/identity.py`) and
   `close()` if you hold resources.
2. Register it in `glasstranslate/translate/factory.py`. Registration is explicit: add your
   backend name to the `_BACKENDS` tuple (this is what `available_backends()` returns and what
   the control window lists) and add an `if backend == "yourname":` branch in
   `create_translator(cfg)` that builds your class from `AppConfig` fields (`models_dir`,
   `translate_device`, `translation_api_url`, `translation_api_key`). Unknown names fall back
   to the identity translator with a warning.
3. Re-export it from `glasstranslate/translate/__init__.py` and add a test in `tests/` that
   exercises it without network or models (monkeypatch any I/O, as `tests/test_cache.py` does
   for LibreTranslate).

Shipped backends: `argos` (offline, CTranslate2), `libretranslate` (online), `identity`
(pass-through, for testing the overlay without models).

## Known limitations

* **Windows-only click-through and capture exclusion.** The overlay relies on the Win32
  extended styles `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE`
  and on `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` (Windows 10 2004 or later) so the
  glass never captures itself. On X11 and macOS click-through can be approximated with
  `Qt.WindowTransparentForInput`, but there is no capture-exclusion equivalent, so the
  overlay would see its own translations. Those platforms are not supported for the live
  overlay. The library packages and the demo run anywhere Python, OpenCV, PIL and onnxruntime
  do (use `onnxruntime` instead of `onnxruntime-directml`, and `mss` instead of `dxcam`).
* **dxcam caches one camera per GPU output.** Hold a single capture instance and call
  `close()` before creating another, or dxcam returns the existing camera with a warning.
  dxcam maps monitors through a private attribute of its factory. If that disappears in a
  future release the code falls back to `EnumDisplayMonitors` order.
* **OCR is the bottleneck.** RapidOCR takes 100 to 250 ms per full frame. The pipeline
  therefore only OCRs dirty regions and debounces during scrolling or video.
* **CPU can beat CUDA for translation.** Argos models are small, so kernel launch overhead
  dominates on the GPU for the short batches an overlay produces. `translate_device: cpu` is
  a reasonable choice.
* **Fit, not overflow.** Translations longer than the original are wrapped (up to 3 lines)
  and shrunk to fit the original quad, down to a minimum size, rather than spilling over
  neighbouring text. Very short boxes with long translations become small.
* **Vertical text outside manga mode.** A vertical segment whose translation is CJK is
  rendered as a column of stacked characters. When the translation is a spaced script the
  text is laid out horizontally along the column, wrapped to the column height in up to six
  lines, and rotated 90° clockwise so it reads top to bottom. Columns are translated
  independently; enable manga mode to translate a bubble as one utterance.
* **Manga mode heuristics.** Bubble detection is a flood fill of the paper, so a bubble
  whose outline is broken, or text that touches the outline, is treated as free text. Blocks
  are grouped by proximity and glyph size; two bubbles that touch can merge into one block.
  Stroke-only erasing over dense artwork can leave faint traces of the original glyphs.
* **Short interjections** (single-word shouts, sound effects) are where translation models
  are weakest. When a model returns nothing usable (empty or all `<unk>`), the original text
  is shown instead of a blank box, and the failure is not cached so a better model installed
  later takes effect immediately.
* **Language detection needs at least three letters** and picks the dominant language of the
  whole page. Mixed-language pages translate as one source language unless a segment is
  unmistakably another script.
