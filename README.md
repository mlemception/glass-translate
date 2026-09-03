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
  overlay opacity, refresh rate, debounce, font, hotkeys, hide-original), Start/Stop, "Grab
  mode", "Show/Hide glass" and a live latency readout (per-stage ms, FPS, segments, cache hit
  rate, devices).
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
text fitting, the translation cache and factory, change detection, language detection and
the demo helpers.

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
* **Vertical text.** A vertical segment whose translation is CJK is rendered as a column of
  stacked characters. When the translation is a spaced script (Japanese manga to English) the
  text is laid out horizontally along the column, wrapped to the column height in up to six
  lines, and rotated 90° clockwise so it reads top to bottom. Wide bubbles with several
  columns are still one segment per column; the translations of adjacent columns are
  independent, so a sentence split across columns is translated in pieces.
* **Short interjections** (single-word shouts, sound effects) are where translation models
  are weakest. When a model returns nothing usable (empty or all `<unk>`), the original text
  is shown instead of a blank box, and the failure is not cached so a better model installed
  later takes effect immediately.
* **Language detection needs at least three letters** and picks the dominant language of the
  whole page. Mixed-language pages translate as one source language unless a segment is
  unmistakably another script.
