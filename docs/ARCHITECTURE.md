# GlassTranslate — architecture & module contracts

Read `glasstranslate/core/types.py` and `glasstranslate/core/interfaces.py` first; every
module builds on those dataclasses/ABCs. `glasstranslate/config/settings.py` is the
persisted `AppConfig`.

## Stack (decided, verified on this machine)
| Concern | Choice | Why |
|---|---|---|
| GUI | PySide6 (Qt 6) | per-pixel alpha (`WA_TranslucentBackground`), always-on-top, frameless; Win32 `WS_EX_TRANSPARENT` gives click-through; `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` hides the glass from its own capture (both verified by probe). |
| Capture | `dxcam` (Desktop Duplication, GPU) primary, `mss` fallback | dxcam gives ~1-3 ms grabs and returns `None` when nothing changed. mss is portable. |
| OCR | `rapidocr` (PP-OCRv6 det/cls/rec via onnxruntime-directml) | rotated quads + confidence per segment; DirectML runs on the RTX 4080 (71 ms vs 249 ms CPU on a small test). CPU fallback is the same package with the CPU provider. |
| Translation (offline) | Argos `.argosmodel` packages executed with `ctranslate2` + `sentencepiece` | offline, ~30 ms per small batch on CPU, CUDA works after `core.gpu.prepare_cuda_path()` (pip `nvidia-cublas-cu12`). No `argostranslate` dependency — we load the model dir directly, avoiding its stanza import. |
| Translation (online, optional) | LibreTranslate-compatible HTTP API (`urllib`, no extra deps) | pluggable, API-key optional. |
| Lang detect | `py3langid` + Unicode-script heuristics for CJK/Cyrillic/etc. | fast, offline. |
| Hotkeys | `pynput` global listener | works while the glass is click-through and unfocused. |

Python 3.13 venv at `.venv` (activate: `.venv\Scripts\python`). Run tests with `.venv\Scripts\python -m pytest -q`.

## Coordinate systems
* **Virtual screen physical pixels**: what capture backends use (`Rect`), and what `Frame.origin_x/y` refer to.
* **Frame pixels**: `Segment.quad` is in the coordinate system of the captured image (i.e. relative to `Frame.origin`).
* **Overlay logical pixels**: Qt widget coords. `physical = logical * devicePixelRatio`. The overlay must divide by DPR when painting frame-coord quads. The pipeline captures exactly the overlay's physical rectangle so frame (0,0) == overlay top-left.

## Public API per package (implement exactly these names)

### `glasstranslate/capture/`
* `backends.py`: `class DXCamCapture(ScreenCapture)` (`name="dxcam"`), `class MSSCapture(ScreenCapture)` (`name="mss"`). `grab(region: Rect) -> Frame | None`. DXCam: one `dxcam.create(output_color="BGR")` per instance, lazily; if `grab` returns None (no change) return the *last frame* re-stamped (the pipeline does its own diffing) — but return None only if we never got a frame. Handle regions crossing monitor bounds by clamping. Must be robust to the region changing between calls.
* `factory.py`: `create_capture(preferred: str = "auto") -> ScreenCapture` — try dxcam, fall back to mss, log which.
* `diff.py`: `class ChangeDetector`:
  * `__init__(self, tile: int = 64, threshold: float = 0.02, min_pixel_delta: int = 24)`
  * `update(self, frame_bgr: np.ndarray) -> list[Rect]` — compares with the previous frame (tile-wise: a tile is dirty if more than `threshold` fraction of its pixels differ by more than `min_pixel_delta` in any channel, computed on a downsampled grayscale for speed); returns merged dirty rectangles (union adjacent/overlapping dirty tiles into rects, in frame coords). First call → whole frame dirty. Size change → whole frame dirty.
  * `reset(self)`.
  * `fraction_changed` attribute (float, last call) so the pipeline can detect "video/scroll storm" and debounce.
  * Must run in <5 ms for 1920×1080 (use numpy, `cv2.resize` with INTER_AREA to tile grid).

### `glasstranslate/ocr/`
* `rapid.py`: `class RapidOCREngine(OCREngine)`; `__init__(self, device: str = "auto", min_confidence: float = 0.5)`. device auto → try DirectML (`RapidOCR(params={"EngineConfig.onnxruntime.use_dml": True})`) and set `self.device="gpu"`; on failure or `device="cpu"` use `RapidOCR()` with `use_dml: False`, `self.device="cpu"`. `recognize(img) -> list[Segment]`: result has `.boxes` (N×4×2 ndarray or None), `.txts`, `.scores`; filter by confidence; ensure quads are `float32 (4,2)` in the order returned (top-left first, clockwise). Silence RapidOCR's INFO logging (logging.getLogger("RapidOCR").setLevel(WARNING)) — it is very chatty.
* `factory.py`: `create_ocr(name: str, device: str, min_confidence: float) -> OCREngine`; `available_engines() -> list[str]` (currently `["rapidocr"]`).
* `langdetect.py`: `class ScriptLanguageDetector(LanguageDetector)`: fast Unicode-range heuristics (Hiragana/Katakana→`ja`, Hangul→`ko`, CJK-only→`zh`, Cyrillic→`ru`, Arabic→`ar`, Greek→`el`, Thai→`th`, Hebrew→`he`) then `py3langid` restricted to a configurable language set for Latin script; returns None for strings with <3 letters. Also `detect_dominant(texts: list[str]) -> str | None` (majority vote weighted by text length) — the pipeline uses this so all segments on one page share a source language unless a segment is unmistakably another script.

### `glasstranslate/render/`
* `style.py`:
  * `quad_angle_deg(quad) -> float`: angle of the text baseline (edge from point 0 to point 1) in degrees, CCW-positive in screen space where y grows downward means: `angle = -degrees(atan2(p1.y-p0.y, p1.x-p0.x))`. Normalise to `(-90, 90]`... **but** if the quad is taller than wide by >1.5× and the text has ≥2 chars, treat it as vertical text: return the angle and set `vertical=True` in `measure_style`.
  * `quad_text_height(quad) -> float`: length of the edge p0→p3 (short side).
  * `extract_colors(img_bgr, quad, pad: int = 2) -> tuple[RGB fg, RGB bg]`: crop the axis-aligned bbox (+pad), run 2-cluster k-means (`cv2.kmeans`, K=2, few iterations, on a subsample ≤ 4000 pixels) on RGB; the cluster with more pixels *touching the bbox border* is background, the other foreground. Fallback to (black, white) if the crop is empty or degenerate. Returns RGB tuples (not BGR!).
  * `measure_style(img_bgr, segment: Segment) -> SegmentStyle`.
* `fit.py` (pure, no Qt): `fit_text(text: str, box_w: float, box_h: float, measure: Callable[[str, float], tuple[float, float]], *, max_lines: int = 3, min_size: float = 6.0, start_size: float | None = None) -> FitResult` where `FitResult(size: float, lines: list[str])`. Start from `start_size` (default `box_h*0.85`), if the single line fits width → done; else try wrapping into up to `max_lines` lines (word-wrap; if a single word is wider than the box, break the word) at a size that fits `box_h/lines`; else shrink until it fits or hits `min_size`; never return an empty `lines`. `measure(text, size)` returns (width, height) — the caller supplies Qt- or PIL-based metrics. Also provide `pil_measurer(font_path_or_name) -> measure` in `fit.py`? No — put it in `compose.py` to keep fit pure.
* `compose.py` (PIL, for the demo and tests — no Qt): `compose(img_bgr, segments: list[TranslatedSegment], font_path: str | None, hide_original: bool = True) -> np.ndarray` draws each segment: fill the (rotated) quad polygon with `style.bg` (if hide_original), then render the translation using `fit_text` with a PIL measurer, rotated by `style.angle_deg` about the quad's centre (render to an RGBA layer, rotate with `Image.rotate(expand=True)`, paste at centre), colour `style.fg`. Vertical segments render each character stacked. Choose a default font with CJK coverage if available on Windows (`C:/Windows/Fonts/msyh.ttc` or `arial.ttf`), fall back to `ImageFont.load_default()`.

### `glasstranslate/translate/`
* `cache.py`: `class TranslationCache`: LRU keyed on `(text, src, tgt)`, `get(text, src, tgt) -> str | None`, `put(text, src, tgt, translation)`, `__len__`, `hits`/`misses` counters, `max_size` (default 5000), thread-safe (Lock). Optional `save(path)`/`load(path)` JSON persistence.
* `argos.py`: `class ArgosCT2Translator(Translator)` (`name="argos"`, `offline=True`). `__init__(models_dir: str | Path, device="auto", compute_type="auto")`: scans `models_dir` for extracted packages (dir containing `metadata.json` with `from_code`/`to_code` + `model/` + `sentencepiece.model`) **and** unextracted `*.argosmodel` zips (extract them on first use). Lazily loads a `ctranslate2.Translator` per pair. Call `glasstranslate.core.gpu.prepare_cuda_path()` before importing ctranslate2; device auto → cuda if `ctranslate2.get_cuda_device_count()>0` and creating the translator succeeds, else cpu. `translate_batch` tokenises with sentencepiece (`encode(out_type=str)`), `translate_batch(beam_size=2, max_decoding_length=256)`, decodes; empty/whitespace strings pass through. For pivoting: if `(src,tgt)` unavailable but `(src,"en")` and `("en",tgt)` are, pivot through English. Provide `available_packages() -> list[dict]` from the argospm index (`https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json`) and `download_package(from_code, to_code, progress_cb=None) -> Path` — **must send a browser-like `User-Agent` header** or argos-net.com returns 403.
* `libre.py`: `class LibreTranslateTranslator(Translator)` (`name="libretranslate"`, `offline=False`), `__init__(url, api_key="")`, POST `/translate` JSON with `q` (list), `source`, `target`, `api_key`; `supported_pairs` from `/languages` (cached); on any network error return inputs unchanged.
* `identity.py`: `class IdentityTranslator(Translator)` (`name="identity"`) returns inputs — useful for testing the overlay without models.
* `factory.py`: `create_translator(cfg: AppConfig) -> Translator`; `available_backends() -> list[str]`.

### `glasstranslate/core/pipeline.py`
`class Pipeline(threading.Thread)`: owns capture, `ChangeDetector`, OCR, detector, translator, cache. Constructor takes `cfg: AppConfig`, `region_provider: Callable[[], Rect]` (physical-pixel rect of the glass), `on_result: Callable[[list[TranslatedSegment], PipelineStats], None]`, `on_status: Callable[[str], None]`. Loop at `cfg.refresh_hz`: grab → diff → if no dirty tiles: emit stats with `skipped_unchanged=True` (cheap) and continue → debounce: if `fraction_changed > 0.4` (scroll/video) wait until two consecutive frames are stable (≤ `debounce_ms`) before OCR → OCR **only the dirty rects** (expand each dirty rect by 32 px, clamp, OCR that crop, offset quads back). Keep a `dict[int, TranslatedSegment]` of live segments; drop live segments whose bbox intersects a dirty rect, add new ones. → style → language detect (dominant, unless `cfg.source_lang != "auto"`) → translate with cache (batch the misses in one call) → `on_result(all live segments, stats)`. Methods: `start()`, `stop()`, `pause()/resume()`, `set_config(cfg)` (hot-swaps engines when engine fields change, in the worker thread), `invalidate()` (force full re-OCR, e.g. after moving the glass). Never let an exception kill the thread — catch, `on_status(str(e))`, sleep, continue.

### `glasstranslate/ui/`
* `overlay.py`: `class GlassOverlay(QWidget)`. Frameless, `Qt.Tool | WindowStaysOnTopHint`, `WA_TranslucentBackground`, `WA_ShowWithoutActivating`. After `show()`, on Windows apply `WS_EX_LAYERED|WS_EX_TRANSPARENT|WS_EX_TOOLWINDOW|WS_EX_NOACTIVATE` and `SetWindowDisplayAffinity(hwnd, 0x11)`. `set_click_through(bool)`; `set_grab_mode(bool)` toggles click-through off and draws a thin border + move/resize handles (drag anywhere to move, drag bottom-right corner to resize; `Escape` or the hotkey exits grab mode). `set_segments(list[TranslatedSegment])` (thread-safe via Signal), `set_background_opacity(float)`, `set_font_family(str)`, `set_hide_original(bool)`. `paintEvent`: fill with `QColor(0,0,0,int(255*opacity))` (or white? use black), then for each segment: `painter.save(); translate to quad centre; rotate(-angle)`; fill a rect of the quad's size with bg (if hide_original) else nothing; use `fit_text` with a `QFontMetricsF` measurer, draw lines in fg; `restore()`. Vertical segments: draw chars stacked. Divide frame coords by `devicePixelRatioF()`. Emit `geometry_changed` Signal when moved/resized. `physical_rect() -> Rect`. On non-Windows, click-through is done via `Qt.WindowTransparentForInput` flag (works on X11/macOS); capture exclusion isn't available there so the pipeline hides the overlay briefly? No: on non-Windows fall back to `setWindowOpacity(0)` for ~1 frame is too flickery — instead just document the limitation and skip.
* `control.py`: `class ControlWindow(QMainWindow)`: all controls listed in the task (source lang combo with Auto, target lang, OCR engine, OCR device, translation backend, API URL/key (enabled only for online backend), models dir + "Download model…" button, opacity slider, refresh Hz, debounce, font family (`QFontComboBox`), hotkey line-edits, hide-original checkbox, Start/Stop, "Grab mode" button, "Show/Hide glass"), and a live readout (`total ms`, per-stage ms, FPS, segments, cache hit rate, devices). Emits `config_changed(AppConfig)`. Saves config on every change (debounced 300 ms).
* `hotkeys.py`: `class HotkeyManager` wrapping `pynput.keyboard.GlobalHotKeys`, `bind(mapping: dict[str, Callable])`, `rebind`, `stop()`; callbacks marshalled to the Qt thread via a Signal.
* `app.py`: `main()` wires everything: load config → QApplication → ControlWindow + GlassOverlay → Pipeline → hotkeys. Overlay geometry persists.

### `glasstranslate/__main__.py`
`from glasstranslate.ui.app import main; main()` and `run.py` at project root doing the same, so `python -m glasstranslate` or `python run.py` works. Add `--demo` passthrough? No: demo is `demo/run_demo.py`.

### `demo/`
* `make_images.py`: generates `demo/images/*.png` with PIL: mixed font sizes, a rotated (~20°) line, a white-on-blue header bar, black-on-white body, coloured text on dark, and a vertical line.
* `run_demo.py`: for each image in `demo/images`: OCR → style → detect → translate (Argos if a model for the pair exists in `models/`, else `IdentityTranslator` with a warning) → `compose` → write `demo/output/<name>_overlay.png` plus a `<name>_debug.png` with quads/angles/colours annotated, and print a per-image latency table. Flags: `--src`, `--tgt`, `--backend`, `--device`.

### `tests/`
`test_style.py` (colors: synthetic white-on-blue & black-on-white boxes, rotated box angle within ±2°, vertical detection), `test_fit.py` (fit-to-box with a fake linear measurer: fits single line, wraps, shrinks, breaks long words, never empty), `test_cache.py` (hit/miss, LRU eviction, thread-safety smoke, persistence), `test_diff.py` (first frame all dirty, unchanged → empty, localized change → small rect containing the change, size change → full, fraction_changed), `test_langdetect.py`, `test_config.py` (roundtrip save/load, corrupt file → defaults). Tests must not require GPU, models, network or a display.
