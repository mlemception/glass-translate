# GlassTranslate quality renderer — sidecar protocol (v1)

The sidecar is a separate local process (`renderer/`, package `glassrenderer`, its own venv with
torch + CUDA + diffusers) that the app launches on demand.  The main app stays torch-free.  This
file is the contract between `glasstranslate/render/quality.py` (client) and
`glassrenderer/server.py` (server).  Both sides ship tests against it; the fake-model mode makes
the whole protocol testable without a GPU.

## Process

```
<sidecar python> -m glassrenderer serve --models-dir <dir> [--fake] [--device cuda|cpu] [--port 0]
```

* Environment `GT_RENDERER_TOKEN` carries the per-session token (32+ random bytes, hex).  It is
  never passed on the command line (visible in process lists).  Without it the server refuses to
  start.
* The server binds **127.0.0.1 only** (never `0.0.0.0`), on `--port` (default 0 = OS-chosen).
* Once listening it prints exactly one line `READY <port>` to **stdout** and flushes; everything
  else (logs, warnings) goes to stderr.  The client waits for that line (30 s in fake mode,
  120 s in real mode: model loading happens *after* READY, see `/health`).
* `--fake`: no torch import at all; the stages are numpy stand-ins (see below).
* The process exits on `POST /shutdown`, on stdin EOF (the parent died), or on SIGTERM.

## HTTP

HTTP/1.1, JSON request and response bodies (`Content-Type: application/json`), UTF-8.

Every request must carry `X-GT-Token: <token>`; the server compares it in constant time and
answers `401 {"error": "unauthorized"}` otherwise.  Requests whose `Host` header is not
`127.0.0.1[:port]` or `localhost[:port]` get `403`.  Bodies above 64 MiB get `413`.  Unknown
paths `404`.  Malformed JSON or schema `400 {"error": "<what>"}`.  Stage failures
`500 {"error": "<type>: <message>"}` (never a traceback).

Images travel as base64-encoded **PNG** strings: `image` is RGB 8-bit (the client converts from
its BGR frames), `mask` is 8-bit greyscale where **255 = fill** (erase and regenerate) and 0 =
keep; both have the same width and height.

### `GET /health`

```json
{"ok": true, "version": "1", "mode": "fake" | "real", "device": "cuda:0" | "cpu",
 "models": {"lama": "ready" | "loading" | "missing", "sdxl": "…", "controlnet": "…"},
 "warm": false, "vram_total_mb": 16376, "vram_used_mb": 9100, "uptime_s": 12.3}
```

`models.*` is `"missing"` when a model file is absent from `--models-dir` (the app downloads them
through the Engines page first), `"loading"` while it is being loaded, `"ready"` after.  `warm` is
true once one real job has run (CUDA kernels compiled).  In fake mode every model is `"ready"`.
`ok` only says the server is alive: a client that needs the models reads `models.*` (all
`"ready"`) before submitting jobs.

### `POST /inpaint`

```json
{"job_id": "p3-1a2b3c", "image": "<base64 png>", "mask": "<base64 png>",
 "params": {"lama": true, "sdxl": true, "strength": 0.4, "steps": 24,
            "controlnet_scale": 0.8, "guidance": 4.0, "seed": 0, "feather_px": 2,
            "prompt": "", "negative_prompt": ""}}
```

All `params` are optional (defaults above; `prompt`/`negative_prompt` extend the built-in monochrome
manga prompts).  A key that is not one of these ten is a `400` (a client typo must never silently
lose a setting); the client therefore sends only keys it knows.  Response:

```json
{"job_id": "p3-1a2b3c", "image": "<base64 png>",
 "timings_ms": {"decode": 3.1, "lama": 91.0, "sdxl": 2850.0, "composite": 4.0, "total": 2960.0},
 "mode": "real", "stages": ["lama", "sdxl"], "size": [1024, 768]}
```

The result has the input's size.  **Every pixel where the mask is 0 is byte-identical to the
input**; the server composites its output back under a `feather_px` feathered mask edge, and the
client composites again as a guarantee.  The server may process a downscaled or padded copy
internally (panels are bucketed to 512 / 768 / 1024 px long side for cuFFT plan reuse) but
resamples back to the input size before compositing.

Fake mode: the "LaMa" stage fills the mask with the median colour of the mask's 6 px surrounding
ring (a numpy stand-in that exercises the same code paths), the "SDXL" stage returns its input.
Timings are reported as measured.

### `POST /shutdown`

`{"ok": true}`, then the process exits within a second.

## Stage interface (inside the sidecar)

```python
class Stage(Protocol):
    name: str
    def run(self, image_rgb: np.ndarray, mask: np.ndarray, params: dict) -> np.ndarray: ...
    # image_rgb uint8 HxWx3, mask uint8 HxW (255 = fill); returns uint8 HxWx3 of the same size
def build_stages(models_dir: Path, device: str, *, fake: bool) -> tuple[Stage, Stage]  # (lama, sdxl)
```

`glassrenderer/pipeline.py` runs `lama` on the panel with the mask, then `sdxl` on the LaMa result
(img2img at `strength`, line-art ControlNet conditioned on the input with the mask region
blanked), then `compositing.composite(input, output, mask, feather_px)`.

## Models

Files are downloaded by the **app** (Engines page progress row, `%LOCALAPPDATA%\GlassTranslate\
models\quality\` in the frozen build, `<project>/models/quality/` in a checkout) from the pinned
Hugging Face revisions listed in `renderer/MODELS.md` with the sha256 / size table shared by
`glasstranslate/render/quality_models.py` and `glassrenderer/models.py` (a test keeps the two
tables identical).  The sidecar never downloads anything itself.

## Security summary

Loopback only; per-session random token in the environment; constant-time compare; `Host`
check; body size cap; no filesystem paths in requests (images are inline); no data leaves the
machine; the sidecar has no network access requirement at run time.
