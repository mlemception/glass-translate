# GlassTranslate quality renderer (sidecar)

A separate local process that makes a translated manga panel look as if the Japanese had never
been on it.  Where the app's eraser can only redraw a bubble or flat-fill a patch of paper, this
process **regenerates** the erased region: LaMa restores the structure, SDXL img2img with a
line-art ControlNet restores the texture, and the result is composited back under a feathered
mask.  Every pixel outside the mask stays byte-identical to the input.

It is a sidecar because it needs torch, CUDA and diffusers, and the app must stay torch-free and
buildable as one PyInstaller exe.  The app launches it on demand, talks to it over JSON on
`127.0.0.1`, and works perfectly well without it: the normal quick fill renders in the usual pass
and a sidecar result later swaps the block's `clean_patch` through the existing patch cache.

## Layout

| Path | What it is |
|---|---|
| `PROTOCOL.md` | **The contract.**  Process, HTTP, schemas, security.  Read this first. |
| `glassrenderer/protocol.py` | The executable half of the contract: params, base64-PNG codec, validation |
| `glassrenderer/compositing.py` | The mask composite (the byte-identity guarantee) and the 512 / 768 / 1024 panel buckets |
| `glassrenderer/fake.py` | numpy stand-ins for the two GPU stages |
| `glassrenderer/pipeline.py` | `Renderer`: `/health`, and one job from bucket to composite |
| `glassrenderer/server.py` | The loopback HTTP front end |
| `glassrenderer/models.py`, `models.json` | The pinned model table (the **app** downloads the files) |
| `glassrenderer/stages/` | The real torch stages — sidecar venv only, imported lazily |
| `MODELS.md` | Model sources, revisions and licences |
| `tests/` | The fake-mode suite: no GPU, no torch |

## Install

```bat
renderer\install.bat
```

That creates `renderer\.venv` and installs `renderer\requirements.txt` (torch + CUDA, diffusers,
the rest).  It is a **second** venv: never install any of it into the app's `.venv`, and never
import torch, diffusers, paddlepaddle or manga-ocr-torchless from `glasstranslate/`.

The model files are not fetched here.  The app downloads them through the Engines page into
`%LOCALAPPDATA%\GlassTranslate\models\quality\` (`<project>\models\quality\` in a checkout); the
sidecar only reads them and has no network requirement at run time.  See `MODELS.md`.

## Run

The app spawns the sidecar itself.  To drive it by hand:

```bat
set PYTHONUTF8=1
set GT_RENDERER_TOKEN=<64 hex chars>
renderer\.venv\Scripts\python -m glassrenderer serve --models-dir models\quality --device auto
```

It prints exactly one line on stdout, `READY <port>`; everything else goes to stderr.

### selftest

`selftest` starts the server in-process, runs one synthetic job through the whole HTTP path and
prints the timings — the fastest way to tell "the sidecar works" from "the app cannot talk to
it".  In `--fake` mode it needs no GPU, no models and no sidecar venv, so the app's own
interpreter can run it:

```bat
set PYTHONUTF8=1
set PYTHONPATH=renderer
.venv\Scripts\python -m glassrenderer selftest --fake
```

```
mode    : fake  device: cpu
models  : {'lama': 'ready', 'sdxl': 'ready', 'controlnet': 'ready'}
stages  : ['lama', 'sdxl']  size: [768, 512]
  decode:       2.0 ms
    lama:      21.9 ms
    sdxl:       0.0 ms
composite:      65.8 ms
   total:     122.9 ms
```

Drop `--fake` (in the sidecar venv, with the models present) to time the real stages.

## Fake mode

`--fake` swaps the two GPU stages for numpy stand-ins and imports no torch at all: `FakeLama`
fills each connected mask component with the median colour of the 6 px ring around it and
`FakeSdxl` is the identity.  Everything else — the protocol, the bucketing, the compositing, the
server, the lifecycle — is the same code the real mode runs, which is why the whole contract is
testable on a laptop.

## Tests

```bat
set PYTHONUTF8=1
.venv\Scripts\python -m pytest renderer\tests -q
```

They run under the **app's** venv (numpy, OpenCV, Pillow, stdlib) and spawn the server as a
subprocess exactly as the app does, with `renderer\` on `PYTHONPATH`.

Run this suite **separately** from the app's `pytest`: both `tests/` directories are packages
named `tests`, so `pytest tests renderer/tests` in one process fails to import the second
`conftest.py`.  The repo's `pytest.ini` already pins `testpaths = tests`, so a bare `pytest` only
collects the app suite and the two never meet.

The GPU stages have their own opt-in test (`GT_GPU_TESTS=1`), which is skipped everywhere else.

## Protocol

`PROTOCOL.md` is the contract between `glasstranslate/render/quality.py` (client) and
`glassrenderer/server.py` (server); both sides ship tests against it.  In short:

* `GET /health` → mode, device, per-model status, warm flag, VRAM, uptime
* `POST /inpaint` → base64 PNG image + mask + params → base64 PNG result + timings
* `POST /shutdown` → `{"ok": true}`, then the process exits

`params` are all optional and default to the contract's values.  An **unknown** `params` key is a
`400`, so a typo in the client never silently loses a setting.

## Security notes

The sidecar is a local HTTP server, so it is treated as one:

* **Loopback only.**  It binds `127.0.0.1`, never `0.0.0.0`, and `allow_reuse_address` is off so
  another process cannot take the port from under it.
* **Per-session token.**  32 random bytes as hex in the environment variable `GT_RENDERER_TOKEN`,
  never on the command line (a process list is world-readable).  Without it the process refuses to
  start: exit code 2 and a message on stderr.  Every request carries it as `X-GT-Token` and it is
  compared with `hmac.compare_digest`.
* **`Host` header check.**  Anything but `127.0.0.1[:port]` / `localhost[:port]` gets a `403`, so
  a page in a browser cannot rebind DNS onto the port.
* **Body cap.**  A `Content-Length` above 64 MiB is refused with `413` before a byte of the body
  is read.
* **No paths on the wire.**  Images travel inline as base64 PNG; the sidecar opens no file a
  request names, and no data leaves the machine.
* **No tracebacks in responses.**  Failures are logged to stderr and answered as
  `{"error": "<type>: <message>"}`.
* **Verified model files.**  `models.py` checks the pinned SHA-256 before a checkpoint is opened —
  `torch.jit.load` deserialises code, so this is load-bearing.
* **Lifecycle.**  The process exits on `POST /shutdown`, on stdin EOF (the parent died) and on
  SIGTERM / SIGINT, so a crashed app never leaves a listening socket behind.  Stdin is *polled*
  (`PeekNamedPipe` on Windows, `select` elsewhere, every 0.25 s): a thread parked in a blocking
  pipe read stalled every new thread — the accept loop included — while torch and diffusers were
  loading, so the sidecar answered nothing during its warm-up.
