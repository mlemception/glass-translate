# Quality renderer — generative stage timings, 2026-09-11

Measured on the target machine, not extrapolated.  Every number below comes from
`tools/quality_probe.py` (per-panel grids on real pages) or from a scratch bucket benchmark using
the same `LamaStage` / `SdxlStage` code, with the models already resident.

**Machine.** Windows 11 Pro 26200, RTX 4080 16 GB (17.17 GB reported by `mem_get_info`, of which
~1.3 GB is already held by the desktop), driver 591.86 / CUDA 13.1, CPython 3.13.11 win_amd64,
models on NVMe.

## 1. Sidecar venv

`renderer/.venv`, created by `renderer\install.bat`, deliberately separate from the app's `.venv`
(the exe must stay torch-free).

| | |
|---|---|
| torch | 2.14.0+cu130 (`cp313-cp313-win_amd64`), CUDA runtime 13.0, cuDNN 9.24.0 |
| torchvision | 0.29.0+cu130 |
| diffusers / transformers / accelerate | 0.40.0 / 5.16.1 / 1.15.0 |
| safetensors / peft / huggingface-hub / tokenizers | 0.8.0 / 0.20.0 / 1.31.0 / 0.23.2 |
| numpy / pillow / opencv-python-headless / omegaconf | 2.5.2 / 12.3.0 / 5.0.0.93 / 2.3.1 |
| attention backend | torch SDPA (diffusers' default) — **no xformers**, no `torch.compile` |

`cu130` is the stable variant for torch 2.14: cu128 was removed from the build matrix in the 2.12
cycle and cu129 stops at 2.9.0.  Driver 591.86 covers CUDA 13.0 under minor-version compatibility.

Verified: `renderer/.venv/Scripts/python -c "import torch;print(torch.__version__,
torch.cuda.is_available())"` → `2.14.0+cu130 True`, device `NVIDIA GeForce RTX 4080`.

## 2. Model table

Full licences and provenance: [`renderer/MODELS.md`](../../renderer/MODELS.md).  Pinned digests:
`renderer/glassrenderer/models.json` (byte-identical copy at
`glasstranslate/render/quality_models.json`).

| Group | Files | Bytes | Licence |
|---|---|---|---|
| LaMa (`lama/`) | `anime_manga_lama.pt` + config/LICENSE/NOTICE | 204,193,932 | MIT + Apache-2.0 notice |
| SDXL checkpoint (`sdxl/`) | `Illustrious-XL-v1.0.safetensors` + `LICENSE.md` | 6,938,054,845 | CreativeML Open RAIL++-M |
| SDXL base configs (`sdxl-base-config/`) | 14 json/txt files for offline `from_single_file` | 3,175,105 | CreativeML Open RAIL++-M |
| ControlNet (`controlnet/`) | union promax fp16 + config | 2,513,343,668 | Apache-2.0 |
| VAE (`vae/`) | `sdxl-vae-fp16-fix` fp16 + config | 334,643,869 | MIT |
| **Total** | **24 files** | **9,993,412,420 (9.31 GiB)** | |

Measured download of the whole table with `ensure_models` (streamed, sha256-verified, atomic
rename): **~9 minutes** end to end.  Throughput was not uniform — the LaMa repo served ~1 MB/s
(204 MB in 216 s) while the Stability, Onoma and xinsir blobs came down at ~77 MB/s, so the
progress row must not extrapolate an ETA from the first file.

Offline loading is verified: the whole pipeline builds with `HF_HUB_OFFLINE=1` and
`local_files_only=True`, with `from_single_file(config=<quality/sdxl-base-config>)` supplying the
component configs that diffusers would otherwise fetch from the Hub.

## 3. Cold start

`build_stages` → `lama.load()` → `sdxl.load()`, i.e. torch import, CUDA context, digest
verification and 9.3 GB of weights onto the device.

| Phase | Time |
|---|---|
| `import torch` + CUDA init + `torch.jit.load` (LaMa) | 3.1 – 4.7 s |
| SDXL single-file + ControlNet + VAE onto the GPU | 15.0 – 16.8 s |
| **Models resident** | **18.0 – 21.5 s** (five runs; median ≈ 19.6 s) |
| Resident VRAM after load | 9.83 GB allocated |

Against the plan's **< 20 s**: met on the median, marginal at the tail.  The variance is page
cache: a second start in the same session is at the 18 s end.

### The one-off TorchScript compile

The LaMa export's **first** forward pass costs **24 – 26 s** (TorchScript codegen plus cuFFT plan
creation for the FFC blocks).  Two findings:

* With torch's profiling executor left on, the cost is paid *twice* — ~11 s on the first call and
  another ~19 s on the second, i.e. a 19 s stall in the middle of the second real job.
  `LamaStage.load()` therefore sets `torch._C._jit_set_profiling_executor(False)` /
  `_jit_set_profiling_mode(False)`, which collapses it to one hit.
* That one hit is **shape-independent**.  A throwaway 64 × 64 run absorbs all of it; afterwards a
  first 512 px panel takes 0.20 s and a first 1024 px panel 0.17 s.  `LamaStage.warm_up()` does
  exactly that.

It is deliberately *not* called from `load()`, because the protocol wants `READY <port>` and a
`/health` with `warm: false` before it.  The server calls `warm_up()` in its preload thread.

### The allocator pool (found during slice 5)

The batch renders showed LaMa taking **6.5 – 15.6 s** on the first job of a page and 40 – 70 ms
afterwards, at shapes that a fresh process handles in 0.1 s.  Replaying the exact jobs ruled out
shape and content; the stall reproduces only with the SDXL weights resident: right after an SDXL
job, a LaMa run whose blocks are not yet in torch's caching-allocator pool takes **14.3 s**, and after
`torch.cuda.empty_cache()` even the next SDXL job takes 14.7 s.  With ~9.2 GB allocated the card
sits near its 16 GB under WDDM, and every fresh `cudaMalloc` from inside a job pays the driver's
paging.  Once the blocks are pooled the same run takes 66 – 73 ms at every shape tried.

Consequences, both in `LamaStage.warm_up()`: after the 64 px compile run it runs one 1024 × 1024
pass (**~21 s**, once, off the request path) so the pool holds every block a real job needs; and
nothing calls `empty_cache()` between jobs (only `unload()` does).  Reserved pool after warm-up:
11.6 GB.

The honest cold-start figure is therefore **~19 s to models-resident and ~90 s to genuinely warm**
(LaMa compile ~26 s + LaMa pool ~21 s + SDXL 1024 warm-up ~22 s); `/health` reports `loading`
until then and the first `/inpaint` simply waits for the preload lock.

## 4. Warm per-stage timings

24 steps at strength 0.4 (= 10 real UNet steps), ControlNet scale 0.8, square panels, median of
3 – 5 runs after warm-up.

| Long side | LaMa | SDXL, guidance 4.0 | SDXL, guidance 1.0 | Peak VRAM (allocated) |
|---|---|---|---|---|
| 512 | **48 ms** | 1.00 s | 1.00 s | 8.9 GB |
| 768 | **46 ms** | 1.66 s | 1.10 s | 11.3 GB |
| 1024 | **94 ms** | 3.20 – 3.45 s | 1.88 – 2.12 s | 12.3 GB plain / **10.4 GB with VAE tiling** |

Notes:

* LaMa at 768 is no slower than at 512 — the FFT sizes happen to be friendlier — so bucketing to
  512/768/1024 costs nothing and keeps the plan cache small.
* At guidance ≤ 1.0 diffusers drops classifier-free guidance and the UNet batch halves, which is
  where the 1.55× at 1024 comes from.  The shipped default stays **4.0** (the protocol's default);
  1.0 is the lever to pull first if a page budget is missed.
* VAE tiling is enabled automatically from a 1024 px generation size (`VAE_TILING_FROM`).  It costs
  ~250 ms and takes peak allocated from 12.31 GB to 10.41 GB (reserved 13.48 → 11.18 GB).  Below
  1024 it is neither needed nor free, so it stays off.

The opt-in GPU test measures the same thing as an assertion: `peak 10.68 GB` at 1024 with
`assert peak_gb < 12.0`.

## 5. Targets

| Target | Measured | Verdict |
|---|---|---|
| LaMa < 0.2 s per 1024 panel | 94 ms | **met** |
| SDXL < 3.5 s per 1024 panel | 3.20 – 3.45 s @ guidance 4.0; 1.9 – 2.1 s @ 1.0 | **met** (with no headroom at 4.0) |
| Cold start < 20 s | 18.0 – 21.5 s to models-resident | **met on the median**; the first job then pays a one-off 24 – 26 s JIT compile unless `warm_up()` ran |
| VRAM < 12 GB | 10.4 – 10.7 GB at 1024 with VAE tiling | **met** (12.3 GB without tiling) |
| Per page < 10 s | not measured here (one job per panel; the app's scheduler decides how many) | pending slice 4 |

### If a target slips: TensorRT

`torch-tensorrt 2.14.0+cu130` does publish a `cp313-win_amd64` wheel on `download.pytorch.org`
(Dynamo frontend only; `tensorrt-cu13-libs` needs `--extra-index-url https://pypi.nvidia.com`
because it has no PyPI wheel).  Realistic payoff for SDXL fp16 is **1.3 – 1.6×**, against 2 – 10
minutes of engine build per shape per engine (UNet and ControlNet), cache invalidation on any
driver/GPU/TensorRT/torch change, and a proprietary NVIDIA licence — the only non-permissive
component that would enter the stack.

**Not worth it yet.** `guidance_scale=1.0` already buys 1.55× at 1024 for one flag and no licence
cost, and VAE tiling already buys the VRAM target.  Keep TensorRT as a documented optional slice,
lazily imported with an eager fallback, and only if slice 5's tuning still misses.

## 6. Parameter grid — what the pages actually look like

`tools/quality_probe.py` sweeps strength × ControlNet scale and writes a greyscale contact sheet
(< 1000 px, < 100 KB) per run.  Both runs used `--steps 24 --guidance 4.0 --seed 0`, mask dilation
2 px, feather 2 px.

### `more_comparisons/2ja.jpg`, block 3 (free text over paper)

`--crop 150,100,250,300 --mask-boxes "197,188,100,111"` → `demo/output/quality/2ja-b3-*`.
Bucket 512; SDXL 0.78 – 1.40 s per cell.

The vertical `ひとひと` sits on clean paper beside a bubble edge and a figure.  LaMa alone already
removes it completely and leaves flat white with no halo; every SDXL cell keeps it clean.  Faces,
the hand, the bubble outline and the background hatching outside the mask are untouched (the
composite guarantees that byte for byte).  The only visible difference across the grid is that
**s0.5 sprinkles faint speckle into the flat white** that s0.3 / s0.4 do not; ControlNet scale
barely matters here because there are no strokes inside the hole to hold.

### `Examples/before.jpg`, region A (vertical text over drawn line art)

`--crop 0,380,230,340 --mask-boxes "27,468,149,222"` → `demo/output/quality/beforeA-*`.
Bucket 512; SDXL 0.82 – 1.60 s per cell.

This is the real test: `「天才」日車 その才能の輝きとは` runs down a drawn fence — straight slats, a
diagonal beam, a curved handrail and dense hatching.  Observations from the sheet:

* **LaMa** continues the slats and the diagonal beam plausibly, but the reconstructed lines are
  softer and slightly wavier than the drawn ones, and the hatch density inside the hole is lower
  than around it.
* **SDXL s0.3 – s0.4** sharpen those lines back to ink weight and rebuild hatching that matches the
  surrounding density.  Geometry tracks the source: the slats stay parallel and the beam stays
  straight.
* **s0.5** starts inventing: extra speckle appears in the lower-left and the handrail curve drifts
  off the surviving stroke on either side of the hole.
* **ControlNet scale** 0.8 and 1.0 are near-identical at these strengths; 0.6 lets the slats wander
  by a pixel or two.  Nothing in the grid re-etched a ghost of the erased glyphs, which was the
  risk of a full-strength line control over a blanked region.
* No colour leaked in, no text was hallucinated, and the `「天` fragment that sits *outside* the
  supplied mask box survives everywhere — as it must.

**Recommended defaults, unchanged from the protocol:** `strength 0.4`, `steps 24`,
`controlnet_scale 0.8`, `guidance 4.0`, `feather_px 2`.  Slice 5 should sweep against the metrics
rather than by eye, but nothing in this grid argues for a different starting point.

## 7. Reproducing

```
renderer\install.bat
:: models land in models\quality (app-side download; ensure_models does the same)
set GT_GPU_TESTS=1
renderer\.venv\Scripts\python -m pytest renderer/tests/test_stages_gpu.py -q -s

set HF_HUB_OFFLINE=1
set DIFFUSERS_DISABLE_REMOTE_CODE=true
renderer\.venv\Scripts\python tools/quality_probe.py ^
    --image Examples/before.jpg --crop 0,380,230,340 --mask-boxes "27,468,149,222" --tag beforeA
```

The table tests need no GPU and run in the main venv:
`.venv\Scripts\python -m pytest renderer/tests/test_models_table.py -q`.
