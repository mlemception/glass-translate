# Quality renderer — models, licences and provenance

Every weight the sidecar loads is pinned by repo, commit, byte size and SHA-256 in
[`glassrenderer/models.json`](glassrenderer/models.json) (copied byte for byte to
`glasstranslate/render/quality_models.json`, which is what the app's downloader reads).  The
sidecar itself **never downloads anything**: it verifies the digests and loads from
`<models_dir>/quality` with `HF_HUB_OFFLINE=1` and `local_files_only=True`.

Total first-run download: **9,993,412,420 bytes (9.31 GiB)** across 24 files.

Licence gate for this feature: Apache-2.0 / MIT / BSD / CreativeML OpenRAIL++-M only.  Nothing
GPL, AGPL or LGPL; nothing with a non-commercial clause; nothing share-alike.  Everything below
was read from the live licence text or the repo's own `LICENSE` file, not from a model card's
front matter alone, during the model research of 2026-09-10
(`demo/output/model_research.json`).

---

## 1. LaMa — structural inpainting

| | |
|---|---|
| Repo | [`TareHimself/AnimeMangaInpainting-torchscript`](https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript) |
| Revision | `b592884b2b6589e57df6916db6bcac7a6307fc58` |
| Files | `anime_manga_lama.pt` (204,190,262 B), `config.json`, `LICENSE`, `NOTICE` |
| Local path | `<models_dir>/quality/lama/` |
| Licence | **MIT** (checkpoint + packaging) with an **Apache-2.0** notice for the vendored architecture |
| Licence text | <https://huggingface.co/TareHimself/AnimeMangaInpainting-torchscript/blob/main/LICENSE> |

The repo ships the grant in-tree rather than only on the card.  `LICENSE` reads "MIT License,
Copyright (c) 2024 dreMaz (lama_large_512px.ckpt), Copyright (c) 2026 Oyintare Ebelo (TorchScript
packaging, wrapper, scripts)" and states that `lama_ffc.py` is Apache-2.0, not MIT.  `NOTICE`
carries the Apache-2.0 attribution for Samsung AI Center Moscow / [advimman/lama][lama] (2021),
from which the FFC generator is derived.  No copyleft, no use restrictions, no non-commercial
clause anywhere in the chain.

These are the same weights IOPaint ships as "anime-lama" and manga-image-translator calls
`lama_large` — but **manga-image-translator's own loaders are GPL-3.0 and are neither copied nor
imported here**; only dreMaz's MIT checkpoint is shared with it.  Because the checkpoint is a
frozen TorchScript module we carry no model code at all: `torch.jit.load`, then
`model(image, mask)`.

`torch.jit.load` deserialises code, so the pinned revision plus the SHA-256 check is a security
control, not bookkeeping: `glassrenderer.models.verify_file` re-hashes the file on **every** load
(`stages/lama.py`), not just after the download.

[lama]: https://github.com/advimman/lama

## 2. SDXL checkpoint — Illustrious-XL-v1.0

| | |
|---|---|
| Repo | [`OnomaAIResearch/Illustrious-XL-v1.0`](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v1.0) |
| Revision | `89d625482edf06d545b740265482f5c8fb2cadb0` |
| File | `Illustrious-XL-v1.0.safetensors` (6,938,040,736 B, all 2514 tensors F16) |
| Local path | `<models_dir>/quality/sdxl/` |
| Licence | **CreativeML Open RAIL++-M** (the repo's `license_link` resolves to the verbatim SDXL 1.0 `LICENSE.md`) |
| Licence text | <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/LICENSE.md> (shipped as `quality/sdxl/LICENSE.md`) |

Verbatim summary of the grant, read from the live text: a perpetual, worldwide, royalty-free,
irrevocable IP licence to reproduce, modify, sublicense and distribute the model and its
derivatives.  **No** non-commercial clause, **no** source copyleft, **no** share-alike on our own
code.  The single propagating obligation is the OpenRAIL use-restriction schedule: if the weights
or a derivative are *redistributed*, the use restrictions must travel with them.  GlassTranslate
does not redistribute the weights — the app downloads them from Hugging Face at the pinned
revision — so the obligation is discharged by this file recording the licence and its URL.

Header verified by range-reading the safetensors metadata: `modelspec.prediction_type: "epsilon"`,
`modelspec.architecture: "stable-diffusion-xl-v1-base"`, no `v_pred`/`ztsnr` tensors.  It therefore
uses the stock scheduler with no `prediction_type` override.

**Rejected alternatives, and why** (all are better-known but fail the gate):

* `Laxhar/noobai-XL-*` — blanket commercial prohibition that on its face extends to generated
  pages, plus a duty to open-source the product.
* `KBlueLeaf/Kohaku-XL-*`, `OnomaAIResearch/Illustrious-xl-early-release-v0` — Fair AI Public
  License 1.0-SD, a genuine share-alike copyleft.
* `cagliostrolab/animagine-xl-4.0` — also OpenRAIL++-M and licence-safe; kept as the alternate.
  It loses only on fidelity: it is aesthetic-tuned, and a low-denoise restoration pass wants the
  weakest style prior available.
* There is **no** SDXL checkpoint trained on monochrome manga pages.  The only manga-specific SDXL
  assets on Hugging Face and Civitai are a provenance-free community merge and a 93 MB LoRA;
  neither is a credible default.

### 2a. SDXL base configuration tree (offline `from_single_file`)

| | |
|---|---|
| Repo | [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) |
| Revision | `462165984030d82259a11f4367a4eed129e94a7b` |
| Files | `model_index.json`, `scheduler/`, `text_encoder{,_2}/config.json`, `tokenizer{,_2}/` (vocab, merges, configs), `unet/config.json`, `vae/config.json` — 14 small files, 3.17 MB total |
| Local path | `<models_dir>/quality/sdxl-base-config/` |
| Licence | **CreativeML Open RAIL++-M**, as above |

`from_single_file` needs the pipeline's component configs, and by default it fetches them from the
Hub.  With the network down (or `HF_HUB_OFFLINE=1`) that either fails or blocks on timeouts, which
alone would eat the whole cold-start budget.  Pinning this tree and passing it as
`config=<that directory>` is what makes the sidecar genuinely offline.  Verified: the pipeline
loads with `HF_HUB_OFFLINE=1` and `local_files_only=True` set.

### 2b. VAE — `sdxl-vae-fp16-fix`

| | |
|---|---|
| Repo | [`madebyollin/sdxl-vae-fp16-fix`](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) |
| Revision | `207b116dae70ace3637169f1ddd2434b91b3a8cd` |
| Files | `diffusion_pytorch_model.safetensors` (334,643,238 B), `config.json` |
| Local path | `<models_dir>/quality/vae/` |
| Licence | **MIT** — <https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/README.md> |

Mandatory, not an optimisation.  Stock SDXL's `vae/config.json` has no `force_upcast` key (so it
defaults to `True`) and diffusers' img2img pipeline branches on that flag in three places: with
the stock VAE we would run the VAE in fp32 twice per panel at 1024 with a multi-GB transient.
This one sets `"force_upcast": false`.

## 3. ControlNet — union promax, line-art mode

| | |
|---|---|
| Repo | [`xinsir/controlnet-union-sdxl-1.0`](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) |
| Revision | `801a4a3fa3d4c936f4feea95b98607bc6726f80c` |
| Files | `diffusion_pytorch_model_promax.safetensors` (2,513,342,408 B, already fp16) and `config_promax.json`, stored under their plain diffusers names |
| Local path | `<models_dir>/quality/controlnet/` |
| Licence | **Apache-2.0** — <https://huggingface.co/xinsir/controlnet-union-sdxl-1.0> |

The only fully unencumbered licence among the plausible ControlNets: no use restrictions, no
copyleft, no attribution rider.  ProMax is also the only candidate that can carry both the
line-art condition (control index 3, "thin line") and an inpainting condition through one network
in one forward pass; every alternative needs two stacked 2.5 GB ControlNets for the same thing.
diffusers ships `ControlNetUnionModel` and `StableDiffusionXLControlNetUnionImg2ImgPipeline`,
which is what `stages/sdxl.py` uses.

**Preprocessor: none.** No learned line detector is downloaded or shipped.  Manga is already
black ink on white paper, so `stages/lineart.py` recovers the strokes with OpenCV — greyscale,
invert, one linear contrast stretch — which is more faithful than a detector trained at 512 px to
hallucinate lines from shaded art, and costs zero download and zero VRAM.  Polarity matters and is
easy to get backwards: the ecosystem convention (and this model's training data) is **white lines
on a black ground**, which is what inverting black ink gives.  The dilated mask region is written
to pure black *after* extraction, so nothing about the Japanese text conditions the generation.

**Rejected alternatives:**

* `Eugeoter/noob-sdxl-controlnet-manga_line` — the only ControlNet trained on monochrome manga
  line extraction, but Fair AI Public License 1.0-SD: genuinely copyleft ("all modifications must
  be provided under this license") with a network source-availability clause.  Harmless for an
  unmodified local sidecar, fatal if we ever fine-tune.  It is also NoobAI-family and would only
  pair with a NoobAI checkpoint, which fails the gate anyway.
* `TheMisto.ai/MistoLine` — OpenRAIL-M plus a visible commercial-attribution requirement.
* `kataragi/*` — OpenRAIL-M; also Animagine-3.1-specific.
* `stabilityai/control-lora` — a bespoke Stability agreement: revocable, non-sublicensable, gated
  above 1M MAU, incorporating a mutable acceptable-use policy by reference.  Not loadable from
  diffusers either.
* `ShermanG/ControlNet-Standard-Lineart-for-SDXL` — trained on exactly the line map we feed, and
  the third most-downloaded line-art ControlNet on the Hub, but it declares **no licence at all**
  (empty tags, null `cardData.license`, no `LICENSE` file).  Unusable until the author adds a
  grant.
* `lllyasviel/Annotators` — declares only `license: other` with no text for ~24 aggregated
  checkpoints.  The individual upstreams (MangaLineExtraction_PyTorch, Anime2Sketch,
  informative-drawings) are MIT, but that is provenance, not a grant from the redistributor.

## 4. Runtime dependencies

Pinned in [`requirements.txt`](requirements.txt); the whole closure (17 direct + ~36 transitive)
was audited for the GPL family with no hits.

| Package | Licence |
|---|---|
| torch 2.14.0+cu130, torchvision 0.29.0+cu130 | BSD-3-Clause |
| diffusers, transformers, accelerate, safetensors, peft, huggingface-hub, tokenizers, opencv-python-headless, requests | Apache-2.0 |
| numpy, omegaconf | BSD-3-Clause |
| pillow | MIT-CMU |
| pytest, einops, PyYAML, urllib3, rich, typer | MIT |
| certifi, tqdm | MPL-2.0 (file-level weak copyleft; no obligation when used unmodified) |

TensorRT is **not** installed.  It is the only proprietary component that was considered, it is
deferred (see `docs/perf/2026-09-11-quality-renderer-timings.md`), and it must stay optional and
lazily imported if it is ever added.

## 5. Hardening

* `HF_HUB_OFFLINE=1` and `local_files_only=True` on every load — no Hub round-trips at run time.
* `DIFFUSERS_DISABLE_REMOTE_CODE=true` is recommended in the sidecar's environment: diffusers'
  attention dispatcher and quantisation backends can otherwise download and execute compute
  kernels from the Hub at run time.
* SHA-256 re-verified at load time, not only at download time (`models.verify_file`).
* No safety checker and no watermarker are loaded; neither is a licence obligation here, and the
  invisible-watermark dependency is not installed.
* File names and subdirectories in the table are validated against path traversal before any
  network request or file write (`models._validate`).
