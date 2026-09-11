"""SDXL img2img stage with the union ControlNet in line-art mode.

LaMa restores structure but paints flat: it cannot invent halftone, hatching or
the continuation of a screentone field.  This stage re-renders the whole panel
at a low denoise strength so the fill grows the surrounding texture back, with
the union ControlNet holding every surviving stroke in place.

Everything is local and pinned (see :mod:`glassrenderer.models`):

* checkpoint — Illustrious-XL-v1.0, a single fp16 SDXL safetensors, epsilon
  prediction, loaded with ``from_single_file`` against the pinned copy of the
  SDXL-base config tree so no Hub round-trip is ever attempted;
* ControlNet — ``xinsir/controlnet-union-sdxl-1.0`` promax, driven on control
  index 3 (thin line / lineart) with the map from :mod:`.lineart`;
* VAE — ``madebyollin/sdxl-vae-fp16-fix``, mandatory: the stock SDXL VAE has no
  ``force_upcast: false`` and would run fp32 twice per panel.

fp16 throughout, SDPA attention (diffusers' default backend — no xformers), no
safety checker, no watermarker, no CPU offload: the models stay resident so a
warm panel is one forward pass.

Prompting is deliberately inverted from the usual anime defaults.  Onoma's own
example negative prompt contains "monochrome, greyscale, comic", which would
fight a manga clean-up; here those are the *positive* tags.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .. import models as M
from . import lineart

log = logging.getLogger(__name__)

#: Union ControlNet control index for "thin line (canny/mlsd/lineart/animelineart/ted)".
LINEART_MODE = 3
#: Long-side buckets: fewer distinct shapes means fewer cuDNN/cuFFT re-plans.
BUCKETS: Tuple[int, ...] = (512, 768, 1024)
#: SDXL latents are an 8x downsample; keep both dimensions on a multiple of 16.
SIZE_MULTIPLE = 16

POSITIVE_PROMPT = (
    "monochrome, greyscale, comic, halftone, traditional media, "
    "simple background, best quality"
)
NEGATIVE_PROMPT = (
    "text, english text, speech bubble, translation request, watermark, "
    "signature, username, spot color, blurry, jpeg artifacts, lowres, worst quality"
)

DEFAULTS: Dict[str, Any] = {
    "strength": 0.4,
    "steps": 24,
    "controlnet_scale": 0.8,
    "guidance": 4.0,
    "seed": 0,
    "control_end": 0.9,
}

#: At 1024 the VAE's transient is what pushes peak VRAM past 12 GB; tiling the
#: decode costs ~0.25 s and takes the peak from 12.3 GB to 10.4 GB.  Below this
#: it is neither needed nor free, so it stays off.
VAE_TILING_FROM = 1024


def bucket_size(height: int, width: int, bucket: Optional[int] = None) -> Tuple[int, int]:
    """Generation size for a ``height`` x ``width`` panel, rounded to 16 px.

    The long side lands on the smallest of :data:`BUCKETS` that still covers it
    (1024 for anything larger), so a small panel is not upscaled to 1024 for
    nothing and a page-sized panel is downscaled once.
    """
    longest = max(height, width)
    target = bucket if bucket else next((b for b in BUCKETS if b >= longest), BUCKETS[-1])
    scale = target / float(longest)
    out_h = max(SIZE_MULTIPLE, int(round(height * scale / SIZE_MULTIPLE)) * SIZE_MULTIPLE)
    out_w = max(SIZE_MULTIPLE, int(round(width * scale / SIZE_MULTIPLE)) * SIZE_MULTIPLE)
    return out_h, out_w


def _resize(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize to ``(height, width)``; area for shrinking, linear for growing."""
    out_h, out_w = size
    if image.shape[:2] == (out_h, out_w):
        return image
    shrinking = out_h * out_w < image.shape[0] * image.shape[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (out_w, out_h), interpolation=interp)


def _joined(built_in: str, extra: object) -> str:
    """Append a caller's prompt fragment to the built-in one."""
    text = str(extra or "").strip()
    return f"{built_in}, {text}" if text else built_in


class SdxlStage:
    """SDXL + union ControlNet as a :class:`glassrenderer.pipeline.Stage`.

    ``run`` takes the LaMa result as the img2img init image and the *original*
    panel's line art as the control image; the caller composites the output
    back under the mask.
    """

    name = "sdxl"

    def __init__(self, models_dir: Union[str, Path], device: str = "cuda") -> None:
        self.models_dir = Path(models_dir)
        self.device = device
        self._pipe: Optional[Any] = None
        self.last_ms: float = 0.0
        self.load_ms: float = 0.0

    # ------------------------------------------------------------ loading
    @property
    def ready(self) -> bool:
        """True once the pipeline is resident on ``device``."""
        return self._pipe is not None

    def specs(self) -> Sequence[M.ModelFile]:
        """Every pinned entry this stage loads (checkpoint, configs, VAE, ControlNet)."""
        return [s for s in M.QUALITY_FILES if M.group_of(s) in (M.SDXL, M.CONTROLNET)]

    def _dir(self, subdir: str) -> str:
        return str(M.quality_dir(self.models_dir).joinpath(*subdir.split("/")))

    def load(self) -> None:
        """Build the pipeline from the local files only (no Hub access)."""
        if self._pipe is not None:
            return
        import torch
        from diffusers import (
            AutoencoderKL,
            ControlNetUnionModel,
            StableDiffusionXLControlNetUnionImg2ImgPipeline,
        )

        started = time.perf_counter()
        checkpoint = M.verify_file(self.models_dir, self._checkpoint_spec())
        vae = AutoencoderKL.from_pretrained(
            self._dir(M.VAE_SUBDIR), torch_dtype=torch.float16, local_files_only=True
        )
        controlnet = ControlNetUnionModel.from_pretrained(
            self._dir(M.CONTROLNET_SUBDIR), torch_dtype=torch.float16, local_files_only=True
        )
        pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_single_file(
            str(checkpoint),
            config=self._dir(M.SDXL_CONFIG_SUBDIR),
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True,
            add_watermarker=False,
            safety_checker=None,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.to(self.device)
        self._pipe = pipe
        self.load_ms = (time.perf_counter() - started) * 1000.0
        log.info("SDXL pipeline ready on %s in %.0f ms", self.device, self.load_ms)

    @staticmethod
    def _checkpoint_spec() -> M.ModelFile:
        return next(s for s in M.QUALITY_FILES if s.name == M.SDXL_CHECKPOINT)

    def unload(self) -> None:
        """Drop the pipeline and free its VRAM."""
        if self._pipe is None:
            return
        import torch

        self._pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warm_up(self, size: int = 1024) -> float:
        """One throwaway 1-step pass so cuDNN picks its algorithms. Returns ms."""
        panel = np.full((size, size, 3), 235, np.uint8)
        mask = np.zeros((size, size), np.uint8)
        mask[size // 2 : size // 2 + 16, size // 2 : size // 2 + 16] = 255
        started = time.perf_counter()
        self.run(panel, mask, {"steps": 4, "strength": 0.25})
        return (time.perf_counter() - started) * 1000.0

    # -------------------------------------------------------------- run
    def run(self, image_rgb: np.ndarray, mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Re-render ``image_rgb`` at a low denoise; returns uint8 RGB, same size."""
        import torch

        self._validate(image_rgb, mask)
        if self._pipe is None:
            self.load()
        opts = {**DEFAULTS, **(params or {})}
        height, width = image_rgb.shape[:2]
        gen_h, gen_w = bucket_size(height, width, opts.get("bucket"))

        self._set_vae_tiling(opts.get("vae_tiling"), max(gen_h, gen_w))
        init = _resize(image_rgb, (gen_h, gen_w))
        control = _resize(lineart.build_control_image(image_rgb, mask), (gen_h, gen_w))
        generator = torch.Generator(device=self.device).manual_seed(int(opts["seed"]))

        started = time.perf_counter()
        with torch.inference_mode():
            output = self._pipe(  # type: ignore[misc]
                prompt=_joined(POSITIVE_PROMPT, opts.get("prompt")),
                negative_prompt=_joined(NEGATIVE_PROMPT, opts.get("negative_prompt")),
                image=self._as_pil(init),
                control_image=[self._as_pil(control)],
                control_mode=[LINEART_MODE],
                controlnet_conditioning_scale=[float(opts["controlnet_scale"])],
                control_guidance_start=0.0,
                control_guidance_end=float(opts["control_end"]),
                strength=float(opts["strength"]),
                num_inference_steps=int(opts["steps"]),
                guidance_scale=float(opts["guidance"]),
                generator=generator,
                output_type="np",
            )
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - started) * 1000.0

        rendered = (np.clip(output.images[0], 0.0, 1.0) * 255.0).round().astype(np.uint8)
        return np.ascontiguousarray(_resize(rendered, (height, width)))

    def _set_vae_tiling(self, requested: Optional[bool], longest: int) -> None:
        """Tile the VAE for large panels so the peak stays under 12 GB."""
        wanted = bool(requested) if requested is not None else longest >= VAE_TILING_FROM
        vae = self._pipe.vae  # type: ignore[union-attr]
        vae.enable_tiling() if wanted else vae.disable_tiling()

    @staticmethod
    def _as_pil(rgb: np.ndarray) -> Any:
        from PIL import Image

        return Image.fromarray(rgb, mode="RGB")

    @staticmethod
    def _validate(image_rgb: np.ndarray, mask: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError(f"image must be uint8 HxWx3 RGB, got {image_rgb.shape} {image_rgb.dtype}")
        if mask.ndim != 2 or mask.shape != image_rgb.shape[:2]:
            raise ValueError(f"mask must be HxW matching the image, got {mask.shape}")
