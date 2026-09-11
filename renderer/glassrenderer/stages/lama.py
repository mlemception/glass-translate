"""LaMa stage: structural inpainting of the erased region.

The checkpoint is the manga/anime fine-tune of big-LaMa packaged as a frozen
TorchScript module (``anime_manga_lama.pt``, MIT + Apache-2.0 notice), so there
is no model code to carry: ``torch.jit.load`` then ``model(image, mask)``.  Its
own ``config.json`` documents the contract this stage implements — float32
``(N,3,H,W)`` RGB in ``[0,1]`` plus a float32 ``(N,1,H,W)`` mask where ``>= 0.5``
erases, H and W multiples of 8 — and states that mask binarisation, the
4-channel concat, the output clamp and the composite are all baked into the
export.  Padding, colour space and uint8 normalisation are ours.

fp32 on purpose: the FFC blocks call ``torch.fft.rfftn``/``irfftn``, half
precision cuFFT is size-restricted and lossy on flat paper, and the generator is
only ~51M parameters, so fp16 would save nothing that matters.

``torch.jit.load`` deserialises code, so the file's SHA-256 is re-checked
against the pinned table on every load — see :func:`glassrenderer.models.verify_file`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .. import models as M

log = logging.getLogger(__name__)

#: The TorchScript export requires both dimensions to be multiples of this.
SIZE_MULTIPLE = 8
#: Mask values at or above this erase (the protocol's mask is 0 or 255).
MASK_ON = 128


def _pad_to_multiple(array: np.ndarray, multiple: int = SIZE_MULTIPLE) -> np.ndarray:
    """Reflect-pad ``array`` (HxW or HxWxC) on the right/bottom to a multiple."""
    h, w = array.shape[:2]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if not pad_h and not pad_w:
        return array
    pads = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (array.ndim - 2)
    # ``reflect`` needs at least 2 px along the padded axis; ``edge`` is a safe
    # fallback for the degenerate strips a caller may hand us.
    mode = "reflect" if h > 1 and w > 1 else "edge"
    return np.pad(array, pads, mode=mode)


class LamaStage:
    """Manga big-LaMa as a :class:`glassrenderer.pipeline.Stage`.

    The model stays resident once :meth:`load` has run; ``run`` is then a pad,
    one forward pass and a crop.  ``device`` is ``"cuda"``/``"cuda:0"``/``"cpu"``.
    """

    name = "lama"

    def __init__(self, models_dir: Union[str, Path], device: str = "cuda") -> None:
        self.models_dir = Path(models_dir)
        self.device = device
        self._model: Optional[Any] = None
        self.last_ms: float = 0.0
        self.load_ms: float = 0.0

    # ------------------------------------------------------------ loading
    @property
    def ready(self) -> bool:
        """True once the TorchScript module is resident on ``device``."""
        return self._model is not None

    def spec(self) -> M.ModelFile:
        """The pinned table entry for the TorchScript weights."""
        return next(s for s in M.QUALITY_FILES if s.name == M.LAMA_WEIGHTS)

    def load(self) -> None:
        """Verify the digest and load the TorchScript module onto ``device``."""
        if self._model is not None:
            return
        import torch

        started = time.perf_counter()
        # The profiling executor re-optimises on the *second* call, which costs
        # ~19 s in the middle of a job.  Off, the whole ~26 s TorchScript /
        # cuFFT compile happens once on the first call instead - see
        # :meth:`warm_up` and ``docs/perf/2026-09-11-quality-renderer-timings.md``.
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        path = M.verify_file(self.models_dir, self.spec())
        log.info("loading %s on %s", path.name, self.device)
        model = torch.jit.load(str(path), map_location=self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self._model = model
        self.load_ms = (time.perf_counter() - started) * 1000.0

    def warm_up(self, size: int = 64, pool_side: int = 1024) -> float:
        """Pay the one-off compile and reserve the allocator pool up front. Returns ms.

        Two throwaway runs.  The ``size`` px one absorbs the ~26 s TorchScript /
        cuFFT compile, which is shape-independent.  The ``pool_side`` square one
        (the largest bucket) makes the caching allocator hold every block a real
        job will need: with the SDXL weights resident the card sits near its
        limit under WDDM, and a fresh ``cudaMalloc`` from inside a job then costs
        14 - 20 s (measured on the RTX 4080; 0.07 s once the blocks are pooled).
        For the same reason nothing here ever calls ``torch.cuda.empty_cache``
        between jobs.  Deliberately *not* called from :meth:`load` — the
        protocol wants ``READY`` (and a ``/health`` with ``warm: false``)
        before that, so the server runs it in its own loading thread.
        """
        started = time.perf_counter()
        for side in (size, pool_side):
            if side <= 0:
                continue
            panel = np.full((side, side, 3), 240, np.uint8)
            mask = np.zeros((side, side), np.uint8)
            mask[side // 4 : side // 2, side // 4 : side // 2] = 255
            self.run(panel, mask, {})
        return (time.perf_counter() - started) * 1000.0

    def unload(self) -> None:
        """Drop the module and free its VRAM."""
        if self._model is None:
            return
        import torch

        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------- run
    def run(self, image_rgb: np.ndarray, mask: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Fill ``mask`` in ``image_rgb`` and return uint8 RGB of the same size.

        ``image_rgb`` is uint8 HxWx3, ``mask`` uint8 HxW with 255 = fill.  The
        export composites internally, so pixels outside the mask come back as
        the model's reconstruction of the input; the pipeline's own compositing
        step is what makes them byte-identical again.
        """
        import torch

        self._validate(image_rgb, mask)
        if self._model is None:
            self.load()
        height, width = image_rgb.shape[:2]
        padded_image = _pad_to_multiple(image_rgb)
        padded_mask = _pad_to_multiple((mask >= MASK_ON).astype(np.uint8) * 255)

        started = time.perf_counter()
        with torch.inference_mode():
            image_t = self._to_tensor(torch, padded_image.transpose(2, 0, 1))
            mask_t = self._to_tensor(torch, padded_mask[None, :, :])
            output = self._model(image_t, mask_t)  # type: ignore[misc]
            filled = output[0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
            result = filled.permute(1, 2, 0).cpu().numpy()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return np.ascontiguousarray(result[:height, :width])

    def _to_tensor(self, torch: Any, chw: np.ndarray) -> Any:
        """``(C,H,W)`` uint8 -> float32 ``(1,C,H,W)`` in ``[0,1]`` on ``device``."""
        tensor = torch.from_numpy(np.ascontiguousarray(chw)).to(self.device)
        return tensor.to(torch.float32).div_(255.0).unsqueeze(0)

    @staticmethod
    def _validate(image_rgb: np.ndarray, mask: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError(f"image must be uint8 HxWx3 RGB, got {image_rgb.shape} {image_rgb.dtype}")
        if mask.ndim != 2 or mask.shape != image_rgb.shape[:2]:
            raise ValueError(f"mask must be HxW matching the image, got {mask.shape}")
