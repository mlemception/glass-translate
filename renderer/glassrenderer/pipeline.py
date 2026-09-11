"""Job orchestration: bucket, LaMa, SDXL, resample, composite.

:class:`Renderer` is the whole of the sidecar's behaviour; :mod:`.server` only
wraps it in HTTP.  One job runs at a time (one GPU), so the class holds no
per-job state beyond the timings it returns.

The real stages live in :mod:`glassrenderer.stages`, which imports torch and
diffusers and exists only in the sidecar venv.  It is imported **lazily**, so
this module - and everything under it - runs unchanged in the app's torch-free
venv: in ``--fake`` mode nothing is imported at all, and in real mode a
missing or broken ``stages`` module degrades to ``models: missing`` in
:meth:`Renderer.health` and a ``500`` from :meth:`Renderer.inpaint`, never a
crash of the process.

Whatever the stages produce, :func:`glassrenderer.compositing.composite` puts
it back under the mask and :meth:`Renderer.inpaint` re-checks the result
against the input outside the mask before it is allowed out.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .compositing import bucket_size, composite, resize_image, resize_mask
from .fake import build_fake_stages
from .protocol import VERSION

log = logging.getLogger(__name__)

MODEL_KEYS: Tuple[str, ...] = ("lama", "sdxl", "controlnet")
STAGE_TIMING_KEYS: Tuple[str, ...] = ("decode", "lama", "sdxl", "composite", "total")


class RendererError(RuntimeError):
    """A job could not be rendered; answered with ``500 {"error": ...}``."""


def _elapsed_ms(since: float) -> float:
    return (time.perf_counter() - since) * 1000.0


class Renderer:
    """Runs one inpainting job at a time against the loaded stages."""

    def __init__(
        self, models_dir: Path, device: str = "auto", fake: bool = False
    ) -> None:
        self.models_dir = Path(models_dir)
        self.fake = bool(fake)
        self._requested_device = device or "auto"
        self._device: Optional[str] = None
        self._stages: Optional[Tuple[Any, Any]] = None
        self._warm = False
        self._started = time.monotonic()
        self._lock = threading.RLock()  # one GPU: stage construction, loading and jobs never overlap
        self._load_state: Optional[str] = None  # None | "loading" | "ready" | "failed"
        self._load_error: Optional[str] = None

    # --- introspection -------------------------------------------------------

    @property
    def mode(self) -> str:
        """``"fake"`` or ``"real"``."""
        return "fake" if self.fake else "real"

    @property
    def device(self) -> str:
        """The resolved device string (``"cuda:0"`` / ``"cpu"``)."""
        if self._device is None:
            self._device = self._resolve_device()
        return self._device

    def _resolve_device(self) -> str:
        if self.fake:
            return "cpu"
        if self._requested_device != "auto":
            return "cuda:0" if self._requested_device == "cuda" else self._requested_device
        try:  # torch only exists in the sidecar venv
            import torch  # type: ignore[import-not-found]

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover - depends on the venv
            return "cpu"

    def _model_status(self) -> Dict[str, str]:
        if self.fake:
            return {key: "ready" for key in MODEL_KEYS}
        try:
            from .stages import model_status  # type: ignore[import-not-found]

            reported = dict(model_status(self.models_dir))
        except Exception as exc:
            log.warning("model status unavailable (%s: %s)", type(exc).__name__, exc)
            return {key: "missing" for key in MODEL_KEYS}
        status = {key: str(reported.get(key, "missing")) for key in MODEL_KEYS}
        if self._load_state == "loading":
            # The files are there; the weights are on their way to the device.
            status = {key: ("loading" if value == "ready" else value) for key, value in status.items()}
        return status

    def _vram_mb(self) -> Tuple[Optional[int], Optional[int]]:
        if self.fake or not self.device.startswith("cuda"):
            return None, None
        try:  # pragma: no cover - needs a CUDA device
            import torch  # type: ignore[import-not-found]

            free, total = torch.cuda.mem_get_info()
            return int(total // (1 << 20)), int((total - free) // (1 << 20))
        except Exception:  # pragma: no cover - no CUDA, no numbers
            return None, None

    def health(self) -> Dict[str, Any]:
        """The ``GET /health`` body of ``renderer/PROTOCOL.md``."""
        total_mb, used_mb = self._vram_mb()
        return {
            "ok": True,
            "version": VERSION,
            "mode": self.mode,
            "device": self.device,
            "models": self._model_status(),
            "warm": self._warm,
            "vram_total_mb": total_mb,
            "vram_used_mb": used_mb,
            "uptime_s": round(time.monotonic() - self._started, 3),
            "load_error": self._load_error,
        }

    # --- stages --------------------------------------------------------------

    def _stage_pair(self) -> Tuple[Any, Any]:
        """The ``(lama, sdxl)`` stages, built once."""
        with self._lock:
            if self._stages is not None:
                return self._stages
            if self.fake:
                self._stages = build_fake_stages()
                return self._stages
            try:
                from .stages import build_stages  # type: ignore[import-not-found]

                self._stages = build_stages(self.models_dir, self.device, fake=False)
            except Exception as exc:
                raise RendererError(
                    f"{type(exc).__name__}: stages unavailable ({exc})"
                ) from None
            return self._stages

    def preload(self, warm: bool = True) -> None:
        """Load the real stages - and pay their one-off compile - off the request path.

        The server calls this in a daemon thread right after ``READY``; meanwhile
        ``/health`` reports the present models as ``"loading"``.  ``warm`` also runs the
        ``warm_up`` of each stage (the LaMa TorchScript compile alone is 25 s on first use).
        Nothing raises: a failure is logged, shown in ``/health`` as ``load_error`` and
        surfaces on ``/inpaint`` as a ``500``.
        """
        if self.fake:
            self._stage_pair()
            return
        self._load_state = "loading"
        try:
            with self._lock:
                stages = self._stage_pair()
                for stage in stages:
                    loader = getattr(stage, "load", None)
                    if callable(loader):
                        loader()
                if warm:
                    for stage in stages:
                        warm_up = getattr(stage, "warm_up", None)
                        if callable(warm_up):
                            warm_up()
                    self._warm = True
            self._load_state = "ready"
            log.info("stages loaded%s", " and warm" if warm else "")
        except Exception as exc:
            self._load_state = "failed"
            self._load_error = f"{type(exc).__name__}: {exc}"
            log.warning("preload failed (%s)", self._load_error)

    @staticmethod
    def _run_stage(
        stage: Any, image: np.ndarray, mask: np.ndarray, params: Mapping[str, Any]
    ) -> np.ndarray:
        try:
            output = stage.run(image, mask, params)
        except Exception as exc:
            raise RendererError(f"{type(exc).__name__}: {exc}") from None
        if not isinstance(output, np.ndarray) or output.shape != image.shape:
            raise RendererError(f"stage {stage.name} returned an unusable array")
        return output.astype(np.uint8, copy=False)

    # --- the job -------------------------------------------------------------

    @staticmethod
    def _check_inputs(image_rgb: np.ndarray, mask: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise RendererError(f"image must be HxWx3, got {image_rgb.shape}")
        if mask.ndim != 2 or mask.shape != image_rgb.shape[:2]:
            raise RendererError("image and mask must be the same size")
        if image_rgb.dtype != np.uint8 or mask.dtype != np.uint8:
            raise RendererError("image and mask must be uint8")

    def _selected_stages(self, params: Mapping[str, Any]) -> List[Any]:
        lama, sdxl = self._stage_pair()
        chosen = [(lama, params.get("lama", True)), (sdxl, params.get("sdxl", True))]
        return [stage for stage, enabled in chosen if bool(enabled)]

    def inpaint(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        params: Mapping[str, Any],
        *,
        decode_ms: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
        """Render one panel.

        Returns ``(image_rgb, timings_ms, stages)`` at the *input* size.
        ``decode_ms`` is the caller's base64/PNG decode time, folded into the
        reported total so the client sees the whole cost of its request.
        Raises :class:`RendererError` for anything the caller should see as a
        ``500``, including a result that is not byte-identical outside the
        mask.
        """
        started = time.perf_counter()
        self._check_inputs(image_rgb, mask)
        timings = {key: 0.0 for key in STAGE_TIMING_KEYS}
        timings["decode"] = float(decode_ms)
        with self._lock:  # a job never overlaps the preload thread
            active = self._selected_stages(params)
            return self._render(image_rgb, mask, params, active, timings, started)

    def _render(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        params: Mapping[str, Any],
        active: List[Any],
        timings: Dict[str, float],
        started: float,
    ) -> Tuple[np.ndarray, Dict[str, float], List[str]]:
        work, work_mask, source_size = self._to_bucket(image_rgb, mask, bool(active))
        names: List[str] = []
        for stage in active:
            stage_started = time.perf_counter()
            work = self._run_stage(stage, work, work_mask, params)
            timings[stage.name] = _elapsed_ms(stage_started)
            names.append(stage.name)
        if source_size is not None:
            work = resize_image(work, source_size)

        composite_started = time.perf_counter()
        result = composite(image_rgb, work, mask, int(params.get("feather_px", 2)))
        timings["composite"] = _elapsed_ms(composite_started)
        self._verify(result, image_rgb, mask)

        timings["total"] = timings["decode"] + _elapsed_ms(started)
        if names and not self.fake:
            self._warm = True
        return result, timings, names

    @staticmethod
    def _to_bucket(
        image_rgb: np.ndarray, mask: np.ndarray, wanted: bool
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
        """Downscale to the panel's bucket; ``None`` when no resampling ran."""
        source_size = (image_rgb.shape[0], image_rgb.shape[1])
        if not wanted:
            return image_rgb, mask, None
        target = bucket_size(*source_size)
        if target == source_size:
            return image_rgb, mask, None
        return resize_image(image_rgb, target), resize_mask(mask, target), source_size

    @staticmethod
    def _verify(result: np.ndarray, image_rgb: np.ndarray, mask: np.ndarray) -> None:
        """Refuse to return anything that moved a pixel outside the mask."""
        if result.shape != image_rgb.shape or result.dtype != np.uint8:
            raise RendererError("the composited result changed shape or dtype")
        keep = mask == 0
        if keep.any() and not np.array_equal(result[keep], image_rgb[keep]):
            raise RendererError("the result differs from the input outside the mask")


__all__: List[str] = ["MODEL_KEYS", "STAGE_TIMING_KEYS", "Renderer", "RendererError"]
