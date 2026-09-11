"""Pipeline side of the quality renderer: per-panel jobs are submitted for free text over
art, results swap the block's ``clean_patch`` and bump ``clean_patch_serial`` at the start of
the next pass, stale results are dropped and everything keeps working without a sidecar.

Fake engines and a fake scheduler only: no sidecar process, no GPU, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pytest

from glasstranslate.config.settings import AppConfig
from glasstranslate.core.pipeline import Pipeline
from glasstranslate.core.types import Rect
from glasstranslate.render import quality as Q

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the fakes live next door
from test_pipeline_blocks import (  # noqa: E402
    FakeCapture,
    FakeOCR,
    FixedDetector,
    RecordingTranslator,
    bubble_page,
)

W, H = 400, 300


class FakeScheduler:
    """Records the submitted jobs; the test plays results back through ``on_result``."""

    def __init__(self, on_result: Callable[[Q.Job, np.ndarray], None], status) -> None:
        self.on_result = on_result
        self.status = status
        self.jobs: List[Q.Job] = []
        self.stopped = 0
        self.available = True

    def submit(self, jobs) -> None:
        self.jobs.extend(jobs)

    def stop(self) -> None:
        self.stopped += 1

    @property
    def stats(self) -> dict:
        return {"done": 0, "failed": 0, "queued": 0}


def _art_page():
    """``bubble_page`` plus a black bar beside the free-standing column, so that block
    counts as free text *over artwork* (the eraser keeps the bar and sets ``outline``) -
    exactly the case the quality renderer exists for."""
    import cv2

    img, segs = bubble_page()
    cv2.rectangle(img, (306, 200), (317, 239), (0, 0, 0), -1)
    return img, segs


def _pipeline(cfg: Optional[AppConfig] = None, *, quality: bool = True):
    img, segs = _art_page()
    cfg = cfg or AppConfig(translation_backend="identity", source_lang="ja", target_lang="en",
                           quality_renderer="auto")
    results: list = []
    made: List[FakeScheduler] = []

    def factory(config, *, on_result, status):
        if not quality:
            return None
        made.append(FakeScheduler(on_result, status))
        return made[-1]

    pipe = Pipeline(
        cfg,
        region_provider=lambda: Rect(0, 0, W, H),
        on_result=lambda segments, stats: results.append((segments, stats)),
        on_status=lambda msg: None,
        capture_factory=lambda c: FakeCapture(img),
        ocr_factory=lambda c: FakeOCR(segs),
        translator_factory=lambda c: RecordingTranslator(),
        detector_factory=FixedDetector,
        quality_factory=factory,
    )
    return pipe, img, results, made


def _free_block(segments):
    """The free-text block over the grey backdrop (the one the sidecar works on)."""
    for seg in segments:
        style = seg.style
        if style.outline and not style.in_bubble and style.erase_mask is not None:
            return seg
    return None


def test_panel_jobs_are_submitted_for_free_text_over_art() -> None:
    pipe, _, results, made = _pipeline()
    pipe.step()
    assert made and made[0].jobs, "no quality job was planned for the free-text block"
    segments, _ = results[-1]
    free = _free_block(segments)
    assert free is not None
    key = Q.block_key(free.source_text, free.style)
    assert any(key in job.members for job in made[0].jobs)
    # Bubble blocks are never sent (they are redrawn, not inpainted).
    bubbles = [s for s in segments if s.style.in_bubble]
    for seg in bubbles:
        assert not any(Q.block_key(seg.source_text, seg.style) in job.members for job in made[0].jobs)


def test_a_result_swaps_the_patch_and_bumps_the_serial_on_the_next_pass() -> None:
    pipe, img, results, made = _pipeline()
    pipe.step()
    segments, _ = results[-1]
    free = _free_block(segments)
    assert free is not None
    before = free.style.clean_patch.copy()
    assert free.style.clean_patch_serial == 0
    key = Q.block_key(free.source_text, free.style)
    job = next(j for j in made[0].jobs if key in j.members)
    made[0].on_result(job, np.zeros((job.rect.h, job.rect.w, 3), np.uint8))

    stats = pipe.step()  # nothing on screen changed: the cheap "skipped" pass still re-emits
    assert stats is not None and stats.skipped_unchanged
    assert free.style.clean_patch_serial == 1
    assert not np.array_equal(free.style.clean_patch, before)
    assert free.style.clean_patch.shape == before.shape
    assert stats.extra.get("quality_patches") == 1
    emitted, _ = results[-1]
    assert any(s is free for s in emitted)  # the same objects, with the upgraded patch


def test_a_stale_result_is_dropped() -> None:
    pipe, _, results, made = _pipeline()
    pipe.step()
    segments, _ = results[-1]
    free = _free_block(segments)
    assert free is not None
    key = Q.block_key(free.source_text, free.style)
    job = next(j for j in made[0].jobs if key in j.members)
    pipe._live.clear()  # the block is gone (a re-read replaced it)
    made[0].on_result(job, np.zeros((job.rect.h, job.rect.w, 3), np.uint8))
    stats = pipe.step()
    assert stats is not None
    assert "quality_patches" not in stats.extra


def test_no_scheduler_means_no_change_at_all() -> None:
    pipe, _, results, made = _pipeline(quality=False)
    stats = pipe.step()
    assert stats is not None and made == []
    segments, _ = results[-1]
    assert segments and all(s.style.clean_patch_serial == 0 for s in segments)


def test_stop_stops_the_scheduler() -> None:
    pipe, _, _, made = _pipeline()
    pipe.step()
    pipe.stop()
    assert made[0].stopped == 1


def test_changing_a_quality_field_rebuilds_the_scheduler() -> None:
    cfg = AppConfig(translation_backend="identity", source_lang="ja", target_lang="en",
                    quality_renderer="auto")
    pipe, _, _, made = _pipeline(cfg)
    pipe.step()
    assert len(made) == 1
    cfg.quality_sidecar_python = "C:/somewhere/python.exe"
    pipe.set_config(cfg)
    pipe.step()
    assert len(made) == 2 and made[0].stopped == 1
    pipe.stop()


@pytest.mark.parametrize("mode, expected", [("auto", "auto"), ("off", "off"), ("nonsense", "off"), ("AUTO", "auto")])
def test_config_normalises_the_quality_mode(mode: str, expected: str) -> None:
    assert AppConfig.from_dict({"quality_renderer": mode}).quality_renderer == expected
