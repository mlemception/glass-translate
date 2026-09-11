"""Background download threads of the Engines page (Argos / Sugoi packages, manga-ocr
bundle, quality-renderer models).

All of them expose the same signal contract so ``ControlBridge`` drives them through one
progress row: ``progress(done, total_or_-1)`` while downloading, then ``finished_ok(path)``
or ``failed(message)``.  ``control.py`` re-exports them; tests monkeypatch the names there.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from PySide6.QtCore import QObject, QThread, Signal

__all__ = ["MangaOcrDownloadWorker", "ModelDownloadWorker", "QualityModelsDownloadWorker"]

log = logging.getLogger(__name__)


class ModelDownloadWorker(QThread):
    """Downloads one Argos package in the background.

    Emits :attr:`progress` (done, total_or_-1) while downloading, then either
    :attr:`finished_ok` with the extracted directory or :attr:`failed`.
    """

    progress = Signal(int, int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        models_dir: str,
        from_code: str,
        to_code: str,
        parent: Optional[QObject] = None,
        *,
        sugoi: bool = False,
    ) -> None:
        super().__init__(parent)
        self._models_dir = models_dir
        self._from = from_code
        self._to = to_code
        self._sugoi = sugoi  # install the Sugoi v4 ja->en model instead of an Argos package

    def run(self) -> None:  # noqa: D401 - QThread API
        try:
            from ..translate.argos import ArgosCT2Translator

            translator = ArgosCT2Translator(self._models_dir, device="cpu")

            def cb(done: int, total: Optional[int]) -> None:
                self.progress.emit(int(done), int(total) if total else -1)

            if self._sugoi:
                path = translator.download_sugoi(progress_cb=cb)
            else:
                path = translator.download_package(self._from, self._to, progress_cb=cb)
            self.finished_ok.emit(str(path))
        except Exception as exc:
            log.exception("model download failed")
            self.failed.emit(str(exc))


class MangaOcrDownloadWorker(QThread):
    """Downloads the manga-ocr ONNX bundle (~200 MB) into ``<models_dir>/manga-ocr`` in the background.

    Same signal contract as :class:`ModelDownloadWorker`: :attr:`progress` ``(done_bytes,
    total_bytes)`` (the per-file percentage from ``ensure_models`` is mapped onto the bundle's total
    size so the shared progress row reads as one download), then :attr:`finished_ok` with the
    directory or :attr:`failed` with a redacted message.
    """

    progress = Signal(int, int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, models_dir: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._models_dir = models_dir

    def run(self) -> None:  # noqa: D401 - QThread API
        from ..ocr.models import MANGA_OCR_FILES, ModelDownloadError, ensure_models

        sizes = {f.name: f.size for f in MANGA_OCR_FILES}
        total = sum(sizes.values())
        done_before: Dict[str, int] = {}

        def cb(filename: str, percent: int) -> None:
            done_before[filename] = sizes.get(filename, 0) * max(0, min(100, int(percent))) // 100
            self.progress.emit(sum(done_before.values()), total)

        try:
            path = ensure_models(self._models_dir, progress=cb)
            self.finished_ok.emit(str(path))
        except (ModelDownloadError, ValueError, OSError) as exc:
            log.warning("manga-ocr download failed: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - a dying QThread would leave the progress row stuck forever
            log.exception("manga-ocr download crashed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class QualityModelsDownloadWorker(QThread):
    """Downloads the quality renderer's models into ``<models_dir>/quality`` in the background.

    The mirror image of :class:`MangaOcrDownloadWorker` for the sidecar's model store
    (``render/quality_models.py``, which reuses the same verified streaming downloader):
    the per-file percentages are mapped onto the bundle's total size so the shared progress
    row reads as one download.  A build without the model table fails immediately with a
    plain message rather than downloading anything.
    """

    # 64-bit: the bundle is several gigabytes, which overflows a Qt ``int``.
    progress = Signal("qlonglong", "qlonglong")
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, models_dir: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._models_dir = models_dir

    def run(self) -> None:  # noqa: D401 - QThread API
        from ..render.quality_models import ModelDownloadError, ensure_models, load_table

        files = load_table()
        total = sum(f.size for f in files)
        # ``ensure_models`` reports progress by bare file name, and the table has the
        # same name in several subdirectories (config.json), so the row is driven by
        # the largest file of each name: the shared small ones are a rounding error
        # against a ~10 GB bundle and the bar can never overshoot.
        sizes: Dict[str, int] = {}
        for f in files:
            sizes[f.name] = max(sizes.get(f.name, 0), f.size)
        done_before: Dict[str, int] = {}

        def cb(filename: str, percent: int) -> None:
            done_before[filename] = sizes.get(filename, 0) * max(0, min(100, int(percent))) // 100
            self.progress.emit(sum(done_before.values()), total)

        try:
            path = ensure_models(self._models_dir, files=files, progress=cb)
            self.finished_ok.emit(str(path))
        except (ModelDownloadError, ValueError, OSError) as exc:
            log.warning("quality model download failed: %s", exc)
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001 - a dying QThread would stick the progress row
            log.exception("quality model download crashed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
