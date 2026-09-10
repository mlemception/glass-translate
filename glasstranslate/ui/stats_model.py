"""``bridge.stats``: the live latency readout of the status strip."""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Property, QObject, Signal

from ..core.types import PipelineStats
from .bridge_fields import format_stats

__all__ = ["StatsModel"]


class StatsModel(QObject):
    """Live latency readout for the status strip (``bridge.stats``).

    ``totalText`` / ``fpsText`` feed the strip, the four detail lines the drawer; every value is
    ``"-"`` until the first pipeline pass.
    """

    changed = Signal()

    _KEYS = ("total", "fps", "stages", "segments", "cache", "devices")

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._values: Dict[str, str] = {k: "-" for k in self._KEYS}

    def update(self, stats: PipelineStats) -> None:
        """Refresh every line from a pipeline pass."""
        self._values = format_stats(stats)
        self.changed.emit()

    def reset(self) -> None:
        self._values = {k: "-" for k in self._KEYS}
        self.changed.emit()

    def _text(self, key: str) -> str:
        return self._values[key]

    totalText = Property(str, lambda self: self._text("total"), notify=changed)
    fpsText = Property(str, lambda self: self._text("fps"), notify=changed)
    stagesText = Property(str, lambda self: self._text("stages"), notify=changed)
    segmentsText = Property(str, lambda self: self._text("segments"), notify=changed)
    cacheText = Property(str, lambda self: self._text("cache"), notify=changed)
    devicesText = Property(str, lambda self: self._text("devices"), notify=changed)
