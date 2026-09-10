"""Settings / status window: the Liquid Glass control panel.

:class:`ControlWindow` is a frameless, per-pixel-translucent ``QQuickView`` that loads
``qrc:/qml/Main.qml`` and exposes two context properties to it: ``bridge``
(:class:`ControlBridge` - every setting, list, state flag and action of the panel) and
``appearance`` (:class:`~glasstranslate.ui.glass.appearance.Appearance` - Windows
transparency / reduced motion / high contrast / dark mode / text scale).  The public Python
contract is unchanged from the old QWidget window so ``app.py`` keeps working: signals
``config_changed(AppConfig)``, ``start_stop_requested(bool)``, ``grab_mode_requested()``,
``toggle_glass_requested()``, ``models_changed()``, ``closed()``; methods ``show()``,
``load_config()``, ``set_running()``, ``update_stats()``, ``show_status()``, ``save_soon()``,
``save_now()`` and the ``config`` property.  Edits made from QML mutate the
:class:`~glasstranslate.config.settings.AppConfig`, emit ``config_changed`` and persist with a
300 ms debounce.

What the glass refracts (docs/GLASS_DESIGN.md section 1.2): while the window is visible, not
minimised and glass is allowed, a :class:`~glasstranslate.ui.glass.backdrop.BackdropGrabber`
thread copies the desktop pixels behind the window at 15 Hz (the window itself is excluded
from capture with ``SetWindowDisplayAffinity``); the GUI thread stores each frame in the
``image://backdrop/<serial>`` provider and updates ``bridge.backdropSerial`` /
``backdropOrigin`` / ``backdropLuma`` / ``inkPolarity``.

Qt resources: this module imports the compiled ``resources_rc``; in a source checkout without
it (the file is generated, see ``tools/build_resources.py``) it builds the resources once.
"""
from __future__ import annotations

import logging

# The implementation lives in three cohesive modules (split 2026-09-10 to get this file back
# under the 800-line ceiling); everything below is re-exported so ``from ..ui import control``
# keeps working for app.py, tools/ and the tests.
from .bridge_fields import (  # noqa: F401 - re-exported for tests and app.py
    CUDA_FROZEN_HINT,
    LANGUAGES,
    download_label,
    download_progress,
    format_stats,
    is_frozen,
    translate_device_items,
)
from .control_bridge import ControlBridge, WINDOW_TITLE  # noqa: F401
from .control_window import ControlWindow, MAIN_QML_URL, WINDOW_DEFAULT_SIZE, WINDOW_FLAGS  # noqa: F401
from .download_workers import MangaOcrDownloadWorker, ModelDownloadWorker  # noqa: F401 - re-exported
from .resources_guard import RESOURCES_OK, _ensure_resources  # noqa: F401
from .stats_model import StatsModel  # noqa: F401

__all__ = [
    "CUDA_FROZEN_HINT",
    "LANGUAGES",
    "MAIN_QML_URL",
    "RESOURCES_OK",
    "WINDOW_DEFAULT_SIZE",
    "WINDOW_FLAGS",
    "WINDOW_TITLE",
    "ControlBridge",
    "ControlWindow",
    "MangaOcrDownloadWorker",
    "ModelDownloadWorker",
    "StatsModel",
    "_ensure_resources",
    "download_label",
    "download_progress",
    "format_stats",
    "is_frozen",
    "translate_device_items",
]

log = logging.getLogger(__name__)
