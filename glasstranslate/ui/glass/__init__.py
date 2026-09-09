"""Liquid Glass plumbing for the control window.

* :mod:`.appearance` - Windows appearance / accessibility signals
  (transparency, reduced motion, high contrast, dark mode, accent, text
  scale) with change notification and the ``GLASSTRANSLATE_APPEARANCE``
  override layer.
* :mod:`.backdrop` - the 15 Hz screen grabber that feeds the glass shader
  with the desktop pixels *behind* the window, the QML image provider that
  serves them, and the luma / ink-polarity helpers.
* :mod:`.win32` - the few Win32 calls a frameless translucent window needs
  (capture exclusion, physical geometry, keyboard move / size).

See ``docs/GLASS_DESIGN.md`` sections 1.2 and 1.3.
"""
from .appearance import Appearance, AppearanceSignals, AppearanceWatcher, read_signals
from .backdrop import BackdropFrame, BackdropGrabber, BackdropProvider, InkPolarity, LumaSmoother, WindowGeometry
from .win32 import exclude_from_capture, keyboard_move, physical_rect

__all__ = [
    "Appearance",
    "AppearanceSignals",
    "AppearanceWatcher",
    "BackdropFrame",
    "BackdropGrabber",
    "BackdropProvider",
    "InkPolarity",
    "LumaSmoother",
    "WindowGeometry",
    "exclude_from_capture",
    "keyboard_move",
    "physical_rect",
    "read_signals",
]
