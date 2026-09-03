"""Global hotkeys via ``pynput`` with callbacks marshalled to the Qt thread.

``pynput.keyboard.GlobalHotKeys`` fires its callbacks on its own listener
thread.  :class:`HotkeyManager` turns each hit into a Qt signal emission, so
the connected slots run on the thread owning the manager (the GUI thread).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Optional

from PySide6.QtCore import QObject, Qt, Signal, Slot

__all__ = ["HotkeyManager", "normalize_hotkey"]

log = logging.getLogger(__name__)


def normalize_hotkey(spec: str) -> str:
    """Return ``spec`` in pynput's canonical form (``<ctrl>+<alt>+g``).

    Accepts the loose forms people type into a line edit: ``Ctrl+Alt+G``,
    ``control + alt + g``, ``<ctrl>+<alt>+G``.  Raises ``ValueError`` when the
    result is not a valid pynput hotkey.
    """
    aliases = {
        "control": "ctrl",
        "ctl": "ctrl",
        "option": "alt",
        "win": "cmd",
        "windows": "cmd",
        "super": "cmd",
        "meta": "cmd",
        "esc": "esc",
        "escape": "esc",
        "return": "enter",
    }
    parts = []
    for raw in spec.split("+"):
        token = raw.strip().strip("<>").lower()
        if not token:
            raise ValueError(f"empty key in hotkey {spec!r}")
        token = aliases.get(token, token)
        parts.append(f"<{token}>" if len(token) > 1 else token)
    result = "+".join(parts)
    from pynput import keyboard

    keyboard.HotKey.parse(result)  # raises ValueError on garbage
    return result


class HotkeyManager(QObject):
    """Registers global hotkeys and runs the callbacks on the Qt thread.

    Usage::

        hk = HotkeyManager()
        hk.bind({"<ctrl>+<alt>+g": overlay.toggle_grab})
        ...
        hk.rebind({...})  # replaces the mapping
        hk.stop()
    """

    triggered = Signal(str)  # hotkey spec that fired (emitted from the listener thread)
    error = Signal(str)  # hotkey registration failed (spec + reason)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._callbacks: Dict[str, Callable[[], None]] = {}
        self._listener = None
        self._lock = threading.Lock()
        self.triggered.connect(self._dispatch, Qt.ConnectionType.QueuedConnection)

    @property
    def active(self) -> bool:
        return self._listener is not None

    def bind(self, mapping: Dict[str, Callable[[], None]]) -> None:
        """Start listening for ``mapping`` (spec -> callback).  Specs are
        normalised; invalid ones are skipped and reported via :attr:`error`."""
        self._stop_listener()
        callbacks: Dict[str, Callable[[], None]] = {}
        for spec, cb in mapping.items():
            try:
                callbacks[normalize_hotkey(spec)] = cb
            except ValueError as exc:
                log.warning("invalid hotkey %r: %s", spec, exc)
                self.error.emit(f"Invalid hotkey {spec!r}: {exc}")
        with self._lock:
            self._callbacks = callbacks
        if not callbacks:
            return
        try:
            from pynput import keyboard

            hotkeys = {spec: self._make_trigger(spec) for spec in callbacks}
            listener = keyboard.GlobalHotKeys(hotkeys)
            listener.daemon = True
            listener.start()
            self._listener = listener
        except Exception as exc:  # pragma: no cover - depends on OS input hooks
            log.warning("could not start global hotkey listener: %s", exc)
            self.error.emit(f"Hotkeys unavailable: {exc}")

    def rebind(self, mapping: Dict[str, Callable[[], None]]) -> None:
        """Replace the current mapping."""
        self.bind(mapping)

    def stop(self) -> None:
        self._stop_listener()
        with self._lock:
            self._callbacks = {}

    # ------------------------------------------------------------ internals
    def _make_trigger(self, spec: str) -> Callable[[], None]:
        def trigger() -> None:
            self.triggered.emit(spec)

        return trigger

    @Slot(str)
    def _dispatch(self, spec: str) -> None:
        with self._lock:
            cb = self._callbacks.get(spec)
        if cb is None:
            return
        try:
            cb()
        except Exception:  # pragma: no cover - callback bug must not kill the GUI
            log.exception("hotkey callback for %s failed", spec)

    def _stop_listener(self) -> None:
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.stop()
            except Exception:  # pragma: no cover
                log.exception("stopping hotkey listener failed")
