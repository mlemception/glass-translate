"""Windows appearance & accessibility signals for the glass panel, plus the QML-facing
``Appearance`` object and the ``GLASSTRANSLATE_APPEARANCE`` override layer.

The first part of this module is ``demo/output/probes/win-a11y/win_a11y.py`` lifted as-is
(verified on Windows 11 build 26200 / PySide6 6.11.2, see the probe scripts and
``demo/output/probes/reports/win-a11y.md``); the only extension is the ``text_scale`` field.

What it reads (all cheap, ~0.1-0.3 ms for a full snapshot):

* transparency  - HKCU\\...\\Themes\\Personalize\\EnableTransparency (DWORD, 1 = effects on)
* reduce_motion - SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION) == FALSE  (Settings > Accessibility >
                  Visual effects > "Animation effects"); mirrors UserPreferencesMask[4] & 0x02
* high_contrast - SystemParametersInfoW(SPI_GETHIGHCONTRAST).dwFlags & HCF_HIGHCONTRASTON, + scheme name
* dark_mode     - HKCU\\...\\Themes\\Personalize\\AppsUseLightTheme == 0
* accent        - HKCU\\Software\\Microsoft\\Windows\\DWM\\AccentColor (stored as 0xAABBGGRR -> converted to RGB),
                  fallback DwmGetColorizationColor (0xAARRGGBB, a *blended* colour, not the exact accent)
* composition   - DwmIsCompositionEnabled (always TRUE on Win8+, kept for completeness)
* rounded_corners - Windows build >= 22000 (DWMWA_WINDOW_CORNER_PREFERENCE verified to return S_OK on 26200)
* windows_build - RtlGetVersion (immune to compatibility-shim lies)
* text_scale    - HKCU\\Software\\Microsoft\\Accessibility\\TextScaleFactor / 100 (Settings > Accessibility >
                  Text size; absent -> 1.0).  Win32 apps must apply Windows text scaling themselves.

Change notification (``AppearanceWatcher``):

* QAbstractNativeEventFilter on the QCoreApplication catching WM_SETTINGCHANGE (0x001A), WM_THEMECHANGED (0x031A),
  WM_DWMCOLORIZATIONCOLORCHANGED (0x0320), WM_DWMCOMPOSITIONCHANGED (0x031E), WM_SYSCOLORCHANGE (0x0015).
  Broadcasts only reach a process that owns at least one top-level HWND, so the watcher creates a hidden QWindow
  (create() only, never shown). Posted messages hit Qt native filters twice (dispatcher + wndproc), so refreshes
  are coalesced through a 50 ms single-shot timer and the snapshot is diffed before emitting.
* Qt's own styleHints().colorSchemeChanged and styleHints().accessibility().contrastPreferenceChanged are wired
  in as well (Qt 6.11 has no contrastPreference on QStyleHints itself - it lives on QAccessibilityHints).
* A 2 s poll is the belt-and-braces path: some toggles (registry edits, policy pushes) arrive without any message.

On non-Windows everything degrades to a static "effects on, light/dark from Qt" snapshot.

Override layer (docs/GLASS_DESIGN.md section 1.3): ``GLASSTRANSLATE_APPEARANCE`` is a comma-separated
token list applied on top of **every** snapshot (initial and each ``changed``), later tokens win:
``glass`` (transparency = composition = True, highContrast = False), ``solid`` (transparency = False),
``hc`` (highContrast = True), ``reducemotion`` / ``motion``, ``dark`` / ``light``,
``textscale=<100..225>``.  Unknown tokens are logged at WARNING and ignored; the override never
writes system settings.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

from PySide6.QtCore import (
    Property,
    QAbstractNativeEventFilter,
    QCoreApplication,
    QObject,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QGuiApplication, QWindow

__all__ = [
    "OVERRIDE_ENV",
    "Appearance",
    "AppearanceSignals",
    "AppearanceWatcher",
    "apply_overrides",
    "mode_for",
    "parse_overrides",
    "read_signals",
]

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

WM_SYSCOLORCHANGE = 0x0015
WM_SETTINGCHANGE = 0x001A
WM_DWMCOMPOSITIONCHANGED = 0x031E
WM_THEMECHANGED = 0x031A
WM_DWMCOLORIZATIONCOLORCHANGED = 0x0320
WATCHED_MESSAGES = frozenset({WM_SYSCOLORCHANGE, WM_SETTINGCHANGE, WM_DWMCOMPOSITIONCHANGED, WM_THEMECHANGED,
                              WM_DWMCOLORIZATIONCOLORCHANGED})

SPI_GETHIGHCONTRAST = 0x0042
SPI_GETCLIENTAREAANIMATION = 0x1042
HCF_HIGHCONTRASTON = 0x00000001

PERSONALIZE_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
DWM_KEY = r"Software\Microsoft\Windows\DWM"
ACCESSIBILITY_KEY = r"Software\Microsoft\Accessibility"

OVERRIDE_ENV = "GLASSTRANSLATE_APPEARANCE"
TEXT_SCALE_MIN, TEXT_SCALE_MAX = 100, 225


@dataclass(frozen=True)
class AppearanceSignals:
    """Snapshot of the OS-level appearance/accessibility preferences the panel cares about."""

    transparency: bool = True          # False -> paint a solid frosted fill instead of real translucency
    reduce_motion: bool = False        # True  -> no elastic/spring motion, instant state changes
    high_contrast: bool = False        # True  -> real 1-2 px borders, palette from GetSysColor
    high_contrast_scheme: str = ""     # e.g. "High Contrast Black" (empty when off)
    dark_mode: bool = False            # AppsUseLightTheme == 0
    accent: Tuple[int, int, int] = (0, 120, 212)  # RGB
    accent_argb: int = 0xFF0078D4
    composition: bool = True           # DWM composition (always True on Win8+)
    rounded_corners: bool = False      # DWMWA_WINDOW_CORNER_PREFERENCE usable (Win11 22000+)
    windows_build: int = 0
    source: str = "default"            # "win32" when read from the OS, "qt" fallback, "default" if nothing worked
    text_scale: float = 1.0            # Windows text size factor (1.0 .. 2.25)

    @property
    def glass_allowed(self) -> bool:
        return self.transparency and self.composition and not self.high_contrast

    @property
    def mode(self) -> str:
        """``"hc"`` | ``"solid"`` | ``"glass"`` exactly as ``Theme.qml`` derives it."""
        return mode_for(self)


def mode_for(sig: AppearanceSignals) -> str:
    """Theme mode: ``hc`` beats ``solid`` beats ``glass``; ``reduce_motion`` is orthogonal."""
    if sig.high_contrast:
        return "hc"
    if not sig.transparency or not sig.composition:
        return "solid"
    return "glass"


# ------------------------------------------------------------------------------------------------ win32 plumbing
if IS_WINDOWS:
    import ctypes.wintypes as wt
    import winreg

    class _HIGHCONTRASTW(ctypes.Structure):
        _fields_ = [("cbSize", wt.UINT), ("dwFlags", wt.DWORD), ("lpszDefaultScheme", wt.LPWSTR)]

    class _RTL_OSVERSIONINFOW(ctypes.Structure):
        _fields_ = [("dwOSVersionInfoSize", wt.DWORD), ("dwMajorVersion", wt.DWORD), ("dwMinorVersion", wt.DWORD),
                    ("dwBuildNumber", wt.DWORD), ("dwPlatformId", wt.DWORD), ("szCSDVersion", wt.WCHAR * 128)]

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.SystemParametersInfoW.argtypes = [wt.UINT, wt.UINT, ctypes.c_void_p, wt.UINT]
    _user32.SystemParametersInfoW.restype = wt.BOOL
    _dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    _dwmapi.DwmIsCompositionEnabled.argtypes = [ctypes.POINTER(wt.BOOL)]
    _dwmapi.DwmIsCompositionEnabled.restype = ctypes.c_long
    _dwmapi.DwmGetColorizationColor.argtypes = [ctypes.POINTER(wt.DWORD), ctypes.POINTER(wt.BOOL)]
    _dwmapi.DwmGetColorizationColor.restype = ctypes.c_long
    _ntdll = ctypes.WinDLL("ntdll")

    def _reg_dword(path: str, name: str) -> Optional[int]:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
                value, kind = winreg.QueryValueEx(key, name)
        except OSError:
            return None
        return int(value) if kind == winreg.REG_DWORD else None

    def _windows_build() -> int:
        info = _RTL_OSVERSIONINFOW()
        info.dwOSVersionInfoSize = ctypes.sizeof(info)
        if _ntdll.RtlGetVersion(ctypes.byref(info)) == 0:
            return int(info.dwBuildNumber)
        return int(sys.getwindowsversion().build)

    def _client_area_animation() -> Optional[bool]:
        flag = wt.BOOL(0)
        if _user32.SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(flag), 0):
            return bool(flag.value)
        return None

    def _high_contrast() -> Tuple[bool, str]:
        hc = _HIGHCONTRASTW()
        hc.cbSize = ctypes.sizeof(hc)
        if _user32.SystemParametersInfoW(SPI_GETHIGHCONTRAST, hc.cbSize, ctypes.byref(hc), 0):
            return bool(hc.dwFlags & HCF_HIGHCONTRASTON), hc.lpszDefaultScheme or ""
        return False, ""

    def _composition() -> bool:
        enabled = wt.BOOL(1)
        if _dwmapi.DwmIsCompositionEnabled(ctypes.byref(enabled)) == 0:
            return bool(enabled.value)
        return True

    def _accent() -> Tuple[Tuple[int, int, int], int]:
        # Registry value is 0xAABBGGRR (little-endian COLORREF-with-alpha). Exact user accent.
        raw = _reg_dword(DWM_KEY, "AccentColor")
        if raw is not None:
            r, g, b = raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF
            return (r, g, b), (0xFF << 24) | (r << 16) | (g << 8) | b
        colour = wt.DWORD(0)
        opaque = wt.BOOL(0)
        if _dwmapi.DwmGetColorizationColor(ctypes.byref(colour), ctypes.byref(opaque)) == 0:
            c = colour.value  # 0xAARRGGBB, blended colorization colour
            return ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF), c | 0xFF000000
        return (0, 120, 212), 0xFF0078D4

    def _text_scale() -> float:
        raw = _reg_dword(ACCESSIBILITY_KEY, "TextScaleFactor")
        if raw is None:
            return 1.0
        return max(TEXT_SCALE_MIN, min(TEXT_SCALE_MAX, raw)) / 100.0


def _qt_dark_mode() -> Optional[bool]:
    app = QGuiApplication.instance()
    if app is None:
        return None
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Dark:
        return True
    if scheme == Qt.ColorScheme.Light:
        return False
    return None


def read_signals() -> AppearanceSignals:
    """Read a fresh snapshot. Safe to call from any thread on Windows (pure Win32 + winreg)."""
    if not IS_WINDOWS:
        dark = _qt_dark_mode()
        return AppearanceSignals(dark_mode=bool(dark), source="qt" if dark is not None else "default")
    build = _windows_build()
    transparency = _reg_dword(PERSONALIZE_KEY, "EnableTransparency")
    light = _reg_dword(PERSONALIZE_KEY, "AppsUseLightTheme")
    anim = _client_area_animation()
    hc_on, hc_scheme = _high_contrast()
    accent_rgb, accent_argb = _accent()
    if light is None:
        qt_dark = _qt_dark_mode()
        dark = bool(qt_dark) if qt_dark is not None else False
    else:
        dark = light == 0
    return AppearanceSignals(
        transparency=True if transparency is None else bool(transparency),
        reduce_motion=(anim is False),
        high_contrast=hc_on,
        high_contrast_scheme=hc_scheme,
        dark_mode=dark,
        accent=accent_rgb,
        accent_argb=accent_argb,
        composition=_composition(),
        rounded_corners=build >= 22000,
        windows_build=build,
        source="win32",
        text_scale=_text_scale(),
    )


# ------------------------------------------------------------------------------------------------ watcher
class _NativeFilter(QAbstractNativeEventFilter):
    """Forwards the interesting broadcast messages to the watcher. Must be kept alive by its owner."""

    def __init__(self, on_message: Callable[[int, int, str], None]) -> None:
        super().__init__()
        self._on_message = on_message

    def nativeEventFilter(self, event_type, message):  # noqa: N802 (Qt override)
        if IS_WINDOWS:
            try:
                msg = wt.MSG.from_address(int(message))
            except (TypeError, ValueError):
                return False, 0
            if msg.message in WATCHED_MESSAGES:
                detail = ""
                if msg.message == WM_SETTINGCHANGE and msg.lParam:
                    try:
                        detail = ctypes.wstring_at(msg.lParam)
                    except (OSError, ValueError):
                        detail = ""
                self._on_message(int(msg.message), int(msg.wParam), detail)
        return False, 0


class AppearanceWatcher(QObject):
    """Owns the current ``AppearanceSignals`` and emits ``changed(AppearanceSignals)`` when any field changes.

    Create it once on the GUI thread after the QGuiApplication exists. ``current`` is always valid.
    ``last_trigger`` tells you what woke it up last ("native:0x1a:ImmersiveColorSet", "poll", "qt:colorScheme", ...).
    """

    changed = Signal(object)  # AppearanceSignals

    def __init__(self, parent: Optional[QObject] = None, *, poll_ms: int = 2000, coalesce_ms: int = 50) -> None:
        super().__init__(parent)
        self.current: AppearanceSignals = read_signals()
        self.last_trigger: str = "init"
        self.refresh_count: int = 0
        self._pending_trigger: str = ""
        app = QCoreApplication.instance()
        if app is None:
            raise RuntimeError("AppearanceWatcher needs a QCoreApplication")

        self._coalesce = QTimer(self)
        self._coalesce.setSingleShot(True)
        self._coalesce.setInterval(coalesce_ms)
        self._coalesce.timeout.connect(self._refresh_from_pending)

        self._poll = QTimer(self)
        self._poll.setInterval(poll_ms)
        self._poll.timeout.connect(lambda: self.refresh("poll"))
        if poll_ms > 0:
            self._poll.start()

        self._filter: Optional[_NativeFilter] = None
        self._sink: Optional[QWindow] = None
        if IS_WINDOWS:
            self._filter = _NativeFilter(self._on_native_message)
            app.installNativeEventFilter(self._filter)
            # Broadcast messages only reach top-level HWNDs; own one so we never depend on app windows existing.
            self._sink = QWindow()
            self._sink.setFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
            self._sink.resize(1, 1)
            self._sink.create()

        gui = QGuiApplication.instance()
        if gui is not None:
            hints = gui.styleHints()
            hints.colorSchemeChanged.connect(lambda _s: self.schedule("qt:colorScheme"))
            accessibility = getattr(hints, "accessibility", None)
            if accessibility is not None:
                acc = accessibility()
                sig = getattr(acc, "contrastPreferenceChanged", None)
                if sig is not None:
                    sig.connect(lambda _c: self.schedule("qt:contrastPreference"))

    # -- wiring
    def _on_native_message(self, message: int, wparam: int, detail: str) -> None:
        self.schedule("native:0x%x:%s" % (message, detail or hex(wparam)))

    def schedule(self, trigger: str) -> None:
        """Coalesce bursts (Qt delivers posted messages to native filters twice) into one refresh."""
        self._pending_trigger = trigger
        self._coalesce.start()

    @Slot()
    def _refresh_from_pending(self) -> None:
        self.refresh(self._pending_trigger or "native")

    def refresh(self, trigger: str = "manual") -> bool:
        """Re-read everything; emit ``changed`` and return True if anything differs."""
        self.refresh_count += 1
        fresh = read_signals()
        if fresh != self.current:
            self.current = fresh
            self.last_trigger = trigger
            self.changed.emit(fresh)
            return True
        return False

    def stop(self) -> None:
        self._poll.stop()
        self._coalesce.stop()
        app = QCoreApplication.instance()
        if app is not None and self._filter is not None:
            app.removeNativeEventFilter(self._filter)
        self._filter = None
        if self._sink is not None:
            self._sink.destroy()
            self._sink = None


# ------------------------------------------------------------------------------------------------ override layer
def parse_overrides(text: Optional[str]) -> List[str]:
    """Split ``GLASSTRANSLATE_APPEARANCE`` into lower-case tokens (empty ones dropped, order kept)."""
    if not text:
        return []
    return [tok.strip().lower() for tok in text.split(",") if tok.strip()]


def apply_overrides(sig: AppearanceSignals, tokens: Sequence[str]) -> AppearanceSignals:
    """Apply the override tokens to ``sig`` in order (later tokens win) and return the result.

    Unknown tokens and out-of-range text scales are logged at WARNING and ignored.
    """
    for tok in tokens:
        if tok == "glass":
            sig = replace(sig, transparency=True, composition=True, high_contrast=False)
        elif tok == "solid":
            sig = replace(sig, transparency=False)
        elif tok == "hc":
            sig = replace(sig, high_contrast=True)
        elif tok == "reducemotion":
            sig = replace(sig, reduce_motion=True)
        elif tok == "motion":
            sig = replace(sig, reduce_motion=False)
        elif tok == "dark":
            sig = replace(sig, dark_mode=True)
        elif tok == "light":
            sig = replace(sig, dark_mode=False)
        elif tok.startswith("textscale="):
            raw = tok.split("=", 1)[1]
            try:
                percent = int(raw)
            except ValueError:
                log.warning("%s: ignoring textscale=%r (not an integer)", OVERRIDE_ENV, raw)
                continue
            if not TEXT_SCALE_MIN <= percent <= TEXT_SCALE_MAX:
                log.warning("%s: ignoring textscale=%d (allowed %d..%d)", OVERRIDE_ENV, percent,
                            TEXT_SCALE_MIN, TEXT_SCALE_MAX)
                continue
            sig = replace(sig, text_scale=percent / 100.0)
        else:
            log.warning("%s: ignoring unknown token %r", OVERRIDE_ENV, tok)
    return sig


# ------------------------------------------------------------------------------------------------ QML object
class Appearance(QObject):
    """QML-facing view of the appearance signals (context property ``appearance``).

    Wraps an :class:`AppearanceWatcher` (created here unless one is passed) and re-applies
    the ``GLASSTRANSLATE_APPEARANCE`` overrides to every snapshot, so a watcher re-emit can
    never undo them.  All properties notify through the single ``changed`` signal.
    ``watch=False`` builds a static object (tests, non-GUI tools): feed it with :meth:`apply`.
    """

    changed = Signal()

    def __init__(
        self,
        parent: Optional[QObject] = None,
        *,
        watcher: Optional[AppearanceWatcher] = None,
        overrides: Optional[str] = None,
        watch: bool = True,
    ) -> None:
        super().__init__(parent)
        env = os.environ.get(OVERRIDE_ENV) if overrides is None else overrides
        self._tokens: List[str] = parse_overrides(env)
        if self._tokens:
            log.info("appearance overrides from %s: %s", OVERRIDE_ENV, ",".join(self._tokens))
        self._watcher: Optional[AppearanceWatcher] = watcher
        if self._watcher is None and watch and QCoreApplication.instance() is not None:
            self._watcher = AppearanceWatcher(self)
        base = self._watcher.current if self._watcher is not None else read_signals()
        self._base: AppearanceSignals = base
        self._signals: AppearanceSignals = apply_overrides(base, self._tokens)
        if self._watcher is not None:
            self._watcher.changed.connect(self._on_watcher_changed)

    # -- python API
    @property
    def signals(self) -> AppearanceSignals:
        """The effective snapshot (system values with the overrides applied)."""
        return self._signals

    @property
    def tokens(self) -> List[str]:
        """The override tokens in effect."""
        return list(self._tokens)

    @property
    def watcher(self) -> Optional[AppearanceWatcher]:
        return self._watcher

    def apply(self, base: AppearanceSignals) -> bool:
        """Adopt a raw system snapshot (overrides re-applied); True and ``changed`` if anything differs."""
        self._base = base
        effective = apply_overrides(base, self._tokens)
        if effective == self._signals:
            return False
        self._signals = effective
        self.changed.emit()
        return True

    def stop(self) -> None:
        """Stop the watcher (native filter, poll timer, sink window)."""
        if self._watcher is not None:
            self._watcher.stop()

    @Slot(object)
    def _on_watcher_changed(self, fresh: object) -> None:
        if isinstance(fresh, AppearanceSignals):
            self.apply(fresh)

    # -- QML properties (one shared notify signal)
    def _mode(self) -> str:
        return mode_for(self._signals)

    def _transparency(self) -> bool:
        return self._signals.transparency

    def _composition(self) -> bool:
        return self._signals.composition

    def _reduce_motion(self) -> bool:
        return self._signals.reduce_motion

    def _high_contrast(self) -> bool:
        return self._signals.high_contrast

    def _dark_mode(self) -> bool:
        return self._signals.dark_mode

    def _accent(self) -> QColor:
        r, g, b = self._signals.accent
        return QColor(r, g, b)

    def _text_scale(self) -> float:
        return float(self._signals.text_scale)

    def _glass_allowed(self) -> bool:
        return self._signals.glass_allowed

    mode = Property(str, _mode, notify=changed)
    transparency = Property(bool, _transparency, notify=changed)
    composition = Property(bool, _composition, notify=changed)
    reduceMotion = Property(bool, _reduce_motion, notify=changed)
    highContrast = Property(bool, _high_contrast, notify=changed)
    darkMode = Property(bool, _dark_mode, notify=changed)
    accent = Property(QColor, _accent, notify=changed)
    textScale = Property(float, _text_scale, notify=changed)
    glassAllowed = Property(bool, _glass_allowed, notify=changed)


if __name__ == "__main__":  # quick manual check: print snapshot + timing
    app = QGuiApplication(sys.argv)
    t0 = time.perf_counter()
    for _ in range(100):
        snap = read_signals()
    dt = (time.perf_counter() - t0) / 100 * 1000
    print(snap)
    print("read_signals(): %.3f ms per call" % dt)
    QTimer.singleShot(0, app.quit)
    app.exec()
