"""Session pieces of the glass profiling driver (dev only, Qt-free).

``tools/profile_glass.py --mode session|pipeline`` builds a real
:class:`~glasstranslate.ui.app.GlassTranslateApp` instead of a bare control window.  Two
things that needs are worth their own module, because both are testable without a display:

* :func:`prepare_config` - a **copy** of the user's ``config.json`` in a temp directory, with
  the three settings a measurement run must not inherit forced off.  Only that one file is
  copied: ``secrets.json`` (the Gemini key) never leaves the user's profile, and the session
  saves into the copy, so the user's own config is never written.
* :class:`AlternatingPageCapture` - two still pages served in turn in place of the screen, so
  the pipeline really OCRs, translates and typesets while the GUI is being measured, with a
  deterministic page cadence instead of whatever happens to be on screen.

:func:`effective_backend` and :func:`warn_if_online` sit next to the config copy for the same
reason: the backend a run would really use is read out of that copy, and anything outside
:data:`OFFLINE_BACKENDS` sends the pages' OCR text to a service with the key stored on this
machine - which a measurement run must say out loud before it starts.
"""
from __future__ import annotations

import gc
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from glasstranslate.capture.static import StaticPageCapture
from glasstranslate.config import settings
from glasstranslate.config.settings import AppConfig
from glasstranslate.core.interfaces import ScreenCapture
from glasstranslate.core.types import Frame, Rect

__all__ = ["DEFAULT_PAGE_PERIOD_S", "OFFLINE_BACKENDS", "PAGES", "AlternatingPageCapture",
           "effective_backend", "install_gc_marks", "name_gc_threads", "prepare_config", "warn_if_online"]

log = logging.getLogger(__name__)

# Both are tracked in git, so every checkout can run the pipeline mode.
PAGES: Tuple[str, ...] = ("Examples/before.jpg", "more_comparisons/1ja.jpg")
DEFAULT_PAGE_PERIOD_S = 2.0
# Backends that answer from a local model; everything else talks to a service over the network.
OFFLINE_BACKENDS: Tuple[str, ...] = ("argos", "identity")

PageMark = Callable[[int, str], None]
# (held page index or None when released, perf_counter at the release)
HoldState = Tuple[Optional[int], float]


def prepare_config(tmp_dir: Union[str, Path], translator: Optional[str] = None) -> Tuple[Path, AppConfig]:
    """Copy ``config.json`` into ``tmp_dir`` and return ``(path, config)`` for the run.

    The quality renderer is forced off (it downloads 9.3 GB and occupies the GPU, which would
    measure the sidecar rather than the GUI), manga mode on and auto-start off; ``translator``
    overrides ``translation_backend`` when it is given.  A user without a config file starts
    from the defaults and the copy is created by the session's first save.
    """
    destination = Path(tmp_dir) / "config.json"
    source = settings.default_config_path()
    if source.is_file():
        shutil.copyfile(source, destination)  # this file only: never secrets.json
        cfg = AppConfig.load(destination)
    else:
        log.info("no config at %s: the profiling session starts from the defaults", source)
        cfg = AppConfig()
    cfg.quality_renderer = "off"
    cfg.manga_mode = True
    cfg.running_on_start = False
    if translator:
        cfg.translation_backend = translator
    return destination, cfg


def warn_if_online(backend: Optional[str]) -> bool:
    """Warn when the run would call a network translator; returns whether it did."""
    name = str(backend or "").strip().lower()
    if not name or name in OFFLINE_BACKENDS:
        return False
    log.warning("translation backend %r is an online service: this run sends the OCR text of the "
                "example pages to it with the key stored on this machine and spends real quota. "
                "Pass --translator %s to keep the run offline.", name, "|".join(OFFLINE_BACKENDS))
    return True


def effective_backend(translator: Optional[str]) -> Optional[str]:
    """Which backend the session modes would use: ``--translator``, else the stored config's."""
    if translator:
        return str(translator)
    scratch = Path(tempfile.mkdtemp(prefix="glass-profile-cfg-"))
    try:
        _path, cfg = prepare_config(scratch, None)
        return str(cfg.translation_backend)
    except Exception:  # noqa: BLE001 - a warning we cannot raise must not stop the run
        log.warning("could not read the configured translation backend", exc_info=True)
        return None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)  # it held a copy of the user's config.json


GcEvent = Tuple[float, str, Dict[str, Any]]


def install_gc_marks() -> Tuple[List[GcEvent], Callable[[], None]]:
    """Record every garbage collection as a ``gc`` event (``gen``, ``ms``, ``thread``).

    A collection holds the interpreter lock on whichever thread started it, so a long one on
    the pipeline thread stalls the GUI thread exactly like one of its own.  Collections never
    nest and run under that lock, so one start time is enough.

    The callback runs **inside the collector**, which any allocation can enter - including the
    one ``Profiler.mark`` makes while it holds its lock.  Calling the profiler from here
    therefore deadlocks the thread against itself; the events go into a plain list instead
    (``list.append`` and ``get_ident`` take no lock) and the driver merges them into the dump
    afterwards, naming the threads with :func:`name_gc_threads`.

    Returns ``(events, uninstall)``; calling ``uninstall`` twice is harmless.
    """
    events: List[GcEvent] = []
    started: List[float] = []

    def on_gc(phase: str, info: Dict[str, Any]) -> None:
        if phase == "start":
            started[:] = [time.perf_counter()]
        elif started:
            now = time.perf_counter()
            events.append((now, "gc", {"gen": int(info.get("generation", -1)),
                                       "thread": threading.get_ident(),  # C call: no lock, no object
                                       "ms": (now - started.pop()) * 1000.0}))

    def uninstall() -> None:
        try:
            gc.callbacks.remove(on_gc)
        except ValueError:  # already removed
            pass

    gc.callbacks.append(on_gc)
    return events, uninstall


def name_gc_threads(events: Sequence[GcEvent]) -> List[GcEvent]:
    """The ``gc`` events with thread idents replaced by names (call while the threads still live).

    The collector callback records ``threading.get_ident()`` only: ``current_thread()`` can take
    a module lock and build objects, which has no place inside a collection.
    """
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    return [(t, kind, {**payload, "thread": names.get(payload.get("thread"), str(payload.get("thread")))})
            for t, kind, payload in events]


class AlternatingPageCapture(ScreenCapture):
    """Two still pages standing in for the screen, swapped every ``period_s``.

    ``hold(index)`` pins one page (the warm-up needs each page OCRed once before the
    measurement starts); ``release()`` starts the alternation from that moment.  ``grab`` runs
    on the pipeline's worker thread, so the hold state is published as **one** tuple
    (``held_index_or_None, released_at``) that the worker reads in a single attribute load: it
    can never see a half-updated pair, and no lock is needed.

    Args:
        paths: the two image files (anything :class:`StaticPageCapture` can decode).
        period_s: seconds each page is served once released.
        clock: monotonic seconds source (injected by the tests).
        on_page: called with ``(index, file_name)`` whenever the served page changes.
    """

    name = "alternating"

    def __init__(self, paths: Sequence[Union[str, Path]], period_s: float = DEFAULT_PAGE_PERIOD_S,
                 clock: Callable[[], float] = time.perf_counter, on_page: Optional[PageMark] = None) -> None:
        if len(paths) < 2:
            raise ValueError(f"AlternatingPageCapture needs two pages, got {len(paths)}")
        self._pages: List[StaticPageCapture] = [StaticPageCapture(p) for p in paths]
        self._names: List[str] = [Path(p).name for p in paths]  # the name only: never a path
        self._period_s = max(1e-3, float(period_s))
        self._clock = clock
        self._on_page = on_page
        self._state: HoldState = (0, 0.0)
        self._serves_since_hold = 0
        self._served = -1

    def hold(self, index: int) -> None:
        """Serve ``index`` until :meth:`release`."""
        self._serves_since_hold = 0
        self._state = (index % len(self._pages), 0.0)

    def release(self) -> None:
        """Start alternating from now."""
        self._state = (None, self._clock())

    def _index_of(self, state: HoldState) -> int:
        held, released_at = state
        if held is not None:
            return held
        return int((self._clock() - released_at) / self._period_s) % len(self._pages)

    @property
    def index(self) -> int:
        """Which page is served right now."""
        return self._index_of(self._state)

    @property
    def serves_since_hold(self) -> int:
        """Grabs of the currently held page since :meth:`hold` (never grows once released)."""
        return self._serves_since_hold

    def grab(self, region: Rect) -> Optional[Frame]:
        state = self._state  # one load: the index and the release time always belong together
        index = self._index_of(state)
        if state[0] is not None:
            self._serves_since_hold += 1
        if index != self._served:
            self._served = index
            if self._on_page is not None:
                self._on_page(index, self._names[index])
        return self._pages[index].grab(region)
