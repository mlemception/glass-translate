"""API keys and other secrets, stored OUTSIDE ``config.json``.

``%LOCALAPPDATA%/GlassTranslate/secrets.json`` (see :func:`~glasstranslate.config
.settings.secrets_dir`; ``<portable root>/config/secrets.json`` in a portable
bundle) holds a flat ``{"name": "value"}`` map.  An environment variable (for Gemini:
``GEMINI_API_KEY``) wins over the file when it is set and non-empty, so the
frozen exe and CI can inject keys without touching the disk.

Guarantees:

* reads never raise: a missing, corrupt or unreadable file yields ``""`` and a
  single logged warning (which names the file, never a value);
* writes are atomic (``.tmp`` + ``os.replace``) and best-effort ``0o600``;
* no function here logs or returns a secret except the ``get_*`` accessors -
  use :func:`redact` for anything user-visible.

``GLASSTRANSLATE_SECRETS_FILE`` overrides the file location (tests).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from .settings import secrets_dir

log = logging.getLogger(__name__)

SECRETS_FILE_NAME = "secrets.json"
SECRETS_FILE_ENV = "GLASSTRANSLATE_SECRETS_FILE"

GEMINI_API_KEY = "gemini_api_key"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

_FILE_MODE = 0o600
_MASK = "••••"  # four bullets
_TAIL_MIN_LEN = 12  # only values at least this long reveal a tail
_TAIL_CHARS = 2


# ------------------------------------------------------------------- paths
def secrets_path() -> Path:
    """Where the secrets file lives: ``GLASSTRANSLATE_SECRETS_FILE`` when set,
    else ``secrets_dir()/secrets.json`` - the user data dir on a normal install
    (never next to ``config.json`` in ``%APPDATA%``; secrets belong with the
    other per-machine data) and ``<portable root>/config`` in a portable
    bundle, whose config folder holds both."""
    override = os.environ.get(SECRETS_FILE_ENV, "").strip()
    if override:
        return Path(override)
    return secrets_dir() / SECRETS_FILE_NAME


# ------------------------------------------------------------------- store
def _read_store(path: Path) -> Dict[str, str]:
    """Load the flat string map.  Any problem degrades to ``{}`` with one
    warning; non-string values are dropped so callers only ever see ``str``."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Could not read secrets file %s (%s); treating it as empty", path, type(exc).__name__)
        return {}
    if not isinstance(data, dict):
        log.warning("Secrets file %s is not a JSON object; treating it as empty", path)
        return {}
    return {str(name): value for name, value in data.items() if isinstance(value, str)}


def _restrict_permissions(path: Path) -> None:
    """Best-effort ``0o600``; Windows only honours the read-only bit."""
    try:
        os.chmod(path, _FILE_MODE)
    except OSError:
        log.debug("Could not restrict permissions on %s", path, exc_info=True)


def _write_store(path: Path, store: Dict[str, str]) -> None:
    """Atomically replace ``path`` with ``store``; a failed write leaves no
    ``.tmp`` behind and re-raises the ``OSError`` for the caller."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        _restrict_permissions(tmp)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------- public API
def get_secret(name: str, *, env: Optional[str] = None) -> str:
    """Value of ``name``: the ``env`` variable (stripped) when set and
    non-empty, else the file entry (stripped), else ``""``.  Never raises."""
    if env:
        from_env = os.environ.get(env, "").strip()
        if from_env:
            return from_env
    return _read_store(secrets_path()).get(name, "").strip()


def has_secret(name: str, *, env: Optional[str] = None) -> bool:
    return bool(get_secret(name, env=env))


def set_secret(name: str, value: str) -> Path:
    """Store ``value`` (stripped) under ``name`` and return the file path.

    An empty value removes the key; other keys are preserved.  Removing from a
    file that does not exist (or does not hold the key) writes nothing, so a
    fresh install never gains an empty ``secrets.json``.  An unreadable
    existing file is replaced.  Write errors (``OSError``) propagate.
    """
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("secret name must not be empty")
    path = secrets_path()
    current = _read_store(path)
    clean_value = (value or "").strip()
    if not clean_value and clean_name not in current:
        return path
    without = {k: v for k, v in current.items() if k != clean_name}
    updated = {**without, clean_name: clean_value} if clean_value else without
    _write_store(path, updated)
    log.debug("Secret %r %s in %s", clean_name, "stored" if clean_value else "removed", path)
    return path


def redact(value: str) -> str:
    """Display form of a secret: ``""`` for empty, ``••••`` plus the last two
    characters for values of 12+ characters, plain ``••••`` otherwise.  The
    result never contains the value itself."""
    clean = (value or "").strip()
    if not clean:
        return ""
    if len(clean) >= _TAIL_MIN_LEN:
        return _MASK + clean[-_TAIL_CHARS:]
    return _MASK


# ---------------------------------------------------------------- gemini
def get_gemini_api_key() -> str:
    return get_secret(GEMINI_API_KEY, env=GEMINI_API_KEY_ENV)


def set_gemini_api_key(value: str) -> Path:
    return set_secret(GEMINI_API_KEY, value)


def has_gemini_api_key() -> bool:
    return has_secret(GEMINI_API_KEY, env=GEMINI_API_KEY_ENV)
