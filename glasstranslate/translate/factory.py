"""Build the translation backend selected in :class:`AppConfig`."""
from __future__ import annotations

import logging
from typing import List

from ..config.settings import AppConfig
from ..core.interfaces import Translator
from .argos import ArgosCT2Translator
from .identity import IdentityTranslator
from .libre import LibreTranslateTranslator

log = logging.getLogger(__name__)

_BACKENDS = ("argos", "libretranslate", "identity")


def available_backends() -> List[str]:
    """Names accepted for ``AppConfig.translation_backend``."""
    return list(_BACKENDS)


def create_translator(cfg: AppConfig) -> Translator:
    """Instantiate the configured backend.  Unknown names fall back to the
    identity translator with a warning rather than failing the pipeline."""
    backend = cfg.translation_backend
    if backend == "argos":
        return ArgosCT2Translator(cfg.models_dir, device=cfg.translate_device)
    if backend == "libretranslate":
        return LibreTranslateTranslator(cfg.translation_api_url, api_key=cfg.translation_api_key)
    if backend != "identity":
        log.warning("Unknown translation backend %r; using identity", backend)
    return IdentityTranslator()
