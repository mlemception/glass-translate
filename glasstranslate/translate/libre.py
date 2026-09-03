"""Online backend speaking the LibreTranslate HTTP API (stdlib ``urllib`` only).

Any network or protocol failure degrades to returning the inputs unchanged so
the overlay keeps working (showing the source text) while the server is down.
"""
from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any, Iterable, List, Optional, Sequence

from ..core.interfaces import Translator

log = logging.getLogger(__name__)

_USER_AGENT = "GlassTranslate/1.0 (+https://github.com/glasstranslate)"


class LibreTranslateTranslator(Translator):
    """Client for a LibreTranslate-compatible server such as
    ``https://libretranslate.com`` or a self-hosted instance."""

    name = "libretranslate"
    device = "remote"
    offline = False

    def __init__(self, url: str, api_key: str = "", timeout: float = 15.0) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._pairs: Optional[set[tuple[str, str]]] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------- helpers
    def _request(self, endpoint: str, payload: Optional[dict] = None) -> Any:
        """POST ``payload`` as JSON (or GET when ``payload`` is None) and return
        the decoded JSON body.  Raises on any transport or decoding error."""
        data = None
        headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.url + endpoint, data=data, headers=headers, method="POST" if data else "GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _load_pairs(self) -> set[tuple[str, str]]:
        """Fetch ``/languages`` once; every language lists its valid targets."""
        pairs: set[tuple[str, str]] = set()
        try:
            languages = self._request("/languages")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            log.warning("LibreTranslate /languages failed (%s); pairs unknown", exc)
            return pairs
        if not isinstance(languages, list):
            return pairs
        codes = [str(lang.get("code")) for lang in languages if isinstance(lang, dict) and lang.get("code")]
        for lang in languages:
            if not isinstance(lang, dict) or not lang.get("code"):
                continue
            src = str(lang["code"])
            targets = lang.get("targets")
            for tgt in (targets if isinstance(targets, list) else codes):
                if str(tgt) != src:
                    pairs.add((src, str(tgt)))
        return pairs

    # -------------------------------------------------------- Translator API
    def supported_pairs(self) -> Iterable[tuple[str, str]]:
        with self._lock:
            if self._pairs is None:
                self._pairs = self._load_pairs()
            return set(self._pairs)

    def supports(self, src: str, tgt: str) -> bool:
        pairs = set(self.supported_pairs())
        # If the server could not be reached we do not know; let translate_batch try.
        return (src, tgt) in pairs if pairs else True

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        texts = list(texts)
        indices = [i for i, t in enumerate(texts) if t.strip()]
        if not indices:
            return texts
        payload = {
            "q": [texts[i] for i in indices],
            "source": src,
            "target": tgt,
            "format": "text",
            "api_key": self.api_key,
        }
        try:
            body = self._request("/translate", payload)
            translated = body["translatedText"]
            if isinstance(translated, str):
                translated = [translated]
            if len(translated) != len(indices):
                raise ValueError("server returned %d translations for %d inputs" % (len(translated), len(indices)))
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("LibreTranslate /translate failed (%s); returning inputs", exc)
            return texts
        out = list(texts)
        for i, tr in zip(indices, translated):
            out[i] = str(tr)
        return out
