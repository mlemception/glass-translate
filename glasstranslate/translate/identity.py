"""Pass-through translator used to exercise the overlay without any model."""
from __future__ import annotations

from typing import Iterable, List, Sequence

from ..core.interfaces import Translator


class IdentityTranslator(Translator):
    """Returns every input unchanged and claims to support every pair."""

    name = "identity"
    device = "cpu"
    offline = True

    def supported_pairs(self) -> Iterable[tuple[str, str]]:
        """The set of pairs is unbounded, so nothing can be enumerated; see
        :meth:`supports`."""
        return ()

    def supports(self, src: str, tgt: str) -> bool:
        return True

    def translate_batch(self, texts: Sequence[str], src: str, tgt: str) -> List[str]:
        return list(texts)
