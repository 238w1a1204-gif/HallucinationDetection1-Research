"""Abstract interface for an evidence source (Wikipedia, Wikidata, DDG, ...).

Every source promises to return a list of EvidenceItem given a claim
string, and to NEVER raise an exception upward (it should catch its own
network/parsing errors and return an empty list instead). This lets the
aggregator continue with whatever sources succeeded, per the requirement
that one source failing should not break retrieval.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EvidenceItem:
    source_name: str
    source_url: str | None
    evidence_text: str


class EvidenceSource(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(self, claim: str) -> list[EvidenceItem]:
        """Return evidence snippets relevant to `claim`. Must not raise."""
        raise NotImplementedError
