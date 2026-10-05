"""Calls all evidence sources concurrently and merges their results.

Each source already guarantees it won't raise, but we wrap the call in a
thread pool + try/except anyway as defense in depth so one truly
unexpected error can never take down the whole retrieval step.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.retrieval.base_source import EvidenceItem, EvidenceSource
from app.retrieval.duckduckgo_source import DuckDuckGoSource
from app.retrieval.wikidata_source import WikidataSource
from app.retrieval.wikipedia_source import WikipediaSource

logger = logging.getLogger(__name__)

_SOURCES: list[EvidenceSource] = [WikipediaSource(), WikidataSource(), DuckDuckGoSource()]


def retrieve_evidence(claim: str) -> list[EvidenceItem]:
    """Query every configured source in parallel and return combined evidence.

    If a source fails or times out, it simply contributes no evidence -
    the overall call still succeeds with whatever the other sources found.
    """
    evidence: list[EvidenceItem] = []
    with ThreadPoolExecutor(max_workers=len(_SOURCES)) as executor:
        futures = {executor.submit(_safe_retrieve, source, claim): source for source in _SOURCES}
        for future in as_completed(futures):
            source = futures[future]
            try:
                items = future.result()
                evidence.extend(items)
            except Exception as exc:  # pragma: no cover - extra safety net
                logger.warning("Source %s raised unexpectedly: %s", source.name, exc)
    return evidence


def _safe_retrieve(source: EvidenceSource, claim: str) -> list[EvidenceItem]:
    try:
        return source.retrieve(claim)
    except Exception as exc:  # pragma: no cover - extra safety net
        logger.warning("Source %s failed for claim '%s': %s", source.name, claim, exc)
        return []
