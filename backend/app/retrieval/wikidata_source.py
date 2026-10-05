"""Wikidata evidence source.

Strategy:
1. Use the wbsearchentities action to find the best-matching entity for
   the claim (using key terms from the claim as the search string).
2. Use the entity's description + label as a short evidence snippet.

Wikidata entries are short factual descriptions (e.g. "Albert Einstein:
German-born theoretical physicist (1879-1955)"), which are useful as a
second, independent evidence source distinct from Wikipedia's prose.
"""
from __future__ import annotations

import logging

import requests

from app.config import get_settings
from app.retrieval.base_source import EvidenceItem, EvidenceSource

logger = logging.getLogger(__name__)
settings = get_settings()

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

_HEADERS = {
    "User-Agent": (
        "HallucinationDetectionBot/1.0 "
        "(Hallucination Detection & Answer Verification project)"
    )
}


class WikidataSource(EvidenceSource):
    name = "wikidata"

    def retrieve(self, claim: str) -> list[EvidenceItem]:
        try:
            entity = self._search_entity(claim)
            if not entity:
                return []
            label = entity.get("label", "")
            description = entity.get("description", "")
            entity_id = entity.get("id")
            if not description:
                return []
            evidence_text = f"{label}: {description}"
            url = f"https://www.wikidata.org/wiki/{entity_id}" if entity_id else None
            return [EvidenceItem(source_name=self.name, source_url=url, evidence_text=evidence_text)]
        except Exception as exc:  # pragma: no cover - defensive network handling
            logger.warning("WikidataSource failed for claim '%s': %s", claim, exc)
            return []

    def _search_entity(self, claim: str) -> dict | None:
        params = {
            "action": "wbsearchentities",
            "search": claim,
            "language": "en",
            "format": "json",
            "limit": 1,
        }
        resp = requests.get(WIKIDATA_API_URL, params=params, headers=_HEADERS, timeout=settings.RETRIEVAL_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("search", [])
        if not results:
            return None
        top = results[0]
        return {
            "id": top.get("id"),
            "label": top.get("label", ""),
            "description": top.get("description", ""),
        }
