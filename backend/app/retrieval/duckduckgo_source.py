"""DuckDuckGo Instant Answer API evidence source.

DDG's Instant Answer API (https://api.duckduckgo.com/) is free and needs
no key, but its coverage is inconsistent - it only returns a good
"AbstractText" for well-known topics/entities. For many claims it will
return nothing, which is expected and handled gracefully (empty list).
"""
from __future__ import annotations

import logging

import requests

from app.config import get_settings
from app.retrieval.base_source import EvidenceItem, EvidenceSource

logger = logging.getLogger(__name__)
settings = get_settings()

DDG_API_URL = "https://api.duckduckgo.com/"


class DuckDuckGoSource(EvidenceSource):
    name = "duckduckgo"

    def retrieve(self, claim: str) -> list[EvidenceItem]:
        try:
            params = {
                "q": claim,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1,
                "skip_disambig": 1,
            }
            resp = requests.get(DDG_API_URL, params=params, timeout=settings.RETRIEVAL_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()

            abstract_text = (data.get("AbstractText") or "").strip()
            abstract_url = data.get("AbstractURL") or None

            if abstract_text:
                return [
                    EvidenceItem(
                        source_name=self.name,
                        source_url=abstract_url,
                        evidence_text=abstract_text[:800],
                    )
                ]

            # Fall back to the first related topic, if present.
            related = data.get("RelatedTopics") or []
            for item in related:
                text = (item.get("Text") or "").strip()
                if text:
                    return [
                        EvidenceItem(
                            source_name=self.name,
                            source_url=item.get("FirstURL"),
                            evidence_text=text[:800],
                        )
                    ]
            return []
        except Exception as exc:  # pragma: no cover - defensive network handling
            logger.warning("DuckDuckGoSource failed for claim '%s': %s", claim, exc)
            return []
