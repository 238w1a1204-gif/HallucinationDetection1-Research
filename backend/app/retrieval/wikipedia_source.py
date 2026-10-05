"""Wikipedia evidence source.

Strategy (updated):
1. Search for up to 5 candidate titles using the Wikipedia search API.
2. Pick the title that best matches the subject entity using a simple
   title-similarity heuristic — NOT always the first result, which may
   be a disambiguation page, a list, or a subsidiary article.
3. Fetch the full intro extract of the chosen article.
4. Return the extract, capped at 1200 chars so downstream models have
   enough context to find dates, names, and relationships.

Changes from original:
- Fetches 5 candidate titles instead of 1, then scores them.
- Title-matching prefers the title that contains the subject entity name
  over unrelated articles (e.g. "Albert Einstein" > "Mileva Marić" for
  a query about Einstein).
- Extract increased to 1200 chars to capture more factual detail.
"""
from __future__ import annotations

import logging
import re

import requests

from app.config import get_settings
from app.retrieval.base_source import EvidenceItem, EvidenceSource

logger = logging.getLogger(__name__)
settings = get_settings()

SEARCH_URL = "https://en.wikipedia.org/w/api.php"

_HEADERS = {
    "User-Agent": (
        "HallucinationDetectionBot/1.0 "
        "(Hallucination Detection & Answer Verification project; "
        "https://github.com/example/hallucination-detection)"
    )
}

# Patterns that indicate a bad article (disambiguation, list, category)
_BAD_TITLE_PATTERNS = [
    re.compile(r'\(disambiguation\)', re.I),
    re.compile(r'^List of', re.I),
    re.compile(r'^Index of', re.I),
    re.compile(r'^Category:', re.I),
    re.compile(r'^Template:', re.I),
    re.compile(r'^Wikipedia:', re.I),
]


def _is_bad_title(title: str) -> bool:
    return any(p.search(title) for p in _BAD_TITLE_PATTERNS)


def _title_matches_subject(title: str, query: str) -> float:
    """Return a score [0..1] for how well a Wikipedia article title matches
    the main subject in the query.

    Higher is better.  We prefer titles where the subject entity name
    appears as the main noun phrase (e.g. "Albert Einstein" matches
    "Albert Einstein" perfectly, but not "Mileva Marić").
    """
    # Tokenise both to lower-case words
    title_l = title.lower()
    query_l = query.lower()

    # Extract meaningful tokens from query (≥4 chars, not stopwords)
    _STOP = {"born", "died", "founded", "founder", "located", "location",
              "capital", "president", "discovered", "invented", "wrote",
              "what", "where", "when", "who", "which", "history", "founded"}
    query_tokens = [t for t in re.findall(r'\b[a-z]{3,}\b', query_l)
                    if t not in _STOP]

    if not query_tokens:
        return 0.5  # neutral

    # Score = fraction of query tokens that appear in the title
    matched = sum(1 for t in query_tokens if t in title_l)
    score = matched / len(query_tokens)

    # Bonus if the title is exactly or nearly the first part of the query
    # (e.g. query="Albert Einstein" → title="Albert Einstein" is perfect)
    if title_l.startswith(query_l[:min(len(query_l), len(title_l))]):
        score = min(1.0, score + 0.3)

    return score


class WikipediaSource(EvidenceSource):
    name = "wikipedia"

    def retrieve(self, claim: str) -> list[EvidenceItem]:
        try:
            title = self._search_best_title(claim)
            if not title:
                return []
            extract = self._fetch_extract(title)
            if not extract:
                return []
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            return [EvidenceItem(source_name=self.name, source_url=url,
                                 evidence_text=extract)]
        except Exception as exc:
            logger.warning("WikipediaSource failed for claim '%s': %s", claim, exc)
            return []

    def _search_best_title(self, claim: str) -> str | None:
        """Search Wikipedia and return the best-matching article title."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": claim,
            "format": "json",
            "srlimit": 5,   # fetch top 5, then pick the best
        }
        resp = requests.get(SEARCH_URL, params=params, headers=_HEADERS,
                            timeout=settings.RETRIEVAL_TIMEOUT_SECONDS)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])

        if not results:
            return None

        # Filter out clearly bad titles
        candidates = [r["title"] for r in results if not _is_bad_title(r["title"])]
        if not candidates:
            candidates = [results[0]["title"]]  # fallback: first result regardless

        # Pick the candidate whose title best matches the claim
        best_title = max(candidates, key=lambda t: _title_matches_subject(t, claim))
        return best_title

    def _fetch_extract(self, title: str) -> str | None:
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
        }
        resp = requests.get(SEARCH_URL, params=params, headers=_HEADERS,
                            timeout=settings.RETRIEVAL_TIMEOUT_SECONDS)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract")
            if extract:
                # Return up to 1200 chars — enough to capture birthplace, founder, etc.
                return extract.strip()[:1200]
        return None
