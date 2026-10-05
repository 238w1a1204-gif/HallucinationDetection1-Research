"""Relationship / predicate checking between a claim and its evidence.

This module answers: "does the evidence actually discuss the same kind of
factual relationship that the claim asserts?"

The relationship taxonomy lives entirely in entity_matcher.ROLE_KEYWORDS.
Adding a new relationship category there automatically makes it checkable
here — no code changes required in this file.

How it works
------------
1. Extract role/predicate keywords from the claim via entity_matcher.
2. Identify which ROLE_KEYWORDS category (or categories) those keywords
   belong to — this is the claimed relationship type.
3. Check whether the evidence text also contains keywords from the SAME
   category.
4. Return a structured result including the matched category (if any)
   and the coverage score.

Coverage score
--------------
A float in [0.0, 1.0] measuring how well the evidence covers the
claimed relationship:

  0.0  — evidence has NO keywords from any category the claim uses
  0.5  — evidence has keywords from a RELATED but different category
          (e.g. claim says "founded", evidence says "established" — same
          category, different keyword)
  1.0  — evidence has keywords from the EXACT SAME category as the claim

DESIGN PRINCIPLE: No entity names, dates, locations, or company names are
hard-coded here.  All generalisation is through the keyword taxonomy.

DOCUMENTED LIMITATIONS:
- Keyword matching is substring-based; unusual phrasings will be missed.
- A claim can belong to multiple relationship categories simultaneously
  (e.g. "she was born in Paris and founded a company") — the module
  returns the FIRST fully matching category and the best coverage score.
- Temporal distinctions ("was CEO" vs "is CEO") are not resolved.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.verification.entity_matcher import ROLE_KEYWORDS, extract_role_keywords


@dataclass
class RelationshipCheckResult:
    """Structured outcome of a relationship check."""
    # True when evidence discusses the SAME relationship category as the claim,
    # OR when the claim contains no relationship keyword (neutral — no penalty).
    relationship_discussed: bool
    # True when at least ONE keyword from the claim appears in the evidence,
    # even if the categories differ.
    any_keyword_match: bool
    # The name of the matched category (e.g. "leadership"), or None.
    matched_category: str | None
    # Coverage score in [0.0, 1.0] — see module docstring.
    coverage_score: float
    # Human-readable explanation.
    reason: str


def check_relationship(claim: str, evidence_text: str) -> RelationshipCheckResult:
    """Determine whether the evidence discusses the same relationship type
    as the claim.

    Returns a RelationshipCheckResult.  Callers should use:
    - relationship_discussed  to decide whether to upgrade/downgrade the verdict
    - coverage_score          as an input to confidence calculation
    - matched_category        for human-readable explanation
    """
    claim_kws     = extract_role_keywords(claim)
    evidence_kws  = extract_role_keywords(evidence_text)

    # ── Claim has no relationship keyword → neutral, no penalty ──────────
    if not claim_kws:
        return RelationshipCheckResult(
            relationship_discussed=True,
            any_keyword_match=False,
            matched_category=None,
            coverage_score=1.0,
            reason="Claim contains no recognisable relationship keyword; check skipped.",
        )

    # ── Find every category the claim's keywords belong to ───────────────
    claim_categories: dict[str, set[str]] = {}   # category → claim kws in it
    for category, keywords in ROLE_KEYWORDS.items():
        cat_kws = set(keywords)
        overlap = claim_kws & cat_kws
        if overlap:
            claim_categories[category] = overlap

    # ── Check if evidence covers any of those categories ─────────────────
    best_category: str | None = None
    best_coverage: float = 0.0
    any_match = bool(claim_kws & evidence_kws)

    for category, claim_cat_kws in claim_categories.items():
        cat_kws = set(ROLE_KEYWORDS[category])
        evidence_in_cat = evidence_kws & cat_kws
        if evidence_in_cat:
            # Full match — evidence has keywords in the same category
            coverage = 1.0
            if coverage > best_coverage:
                best_coverage = coverage
                best_category = category

    if best_category is not None:
        evidence_in_cat = evidence_kws & set(ROLE_KEYWORDS[best_category])
        return RelationshipCheckResult(
            relationship_discussed=True,
            any_keyword_match=True,
            matched_category=best_category,
            coverage_score=best_coverage,
            reason=(
                f"Evidence discusses the '{best_category}' relationship "
                f"(evidence keywords: {sorted(evidence_in_cat)})."
            ),
        )

    # ── No category match — check for any keyword overlap ────────────────
    if any_match:
        # Keywords overlap but not in a recognised shared category — partial
        return RelationshipCheckResult(
            relationship_discussed=False,
            any_keyword_match=True,
            matched_category=None,
            coverage_score=0.3,
            reason=(
                f"Evidence shares some keywords with the claim "
                f"({sorted(claim_kws & evidence_kws)}) but they do not "
                f"belong to the same relationship category."
            ),
        )

    # ── No keyword overlap at all ─────────────────────────────────────────
    return RelationshipCheckResult(
        relationship_discussed=False,
        any_keyword_match=False,
        matched_category=None,
        coverage_score=0.0,
        reason=(
            f"Claim asserts {sorted(claim_kws)} relationship but evidence "
            f"contains no matching relationship keywords."
        ),
    )
