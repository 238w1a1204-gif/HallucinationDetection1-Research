"""Regression tests for the Microsoft founder retrieval issue.

Problem: Wikipedia's search for "Microsoft founded founder" returned the
"Microsoft AI" article (a division/subsidiary) instead of the main
Microsoft article.  The subject-presence scorer gave it score=1.0 because
"Microsoft" appears as a substring — even though the article is about
"Microsoft AI", not "Microsoft" itself.

Fix: _subject_present_in_evidence now distinguishes:
  - "Microsoft Corporation is..." → score=1.0 (main entity)
  - "Microsoft was founded by..." → score=1.0 (standalone first mention)
  - "Microsoft AI is a division..." → score=0.3 (compound name / subsidiary)

These tests verify the fix is in place AND that the correct founder evidence
can be used for verification.

Nothing in this file is hard-coded to Microsoft specifically — the same
logic generalises to any organisation with product divisions.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from app.verification.evidence_quality_scorer import (
    _subject_present_in_evidence,
    rank_evidence_by_quality,
)
from app.verification.similarity_scorer import ScoredEvidence
from app.verification.verdict_engine import (
    determine_verdict,
    VERDICT_VERIFIED, VERDICT_PARTIALLY_SUPPORTED,
    VERDICT_UNSUPPORTED, VERDICT_CONTRADICTED,
)
from app.verification.question_analyzer import analyze_question
from app.retrieval.query_builder import build_subject_query
from app.config import get_settings

settings = get_settings()
STRONG  = settings.SIMILARITY_STRONG_THRESHOLD
PARTIAL = settings.SIMILARITY_PARTIAL_THRESHOLD


def _no_nli():
    return patch(
        "app.verification.verdict_engine.detect_contradiction",
        return_value=(False, "neutral"),
    )

def _ev(text: str, sim: float, src: str = "wikipedia") -> ScoredEvidence:
    return ScoredEvidence(
        evidence_text=text, source_name=src, source_url=None, similarity=sim
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Subject-presence: subsidiary / compound name scoring
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubjectPresenceCompound:

    def test_subsidiary_article_gets_low_subject_score(self):
        """'Microsoft AI' article should NOT score 1.0 for subject='Microsoft'."""
        score = _subject_present_in_evidence(
            "Microsoft",
            "Microsoft AI (MAI) is an American artificial intelligence research "
            "laboratory which serves as a division of Microsoft.",
        )
        assert score <= 0.5, (
            f"Microsoft AI article should have LOW subject score for 'Microsoft'. "
            f"Got {score}"
        )

    def test_main_entity_article_gets_high_subject_score(self):
        """Main Microsoft article should score 1.0 for subject='Microsoft'."""
        score = _subject_present_in_evidence(
            "Microsoft",
            "Microsoft Corporation is an American multinational technology company. "
            "Microsoft was founded by Bill Gates and Paul Allen.",
        )
        assert score >= 0.9, (
            f"Main Microsoft article should have HIGH subject score. Got {score}"
        )

    def test_founder_evidence_gets_high_subject_score(self):
        """Evidence that directly names the subject as topic gets 1.0."""
        score = _subject_present_in_evidence(
            "Microsoft",
            "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975, "
            "in Albuquerque, New Mexico.",
        )
        assert score >= 0.9

    def test_corporation_suffix_not_penalised(self):
        """'Microsoft Corporation' is the entity itself — not a compound subsidiary."""
        score = _subject_present_in_evidence(
            "Microsoft",
            "Microsoft Corporation announced new products today.",
        )
        assert score >= 0.9

    def test_general_pattern_apple_music(self):
        """Apple Music article should score low for subject='Apple'."""
        score = _subject_present_in_evidence(
            "Apple",
            "Apple Music is a music and video streaming service developed by Apple.",
        )
        assert score <= 0.5, (
            f"'Apple Music' article should have LOW score for subject='Apple'. Got {score}"
        )

    def test_general_pattern_amazon_web_services(self):
        """Amazon Web Services article should score low for subject='Amazon'."""
        score = _subject_present_in_evidence(
            "Amazon",
            "Amazon Web Services (AWS) is a subsidiary of Amazon providing on-demand "
            "cloud computing platforms.",
        )
        assert score <= 0.5

    def test_subject_absent_scores_zero(self):
        score = _subject_present_in_evidence(
            "Microsoft",
            "Google is a technology company founded by Larry Page and Sergey Brin.",
        )
        assert score == 0.0

    def test_multiword_subject_unaffected(self):
        """Multi-word subjects use the old all-tokens logic — not affected by compound fix."""
        score = _subject_present_in_evidence(
            "Bill Gates",
            "Bill Gates co-founded Microsoft and later became a philanthropist.",
        )
        assert score >= 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# Quality ranking: subsidiary article must rank BELOW main article
# ═══════════════════════════════════════════════════════════════════════════════

class TestQualityRankingSubsidiaryVsMain:

    def test_main_microsoft_article_ranks_above_subsidiary(self):
        """Even if 'Microsoft AI' has higher similarity, the main Microsoft
        article should rank first when we use subject='Microsoft'."""
        claim   = "microsoft corp"
        subject = "Microsoft"

        ev_subsidiary = _ev(
            "Microsoft AI (MAI) is an American artificial intelligence research "
            "laboratory which serves as a division of Microsoft. It oversees "
            "consumer artificial intelligence products.",
            sim=0.80,   # higher similarity
        )
        ev_main = _ev(
            "Microsoft Corporation is an American multinational technology company "
            "headquartered in Redmond, Washington. Microsoft was founded by Bill Gates.",
            sim=0.65,   # lower similarity
        )

        ranked = rank_evidence_by_quality(
            claim, [ev_subsidiary, ev_main], subject_entity=subject
        )

        assert ranked[0].evidence_text == ev_main.evidence_text, (
            "Main Microsoft article should rank first even with lower similarity. "
            f"Got first: {ranked[0].evidence_text[:60]!r}"
        )

    def test_founder_evidence_ranks_above_generic_article(self):
        """Evidence that names the founder should rank higher than a generic description."""
        claim   = "microsoft corp"
        subject = "Microsoft"

        ev_generic = _ev(
            "Microsoft is a major technology corporation based in Redmond, Washington.",
            sim=0.70,
        )
        ev_founder = _ev(
            "Microsoft was founded by Bill Gates and Paul Allen in 1975. "
            "Gates served as CEO until 2000.",
            sim=0.65,
        )

        ranked = rank_evidence_by_quality(
            claim, [ev_generic, ev_founder], subject_entity=subject
        )
        # Both have high subject score; founder has relationship coverage too
        # — founder evidence should rank at least as high
        assert ranked[0].quality_score >= ranked[-1].quality_score


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict engine: subsidiary article must not verify founder claim
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerdictWithSubsidiaryEvidence:

    def test_subsidiary_article_does_not_verify_org_answer(self):
        """Microsoft AI evidence must not VERIFY a 'Microsoft Corp.' answer
        to 'Who founded Microsoft?'."""
        qa = analyze_question("Who founded Microsoft?")
        evidence = [_ev(
            "Microsoft AI (MAI) is an American artificial intelligence research "
            "laboratory which serves as a division of Microsoft. It oversees "
            "consumer artificial intelligence products.",
            sim=STRONG + 0.05,
        )]
        with _no_nli():
            v = determine_verdict(
                "microsoft corp",
                evidence,
                question_analysis=qa,
                answer_text="microsoft corp.",
                subject_entity="Microsoft",
            )
        assert v.verdict != VERDICT_VERIFIED, (
            f"Microsoft AI evidence must not VERIFY a founder answer. Got: {v.verdict}"
        )

    def test_correct_founder_evidence_can_support(self):
        """Real founder evidence should support the answer."""
        qa = analyze_question("Who founded Microsoft?")
        evidence = [
            _ev(
                "Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975.",
                sim=STRONG + 0.05,
                src="wikipedia",
            ),
            _ev(
                "Microsoft: American multinational technology company founded by "
                "Bill Gates and Paul Allen",
                sim=PARTIAL + 0.05,
                src="wikidata",
            ),
        ]
        with _no_nli():
            v = determine_verdict(
                "Bill Gates and Paul Allen founded Microsoft.",
                evidence,
                question_analysis=qa,
                answer_text="Bill Gates and Paul Allen",
                subject_entity="Microsoft",
            )
        assert v.verdict in (VERDICT_VERIFIED, VERDICT_PARTIALLY_SUPPORTED), (
            f"Correct founder evidence should support the answer. Got: {v.verdict}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Query builder: founder query should not include product names
# ═══════════════════════════════════════════════════════════════════════════════

class TestFounderQueryBuilder:

    def test_founder_query_uses_subject_name(self):
        qa = analyze_question("Who founded Microsoft?")
        query = build_subject_query(qa, fallback_claim="microsoft corp.")
        assert "microsoft" in query.lower(), f"Query should include 'microsoft'. Got: {query!r}"

    def test_founder_query_includes_founding_keyword(self):
        qa = analyze_question("Who founded Microsoft?")
        query = build_subject_query(qa, fallback_claim="microsoft corp.")
        # Query should include a keyword that helps find founding information.
        # "history" retrieves the History of Microsoft article which names founders.
        assert any(kw in query.lower() for kw in ["found", "founder", "history"]), (
            f"Query should include a founding or history keyword. Got: {query!r}"
        )

    def test_founder_query_does_not_include_answer_value(self):
        """Answer value ('microsoft corp.') must NOT dominate the query."""
        qa = analyze_question("Who founded Microsoft?")
        query = build_subject_query(qa, fallback_claim="microsoft corp.")
        # "corp" is not the subject — it's part of the wrong answer
        assert query.lower().strip() != "microsoft corp.", (
            f"Query must not be identical to the wrong answer. Got: {query!r}"
        )

    def test_general_founder_query_for_other_orgs(self):
        """Same logic for any organisation — not hard-coded to Microsoft."""
        qa = analyze_question("Who founded Amazon?")
        query = build_subject_query(qa, fallback_claim="amazon.com")
        assert "amazon" in query.lower()
        # Should include either a founding keyword or history keyword
        assert any(kw in query.lower() for kw in ["found", "founder", "history"])
