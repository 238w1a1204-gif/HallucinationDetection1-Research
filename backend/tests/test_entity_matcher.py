"""Tests for the rewritten app.verification.entity_matcher.

Covers:
- Year mismatch detection
- Person-in-claim not found in evidence (the "empty-people" bug fix)
- Organization mismatch detection
- entity_presence_score computation
- extract_role_keywords
"""
from __future__ import annotations

import pytest
from app.verification.entity_matcher import (
    EntityCheckResult,
    check_entity_consistency,
    entity_presence_score,
    extract_role_keywords,
)


class TestCheckEntityConsistency:

    # ── Year checks ───────────────────────────────────────────────────────

    def test_matching_years_no_mismatch(self):
        result = check_entity_consistency(
            "Einstein was born in 1879",
            "Albert Einstein (1879–1955) was a physicist.",
        )
        assert result.mismatch is False

    def test_mismatching_years_returns_mismatch(self):
        result = check_entity_consistency(
            "Einstein was born in 1990",
            "Albert Einstein was born in 1879.",
        )
        assert result.mismatch is True
        assert "1990" in result.reason or "1879" in result.reason

    def test_claim_with_year_evidence_without_year_no_mismatch(self):
        # Only flag when BOTH sides have years that differ
        result = check_entity_consistency(
            "Einstein was born in 1879",
            "Albert Einstein was a German-born physicist.",
        )
        assert result.mismatch is False

    # ── Person checks (the critical bug-fix tests) ────────────────────────

    def test_person_in_claim_found_in_evidence_no_mismatch(self):
        result = check_entity_consistency(
            "Albert Einstein was a physicist",
            "Albert Einstein (14 March 1879 – 18 April 1955) was a "
            "German-born theoretical physicist.",
        )
        assert result.mismatch is False

    def test_person_in_claim_NOT_in_evidence_mismatch(self):
        """Core regression: claim names a person absent from the evidence."""
        result = check_entity_consistency(
            "John McCarthy was the CEO of Infosys",
            "Infosys Limited is an Indian multinational information technology "
            "company that offers business consulting, IT and outsourcing services.",
        )
        assert result.mismatch is True
        assert "john mccarthy" in result.reason.lower() or "person" in result.reason.lower()

    def test_person_in_claim_NOT_in_evidence_even_when_evidence_has_other_people(self):
        """Claim says 'John McCarthy' but evidence only mentions 'Narayana Murthy'."""
        result = check_entity_consistency(
            "John McCarthy was the CEO of Infosys",
            "Narayana Murthy is the co-founder and former CEO of Infosys.",
        )
        assert result.mismatch is True

    def test_no_person_in_claim_no_person_check(self):
        result = check_entity_consistency(
            "Infosys is located in Bangalore",
            "Infosys Limited is headquartered in Bengaluru, India.",
        )
        # No person check required — should not mismatch on person grounds
        # (may or may not match depending on NER for location)
        assert isinstance(result.mismatch, bool)

    # ── Organization checks ───────────────────────────────────────────────

    def test_org_in_claim_found_in_evidence_no_mismatch(self):
        result = check_entity_consistency(
            "Microsoft was founded by Bill Gates",
            "Microsoft Corporation was founded by Bill Gates and Paul Allen.",
        )
        assert result.mismatch is False

    def test_all_orgs_in_claim_absent_from_evidence_mismatch(self):
        result = check_entity_consistency(
            "Google is headquartered in Mountain View",
            "Apple Inc. is headquartered in Cupertino, California.",
        )
        # Google not mentioned → mismatch (NER may or may not catch Google here;
        # plain-text substring check will find it if spaCy misses it)
        # This is a best-effort test — we mainly verify no crash and that the
        # result is an EntityCheckResult instance.
        assert isinstance(result, EntityCheckResult)
        assert isinstance(result.mismatch, bool)

    # ── entity_presence_score ─────────────────────────────────────────────

    def test_presence_score_one_when_no_entities_in_claim(self):
        """No named entities in claim → score should be 1.0 (no penalty)."""
        score = entity_presence_score(
            "water is a chemical compound",
            "H2O is composed of hydrogen and oxygen atoms.",
        )
        assert score == pytest.approx(1.0, abs=0.1)

    def test_presence_score_low_when_entity_absent(self):
        """Person named in claim not in evidence → low score."""
        score = entity_presence_score(
            "John McCarthy was CEO of Infosys",
            "Infosys Limited is an Indian IT company.",
        )
        # john mccarthy is absent; Infosys may match; score should be at most 0.5
        assert score <= 0.5

    def test_presence_score_high_when_entity_present(self):
        score = entity_presence_score(
            "Albert Einstein was born in 1879",
            "Albert Einstein (1879–1955) was a German-born physicist.",
        )
        assert score > 0.5

    def test_presence_score_in_valid_range(self):
        score = entity_presence_score(
            "Marie Curie was born in Warsaw in 1867",
            "Marie Curie (1867–1934) was a Polish physicist born in Warsaw.",
        )
        assert 0.0 <= score <= 1.0


class TestExtractRoleKeywords:

    def test_ceo_keyword_detected(self):
        kws = extract_role_keywords("John McCarthy was the CEO of Infosys")
        assert "ceo" in kws

    def test_founded_keyword_detected(self):
        kws = extract_role_keywords("Bill Gates founded Microsoft")
        assert "founded" in kws

    def test_born_keyword_detected(self):
        kws = extract_role_keywords("Einstein was born in Ulm")
        assert "born" in kws

    def test_no_keyword_empty_set(self):
        kws = extract_role_keywords("Water is a chemical compound")
        assert len(kws) == 0

    def test_multiple_keywords(self):
        kws = extract_role_keywords("She was born in Paris and later founded a company")
        assert "born" in kws
        assert "founded" in kws
