"""Tests for app.verification.relationship_checker."""
from __future__ import annotations

import pytest
from app.verification.relationship_checker import check_relationship, RelationshipCheckResult


class TestCheckRelationship:

    def test_ceo_relationship_in_evidence(self):
        result = check_relationship(
            "John McCarthy was the CEO of Infosys",
            "Narayana Murthy is the co-founder and former CEO of Infosys.",
        )
        assert result.relationship_discussed is True
        assert result.matched_category == "leadership"

    def test_ceo_relationship_NOT_in_evidence(self):
        result = check_relationship(
            "John McCarthy was the CEO of Infosys",
            "Infosys Limited is an Indian multinational information technology company.",
        )
        assert result.relationship_discussed is False
        assert result.matched_category is None

    def test_born_relationship_in_evidence(self):
        result = check_relationship(
            "Albert Einstein was born in Ulm",
            "Einstein was born on 14 March 1879 in Ulm, Kingdom of Württemberg.",
        )
        assert result.relationship_discussed is True
        assert result.matched_category == "birth"

    def test_born_relationship_NOT_in_evidence(self):
        result = check_relationship(
            "Marie Curie was born in Warsaw",
            "Marie Curie was a Polish physicist and chemist.",
        )
        assert result.relationship_discussed is False

    def test_founded_relationship_in_evidence(self):
        result = check_relationship(
            "Bill Gates founded Microsoft",
            "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
        )
        assert result.relationship_discussed is True
        assert result.matched_category == "founding"

    def test_no_keyword_in_claim_returns_neutral(self):
        result = check_relationship(
            "Water is a chemical compound",
            "H2O consists of two hydrogen atoms and one oxygen atom.",
        )
        # No relationship keyword → relationship_discussed treated as True
        # (we don't penalise claims that have no relationship keyword)
        assert result.relationship_discussed is True
        assert result.matched_category is None

    def test_returns_relationship_check_result_object(self):
        result = check_relationship("test claim", "test evidence")
        assert isinstance(result, RelationshipCheckResult)
        assert isinstance(result.relationship_discussed, bool)
        assert isinstance(result.reason, str)
