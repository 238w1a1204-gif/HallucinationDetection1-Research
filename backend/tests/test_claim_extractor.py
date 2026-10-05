"""Tests for app.nlp.claim_extractor — actual behaviour, not mere imports."""
from __future__ import annotations

import pytest
from app.nlp.claim_extractor import extract_claims, MIN_CLAIM_LENGTH_CHARS


class TestExtractClaims:
    def test_empty_string_returns_empty_list(self):
        assert extract_claims("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert extract_claims("   \n\t  ") == []

    def test_single_sentence_produces_one_claim(self):
        claims = extract_claims("Albert Einstein was a German-born theoretical physicist.")
        assert len(claims) == 1
        assert "Einstein" in claims[0]

    def test_two_sentences_produce_two_claims(self):
        text = "Paris is the capital of France. It is also the most populous city."
        claims = extract_claims(text)
        assert len(claims) >= 2

    def test_conjunction_splitting(self):
        # "; " is a splitter — both halves should appear as separate claims
        text = "Paris is the capital of France; Rome is the capital of Italy."
        claims = extract_claims(text)
        assert len(claims) >= 2

    def test_question_is_excluded(self):
        # A sentence ending in "?" should be filtered out
        text = "What is the capital of France? Paris is the capital of France."
        claims = extract_claims(text)
        for c in claims:
            assert not c.strip().endswith("?")

    def test_too_short_fragments_excluded(self):
        # Any fragment shorter than MIN_CLAIM_LENGTH_CHARS should be dropped
        text = "Hi. Albert Einstein was born in Ulm in 1879."
        claims = extract_claims(text)
        for c in claims:
            assert len(c) >= MIN_CLAIM_LENGTH_CHARS

    def test_duplicates_removed(self):
        # Repeat the same sentence — de-duplication should give exactly one copy
        sentence = "Water is a chemical compound made of hydrogen and oxygen."
        text = f"{sentence} {sentence}"
        claims = extract_claims(text)
        lower_claims = [c.lower() for c in claims]
        # The same lowercase key must not appear twice
        assert len(lower_claims) == len(set(lower_claims))

    def test_returns_list_of_strings(self):
        claims = extract_claims("The sun is a star. Earth orbits the sun.")
        assert isinstance(claims, list)
        assert all(isinstance(c, str) for c in claims)

    def test_multisentence_paragraph(self):
        text = (
            "Marie Curie was a Polish physicist and chemist. "
            "She was born in Warsaw in 1867. "
            "She won the Nobel Prize in Physics in 1903."
        )
        claims = extract_claims(text)
        assert len(claims) >= 2
