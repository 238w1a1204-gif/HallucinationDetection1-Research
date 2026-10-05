"""Tests for app.verification.similarity_scorer.

We mock the embedding layer so that tests run without downloading the
sentence-transformers model (CI-friendly). We verify the scoring and
sorting logic.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from app.verification.similarity_scorer import (
    ScoredEvidence,
    score_evidence_against_claim,
)


def _unit_vec(n: int, dim: int = 4) -> np.ndarray:
    """Return a simple unit vector with value 1 at position n (dim=4 space)."""
    v = np.zeros(dim, dtype=np.float32)
    v[n % dim] = 1.0
    return v


def _make_embed(vecs: list[np.ndarray]):
    """Return a mock `embed` function that returns vecs in order."""
    call_count = [0]

    def embed(texts):
        result = np.array(vecs[call_count[0] : call_count[0] + len(texts)])
        call_count[0] += len(texts)
        return result

    return embed


class TestScoreEvidenceAgainstClaim:
    def test_empty_evidence_returns_empty_list(self):
        result = score_evidence_against_claim("some claim", [])
        assert result == []

    def test_returns_scored_evidence_objects(self):
        # identical vectors → cosine similarity = 1.0
        vec = _unit_vec(0)
        mock_embed = _make_embed([vec, vec])  # claim vec, then evidence vec
        with patch("app.verification.similarity_scorer.embed", side_effect=mock_embed):
            results = score_evidence_against_claim(
                "claim", [("evidence text", "wikipedia", "http://example.com")]
            )
        assert len(results) == 1
        assert isinstance(results[0], ScoredEvidence)
        assert results[0].similarity == pytest.approx(1.0, abs=1e-5)
        assert results[0].source_name == "wikipedia"
        assert results[0].source_url == "http://example.com"
        assert results[0].evidence_text == "evidence text"

    def test_orthogonal_vectors_give_zero_similarity(self):
        claim_vec = _unit_vec(0)
        ev_vec = _unit_vec(1)
        mock_embed = _make_embed([claim_vec, ev_vec])
        with patch("app.verification.similarity_scorer.embed", side_effect=mock_embed):
            results = score_evidence_against_claim(
                "claim", [("text", "wikidata", None)]
            )
        assert results[0].similarity == pytest.approx(0.0, abs=1e-5)

    def test_results_sorted_descending(self):
        # Three evidence items with known similarities: 1.0, 0.5, 0.0
        # achieved by choosing vectors deliberately
        claim_vec = _unit_vec(0)  # [1, 0, 0, 0]
        ev1 = _unit_vec(0)        # sim = 1.0 (identical)
        ev2 = np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float32)  # sim ≈ 0.707
        ev3 = _unit_vec(1)        # sim = 0.0 (orthogonal)

        def mock_embed(texts):
            if len(texts) == 1:
                return np.array([claim_vec])
            return np.array([ev1, ev2, ev3])

        with patch("app.verification.similarity_scorer.embed", side_effect=mock_embed):
            results = score_evidence_against_claim(
                "claim",
                [("a", "wikipedia", None), ("b", "wikidata", None), ("c", "duckduckgo", None)],
            )

        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True), "Results must be sorted descending"

    def test_similarity_clamped_to_nonnegative(self):
        # Negative cosine similarity (anti-parallel) should be clamped to 0
        claim_vec = _unit_vec(0)
        ev_vec = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def mock_embed(texts):
            if len(texts) == 1:
                return np.array([claim_vec])
            return np.array([ev_vec])

        with patch("app.verification.similarity_scorer.embed", side_effect=mock_embed):
            results = score_evidence_against_claim("claim", [("text", "src", None)])
        assert results[0].similarity >= 0.0

    def test_source_name_and_url_preserved(self):
        vec = _unit_vec(0)
        mock_embed = _make_embed([vec, vec])
        with patch("app.verification.similarity_scorer.embed", side_effect=mock_embed):
            results = score_evidence_against_claim(
                "claim",
                [("text", "duckduckgo", "https://ddg.com")],
            )
        assert results[0].source_name == "duckduckgo"
        assert results[0].source_url == "https://ddg.com"
