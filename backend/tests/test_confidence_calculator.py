"""Tests for app.verification.confidence_calculator.

Verifies the heuristic weighted-sum formula and boundary conditions.

Updated weights (v2):
  similarity=0.35, agreement=0.25, consistency=0.20, entity_presence=0.20
"""
from __future__ import annotations

import pytest
from app.verification.confidence_calculator import (
    calculate_confidence,
    TOTAL_SOURCES_QUERIED,
    WEIGHT_SIMILARITY,
    WEIGHT_AGREEMENT,
    WEIGHT_CONSISTENCY,
    WEIGHT_ENTITY_PRESENCE,
)


class TestCalculateConfidence:
    # ---- formula correctness ------------------------------------------------

    def test_perfect_score(self):
        # sim=1.0, all 3 sources agree, no contradiction, no entity mismatch,
        # entity_presence=1.0 (default)
        score = calculate_confidence(
            similarity=1.0,
            agree_count=3,
            contradiction_detected=False,
            entity_mismatch=False,
        )
        # 0.35*1.0 + 0.25*(3/3) + 0.20*1.0 + 0.20*1.0 = 1.0
        expected = WEIGHT_SIMILARITY + WEIGHT_AGREEMENT + WEIGHT_CONSISTENCY + WEIGHT_ENTITY_PRESENCE
        assert score == pytest.approx(expected, abs=1e-4)

    def test_zero_score_when_contradiction_and_no_evidence(self):
        # When there is no evidence at all, entity_presence_score should be 0.0,
        # AND contradiction kills consistency_signal → all components zero.
        score = calculate_confidence(
            similarity=0.0,
            agree_count=0,
            contradiction_detected=True,
            entity_mismatch=False,
            entity_presence_score=0.0,   # no evidence → no entity overlap
        )
        # 0.35*0 + 0.25*(0/3) + 0.20*0 + 0.20*0 = 0.0
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_contradiction_kills_consistency_signal(self):
        # Even with high similarity and agreement, contradiction zeroes consistency
        with_contradiction = calculate_confidence(
            similarity=1.0, agree_count=3,
            contradiction_detected=True, entity_mismatch=False
        )
        without_contradiction = calculate_confidence(
            similarity=1.0, agree_count=3,
            contradiction_detected=False, entity_mismatch=False
        )
        assert with_contradiction < without_contradiction

    def test_entity_mismatch_kills_consistency_signal(self):
        with_mismatch = calculate_confidence(
            similarity=1.0, agree_count=3,
            contradiction_detected=False, entity_mismatch=True
        )
        without_mismatch = calculate_confidence(
            similarity=1.0, agree_count=3,
            contradiction_detected=False, entity_mismatch=False
        )
        assert with_mismatch < without_mismatch

    def test_partial_agreement_new_formula(self):
        # 1 out of 3 sources agree, entity_presence defaults to 1.0
        score = calculate_confidence(
            similarity=0.8,
            agree_count=1,
            contradiction_detected=False,
            entity_mismatch=False,
        )
        # 0.35*0.8 + 0.25*(1/3) + 0.20*1.0 + 0.20*1.0
        # = 0.28 + 0.0833... + 0.20 + 0.20 = 0.7633...
        expected = (
            WEIGHT_SIMILARITY * 0.8
            + WEIGHT_AGREEMENT * (1 / 3)
            + WEIGHT_CONSISTENCY * 1.0
            + WEIGHT_ENTITY_PRESENCE * 1.0
        )
        assert score == pytest.approx(expected, abs=1e-3)

    def test_entity_presence_penalises_wrong_entity(self):
        # entity_presence_score=0.0 means no claim entities found in evidence
        # This should give a lower score than entity_presence_score=1.0
        score_no_entity = calculate_confidence(
            similarity=0.7,
            agree_count=1,
            contradiction_detected=False,
            entity_mismatch=False,
            entity_presence_score=0.0,
        )
        score_full_entity = calculate_confidence(
            similarity=0.7,
            agree_count=1,
            contradiction_detected=False,
            entity_mismatch=False,
            entity_presence_score=1.0,
        )
        assert score_no_entity < score_full_entity

    def test_entity_presence_is_zero_with_wrong_entity_and_mismatch(self):
        # Simulate the Infosys/McCarthy scenario:
        # moderate similarity, entity mismatch, zero entity presence
        score = calculate_confidence(
            similarity=0.6,
            agree_count=1,
            contradiction_detected=False,
            entity_mismatch=True,
            entity_presence_score=0.0,
        )
        # consistency_signal=0 (entity_mismatch), entity_presence=0
        # 0.35*0.6 + 0.25*(1/3) + 0.20*0 + 0.20*0 = 0.21 + 0.0833 = 0.2933
        expected = (
            WEIGHT_SIMILARITY * 0.6
            + WEIGHT_AGREEMENT * (1 / 3)
        )
        assert score == pytest.approx(expected, abs=1e-3)
        # Must be well below 0.5 — not "high confidence"
        assert score < 0.5

    def test_result_rounded_to_four_decimal_places(self):
        score = calculate_confidence(
            similarity=0.333333,
            agree_count=1,
            contradiction_detected=False,
            entity_mismatch=False,
        )
        assert score == round(score, 4)

    # ---- boundary clamping ---------------------------------------------------

    def test_output_never_exceeds_1(self):
        score = calculate_confidence(
            similarity=2.0,  # deliberately out-of-range
            agree_count=5,
            contradiction_detected=False,
            entity_mismatch=False,
        )
        assert score <= 1.0

    def test_output_never_below_0(self):
        score = calculate_confidence(
            similarity=-5.0,  # deliberately out-of-range
            agree_count=0,
            contradiction_detected=True,
            entity_mismatch=True,
            entity_presence_score=0.0,
        )
        assert score >= 0.0

    # ---- edge cases ----------------------------------------------------------

    def test_zero_sources_queried_does_not_crash(self):
        score = calculate_confidence(
            similarity=0.5,
            agree_count=0,
            contradiction_detected=False,
            entity_mismatch=False,
            total_sources_queried=0,
        )
        assert 0.0 <= score <= 1.0

    def test_total_sources_default_is_three(self):
        assert TOTAL_SOURCES_QUERIED == 3

    def test_weights_sum_to_one(self):
        total = (
            WEIGHT_SIMILARITY
            + WEIGHT_AGREEMENT
            + WEIGHT_CONSISTENCY
            + WEIGHT_ENTITY_PRESENCE
        )
        assert total == pytest.approx(1.0, abs=1e-9)
