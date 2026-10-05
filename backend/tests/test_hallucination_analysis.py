"""Tests for app.verification.hallucination_analysis."""
from __future__ import annotations

import pytest
from app.verification.hallucination_analysis import (
    analyze,
    HallucinationAnalysis,
    OVERALL_HALLUCINATED,
    OVERALL_CLEAN,
    OVERALL_PARTIAL,
    OVERALL_NO_CLAIMS,
)
from app.verification.verdict_engine import (
    VERDICT_VERIFIED,
    VERDICT_PARTIALLY_SUPPORTED,
    VERDICT_UNSUPPORTED,
    VERDICT_CONTRADICTED,
)


class TestAnalyze:
    def test_empty_claims_returns_no_claims_verdict(self):
        result = analyze([], [])
        assert result.overall_verdict == OVERALL_NO_CLAIMS
        assert result.overall_confidence == 0.0
        assert result.num_claims == 0

    def test_all_verified_returns_no_hallucination(self):
        verdicts = [VERDICT_VERIFIED, VERDICT_VERIFIED, VERDICT_VERIFIED]
        confidences = [0.9, 0.85, 0.95]
        result = analyze(verdicts, confidences)
        assert result.overall_verdict == OVERALL_CLEAN
        assert result.num_verified == 3
        assert result.num_contradicted == 0

    def test_any_contradicted_returns_contains_hallucination(self):
        verdicts = [VERDICT_VERIFIED, VERDICT_CONTRADICTED, VERDICT_PARTIALLY_SUPPORTED]
        confidences = [0.9, 0.1, 0.6]
        result = analyze(verdicts, confidences)
        assert result.overall_verdict == OVERALL_HALLUCINATED
        assert result.num_contradicted == 1

    def test_mix_without_contradiction_returns_partial(self):
        verdicts = [VERDICT_VERIFIED, VERDICT_PARTIALLY_SUPPORTED, VERDICT_UNSUPPORTED]
        confidences = [0.9, 0.5, 0.2]
        result = analyze(verdicts, confidences)
        assert result.overall_verdict == OVERALL_PARTIAL

    def test_all_unsupported_returns_partial(self):
        verdicts = [VERDICT_UNSUPPORTED, VERDICT_UNSUPPORTED]
        confidences = [0.1, 0.1]
        result = analyze(verdicts, confidences)
        # No contradiction, not all verified → PARTIAL
        assert result.overall_verdict == OVERALL_PARTIAL

    def test_overall_confidence_is_mean_of_claim_confidences(self):
        verdicts = [VERDICT_VERIFIED, VERDICT_VERIFIED]
        confidences = [0.8, 0.6]
        result = analyze(verdicts, confidences)
        assert result.overall_confidence == pytest.approx(0.7, abs=1e-4)

    def test_counts_are_correct(self):
        verdicts = [
            VERDICT_VERIFIED,
            VERDICT_VERIFIED,
            VERDICT_PARTIALLY_SUPPORTED,
            VERDICT_UNSUPPORTED,
            VERDICT_CONTRADICTED,
        ]
        confidences = [0.9, 0.85, 0.6, 0.2, 0.1]
        result = analyze(verdicts, confidences)
        assert result.num_claims == 5
        assert result.num_verified == 2
        assert result.num_partially_supported == 1
        assert result.num_unsupported == 1
        assert result.num_contradicted == 1

    def test_returns_hallucination_analysis_object(self):
        result = analyze([VERDICT_VERIFIED], [0.9])
        assert isinstance(result, HallucinationAnalysis)

    def test_single_contradicted_claim(self):
        result = analyze([VERDICT_CONTRADICTED], [0.05])
        assert result.overall_verdict == OVERALL_HALLUCINATED
        assert result.overall_confidence == pytest.approx(0.05, abs=1e-4)

    def test_confidence_rounded_to_four_places(self):
        result = analyze([VERDICT_VERIFIED] * 3, [1 / 3, 1 / 3, 1 / 3])
        assert result.overall_confidence == round(result.overall_confidence, 4)
