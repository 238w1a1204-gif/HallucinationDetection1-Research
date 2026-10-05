"""Regression tests for entity resolution and the wrong-entity
contradiction guard.

Problems fixed:
  1. "albert einstien" (typo) → entity resolver should find "Albert Einstein"
     not just "albert".
  2. Evidence about Prince Albert must NOT produce CONTRADICTED for a
     question about Albert Einstein — it is the wrong entity.
  3. Corrected answer must not paste "Albert may refer to:" as a correction.
  4. Direct contradiction (same subject, conflicting descriptor) must still
     produce CONTRADICTED.
  5. Correct biographical answer must still produce VERIFIED/SUPPORTED.

All tests are entity-agnostic: the same mechanism must work for arbitrary
entities (Shakespeare, Tesla, Newton, …). Einstein is only used in the
concrete examples that mirror the actual production failure.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from app.correction.answer_corrector import (
    ClaimCorrectionInput,
    _evidence_is_usable,
    generate_corrected_answer,
)
from app.verification.answer_value_checker import (
    check_answer_value,
    VALUE_SUPPORTED, VALUE_CONTRADICTED, VALUE_UNVERIFIABLE,
)
from app.verification.question_analyzer import (
    analyze_question,
    _char_similarity,
    _token_overlap_sufficient,
    build_claim_from_question,
)
from app.verification.similarity_scorer import ScoredEvidence
from app.verification.verdict_engine import (
    determine_verdict,
    VERDICT_VERIFIED, VERDICT_PARTIALLY_SUPPORTED,
    VERDICT_UNSUPPORTED, VERDICT_CONTRADICTED,
)
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
    return ScoredEvidence(evidence_text=text, source_name=src, source_url=None, similarity=sim)


# ═══════════════════════════════════════════════════════════════════════════════
# Fuzzy similarity helpers (used by entity resolver)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFuzzyHelpers:

    def test_char_similarity_identical(self):
        assert _char_similarity("einstein", "einstein") == pytest.approx(1.0)

    def test_char_similarity_close_typo(self):
        score = _char_similarity("einstien", "einstein")
        assert score > 0.8, f"Expected > 0.8 for close typo, got {score}"

    def test_char_similarity_unrelated(self):
        score = _char_similarity("apple", "einstein")
        assert score < 0.4

    def test_token_overlap_typo(self):
        # "albert einstien" vs "Albert Einstein" — token-level
        assert _token_overlap_sufficient("albert einstien", "Albert Einstein")

    def test_token_overlap_different(self):
        assert not _token_overlap_sufficient("william shakespeare", "Isaac Newton")

    def test_token_overlap_generic_spelling_errors(self):
        # General misspellings — not entity-specific
        assert _token_overlap_sufficient("shakespear", "Shakespeare")
        assert _token_overlap_sufficient("teslla", "Tesla")
        assert _token_overlap_sufficient("nikola teslla", "Nikola Tesla")


# ═══════════════════════════════════════════════════════════════════════════════
# Entity resolution via question analyzer
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntityResolution:

    def test_correct_spelling_extracts_correctly(self):
        qa = analyze_question("Who was Albert Einstein?")
        assert "einstein" in qa.focus_entity.lower()

    def test_typo_does_not_produce_single_word_albert(self):
        """'albert einstien' must not extract just 'albert' as focus entity.
        It should either resolve to 'Albert Einstein' or return the full
        candidate 'albert einstien' — never just the first word.
        """
        qa = analyze_question("who was albert einstien")
        focus = qa.focus_entity.lower()
        # Must contain more than just "albert"
        assert focus != "albert", (
            f"Focus entity should not be just 'albert'. Got: {qa.focus_entity!r}"
        )
        # Should contain either 'einstien', 'einstein', or both words
        assert "einstein" in focus or "einstien" in focus or len(focus.split()) >= 2, (
            f"Focus entity should contain the full name. Got: {qa.focus_entity!r}"
        )

    def test_full_name_extracted_for_multiword(self):
        """Generic multi-word names must be extracted fully."""
        qa = analyze_question("Who was Marie Curie?")
        assert "curie" in qa.focus_entity.lower()
        assert "marie" in qa.focus_entity.lower() or len(qa.focus_entity) > 5

    def test_organization_extracted(self):
        qa = analyze_question("Who founded Microsoft?")
        assert "microsoft" in qa.focus_entity.lower()

    def test_landmark_extracted(self):
        qa = analyze_question("Where is the Eiffel Tower located?")
        assert "eiffel" in qa.focus_entity.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Wrong-entity evidence must NOT produce CONTRADICTED
# ═══════════════════════════════════════════════════════════════════════════════

class TestWrongEntityNotContradicted:

    def test_prince_albert_evidence_not_contradicted_for_einstein_question(self):
        """The core regression: Prince Albert evidence for an Einstein question
        must return VALUE_UNVERIFIABLE, not VALUE_CONTRADICTED.
        """
        qa = analyze_question("Who was Albert Einstein?")
        result = check_answer_value(
            qa,
            "a German politician",
            "Albert: Prince Albert of Saxe-Coburg and Gotha was a German-born "
            "prince and the husband of Queen Victoria of the United Kingdom.",
            subject_entity="Albert Einstein",
        )
        assert result.status != VALUE_CONTRADICTED, (
            f"Wrong-entity evidence must not produce VALUE_CONTRADICTED. "
            f"Got: {result.status}. Reason: {result.reason}"
        )
        assert result.status == VALUE_UNVERIFIABLE

    def test_disambiguation_page_not_contradicted(self):
        """A Wikipedia disambiguation page must not contradict anything."""
        qa = analyze_question("Who was Albert Einstein?")
        result = check_answer_value(
            qa,
            "a German politician",
            "Albert may refer to: Albert (name), Albert of Habsburg, "
            "Albert I of Germany, Prince Albert...",
            subject_entity="Albert Einstein",
        )
        assert result.status == VALUE_UNVERIFIABLE

    def test_same_subject_wrong_descriptor_is_contradicted(self):
        """When evidence IS about the same subject AND contradicts the answer,
        CONTRADICTED is correct."""
        qa = analyze_question("Who was Albert Einstein?")
        result = check_answer_value(
            qa,
            "a German politician",
            "Albert Einstein was a German-born theoretical physicist widely "
            "regarded as one of the greatest scientists of all time.",
            subject_entity="Albert Einstein",
        )
        assert result.status == VALUE_CONTRADICTED, (
            f"Same-subject contradictory evidence must be CONTRADICTED. "
            f"Got: {result.status}. Reason: {result.reason}"
        )

    def test_correct_biographical_answer_supported(self):
        """'physicist' for Einstein with Einstein evidence → SUPPORTED."""
        qa = analyze_question("Who was Albert Einstein?")
        result = check_answer_value(
            qa,
            "physicist",
            "Albert Einstein was a German-born theoretical physicist.",
            subject_entity="Albert Einstein",
        )
        assert result.status == VALUE_SUPPORTED

    def test_wrong_entity_unverifiable_for_any_person(self):
        """Pattern is general: using unrelated person's article is UNVERIFIABLE."""
        qa = analyze_question("Who was Isaac Newton?")
        result = check_answer_value(
            qa,
            "an astronomer",
            "Charles Darwin was an English naturalist who proposed the theory "
            "of evolution by natural selection.",
            subject_entity="Isaac Newton",
        )
        # Darwin evidence doesn't contain "isaac newton" → UNVERIFIABLE
        assert result.status == VALUE_UNVERIFIABLE


# ═══════════════════════════════════════════════════════════════════════════════
# Verdict engine: wrong-entity evidence
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerdictEngineWrongEntity:

    def test_prince_albert_evidence_not_verified_for_einstein(self):
        """With Prince Albert evidence and subject=Albert Einstein,
        the verdict must NOT be VERIFIED."""
        qa = analyze_question("Who was Albert Einstein?")
        evidence = [_ev(
            "Albert: Prince Albert of Saxe-Coburg and Gotha was a German-born "
            "prince and the husband of Queen Victoria.",
            sim=STRONG + 0.05,
        )]
        with _no_nli():
            v = determine_verdict(
                "a German politician",
                evidence,
                question_analysis=qa,
                answer_text="a German politician",
                subject_entity="Albert Einstein",
            )
        assert v.verdict != VERDICT_VERIFIED, (
            f"Wrong-entity evidence must not produce VERIFIED. Got: {v.verdict}"
        )

    def test_disambiguation_evidence_not_verified(self):
        """Disambiguation page must not verify anything."""
        qa = analyze_question("Who was Albert Einstein?")
        evidence = [_ev(
            "Albert may refer to: Albert (name), Albert of Habsburg...",
            sim=0.72,
        )]
        with _no_nli():
            v = determine_verdict(
                "a German politician",
                evidence,
                question_analysis=qa,
                answer_text="a German politician",
                subject_entity="Albert Einstein",
            )
        assert v.verdict != VERDICT_VERIFIED

    def test_real_einstein_evidence_contradicts_politician(self):
        """With real Einstein evidence, 'politician' answer is CONTRADICTED."""
        qa = analyze_question("Who was Albert Einstein?")
        evidence = [
            _ev(
                "Albert Einstein was a German-born theoretical physicist widely "
                "regarded as one of the greatest scientists of all time.",
                sim=STRONG + 0.05,
                src="wikipedia",
            ),
            _ev(
                "Albert Einstein: German-born theoretical physicist (1879–1955)",
                sim=PARTIAL + 0.05,
                src="wikidata",
            ),
        ]
        with _no_nli():
            v = determine_verdict(
                "Albert Einstein was a German politician.",
                evidence,
                question_analysis=qa,
                answer_text="a German politician",
                subject_entity="Albert Einstein",
            )
        assert v.verdict in (VERDICT_CONTRADICTED, VERDICT_UNSUPPORTED), (
            f"Einstein physicist evidence should not verify 'politician'. "
            f"Got: {v.verdict}"
        )

    def test_real_einstein_evidence_supports_physicist(self):
        """'physicist' for Einstein with good evidence → VERIFIED or PARTIALLY_SUPPORTED."""
        qa = analyze_question("Who was Albert Einstein?")
        evidence = [
            _ev(
                "Albert Einstein was a German-born theoretical physicist.",
                sim=STRONG + 0.05,
                src="wikipedia",
            ),
            _ev(
                "Albert Einstein: German-born theoretical physicist (1879-1955)",
                sim=PARTIAL + 0.05,
                src="wikidata",
            ),
        ]
        with _no_nli():
            v = determine_verdict(
                "Albert Einstein was a physicist.",
                evidence,
                question_analysis=qa,
                answer_text="physicist",
                subject_entity="Albert Einstein",
            )
        assert v.verdict in (VERDICT_VERIFIED, VERDICT_PARTIALLY_SUPPORTED), (
            f"Good Einstein evidence should support 'physicist'. Got: {v.verdict}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Corrected answer: must not paste disambiguation/wrong-entity evidence
# ═══════════════════════════════════════════════════════════════════════════════

class TestCorrectedAnswerWrongEntity:

    def test_disambiguation_evidence_not_used_as_correction(self):
        """'Albert may refer to:' must NOT appear as the corrected answer."""
        inputs = [ClaimCorrectionInput(
            claim_text="a German politician",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text="Albert may refer to: Albert (name), Albert of Habsburg...",
            entity_mismatch=False,
            subject_entity="Albert Einstein",
        )]
        out = generate_corrected_answer(inputs)
        assert "albert may refer to" not in out.lower(), (
            "Disambiguation page must NOT be used as a corrected answer."
        )
        assert "could not be verified" in out.lower() or "unrelated entity" in out.lower()

    def test_wrong_person_evidence_not_used_as_correction(self):
        """Prince Albert evidence must NOT be pasted as Einstein's corrected answer."""
        inputs = [ClaimCorrectionInput(
            claim_text="a German politician",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Albert: Prince Albert of Saxe-Coburg and Gotha was the husband "
                "of Queen Victoria."
            ),
            entity_mismatch=False,
            subject_entity="Albert Einstein",
        )]
        out = generate_corrected_answer(inputs)
        assert "prince albert" not in out.lower() or "could not be verified" in out.lower(), (
            "Wrong-entity evidence must not be pasted as a correction."
        )

    def test_real_evidence_used_as_correction(self):
        """When evidence IS about the correct entity, use it."""
        inputs = [ClaimCorrectionInput(
            claim_text="a German politician",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Albert Einstein was a German-born theoretical physicist."
            ),
            entity_mismatch=False,
            subject_entity="Albert Einstein",
        )]
        out = generate_corrected_answer(inputs)
        assert "physicist" in out.lower()
        assert "einstein" in out.lower()

    def test_evidence_usability_check(self):
        assert _evidence_is_usable(
            "Albert Einstein was a physicist.", "Albert Einstein"
        ) is True
        assert _evidence_is_usable(
            "Albert may refer to: Albert (name)...", "Albert Einstein"
        ) is False
        assert _evidence_is_usable(
            "Prince Albert was the husband of Queen Victoria.", "Albert Einstein"
        ) is False
        assert _evidence_is_usable(None, "Albert Einstein") is False
        assert _evidence_is_usable("", "Albert Einstein") is False
        # No subject constraint → any substantive text is usable
        assert _evidence_is_usable("Some evidence about something.", "") is True


# ═══════════════════════════════════════════════════════════════════════════════
# General entity patterns — not Einstein-specific
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeneralEntityPatterns:

    def test_wrong_entity_unverifiable_shakespeare(self):
        """Marlowe evidence for Shakespeare question → UNVERIFIABLE."""
        qa = analyze_question("Who was William Shakespeare?")
        result = check_answer_value(
            qa,
            "a playwright",
            "Christopher Marlowe was an English playwright, poet, and translator.",
            subject_entity="William Shakespeare",
        )
        # Marlowe evidence has no "shakespeare" → cannot confirm or deny
        assert result.status == VALUE_UNVERIFIABLE

    def test_correct_entity_supported_newton(self):
        """Newton physicist answer with Newton evidence → SUPPORTED."""
        qa = analyze_question("Who was Isaac Newton?")
        result = check_answer_value(
            qa,
            "a physicist",
            "Isaac Newton was an English mathematician, physicist, and astronomer.",
            subject_entity="Isaac Newton",
        )
        assert result.status == VALUE_SUPPORTED

    def test_wrong_descriptor_contradicted_same_subject(self):
        """Wrong job title with correct subject → CONTRADICTED."""
        qa = analyze_question("Who was Marie Curie?")
        result = check_answer_value(
            qa,
            "a German politician",
            "Marie Curie was a Polish-French physicist and chemist who conducted "
            "pioneering research on radioactivity.",
            subject_entity="Marie Curie",
        )
        assert result.status == VALUE_CONTRADICTED

    def test_eiffel_tower_still_works(self):
        """Make sure existing location verification is not broken."""
        qa = analyze_question("Where is the Eiffel Tower located?")
        result = check_answer_value(
            qa,
            "Paris",
            "The Eiffel Tower is located on the Champ de Mars in Paris, France.",
        )
        assert result.status == VALUE_SUPPORTED

    def test_wrong_location_still_contradicted(self):
        qa = analyze_question("Where is the Eiffel Tower located?")
        result = check_answer_value(
            qa,
            "London",
            "The Eiffel Tower is located in Paris, France.",
        )
        assert result.status == VALUE_CONTRADICTED
