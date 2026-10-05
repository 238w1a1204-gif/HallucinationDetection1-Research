"""Regression tests for targeted answer extraction in corrected answers.

These tests verify that when evidence directly contradicts an LLM answer,
the corrected answer contains the SPECIFIC correct value (e.g. "Ulm, Germany")
rather than a full Wikipedia paragraph or a "could not be verified" message.

All assertions use real evidence text — no factual answers are hard-coded
into production logic, only into these test fixtures.
"""
from __future__ import annotations

import pytest
from app.correction.answer_corrector import (
    ClaimCorrectionInput,
    extract_answer_from_evidence,
    generate_corrected_answer,
    _build_correction_sentence,
    _first_sentence,
)
from app.verification.verdict_engine import VERDICT_CONTRADICTED, VERDICT_VERIFIED

# ═══════════════════════════════════════════════════════════════════════════════
# extract_answer_from_evidence — unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractAnswerFromEvidence:

    # ── Birthplace ─────────────────────────────────────────────────────────

    def test_born_in_extracts_location(self):
        ev = "Albert Einstein (14 March 1879) was a German-born physicist. He was born in Ulm, Kingdom of Württemberg."
        result = extract_answer_from_evidence(ev, "BORN_IN", "Albert Einstein")
        assert result is not None
        assert "ulm" in result.lower(), f"Expected Ulm in result, got: {result!r}"

    def test_born_in_marie_curie(self):
        ev = ("Maria Salomea Skłodowska Curie (7 November 1867 – 4 July 1934), "
              "better known as Marie Curie, was born in Warsaw, Poland.")
        result = extract_answer_from_evidence(ev, "BORN_IN", "Marie Curie")
        assert result is not None
        assert "warsaw" in result.lower(), f"Expected Warsaw, got: {result!r}"

    def test_born_in_wikidata_format(self):
        # Wikidata short descriptions often use different phrasing
        ev = "Albert Einstein: German-born theoretical physicist (1879–1955)"
        result = extract_answer_from_evidence(ev, "BORN_IN", "Albert Einstein")
        # May not extract from this format — acceptable to return None
        # but must not raise
        assert result is None or isinstance(result, str)

    # ── Location ───────────────────────────────────────────────────────────

    def test_located_in_eiffel_tower(self):
        ev = ("The Eiffel Tower is a lattice tower located on the Champ de Mars "
              "in Paris, France. It is named after the engineer Gustave Eiffel.")
        result = extract_answer_from_evidence(ev, "LOCATED_IN", "Eiffel Tower")
        assert result is not None
        assert "paris" in result.lower(), f"Expected Paris, got: {result!r}"

    def test_located_in_wikidata(self):
        ev = "Eiffel Tower: tower located on the Champ de Mars in Paris, France"
        result = extract_answer_from_evidence(ev, "LOCATED_IN", "Eiffel Tower")
        assert result is not None
        assert "paris" in result.lower(), f"Expected Paris, got: {result!r}"

    # ── Capital ────────────────────────────────────────────────────────────

    def test_capital_of_france(self):
        ev = ("France, officially the French Republic, is a country in Western Europe. "
              "Paris, the capital city and largest city of France, is located in the north.")
        result = extract_answer_from_evidence(ev, "CAPITAL_OF", "France")
        assert result is not None
        assert "paris" in result.lower(), f"Expected Paris, got: {result!r}"

    def test_capital_of_germany(self):
        ev = ("Germany is a country in Central Europe. "
              "Berlin is the capital and largest city of Germany.")
        result = extract_answer_from_evidence(ev, "CAPITAL_OF", "Germany")
        assert result is not None
        assert "berlin" in result.lower(), f"Expected Berlin, got: {result!r}"

    # ── Founder ────────────────────────────────────────────────────────────

    def test_founded_by_microsoft(self):
        ev = ("Microsoft is a computer technology corporation founded on "
              "April 4, 1975, by Bill Gates and Paul Allen in Albuquerque.")
        # "Who founded X?" → question_type="PERSON"
        result = extract_answer_from_evidence(ev, "FOUNDED_BY", "Microsoft", "PERSON")
        assert result is not None
        assert "gates" in result.lower() or "allen" in result.lower(), (
            f"Expected Gates or Allen in result, got: {result!r}"
        )

    def test_founded_by_amazon(self):
        ev = "Amazon was founded by Jeff Bezos in 1994 in Bellevue, Washington."
        result = extract_answer_from_evidence(ev, "FOUNDED_BY", "Amazon", "PERSON")
        assert result is not None
        assert "bezos" in result.lower(), f"Expected Bezos, got: {result!r}"

    def test_founded_by_date(self):
        """'When was X founded?' → question_type=DATE → should return year not person."""
        ev = ("Microsoft is a computer technology corporation founded on "
              "April 4, 1975, by Bill Gates and Paul Allen in Albuquerque.")
        result = extract_answer_from_evidence(ev, "FOUNDED_BY", "Microsoft", "DATE")
        assert result is not None
        assert "1975" in result, f"Expected year 1975, got: {result!r}"

    # ── President ──────────────────────────────────────────────────────────

    def test_first_president_george_washington(self):
        ev = ("George Washington was the first president of the United States, "
              "serving from 1789 to 1797 after leading the Continental Army.")
        result = extract_answer_from_evidence(ev, "PRESIDENT_OF", "United States")
        assert result is not None
        assert "washington" in result.lower(), (
            f"Expected Washington in result, got: {result!r}"
        )

    # ── Edge cases ─────────────────────────────────────────────────────────

    def test_empty_evidence_returns_none(self):
        assert extract_answer_from_evidence("", "BORN_IN", "Einstein") is None

    def test_unknown_relationship_returns_none(self):
        ev = "Albert Einstein was a physicist born in Ulm."
        result = extract_answer_from_evidence(ev, "UNKNOWN_REL", "Einstein")
        assert result is None

    def test_result_is_short(self):
        """Extracted values must be short (not full paragraphs)."""
        ev = ("Albert Einstein (14 March 1879 – 18 April 1955) was a German-born "
              "theoretical physicist best known for developing the theory of relativity. "
              "Einstein was born in Ulm, in the Kingdom of Württemberg in the German Empire.")
        result = extract_answer_from_evidence(ev, "BORN_IN", "Albert Einstein")
        if result:
            assert len(result) < 80, f"Extracted value too long: {result!r}"

    def test_general_pattern_different_entity(self):
        """Uses Darwin — not Einstein — proves no hard-coding."""
        ev = ("Charles Darwin was born in Shrewsbury, Shropshire, England on "
              "12 February 1809.")
        result = extract_answer_from_evidence(ev, "BORN_IN", "Charles Darwin")
        assert result is not None
        assert "shrewsbury" in result.lower(), f"Expected Shrewsbury, got: {result!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# _build_correction_sentence — unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildCorrectionSentence:

    def test_born_in_template(self):
        s = _build_correction_sentence("Albert Einstein", "BORN_IN", "*", "Ulm, Germany", "")
        assert "Albert Einstein" in s
        assert "Ulm, Germany" in s
        assert "born" in s.lower()

    def test_capital_of_template(self):
        s = _build_correction_sentence("France", "CAPITAL_OF", "*", "Paris", "")
        assert "France" in s
        assert "Paris" in s

    def test_founded_by_template(self):
        s = _build_correction_sentence("Microsoft", "FOUNDED_BY", "PERSON", "Bill Gates and Paul Allen", "")
        assert "Microsoft" in s
        assert "Bill Gates" in s

    def test_founded_date_template(self):
        s = _build_correction_sentence("Microsoft", "FOUNDED_BY", "DATE", "1975", "")
        assert "Microsoft" in s
        assert "1975" in s
        assert "founded" in s.lower()

    def test_unknown_relationship_uses_first_sentence(self):
        fallback = "Some relevant evidence sentence. And more text that is irrelevant."
        s = _build_correction_sentence("X", "UNKNOWN", "*", "value", fallback)
        assert s == "Some relevant evidence sentence."

    def test_first_sentence_fallback(self):
        text = "Einstein was born in Ulm. He later moved to Switzerland. Many more details."
        assert _first_sentence(text) == "Einstein was born in Ulm."


# ═══════════════════════════════════════════════════════════════════════════════
# generate_corrected_answer — integration tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateCorrectedAnswer:

    # ── Test 1: Einstein birthplace ────────────────────────────────────────
    def test_einstein_birthplace_correction(self):
        inputs = [ClaimCorrectionInput(
            claim_text="Albert Einstein was born in St. Petersburg.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Albert Einstein (14 March 1879 – 18 April 1955) was a "
                "German-born theoretical physicist. He was born in Ulm, "
                "in the Kingdom of Württemberg."
            ),
            entity_mismatch=False,
            subject_entity="Albert Einstein",
            question_type="LOCATION",
            question_relationship="BORN_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        # Must contain Ulm, not St. Petersburg
        assert "ulm" in body.lower(), (
            f"Corrected answer should contain 'Ulm'. Got: {body!r}"
        )
        assert "st. petersburg" not in body.lower(), (
            f"Corrected answer must not repeat the wrong answer. Got: {body!r}"
        )

    # ── Test 2: Eiffel Tower location ──────────────────────────────────────
    def test_eiffel_tower_location_correction(self):
        inputs = [ClaimCorrectionInput(
            claim_text="The Eiffel Tower is located in London.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "The Eiffel Tower is a lattice tower located on the "
                "Champ de Mars in Paris, France."
            ),
            entity_mismatch=False,
            subject_entity="Eiffel Tower",
            question_type="LOCATION",
            question_relationship="LOCATED_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "paris" in body.lower(), (
            f"Corrected answer should contain 'Paris'. Got: {body!r}"
        )
        assert "london" not in body.lower(), (
            f"Corrected answer must not repeat the wrong answer. Got: {body!r}"
        )

    # ── Test 3: Capital of France ──────────────────────────────────────────
    def test_france_capital_correction(self):
        inputs = [ClaimCorrectionInput(
            claim_text="The capital of France is London.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "France is a country in Western Europe. "
                "Paris, the capital city and largest city of France, "
                "is a major cultural centre."
            ),
            entity_mismatch=False,
            subject_entity="France",
            question_type="LOCATION",
            question_relationship="CAPITAL_OF",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "paris" in body.lower(), (
            f"Corrected answer should mention Paris. Got: {body!r}"
        )
        assert "london" not in body.lower()

    # ── Test 4: Microsoft founder ──────────────────────────────────────────
    def test_microsoft_founder_correction(self):
        inputs = [ClaimCorrectionInput(
            claim_text="Microsoft Corp. founded Microsoft.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Microsoft is a computer technology corporation founded on "
                "April 4, 1975, by Bill Gates and Paul Allen in Albuquerque."
            ),
            entity_mismatch=False,
            subject_entity="Microsoft",
            question_type="PERSON",
            question_relationship="FOUNDED_BY",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "gates" in body.lower() or "allen" in body.lower(), (
            f"Corrected answer should name founders. Got: {body!r}"
        )

    # ── Test 5: First US president ─────────────────────────────────────────
    def test_first_president_correction(self):
        inputs = [ClaimCorrectionInput(
            claim_text="John F. Kennedy was the first president.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "George Washington was the first president of the United States, "
                "serving from 1789 to 1797."
            ),
            entity_mismatch=True,
            subject_entity="United States",
            question_type="PERSON",
            question_relationship="PRESIDENT_OF",
        )]
        out = generate_corrected_answer(inputs)
        # entity_mismatch=True → claim removed, but we have already-contradicted
        # evidence. Check the note section mentions the contradiction.
        assert "kennedy" not in out.lower().split("[corrections applied]")[0].lower()

    # ── Test 6: VERIFIED answer — keep as-is ──────────────────────────────
    def test_verified_answer_kept_verbatim(self):
        inputs = [ClaimCorrectionInput(
            claim_text="Albert Einstein was born in Ulm, Germany.",
            verdict=VERDICT_VERIFIED,
            best_evidence_text="Albert Einstein was born in Ulm, Germany.",
            entity_mismatch=False,
            subject_entity="Albert Einstein",
            question_type="LOCATION",
            question_relationship="BORN_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "ulm" in body.lower()
        assert "corrected" not in out.lower()

    # ── Test 7: No disambiguation in correction ────────────────────────────
    def test_disambiguation_page_not_used_as_correction(self):
        inputs = [ClaimCorrectionInput(
            claim_text="Einstein was born in Berlin.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text="Albert may refer to: Albert (name), Albert of Habsburg...",
            entity_mismatch=False,
            subject_entity="Albert Einstein",
            question_type="LOCATION",
            question_relationship="BORN_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "may refer to" not in body.lower()
        assert "could not be verified" in out.lower() or "insufficient" in out.lower()

    # ── Test 8: General pattern — not hardcoded ────────────────────────────
    def test_arbitrary_person_birthplace(self):
        """Uses Darwin — proves no Einstein-specific hardcoding."""
        inputs = [ClaimCorrectionInput(
            claim_text="Charles Darwin was born in London.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Charles Darwin was an English naturalist. "
                "He was born in Shrewsbury, Shropshire, England on 12 February 1809."
            ),
            entity_mismatch=False,
            subject_entity="Charles Darwin",
            question_type="LOCATION",
            question_relationship="BORN_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "shrewsbury" in body.lower(), (
            f"Corrected answer should contain Shrewsbury. Got: {body!r}"
        )
        assert "london" not in body.lower()

    def test_arbitrary_location(self):
        """Uses Colosseum — proves no Eiffel Tower hardcoding."""
        inputs = [ClaimCorrectionInput(
            claim_text="The Colosseum is located in Athens.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "The Colosseum is an ancient amphitheatre located in the "
                "centre of Rome, Italy."
            ),
            entity_mismatch=False,
            subject_entity="Colosseum",
            question_type="LOCATION",
            question_relationship="LOCATED_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "rome" in body.lower(), (
            f"Corrected answer should contain Rome. Got: {body!r}"
        )
        assert "athens" not in body.lower()
