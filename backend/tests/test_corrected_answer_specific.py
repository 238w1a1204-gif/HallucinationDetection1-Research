"""Regression tests for specific, relationship-aware corrected answers.

These tests verify that the corrected answer extracts the PRECISE value
needed to answer the original question — not a generic first sentence.

Key scenarios:
  Capital:       "What is the capital of France?" → "The capital of France is Paris."
  Inventor:      "Who invented the telephone?"    → "... credited to Alexander Graham Bell."
  First president: UNSUPPORTED case still attempts extraction from evidence.
  Biographical:  "Who was Gandhi?" → descriptive sentence about Gandhi.
  Date vs Person: "When was X founded?" → year; "Who founded X?" → person name.
  Preservation:  All previously working cases (Microsoft, Marie Curie, Eiffel Tower)
                 continue to produce correct specific answers.

Nothing is hard-coded in production logic — all values come from the
evidence text supplied at runtime.
"""
from __future__ import annotations

import pytest
from app.correction.answer_corrector import (
    ClaimCorrectionInput,
    extract_answer_from_evidence,
    generate_corrected_answer,
    extract_biographical_sentence,
)
from app.verification.verdict_engine import VERDICT_CONTRADICTED, VERDICT_UNSUPPORTED, VERDICT_VERIFIED

# ═══════════════════════════════════════════════════════════════════════════════
# Capital question
# ═══════════════════════════════════════════════════════════════════════════════

class TestCapitalCorrection:

    def test_extract_capital_from_x_is_capital_pattern(self):
        """France article typically says 'Paris, the capital city...'"""
        ev = ("France is a country in Western Europe. "
              "Paris, the capital city and largest city of France, "
              "is a major cultural centre.")
        val = extract_answer_from_evidence(ev, "CAPITAL_OF", "France", "*")
        assert val is not None
        assert "paris" in val.lower(), f"Expected Paris, got {val!r}"

    def test_extract_capital_from_capital_is_pattern(self):
        ev = "Germany is a federal republic. The capital city is Berlin."
        val = extract_answer_from_evidence(ev, "CAPITAL_OF", "Germany", "*")
        assert val is not None
        assert "berlin" in val.lower(), f"Expected Berlin, got {val!r}"

    def test_extract_capital_reverse_pattern(self):
        ev = "Tokyo is the capital of Japan and its most populous city."
        val = extract_answer_from_evidence(ev, "CAPITAL_OF", "Japan", "*")
        assert val is not None
        assert "tokyo" in val.lower(), f"Expected Tokyo, got {val!r}"

    def test_corrected_answer_for_capital_is_specific(self):
        """Corrected answer must say 'The capital of France is Paris.'
        NOT 'France, officially the French Republic, is a country...'"""
        inputs = [ClaimCorrectionInput(
            claim_text="The capital of France is London.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "France is a country in Western Europe. "
                "Paris, the capital city and largest city of France, "
                "is a major cultural centre."
            ),
            subject_entity="France",
            question_type="LOCATION",
            question_relationship="CAPITAL_OF",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "paris" in body.lower(), f"Correction must mention Paris. Got: {body!r}"
        assert "london" not in body.lower(), f"Must not repeat wrong answer. Got: {body!r}"
        # Must NOT start with the generic France intro
        assert not body.lower().startswith("france, officially"), (
            f"Correction must be specific, not a generic intro. Got: {body!r}"
        )

    def test_arbitrary_capital_different_country(self):
        """General test with Japan — proves no France-specific hardcoding."""
        inputs = [ClaimCorrectionInput(
            claim_text="The capital of Japan is Osaka.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text="Tokyo is the capital of Japan and its most populous city.",
            subject_entity="Japan",
            question_type="LOCATION",
            question_relationship="CAPITAL_OF",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "tokyo" in body.lower()
        assert "osaka" not in body.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Inventor question
# ═══════════════════════════════════════════════════════════════════════════════

class TestInventorCorrection:

    def test_extract_invented_by_pattern(self):
        ev = "The telephone was invented by Alexander Graham Bell in 1876."
        val = extract_answer_from_evidence(ev, "DISCOVERED_BY", "telephone", "*")
        assert val is not None
        assert "bell" in val.lower(), f"Expected Bell, got {val!r}"

    def test_extract_credited_to_pattern(self):
        ev = ("The invention of the telephone is commonly credited to "
              "Alexander Graham Bell, who was awarded the first patent in 1876.")
        val = extract_answer_from_evidence(ev, "DISCOVERED_BY", "telephone", "*")
        assert val is not None
        assert "bell" in val.lower(), f"Expected Bell, got {val!r}"

    def test_extract_who_invented_pattern(self):
        ev = "Alexander Graham Bell, who invented the telephone, was born in Scotland."
        val = extract_answer_from_evidence(ev, "DISCOVERED_BY", "telephone", "*")
        assert val is not None
        assert "bell" in val.lower(), f"Expected Bell, got {val!r}"

    def test_corrected_answer_for_inventor_is_specific(self):
        """Must name the inventor, not describe the device."""
        inputs = [ClaimCorrectionInput(
            claim_text="John D. Humphrey invented the telephone.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "The telephone was invented by Alexander Graham Bell in 1876. "
                "Bell was awarded the first patent for the telephone."
            ),
            subject_entity="telephone",
            question_type="PERSON",
            question_relationship="DISCOVERED_BY",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "bell" in body.lower(), f"Must name Bell. Got: {body!r}"
        assert "humphrey" not in body.lower()

    def test_arbitrary_inventor_radium(self):
        """General test — proves no telephone-specific hardcoding."""
        inputs = [ClaimCorrectionInput(
            claim_text="Thomas Edison discovered radium.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Radium was discovered by Marie Curie and Pierre Curie in 1898."
            ),
            subject_entity="radium",
            question_type="PERSON",
            question_relationship="DISCOVERED_BY",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "curie" in body.lower(), f"Must name Curie. Got: {body!r}"
        assert "edison" not in body.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# First president — UNSUPPORTED case
# ═══════════════════════════════════════════════════════════════════════════════

class TestFirstPresidentCorrection:

    def test_extract_first_president_pattern(self):
        ev = ("George Washington was the first president of the United States, "
              "serving from 1789 to 1797.")
        val = extract_answer_from_evidence(ev, "PRESIDENT_OF", "United States", "*")
        assert val is not None
        assert "washington" in val.lower(), f"Expected Washington, got {val!r}"

    def test_corrected_answer_for_unsupported_when_evidence_present(self):
        """UNSUPPORTED verdict with valid evidence → should attempt extraction."""
        inputs = [ClaimCorrectionInput(
            claim_text="john f kennedy",
            verdict=VERDICT_UNSUPPORTED,
            best_evidence_text=(
                "George Washington was the first president of the United States, "
                "serving from 1789 to 1797 after leading the Continental Army."
            ),
            subject_entity="United States",
            question_type="PERSON",
            question_relationship="PRESIDENT_OF",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "washington" in body.lower(), (
            f"UNSUPPORTED with evidence should extract Washington. Got: {body!r}"
        )
        assert "kennedy" not in body.lower()

    def test_unsupported_without_evidence_keeps_fallback(self):
        """UNSUPPORTED with no evidence → keeps 'insufficient evidence' message."""
        inputs = [ClaimCorrectionInput(
            claim_text="john f kennedy",
            verdict=VERDICT_UNSUPPORTED,
            best_evidence_text=None,
            subject_entity="United States",
            question_type="PERSON",
            question_relationship="PRESIDENT_OF",
        )]
        out = generate_corrected_answer(inputs)
        assert "could not be verified" in out.lower() or "insufficient" in out.lower()

    def test_arbitrary_president_different_country(self):
        """General test — not specific to US."""
        inputs = [ClaimCorrectionInput(
            claim_text="Napoleon was the first president of France.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Louis-Napoléon Bonaparte became the first president of France "
                "in December 1848."
            ),
            subject_entity="France",
            question_type="PERSON",
            question_relationship="PRESIDENT_OF",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "napoléon" in body.lower() or "napoleon" in body.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Biographical question
# ═══════════════════════════════════════════════════════════════════════════════

class TestBiographicalCorrection:

    def test_extract_biographical_sentence_gandhi(self):
        ev = ("Mohandas Karamchand Gandhi was an Indian lawyer, anti-colonial "
              "nationalist and political ethicist who employed nonviolent resistance. "
              "Gandhi led India's independence movement against British rule.")
        bio = extract_biographical_sentence(ev, "Mahatma Gandhi")
        assert bio is not None
        # Should contain a description
        assert "indian" in bio.lower() or "lawyer" in bio.lower() or "nationalist" in bio.lower()

    def test_extract_biographical_sentence_generic(self):
        """Works for any person — not Gandhi-specific."""
        ev = ("Alan Turing was an English mathematician, computer scientist, "
              "logician and cryptanalyst. He was highly influential in the "
              "development of theoretical computer science.")
        bio = extract_biographical_sentence(ev, "Alan Turing")
        assert bio is not None
        assert "mathematician" in bio.lower() or "computer" in bio.lower()

    def test_corrected_answer_for_biographical_unsupported(self):
        inputs = [ClaimCorrectionInput(
            claim_text="a revolutionary",
            verdict=VERDICT_UNSUPPORTED,
            best_evidence_text=(
                "Mahatma Gandhi was an Indian lawyer, anti-colonial nationalist "
                "and political ethicist who employed nonviolent resistance."
            ),
            subject_entity="Mahatma Gandhi",
            question_type="BIOGRAPHICAL",
            question_relationship="BIOGRAPHICAL_DESCRIPTION",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "gandhi" in body.lower() or "indian" in body.lower(), (
            f"Biographical correction must describe Gandhi. Got: {body!r}"
        )

    def test_biographical_insufficient_evidence_keeps_fallback(self):
        """When evidence doesn't describe the subject, keep fallback."""
        inputs = [ClaimCorrectionInput(
            claim_text="a revolutionary",
            verdict=VERDICT_UNSUPPORTED,
            best_evidence_text="Apple Inc. is a technology company based in California.",
            subject_entity="Mahatma Gandhi",
            question_type="BIOGRAPHICAL",
            question_relationship="BIOGRAPHICAL_DESCRIPTION",
        )]
        out = generate_corrected_answer(inputs)
        assert "apple" not in out.lower().split("[corrections applied]")[0].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Date vs Person disambiguation for founding questions
# ═══════════════════════════════════════════════════════════════════════════════

class TestFoundingDateVsPerson:

    FOUNDING_EV = ("Microsoft is a computer technology corporation founded on "
                   "April 4, 1975, by Bill Gates and Paul Allen in Albuquerque.")

    def test_who_founded_extracts_person(self):
        """question_type=PERSON → extract the founder name(s)."""
        val = extract_answer_from_evidence(
            self.FOUNDING_EV, "FOUNDED_BY", "Microsoft", "PERSON"
        )
        assert val is not None
        assert "gates" in val.lower() or "allen" in val.lower(), (
            f"Expected person names, got: {val!r}"
        )

    def test_when_founded_extracts_date(self):
        """question_type=DATE → extract the year, NOT the person."""
        val = extract_answer_from_evidence(
            self.FOUNDING_EV, "FOUNDED_BY", "Microsoft", "DATE"
        )
        assert val is not None
        assert "1975" in val, f"Expected 1975, got: {val!r}"
        assert "gates" not in val.lower(), (
            f"Date question must not return person name. Got: {val!r}"
        )

    def test_corrected_answer_who_founded_has_person(self):
        inputs = [ClaimCorrectionInput(
            claim_text="microsoft corp",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=self.FOUNDING_EV,
            subject_entity="Microsoft",
            question_type="PERSON",
            question_relationship="FOUNDED_BY",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "gates" in body.lower() or "allen" in body.lower()
        assert "1975" not in body or "gates" in body.lower()  # date ok if name also present

    def test_corrected_answer_when_founded_has_date(self):
        inputs = [ClaimCorrectionInput(
            claim_text="Microsoft was founded in 1990.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=self.FOUNDING_EV,
            subject_entity="Microsoft",
            question_type="DATE",
            question_relationship="FOUNDED_BY",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "1975" in body, f"Date correction must contain 1975. Got: {body!r}"
        # Must not be about the person for a date question
        # (body may say "Microsoft was founded in 1975.")
        assert "microsoft" in body.lower()

    def test_general_date_vs_person_amazon(self):
        """Proves not Microsoft-specific."""
        ev = "Amazon was founded by Jeff Bezos in 1994 in Bellevue, Washington."
        person = extract_answer_from_evidence(ev, "FOUNDED_BY", "Amazon", "PERSON")
        date   = extract_answer_from_evidence(ev, "FOUNDED_BY", "Amazon", "DATE")
        assert person is not None and "bezos" in person.lower()
        assert date   is not None and "1994" in date


# ═══════════════════════════════════════════════════════════════════════════════
# Preservation: previously working cases still work
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreservePreviouslyWorkingCases:

    def test_einstein_birthplace_still_works(self):
        inputs = [ClaimCorrectionInput(
            claim_text="Albert Einstein was born in St. Petersburg.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Albert Einstein (14 March 1879) was a German-born physicist. "
                "He was born in Ulm, Kingdom of Württemberg."
            ),
            subject_entity="Albert Einstein",
            question_type="LOCATION",
            question_relationship="BORN_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "ulm" in body.lower()
        assert "st. petersburg" not in body.lower()

    def test_microsoft_founded_by_still_works(self):
        inputs = [ClaimCorrectionInput(
            claim_text="microsoft corp",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Microsoft is a computer technology corporation founded on "
                "April 4, 1975, by Bill Gates and Paul Allen in Albuquerque."
            ),
            subject_entity="Microsoft",
            question_type="PERSON",
            question_relationship="FOUNDED_BY",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "gates" in body.lower() or "allen" in body.lower()

    def test_eiffel_tower_location_still_works(self):
        inputs = [ClaimCorrectionInput(
            claim_text="The Eiffel Tower is in London.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "The Eiffel Tower is a lattice tower located on the "
                "Champ de Mars in Paris, France."
            ),
            subject_entity="Eiffel Tower",
            question_type="LOCATION",
            question_relationship="LOCATED_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "paris" in body.lower()
        assert "london" not in body.lower()

    def test_marie_curie_birthplace_still_works(self):
        inputs = [ClaimCorrectionInput(
            claim_text="Marie Curie was born in Paris.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Marie Curie (7 November 1867) was born in Warsaw, Poland. "
                "She was a Polish physicist and chemist."
            ),
            subject_entity="Marie Curie",
            question_type="LOCATION",
            question_relationship="BORN_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "warsaw" in body.lower()
        assert "paris" not in body.lower()

    def test_darwin_birthplace_general(self):
        """Confirms generalisation — not hardcoded for specific people."""
        inputs = [ClaimCorrectionInput(
            claim_text="Charles Darwin was born in London.",
            verdict=VERDICT_CONTRADICTED,
            best_evidence_text=(
                "Charles Darwin was born in Shrewsbury, Shropshire, England."
            ),
            subject_entity="Charles Darwin",
            question_type="LOCATION",
            question_relationship="BORN_IN",
        )]
        out = generate_corrected_answer(inputs)
        body = out.split("[Corrections applied]")[0].strip()
        assert "shrewsbury" in body.lower()
        assert "london" not in body.lower()
