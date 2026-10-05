"""Coordinates the full pipeline:

User Question
  → LLM Answer Generation
  → Question Analysis
  → Claim Extraction  (short-answer fallback)
  → Subject-Centred Evidence Retrieval  ← KEY FIX
  → Semantic Similarity Scoring
  → Fact Verification  (subject-presence + answer-value checks)
  → Hallucination Analysis
  → Confidence Score
  → Corrected Answer
  → Verification Report

CRITICAL FIX: Subject-centred retrieval
----------------------------------------
Previously, retrieval queries were built from the claim text, e.g.:
  "Albert Einstein was born in St. Petersburg."
Wikipedia would find the Saint Petersburg article, which looks highly similar
to the claim but does NOT establish "Einstein → born in → St. Petersburg".

Now, when a QuestionAnalysis is available, we:
  1. Query using the SUBJECT + RELATIONSHIP keywords (not the answer value).
     E.g. "Albert Einstein born birthplace" instead of "...St. Petersburg".
  2. Pass the subject entity to evidence quality scoring, which penalises
     evidence that does NOT mention the subject (subject_presence=0.0).

Both changes are necessary: Fix 1 fetches subject-centric articles;
Fix 2 rejects any stray evidence that slipped through without the subject.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.correction.answer_corrector import ClaimCorrectionInput, generate_corrected_answer
from app.db import models
from app.llm.factory import get_llm_provider
from app.nlp.claim_extractor import extract_claims, build_claims_from_question_answer
from app.report.report_generator import ClaimReportEntry, build_report, report_to_json_string
from app.retrieval.aggregator import retrieve_evidence
from app.retrieval.query_builder import build_subject_query
from app.verification.confidence_calculator import calculate_confidence
from app.verification.explainability import explain_claim
from app.verification.hallucination_analysis import analyze
from app.verification.question_analyzer import analyze_question
from app.verification.similarity_scorer import score_evidence_against_claim
from app.verification.verdict_engine import determine_verdict

logger = logging.getLogger(__name__)

_llm_provider = None


def _get_llm():
    global _llm_provider
    if _llm_provider is None:
        _llm_provider = get_llm_provider()
    return _llm_provider


def run_pipeline(db: Session, question_text: str) -> models.Question:
    """Run the full pipeline and persist results."""
    llm = _get_llm()

    # 1. Persist the question.
    question = models.Question(question_text=question_text)
    db.add(question)
    db.flush()

    # 2. LLM answer generation.
    raw_answer = llm.generate_answer(question_text)
    provider_name = type(llm).__name__

    answer = models.Answer(
        question_id=question.id, raw_answer=raw_answer, llm_provider=provider_name
    )
    db.add(answer)
    db.flush()

    # 3a. Question analysis.
    try:
        qa = analyze_question(question_text)
    except Exception as exc:
        logger.warning("Question analysis failed (%s); continuing without it.", exc)
        qa = None

    # Extract subject entity once — used for retrieval query and quality scoring.
    subject_entity = qa.focus_entity if qa else ""

    # 3b. Claim extraction — sentence splitting first, short-answer fallback.
    claim_texts = extract_claims(raw_answer)
    if not claim_texts:
        try:
            claim_texts = build_claims_from_question_answer(question_text, raw_answer)
        except Exception as exc:
            logger.warning("Short-answer claim builder failed (%s).", exc)

    claim_verdicts_for_analysis: list[str] = []
    claim_confidences_for_analysis: list[float] = []
    claim_report_entries: list[ClaimReportEntry] = []
    correction_inputs: list[ClaimCorrectionInput] = []

    for idx, claim_text in enumerate(claim_texts):
        claim = models.Claim(answer_id=answer.id, claim_text=claim_text, claim_index=idx)
        db.add(claim)
        db.flush()

        # 4. Evidence retrieval — use SUBJECT-CENTRED query.
        #
        # KEY FIX: instead of searching for the claim text (which contains
        # the answer value), we search for the SUBJECT + RELATIONSHIP.
        # This prevents retrieval from returning articles about the answer
        # value that don't mention the subject.
        #
        # Example:
        #   claim = "Albert Einstein was born in St. Petersburg."
        #   retrieval_query = "Albert Einstein born birthplace"
        #                      ↑ not "st. petersburg"
        #
        # Fallback: if no question analysis, use the claim text as before.
        retrieval_query = build_subject_query(qa, fallback_claim=claim_text)
        logger.debug("Retrieval query: %r (claim: %r)", retrieval_query, claim_text)

        evidence_items = retrieve_evidence(retrieval_query)
        evidence_tuples = [
            (item.evidence_text, item.source_name, item.source_url)
            for item in evidence_items
        ]

        # 5. Semantic similarity scoring — score against the CLAIM text
        #    (not the retrieval query), so the similarity is between the
        #    claim and the evidence.
        scored_evidence = score_evidence_against_claim(claim_text, evidence_tuples)

        for scored in scored_evidence:
            db.add(
                models.Evidence(
                    claim_id=claim.id,
                    source_name=scored.source_name,
                    source_url=scored.source_url,
                    evidence_text=scored.evidence_text,
                    similarity_score=scored.similarity,
                )
            )

        # 6. Fact verification — passes subject_entity so the quality scorer
        #    can penalise evidence that doesn't mention the subject.
        verdict = determine_verdict(
            claim_text,
            scored_evidence,
            question_analysis=qa,
            answer_text=raw_answer,
            subject_entity=subject_entity,
        )

        db.add(
            models.VerificationResult(
                claim_id=claim.id,
                verdict=verdict.verdict,
                max_similarity_score=verdict.max_similarity,
                num_sources_agreeing=verdict.num_sources_agreeing,
                contradiction_detected=verdict.contradiction_detected,
                entity_mismatch=verdict.entity_mismatch,
                explanation=verdict.explanation,
            )
        )

        # 7. Confidence score.
        confidence = calculate_confidence(
            similarity=verdict.max_similarity,
            agree_count=verdict.num_sources_agreeing,
            contradiction_detected=verdict.contradiction_detected,
            entity_mismatch=verdict.entity_mismatch,
            entity_presence_score=verdict.entity_presence_score,
        )
        db.add(models.ConfidenceScore(answer_id=answer.id, claim_id=claim.id, score=confidence))

        claim_verdicts_for_analysis.append(verdict.verdict)
        claim_confidences_for_analysis.append(confidence)

        best_evidence_text = scored_evidence[0].evidence_text if scored_evidence else None
        all_evidence_texts = [s.evidence_text for s in scored_evidence[1:] if s.evidence_text]

        correction_inputs.append(
            ClaimCorrectionInput(
                claim_text=claim_text,
                verdict=verdict.verdict,
                best_evidence_text=best_evidence_text,
                entity_mismatch=verdict.entity_mismatch,
                subject_entity=subject_entity,
                question_type=qa.question_type if qa else "",
                question_relationship=qa.relationship if qa else "",
                all_evidence_texts=all_evidence_texts,
            )
        )

        claim_report_entries.append(
            ClaimReportEntry(
                claim_text=claim_text,
                verdict=verdict.verdict,
                confidence=confidence,
                explanation=explain_claim(claim_text, verdict),
                evidence=[
                    {
                        "source_name": s.source_name,
                        "source_url": s.source_url,
                        "evidence_text": s.evidence_text,
                        "similarity_score": s.similarity,
                    }
                    for s in scored_evidence
                ],
            )
        )

    # 8. Hallucination analysis.
    analysis = analyze(claim_verdicts_for_analysis, claim_confidences_for_analysis)
    db.add(
        models.ConfidenceScore(
            answer_id=answer.id, claim_id=None, score=analysis.overall_confidence
        )
    )

    # 9. Corrected answer.
    corrected_answer = generate_corrected_answer(correction_inputs)

    answer.corrected_answer = corrected_answer
    answer.overall_verdict = analysis.overall_verdict
    answer.overall_confidence = analysis.overall_confidence

    # 10. Verification report.
    summary_counts = {
        "num_claims": analysis.num_claims,
        "num_verified": analysis.num_verified,
        "num_partially_supported": analysis.num_partially_supported,
        "num_unsupported": analysis.num_unsupported,
        "num_contradicted": analysis.num_contradicted,
    }
    report_dict, report_text = build_report(
        question=question_text,
        raw_answer=raw_answer,
        corrected_answer=corrected_answer,
        overall_verdict=analysis.overall_verdict,
        overall_confidence=analysis.overall_confidence,
        claim_entries=claim_report_entries,
        summary_counts=summary_counts,
    )
    db.add(
        models.VerificationReport(
            question_id=question.id,
            report_json=report_to_json_string(report_dict),
            report_text=report_text,
            overall_verdict=analysis.overall_verdict,
        )
    )

    db.commit()
    db.refresh(question)
    return question
