"""Combines similarity, subject-presence, entity-consistency, relationship,
answer-value, and contradiction signals into one of four verdicts per claim:

    VERIFIED | PARTIALLY_SUPPORTED | UNSUPPORTED | CONTRADICTED

--- Architecture (three stages) ---

STAGE 1 — Subject-aware quality-based evidence selection
  Evidence items are re-ranked by a quality score.  Crucially, evidence that
  does NOT mention the question's subject entity receives subject_presence=0,
  dragging its quality_score well below evidence that does mention the subject.
  This is the fix for: "Albert Einstein born in St. Petersburg" retrieving the
  Saint Petersburg article and treating it as supporting evidence.

STAGE 2 — Answer-value check
  When QuestionAnalysis is supplied, the answer VALUE is compared against
  the value the evidence gives for the same relationship.  Catches
  "St. Petersburg" vs "Ulm, Germany" even at high semantic similarity.

STAGE 3 — Multi-signal decision rules (strict priority order)
  Rule 0  Answer-value contradiction               → CONTRADICTED
  Rule 1  NLI contradiction                         → CONTRADICTED
  Rule 2  Entity mismatch + similarity ≥ PARTIAL   → CONTRADICTED
  Rule 3  Entity mismatch + similarity < PARTIAL   → UNSUPPORTED
  Rule 4  Strong sim + ≥2 sources + rel discussed + AV ok
          → VERIFIED
  Rule 5  Strong sim (partial conditions)          → PARTIALLY_SUPPORTED
  Rule 6  Partial sim                              → PARTIALLY_SUPPORTED
  Rule 7  All else                                 → UNSUPPORTED
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings
from app.verification.agreement_checker import count_agreeing_sources
from app.verification.contradiction_detector import detect_contradiction
from app.verification.entity_matcher import check_entity_consistency
from app.verification.evidence_quality_scorer import rank_evidence_by_quality
from app.verification.relationship_checker import check_relationship
from app.verification.similarity_scorer import ScoredEvidence

settings = get_settings()

VERDICT_VERIFIED            = "VERIFIED"
VERDICT_PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
VERDICT_UNSUPPORTED         = "UNSUPPORTED"
VERDICT_CONTRADICTED        = "CONTRADICTED"


@dataclass
class ClaimVerdict:
    verdict: str
    max_similarity: float
    num_sources_agreeing: int
    contradiction_detected: bool
    entity_mismatch: bool
    explanation: str
    entity_presence_score: float = field(default=1.0)
    relationship_discussed: bool = field(default=True)
    primary_evidence_quality: float = field(default=0.0)
    answer_value_status: str = field(default="VALUE_UNVERIFIABLE")


def determine_verdict(
    claim: str,
    scored_evidence: list[ScoredEvidence],
    question_analysis=None,
    answer_text: str = "",
    subject_entity: str = "",
) -> ClaimVerdict:
    """Return a ClaimVerdict for *claim* given its pre-scored evidence list.

    Parameters
    ----------
    claim:
        The factual claim string being verified.
    scored_evidence:
        Evidence items scored by semantic similarity (sorted descending).
    question_analysis:
        Optional QuestionAnalysis — enables answer-value checking.
    answer_text:
        The original LLM answer (used for answer-value checking).
    subject_entity:
        The focus entity from the question (e.g. "Albert Einstein").
        Evidence not mentioning this entity receives a quality penalty.
    """

    # ── Stage 0: No evidence ──────────────────────────────────────────────
    if not scored_evidence:
        return ClaimVerdict(
            verdict=VERDICT_UNSUPPORTED,
            max_similarity=0.0,
            num_sources_agreeing=0,
            contradiction_detected=False,
            entity_mismatch=False,
            entity_presence_score=0.0,
            relationship_discussed=False,
            primary_evidence_quality=0.0,
            answer_value_status="VALUE_UNVERIFIABLE",
            explanation=(
                "No evidence could be retrieved from Wikipedia, Wikidata, or "
                "DuckDuckGo for this claim, so it cannot be verified."
            ),
        )

    # ── Stage 1: Re-rank by quality (subject-aware) ───────────────────────
    quality_ranked = rank_evidence_by_quality(
        claim, scored_evidence, subject_entity=subject_entity
    )
    primary = quality_ranked[0]
    max_similarity = scored_evidence[0].similarity

    # Agreement counting (entity-filtered)
    agree_count = count_agreeing_sources(scored_evidence, claim=claim)

    # ── Entity / relationship / NLI checks on PRIMARY evidence ───────────
    entity_check = check_entity_consistency(claim, primary.evidence_text)
    rel_check     = check_relationship(claim, primary.evidence_text)
    contradiction_detected, nli_label = detect_contradiction(
        claim, primary.evidence_text
    )

    # ── Stage 2: Answer-value check ───────────────────────────────────────
    av_status = "VALUE_UNVERIFIABLE"
    av_reason = ""
    if question_analysis is not None:
        try:
            from app.verification.answer_value_checker import check_answer_value
            check_text = answer_text.strip() if answer_text.strip() else claim
            # Check against the quality-primary evidence first
            av_result = check_answer_value(
                question_analysis,
                check_text,
                primary.evidence_text,
                subject_entity=subject_entity,
            )
            av_status = av_result.status
            av_reason = av_result.reason
            # If primary gave CONTRADICTED, also check other evidence items —
            # a different source might support the claim.  Only upgrade
            # CONTRADICTED → UNVERIFIABLE if another item says SUPPORTED.
            if av_status == "VALUE_CONTRADICTED" and len(quality_ranked) > 1:
                for alt in quality_ranked[1:]:
                    alt_result = check_answer_value(
                        question_analysis, check_text, alt.evidence_text,
                        subject_entity=subject_entity,
                    )
                    if alt_result.status == "VALUE_SUPPORTED":
                        av_status = "VALUE_SUPPORTED"
                        av_reason = f"[alt evidence] {alt_result.reason}"
                        break
        except Exception as exc:
            av_reason = f"Answer-value check skipped: {exc}"

    # ── Build explanation ─────────────────────────────────────────────────
    reasons: list[str] = [
        f"Best similarity: {max_similarity:.2f} (from '{scored_evidence[0].source_name}').",
        f"Primary evidence quality {primary.quality_score:.2f} "
        f"(subject_presence={primary.subject_presence:.1f}) "
        f"from '{primary.source_name}'.",
        f"{agree_count} of up to 3 sources provide supporting evidence.",
        f"Entity presence: {entity_check.entity_presence_score:.2f}.",
    ]
    if av_reason:
        reasons.append(f"Answer-value: {av_reason}")
    if rel_check.matched_category:
        reasons.append(f"Relationship '{rel_check.matched_category}' found in evidence.")
    elif not rel_check.relationship_discussed:
        reasons.append(rel_check.reason)

    # ── Rule 0: Answer-value contradiction ───────────────────────────────
    if av_status == "VALUE_CONTRADICTED":
        reasons.append(
            "Answer value directly contradicts the value stated in evidence "
            "for the requested relationship."
        )
        return ClaimVerdict(
            verdict=VERDICT_CONTRADICTED,
            max_similarity=max_similarity,
            num_sources_agreeing=agree_count,
            contradiction_detected=contradiction_detected,
            entity_mismatch=entity_check.mismatch,
            entity_presence_score=entity_check.entity_presence_score,
            relationship_discussed=rel_check.relationship_discussed,
            primary_evidence_quality=primary.quality_score,
            answer_value_status=av_status,
            explanation=" ".join(reasons),
        )

    # ── Rule 1: NLI contradiction ─────────────────────────────────────────
    if contradiction_detected and rel_check.relationship_discussed :
        reasons.append(
            f"NLI model classified primary evidence as '{nli_label}'."
        )
        return ClaimVerdict(
            verdict=VERDICT_CONTRADICTED,
            max_similarity=max_similarity,
            num_sources_agreeing=agree_count,
            contradiction_detected=True,
            entity_mismatch=entity_check.mismatch,
            entity_presence_score=entity_check.entity_presence_score,
            relationship_discussed=rel_check.relationship_discussed,
            primary_evidence_quality=primary.quality_score,
            answer_value_status=av_status,
            explanation=" ".join(reasons),
        )

    # ── Rule 2: Entity mismatch at partial-or-stronger similarity ─────────
    if entity_check.mismatch and max_similarity >= settings.SIMILARITY_PARTIAL_THRESHOLD:
        reasons.append(entity_check.reason)
        reasons.append(
            "Evidence semantically related but key entity in claim does not match."
        )
        return ClaimVerdict(
            verdict=VERDICT_CONTRADICTED,
            max_similarity=max_similarity,
            num_sources_agreeing=agree_count,
            contradiction_detected=False,
            entity_mismatch=True,
            entity_presence_score=entity_check.entity_presence_score,
            relationship_discussed=rel_check.relationship_discussed,
            primary_evidence_quality=primary.quality_score,
            answer_value_status=av_status,
            explanation=" ".join(reasons),
        )

    # ── Rule 3: Entity mismatch at low similarity ─────────────────────────
    if entity_check.mismatch:
        reasons.append(entity_check.reason)
        return ClaimVerdict(
            verdict=VERDICT_UNSUPPORTED,
            max_similarity=max_similarity,
            num_sources_agreeing=agree_count,
            contradiction_detected=False,
            entity_mismatch=True,
            entity_presence_score=entity_check.entity_presence_score,
            relationship_discussed=rel_check.relationship_discussed,
            primary_evidence_quality=primary.quality_score,
            answer_value_status=av_status,
            explanation=" ".join(reasons),
        )

    # ── Rules 4–7: No entity mismatch ─────────────────────────────────────
    av_supported = (av_status == "VALUE_SUPPORTED")

    if max_similarity >= settings.SIMILARITY_STRONG_THRESHOLD:
        if agree_count >= settings.AGREEMENT_MIN_SOURCES:
            if rel_check.relationship_discussed and (av_supported or av_status == "VALUE_UNVERIFIABLE"):
                verdict = VERDICT_VERIFIED
                reasons.append("Strong similarity, multi-source agreement, relationship discussed.")
            else:
                verdict = VERDICT_PARTIALLY_SUPPORTED
                reasons.append("Strong similarity + agreement, but relationship/value inconclusive.")
        else:
            verdict = VERDICT_PARTIALLY_SUPPORTED
            reasons.append("Strong similarity but only one source.")
    elif max_similarity >= settings.SIMILARITY_PARTIAL_THRESHOLD:
        verdict = VERDICT_PARTIALLY_SUPPORTED
        reasons.append("Medium similarity; related but not strongly matched.")
    else:
        verdict = VERDICT_UNSUPPORTED
        reasons.append("Similarity too low to consider claim supported.")

    return ClaimVerdict(
        verdict=verdict,
        max_similarity=max_similarity,
        num_sources_agreeing=agree_count,
        contradiction_detected=False,
        entity_mismatch=entity_check.mismatch,
        entity_presence_score=entity_check.entity_presence_score,
        relationship_discussed=rel_check.relationship_discussed,
        primary_evidence_quality=primary.quality_score,
        answer_value_status=av_status,
        explanation=" ".join(reasons),
    )
