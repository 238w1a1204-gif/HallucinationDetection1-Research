"""
FastAPI application entry point for the Hallucination Detection & Answer
Verification backend.

Endpoints
---------
GET  /api/health                   – liveness check
POST /api/questions                – submit a question and run the full pipeline
GET  /api/questions                – list all questions (summary)
GET  /api/questions/{id}           – full detail for one question/answer
GET  /api/questions/{id}/claims    – per-claim verification breakdown
GET  /api/questions/{id}/report    – full verification report (JSON + text)
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.database import get_db, init_db
from app.db import models
from app.db.schemas import (
    ClaimOut,
    EvidenceOut,
    HealthOut,
    QuestionCreateRequest,
    QuestionCreateResponse,
    QuestionDetailOut,
    QuestionSummaryOut,
    ReportOut,
    VerificationResultOut,
)
from app.services.pipeline_orchestrator import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event("startup"))
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database …")
    init_db()
    logger.info("Database ready.")
    yield  # application runs here


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hallucination Detection & Answer Verification",
    description=(
        "Hybrid multi-source fact verification pipeline. "
        "Submits a question to an LLM, extracts claims, retrieves evidence "
        "from Wikipedia/Wikidata/DuckDuckGo, and verifies each claim."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN, "http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helper: load question or 404
# ---------------------------------------------------------------------------

def _get_question_or_404(question_id: int, db: Session) -> models.Question:
    question = db.query(models.Question).filter(models.Question.id == question_id).first()
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question {question_id} not found.",
        )
    return question


def _build_claims_out(answer: models.Answer) -> list[ClaimOut]:
    """Convert an Answer ORM object's claims into a list of ClaimOut schemas."""
    claims_out: list[ClaimOut] = []
    for claim in sorted(answer.claims, key=lambda c: c.claim_index):
        evidence_out = [
            EvidenceOut(
                id=ev.id,
                source_name=ev.source_name,
                source_url=ev.source_url,
                evidence_text=ev.evidence_text,
                similarity_score=ev.similarity_score,
            )
            for ev in claim.evidence
        ]
        vr = claim.verification_result
        vr_out = (
            VerificationResultOut(
                verdict=vr.verdict,
                max_similarity_score=vr.max_similarity_score,
                num_sources_agreeing=vr.num_sources_agreeing,
                contradiction_detected=vr.contradiction_detected,
                entity_mismatch=vr.entity_mismatch,
                explanation=vr.explanation,
            )
            if vr
            else None
        )
        claim_confidence: float | None = None
        for cs in answer.confidence_scores:
            if cs.claim_id == claim.id:
                claim_confidence = cs.score
                break

        claims_out.append(
            ClaimOut(
                id=claim.id,
                claim_text=claim.claim_text,
                claim_index=claim.claim_index,
                evidence=evidence_out,
                verification_result=vr_out,
                confidence=claim_confidence,
            )
        )
    return claims_out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthOut, tags=["meta"])
def health_check() -> HealthOut:
    """Returns the service status and which LLM provider is configured."""
    return HealthOut(status="ok", llm_provider=settings.LLM_PROVIDER)


@app.post(
    "/api/questions",
    response_model=QuestionCreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["questions"],
)
def create_question(
    body: QuestionCreateRequest,
    db: Session = Depends(get_db),
) -> QuestionCreateResponse:
    """Submit a question and run the full verification pipeline.

    This is the main endpoint. It may take 10-60 seconds depending on the
    configured LLM provider and network latency for evidence retrieval.
    """
    if not body.question or not body.question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question text must not be empty.",
        )

    try:
        question = run_pipeline(db=db, question_text=body.question.strip())
    except Exception as exc:
        logger.exception("Pipeline failed for question: %s", body.question)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        ) from exc

    # The question ORM object now has a related Answer with claims persisted.
    answer = question.answers[0] if question.answers else None
    if answer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline completed but no answer was persisted.",
        )

    claims_out = _build_claims_out(answer)

    # Populate question analysis fields (best-effort; never raises)
    q_type = None
    q_rel = None
    try:
        from app.verification.question_analyzer import analyze_question
        _qa = analyze_question(question.question_text)
        q_type = _qa.question_type
        q_rel = _qa.relationship_label
    except Exception:
        pass

    return QuestionCreateResponse(
        question_id=question.id,
        question=question.question_text,
        answer=answer.raw_answer,
        corrected_answer=answer.corrected_answer,
        overall_verdict=answer.overall_verdict or "UNKNOWN",
        overall_confidence=answer.overall_confidence or 0.0,
        claims=claims_out,
        status="completed",
        question_type=q_type,
        question_relationship=q_rel,
    )


@app.get(
    "/api/questions",
    response_model=list[QuestionSummaryOut],
    tags=["questions"],
)
def list_questions(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[QuestionSummaryOut]:
    """Return a paginated list of questions with their overall verdict."""
    questions = (
        db.query(models.Question)
        .order_by(models.Question.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    result = []
    for q in questions:
        overall_verdict: str | None = None
        if q.answers:
            overall_verdict = q.answers[-1].overall_verdict
        result.append(
            QuestionSummaryOut(
                question_id=q.id,
                question=q.question_text,
                created_at=q.created_at,
                overall_verdict=overall_verdict,
            )
        )
    return result


@app.get(
    "/api/questions/{question_id}",
    response_model=QuestionDetailOut,
    tags=["questions"],
)
def get_question(
    question_id: int,
    db: Session = Depends(get_db),
) -> QuestionDetailOut:
    """Return full detail for a single question including answer and aggregate stats."""
    question = _get_question_or_404(question_id, db)
    answer = question.answers[0] if question.answers else None

    num_claims = 0
    num_verified = 0
    num_partially_supported = 0
    num_unsupported = 0
    num_contradicted = 0

    if answer:
        for claim in answer.claims:
            num_claims += 1
            vr = claim.verification_result
            if vr:
                v = vr.verdict
                if v == "VERIFIED":
                    num_verified += 1
                elif v == "PARTIALLY_SUPPORTED":
                    num_partially_supported += 1
                elif v == "UNSUPPORTED":
                    num_unsupported += 1
                elif v == "CONTRADICTED":
                    num_contradicted += 1

    return QuestionDetailOut(
        question_id=question.id,
        question=question.question_text,
        answer=answer.raw_answer if answer else "",
        corrected_answer=answer.corrected_answer if answer else None,
        overall_verdict=answer.overall_verdict if answer else None,
        overall_confidence=answer.overall_confidence if answer else None,
        created_at=question.created_at,
        num_claims=num_claims,
        num_verified=num_verified,
        num_partially_supported=num_partially_supported,
        num_unsupported=num_unsupported,
        num_contradicted=num_contradicted,
    )


@app.get(
    "/api/questions/{question_id}/claims",
    response_model=list[ClaimOut],
    tags=["questions"],
)
def get_claims(
    question_id: int,
    db: Session = Depends(get_db),
) -> list[ClaimOut]:
    """Return all claims and their verification results for a question."""
    question = _get_question_or_404(question_id, db)
    answer = question.answers[0] if question.answers else None
    if answer is None:
        return []

    return _build_claims_out(answer)


@app.get(
    "/api/questions/{question_id}/report",
    response_model=ReportOut,
    tags=["questions"],
)
def get_report(
    question_id: int,
    db: Session = Depends(get_db),
) -> ReportOut:
    """Return the full verification report for a question."""
    question = _get_question_or_404(question_id, db)
    report = (
        db.query(models.VerificationReport)
        .filter(models.VerificationReport.question_id == question_id)
        .order_by(models.VerificationReport.created_at.desc())
        .first()
    )
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No report found for question {question_id}.",
        )
    try:
        report_json = json.loads(report.report_json)
    except json.JSONDecodeError:
        report_json = {}

    return ReportOut(
        question_id=question_id,
        report_json=report_json,
        report_text=report.report_text,
        overall_verdict=report.overall_verdict,
    )
