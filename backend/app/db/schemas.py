"""Pydantic request/response schemas used by the API layer."""
from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict


# ---------- Requests ----------

class QuestionCreateRequest(BaseModel):
    question: str


# ---------- Evidence ----------

class EvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_name: str
    source_url: str | None
    evidence_text: str
    similarity_score: float | None


# ---------- Verification result ----------

class VerificationResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    verdict: str
    max_similarity_score: float | None
    num_sources_agreeing: int
    contradiction_detected: bool
    entity_mismatch: bool
    explanation: str | None


# ---------- Claim ----------

class ClaimOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    claim_text: str
    claim_index: int
    evidence: list[EvidenceOut] = []
    verification_result: VerificationResultOut | None = None
    confidence: float | None = None


# ---------- Question / answer summary ----------

class QuestionSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    question_id: int
    question: str
    created_at: datetime.datetime
    overall_verdict: str | None = None


class QuestionCreateResponse(BaseModel):
    question_id: int
    question: str
    answer: str
    corrected_answer: str | None
    overall_verdict: str
    overall_confidence: float
    claims: list[ClaimOut]
    status: str = "completed"
    # Question analysis fields (optional — None when analysis was unavailable)
    question_type: str | None = None
    question_relationship: str | None = None


class QuestionDetailOut(BaseModel):
    question_id: int
    question: str
    answer: str
    corrected_answer: str | None
    overall_verdict: str | None
    overall_confidence: float | None
    created_at: datetime.datetime
    num_claims: int
    num_verified: int
    num_partially_supported: int
    num_unsupported: int
    num_contradicted: int


class ReportOut(BaseModel):
    question_id: int
    report_json: dict
    report_text: str
    overall_verdict: str


class HealthOut(BaseModel):
    status: str
    llm_provider: str
