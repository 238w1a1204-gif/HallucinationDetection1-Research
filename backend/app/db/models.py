"""SQLAlchemy ORM models mirroring the schema in docs/architecture.md."""
from __future__ import annotations

import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    answers: Mapped[list["Answer"]] = relationship(back_populates="question", cascade="all, delete-orphan")
    reports: Mapped[list["VerificationReport"]] = relationship(back_populates="question", cascade="all, delete-orphan")


class Answer(Base):
    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), nullable=False)
    raw_answer: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    overall_verdict: Mapped[str | None] = mapped_column(String(50), nullable=True)
    overall_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    question: Mapped["Question"] = relationship(back_populates="answers")
    claims: Mapped[list["Claim"]] = relationship(back_populates="answer", cascade="all, delete-orphan")
    confidence_scores: Mapped[list["ConfidenceScore"]] = relationship(back_populates="answer", cascade="all, delete-orphan")


class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id"), nullable=False)
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    answer: Mapped["Answer"] = relationship(back_populates="claims")
    evidence: Mapped[list["Evidence"]] = relationship(back_populates="claim", cascade="all, delete-orphan")
    verification_result: Mapped["VerificationResult | None"] = relationship(
        back_populates="claim", cascade="all, delete-orphan", uselist=False
    )
    confidence_scores: Mapped[list["ConfidenceScore"]] = relationship(back_populates="claim", cascade="all, delete-orphan")


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), nullable=False)
    source_name: Mapped[str] = mapped_column(String(50), nullable=False)  # wikipedia | wikidata | duckduckgo
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieved_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    claim: Mapped["Claim"] = relationship(back_populates="evidence")


class VerificationResult(Base):
    __tablename__ = "verification_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id"), nullable=False, unique=True)
    verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    max_similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_sources_agreeing: Mapped[int] = mapped_column(Integer, default=0)
    contradiction_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    entity_mismatch: Mapped[bool] = mapped_column(Boolean, default=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    claim: Mapped["Claim"] = relationship(back_populates="verification_result")


class ConfidenceScore(Base):
    __tablename__ = "confidence_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id"), nullable=False)
    claim_id: Mapped[int | None] = mapped_column(ForeignKey("claims.id"), nullable=True)  # NULL = answer-level
    score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    answer: Mapped["Answer"] = relationship(back_populates="confidence_scores")
    claim: Mapped["Claim | None"] = relationship(back_populates="confidence_scores")


class VerificationReport(Base):
    __tablename__ = "verification_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), nullable=False)
    report_json: Mapped[str] = mapped_column(Text, nullable=False)
    report_text: Mapped[str] = mapped_column(Text, nullable=False)
    overall_verdict: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=datetime.datetime.utcnow)

    question: Mapped["Question"] = relationship(back_populates="reports")
