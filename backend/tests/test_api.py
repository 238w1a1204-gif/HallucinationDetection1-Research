"""Tests for the FastAPI routes.

The full pipeline (LLM + embedding models + network calls) is mocked so
that API tests run in milliseconds without any model downloads or network
access.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _mock_pipeline(db, question_text):
    """
    Minimal stand-in for run_pipeline.
    Persists a Question, Answer, Claim, VerificationResult, ConfidenceScore
    and VerificationReport using the real ORM models so that all GET
    endpoints can query them normally.
    """
    import datetime
    from app.db import models
    import json

    question = models.Question(question_text=question_text)
    db.add(question)
    db.flush()

    answer = models.Answer(
        question_id=question.id,
        raw_answer="Mock answer about the topic.",
        corrected_answer="Mock corrected answer.",
        llm_provider="MockProvider",
        overall_verdict="NO_HALLUCINATION_DETECTED",
        overall_confidence=0.85,
    )
    db.add(answer)
    db.flush()

    claim = models.Claim(
        answer_id=answer.id,
        claim_text="Mock claim text",
        claim_index=0,
    )
    db.add(claim)
    db.flush()

    ev = models.Evidence(
        claim_id=claim.id,
        source_name="wikipedia",
        source_url="https://en.wikipedia.org/wiki/Mock",
        evidence_text="Mock evidence retrieved from Wikipedia.",
        similarity_score=0.82,
    )
    db.add(ev)

    vr = models.VerificationResult(
        claim_id=claim.id,
        verdict="VERIFIED",
        max_similarity_score=0.82,
        num_sources_agreeing=2,
        contradiction_detected=False,
        entity_mismatch=False,
        explanation="Mock explanation: strong similarity with multi-source agreement.",
    )
    db.add(vr)

    cs_claim = models.ConfidenceScore(answer_id=answer.id, claim_id=claim.id, score=0.85)
    cs_answer = models.ConfidenceScore(answer_id=answer.id, claim_id=None, score=0.85)
    db.add(cs_claim)
    db.add(cs_answer)

    report_dict = {
        "question": question_text,
        "original_answer": "Mock answer.",
        "overall_verdict": "NO_HALLUCINATION_DETECTED",
    }
    report = models.VerificationReport(
        question_id=question.id,
        report_json=json.dumps(report_dict),
        report_text="MOCK REPORT TEXT",
        overall_verdict="NO_HALLUCINATION_DETECTED",
    )
    db.add(report)

    db.commit()
    db.refresh(question)
    return question


# ── Health endpoint ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_health_includes_llm_provider(self, client):
        data = client.get("/api/health").json()
        assert "llm_provider" in data
        assert isinstance(data["llm_provider"], str)


# ── POST /api/questions ───────────────────────────────────────────────────────


class TestCreateQuestion:
    @pytest.fixture(autouse=True)
    def _mock_pipeline(self):
        with patch(
            "app.main.run_pipeline",
            side_effect=_mock_pipeline,
        ):
            yield

    def test_create_question_returns_201(self, client):
        resp = client.post("/api/questions", json={"question": "What is gravity?"})
        assert resp.status_code == 201

    def test_response_contains_required_fields(self, client):
        data = client.post(
            "/api/questions", json={"question": "What is gravity?"}
        ).json()
        for field in (
            "question_id",
            "question",
            "answer",
            "corrected_answer",
            "overall_verdict",
            "overall_confidence",
            "claims",
            "status",
        ):
            assert field in data, f"Missing field: {field}"

    def test_question_text_echoed_in_response(self, client):
        q = "What is photosynthesis?"
        data = client.post("/api/questions", json={"question": q}).json()
        assert data["question"] == q

    def test_claims_is_a_list(self, client):
        data = client.post(
            "/api/questions", json={"question": "Tell me about DNA."}
        ).json()
        assert isinstance(data["claims"], list)

    def test_empty_question_returns_422(self, client):
        resp = client.post("/api/questions", json={"question": "   "})
        assert resp.status_code == 422

    def test_missing_question_field_returns_422(self, client):
        resp = client.post("/api/questions", json={})
        assert resp.status_code == 422


# ── GET /api/questions ────────────────────────────────────────────────────────


class TestListQuestions:
    @pytest.fixture(autouse=True)
    def _mock_pipeline(self):
        with patch("app.main.run_pipeline", side_effect=_mock_pipeline):
            yield

    def test_empty_db_returns_empty_list(self, client):
        data = client.get("/api/questions").json()
        assert data == []

    def test_returns_list_after_submission(self, client):
        client.post("/api/questions", json={"question": "Test question?"})
        data = client.get("/api/questions").json()
        assert len(data) == 1

    def test_list_item_has_question_id(self, client):
        client.post("/api/questions", json={"question": "Another question?"})
        items = client.get("/api/questions").json()
        assert "question_id" in items[0]
        assert "question" in items[0]
        assert "overall_verdict" in items[0]


# ── GET /api/questions/{id} ───────────────────────────────────────────────────


class TestGetQuestion:
    @pytest.fixture(autouse=True)
    def _mock_pipeline(self):
        with patch("app.main.run_pipeline", side_effect=_mock_pipeline):
            yield

    def test_not_found_returns_404(self, client):
        resp = client.get("/api/questions/999")
        assert resp.status_code == 404

    def test_returns_detail_for_existing_question(self, client):
        created = client.post(
            "/api/questions", json={"question": "What is quantum mechanics?"}
        ).json()
        qid = created["question_id"]
        detail = client.get(f"/api/questions/{qid}").json()
        assert detail["question_id"] == qid
        assert detail["question"] == "What is quantum mechanics?"

    def test_detail_includes_verdict_and_confidence(self, client):
        created = client.post(
            "/api/questions", json={"question": "Explain DNA."}
        ).json()
        detail = client.get(f"/api/questions/{created['question_id']}").json()
        assert "overall_verdict" in detail
        assert "overall_confidence" in detail


# ── GET /api/questions/{id}/claims ────────────────────────────────────────────


class TestGetClaims:
    @pytest.fixture(autouse=True)
    def _mock_pipeline(self):
        with patch("app.main.run_pipeline", side_effect=_mock_pipeline):
            yield

    def test_returns_claims_list(self, client):
        created = client.post(
            "/api/questions", json={"question": "What is DNA?"}
        ).json()
        claims = client.get(
            f"/api/questions/{created['question_id']}/claims"
        ).json()
        assert isinstance(claims, list)
        assert len(claims) == 1  # our mock creates exactly 1 claim

    def test_claim_has_verdict(self, client):
        created = client.post(
            "/api/questions", json={"question": "Who invented the telephone?"}
        ).json()
        claims = client.get(
            f"/api/questions/{created['question_id']}/claims"
        ).json()
        assert claims[0]["verification_result"]["verdict"] == "VERIFIED"

    def test_claim_has_evidence(self, client):
        created = client.post(
            "/api/questions", json={"question": "What is the speed of light?"}
        ).json()
        claims = client.get(
            f"/api/questions/{created['question_id']}/claims"
        ).json()
        assert len(claims[0]["evidence"]) == 1
        assert claims[0]["evidence"][0]["source_name"] == "wikipedia"


# ── GET /api/questions/{id}/report ────────────────────────────────────────────


class TestGetReport:
    @pytest.fixture(autouse=True)
    def _mock_pipeline(self):
        with patch("app.main.run_pipeline", side_effect=_mock_pipeline):
            yield

    def test_report_not_found_returns_404(self, client):
        resp = client.get("/api/questions/999/report")
        assert resp.status_code == 404

    def test_returns_report_for_existing_question(self, client):
        created = client.post(
            "/api/questions", json={"question": "What is evolution?"}
        ).json()
        report = client.get(
            f"/api/questions/{created['question_id']}/report"
        ).json()
        assert "report_text" in report
        assert "report_json" in report
        assert "overall_verdict" in report

    def test_report_json_is_a_dict(self, client):
        created = client.post(
            "/api/questions", json={"question": "What is entropy?"}
        ).json()
        report = client.get(
            f"/api/questions/{created['question_id']}/report"
        ).json()
        assert isinstance(report["report_json"], dict)
