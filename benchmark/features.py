"""Per-claim feature extraction on FROZEN evidence (no live retrieval).

Computes every intermediate signal once (similarity, entity check, relationship,
NLI label, answer-value status) so ablations / threshold tuning / confidence
analysis can be re-run instantly. NLI results are cached on disk because
bart-large-mnli is the slow step.

Run scripts from backend/ so that `import app...` works.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())  # backend/ must be the working directory


def read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def evidence_for(row):
    out = []
    for n in (1, 2):
        text = (row.get(f"evidence_{n}_text") or "").strip()
        if text:
            out.append((text, row.get(f"evidence_{n}_source") or f"source_{n}",
                        (row.get(f"evidence_{n}_url") or "").strip() or None))
    return out


def make_offline():
    """analyze_question() otherwise calls Wikipedia for spelling resolution,
    which makes a 'frozen' benchmark non-deterministic. Disable it."""
    import app.verification.question_analyzer as qmod
    qmod._wikidata_resolve = lambda candidate: None


class NLICache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def _key(claim, evidence):
        return hashlib.sha1(f"{claim}\x1f{evidence}".encode("utf-8")).hexdigest()

    def label(self, claim, evidence):
        k = self._key(claim, evidence)
        if k not in self.data:
            from app.verification.contradiction_detector import detect_contradiction
            _, lab = detect_contradiction(claim, evidence)
            if lab in ("unavailable", "error"):
                raise RuntimeError(f"NLI model unavailable ({lab}); check transformers/torch install.")
            self.data[k] = lab
        return self.data[k]
    def scores(self, claim, evidence):
        """Full NLI probability dict {entailment, neutral, contradiction} (cached)."""
        k = self._key(claim, evidence) + "|p"
        if k not in self.data:
            from app.verification.contradiction_detector import _get_nli_pipeline
            pipe = _get_nli_pipeline()
            if pipe is None:
                raise RuntimeError("NLI model unavailable")
            res = pipe({"text": evidence, "text_pair": claim})
            if isinstance(res, list) and res and isinstance(res[0], list):
                res = res[0]
            self.data[k] = {r["label"].lower(): float(r["score"]) for r in res}
        return self.data[k]
    def save(self):
        self.path.write_text(json.dumps(self.data))


def extract_features(row, nli: NLICache) -> dict:
    from unittest.mock import patch

    from app.verification.agreement_checker import count_agreeing_sources
    from app.verification.answer_value_checker import check_answer_value
    from app.verification.confidence_calculator import calculate_confidence
    from app.verification.entity_matcher import check_entity_consistency
    from app.verification.evidence_quality_scorer import rank_evidence_by_quality
    from app.verification.question_analyzer import analyze_question
    from app.verification.relationship_checker import check_relationship
    from app.verification.similarity_scorer import score_evidence_against_claim
    from app.verification.verdict_engine import determine_verdict

    claim = row["claim"].strip()
    question = (row.get("question") or "").strip()
    qa = analyze_question(question)
    subject = getattr(qa, "focus_entity", "") or ""
    scored = score_evidence_against_claim(claim, evidence_for(row))

    f = {"id": row.get("id"), "category": row.get("category", ""),
         "gold": row.get("gold_verdict"), "claim": claim, "n_evidence": len(scored),
         "max_sim": 0.0, "agree": 0, "ent_mismatch": False, "ent_presence": 0.0,
         "rel_ok": False, "nli_primary": "none","nli_all": [], "nli_probs_all": [], "av" : "VALUE_UNVERIFIABLE",
         "primary_quality": 0.0}

    if scored:
        ranked = rank_evidence_by_quality(claim, scored, subject_entity=subject)
        primary = ranked[0]
        ent = check_entity_consistency(claim, primary.evidence_text)
        rel = check_relationship(claim, primary.evidence_text)
        # same logic as verdict_engine Stage 2 (answer_text == claim in the benchmark)
        av = check_answer_value(qa, claim, primary.evidence_text, subject_entity=subject).status
        if av == "VALUE_CONTRADICTED":
            for alt in ranked[1:]:
                if check_answer_value(qa, claim, alt.evidence_text,
                                      subject_entity=subject).status == "VALUE_SUPPORTED":
                    av = "VALUE_SUPPORTED"
                    break
        f.update({
            "max_sim": float(scored[0].similarity),
            "agree": int(count_agreeing_sources(scored, claim=claim)),
            "ent_mismatch": bool(ent.mismatch),
            "ent_presence": float(ent.entity_presence_score),
            "rel_ok": bool(rel.relationship_discussed),
            "nli_primary": nli.label(claim, primary.evidence_text),
            "nli_all": [nli.label(claim, s.evidence_text) for s in scored],
            "nli_probs_all": [nli.scores(claim, s.evidence_text) for s in scored],
            "av": av,
            "primary_quality": float(primary.quality_score),
        })

    def _cached_detect(c, e):
        lab = nli.label(c, e)
        return lab == "contradiction", lab

    with patch("app.verification.verdict_engine.detect_contradiction", side_effect=_cached_detect):
        v = determine_verdict(claim, scored, question_analysis=qa,
                              answer_text=claim, subject_entity=subject)
    conf = calculate_confidence(
        similarity=v.max_similarity, agree_count=v.num_sources_agreeing,
        contradiction_detected=v.contradiction_detected, entity_mismatch=v.entity_mismatch,
        entity_presence_score=v.entity_presence_score)
    f.update({"engine_verdict": v.verdict, "engine_conf": float(conf),
              "explanation": v.explanation})
    return f


def load_features(path):
    data = json.loads(Path(path).read_text())
    return [f for f in data if not f.get("error")]
