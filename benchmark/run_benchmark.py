#!/usr/bin/env python3
"""Evaluate the existing claim verifier on fixed evidence packets.

Run from backend/:
  python ../benchmark/run_benchmark.py --input ../benchmark/claims.csv --output ../benchmark/results.csv

This bypasses LLM answer generation and live retrieval so each code revision sees the same evidence.
"""
from __future__ import annotations
import argparse, csv, json, platform, sys
from datetime import datetime, timezone
from pathlib import Path
from app.config import get_settings
from app.verification.question_analyzer import analyze_question
from app.verification.similarity_scorer import score_evidence_against_claim
from app.verification.verdict_engine import determine_verdict
from app.verification.confidence_calculator import calculate_confidence

LABELS = ["VERIFIED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]

def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def evidence_for(row):
    out = []
    for n in (1, 2):
        text = (row.get(f"evidence_{n}_text") or "").strip()
        if text:
            out.append((text, row.get(f"evidence_{n}_source") or f"source_{n}",
                        (row.get(f"evidence_{n}_url") or "").strip() or None))
    return out

def compute_metrics(rows):
    valid = [r for r in rows if r.get("predicted_verdict") in LABELS and r.get("gold_verdict") in LABELS]
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for r in valid:
        matrix[r["gold_verdict"]][r["predicted_verdict"]] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in LABELS if g != label)
        fn = sum(matrix[label][p] for p in LABELS if p != label)
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class[label] = {"precision": round(precision,4), "recall": round(recall,4),
                            "f1": round(f1,4), "support": sum(matrix[label].values())}
    unsupported = [r for r in valid if r["gold_verdict"] == "UNSUPPORTED"]
    false_verified = sum(r["predicted_verdict"] == "VERIFIED" for r in unsupported)
    return {
        "evaluated": len(valid),
        "accuracy": round(sum(matrix[l][l] for l in LABELS)/len(valid),4) if valid else 0.0,
        "macro_f1": round(sum(x["f1"] for x in per_class.values())/len(LABELS),4),
        "per_class": per_class,
        "confusion_matrix_rows_gold_columns_predicted": matrix,
        "gold_unsupported_predicted_verified_count": false_verified,
        "gold_unsupported_predicted_verified_rate": round(false_verified/len(unsupported),4) if unsupported else None,
        "note": "UNSUPPORTED means the supplied evidence is insufficient; it does not necessarily mean the claim is false."
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../benchmark/claims.csv")
    ap.add_argument("--output", default="../benchmark/results.csv")
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp.resolve()}")
    rows = read_rows(inp)
    results = []
    for i, row in enumerate(rows, 1):
        result = dict(row)
        for k in ("predicted_verdict","predicted_confidence","max_similarity","agreeing_sources",
                  "contradiction_detected","entity_mismatch","explanation","evidence_used_json","error"):
            result[k] = ""
        try:
            claim, question = row["claim"].strip(), row["question"].strip()
            qa = analyze_question(question)
            scored = score_evidence_against_claim(claim, evidence_for(row))
            verdict = determine_verdict(
                claim, scored, question_analysis=qa, answer_text=claim,
                subject_entity=getattr(qa, "focus_entity", "") if qa else ""
            )
            conf = calculate_confidence(
                similarity=verdict.max_similarity, agree_count=verdict.num_sources_agreeing,
                contradiction_detected=verdict.contradiction_detected,
                entity_mismatch=verdict.entity_mismatch,
                entity_presence_score=verdict.entity_presence_score
            )
            result.update({
                "predicted_verdict": verdict.verdict,
                "predicted_confidence": round(float(conf),6),
                "max_similarity": round(float(verdict.max_similarity),6),
                "agreeing_sources": int(verdict.num_sources_agreeing),
                "contradiction_detected": bool(verdict.contradiction_detected),
                "entity_mismatch": bool(verdict.entity_mismatch),
                "explanation": verdict.explanation,
                "evidence_used_json": json.dumps([{
                    "source_name": x.source_name, "source_url": x.source_url,
                    "similarity": round(float(x.similarity),6), "evidence_text": x.evidence_text
                } for x in scored], ensure_ascii=False)
            })
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        results.append(result)
        print(f"[{i:02d}/{len(rows)}] {row.get('id')} -> {result['predicted_verdict'] or 'ERROR'}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)
    successful = [r for r in results if not r["error"]]
    settings = get_settings()
    meta = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(inp.resolve()), "output_file": str(out.resolve()),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "configured_llm_provider": getattr(settings, "LLM_PROVIDER", None),
        "embedding_model_config": {k:getattr(settings,k,None) for k in
            ("EMBEDDING_MODEL","SENTENCE_TRANSFORMER_MODEL","SIMILARITY_MODEL") if getattr(settings,k,None)},
        "nli_model_config": {k:getattr(settings,k,None) for k in
            ("NLI_MODEL","CONTRADICTION_MODEL") if getattr(settings,k,None)},
        "dataset_rows": len(rows), "successful_rows": len(successful),
        "failed_rows": len(rows)-len(successful), "metrics": compute_metrics(successful),
        "method_note": "Fixed evidence packets; no LLM generation or live retrieval. Preserve output CSV and metrics JSON for every run."
    }
    mp = out.with_name(out.stem + "_metrics.json")
    mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults: {out.resolve()}\nMetrics: {mp.resolve()}")
    print(f"Accuracy={meta['metrics']['accuracy']:.3f}; macro-F1={meta['metrics']['macro_f1']:.3f}")
    print(f"Successful rows={len(successful)}/{len(rows)}")
if __name__ == "__main__":
    main()
