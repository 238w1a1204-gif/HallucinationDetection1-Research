"""Builds the final structured (JSON) and human-readable (text) report
for a question, combining every pipeline stage's output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ClaimReportEntry:
    claim_text: str
    verdict: str
    confidence: float
    explanation: str
    evidence: list[dict]


def build_report(
    question: str,
    raw_answer: str,
    corrected_answer: str,
    overall_verdict: str,
    overall_confidence: float,
    claim_entries: list[ClaimReportEntry],
    summary_counts: dict,
) -> tuple[dict, str]:
    report_dict = {
        "question": question,
        "original_answer": raw_answer,
        "corrected_answer": corrected_answer,
        "overall_verdict": overall_verdict,
        "overall_confidence": overall_confidence,
        "summary": summary_counts,
        "claims": [
            {
                "claim_text": c.claim_text,
                "verdict": c.verdict,
                "confidence": c.confidence,
                "explanation": c.explanation,
                "evidence": c.evidence,
            }
            for c in claim_entries
        ],
        "disclaimer": (
            "Confidence scores are heuristic (weighted-sum) indicators of evidence "
            "support, not calibrated probabilities of truth. Multi-source agreement "
            "is supporting evidence, not proof of correctness."
        ),
    }

    lines = [
        "HALLUCINATION DETECTION & ANSWER VERIFICATION REPORT",
        "=" * 55,
        f"Question: {question}",
        "",
        "Original Answer:",
        raw_answer,
        "",
        f"Overall Verdict: {overall_verdict}",
        f"Overall Confidence: {overall_confidence:.2f}",
        (
            f"Claims: {summary_counts.get('num_claims', 0)} total | "
            f"{summary_counts.get('num_verified', 0)} verified | "
            f"{summary_counts.get('num_partially_supported', 0)} partially supported | "
            f"{summary_counts.get('num_unsupported', 0)} unsupported | "
            f"{summary_counts.get('num_contradicted', 0)} contradicted"
        ),
        "",
        "Claim-by-claim breakdown:",
        "-" * 55,
    ]
    for idx, c in enumerate(claim_entries, start=1):
        lines.append(f"{idx}. \"{c.claim_text}\"")
        lines.append(f"   Verdict: {c.verdict} | Confidence: {c.confidence:.2f}")
        lines.append(f"   Explanation: {c.explanation}")
        for ev in c.evidence:
            lines.append(
                f"   - [{ev.get('source_name')}] similarity={ev.get('similarity_score'):.2f} "
                f"url={ev.get('source_url')}"
            )
        lines.append("")

    lines.append("Corrected Answer:")
    lines.append(corrected_answer)
    lines.append("")
    lines.append(
        "Disclaimer: Confidence scores are heuristic indicators, not calibrated "
        "probabilities. Multi-source agreement is supporting evidence, not proof."
    )

    report_text = "\n".join(lines)
    return report_dict, report_text


def report_to_json_string(report_dict: dict) -> str:
    return json.dumps(report_dict, indent=2, default=str)
