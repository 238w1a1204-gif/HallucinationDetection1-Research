#!/usr/bin/env python3
"""LLM-as-judge baseline (Gemini) on the same frozen evidence.

From backend/ (needs GEMINI_API_KEY and a valid GEMINI_MODEL in backend/.env):
  python ../benchmark/baseline_gemini_judge.py --input ../benchmark/data/vitaminc_test.csv --name vc_test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_utils import bootstrap_ci, compute_metrics  # noqa: E402
from features import evidence_for, read_rows  # noqa: E402

PROMPT = """You are a strict fact-verification judge. Use ONLY the evidence below.
Choose exactly one label:
VERIFIED = evidence clearly supports the claim
CONTRADICTED = evidence clearly conflicts with the claim
UNSUPPORTED = evidence is insufficient to decide
PARTIALLY_SUPPORTED = evidence supports only part of the claim

Claim: {claim}
Evidence:
{evidence}

Answer with the label only."""
LABEL_RE = re.compile(r"PARTIALLY_SUPPORTED|UNSUPPORTED|CONTRADICTED|VERIFIED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="../benchmark/results")
    ap.add_argument("--cache", default="../benchmark/cache/gemini_cache.json")
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()

    from google import genai
    from google.genai import types

    from app.config import get_settings
    st = get_settings()
    client = genai.Client(api_key=st.GEMINI_API_KEY)

    cache_p = Path(a.cache)
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    rows = read_rows(a.input)
    gold, pred = [], []
    for i, row in enumerate(rows, 1):
        ev = "\n".join(f"- {t}" for t, _, _ in evidence_for(row))
        prompt = PROMPT.format(claim=row["claim"], evidence=ev)
        key = f"{st.GEMINI_MODEL}|{prompt}"
        if key not in cache:
            for attempt in range(4):
                try:
                    r = client.models.generate_content(
                        model=st.GEMINI_MODEL, contents=prompt,
                        config=types.GenerateContentConfig(temperature=0))
                    cache[key] = (r.text or "").upper()
                    break
                except Exception as exc:
                    print("retry", attempt, exc)
                    time.sleep(2 ** attempt * 2)
            else:
                cache[key] = ""
            time.sleep(a.sleep)
        m = LABEL_RE.search(cache[key])
        gold.append(row["gold_verdict"])
        pred.append(m.group(0) if m else "UNSUPPORTED")
        if i % 20 == 0:
            cache_p.write_text(json.dumps(cache))
            print(f"{i}/{len(rows)}")
    cache_p.write_text(json.dumps(cache))

    m = compute_metrics(gold, pred)
    m["accuracy_ci95"] = bootstrap_ci(gold, pred, "accuracy")
    m["model"] = st.GEMINI_MODEL
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}_gemini_judge_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: m[k] for k in ("accuracy", "accuracy_ci95", "macro_f1_populated",
                                        "false_verified_rate")}, indent=2))


if __name__ == "__main__":
    main()
