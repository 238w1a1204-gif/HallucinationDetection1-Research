#!/usr/bin/env python3
"""DEV-only diagnostic: can a numeric-conflict rule fix NLI misses (esp. CONTRADICTED -> UNSUPPORTED)?

From backend/:
  python ../benchmark/numeric_conflict.py \
    --dev-features ../benchmark/results/vc_dev_deberta_features.json \
    --csv ../benchmark/data/vitaminc_dev.csv
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import bootstrap_ci, compute_metrics  # noqa: E402
from features import load_features, read_rows  # noqa: E402
from variants import predict  # noqa: E402

NUM = re.compile(r"\d+(?:[.,]\d+)*")


def nums(t):
    return {n.replace(",", "") for n in NUM.findall(t or "")}


def flags(claim, evidence):
    c, e = nums(claim), nums(evidence)
    return {"has_num": bool(c), "missing": bool(c and e and (c - e)),
            "disjoint": bool(c and e and not (c & e))}


def p_ent(f):
    pr = f.get("nli_probs_all") or []
    return max((d.get("entailment", 0.0) for d in pr), default=1.0)


def report(tag, gold, pred):
    m = compute_metrics(gold, pred)
    print(f"{tag:46s} acc={m['accuracy']:.3f} {bootstrap_ci(gold, pred, 'accuracy')} "
          f"falseVER={m['false_verified_rate']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-features", required=True)
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    fs = load_features(a.dev_features)
    ev = {r["id"]: r["evidence_1_text"] for r in read_rows(a.csv)}
    fl = [flags(f["claim"], ev.get(f["id"], "")) for f in fs]
    gold = [f["gold"] for f in fs]
    base = [predict(f, "nli_only") for f in fs]

    print("Share of claims with each numeric pattern, by GOLD class:")
    print(f"{'gold':14s}{'n':>5s}{'has_num':>9s}{'missing':>9s}{'disjoint':>10s}")
    for g in ("VERIFIED", "UNSUPPORTED", "CONTRADICTED"):
        idx = [i for i, x in enumerate(gold) if x == g]
        n = len(idx) or 1
        print(f"{g:14s}{len(idx):5d}" + "".join(f"{100*sum(fl[i][k] for i in idx)/n:8.1f}%" for k in ("has_num", "missing", "disjoint")))

    miss = [i for i in range(len(fs)) if gold[i] == "CONTRADICTED" and base[i] == "UNSUPPORTED"]
    print(f"\nGold CONTRADICTED but NLI says UNSUPPORTED: {len(miss)} claims; "
          f"{sum(fl[i]['disjoint'] for i in miss)} have disjoint numbers, "
          f"{sum(fl[i]['missing'] for i in miss)} have a claim number missing from evidence")
    wrong = defaultdict(int)
    for i in range(len(fs)):
        if base[i] == "UNSUPPORTED" and fl[i]["disjoint"]:
            wrong[gold[i]] += 1
    print("Gold class of claims NLI calls UNSUPPORTED that also have disjoint numbers:", dict(wrong))

    print()
    report("Baseline: NLI argmax", gold, base)
    rules = {
        "Rule A: NLI=UNS & disjoint -> CONTRADICTED": lambda i: "CONTRADICTED" if base[i] == "UNSUPPORTED" and fl[i]["disjoint"] else base[i],
        "Rule B: NLI=UNS & missing  -> CONTRADICTED": lambda i: "CONTRADICTED" if base[i] == "UNSUPPORTED" and fl[i]["missing"] else base[i],
        "Rule C: disjoint & P(ent)<0.9 -> CONTRADICTED": lambda i: "CONTRADICTED" if base[i] in ("UNSUPPORTED", "VERIFIED") and fl[i]["disjoint"] and p_ent(fs[i]) < 0.9 else base[i],
    }
    for name, fn in rules.items():
        report(name, gold, [fn(i) for i in range(len(fs))])


if __name__ == "__main__":
    main()
