#!/usr/bin/env python3
"""NLI-probability verdicts with abstention thresholds (tuned on DEV only) and a
comparison of NLI-probability confidence vs the heuristic weighted-sum confidence.

From backend/ (features must have been re-extracted so they contain nli_probs_all):
  python ../benchmark/nli_threshold.py --dev-features ../benchmark/results/vc_dev_deberta_features.json \
      [--max-false-ver 0.05] [--test-features ...]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import bootstrap_ci, compute_metrics  # noqa: E402
from features import load_features  # noqa: E402
from variants import predict  # noqa: E402

V, U, C = "VERIFIED", "UNSUPPORTED", "CONTRADICTED"


def pe_pc(f):
    pr = f.get("nli_probs_all") or []
    if not pr:
        return 0.0, 0.0
    return max(d.get("entailment", 0.0) for d in pr), max(d.get("contradiction", 0.0) for d in pr)


def decide_prob(f, te, tc):
    if f["n_evidence"] == 0:
        return U
    pe, pc = pe_pc(f)
    if pc >= tc and pc > pe:
        return C
    if pe >= te and pe > pc:
        return V
    return U


def ece(correct, conf, bins=10):
    correct, conf = np.asarray(correct, float), np.asarray(conf, float)
    edges, tot = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def report(tag, gold, pred):
    m = compute_metrics(gold, pred)
    print(f"{tag:34s} acc={m['accuracy']:.3f} {bootstrap_ci(gold, pred, 'accuracy')}  "
          f"macroF1={m['macro_f1_populated']:.3f}  falseVER={m['false_verified_rate']}")


def conf_compare(tag, fs):
    gold = [f["gold"] for f in fs]
    # heuristic engine confidence vs engine correctness
    c1 = np.array([f["engine_verdict"] == f["gold"] for f in fs])
    s1 = np.array([f["engine_conf"] for f in fs])
    # NLI-probability confidence vs NLI-only correctness
    c2 = np.array([predict(f, "nli_only") == f["gold"] for f in fs])
    s2 = np.array([max(pe_pc(f)[0], pe_pc(f)[1], max((d.get("neutral", 0) for d in f["nli_probs_all"]), default=0))
                   for f in fs])
    print(f"\n[{tag}] confidence quality (AUROC of confidence for 'verdict is correct'; ECE)")
    for name, c, s in (("heuristic weighted-sum / engine", c1, s1), ("max NLI probability / NLI-only", c2, s2)):
        au = roc_auc_score(c, s) if 0 < c.sum() < len(c) else float("nan")
        order = np.argsort(-s)
        sel = c[order][: int(0.8 * len(c))].mean()
        print(f"  {name:34s} AUROC={au:.3f}  ECE={ece(c, s):.3f}  acc@80%coverage={sel:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-features", required=True)
    ap.add_argument("--test-features", default=None)
    ap.add_argument("--max-false-ver", type=float, default=None)
    a = ap.parse_args()
    dev = [f for f in load_features(a.dev_features) if f.get("nli_probs_all") is not None]
    gd = [f["gold"] for f in dev]

    grid = []
    for te in np.arange(0.30, 0.991, 0.05):
        for tc in np.arange(0.30, 0.991, 0.05):
            p = [decide_prob(f, te, tc) for f in dev]
            m = compute_metrics(gd, p)
            fv = m["false_verified_rate"] or 0.0
            if a.max_false_ver is not None and fv > a.max_false_ver:
                continue
            grid.append((m["macro_f1_populated"], round(float(te), 2), round(float(tc), 2), fv))
    grid.sort(reverse=True)
    print("Top-5 DEV settings (macroF1, tau_ent, tau_con, falseVER):")
    for g in grid[:5]:
        print("  ", g)
    _, te, tc, _ = grid[0]
    print(f"\nChosen on DEV: tau_ent={te} tau_con={tc}")
    report("DEV  NLI argmax (nli_only)", gd, [predict(f, "nli_only") for f in dev])
    report("DEV  NLI prob + thresholds", gd, [decide_prob(f, te, tc) for f in dev])
    conf_compare("DEV", dev)

    if a.test_features:
        test = [f for f in load_features(a.test_features) if f.get("nli_probs_all") is not None]
        gt = [f["gold"] for f in test]
        print("\n--- TEST (single evaluation) ---")
        report("TEST NLI argmax (nli_only)", gt, [predict(f, "nli_only") for f in test])
        report("TEST NLI prob + thresholds", gt, [decide_prob(f, te, tc) for f in test])
        conf_compare("TEST", test)


if __name__ == "__main__":
    main()
