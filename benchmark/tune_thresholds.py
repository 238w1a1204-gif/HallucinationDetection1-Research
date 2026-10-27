#!/usr/bin/env python3
"""Tune similarity thresholds on DEV only, then evaluate ONCE on TEST.
Also trains a scikit-learn logistic-regression 'learned combiner' on DEV as a baseline.

From backend/:
  python ../benchmark/tune_thresholds.py \
    --dev-features ../benchmark/results/vc_dev_features.json \
    --test-features ../benchmark/results/vc_test_features.json --min-sources 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_utils import bootstrap_ci, compute_metrics  # noqa: E402
from features import load_features  # noqa: E402
from variants import FULL, decide  # noqa: E402


def to_matrix(fs):
    return np.array([[
        f["max_sim"], f["agree"], f["ent_presence"], float(f["ent_mismatch"]),
        float(f["rel_ok"]), float(f["nli_primary"] == "contradiction"),
        float(f["nli_primary"] == "entailment"), float(f["av"] == "VALUE_CONTRADICTED"),
        float(f["av"] == "VALUE_SUPPORTED"), f["primary_quality"]] for f in fs])


def report(tag, gold, pred):
    m = compute_metrics(gold, pred)
    ci = bootstrap_ci(gold, pred, "accuracy")
    print(f"{tag:34s} acc={m['accuracy']:.3f} {ci}  macroF1={m['macro_f1_populated']:.3f}  "
          f"falseVER={m['false_verified_rate']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-features", required=True)
    ap.add_argument("--test-features", required=True)
    ap.add_argument("--min-sources", type=int, default=2)
    a = ap.parse_args()
    dev, test = load_features(a.dev_features), load_features(a.test_features)
    gd, gt = [f["gold"] for f in dev], [f["gold"] for f in test]

    best = (-1, None, None)
    grid = []
    for partial in np.arange(0.35, 0.701, 0.025):
        for strong in np.arange(partial + 0.05, 0.951, 0.025):
            p = [decide(f, FULL, strong, partial, a.min_sources) for f in dev]
            s = compute_metrics(gd, p)["macro_f1_populated"]
            grid.append((s, round(strong, 3), round(partial, 3)))
            if s > best[0]:
                best = (s, strong, partial)
    grid.sort(reverse=True)
    print("Top-5 dev settings (macroF1, strong, partial):", grid[:5])
    _, bs, bp = best
    print(f"\nChosen on DEV: strong={bs:.3f} partial={bp:.3f}\n--- TEST (single evaluation) ---")
    report("Default thresholds (0.75/0.55)", gt, [decide(f, FULL, 0.75, 0.55, a.min_sources) for f in test])
    report("Tuned thresholds", gt, [decide(f, FULL, bs, bp, a.min_sources) for f in test])

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(to_matrix(dev), gd)
    report("Learned combiner (LogReg, dev-fit)", gt, list(clf.predict(to_matrix(test))))


if __name__ == "__main__":
    main()
