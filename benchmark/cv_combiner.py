#!/usr/bin/env python3
"""Cross-validated learned combiner on DEV only (GroupKFold by claim).

From backend/:
  python ../benchmark/cv_combiner.py --dev-features ../benchmark/results/vc_dev_features.json --min-sources 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import bootstrap_ci, compute_metrics  # noqa: E402
from features import load_features  # noqa: E402
from variants import FULL, decide, predict  # noqa: E402


def matrix(fs):
    rows = []
    for f in fs:
        labs = f["nli_all"]
        rows.append([
            f["max_sim"], f["agree"], f["ent_presence"], float(f["ent_mismatch"]),
            float(f["rel_ok"]), f["primary_quality"],
            float(f["nli_primary"] == "entailment"), float(f["nli_primary"] == "contradiction"),
            float(f["nli_primary"] == "neutral"), labs.count("entailment"), labs.count("contradiction"),
            float(f["av"] == "VALUE_CONTRADICTED"), float(f["av"] == "VALUE_SUPPORTED")])
    return np.array(rows)


def report(tag, gold, pred):
    m = compute_metrics(gold, pred)
    print(f"{tag:30s} acc={m['accuracy']:.3f} {bootstrap_ci(gold, pred, 'accuracy')}  "
          f"macroF1={m['macro_f1_populated']:.3f}  falseVER={m['false_verified_rate']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-features", required=True)
    ap.add_argument("--min-sources", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    fs = load_features(a.dev_features)
    y = np.array([f["gold"] for f in fs])
    ids = {}
    groups = np.array([ids.setdefault(f["claim"], len(ids)) for f in fs])
    X = matrix(fs)
    print(f"{len(fs)} claims, {len(ids)} distinct claims (groups)\n")

    report("Rule engine (Full)", list(y), [decide(f, FULL, min_sources=a.min_sources) for f in fs])
    report("NLI only", list(y), [predict(f, "nli_only") for f in fs])
    report("NLI-gated +Entity", list(y), [predict(f, "gated_entity") for f in fs])

    models = {
        "LogReg (balanced)": make_pipeline(StandardScaler(),
                                           LogisticRegression(max_iter=2000, class_weight="balanced")),
        "GradBoost (depth 3)": HistGradientBoostingClassifier(max_depth=3, max_iter=100,
                                                              random_state=a.seed),
    }
    cv = GroupKFold(n_splits=a.folds)
    for name, clf in models.items():
        pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
        report(f"CV {name}", list(y), list(pred))


if __name__ == "__main__":
    main()
