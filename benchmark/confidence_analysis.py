#!/usr/bin/env python3
"""Turn the 'confidence is uncalibrated' disclaimer into a measured result.

From backend/:
  python ../benchmark/confidence_analysis.py \
    --test-features ../benchmark/results/vc_test_features.json \
    --dev-features  ../benchmark/results/vc_dev_features.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_features  # noqa: E402


def ece(correct, conf, bins=10):
    correct, conf = np.asarray(correct, float), np.asarray(conf, float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(tot)


def risk_coverage(correct, conf):
    order = np.argsort(-np.asarray(conf))
    c = np.asarray(correct, float)[order]
    cov = np.arange(1, len(c) + 1) / len(c)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    return cov, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-features", required=True)
    ap.add_argument("--dev-features", default=None)
    ap.add_argument("--out-dir", default="../benchmark/results")
    ap.add_argument("--name", default="conf")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    test = load_features(a.test_features)
    correct = np.array([f["engine_verdict"] == f["gold"] for f in test])
    conf = np.array([f["engine_conf"] for f in test])
    sim = np.array([f["max_sim"] for f in test])
    res = {"n": len(test), "verdict_accuracy": float(correct.mean())}
    if 0 < correct.sum() < len(correct):
        res["auroc_engine_conf"] = float(roc_auc_score(correct, conf))
        res["auroc_similarity_only"] = float(roc_auc_score(correct, sim))
    res["ece_raw"] = ece(correct, conf)

    cov, acc = risk_coverage(correct, conf)
    for c in (0.5, 0.8):
        res[f"selective_accuracy_at_{int(c*100)}pct_coverage"] = float(acc[int(c * len(acc)) - 1])
    res["aurc_risk"] = float(np.mean(1 - acc))

    recal = None
    if a.dev_features:
        dev = load_features(a.dev_features)
        cd = np.array([f["engine_verdict"] == f["gold"] for f in dev])
        xd = np.array([f["engine_conf"] for f in dev])
        if 0 < cd.sum() < len(cd):
            platt = LogisticRegression().fit(xd.reshape(-1, 1), cd)
            iso = IsotonicRegression(out_of_bounds="clip").fit(xd, cd.astype(float))
            p_platt = platt.predict_proba(conf.reshape(-1, 1))[:, 1]
            p_iso = iso.predict(conf)
            res["ece_platt_dev_fit"] = ece(correct, p_platt)
            res["ece_isotonic_dev_fit"] = ece(correct, p_iso)
            recal = p_platt

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    for label, p in (("raw confidence", conf), ("Platt (dev-fit)", recal)):
        if p is None:
            continue
        try:
            fy, fx = calibration_curve(correct, p, n_bins=6, strategy="quantile")
            ax.plot(fx, fy, "o-", ms=3, label=label)
        except ValueError:
            pass
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Verdict accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"{a.name}_reliability.png", dpi=300)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot(cov, acc)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Accuracy on retained claims")
    fig.tight_layout()
    fig.savefig(out / f"{a.name}_risk_coverage.png", dpi=300)

    (out / f"{a.name}_confidence_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
