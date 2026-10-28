#!/usr/bin/env python3
"""Probability-averaging ensemble of two (or more) NLI backbones + abstention thresholds.
Weight and thresholds are tuned on DEV only; TEST is evaluated once (optional).

Each features file must come from run_ablation.py on the SAME csv, with a different NLI_MODEL
and a different --nli-cache. From backend/:

  python ../benchmark/ensemble_nli.py \
     --dev-features  ../benchmark/results/vc_dev_deberta_features.json ../benchmark/results/vc_dev_large_features.json \
     [--max-false-ver 0.05] [--test-features <testA.json> <testB.json>]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import bootstrap_ci, compute_metrics  # noqa: E402
from features import load_features  # noqa: E402
from nli_threshold import conf_compare, decide_prob  # noqa: E402

LABS = ("entailment", "neutral", "contradiction")


def load_aligned(paths):
    sets = [load_features(p) for p in paths]
    n = len(sets[0])
    for s in sets:
        assert len(s) == n and all(a["id"] == b["id"] for a, b in zip(sets[0], s)), \
            "feature files are not aligned (different rows/ids); rerun run_ablation on the same CSV"
    return sets


def blend(sets, weights):
    out = []
    for rows in zip(*sets):
        f = copy.deepcopy(rows[0])
        probs = []
        for ev_i in range(len(rows[0]["nli_probs_all"])):
            probs.append({l: float(sum(w * r["nli_probs_all"][ev_i].get(l, 0.0)
                                       for w, r in zip(weights, rows))) for l in LABS})
        f["nli_probs_all"] = probs
        # argmax labels of the blend, so variants.predict('nli_only') stays consistent
        f["nli_all"] = [max(p, key=p.get) for p in probs]
        f["nli_primary"] = f["nli_all"][0] if f["nli_all"] else "none"
        out.append(f)
    return out


def best_thresholds(fs, max_fv):
    gold = [f["gold"] for f in fs]
    best = None
    for te in np.arange(0.30, 0.991, 0.05):
        for tc in np.arange(0.30, 0.991, 0.05):
            m = compute_metrics(gold, [decide_prob(f, te, tc) for f in fs])
            fv = m["false_verified_rate"] or 0.0
            if max_fv is not None and fv > max_fv:
                continue
            cand = (m["macro_f1_populated"], round(float(te), 2), round(float(tc), 2), m["accuracy"], fv)
            if best is None or cand[0] > best[0]:
                best = cand
    return best


def report(tag, gold, pred):
    m = compute_metrics(gold, pred)
    print(f"{tag:36s} acc={m['accuracy']:.3f} {bootstrap_ci(gold, pred, 'accuracy')}  "
          f"macroF1={m['macro_f1_populated']:.3f}  falseVER={m['false_verified_rate']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-features", nargs="+", required=True)
    ap.add_argument("--test-features", nargs="*", default=None)
    ap.add_argument("--max-false-ver", type=float, default=None)
    a = ap.parse_args()

    dev_sets = load_aligned(a.dev_features)
    k = len(dev_sets)
    grid = [(w, 1 - w) for w in np.arange(0, 1.001, 0.1)] if k == 2 else [tuple([1 / k] * k)]
    print(f"{len(dev_sets[0])} dev claims, {k} models\n\nDEV: weight on model 1 -> best (macroF1, tau_ent, tau_con, acc, falseVER)")
    best = None
    for ws in grid:
        res = best_thresholds(blend(dev_sets, ws), a.max_false_ver)
        if res is None:
            continue
        print(f"  w={ws[0]:.1f}: {res}")
        if best is None or res[0] > best[0][0]:
            best = (res, ws)
    (_, te, tc, _, _), ws = best
    print(f"\nChosen on DEV: weights={tuple(round(float(x), 2) for x in ws)} tau_ent={te} tau_con={tc}")
    dev_b = blend(dev_sets, ws)
    gd = [f["gold"] for f in dev_b]
    report("DEV ensemble + thresholds", gd, [decide_prob(f, te, tc) for f in dev_b])
    for i, s in enumerate(dev_sets):
        report(f"DEV model {i + 1} alone (same taus)", gd, [decide_prob(f, te, tc) for f in s])
    conf_compare("DEV ensemble", dev_b)

    if a.test_features:
        test_sets = load_aligned(a.test_features)
        tb = blend(test_sets, ws)
        gt = [f["gold"] for f in tb]
        print("\n--- TEST (single evaluation) ---")
        report("TEST ensemble + thresholds", gt, [decide_prob(f, te, tc) for f in tb])
        for i, s in enumerate(test_sets):
            report(f"TEST model {i + 1} alone (same taus)", gt, [decide_prob(f, te, tc) for f in s])


if __name__ == "__main__":
    main()
