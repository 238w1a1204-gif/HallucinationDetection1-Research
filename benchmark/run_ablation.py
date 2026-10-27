#!/usr/bin/env python3
"""Ablation study on frozen evidence. Run from backend/:

  python ../benchmark/run_ablation.py --input ../benchmark/claims.csv --name challenge --min-sources 2
  python ../benchmark/run_ablation.py --input ../benchmark/data/vitaminc_dev.csv --name vc_dev --min-sources 1

Outputs in ../benchmark/results/: <name>_features.json, <name>_ablation.csv,
<name>_ablation_metrics.json, <name>_ablation.tex
NOTE: VitaminC has ONE evidence snippet per claim, so VERIFIED needs --min-sources 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as F  # noqa: E402
from eval_utils import bootstrap_ci, compute_metrics, latex_table  # noqa: E402
from variants import CONFIGS, FULL, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="../benchmark/results")
    ap.add_argument("--nli-cache", default="../benchmark/cache/nli_cache.json")
    ap.add_argument("--min-sources", type=int, default=2)
    ap.add_argument("--online", action="store_true", help="allow Wikipedia spelling resolution")
    a = ap.parse_args()

    from app.config import get_settings
    st = get_settings()
    st.AGREEMENT_MIN_SOURCES = a.min_sources  # same object used by verdict_engine
    strong, partial = st.SIMILARITY_STRONG_THRESHOLD, st.SIMILARITY_PARTIAL_THRESHOLD
    if not a.online:
        F.make_offline()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = F.read_rows(a.input)
    nli = F.NLICache(a.nli_cache)
    feats = []
    for i, row in enumerate(rows, 1):
        try:
            feats.append(F.extract_features(row, nli))
        except Exception as exc:
            feats.append({"id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"})
        if i % 25 == 0:
            nli.save()
            print(f"  features {i}/{len(rows)}")
    nli.save()
    (out / f"{a.name}_features.json").write_text(json.dumps(feats, indent=1))
    ok = [f for f in feats if not f.get("error")]
    print(f"features ok: {len(ok)}/{len(rows)}")

    gold = [f["gold"] for f in ok]
    kw = dict(strong=strong, partial=partial, min_sources=a.min_sources)
    preds = {name: [predict(f, cfg, **kw) for f in ok] for name, cfg in CONFIGS}
    preds["Engine (determine_verdict)"] = [f["engine_verdict"] for f in ok]
    agree = sum(p == e for p, e in zip(preds["Full (+AnswerValue)"], preds["Engine (determine_verdict)"]))
    print(f"\nFull-variant vs real engine agreement: {agree}/{len(ok)} (should be {len(ok)})")

    metrics, tex_rows = {}, []
    print(f"\n{'config':28s} {'acc [95% CI]':24s} {'macroF1pop [95% CI]':26s} falseVER")
    for name, p in preds.items():
        m = compute_metrics(gold, p)
        a_ci = bootstrap_ci(gold, p, "accuracy")
        f_ci = bootstrap_ci(gold, p, "macro_f1_populated")
        m["accuracy_ci95"], m["macro_f1_populated_ci95"] = a_ci, f_ci
        metrics[name] = m
        fv = m["false_verified_rate"]
        print(f"{name:28s} {m['accuracy']:.3f} [{a_ci[0]:.2f},{a_ci[1]:.2f}]   "
              f"{m['macro_f1_populated']:.3f} [{f_ci[0]:.2f},{f_ci[1]:.2f}]   {fv}")
        tex_rows.append([name, f"{m['accuracy']:.3f} [{a_ci[0]:.2f},{a_ci[1]:.2f}]",
                         f"{m['macro_f1_populated']:.3f}", f"{fv:.3f}" if fv is not None else "-"])

    (out / f"{a.name}_ablation_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / f"{a.name}_ablation.tex").write_text(latex_table(
        ["Configuration", "Acc [95\\% CI]", "Macro-F1", "False-VER"], tex_rows,
        f"Ablation on {a.name} (frozen evidence).", f"tab:ablation_{a.name}"))
    with (out / f"{a.name}_ablation.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "gold", "claim"] + list(preds))
        for i, f in enumerate(ok):
            w.writerow([f["id"], f["gold"], f["claim"]] + [preds[k][i] for k in preds])
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
