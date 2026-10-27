#!/usr/bin/env python3
"""Tag every misclassified claim with a failure category (heuristic, from the engine
explanation). Review <name>_errors.csv by hand and fix tags before publishing.

From backend/:
  python ../benchmark/error_analysis.py --features ../benchmark/results/challenge_features.json --name challenge
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import latex_table  # noqa: E402
from features import load_features  # noqa: E402


def tag(f):
    g, p, ex = f["gold"], f["engine_verdict"], f.get("explanation", "")
    if g == "VERIFIED" and p == "CONTRADICTED":
        if "organization(s)" in ex:
            return "NER label noise (ORG)"
        if "person(s)" in ex:
            return "Name-form mismatch (NER/surname)"
        return "False contradiction (other)"
    if g == "UNSUPPORTED" and p == "CONTRADICTED":
        return "Abstention failure (absence treated as contradiction)"
    if g == "UNSUPPORTED" and p in ("VERIFIED", "PARTIALLY_SUPPORTED"):
        return "Unsafe support of unverifiable claim"
    if g == "CONTRADICTED" and p in ("VERIFIED", "PARTIALLY_SUPPORTED"):
        if "no structured extractor" in ex:
            return "Missed contradiction (no value extractor)"
        return "Missed contradiction (other)"
    if g == "CONTRADICTED" and p == "UNSUPPORTED":
        return "Contradiction downgraded to unsupported"
    if g == "VERIFIED" and p in ("PARTIALLY_SUPPORTED", "UNSUPPORTED"):
        return "Conservative downgrade (agreement/relationship)"
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="../benchmark/results")
    a = ap.parse_args()
    fs = load_features(a.features)
    errs = [f for f in fs if f["gold"] != f["engine_verdict"]]
    by = defaultdict(list)
    for f in errs:
        by[tag(f)].append(f)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{a.name}_errors.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "gold", "pred", "auto_tag", "manual_tag", "claim", "explanation"])
        for f in errs:
            w.writerow([f["id"], f["gold"], f["engine_verdict"], tag(f), "", f["claim"], f["explanation"]])

    cnt = Counter({k: len(v) for k, v in by.items()})
    rows = [[k, n, f"{100*n/len(fs):.1f}\\%"] for k, n in cnt.most_common()]
    (out / f"{a.name}_errors.tex").write_text(latex_table(
        ["Failure category", "Count", "Share of all claims"], rows,
        f"Error taxonomy on {a.name} ({len(errs)} errors / {len(fs)} claims).",
        f"tab:errors_{a.name}"))
    print(f"{len(errs)} errors / {len(fs)} claims")
    for k, n in cnt.most_common():
        ex = by[k][0]
        print(f"{n:3d}  {k}\n       e.g. [{ex['id']}] {ex['claim'][:90]}")


if __name__ == "__main__":
    main()
