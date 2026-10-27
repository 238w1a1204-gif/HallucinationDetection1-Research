#!/usr/bin/env python3
"""Convert VitaminC (claim + evidence + SUPPORTS/REFUTES/NEI) to your CSV schema
and make a leakage-free dev/test split.

Run from backend/:
  python ../benchmark/prepare_vitaminc.py --n-per-class 300 --out-dir ../benchmark/data
Fallback if the HF loader fails:
  python ../benchmark/prepare_vitaminc.py --jsonl path/to/vitaminc_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

MAP = {"SUPPORTS": "VERIFIED", "REFUTES": "CONTRADICTED", "NOT ENOUGH INFO": "UNSUPPORTED"}
COLS = ["id", "category", "question", "claim", "gold_verdict",
        "evidence_1_text", "evidence_1_source", "evidence_1_url",
        "evidence_2_text", "evidence_2_source", "evidence_2_url", "gold_rationale"]


def load(args) -> pd.DataFrame:
    if args.jsonl:
        rows = [json.loads(l) for l in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
        return pd.DataFrame(rows)
    from datasets import load_dataset
    ds = load_dataset(args.hf_name, split=args.split)
    print("Columns:", ds.column_names)
    df = ds.to_pandas()
    feat = ds.features.get("label")
    if feat is not None and hasattr(feat, "int2str"):
        df["label"] = [feat.int2str(int(x)) for x in df["label"]]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-name", default="tals/vitaminc")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--dev-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="../benchmark/data")
    a = ap.parse_args()

    df = load(a)
    df["label"] = df["label"].astype(str).str.upper().str.replace("_", " ")
    df = df[df["label"].isin(MAP)].dropna(subset=["claim", "evidence"])
    df = df.drop_duplicates(subset=["claim", "evidence"])
    parts = [g.sample(min(a.n_per_class, len(g)), random_state=a.seed) for _, g in df.groupby("label")]
    df = pd.concat(parts).reset_index(drop=True)

    out = pd.DataFrame({
        "id": [f"V{i:04d}" for i in range(len(df))],
        "category": df["label"].str.lower().str.replace(" ", "_"),
        "question": "",  # claim-only mode (VitaminC has no question)
        "claim": df["claim"],
        "gold_verdict": df["label"].map(MAP),
        "evidence_1_text": df["evidence"],
        "evidence_1_source": "Wikipedia",
        "evidence_1_url": "",
        "evidence_2_text": "", "evidence_2_source": "", "evidence_2_url": "",
        "gold_rationale": "VitaminC " + df["label"],
    })[COLS]

    # group by claim: contrastive pairs share claims -> keep them on the same side
    gss = GroupShuffleSplit(n_splits=1, train_size=a.dev_frac, random_state=a.seed)
    dev_idx, test_idx = next(gss.split(out, groups=out["claim"]))
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    out.iloc[dev_idx].to_csv(od / "vitaminc_dev.csv", index=False)
    out.iloc[test_idx].to_csv(od / "vitaminc_test.csv", index=False)
    print("dev:", out.iloc[dev_idx]["gold_verdict"].value_counts().to_dict())
    print("test:", out.iloc[test_idx]["gold_verdict"].value_counts().to_dict())


if __name__ == "__main__":
    main()
