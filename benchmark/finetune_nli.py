#!/usr/bin/env python3
"""Fine-tune an NLI checkpoint on the VitaminC TRAIN split (run on a Colab GPU).

Input order matches the verifier pipeline: premise = evidence, hypothesis = claim.
Claims that appear in any --exclude-csv file (your dev/test sets) are removed from training.

Example (from backend/):
  python ../benchmark/finetune_nli.py --out ../benchmark/models/deberta_vc \
     --exclude-csv ../benchmark/data/vitaminc_dev.csv ../benchmark/data/vitaminc_test.csv \
                   ../benchmark/data_test2/vitaminc_dev.csv ../benchmark/data_test2/vitaminc_test.csv
Then evaluate with:  NLI_MODEL=../benchmark/models/deberta_vc python ../benchmark/run_ablation.py ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

LAB2NLI = {"SUPPORTS": "entailment", "REFUTES": "contradiction", "NOT ENOUGH INFO": "neutral"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf-name", default="tals/vitaminc")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-per-class", type=int, default=6000)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-csv", nargs="*", default=[])
    a = ap.parse_args()

    from datasets import Dataset, load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    set_seed(a.seed)

    ds = load_dataset(a.hf_name, split=a.split)
    df = ds.to_pandas()
    feat = ds.features.get("label")
    if feat is not None and hasattr(feat, "int2str"):
        df["label"] = [feat.int2str(int(x)) for x in df["label"]]
    df["label"] = df["label"].astype(str).str.upper().str.replace("_", " ")
    df = df[df["label"].isin(LAB2NLI)].dropna(subset=["claim", "evidence"])
    df = df.drop_duplicates(subset=["claim", "evidence"])

    banned = set()
    for p in a.exclude_csv:
        if Path(p).exists():
            banned |= set(pd.read_csv(p)["claim"].astype(str))
        else:
            print("WARNING: exclude file not found:", p)
    n0 = len(df)
    df = df[~df["claim"].astype(str).isin(banned)]
    print(f"train pool {n0} -> {len(df)} after removing {n0 - len(df)} claims present in eval sets")

    parts = [g.sample(min(a.n_per_class, len(g)), random_state=a.seed) for _, g in df.groupby("label")]
    df = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    print("training label counts:", df["label"].value_counts().to_dict())

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base)
    l2i = {k.lower(): v for k, v in model.config.label2id.items()}
    assert set(LAB2NLI.values()) <= set(l2i), f"unexpected label names in base model: {l2i}"
    df["labels"] = df["label"].map(lambda x: l2i[LAB2NLI[x]])

    def enc(b):
        return tok(b["evidence"], b["claim"], truncation=True, max_length=a.max_len)

    train = Dataset.from_pandas(df[["claim", "evidence", "labels"]]).map(
        enc, batched=True, remove_columns=["claim", "evidence"])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out / "ckpt"), per_device_train_batch_size=a.bs, learning_rate=a.lr,
        num_train_epochs=a.epochs, weight_decay=0.01, warmup_ratio=0.06,
        fp16=torch.cuda.is_available(), save_strategy="no", logging_steps=100,
        report_to=[], seed=a.seed)
    Trainer(model=model, args=args, train_dataset=train,
            data_collator=DataCollatorWithPadding(tok)).train()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "finetune_config.json").write_text(json.dumps(
        {**vars(a), "n_train": len(df), "n_excluded_claims": len(banned)}, indent=2))
    print("saved to", out.resolve())


if __name__ == "__main__":
    main()
