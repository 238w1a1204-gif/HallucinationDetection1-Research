"""Shared evaluation helpers (scikit-learn based). Used by every benchmark script."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = ["VERIFIED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]


def macro_f1_populated(y_true, y_pred) -> float:
    """Macro-F1 over classes that actually occur in the gold labels.

    (The original run_benchmark.py averages over all 4 classes, so an empty
    gold class such as PARTIALLY_SUPPORTED drags macro-F1 down by construction.)
    """
    present = [l for l in LABELS if l in set(y_true)]
    if not present:
        return 0.0
    _, _, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present, average=None, zero_division=0
    )
    return float(np.mean(f))


def compute_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = list(y_true), list(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    n_nonver = sum(t != "VERIFIED" for t in y_true)
    false_ver = sum(t != "VERIFIED" and q == "VERIFIED" for t, q in zip(y_true, y_pred))
    n_uns = sum(t == "UNSUPPORTED" for t in y_true)
    uns_ver = sum(t == "UNSUPPORTED" and q == "VERIFIED" for t, q in zip(y_true, y_pred))
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1_populated": round(macro_f1_populated(y_true, y_pred), 4),
        "macro_f1_all4": round(float(np.mean(f)), 4),
        "per_class": {
            l: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4), "support": int(s[i])}
            for i, l in enumerate(LABELS)
        },
        "confusion_matrix_gold_rows_pred_cols": cm.tolist(),
        # safety metrics: how often is something NOT supported by gold called VERIFIED?
        "false_verified_rate": round(false_ver / n_nonver, 4) if n_nonver else None,
        "unsupported_called_verified_rate": round(uns_ver / n_uns, 4) if n_uns else None,
    }


def bootstrap_ci(y_true, y_pred, metric="accuracy", n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI. metric in {'accuracy','macro_f1_populated'}."""
    yt, yp = np.array(y_true), np.array(y_pred)
    rng = np.random.default_rng(seed)
    fn = (lambda a, b: accuracy_score(a, b)) if metric == "accuracy" else (
        lambda a, b: macro_f1_populated(list(a), list(b)))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        vals.append(fn(yt[idx], yp[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)


def latex_table(header, rows, caption, label) -> str:
    cols = "l" + "c" * (len(header) - 1)
    esc = lambda x: str(x).replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\small", rf"\begin{{tabular}}{{{cols}}}", r"\hline",
             " & ".join(esc(h) for h in header) + r" \\", r"\hline"]
    for r in rows:
        lines.append(" & ".join(esc(c) for c in r) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
