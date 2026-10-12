# Frozen-Evidence Claim Verification Benchmark

## Purpose
This initial 40-item seed benchmark evaluates the claim-verification stage separately from answer generation. Each run uses fixed evidence snippets and URLs, avoiding changes caused by LLM answer wording or live retrieval.

## Categories
- `supported` (10): expected `VERIFIED`
- `wrong_entity_date_number` (10): expected `CONTRADICTED`
- `contradicted_relation` (10): expected `CONTRADICTED`
- `insufficient_evidence` (10): expected `UNSUPPORTED`

For `insufficient_evidence`, the gold label means the supplied evidence packet does not establish the precise claim. It does not mean the claim is necessarily false.

## Run from `backend/`
```bash
python ../benchmark/run_benchmark.py --input ../benchmark/claims.csv --output ../benchmark/results.csv
```
Use the project's existing Python 3.11 virtual environment. The script writes a per-claim CSV and a JSON metrics file with accuracy, per-class precision/recall/F1, macro-F1, confusion matrix, and the rate at which gold-UNSUPPORTED claims are predicted VERIFIED.

## Limitations and protocol
1. This is a seed benchmark, not yet publication-grade. Manually review gold labels and evidence packets before reporting results.
2. It reuses a small set of reference pages, so topical coverage is limited and repeated-source effects exist.
3. Evidence snippets are concise paraphrases of listed pages. Check each source and refine annotations if strict evidence-level adjudication is required.
4. This evaluates verification on fixed evidence. It does not evaluate retrieval recall or LLM answer generation.
5. Run the baseline before further changes. Preserve each results CSV and metrics JSON. Do not tune thresholds on this same set and call it held-out performance.
6. A future ablation runner should compare similarity-only, NLI-only, entity/value checks, and the full verifier on the same fixed evidence.
