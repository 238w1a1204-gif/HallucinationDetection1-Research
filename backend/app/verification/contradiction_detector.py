"""Contradiction detection using a Hugging Face NLI (Natural Language
Inference) model.

Default model: facebook/bart-large-mnli, which classifies a
(premise, hypothesis) pair into entailment / neutral / contradiction.
We treat the evidence text as the premise and the claim as the
hypothesis: "does the evidence entail, is neutral to, or contradict the
claim?".

DOCUMENTED LIMITATION:
General-purpose NLI models are trained on broad sentence-pair datasets
(e.g. MNLI), not specifically on fact-checking data. They can struggle
with:
- Numerical/date contradictions phrased subtly (e.g. "1879" vs "1880")
  which is why we ALSO run a dedicated entity/date checker
  (entity_matcher.py) rather than relying on NLI alone for this project.
- Long or multi-fact evidence paragraphs (the model works best on
  single, focused sentences).
This is a practical, "good enough for a student project" component, not
a state-of-the-art fact-verification-specific NLI model. This limitation
is documented in the README as well.
"""
from __future__ import annotations

import logging
import threading

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_lock = threading.Lock()
_nli_pipeline = None
_load_failed = False


def _get_nli_pipeline():
    global _nli_pipeline, _load_failed
    if _nli_pipeline is None and not _load_failed:
        with _lock:
            if _nli_pipeline is None and not _load_failed:
                try:
                    from transformers import pipeline

                    logger.info("Loading NLI model: %s", settings.NLI_MODEL)
                    _nli_pipeline = pipeline("text-classification", model=settings.NLI_MODEL, top_k=None)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Could not load NLI model '%s' (%s). Contradiction detection via NLI "
                        "will be disabled; entity/date checks still run.",
                        settings.NLI_MODEL,
                        exc,
                    )
                    _load_failed = True
    return _nli_pipeline


def detect_contradiction(claim: str, evidence_text: str) -> tuple[bool, str]:
    """Return (is_contradicted, label) for a single (claim, evidence) pair.

    If the NLI model could not be loaded, returns (False, "unavailable")
    rather than silently pretending a contradiction was ruled out - callers
    should treat "unavailable" as "not checked", and rely more heavily on
    the entity/date checker in that case.
    """
    pipe = _get_nli_pipeline()
    if pipe is None:
        return False, "unavailable"

    try:
        # bart-large-mnli's pipeline expects "text" (premise) and treats
        # the model as sequence-pair classification via `text_pair`.
        results = pipe({"text": evidence_text, "text_pair": claim})
        # `top_k=None` returns all label scores; find the top one.
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]
        top = max(results, key=lambda r: r["score"])
        label = top["label"].lower()
        return label == "contradiction", label
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("NLI inference failed for claim '%s': %s", claim, exc)
        return False, "error"
