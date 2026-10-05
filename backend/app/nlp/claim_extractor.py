"""Splits an LLM answer into individual, (mostly) independently-checkable claims.

Approach (kept intentionally simple for reliability):
1. Use spaCy to split the answer into sentences.
2. Further split sentences on coordinating conjunctions that join two
   independent clauses (e.g. "X was born in 1879, and he later moved to
   the US.") so each half becomes its own claim when practical.
3. Discard trivial/non-factual fragments (too short, or pure questions).

This is a heuristic, rule-based segmentation - not a trained "claim
extraction" model. It is a practical starting point; more sophisticated
claim decomposition (e.g. using an LLM prompt) is listed as future work.
"""
from __future__ import annotations

import logging
import threading

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_lock = threading.Lock()
_nlp = None

MIN_CLAIM_LENGTH_CHARS = 8


def _get_nlp():
    global _nlp
    if _nlp is None:
        with _lock:
            if _nlp is None:
                import spacy

                try:
                    _nlp = spacy.load(settings.SPACY_MODEL)
                except OSError:
                    logger.warning(
                        "spaCy model '%s' not found; downloading it now.", settings.SPACY_MODEL
                    )
                    from spacy.cli import download

                    download(settings.SPACY_MODEL)
                    _nlp = spacy.load(settings.SPACY_MODEL)
    return _nlp


def extract_claims(answer_text: str) -> list[str]:
    """Return a list of atomic claim strings extracted from `answer_text`."""
    if not answer_text or not answer_text.strip():
        return []

    nlp = _get_nlp()
    doc = nlp(answer_text)

    claims: list[str] = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        sub_claims = _split_on_conjunctions(sent)
        for sc in sub_claims:
            sc = sc.strip(" ,.;")
            if len(sc) >= MIN_CLAIM_LENGTH_CHARS and not sc.endswith("?"):
                claims.append(sc)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique_claims: list[str] = []
    for c in claims:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            unique_claims.append(c)
    return unique_claims


def _split_on_conjunctions(sent) -> list[str]:
    """Split a spaCy sentence span on top-level coordinating conjunctions
    that connect two clauses each containing their own subject/verb.

    This catches cases like "Paris is the capital of France, and it has a
    population of about 2 million." -> two claims.
    Falls back to returning the whole sentence if no clean split is found.
    """
    text = sent.text
    # Simple, safe heuristic: split on ", and " / ", but " only if both
    # sides look long enough to be independent statements.
    for conj in [", and ", ", but ", "; "]:
        if conj in text:
            parts = text.split(conj)
            if all(len(p.strip()) >= MIN_CLAIM_LENGTH_CHARS for p in parts):
                return parts
    return [text]


# ── Short-answer / question-context claim builder ─────────────────────────────

def build_claims_from_question_answer(question: str, answer: str) -> list[str]:
    """Construct verifiable claim strings when the LLM answer is too short
    to produce claims via sentence splitting.

    ``extract_claims("Paris")`` returns [] because "Paris" is 5 chars and
    has no sentence structure.  But "Paris" is a perfectly valid answer that
    must be verified.  This function uses the question to build a full
    declarative sentence:

        question = "Where is the Eiffel Tower located?"
        answer   = "Paris"
        returns  = ["The Eiffel Tower is located in Paris.", "Paris"]

    The short answer itself is also included so retrieval queries are
    enriched with the original answer token.

    Parameters
    ----------
    question:
        The original user question.
    answer:
        The LLM's answer (may be a single word or very short phrase).

    Returns
    -------
    list[str]
        Deduplicated claim strings suitable for evidence retrieval and
        verification.  Returns [] when the answer is blank.

    Notes
    -----
    This function is called by the pipeline orchestrator ONLY when
    ``extract_claims(raw_answer)`` returns an empty list, so it does not
    interfere with normal multi-sentence answers.
    """
    answer = (answer or "").strip()
    if not answer:
        return []

    # Import here to avoid circular imports — question_analyzer imports
    # _get_nlp() from this module, not the functions above.
    from app.verification.question_analyzer import build_claim_from_question

    full_claim = build_claim_from_question(question, answer)

    claims: list[str] = []
    if full_claim and full_claim.lower().strip() != answer.lower().strip():
        claims.append(full_claim)

    # Always include the answer itself if long enough to be informative
    if len(answer) >= MIN_CLAIM_LENGTH_CHARS:
        claims.append(answer)

    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for c in claims:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result
