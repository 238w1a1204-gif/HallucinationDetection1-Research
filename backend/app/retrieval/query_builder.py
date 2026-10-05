"""Subject-centred retrieval query builder.

Builds queries centred on the SUBJECT entity + relationship — NOT on the
LLM answer value.  This prevents cases like "Albert Einstein was born in
St. Petersburg" causing Wikipedia to retrieve the Saint Petersburg article.

KEY INSIGHT (2025 update):
Wikipedia's search engine works best with SHORT, CLEAN queries.  Adding
relationship-keyword suffixes like "born birthplace" can confuse it and
return wrong articles (e.g. "birthplace" → capital-punishment article for
France queries).

Updated strategy:
- BIOGRAPHICAL, GENERIC   → just the subject name
- LOCATION (born/died)    → "subject biography" (gets full life article)
- LOCATION (located_in)   → "subject" alone  (main article has location)
- FOUNDING                → "subject history" (history section has founders)
- CEO / LEADERSHIP        → "subject"          (main article has leadership)
- PRESIDENT               → "subject"
- DATE queries            → "subject"
- DISCOVERY               → "subject discovered invented"
- AUTHORSHIP              → "subject author"
- CAPITAL                 → "capital of subject" (reversed for clarity)

All queries are built from the subject entity extracted by question_analyzer.
No factual answers or entity names are hard-coded.
"""
from __future__ import annotations

import re

from app.verification.question_analyzer import (
    QuestionAnalysis,
    REL_BORN_IN, REL_DIED_IN, REL_FOUNDED_BY, REL_CEO_OF,
    REL_LOCATED_IN, REL_DISCOVERED, REL_AUTHORED_BY,
    REL_PRESIDENT_OF, REL_CAPITAL_OF, REL_GENERIC,
    REL_BIO_DESCRIPTION,
)

# ── Query suffix map ──────────────────────────────────────────────────────────
# Short, targeted suffixes that help Wikipedia find the right article.
# Empty string → use subject alone (most reliable for many cases).

_REL_SUFFIX: dict[str, str] = {
    REL_BORN_IN:         "",          # "Albert Einstein" → main bio article
    REL_DIED_IN:         "",
    REL_FOUNDED_BY:      "history",   # "Microsoft history" → founding section
    REL_CEO_OF:          "",          # "Microsoft" → main article has CEO
    REL_LOCATED_IN:      "",          # "Eiffel Tower" → main article has location
    REL_DISCOVERED:      "discovery",
    REL_AUTHORED_BY:     "author",
    REL_PRESIDENT_OF:    "president", # "United States president" → president article
    REL_CAPITAL_OF:      "",          # handled specially — subject alone
    REL_BIO_DESCRIPTION: "",          # subject name alone
    REL_GENERIC:         "",
}

_LEADING_THE = re.compile(r"^the\s+", re.I)


def _clean_subject(subject: str) -> str:
    """Strip leading 'the'/'The' — improves Wikipedia search precision."""
    return _LEADING_THE.sub("", subject).strip()


def build_subject_query(
    qa: "QuestionAnalysis | None",
    fallback_claim: str,
) -> str:
    """Return a Wikipedia-friendly search query centred on the subject.

    Parameters
    ----------
    qa:
        QuestionAnalysis from analyze_question().  When None, returns
        fallback_claim unchanged.
    fallback_claim:
        Used when no question analysis is available.
    """
    if qa is None or not qa.focus_entity:
        return fallback_claim

    focus = _clean_subject(qa.focus_entity.strip())
    if not focus:
        return fallback_claim

    rel = qa.relationship

    # Special case: capital questions work better as "subject" alone
    # then Wikipedia returns the country article which mentions the capital.
    # "France capital city" → "Capital city" (generic, wrong)
    # "France" → France article which says "Paris is the capital"
    if rel == REL_CAPITAL_OF:
        return focus

    suffix = _REL_SUFFIX.get(rel, "")
    if suffix:
        return f"{focus} {suffix}"

    # No suffix — subject name alone is the most reliable Wikipedia query
    return focus
