"""Answer-value verification: does the answer VALUE match what the evidence
says for the requested relationship?

THE CORE PROBLEM
----------------
Semantic similarity alone cannot distinguish:

  Q: "Where was Einstein born?"  A: "St. Petersburg"
  Evidence: "Einstein was born in Ulm, Germany."

Both texts share the same entities (Einstein) and the same relationship
(born in), but the LOCATION VALUES differ.  Embedding similarity is high
because both are about Einstein's birthplace — yet the answer is wrong.

This module makes the distinction explicit:
  - It extracts the relationship-relevant value from the evidence.
  - It checks whether the answer value matches that extracted value.
  - It returns VALUE_SUPPORTED, VALUE_CONTRADICTED, or VALUE_UNVERIFIABLE.

DESIGN PRINCIPLES
-----------------
• No factual answers are hard-coded.  All conclusions come from retrieved
  evidence text passed at runtime.
• Works for locations, dates, person names, organisation names, and generic
  text values.
• Designed as a standalone module so it can be unit-tested without the full
  pipeline.

DOCUMENTED LIMITATIONS
-----------------------
• NER accuracy depends on spaCy en_core_web_sm (moderate recall for
  uncommon names / spellings).
• Proximity-based value extraction is heuristic; complex sentences may fail.
• Numbers/quantities are only partially supported.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.verification.question_analyzer import (
    QT_DATE, QT_LOCATION, QT_NUMBER, QT_ORG, QT_PERSON, QT_BIO,
    REL_BORN_IN, REL_CEO_OF, REL_DIED_IN, REL_FOUNDED_BY,
    REL_LOCATED_IN, REL_PRESIDENT_OF, REL_DISCOVERED, REL_AUTHORED_BY,
    REL_BIO_DESCRIPTION,
    QuestionAnalysis,
)

# ── Status constants ──────────────────────────────────────────────────────────
VALUE_SUPPORTED    = "VALUE_SUPPORTED"
VALUE_CONTRADICTED = "VALUE_CONTRADICTED"
VALUE_UNVERIFIABLE = "VALUE_UNVERIFIABLE"


@dataclass
class AnswerValueCheckResult:
    status: str                    # one of VALUE_* constants
    answer_value: str              # normalised answer we checked
    evidence_values: list[str] = field(default_factory=list)
    reason: str = ""


def check_answer_value(
    qa: QuestionAnalysis,
    answer_text: str,
    evidence_text: str,
    subject_entity: str = "",
) -> AnswerValueCheckResult:
    """Compare the answer value against evidence for the requested relationship.

    Parameters
    ----------
    qa:
        Result of question_analyzer.analyze_question().
    answer_text:
        The LLM's answer (possibly short: "Paris", "1879", "Bill Gates").
    evidence_text:
        A single evidence snippet from a retrieval source.
    subject_entity:
        The focus entity from the question (e.g. "Albert Einstein").
        Used to guard biographical contradiction: only fire CONTRADICTED
        for biographical checks when the subject is actually in the evidence.

    Returns
    -------
    AnswerValueCheckResult
    """
    ans = (answer_text or "").strip()
    evid = (evidence_text or "").strip()
    if not ans or not evid:
        return AnswerValueCheckResult(
            status=VALUE_UNVERIFIABLE,
            answer_value=ans,
            reason="Answer or evidence is empty.",
        )

    ans_l  = ans.lower()
    evid_l = evid.lower()

    # ── Answer type mismatch check ────────────────────────────────────────
    if qa.question_type == QT_PERSON and qa.relationship not in (
        REL_BIO_DESCRIPTION,
    ):
        type_check = _check_answer_type_is_person(ans, evid)
        if type_check is not None:
            return type_check

    # ── Route by question type / relationship ─────────────────────────────
    if qa.question_type == QT_LOCATION or qa.relationship in (
        REL_BORN_IN, REL_DIED_IN, REL_LOCATED_IN
    ):
        return _check_location(ans_l, evid, evid_l, qa.relationship)

    if qa.question_type == QT_DATE or qa.relationship in (REL_BORN_IN, REL_DIED_IN):
        return _check_date(ans_l, evid_l)

    if qa.question_type in (QT_PERSON,) or qa.relationship in (
        REL_FOUNDED_BY, REL_CEO_OF, REL_PRESIDENT_OF, REL_DISCOVERED, REL_AUTHORED_BY
    ):
        return _check_person(ans_l, ans, evid, evid_l)

    if qa.question_type == QT_BIO:
        return _check_biographical(ans_l, evid_l, subject_entity=subject_entity)

    if qa.question_type == QT_NUMBER:
        return _check_number(ans_l, evid_l)

    return _check_generic(ans_l, evid_l)


# ── Type-mismatch check ───────────────────────────────────────────────────────

def _check_answer_type_is_person(
    answer: str, evidence: str
) -> AnswerValueCheckResult | None:
    """Return a VALUE_CONTRADICTED result if the answer looks like an
    organisation when a person was requested, else return None.

    This catches "Microsoft Corp." as the answer to "Who founded Microsoft?"
    """
    try:
        from app.nlp.entity_extractor import extract_entities
        ans_ents = extract_entities(answer)
        # If answer has organisations but no people, it's the wrong type
        if ans_ents.organizations and not ans_ents.people:
            return AnswerValueCheckResult(
                status=VALUE_CONTRADICTED,
                answer_value=answer.lower(),
                reason=(
                    f"The answer '{answer}' appears to name an organisation, "
                    "but the question asks for a person. "
                    "This does not satisfy the requested relationship."
                ),
            )
    except Exception:
        pass
    return None


# ── Location check ────────────────────────────────────────────────────────────

_LOC_PROXIMITY_PATTERNS: list[str] = [
    r"\bborn in ([A-Z][A-Za-z ,\-]+?)(?:[,.]|\s+in\b|\s+on\b|$)",
    r"\bdied in ([A-Z][A-Za-z ,\-]+?)(?:[,.]|$)",
    r"\blocated in ([A-Z][A-Za-z ,\-]+?)(?:[,.]|$)",
    r"\bheadquartered in ([A-Z][A-Za-z ,\-]+?)(?:[,.]|$)",
    r"\bbased in ([A-Z][A-Za-z ,\-]+?)(?:[,.]|$)",
    r"\bsituated in ([A-Z][A-Za-z ,\-]+?)(?:[,.]|$)",
    r"\bin ([A-Z][A-Za-z ,\-]+?),\s+[A-Z]",  # "in Paris, France"
]


def _extract_location_values(evidence: str) -> list[str]:
    """Extract location strings from evidence using proximity patterns + NER."""
    locs: set[str] = set()

    # Proximity patterns
    for pat in _LOC_PROXIMITY_PATTERNS:
        for m in re.finditer(pat, evidence):
            val = m.group(1).strip(" ,.")
            if 1 < len(val) < 60:
                locs.add(val.lower())

    # spaCy NER
    try:
        from app.nlp.entity_extractor import extract_entities
        ev_ents = extract_entities(evidence)
        for loc in ev_ents.locations | ev_ents.other:
            locs.add(loc.lower())
    except Exception:
        pass

    return sorted(locs)


def _check_location(
    ans_l: str, evidence: str, evid_l: str, rel: str
) -> AnswerValueCheckResult:
    # Direct substring check first — most reliable
    if ans_l in evid_l:
        return AnswerValueCheckResult(
            status=VALUE_SUPPORTED,
            answer_value=ans_l,
            reason=f"Answer value '{ans_l}' found directly in evidence.",
        )

    # Token-level match: "Ulm, Germany" → check for "ulm" OR "germany" individually.
    # This handles cases like answer="Ulm, Germany" vs evidence="...Ulm, Kingdom..."
    ans_tokens = [t.strip(",.;:") for t in ans_l.split() if len(t.strip(",.;:")) >= 3]
    if ans_tokens and any(tok in evid_l for tok in ans_tokens):
        return AnswerValueCheckResult(
            status=VALUE_SUPPORTED,
            answer_value=ans_l,
            reason=f"Answer token(s) {ans_tokens} found in evidence.",
        )

    ev_locs = _extract_location_values(evidence)

    if ev_locs:
        # Return CONTRADICTED when evidence clearly names different locations
        # for this relationship. We don't require a person/org in the evidence —
        # location articles are also legitimate contradiction sources.
        return AnswerValueCheckResult(
            status=VALUE_CONTRADICTED,
            answer_value=ans_l,
            evidence_values=ev_locs[:5],
            reason=(
                f"Answer location '{ans_l}' not found in evidence. "
                f"Evidence mentions: {ev_locs[:3]}."
            ),
        )

    return AnswerValueCheckResult(
        status=VALUE_UNVERIFIABLE,
        answer_value=ans_l,
        reason="Could not extract matching or contradicting location from evidence.",
    )


# ── Date check ────────────────────────────────────────────────────────────────

def _check_date(ans_l: str, evid_l: str) -> AnswerValueCheckResult:
    from app.nlp.entity_extractor import extract_years

    ans_years  = extract_years(ans_l)
    evid_years = extract_years(evid_l)

    if ans_years and evid_years:
        if ans_years & evid_years:
            return AnswerValueCheckResult(
                status=VALUE_SUPPORTED,
                answer_value=ans_l,
                evidence_values=sorted(evid_years),
                reason=f"Year(s) {sorted(ans_years)} found in evidence.",
            )
        return AnswerValueCheckResult(
            status=VALUE_CONTRADICTED,
            answer_value=ans_l,
            evidence_values=sorted(evid_years),
            reason=(
                f"Answer year(s) {sorted(ans_years)} differ from "
                f"evidence year(s) {sorted(evid_years)}."
            ),
        )

    if ans_l in evid_l:
        return AnswerValueCheckResult(
            status=VALUE_SUPPORTED,
            answer_value=ans_l,
            reason=f"Answer value '{ans_l}' found verbatim in evidence.",
        )

    return AnswerValueCheckResult(
        status=VALUE_UNVERIFIABLE,
        answer_value=ans_l,
        evidence_values=sorted(evid_years),
        reason="Could not match date value between answer and evidence.",
    )


# ── Person check ──────────────────────────────────────────────────────────────

def _check_person(
    ans_l: str, ans_orig: str, evidence: str, evid_l: str
) -> AnswerValueCheckResult:
    # Direct substring
    if ans_l in evid_l:
        return AnswerValueCheckResult(
            status=VALUE_SUPPORTED,
            answer_value=ans_l,
            reason=f"Answer value '{ans_l}' found in evidence.",
        )

    # Check NER people in evidence
    try:
        from app.nlp.entity_extractor import extract_entities
        ev_ents = extract_entities(evidence)
        ev_people = {p.lower() for p in ev_ents.people}
    except Exception:
        ev_people = set()

    if ev_people:
        return AnswerValueCheckResult(
            status=VALUE_CONTRADICTED,
            answer_value=ans_l,
            evidence_values=sorted(ev_people)[:5],
            reason=(
                f"Answer person '{ans_l}' not found in evidence. "
                f"Evidence mentions: {sorted(ev_people)[:3]}."
            ),
        )

    return AnswerValueCheckResult(
        status=VALUE_UNVERIFIABLE,
        answer_value=ans_l,
        reason="Could not extract person values from evidence to compare.",
    )


# ── Biographical check ────────────────────────────────────────────────────────

def _check_biographical(ans_l: str, evid_l: str, subject_entity: str = "") -> AnswerValueCheckResult:
    """Check biographical claims.

    Logic:
    1. If the subject entity is provided but NOT found in evidence → UNVERIFIABLE.
       (Cannot confirm or deny claims about Einstein using Prince Albert's article.)
    2. If subject IS in evidence and answer descriptors are in evidence → SUPPORTED.
    3. If subject IS in evidence and answer contains a role-word NOT in evidence
       AND evidence contains a DIFFERENT role-word → CONTRADICTED.
    4. Otherwise → UNVERIFIABLE.

    Role words: occupation/role descriptors (physicist, politician, painter, …)
    Generic nationality/adj words (german, french, british) are NOT sufficient
    for either support or contradiction on their own.
    """
    # ── Role-word sets ─────────────────────────────────────────────────────
    # Words that specifically describe a person's occupation/role.
    # These distinguish "physicist" from "politician" from "painter".
    ROLE_WORDS = {
        "physicist", "chemist", "biologist", "mathematician", "astronomer",
        "scientist", "engineer", "inventor", "researcher",
        "politician", "senator", "congressman", "minister", "president",
        "prime minister", "chancellor", "governor", "diplomat",
        "painter", "artist", "sculptor", "architect", "composer", "musician",
        "singer", "actor", "actress", "director", "author", "writer", "poet",
        "soldier", "general", "admiral", "officer", "commander",
        "philosopher", "theologian", "economist", "historian",
        "athlete", "footballer", "cricketer", "swimmer",
        "doctor", "physician", "surgeon",
        "lawyer", "judge",
        "entrepreneur", "businessman",
        "explorer", "navigator",
    }

    # ── Step 1: Subject presence gate ─────────────────────────────────────
    subject_in_evidence = False
    if subject_entity:
        subj_l = subject_entity.lower().strip()
        subj_tokens = [t for t in subj_l.split() if len(t) >= 4]
        if subj_l in evid_l:
            subject_in_evidence = True
        elif len(subj_tokens) >= 1 and all(t in evid_l for t in subj_tokens):
            subject_in_evidence = True
    else:
        # No subject constraint provided — allow check to proceed
        subject_in_evidence = True

    if not subject_in_evidence:
        return AnswerValueCheckResult(
            status=VALUE_UNVERIFIABLE,
            answer_value=ans_l,
            reason=(
                "Evidence could not be confirmed to be about the question subject. "
                "Cannot support or contradict without subject-matched evidence."
            ),
        )

    # ── Step 2: Role-word matching ─────────────────────────────────────────
    ans_words  = set(re.findall(r"\b[a-z]+\b", ans_l))
    evid_words = set(re.findall(r"\b[a-z]+\b", evid_l))

    ans_roles  = ROLE_WORDS & ans_words
    evid_roles = ROLE_WORDS & evid_words

    # SUPPORTED: answer role word appears in evidence
    if ans_roles and (ans_roles & evid_words):
        return AnswerValueCheckResult(
            status=VALUE_SUPPORTED,
            answer_value=ans_l,
            reason=f"Role descriptor(s) {sorted(ans_roles & evid_words)[:2]} found in evidence.",
        )

    # CONTRADICTED: answer has a role, evidence has a DIFFERENT role, subject confirmed
    if ans_roles and evid_roles and not (ans_roles & evid_roles):
        return AnswerValueCheckResult(
            status=VALUE_CONTRADICTED,
            answer_value=ans_l,
            reason=(
                f"Answer role {sorted(ans_roles)} conflicts with "
                f"evidence role {sorted(evid_roles)} for the same subject."
            ),
        )

    # SUPPORTED (fallback): significant non-role word overlap
    # Use only content words ≥ 5 chars to avoid "german"/"born" noise
    content_ans  = {w for w in ans_words  if len(w) >= 5} - ROLE_WORDS
    content_evid = {w for w in evid_words if len(w) >= 5} - ROLE_WORDS
    overlap = content_ans & content_evid
    if overlap:
        return AnswerValueCheckResult(
            status=VALUE_SUPPORTED,
            answer_value=ans_l,
            reason=f"Content descriptor(s) {sorted(overlap)[:3]} found in evidence.",
        )

    return AnswerValueCheckResult(
        status=VALUE_UNVERIFIABLE,
        answer_value=ans_l,
        reason="Could not match biographical description against evidence.",
    )

    return AnswerValueCheckResult(
        status=VALUE_UNVERIFIABLE,
        answer_value=ans_l,
        reason="Could not match biographical description against evidence.",
    )


# ── Number check ──────────────────────────────────────────────────────────────

def _check_number(ans_l: str, evid_l: str) -> AnswerValueCheckResult:
    digits = re.findall(r"\d[\d,.]*", ans_l)
    for d in digits:
        if d in evid_l:
            return AnswerValueCheckResult(
                status=VALUE_SUPPORTED,
                answer_value=ans_l,
                evidence_values=[d],
                reason=f"Number '{d}' found in evidence.",
            )
    if digits:
        return AnswerValueCheckResult(
            status=VALUE_UNVERIFIABLE,
            answer_value=ans_l,
            reason=f"Numeric value(s) {digits} not found in evidence.",
        )
    return _check_generic(ans_l, evid_l)


# ── Generic check ─────────────────────────────────────────────────────────────

def _check_generic(ans_l: str, evid_l: str) -> AnswerValueCheckResult:
    if ans_l in evid_l:
        return AnswerValueCheckResult(
            status=VALUE_SUPPORTED,
            answer_value=ans_l,
            reason=f"Answer '{ans_l}' found in evidence text.",
        )
    return AnswerValueCheckResult(
        status=VALUE_UNVERIFIABLE,
        answer_value=ans_l,
        reason=(
            f"Could not confirm or contradict '{ans_l}' from evidence "
            "(no structured extractor for this answer type)."
        ),
    )
