"""Generates a corrected answer from per-claim verdicts and evidence.

DESIGN PRINCIPLES
-----------------
• Never repeat an LLM claim verbatim if it was not verified.
• Never invent information — only use text from retrieved evidence.
• Be transparent: every modification is noted in [Corrections applied].
• Never use irrelevant/wrong-entity evidence as a correction.
• When evidence contradicts or doesn't support the answer, extract the
  SPECIFIC value that answers the question (not a generic intro sentence).

RELATIONSHIP-AWARE EXTRACTION
------------------------------
For every question relationship we maintain two separate extraction paths:

  FOUNDED_BY + QT_PERSON → extract person name(s) after "by"
  FOUNDED_BY + QT_DATE   → extract year / date after "founded in/on"

  DISCOVERED_BY          → extract person name after "invented/credited to"
  CAPITAL_OF             → extract city name before/after "capital"
  BORN_IN                → extract location after "born in"
  PRESIDENT_OF           → extract person name before/after "president"
  BIOGRAPHICAL           → extract first descriptive sentence about the subject

For UNSUPPORTED verdicts where the corrector previously returned
"insufficient evidence", we now also attempt extraction so that when
evidence IS available but the LLM answer was simply unrelated, we can
still surface the correct answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.verification.verdict_engine import (
    VERDICT_CONTRADICTED,
    VERDICT_PARTIALLY_SUPPORTED,
    VERDICT_UNSUPPORTED,
    VERDICT_VERIFIED,
)

_PARTIAL_DISCLAIMER = " (partially supported — not independently verified)"

_USELESS_EVIDENCE_PATTERNS = [
    re.compile(r"may refer to:", re.I),
    re.compile(r"can refer to:", re.I),
    re.compile(r"is a disambiguation", re.I),
    re.compile(r"^[A-Za-z\s]+ \(disambiguation\)", re.I),
    re.compile(r"^[A-Za-z\s]+ may refer to", re.I),
]

# ── Relationship → value-extraction patterns ──────────────────────────────────
# Each key is a (relationship, question_type) tuple.  When question_type
# doesn't matter, use "*" as the type wildcard.
# Patterns capture GROUP 1 = the answer value.

_REL_EXTRACTION: dict[tuple, list[re.Pattern]] = {

    # ── Birthplace ────────────────────────────────────────────────────────
    ("BORN_IN", "*"): [
        re.compile(r"born (?:in|on|at)\s+([A-Z][^,.;()\n]+?)(?:\s*,|\s*\(|\.)", re.I),
        re.compile(r"birthplace[:\s]+([A-Z][^,.;()\n]+?)(?:\s*,|\.|\n)", re.I),
        re.compile(r"native of\s+([A-Z][^,.;()\n]+?)(?:\s*,|\.|\n)", re.I),
        # Wikidata: "Polish-born French physicist" → "Poland"
        re.compile(r"\b([A-Z][a-z]+)-born\b", re.I),
    ],

    # ── Death place ───────────────────────────────────────────────────────
    ("DIED_IN", "*"): [
        re.compile(r"died (?:in|on|at)\s+([A-Z][^,.;()\n]+?)(?:\s*,|\.|\n)", re.I),
        re.compile(r"death (?:in|at)\s+([A-Z][^,.;()\n]+?)(?:\s*,|\.|\n)", re.I),
    ],

    # ── Location ──────────────────────────────────────────────────────────
    ("LOCATED_IN", "*"): [
        re.compile(r"located (?:in|on|at)\s+([A-Z][^,.;()\n]+?)(?:\s*,|\.|\n)", re.I),
        re.compile(r"situated (?:in|on|at)\s+([A-Z][^,.;()\n]+?)(?:\s*,|\.|\n)", re.I),
        # "tower on the Champ de Mars in Paris, France"
        re.compile(r"\bin\s+([A-Z][A-Za-z\s]+?),\s+(?:France|England|Germany|Italy|Spain|USA|US|UK|Japan|China|India|Russia|Canada|Australia)\b", re.I),
    ],

    # ── Capital city ──────────────────────────────────────────────────────
    # The France Wikipedia article says "Paris, the capital city and largest city"
    # We must match THAT pattern first, then fallbacks.
    ("CAPITAL_OF", "*"): [
        # "Paris, the capital city and largest city of France"
        re.compile(r"([A-Z][A-Za-z\s]{1,25}?),\s+the capital(?:\s+city)?", re.I),
        # "capital city is Paris" / "capital is Paris"
        re.compile(r"capital\s+(?:city\s+)?(?:of\s+\w+\s+)?(?:is|:)\s*([A-Z][A-Za-z\s]+?)(?:\s*[,.]|\n)", re.I),
        # "Paris is the capital of France"
        re.compile(r"([A-Z][A-Za-z\s]{1,25}?) is (?:the )?capital\b", re.I),
        # DuckDuckGo format: "capital: Paris"
        re.compile(r"capital[:\s]+([A-Z][A-Za-z\s]{1,25})(?:\s*[,.\n])", re.I),
    ],

    # ── Founder — PERSON question ("Who founded X?") ─────────────────────
    ("FOUNDED_BY", "PERSON"): [
        re.compile(r"founded\b.{0,40}?\bby\s+([A-Z][^.;()\n]+?)(?:\s+in\s+[A-Z]|\s+in\s+\d|\.|$)", re.I),
        re.compile(r"co-founded by\s+([A-Z][^,.;()\n]+?)(?:\s+in\s+\d|\.|,)", re.I),
        # "Bill Gates and Paul Allen founded Microsoft"
        re.compile(r"([A-Z][A-Za-z\s]+?)\s+(?:and\s+[A-Z][A-Za-z\s]+?\s+)?co?-?founded\b", re.I),
    ],

    # ── Founding date — DATE question ("When was X founded?") ────────────
    ("FOUNDED_BY", "DATE"): [
        # "founded on April 4, 1975"
        re.compile(r"founded\s+(?:on\s+)?([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})", re.I),
        # "founded in 1975"
        re.compile(r"founded\s+in\s+(\d{4})", re.I),
        # "founded by X in 1994"  (year appears later)
        re.compile(r"founded\b[^.]*?\bin\s+(\d{4})\b", re.I),
        re.compile(r"established\s+in\s+(\d{4})", re.I),
        # "(1975)" bare year in parentheses after name
        re.compile(r"\((\d{4})[–-]", re.I),
    ],

    # ── CEO ───────────────────────────────────────────────────────────────
    ("CEO_OF", "*"): [
        re.compile(r"(?:CEO|chief executive officer)[:\s]+([A-Z][^,.;()\n]+?)(?:\s*[,.]|\n)", re.I),
        re.compile(r"([A-Z][^,.;()\n]+?)\s+(?:is|was|serves as|became)\s+(?:the\s+)?CEO", re.I),
    ],

    # ── President / leader ────────────────────────────────────────────────
    ("PRESIDENT_OF", "*"): [
        # "George Washington was the first president"
        re.compile(r"([A-Z][A-Za-z\s]+?)\s+was the first president", re.I),
        # "first president … George Washington"  (reverse order)
        re.compile(r"first president[^,\.]*?(?:was|:)\s*([A-Z][^,.;()\n]+?)(?:\s*[,.]|\n)", re.I),
        re.compile(r"([A-Z][^,.;()\n]+?)\s+(?:was|is|became|served as)(?:\s+the)?\s+(?:first\s+)?president", re.I),
    ],

    # ── Discovery / invention (person) ───────────────────────────────────
    ("DISCOVERED_BY", "*"): [
        re.compile(r"invented by\s+([A-Z][^,.;()\n]+?)(?:\s*[,.]|\n)", re.I),
        re.compile(r"discovered by\s+([A-Z][^,.;()\n]+?)(?:\s*[,.]|\n)", re.I),
        # "credited to Alexander Graham Bell"
        re.compile(r"credited to\s+([A-Z][^,.;()\n]+?)(?:\s*[,.]|\n)", re.I),
        # "attributed to Alexander Graham Bell"
        re.compile(r"attributed to\s+([A-Z][^,.;()\n]+?)(?:\s*[,.]|\n)", re.I),
        # "Alexander Graham Bell, who invented the telephone"
        re.compile(r"([A-Z][A-Za-z\s]+?),\s+who\s+(?:invented|discovered)", re.I),
        # "Bell invented …" — matches "X invented"
        re.compile(r"([A-Z][A-Za-z\s]{2,30}?)\s+invented\b", re.I),
        # "credited with inventing the telephone by Alexander Graham Bell"
        re.compile(r"credited with inventing[^.]*?(?:by|to)\s+([A-Z][^,.;()\n]+?)(?:\s*[,.]|\n)", re.I),
    ],

    # ── Authorship ────────────────────────────────────────────────────────
    ("AUTHORED_BY", "*"): [
        re.compile(r"(?:written|authored) by\s+([A-Z][^,.;()\n]+?)(?:\s*[,.]|\n)", re.I),
        re.compile(r"author[:\s]+([A-Z][^,.;()\n]+?)(?:\s*[,.]|\n)", re.I),
    ],
}


def _get_patterns(relationship: str, question_type: str) -> list[re.Pattern]:
    """Return extraction patterns for (relationship, question_type),
    falling back to (relationship, "*") wildcard."""
    exact = _REL_EXTRACTION.get((relationship, question_type), [])
    if exact:
        return exact
    return _REL_EXTRACTION.get((relationship, "*"), [])


# ── Nationality adjective → country name map ──────────────────────────────────
# Used to normalise "German" → "Germany", "Polish" → "Poland" etc.
# when the only birthplace evidence is a Wikidata "X-born" adjective.
# General-purpose: covers common nationalities. NOT specific to any person.
_NATIONALITY_TO_COUNTRY: dict[str, str] = {
    "american":    "the United States",
    "argentinian": "Argentina",
    "australian":  "Australia",
    "austrian":    "Austria",
    "belgian":     "Belgium",
    "brazilian":   "Brazil",
    "british":     "the United Kingdom",
    "canadian":    "Canada",
    "chinese":     "China",
    "czech":       "the Czech Republic",
    "danish":      "Denmark",
    "dutch":       "the Netherlands",
    "egyptian":    "Egypt",
    "english":     "England",
    "finnish":     "Finland",
    "french":      "France",
    "german":      "Germany",
    "greek":       "Greece",
    "hungarian":   "Hungary",
    "indian":      "India",
    "iranian":     "Iran",
    "irish":       "Ireland",
    "israeli":     "Israel",
    "italian":     "Italy",
    "japanese":    "Japan",
    "korean":      "Korea",
    "mexican":     "Mexico",
    "norwegian":   "Norway",
    "pakistani":   "Pakistan",
    "peruvian":    "Peru",
    "polish":      "Poland",
    "portuguese":  "Portugal",
    "romanian":    "Romania",
    "russian":     "Russia",
    "scottish":    "Scotland",
    "south african": "South Africa",
    "spanish":     "Spain",
    "swedish":     "Sweden",
    "swiss":       "Switzerland",
    "turkish":     "Turkey",
    "ukrainian":   "Ukraine",
    "welsh":       "Wales",
}


def _normalise_extracted_value(value: str, relationship: str) -> str:
    """Post-process an extracted value to make it readable.

    For BORN_IN, if the extracted value is a nationality adjective
    (e.g. "German", "Polish"), map it to the country name so the
    corrected sentence reads "born in Germany" not "born in German".
    """
    if relationship != "BORN_IN":
        return value
    mapped = _NATIONALITY_TO_COUNTRY.get(value.lower().strip())
    return mapped if mapped else value


def extract_answer_from_evidence(
    evidence_text: str,
    question_relationship: str,
    subject_entity: str,
    question_type: str = "*",
) -> str | None:
    """Extract the specific answer value from evidence text.

    Uses (relationship, question_type) to pick the right patterns — so
    FOUNDED_BY + DATE extracts "1975", not "Bill Gates".

    Returns the extracted value or None on failure.
    """
    if not evidence_text or not question_relationship:
        return None

    patterns = _get_patterns(question_relationship, question_type)
    for pat in patterns:
        m = pat.search(evidence_text)
        if m:
            value = m.group(1).strip(" ,.()")
            if 1 < len(value) < 80 and not value.endswith(" and"):
                return _normalise_extracted_value(value, question_relationship)

    return None


def extract_biographical_sentence(
    evidence_text: str,
    subject_entity: str,
) -> str | None:
    """Extract a concise biographical description of the subject from evidence.

    Looks for sentences that contain the subject name and a descriptive
    predicate (was a …, is a …).  Returns the first such sentence, or None.
    """
    if not evidence_text or not subject_entity:
        return None

    subj_l = subject_entity.lower()
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', evidence_text.strip())
    for sent in sentences:
        sent_l = sent.lower()
        # Sentence must mention the subject
        subj_tokens = [t for t in subj_l.split() if len(t) >= 4]
        if not any(t in sent_l for t in subj_tokens):
            continue
        # Sentence must have a descriptive copula ("was a", "is a", "was an")
        if re.search(r'\bwas\s+(?:a|an|the)\b|\bis\s+(?:a|an|the)\b', sent, re.I):
            # Keep it short (≤200 chars)
            return sent.strip()[:200]

    return None


def _first_sentence(text: str) -> str:
    """Return the first complete sentence from text."""
    m = re.search(r'\.(?:\s+[A-Z]|\s*\n)', text)
    if m:
        return text[: m.start() + 1].strip()
    return text[:250].strip()


def _evidence_is_usable(evidence_text: str | None, subject_entity: str = "") -> bool:
    if not evidence_text or not evidence_text.strip():
        return False
    evid_l = evidence_text.lower()
    for pat in _USELESS_EVIDENCE_PATTERNS:
        if pat.search(evidence_text[:200]):
            return False
    if subject_entity:
        subj_l = subject_entity.lower().strip()
        tokens = [t for t in subj_l.split() if len(t) >= 4]
        if len(tokens) >= 2:
            if not all(t in evid_l for t in tokens):
                return False
        elif tokens:
            if tokens[0] not in evid_l:
                return False
    return True


def _evidence_contains_answer_info(
    evidence_text: str,
    subject_entity: str,
    question_type: str = "",
    question_relationship: str = "",
) -> bool:
    """Check whether the evidence snippet is likely to contain the answer."""
    if not evidence_text:
        return False
    evid_l = evidence_text.lower()

    if question_type in ("LOCATION",):
        loc_signals = [
            "located in", "born in", "died in", "capital of", "capital is",
            "capital city", "based in", "headquartered in", "situated in",
            "in the city", " is a city", " is a country", " is a nation",
            "champ de mars", "the capital", "-born",
        ]
        if any(sig in evid_l for sig in loc_signals):
            return True
        try:
            from app.nlp.entity_extractor import extract_entities
            ents = extract_entities(evidence_text)
            return bool(ents.locations)
        except Exception:
            return True

    if question_type in ("PERSON",):
        try:
            from app.nlp.entity_extractor import extract_entities
            ents = extract_entities(evidence_text)
            return bool(ents.people)
        except Exception:
            pass
        return True

    if question_type in ("DATE",):
        from app.nlp.entity_extractor import extract_years
        return bool(extract_years(evid_l))

    return True


# ── Sentence templates ─────────────────────────────────────────────────────────
# Keyed by (relationship, question_type).  Wildcards ("*") used as fallback.

_REL_TEMPLATES: dict[tuple, str] = {
    ("BORN_IN",       "*"):      "{subject} was born in {value}.",
    ("DIED_IN",       "*"):      "{subject} died in {value}.",
    ("LOCATED_IN",    "*"):      "{subject} is located in {value}.",
    ("CAPITAL_OF",    "*"):      "The capital of {subject} is {value}.",
    ("FOUNDED_BY",    "PERSON"): "{subject} was founded by {value}.",
    ("FOUNDED_BY",    "DATE"):   "{subject} was founded in {value}.",
    ("CEO_OF",        "*"):      "The CEO of {subject} is {value}.",
    ("PRESIDENT_OF",  "*"):      "{value} was the president of {subject}.",
    ("DISCOVERED_BY", "*"):      "{subject} was invented/discovered by {value}.",
    ("AUTHORED_BY",   "*"):      "{subject} was written by {value}.",
}


def _get_template(relationship: str, question_type: str) -> str | None:
    exact = _REL_TEMPLATES.get((relationship, question_type))
    if exact:
        return exact
    return _REL_TEMPLATES.get((relationship, "*"))


def _build_correction_sentence(
    subject: str,
    relationship: str,
    question_type: str,
    extracted_value: str,
    fallback_evidence: str,
) -> str:
    """Build a clean, specific corrected sentence."""
    template = _get_template(relationship, question_type)
    if template and extracted_value:
        return template.format(subject=subject, value=extracted_value)
    return _first_sentence(fallback_evidence)


# ── Main corrector ─────────────────────────────────────────────────────────────

@dataclass
class ClaimCorrectionInput:
    claim_text: str
    verdict: str
    best_evidence_text: str | None
    entity_mismatch: bool = field(default=False)
    subject_entity: str = field(default="")
    question_type: str = field(default="")
    question_relationship: str = field(default="")
    # All available evidence texts — corrector tries each for extraction
    all_evidence_texts: list[str] = field(default_factory=list)


def generate_corrected_answer(claim_inputs: list[ClaimCorrectionInput]) -> str:
    """Produce a corrected answer string from per-claim inputs."""
    if not claim_inputs:
        return (
            "No claims were extracted from the answer, so no corrected "
            "answer could be generated."
        )

    kept_sentences:   list[str] = []
    corrected_notes:  list[str] = []
    unresolved_notes: list[str] = []

    for item in claim_inputs:
        evid     = item.best_evidence_text or ""
        rel      = item.question_relationship
        qtype    = item.question_type
        subject  = item.subject_entity

        # ── VERIFIED ──────────────────────────────────────────────────────
        if item.verdict == VERDICT_VERIFIED:
            kept_sentences.append(item.claim_text)

        # ── PARTIALLY_SUPPORTED ──────────────────────────────────────────
        elif item.verdict == VERDICT_PARTIALLY_SUPPORTED:
            if item.entity_mismatch:
                unresolved_notes.append(
                    f'Removed: "{item.claim_text}" — partially supported but a '
                    "key entity was not found in the evidence (could not be verified)."
                )
            else:
                kept_sentences.append(item.claim_text + _PARTIAL_DISCLAIMER)
                corrected_notes.append(
                    f'Partial: "{item.claim_text}" is only partially supported '
                    "and has not been independently verified."
                )

        # ── CONTRADICTED or UNSUPPORTED ──────────────────────────────────
        elif item.verdict in (VERDICT_CONTRADICTED, VERDICT_UNSUPPORTED):
            # Try extraction from ALL evidence items, not just best_evidence_text.
            # This is critical: the highest-similarity item may be a generic
            # article (e.g. a telephone description) while another item from
            # Wikidata or DuckDuckGo directly names the inventor/birthplace.
            all_evids = list(dict.fromkeys(filter(None, [
                item.best_evidence_text
            ] + (item.all_evidence_texts or []))))

            correction = None
            for evid_candidate in all_evids:
                correction = _try_build_correction(
                    evid_candidate, rel, qtype, subject
                )
                if correction:
                    break

            if correction:
                kept_sentences.append(correction)
                corrected_notes.append(
                    f'Corrected: "{item.claim_text}" → "{correction}"'
                )
            else:
                unresolved_notes.append(
                    f'Removed: "{item.claim_text}" — '
                    + ("contradicted but " if item.verdict == VERDICT_CONTRADICTED
                       else "unsupported and ")
                    + "retrieved evidence was insufficient for a reliable correction. "
                    "The original answer could not be independently verified."
                )

    corrected_answer = " ".join(kept_sentences).strip()

    if not corrected_answer:
        corrected_answer = (
            "The available evidence was insufficient to confirm any part of "
            "the original answer.  The original answer could not be verified."
        )

    notes = corrected_notes + unresolved_notes
    if notes:
        corrected_answer += "\n\n[Corrections applied]\n" + "\n".join(notes)

    return corrected_answer


def _try_build_correction(
    evid: str,
    relationship: str,
    question_type: str,
    subject: str,
) -> str | None:
    """Attempt to build a specific correction sentence from evidence.

    Returns a clean string or None if the evidence is unusable or
    insufficient to answer the question reliably.
    """
    if not _evidence_is_usable(evid, subject):
        return None

    if not _evidence_contains_answer_info(evid, subject, question_type, relationship):
        return None

    # ── Biographical questions ────────────────────────────────────────────
    if question_type == "BIOGRAPHICAL":
        bio = extract_biographical_sentence(evid, subject)
        if bio:
            return bio
        # Fall back to first sentence only if it mentions the subject
        fs = _first_sentence(evid)
        subj_l = subject.lower()
        if any(t in fs.lower() for t in subj_l.split() if len(t) >= 4):
            return fs
        return None

    # ── Structured relationship questions ─────────────────────────────────
    extracted = extract_answer_from_evidence(evid, relationship, subject, question_type)
    if extracted:
        return _build_correction_sentence(subject, relationship, question_type,
                                          extracted, evid)

    # ── No specific value extracted ────────────────────────────────────────
    # When no relationship is known, fall back to first sentence (old behaviour).
    if not relationship:
        return _first_sentence(evid)

    # When relationship IS known, only use the first sentence if it actually
    # answers the question (avoids "France, officially..." for capital query).
    if _first_sentence_answers_question(evid, relationship, question_type, subject):
        return _first_sentence(evid)

    return None


def _first_sentence_answers_question(
    evidence_text: str,
    relationship: str,
    question_type: str,
    subject: str,
) -> bool:
    """Return True only when the first sentence of the evidence is specific
    enough to answer the question — i.e. it contains both the subject AND
    a relationship keyword.

    This prevents returning the generic France intro as the answer to
    "What is the capital of France?"
    """
    fs = _first_sentence(evidence_text).lower()
    subj_l = subject.lower()

    # Subject must appear
    if not any(t in fs for t in subj_l.split() if len(t) >= 4):
        return False

    # Relationship keyword must appear
    rel_signals: dict[str, list[str]] = {
        "BORN_IN":       ["born in", "born on", "birthplace"],
        "DIED_IN":       ["died in", "died on"],
        "LOCATED_IN":    ["located in", "situated in", "in paris", "in london",
                          "champ de mars"],
        "CAPITAL_OF":    ["capital", "capital city"],
        "FOUNDED_BY":    ["founded by", "co-founded", "established by"],
        "FOUNDED_BY_DATE": ["founded in", "founded on"],
        "CEO_OF":        ["ceo", "chief executive"],
        "PRESIDENT_OF":  ["president"],
        "DISCOVERED_BY": ["invented by", "discovered by", "credited to",
                          "attributed to"],
        "AUTHORED_BY":   ["written by", "authored by"],
        "BIOGRAPHICAL":  ["was a", "is a", "was an", "is an"],
    }
    key = relationship if question_type != "DATE" else f"{relationship}_DATE"
    signals = rel_signals.get(key, rel_signals.get(relationship, []))
    return any(sig in fs for sig in signals)
