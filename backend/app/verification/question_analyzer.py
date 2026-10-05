"""General-purpose question analysis.

Parses a natural-language factual question and produces structured metadata:
  - question_type       : what kind of answer is expected (PERSON, LOCATION, DATE, …)
  - relationship        : the specific predicate being asked (BORN_IN, FOUNDED_BY, …)
  - focus_entity        : the subject the question is about
  - answer_template     : a sentence template for expanding short answers
  - relationship_label  : human-readable label for explanations

DESIGN PRINCIPLES
-----------------
• Zero hard-coded entity names, answer facts, or domain-specific knowledge.
• Classification is based purely on question structure (wh-word, verb phrase,
  keyword patterns) — it generalises to arbitrary factual questions.
• Extensible: adding a new question pattern requires only one new tuple in
  _PATTERNS; no logic changes.
• Falls back gracefully to GENERIC when no pattern matches, preserving the
  existing similarity-based verification behaviour.

DOCUMENTED LIMITATIONS
-----------------------
• Regex-based pattern matching; unusual phrasings may not match.
• Focus-entity extraction is heuristic (NER + text stripping).
• Temporal/numerical questions are detected but value comparison is limited.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── Answer-type constants ─────────────────────────────────────────────────────
QT_PERSON   = "PERSON"
QT_LOCATION = "LOCATION"
QT_DATE     = "DATE"
QT_ORG      = "ORGANIZATION"
QT_NUMBER   = "NUMBER"
QT_DISCOVERY= "DISCOVERY"
QT_THING    = "THING"
QT_BIO      = "BIOGRAPHICAL"
QT_GENERIC  = "GENERIC"

# ── Relationship constants ────────────────────────────────────────────────────
REL_BORN_IN       = "BORN_IN"
REL_DIED_IN       = "DIED_IN"
REL_FOUNDED_BY    = "FOUNDED_BY"
REL_CEO_OF        = "CEO_OF"
REL_LOCATED_IN    = "LOCATED_IN"
REL_DISCOVERED    = "DISCOVERED_BY"
REL_AUTHORED_BY   = "AUTHORED_BY"
REL_PRESIDENT_OF  = "PRESIDENT_OF"
REL_CAPITAL_OF    = "CAPITAL_OF"
REL_BIO_DESCRIPTION = "BIOGRAPHICAL_DESCRIPTION"
REL_GENERIC       = "GENERIC"


@dataclass
class QuestionAnalysis:
    """Structured interpretation of a natural-language factual question."""
    question_type: str
    relationship: str
    focus_entity: str
    # Template: use {answer} and {focus} as placeholders.
    # E.g. "{focus} was born in {answer}."
    answer_template: str
    relationship_label: str


# ── Pattern table ─────────────────────────────────────────────────────────────
# Tuples: (compiled_regex, question_type, relationship, answer_template, label)
# Checked in order; first match wins.
# NO entity names appear here — only linguistic structure.

_P = re.compile  # shorthand

_PATTERNS: list[tuple] = [
    # ── Birthplace ──────────────────────────────────────────────────────────
    (_P(r"\bwhere\b.{0,40}\bborn\b", re.I),
     QT_LOCATION, REL_BORN_IN,
     "{focus} was born in {answer}.",
     "born in"),

    # ── Birth date ──────────────────────────────────────────────────────────
    (_P(r"\bwhen\b.{0,40}\bborn\b|\bdate of birth\b|\bbirthdate\b", re.I),
     QT_DATE, REL_BORN_IN,
     "{focus} was born on {answer}.",
     "date of birth"),

    # ── Death place ─────────────────────────────────────────────────────────
    (_P(r"\bwhere\b.{0,40}(died|death)\b", re.I),
     QT_LOCATION, REL_DIED_IN,
     "{focus} died in {answer}.",
     "died in"),

    # ── Death date ──────────────────────────────────────────────────────────
    (_P(r"\bwhen\b.{0,40}(died|death)\b|\bdate of death\b", re.I),
     QT_DATE, REL_DIED_IN,
     "{focus} died in {answer}.",
     "date of death"),

    # ── Founder ─────────────────────────────────────────────────────────────
    (_P(r"\bwho\b.{0,60}\b(found(ed|er|ers|ing)|co.?found(ed|er))\b", re.I),
     QT_PERSON, REL_FOUNDED_BY,
     "{focus} was founded by {answer}.",
     "founded by"),

    # ── Founding date ───────────────────────────────────────────────────────
    (_P(r"\bwhen\b.{0,60}\bfound(ed|ing)\b|\bfounding date\b", re.I),
     QT_DATE, REL_FOUNDED_BY,
     "{focus} was founded in {answer}.",
     "founding date"),

    # ── CEO / executive ─────────────────────────────────────────────────────
    (_P(r"\bwho\b.{0,60}\b(ceo|chief executive|head(s)?|leads|led)\b", re.I),
     QT_PERSON, REL_CEO_OF,
     "The CEO of {focus} is {answer}.",
     "CEO of"),

    # ── First president ─────────────────────────────────────────────────────
    (_P(r"\bwho\b.{0,40}\bfirst\b.{0,30}\bpresident\b", re.I),
     QT_PERSON, REL_PRESIDENT_OF,
     "The first president of {focus} was {answer}.",
     "first president of"),

    # ── President ───────────────────────────────────────────────────────────
    (_P(r"\bwho\b.{0,60}\bpresident\b", re.I),
     QT_PERSON, REL_PRESIDENT_OF,
     "The president of {focus} was {answer}.",
     "president of"),

    # ── Location / headquarters ─────────────────────────────────────────────
    (_P(r"\bwhere\b.{0,60}\b(is|are|was|were|located|situated|headquartered|based)\b", re.I),
     QT_LOCATION, REL_LOCATED_IN,
     "{focus} is located in {answer}.",
     "located in"),

    # ── Capital ─────────────────────────────────────────────────────────────
    (_P(r"\bwhat\b.{0,40}\bcapital\b", re.I),
     QT_LOCATION, REL_CAPITAL_OF,
     "The capital of {focus} is {answer}.",
     "capital of"),

    # ── Discovery / invention ───────────────────────────────────────────────
    (_P(r"\bwho\b.{0,60}\b(discover(ed)?|invent(ed)?|develop(ed)?|formul(ated)?)\b", re.I),
     QT_PERSON, REL_DISCOVERED,
     "{focus} was discovered/invented by {answer}.",
     "discovered by"),

    (_P(r"\bwhat\b.{0,60}\b(discover(ed)?|invent(ed)?)\b", re.I),
     QT_DISCOVERY, REL_DISCOVERED,
     "{focus} discovered/invented {answer}.",
     "discovered"),

    # ── Authorship ──────────────────────────────────────────────────────────
    (_P(r"\bwho\b.{0,60}\b(wrote|written|authored)\b", re.I),
     QT_PERSON, REL_AUTHORED_BY,
     "{focus} was written by {answer}.",
     "written by"),

    # ── Biographical / identity ("who was/is X") ────────────────────────────
    # Must come AFTER all more specific "who" patterns
    (_P(r"\bwho\b.{0,20}\b(was|is|were|are)\b.{0,60}", re.I),
     QT_BIO, REL_BIO_DESCRIPTION,
     "{answer} is/was {focus}.",
     "biographical description"),

    # ── Generic "who" ───────────────────────────────────────────────────────
    (_P(r"^\s*who\b", re.I),
     QT_PERSON, REL_GENERIC,
     "{answer} is associated with {focus}.",
     "person"),

    # ── Generic "where" ─────────────────────────────────────────────────────
    (_P(r"^\s*where\b", re.I),
     QT_LOCATION, REL_LOCATED_IN,
     "{focus} is in {answer}.",
     "location"),

    # ── Generic "when" ──────────────────────────────────────────────────────
    (_P(r"^\s*when\b", re.I),
     QT_DATE, REL_GENERIC,
     "{focus} occurred in {answer}.",
     "date"),

    # ── How many / much ─────────────────────────────────────────────────────
    (_P(r"\bhow (many|much|tall|long|far|old|big)\b", re.I),
     QT_NUMBER, REL_GENERIC,
     "{focus} is {answer}.",
     "quantity"),

    # ── Generic "what" ──────────────────────────────────────────────────────
    (_P(r"^\s*what\b", re.I),
     QT_THING, REL_GENERIC,
     "{focus} is {answer}.",
     "description"),
]


def analyze_question(question: str) -> QuestionAnalysis:
    """Parse a question and return structured metadata.

    Always returns a result; falls back to GENERIC when no pattern matches.
    """
    q = (question or "").strip()
    focus = _extract_focus_entity(q)

    for pattern, q_type, rel, template, label in _PATTERNS:
        if pattern.search(q):
            return QuestionAnalysis(
                question_type=q_type,
                relationship=rel,
                focus_entity=focus,
                answer_template=template,
                relationship_label=label,
            )

    return QuestionAnalysis(
        question_type=QT_GENERIC,
        relationship=REL_GENERIC,
        focus_entity=focus,
        answer_template="{focus} is related to {answer}.",
        relationship_label="general",
    )


def build_claim_from_question(question: str, answer: str) -> str:
    """Construct a full declarative claim by combining question context with
    the answer value.

    Used for SHORT answers ("Paris", "1879", "Bill Gates") that cannot be
    verified standalone.

    Examples
    --------
    question="Where is the Eiffel Tower located?", answer="Paris"
    → "The Eiffel Tower is located in Paris."

    question="When was Microsoft founded?", answer="1975"
    → "Microsoft was founded in 1975."
    """
    answer = (answer or "").strip()
    if not answer:
        return ""

    qa = analyze_question(question)
    focus = qa.focus_entity or "the subject"

    try:
        claim = qa.answer_template.format(focus=focus, answer=answer)
    except (KeyError, IndexError):
        claim = f"{focus} is {answer}."
    return claim.strip()


# ── Focus entity extraction ───────────────────────────────────────────────────

_WH_STRIP = re.compile(
    r"^(who|what|where|when|which|whose|whom|how)\b[\s,]*(was|is|were|are|did|does|do)?\s*",
    re.I,
)
_TRAILING_STRIP = re.compile(r"\?$")
_AUX_STRIP = re.compile(
    r"^(of|for|about|by|with|to|from|in|on|at|the|a|an)\b\s*",
    re.I,
)
_VERB_BOUNDARY = re.compile(
    r"\b(was|is|were|are|did|does|do|has|have|had|born|died|located|founded|discovered|"
    r"led|heads|wrote|built|created|established|invented|formulated|won|received)\b",
    re.I,
)


def _extract_focus_entity(question: str) -> str:
    """Heuristically extract the subject entity from a question.

    Priority:
    1. spaCy NER — most reliable for correctly-spelled known entities.
       BUT: if NER returns only a single short word (≤8 chars) that could
       be the first name of a misspelled multi-word name, we fall through
       to the full-text extractor so the resolver can see the whole name.
    2. Full text after stripping the wh-word + auxiliary block.
    3. Fuzzy resolution (Wikipedia search) applied to the full candidate.
    """
    # Strip wh+aux block once — used by both NER fallback and text path
    cleaned = _WH_STRIP.sub("", question)
    cleaned = _TRAILING_STRIP.sub("", cleaned).strip()
    cleaned = _AUX_STRIP.sub("", cleaned).strip()

    # ── 1. spaCy NER ──────────────────────────────────────────────────────
    try:
        from app.nlp.claim_extractor import _get_nlp
        nlp = _get_nlp()
        doc = nlp(question)
        preferred_labels = {"ORG", "PERSON", "GPE", "LOC", "FAC", "WORK_OF_ART",
                            "NORP", "EVENT", "PRODUCT"}
        ents = [e for e in doc.ents if e.label_ in preferred_labels]
        if ents:
            entity = ents[0].text.strip()
            # Accept full multi-word entities directly (resolver still applied
            # for minor typos like "Einstien" in multi-word names).
            # REJECT single short words when the cleaned question has more text
            # that looks like additional name tokens — this handles "albert"
            # from "who was albert einstien" (spaCy only tags "albert").
            is_short_first_name = (
                len(entity.split()) == 1
                and len(entity) <= 8
                and cleaned.lower().startswith(entity.lower())
                and len(cleaned) > len(entity) + 1
            )
            if not is_short_first_name:
                return _maybe_resolve_entity(entity)
            # Fall through: treat whole cleaned text as candidate
    except Exception:
        pass

    # ── 2. Full text candidate (up to verb boundary) ───────────────────────
    m = _VERB_BOUNDARY.search(cleaned)
    if m and m.start() > 0:
        candidate = cleaned[: m.start()].strip(" ,.")
        if len(candidate) >= 2:
            return _maybe_resolve_entity(candidate)

    # ── 3. Whole cleaned text ──────────────────────────────────────────────
    raw = cleaned[:60].strip(" ,?.")
    return _maybe_resolve_entity(raw)


# ── Fuzzy entity resolution ───────────────────────────────────────────────────

def _maybe_resolve_entity(candidate: str) -> str:
    """Attempt to resolve a possibly misspelled entity name using Wikidata
    search as a spelling oracle.

    Strategy:
    - Query Wikidata's wbsearchentities for the candidate string.
    - Compare the returned label to the candidate using character-level
      similarity (edit distance ratio).
    - If the top result is sufficiently close (>= 0.6 similarity) AND
      differs from the candidate (i.e. looks like a correction), return
      the canonical form.
    - Otherwise return the candidate unchanged.

    This is general-purpose: it does not know about specific people.
    It relies on Wikidata's own spelling correction and entity search.

    DOCUMENTED LIMITATIONS:
    - Requires network access; falls back silently on failure.
    - Only corrects towards Wikidata's top result — ambiguous cases
      (multiple plausible candidates) are not resolved.
    - Edit-distance threshold is heuristic.
    """
    if not candidate or len(candidate) < 4:
        return candidate
    # Only try resolution for multi-word or clearly name-like candidates
    # (skip very short single words like "who" residuals)
    if len(candidate.split()) < 1:
        return candidate
    try:
        resolved = _wikidata_resolve(candidate)
        if resolved and resolved.lower() != candidate.lower():
            return resolved
    except Exception:
        pass
    return candidate


def _wikidata_resolve(candidate: str) -> str | None:
    """Query Wikipedia's search API (which handles typos better than Wikidata)
    and return the canonical article title if it is a plausible spelling
    correction of *candidate*.

    Returns None when no confident resolution is found.
    """
    import requests
    from difflib import SequenceMatcher

    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": candidate,
                "format": "json",
                "srlimit": 3,
                "srprop": "snippet",
            },
            headers={
                "User-Agent": (
                    "HallucinationDetectionBot/1.0 "
                    "(Hallucination Detection & Answer Verification project)"
                )
            },
            timeout=6,
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
    except Exception:
        return None

    if not results:
        return None

    # Check each result's title for similarity to candidate
    cand_l = candidate.lower()
    for item in results:
        title = item.get("title", "")
        if not title:
            continue
        # Direct token-overlap check (works for typos like "einstien" → "Einstein")
        if _token_overlap_sufficient(candidate, title):
            return title
        # Fallback: char-similarity ratio
        sim = SequenceMatcher(None, cand_l, title.lower()).ratio()
        if 0.55 <= sim < 1.0:
            return title

    return None


def _char_similarity(a: str, b: str) -> float:
    """Character-level similarity in [0,1] using the Ratcliff/Obershelp
    algorithm approximated via difflib.SequenceMatcher.
    """
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _token_overlap_sufficient(candidate: str, label: str) -> bool:
    """Return True if every token in *candidate* fuzzy-matches a token
    in *label* with edit-distance ratio >= 0.75."""
    from difflib import SequenceMatcher

    c_tokens = candidate.lower().split()
    l_tokens = label.lower().split()

    if not c_tokens or not l_tokens:
        return False

    for ct in c_tokens:
        if len(ct) < 3:
            continue  # skip noise like "a", "of"
        best = max(
            SequenceMatcher(None, ct, lt).ratio()
            for lt in l_tokens
        )
        if best < 0.75:
            return False
    return True
