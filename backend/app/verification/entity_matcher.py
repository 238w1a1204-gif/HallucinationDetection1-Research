"""Entity/date/organization/role consistency checking.

Checks that the KEY entities and the claimed relationship mentioned in the
claim are actually present in the evidence.

DOCUMENTED LIMITATIONS:
- Uses spaCy en_core_web_sm which has moderate NER accuracy.  Named entities
  in short claims can be mis-tagged or missed entirely.
- Fuzzy substring matching is intentionally liberal to tolerate
  "Einstein" vs "Albert Einstein"; this can produce false passes for very
  short common names.
- Relationship keyword matching is keyword-list-based, not deep NLP parsing.
  It will miss unusual phrasings.
- The keyword dictionary covers common relationship categories; rare or
  highly domain-specific relationships are not covered.

Design principle: NO hard-coded entity names (people, companies, places,
etc.).  All generalisation comes from linguistic patterns (NER labels,
keyword categories) that work across arbitrary entities.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.nlp.entity_extractor import extract_entities, extract_years

# ── Relationship-category keyword dictionary ─────────────────────────────────
#
# Each category maps a human-readable relationship type to a list of keyword
# strings that signal that relationship in natural language text.
#
# Rules for extending this dictionary:
# 1. Add a new category key (snake_case string).
# 2. Provide at least 2 keyword phrases that clearly signal the relationship.
# 3. Keywords are matched as substrings of the lowercased text, so shorter
#    phrases match more broadly — prefer specific phrases over single words.
# 4. Do NOT add individual entity names (people/companies/places) here.
#    Only add linguistic predicates that generalise across entities.
#
# The dictionary is intentionally kept here (not in relationship_checker.py)
# so that entity_matcher can use the same vocabulary for its
# extract_role_keywords() utility without a circular import.

ROLE_KEYWORDS: dict[str, list[str]] = {
    # ── Leadership & executive roles ──────────────────────────────────────
    "leadership": [
        "ceo", "chief executive", "chief executive officer",
        "president", "chairman", "chairwoman", "chairperson",
        "managing director", "md", "executive director",
        "cto", "chief technology officer",
        "cfo", "chief financial officer",
        "coo", "chief operating officer",
        "director", "vice president", "vp",
        "head of", "leads", "led", "lead",
        "prime minister", "premier", "chancellor",
        "governor", "mayor", "secretary general",
    ],

    # ── Founding & creation ───────────────────────────────────────────────
    "founding": [
        "founded", "co-founded", "cofounded",
        "established", "co-established",
        "created", "co-created",
        "started", "co-started",
        "launched", "co-launched",
        "built", "set up",
        "founder", "co-founder", "cofounders", "founders",
    ],

    # ── Birth & origin ────────────────────────────────────────────────────
    "birth": [
        "born", "birth", "birthplace",
        "native of", "grew up in",
        "birthdate", "date of birth", "dob",
        "nativity",
    ],

    # ── Death & end of life ───────────────────────────────────────────────
    "death": [
        "died", "death", "passed away", "deceased",
        "date of death", "died in", "death in",
        "killed", "executed", "perished",
    ],

    # ── Location / headquarters / geography ──────────────────────────────
    "location": [
        "located", "location", "located in", "location in",
        "headquartered", "headquarters", "based in",
        "situated", "situated in",
        "resides", "residing", "resides in",
        "lives in", "lived in",
        "capital of", "is the capital",
        "in the country", "in the city",
    ],

    # ── Scientific & academic discovery / invention ───────────────────────
    "discovery": [
        "discovered", "discovery", "discovered by",
        "invented", "invention", "invented by",
        "developed", "developed by",
        "proposed", "proposed by",
        "formulated", "formulated by",
        "coined", "coined by",
        "pioneered", "pioneered by",
        "theorised", "theorized",
        "devised", "devised by",
        "composed by",    # "composed" alone is too generic (e.g. "composed of")
    ],

    # ── Authorship & creation of works ───────────────────────────────────
    "authorship": [
        "wrote", "written by", "author of", "authored",
        "composed by",    # "composed" alone is too generic (e.g. "composed of")
        "composer",
        "directed", "directed by", "director of",
        "produced", "produced by", "producer of",
        "published", "published by",
        "painted", "painted by", "artist",
        "designed", "designed by", "designer",
    ],

    # ── Political rule & governance ───────────────────────────────────────
    "rule": [
        "ruled", "reign", "reigned",
        "governed", "governed by",
        "king of", "queen of", "emperor of", "empress of",
        "king", "queen", "emperor", "empress",
        "monarch", "monarchy",
        "pharaoh", "sultan", "tsar", "tsarina",
    ],

    # ── Employment & membership ───────────────────────────────────────────
    "employment": [
        "works for", "worked for",
        "employed by", "employed at",
        "works at", "working at",
        "member of", "member",
        "joined", "belongs to",
        "affiliated with", "affiliated",
        "served", "served as",
        "served in",
    ],

    # ── Ownership & possession ────────────────────────────────────────────
    "ownership": [
        "owns", "owned by", "owner of",
        "acquired", "acquired by",
        "subsidiary of", "parent company",
        "belongs to", "property of",
        "controlled by",
    ],

    # ── Nationality & citizenship ─────────────────────────────────────────
    "nationality": [
        "citizen of", "nationality",
        "national of", "naturalized",
        "passport",
    ],

    # ── Award & recognition ───────────────────────────────────────────────
    "award": [
        "won", "awarded", "received",
        "nobel prize", "prize", "award",
        "medal", "honor", "honour",
        "laureate", "recipient",
    ],

    # ── Occurrence & events ───────────────────────────────────────────────
    "event": [
        "occurred", "happened", "took place",
        "event", "incident",
        "battle of", "war of",
        "signed", "ratified",
    ],
}

# Flat set for quick membership tests (used by extract_role_keywords)
ALL_ROLE_KEYWORDS: set[str] = {
    kw for kws in ROLE_KEYWORDS.values() for kw in kws
}


@dataclass
class EntityCheckResult:
    mismatch: bool
    reason: str
    # Score in [0.0, 1.0] — fraction of claim entities found in evidence
    entity_presence_score: float = 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def check_entity_consistency(claim: str, evidence_text: str) -> EntityCheckResult:
    """Return an EntityCheckResult describing how well the evidence's
    entities match those in the claim.

    Decision order (each check is general — no entity names are hard-coded):
    1. Year mismatch: both texts have years and they differ → mismatch
    2. Person check: a person named in the claim cannot be found anywhere
       in the evidence (fixed: no longer requires the evidence to also have
       people — that was the original empty-evidence-people bug)
    3. Organization check: ALL claimed organizations absent from evidence
    4. Otherwise → no mismatch; attach entity_presence_score
    """
    evidence_l = (evidence_text or "").lower()

    # ── 1. Year check ─────────────────────────────────────────────────────
    claim_years = extract_years(claim)
    evidence_years = extract_years(evidence_text)
    if claim_years and evidence_years and not (claim_years & evidence_years):
        score = _compute_entity_presence_score(claim, evidence_text)
        return EntityCheckResult(
            mismatch=True,
            reason=(
                f"Claim mentions year(s) {sorted(claim_years)} but evidence mentions "
                f"different year(s) {sorted(evidence_years)}."
            ),
            entity_presence_score=score,
        )

    claim_entities = extract_entities(claim)
    evidence_entities = extract_entities(evidence_text)

    # ── 2. Person check ───────────────────────────────────────────────────
    # No longer guarded by `evidence_entities.people` — if the claim names a
    # person and that person doesn't appear anywhere in the evidence text
    # (by NER or plain substring), that's a mismatch.
    if claim_entities.people:
        missing: set[str] = set()
        for person in claim_entities.people:
            if not _entity_in_text(person, evidence_entities.people, evidence_l):
                missing.add(person)
        if missing:
            score = _compute_entity_presence_score(claim, evidence_text)
            return EntityCheckResult(
                mismatch=True,
                reason=(
                    f"Claim refers to person(s) {sorted(missing)} which "
                    "could not be found in the evidence."
                ),
                entity_presence_score=score,
            )

    # ── 3. Organization check ─────────────────────────────────────────────
    # Only flag as mismatch when ALL claimed orgs are absent — if at least
    # one org matches the evidence is at least partly relevant.
    if claim_entities.organizations:
        missing_orgs: set[str] = set()
        for org in claim_entities.organizations:
            if not _entity_in_text(org, evidence_entities.organizations, evidence_l):
                missing_orgs.add(org)
        if missing_orgs and len(missing_orgs) == len(claim_entities.organizations):
            score = _compute_entity_presence_score(claim, evidence_text)
            return EntityCheckResult(
                mismatch=True,
                reason=(
                    f"Claim refers to organization(s) {sorted(missing_orgs)} which "
                    "could not be found in the evidence."
                ),
                entity_presence_score=score,
            )

    # ── 4. No structural mismatch ─────────────────────────────────────────
    score = _compute_entity_presence_score(claim, evidence_text)
    return EntityCheckResult(
        mismatch=False,
        reason="No entity/date inconsistency detected.",
        entity_presence_score=score,
    )


def entity_presence_score(claim: str, evidence_text: str) -> float:
    """Compute the fraction [0.0–1.0] of named entities in *claim* that
    appear in *evidence_text*.

    Returns 1.0 when the claim has no named entities (no penalty for
    generic claims that don't name specific entities).
    Returns 0.0 when the claim has named entities but none appear in the evidence.
    """
    return _compute_entity_presence_score(claim, evidence_text)


def extract_role_keywords(text: str) -> set[str]:
    """Return the set of role/predicate keywords from ROLE_KEYWORDS that
    are present in *text* (case-insensitive whole-phrase match).

    Uses word-boundary-aware matching so that "composed" does NOT match
    the multi-word phrase "composed by" — the full phrase must be present.

    This function is general-purpose: it works on any text without knowing
    in advance which entities the text mentions.
    """
    t = text.lower()
    matched: set[str] = set()
    for kw in ALL_ROLE_KEYWORDS:
        # For single-word keywords use word-boundary regex to avoid partial
        # hits (e.g. "led" inside "fled"). For multi-word phrases a plain
        # substring check is sufficient since the phrase itself is specific.
        if " " in kw:
            if kw in t:
                matched.add(kw)
        else:
            import re as _re
            if _re.search(r"\b" + _re.escape(kw) + r"\b", t):
                matched.add(kw)
    return matched


# ── Internal helpers ─────────────────────────────────────────────────────────

_GENERIC_TOKENS = {"university", "college", "institute", "company", "corporation",
                   "limited", "group", "national", "united", "states", "school",
                   "association", "foundation"}

def _entity_in_text(entity: str, ner_set: set[str], text_lower: str) -> bool:
    entity_l = entity.lower().strip()
    if entity_l in text_lower:
        return True
    if _fuzzy_contains(entity_l, ner_set):
        return True
    # surname / last-token fallback for multi-word names
    tokens = re.findall(r"[a-z0-9\u00c0-\u024f'’\-]+", entity_l)
    if len(tokens) >= 2:
        last = tokens[-1]
        if (last.isalpha() and len(last) >= 4 and last not in _GENERIC_TOKENS
                and re.search(r"\b" + re.escape(last) + r"\b", text_lower)):
            return True
    return False


def _fuzzy_contains(needle: str, haystack: set[str]) -> bool:
    """Substring match in either direction: needle in h, or h in needle."""
    return any(needle in h or h in needle for h in haystack)


def _compute_entity_presence_score(claim: str, evidence_text: str) -> float:
    """Fraction of named entities in the claim that appear in the evidence.

    Named entities counted: people, organizations, locations, dates (NER)
    plus 4-digit years from regex.

    Returns 1.0 when the claim has no named entities (no penalty).
    """
    evidence_l = (evidence_text or "").lower()
    claim_entities = extract_entities(claim)
    claim_years = extract_years(claim)
    evidence_years = extract_years(evidence_text)
    evidence_entities = extract_entities(evidence_text)

    all_claim_entities: list[str] = []
    for group in (
        claim_entities.people,
        claim_entities.organizations,
        claim_entities.locations,
        claim_entities.dates,
        claim_entities.other,
    ):
        all_claim_entities.extend(group)
    all_claim_entities.extend(claim_years)

    if not all_claim_entities:
        return 1.0  # nothing to check — don't penalise

    all_evidence_ner: set[str] = (
        evidence_entities.people
        | evidence_entities.organizations
        | evidence_entities.locations
        | evidence_entities.dates
        | evidence_entities.other
        | evidence_years
    )

    found = sum(
        1
        for e in all_claim_entities
        if _entity_in_text(e, all_evidence_ner, evidence_l)
    )
    return round(found / len(all_claim_entities), 4)
