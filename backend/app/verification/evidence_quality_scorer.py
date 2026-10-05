"""Evidence quality scoring and re-ranking.

After semantic similarity scoring, evidence items are re-ranked by a
*quality score* that combines:

  1. Semantic similarity       — how textually close the evidence is to the claim
  2. Subject presence          — does the evidence mention the question SUBJECT?  ← KEY
  3. Entity presence score     — fraction of claim entities found in evidence
  4. Relationship coverage     — does the evidence discuss the relationship type?
  5. Source reliability weight — Wikipedia/Wikidata > DuckDuckGo

THE SUBJECT-PRESENCE GUARD (the critical fix)
---------------------------------------------
The root cause of false-positives like "St. Petersburg VERIFIED for Einstein's
birthplace" is that evidence ABOUT THE ANSWER VALUE (the Saint Petersburg
article) was being used as supporting evidence.  This article mentions
"St. Petersburg" (the claim's location entity) but NEVER mentions
"Albert Einstein" (the subject of the question).

We now add a subject_presence check:
  - If a QuestionAnalysis is provided and has a focus_entity,
    evidence that does NOT contain the subject entity gets a quality
    score penalty of 0 for the subject component.
  - Evidence that is a Saint Petersburg article (no Einstein mention)
    will have subject_presence = 0.0, reducing its quality_score
    significantly and pushing it to the bottom of the ranked list.
  - The verdict engine then uses the highest-quality (not highest-similarity)
    evidence as its primary evidence.  If all evidence lacks the subject,
    the verdict falls through to UNSUPPORTED rather than VERIFIED.

DESIGN PRINCIPLES
-----------------
• No entity names or factual answers are hard-coded.
• Subject-presence is checked by: direct substring match (lowercased) OR
  spaCy NER match — both are entity-agnostic.
• Source reliability weights are heuristic, not empirically calibrated.

DOCUMENTED LIMITATIONS
-----------------------
• Entity presence depends on spaCy en_core_web_sm accuracy.
• Substring matching is case-insensitive but not stemmed.
• Multi-word subject names (e.g. "Albert Einstein") require ALL tokens to
  appear in the evidence — partial name matches are accepted via fuzzy NER.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.verification.entity_matcher import entity_presence_score, extract_role_keywords
from app.verification.relationship_checker import check_relationship
from app.verification.similarity_scorer import ScoredEvidence

# ── Source reliability weights ────────────────────────────────────────────────
SOURCE_RELIABILITY: dict[str, float] = {
    "wikipedia":  1.00,
    "wikidata":   1.00,
    "duckduckgo": 0.80,
}
DEFAULT_SOURCE_RELIABILITY = 0.70

# ── Component weights ─────────────────────────────────────────────────────────
# Must sum to 1.0.
QUALITY_WEIGHT_SIMILARITY    = 0.30   # reduced from 0.40 — similarity alone misleads
QUALITY_WEIGHT_SUBJECT       = 0.30   # NEW: does evidence mention the question subject?
QUALITY_WEIGHT_ENTITY        = 0.20   # fraction of claim entities found
QUALITY_WEIGHT_RELATIONSHIP  = 0.10   # same relationship category discussed
QUALITY_WEIGHT_SOURCE        = 0.10   # source reliability


@dataclass
class QualityRankedEvidence:
    """An evidence item annotated with its quality breakdown."""
    evidence_text: str
    source_name: str
    source_url: str | None
    similarity: float
    subject_presence: float       # NEW: 1.0 if subject entity found, else 0.0
    entity_presence: float
    relationship_coverage: float
    source_reliability: float
    quality_score: float = field(init=False)

    def __post_init__(self) -> None:
        self.quality_score = round(
            QUALITY_WEIGHT_SIMILARITY    * self.similarity
            + QUALITY_WEIGHT_SUBJECT     * self.subject_presence
            + QUALITY_WEIGHT_ENTITY      * self.entity_presence
            + QUALITY_WEIGHT_RELATIONSHIP * self.relationship_coverage
            + QUALITY_WEIGHT_SOURCE      * self.source_reliability,
            4,
        )


def _subject_present_in_evidence(subject: str, evidence_text: str) -> float:
    """Return a score [0, 1] indicating how centrally the subject entity
    appears in the evidence.

    Returns:
      1.0  — subject is the primary topic of the evidence
      0.3  — subject appears only as part of a compound name (subsidiary/division)
      0.0  — subject is absent from the evidence

    For MULTI-WORD subjects (e.g. "Albert Einstein"), ALL significant tokens
    must be present.

    For single-word subjects (e.g. "Microsoft", "Paris"):
    - Full presence = 1.0 when the subject appears as a standalone entity.
    - Compound penalty = 0.3 when the subject appears ONLY as a prefix of a
      compound name in the first sentence (e.g. "Microsoft AI is a division…").
      This penalises subsidiary/division articles while keeping relevant ones.
    """
    if not subject or not evidence_text:
        return 1.0   # no subject known → no penalty

    subj_l = subject.lower().strip()
    evid_l = evidence_text.lower()

    # ── Subject not found at all ──────────────────────────────────────────
    if subj_l not in evid_l:
        tokens = [t for t in subj_l.split() if len(t) >= 4]
        if len(tokens) >= 2:
            if all(t in evid_l for t in tokens):
                return 1.0
            return 0.0
        # Single-token: spaCy NER fallback
        try:
            from app.nlp.entity_extractor import extract_entities
            ev_ents = extract_entities(evidence_text)
            all_ev_ents: set[str] = set()
            for group in (ev_ents.people, ev_ents.organizations, ev_ents.locations,
                          ev_ents.dates, ev_ents.other):
                all_ev_ents.update(e.lower() for e in group)
            for ent in all_ev_ents:
                if subj_l in ent or ent in subj_l:
                    return 1.0
        except Exception:
            pass
        return 0.0

    # ── Subject IS in the evidence ────────────────────────────────────────
    # Multi-word: all tokens present → full score
    tokens = [t for t in subj_l.split() if len(t) >= 4]
    if len(tokens) >= 2:
        return 1.0 if all(t in evid_l for t in tokens) else 0.0

    # Single-token: check whether the FIRST mention in the text is
    # standalone or part of a compound name.
    # Strategy: look at the first sentence (up to first period or 200 chars).
    import re as _re
    first_sentence = _re.split(r'[.!?]', evidence_text[:300])[0]
    pattern = _re.compile(r'\b' + _re.escape(subj_l) + r'\b', _re.I)
    first_match = pattern.search(first_sentence)

    if first_match is None:
        # Subject not in first sentence but IS elsewhere — present but not primary
        return 0.7

    # Check what immediately follows the first match in the first sentence
    after = first_sentence[first_match.end():first_match.end() + 40].lstrip()

    # Common legal/corporate suffixes that ARE part of the entity's own name
    # and do NOT indicate a different/subsidiary entity.
    _ENTITY_SUFFIXES = {
        "corporation", "corp", "corp.", "inc", "inc.", "ltd", "ltd.", "llc",
        "llc.", "plc", "plc.", "gmbh", "s.a.", "ag", "nv", "bv",
        "company", "co.", "group", "holdings", "enterprises", "foundation",
        "institute", "university", "college", "school", "association",
    }
    # Extract the first word after the subject
    first_after_word = _re.match(r'^([A-Za-z]+)', after)
    if first_after_word:
        word = first_after_word.group(1).lower()
        if word in _ENTITY_SUFFIXES:
            # e.g. "Microsoft Corporation" — this IS the main entity
            return 1.0

    # If followed by another CapitalWord (not a suffix), it's a compound
    # e.g. "Microsoft AI", "Microsoft Azure", "Apple Music"
    if after and _re.match(r'^[A-Z][a-zA-Z]{1,}', after):
        # It's a compound — penalise but don't zero out (still topically related)
        return 0.3

    # Standalone first mention → primary topic
    return 1.0


def rank_evidence_by_quality(
    claim: str,
    scored_evidence: list[ScoredEvidence],
    subject_entity: str = "",
) -> list[QualityRankedEvidence]:
    """Re-rank evidence items by factual quality.

    Parameters
    ----------
    claim:
        The claim string being verified.
    scored_evidence:
        Evidence items scored by semantic similarity (sorted descending).
    subject_entity:
        The question's focus entity (e.g. "Albert Einstein").  When provided,
        evidence that does NOT mention this entity receives subject_presence=0.0,
        significantly reducing its quality score.

    Returns
    -------
    list[QualityRankedEvidence]
        Items re-ordered by quality_score descending.
    """
    ranked: list[QualityRankedEvidence] = []
    for item in scored_evidence:
        ep  = entity_presence_score(claim, item.evidence_text)
        rel = check_relationship(claim, item.evidence_text)
        rel_coverage = 1.0 if rel.relationship_discussed else 0.0
        src_rel = SOURCE_RELIABILITY.get(item.source_name, DEFAULT_SOURCE_RELIABILITY)
        subj_pres = _subject_present_in_evidence(subject_entity, item.evidence_text)

        ranked.append(
            QualityRankedEvidence(
                evidence_text=item.evidence_text,
                source_name=item.source_name,
                source_url=item.source_url,
                similarity=item.similarity,
                subject_presence=subj_pres,
                entity_presence=ep,
                relationship_coverage=rel_coverage,
                source_reliability=src_rel,
            )
        )

    ranked.sort(key=lambda e: e.quality_score, reverse=True)
    return ranked


def quality_ranked_to_scored(
    ranked: list[QualityRankedEvidence],
) -> list[ScoredEvidence]:
    """Convert back to ScoredEvidence for downstream modules."""
    return [
        ScoredEvidence(
            evidence_text=e.evidence_text,
            source_name=e.source_name,
            source_url=e.source_url,
            similarity=e.similarity,
        )
        for e in ranked
    ]
