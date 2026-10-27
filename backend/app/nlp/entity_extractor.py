"""Entity/date extraction using spaCy NER, used for consistency checking.

We reuse the same spaCy pipeline loader as claim_extractor via a shared
cache so the model is only loaded once per process.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.nlp.claim_extractor import _get_nlp

# Entity labels we care about for fact-consistency checks.
RELEVANT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "NORP", "FAC", "EVENT"}


@dataclass
class ExtractedEntities:
    people: set[str] = field(default_factory=set)
    organizations: set[str] = field(default_factory=set)
    locations: set[str] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    other: set[str] = field(default_factory=set)

    def all_values(self) -> set[str]:
        return self.people | self.organizations | self.locations | self.dates | self.other


def extract_entities(text: str) -> ExtractedEntities:
    """Run spaCy NER over `text` and bucket entities by type."""
    result = ExtractedEntities()
    if not text or not text.strip():
        return result

    nlp = _get_nlp()
    doc = nlp(text)
    for ent in doc.ents:
        value = ent.text.strip().lower()
        if not value:
            continue
        if (ent.label_ == "ORG" and len(ent) == 1 and ent.start > 0
                and ent.doc[ent.start - 1].lower_ == "the"):
            continue
        if ent.label_ == "PERSON":
            result.people.add(value)
        elif ent.label_ == "ORG":
            result.organizations.add(value)
        elif ent.label_ in ("GPE", "LOC"):
            result.locations.add(value)
        elif ent.label_ == "DATE":
            result.dates.add(value)
        elif ent.label_ in RELEVANT_LABELS:
            result.other.add(value)
    return result


def extract_years(text: str) -> set[str]:
    """Very small helper: pull 4-digit year-like tokens out of text.

    Complements spaCy DATE entities, which sometimes miss bare years
    embedded in larger phrases.
    """
    import re

    return set(re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text or ""))
