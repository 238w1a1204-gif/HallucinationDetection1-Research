"""Component-toggled re-implementation of verdict_engine rules 0-7.

`FULL` must agree with the real engine (run_ablation prints the agreement rate).
"""
from __future__ import annotations

V, P, U, C = "VERIFIED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"

FULL = dict(nli=True, entity=True, rel=True, av=True)

CONFIGS = [
    ("Sim+Agreement", {}),
    ("NLI only", "nli_only"),
    ("Sim+NLI", dict(nli=True)),
    ("Sim+NLI+Entity", dict(nli=True, entity=True)),
    ("Sim+NLI+Entity+Rel", dict(nli=True, entity=True, rel=True)),
    ("Full (+AnswerValue)", FULL),
    ("NLI-gated (+AV)", "gated"),
    ("NLI-gated +Entity", "gated_entity"),
]


def decide(f, cfg, strong=0.75, partial=0.55, min_sources=2):
    if f["n_evidence"] == 0:
        return U
    if cfg.get("av") and f["av"] == "VALUE_CONTRADICTED":
        return C
    if cfg.get("nli") and f["nli_primary"] == "contradiction" and (f["rel_ok"] or not cfg.get("rel")):
        return C
    if cfg.get("entity") and f["ent_mismatch"]:
        return C if f["max_sim"] >= partial else U
    rel_ok = f["rel_ok"] if cfg.get("rel") else True
    sim = f["max_sim"]
    if sim >= strong:
        return V if (f["agree"] >= min_sources and rel_ok) else P
    if sim >= partial:
        return P
    return U


def decide_nli_only(f):
    labs = f["nli_all"]
    if not labs:
        return U
    ent, con = "entailment" in labs, "contradiction" in labs
    if ent and con:
        return P
    if ent:
        return V
    if con:
        return C
    return U

def decide_gated(f, use_entity=False, partial=0.55, **_):
    """NLI decides support/contradiction/abstention; answer-value and entity
    checks act as gates. Similarity is no longer a verdict threshold."""
    if f["n_evidence"] == 0:
        return U
    if f["av"] == "VALUE_CONTRADICTED":
        return C
    labs = f["nli_all"]
    ent, con = "entailment" in labs, "contradiction" in labs
    if con and not ent:
        return C
    if ent and con:
        return P
    if ent:
        return P if (use_entity and f["ent_mismatch"]) else V
    if use_entity and f["ent_mismatch"] and f["max_sim"] >= partial:
        return C
    return U

def predict(f, cfg, **kw):
    if cfg == "nli_only":
        return decide_nli_only(f)
    if cfg == "gated":
        return decide_gated(f, False, **kw)
    if cfg == "gated_entity":
        return decide_gated(f, True, **kw)
    return decide(f, cfg, **kw)