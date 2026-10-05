# Hallucination Detection & Answer Verification

Hybrid multi-source fact-verification system that detects hallucinations in LLM-generated answers using Wikipedia, Wikidata, and DuckDuckGo.

---

## 1. Project Objective

Large Language Models (LLMs) frequently generate plausible-sounding but factually incorrect text — known as *hallucination*. This system:

1. Accepts a natural-language question.
2. Generates an answer using a configurable LLM provider (Hugging Face / Gemini / OpenAI).
3. **Analyses the question** to determine the expected answer type, subject entity, and relationship being asked (e.g. `BORN_IN`, `FOUNDED_BY`, `LOCATED_IN`).
4. Extracts individual factual claims from the answer (sentence splitting + short-answer expansion using question context).
5. Retrieves evidence from **Wikipedia, Wikidata, and DuckDuckGo** using subject-centred queries — not queries dominated by the answer value.
6. **Validates that evidence is about the right subject** (not a different entity with a similar name).
7. Scores evidence by semantic similarity AND by a multi-factor quality score (subject presence, entity overlap, relationship coverage, source reliability).
8. Checks the **answer value** against the relationship-specific value in the evidence (e.g. "St. Petersburg" vs "Ulm, Germany").
9. Detects contradictions via NLI (bart-large-mnli) and explicit entity/year mismatch checks.
10. Assigns per-claim verdicts: `VERIFIED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`.
11. Computes a heuristic confidence score.
12. Generates a **corrected answer** from verified evidence only — never repeating a hallucinated answer.
13. Produces a full structured verification report (JSON + plain text).
14. Persists all results to SQLite.
15. Exposes the full pipeline through a FastAPI REST API consumed by a React/Vite frontend.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        React Frontend (Vite)                          │
│  QuestionForm → VerificationSummary (question type + relationship)   │
│  AnswerPanel (original + corrected) → ClaimsList → EvidenceList      │
│  ReportSection (download JSON/text)                                   │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ HTTP/Axios
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Uvicorn)                         │
│                                                                       │
│  POST /api/questions  ──►  pipeline_orchestrator.run_pipeline()       │
│                                                                       │
│  ① LLM Answer Generation    (HuggingFace / Gemini / OpenAI)          │
│  ② Question Analysis        (question_analyzer)                      │
│     → question_type, relationship, focus_entity                      │
│     → fuzzy entity resolution (Wikipedia search oracle)              │
│  ③ Claim Extraction         (spaCy sentence split)                   │
│     → short-answer expansion: "Paris" + "Where is the Eiffel Tower?" │
│       → "The Eiffel Tower is located in Paris."                      │
│  ④ Subject-Centred Retrieval (query_builder + aggregator)            │
│     → query = "Albert Einstein born"  NOT "st. petersburg"           │
│     ├─ Wikipedia API                                                  │
│     ├─ Wikidata API                                                   │
│     └─ DuckDuckGo Instant Answer API                                 │
│  ⑤ Evidence Quality Ranking  (evidence_quality_scorer)               │
│     → subject_presence × 0.30  ← KEY: rejects wrong-entity evidence │
│     → entity_presence  × 0.20                                        │
│     → relationship     × 0.10                                        │
│     → similarity       × 0.30                                        │
│     → source_reliability × 0.10                                      │
│  ⑥ Fact Verification                                                  │
│     ├─ answer_value_checker  (value vs relationship in evidence)     │
│     ├─ entity_matcher        (NER + year consistency)                │
│     ├─ contradiction_detector (NLI bart-large-mnli)                  │
│     ├─ relationship_checker  (predicate keyword matching)            │
│     └─ verdict_engine        (7-rule priority decision)              │
│  ⑦ Hallucination Analysis    (answer-level aggregation)              │
│  ⑧ Confidence Score          (heuristic weighted sum)                │
│  ⑨ Corrected Answer          (evidence-only, never hallucinations)   │
│  ⑩ Report Generation         (JSON + plain text)                     │
│  ⑪ Database Persistence      (SQLAlchemy + SQLite)                   │
└──────────────────────────────────────────────────────────────────────┘
```

### Verdict decision rules (priority order)

| Rule | Condition | Verdict |
|------|-----------|---------|
| 0 | Answer-value contradicts evidence value for the requested relationship | `CONTRADICTED` |
| 1 | NLI model says evidence contradicts claim | `CONTRADICTED` |
| 2 | Entity mismatch + similarity ≥ partial threshold | `CONTRADICTED` |
| 3 | Entity mismatch + similarity < partial threshold | `UNSUPPORTED` |
| 4 | Strong similarity + ≥2 sources + relationship discussed + value ok | `VERIFIED` |
| 5 | Strong similarity but single source or relationship unclear | `PARTIALLY_SUPPORTED` |
| 6 | Partial similarity range | `PARTIALLY_SUPPORTED` |
| 7 | Low similarity | `UNSUPPORTED` |

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI 0.115 + Uvicorn |
| ORM / database | SQLAlchemy 2.0 + SQLite |
| Data validation | Pydantic v2 + pydantic-settings |
| LLM (default) | HuggingFace Transformers — google/flan-t5-base |
| LLM (optional) | Google Gemini API / OpenAI API |
| Claim extraction | spaCy 3.7 (en_core_web_sm) |
| Semantic similarity | Sentence-Transformers (all-MiniLM-L6-v2) |
| NLI / contradiction | HuggingFace — facebook/bart-large-mnli |
| Evidence retrieval | Wikipedia REST API, Wikidata API, DuckDuckGo Instant Answer |
| Entity resolution | Wikipedia search (typo-tolerant) + difflib token-overlap |
| Testing | pytest 8.3 + pytest-mock |
| Frontend | React 19 + Vite 8 + Tailwind CSS 3 + Axios |

---

## 4. Backend Setup

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Default `.env` uses local Hugging Face — no API key required.

---

## 5. Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_BASE_URL=http://localhost:8000
```

---

## 6. Environment Variables

### Backend (`backend/.env`)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `huggingface` | `huggingface` \| `gemini` \| `openai` |
| `HF_GENERATION_MODEL` | `google/flan-t5-base` | HuggingFace model for answer generation |
| `GEMINI_API_KEY` | _(empty)_ | Required when `LLM_PROVIDER=gemini` |
| `OPENAI_API_KEY` | _(empty)_ | Required when `LLM_PROVIDER=openai` |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy NER model |
| `SENTENCE_TRANSFORMER_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `NLI_MODEL` | `facebook/bart-large-mnli` | NLI contradiction model |
| `SIMILARITY_STRONG_THRESHOLD` | `0.75` | Threshold for VERIFIED |
| `SIMILARITY_PARTIAL_THRESHOLD` | `0.55` | Threshold for PARTIALLY_SUPPORTED |
| `AGREEMENT_MIN_SOURCES` | `2` | Minimum sources for VERIFIED |
| `DATABASE_URL` | `sqlite:///./hallucination_detection.db` | DB connection string |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | CORS allowed origin |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8000` | Backend API base URL |

---

## 7. How to Run the Backend

```bash
cd backend
venv\Scripts\activate          # Windows
# source venv/bin/activate    # macOS/Linux

uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- **API base**: http://127.0.0.1:8000/api
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

> **Note:** First request downloads HuggingFace models (~2.5 GB total). Subsequent requests are much faster.

---

## 8. How to Run the Frontend

```bash
cd frontend
npm run dev    # Development (hot-reload) → http://localhost:5173
npm run build  # Production build → frontend/dist/
```

---

## 9. API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check + LLM provider |
| `POST` | `/api/questions` | Submit question, run full pipeline |
| `GET` | `/api/questions` | List questions (paginated) |
| `GET` | `/api/questions/{id}` | Question detail + stats |
| `GET` | `/api/questions/{id}/claims` | Per-claim breakdown with evidence |
| `GET` | `/api/questions/{id}/report` | Full verification report |

### POST /api/questions — Response fields (key)

```json
{
  "question_id": 1,
  "question": "Where was Albert Einstein born?",
  "answer": "st. petersburg",
  "question_type": "LOCATION",
  "question_relationship": "born in",
  "overall_verdict": "CONTAINS_HALLUCINATION",
  "overall_confidence": 0.40,
  "corrected_answer": "The original answer could not be verified.",
  "claims": [
    {
      "claim_text": "st. petersburg",
      "verification_result": {
        "verdict": "CONTRADICTED",
        "entity_mismatch": false,
        "explanation": "..."
      },
      "evidence": [...]
    }
  ]
}
```

---

## 10. Verification Methodology

### Subject-centred retrieval (key fix)

When the LLM says "Albert Einstein was born in St. Petersburg", the naive retrieval query would be the claim text, causing Wikipedia to return the **Saint Petersburg** article. This creates a false positive: the Saint Petersburg article has high embedding similarity to the claim but says nothing about Einstein's birthplace.

The fix: retrieval queries are built from the **subject entity + relationship keyword**, not the answer value:

| Question | Retrieval query |
|---|---|
| Where was Albert Einstein born? | `Albert Einstein born` |
| Who founded Microsoft? | `Microsoft founded founder` |
| Where is the Eiffel Tower located? | `Eiffel Tower location` |
| Who was Albert Einstein? | `Albert Einstein` |

### Subject-presence evidence guard

After retrieval, every evidence item is scored by whether it mentions the question's subject entity. Evidence that does NOT mention "Albert Einstein" (e.g. the Saint Petersburg article) receives `subject_presence = 0.0`, dramatically lowering its quality score. This prevents it from being selected as the primary evidence for checks.

For multi-word subjects (e.g. "Albert Einstein"), ALL significant tokens must appear in the evidence.

### Answer-value checking

For structured relationship questions (BORN_IN, LOCATED_IN, FOUNDED_BY, etc.), the system extracts the relationship-specific value from the evidence and compares it against the answer value:

- `answer="St. Petersburg"`, `evidence="Einstein was born in Ulm"` → `VALUE_CONTRADICTED`
- `answer="Ulm"`, `evidence="Einstein was born in Ulm"` → `VALUE_SUPPORTED`
- `answer="Paris"`, `evidence="...the Eiffel Tower..."` → `VALUE_SUPPORTED`

For **biographical** questions (Who was X?), the system requires the subject to be confirmed in the evidence before firing CONTRADICTED — preventing wrong-entity articles from generating false contradictions.

### Entity resolution (typo handling)

When spaCy NER fails on a misspelled name (e.g. "albert einstien"), the system:
1. Detects that NER returned only a fragment (single short word with more text following)
2. Uses the full candidate text as the entity
3. Queries Wikipedia's typo-tolerant search API
4. Applies `difflib.SequenceMatcher` token-level comparison to confirm the resolved name
5. Returns the canonical Wikipedia article title ("Albert Einstein") as the focus entity

This is general-purpose — it does not know about specific people. The same mechanism handles any misspelled entity.

---

## 11. Confidence Score Methodology

> **Important:** The confidence score is a **heuristic weighted sum**, NOT a calibrated probability. It is not statistically validated against a labelled dataset.

```
confidence = 0.35 × similarity
           + 0.25 × (agreeing_sources / 3)
           + 0.20 × consistency_signal      (0 if contradiction or entity mismatch)
           + 0.20 × entity_presence_score   (fraction of claim entities in evidence)
```

High confidence requires: strong evidence + multiple agreeing sources + no entity/contradiction flags + the claim's entities appearing in the evidence.

A claim with high similarity but wrong entity (e.g. "St. Petersburg" claim matched to the Saint Petersburg article) scores near zero on `entity_presence` and zero on `consistency_signal`, keeping confidence low.

---

## 12. Testing

```bash
cd backend
venv\Scripts\activate
pytest tests/ -v
```

### Test suite summary (263 tests)

| File | Tests | Coverage |
|---|---|---|
| `test_api.py` | 21 | All 6 API endpoints |
| `test_claim_extractor.py` | 10 | Sentence splitting, deduplication |
| `test_confidence_calculator.py` | 13 | Formula, weights, boundary conditions |
| `test_entity_matcher.py` | 18 | NER checks, org checks, year mismatch |
| `test_entity_resolution.py` | 29 | Typo resolution, wrong-entity guard, corrected-answer guard |
| `test_general_verification.py` | 47 | 16 factual categories |
| `test_hallucination_analysis.py` | 10 | Verdict aggregation |
| `test_multi_factor_verification.py` | 16 | John McCarthy / Infosys regression |
| `test_question_aware_verification.py` | 55 | Question analyzer, short-answer, answer-value |
| `test_relationship_checker.py` | 7 | Predicate keyword matching |
| `test_similarity_scorer.py` | 6 | Cosine similarity, sorting |
| `test_subject_centred_verification.py` | 30 | St. Petersburg false-positive regression |
| `test_verdict_engine.py` | 10 | All verdict branches |
| **Total** | **263** | |

Tests use an in-memory SQLite database, mocked NLI calls, and mocked LLM. All 263 pass in under 3 minutes.

---

## 13. Known Limitations

### LLM generation quality
`google/flan-t5-base` is a 250M-parameter model. It produces short, often incorrect answers. Use `LLM_PROVIDER=gemini` or `LLM_PROVIDER=openai` for meaningful verification results.

### First-run latency
On first run, the system downloads:
- `google/flan-t5-base`: ~990 MB
- `sentence-transformers/all-MiniLM-L6-v2`: ~90 MB
- `facebook/bart-large-mnli`: ~1.6 GB

Subsequent requests use the cached models.

### Wikipedia retrieval variability
Wikipedia's search API can return incorrect articles even with subject-centred queries (e.g. "Albert Einstein born" may return "Hans Albert Einstein" — his son). The subject-presence guard reduces the damage from such cases, but cannot eliminate it entirely.

### NER accuracy
`en_core_web_sm` has moderate recall for less common names, non-English entities, and misspellings. The entity resolver mitigates typos via Wikipedia's spelling-correction, but unusual names may still be missed.

### Entity resolver coverage
The fuzzy resolver uses Wikipedia's search API with a timeout of 6 seconds. Network failures cause graceful fallback to the raw candidate.

### Confidence scores are not calibrated
Treat them as a relative ordering signal ("this claim has more evidence support than that one"), not as probabilities of truth. Proper calibration requires a labelled evaluation dataset.

### Single-word LLM answers
`flan-t5-base` often returns one-word answers (e.g. "Paris"). The short-answer expander constructs a full claim from question context, but the claim extractor's minimum-length filter (`≥ 8 chars`) must still be satisfied.

### Multi-part questions
Questions like "Where was Marie Curie born and what did she discover?" are analysed using the first detectable relationship pattern. Multi-relationship extraction is a documented future improvement.
