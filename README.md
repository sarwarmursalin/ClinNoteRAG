# ClinNoteRAG

**Agentic RAG system for auto-grading medical student patient notes against NBME rubric concepts.**

Capstone project — ENGI981A, Memorial University of Newfoundland.

## What it does

Takes a free-text patient history note written by a medical student and automatically determines, for each clinical concept in the case rubric, whether the student documented it — with quoted evidence. Evaluated against 14,300 expert physician labels from the NBME USMLE Step 2 dataset.

```
features.csv (143 rubric concepts)
        ↓ embed with sentence-transformers
    ChromaDB knowledge base
        ↓
Patient note + case ID
        ↓
  [Pydantic AI Agent]
  For each concept:
    → calls retrieve_concept_info tool (ChromaDB lookup)
    → reasons over note + synonyms
    → returns present/absent + evidence
        ↓
Compare to expert labels → Precision / Recall / F1
```

## Stack

| Component | Technology |
|---|---|
| Web framework | Django 5 |
| Vector DB | ChromaDB |
| LLM | Claude (Anthropic) / Ollama |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Agent | Pydantic AI |
| Metrics | scikit-learn |

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Create .env
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

# 3. Set up database
python manage.py migrate

# 4. Ingest NBME concepts into ChromaDB (one-time)
python scripts/ingest_concepts.py

# 5. Smoke test agent on one note
python services/agent.py

# 6. Run full evaluation (1,000 notes)
python scripts/evaluate.py

# 7. Start web dashboard
python manage.py runserver
# → http://127.0.0.1:8000
```

## Data

NBME Score Clinical Patient Notes dataset (Kaggle).
Place the three CSV files in `../NMBE data/` relative to this project:
- `features.csv` — 143 rubric concepts across 10 clinical cases
- `train.csv` — 14,300 expert-annotated concept-level labels
- `patient_notes.csv` — 42,146 free-text patient notes

## Project structure

```
ClinNoteRAG/
├── apps/assessment/     Django app — models, views, admin
├── services/agent.py    Pydantic AI agentic RAG pipeline
├── services/embedder.py sentence-transformers wrapper
├── scripts/ingest_concepts.py   Task 1: features.csv → ChromaDB
├── scripts/evaluate.py          Task 3: F1 evaluation over 1,000 notes
└── templates/           Web dashboard
```
