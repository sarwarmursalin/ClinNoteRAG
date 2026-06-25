# ClinNoteRAG

**Automated patient history note assessment using Agentic Retrieval-Augmented Generation.**

ClinNoteRAG evaluates medical student patient notes against NBME clinical rubric concepts — giving instant, concept-level feedback on what was documented and what was missed. Built as a full-stack web application for deployment in medical education settings.

> Capstone Project — ENGI981A, Memorial University of Newfoundland  
> Supervised by Dr. Elif Ak

---

## Features

- **Agentic RAG pipeline** — LLM agent iteratively queries ChromaDB per concept, extracting evidence from free-text notes
- **Role-based access** — separate student and faculty/grader portals with Django authentication
- **Evidence highlighting** — matched phrases color-highlighted directly in the submitted note
- **Before/after comparison** — edit and resubmit to track improvement across attempts
- **Faculty dashboard** — graders see all student submissions with coverage scores
- **Validated on 2,839 notes** — evaluated against NBME expert physician annotations

---

## System Architecture

```
Student Note (free text)
        │
        ▼
  [Agentic LLM — Pydantic AI]
  For each rubric concept:
    1. Query ChromaDB → retrieve synonyms & accepted phrasings
    2. Reason: is this concept documented in the note?
    3. Extract supporting evidence phrase
        │
        ▼
  Concept verdicts (present / absent + evidence)
        │
        ▼
  Score report — concept breakdown, highlighted note, missing concepts
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Django 5 |
| Agent framework | Pydantic AI |
| Vector database | ChromaDB |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | IBM Granite 4.1 30B (via CAIR LiteLLM) |
| Database | SQLite |
| Frontend | Django templates + Chart.js |
| Language | Python 3.13 |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Access to an LLM endpoint (configure in `config/settings.py`)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/sarwarmursalin/ClinNoteRAG.git
cd ClinNoteRAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Apply database migrations
python manage.py migrate

# 4. Create an admin account
python manage.py createsuperuser

# 5. Start the development server
python manage.py runserver
```

Visit `http://127.0.0.1:8000` — the landing page is publicly accessible.  
The vector database (`chroma_db/`) is included in the repo, so no ingestion step is needed.

### Creating user accounts

| Role | How to create |
|---|---|
| **Student** | Self-register at `/register/` |
| **Faculty / Grader** | Admin creates via `/admin/` → Users → Add, then set Role to *Faculty / Grader* in the Profile section |
| **Admin** | `python manage.py createsuperuser` |

---

## Project Structure

```
ClinNoteRAG/
├── apps/
│   └── assessment/          Django app
│       ├── models.py         EvaluationRun, ConceptVerdict, UserProfile
│       ├── views.py          Role-based views + decorators
│       ├── forms.py          Auth and evaluation forms
│       ├── admin.py          Django admin with UserProfile inline
│       └── migrations/
├── services/
│   ├── agent.py              Agentic RAG pipeline (Pydantic AI)
│   ├── naive_rag.py          Naive RAG baseline
│   ├── no_rag.py             No-RAG baseline
│   └── embedder.py           sentence-transformers wrapper
├── scripts/
│   ├── evaluate.py           Batch evaluation over NBME dataset
│   ├── ingest_concepts.py    Load rubric concepts into ChromaDB
│   ├── error_analysis.py     Cohen's κ, per-case metrics, FN analysis
│   └── llm_judge.py          LLM-as-Judge inter-rater validation
├── chroma_db/                Pre-built vector database (143 concepts)
├── templates/                All HTML templates
│   ├── base.html
│   ├── base_auth.html
│   └── assessment/
│       ├── landing.html
│       ├── evaluate.html
│       ├── note_result.html
│       ├── faculty_dashboard.html
│       ├── ablation.html
│       └── ...
├── config/                   Django settings and URL config
├── requirements.txt
└── manage.py
```

---

## Dataset

NBME Score Clinical Patient Notes (Kaggle).  
Place the three CSV files in `../NBME data/` relative to the project root:

| File | Description |
|---|---|
| `features.csv` | 143 rubric concepts across 10 clinical cases |
| `train.csv` | 14,300 expert-annotated concept-level labels |
| `patient_notes.csv` | 42,146 free-text patient notes |

The dataset is not included in this repository. Download from [Kaggle NBME Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes).

---

## Evaluation Results

Ablation study across three retrieval strategies on 2,839 notes:

| Strategy | F1 | Cohen's κ | Notes |
|---|---|---|---|
| No-RAG Baseline | 0.9423 | 0.8134 | Concept names only |
| Naive RAG | 0.9233 | 0.7691 | Bulk ChromaDB retrieval |
| **Agentic RAG** | **0.7886** | **0.4804** | Per-concept tool-use ← production |

LLM-as-Judge inter-rater agreement: κ = 0.8415 (system vs judge).

> The production system uses Agentic RAG for its interpretability and evidence extraction capability, despite the F1 trade-off. See the ablation study page in the app for full analysis.

---

## Team

| Name | Role |
|---|---|
| Golam Sarwar Md. Mursalin | Backend & AI/ML — RAG pipeline, LLM integration, evaluation harness |
| S M Ziauddin | Frontend & UX — UI design, templates, dashboards |

---

## License

This project was developed for academic research at Memorial University of Newfoundland.  
Dataset usage is subject to [NBME competition rules](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/rules).
