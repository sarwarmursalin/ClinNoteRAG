"""ClinNoteRAG — Agentic RAG concept coverage agent.

The agent receives a patient note and a list of NBME feature numbers.
For each feature number it calls the `retrieve_concept_info` tool to fetch
the concept text and synonyms from ChromaDB, then determines whether the
note covers that concept, returning a structured verdict.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import chromadb
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# Allow running standalone: python services/agent.py
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django
    django.setup()

from django.conf import settings


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class ConceptVerdict(BaseModel):
    feature_num: str = Field(description="Feature number, e.g. '000'")
    concept: str = Field(description="Human-readable concept name")
    present: bool = Field(description="True if the note documents this concept")
    evidence: str | None = Field(
        default=None,
        description="Verbatim or paraphrased text from the note that supports the verdict",
    )


class CoverageOutput(BaseModel):
    verdicts: list[ConceptVerdict] = Field(
        description="One verdict per feature_num provided"
    )


# ---------------------------------------------------------------------------
# Agent dependencies
# ---------------------------------------------------------------------------


@dataclass
class NBMEDeps:
    case_num: int
    chroma_collection: Any  # chromadb.Collection


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are ClinNoteRAG, an expert medical education AI that evaluates whether a medical \
student's patient history note documents specific clinical concepts.

For EACH feature number you are given:
1. Call the `retrieve_concept_info` tool with that feature_num to get the concept \
   name and its accepted synonym variants.
2. Read the patient note carefully for ANY mention of the concept or its synonyms \
   — including paraphrases, abbreviations, and misspellings.
3. Return:
   - present=True  if the note contains clear textual evidence of the concept
   - present=False if the concept is absent or not mentioned
   - evidence: the exact phrase or sentence from the note that supports present=True \
     (leave null if absent)

Be conservative: only mark present=True when there is direct textual evidence in the note.
Do not infer from general clinical context. Cover ALL feature numbers — do not skip any.
"""

_model = OpenAIModel(
    settings.CAIR_LLM_MODEL,
    provider=OpenAIProvider(
        base_url=settings.CAIR_LLM_URL,
        api_key=settings.CAIR_LLM_API_KEY,
        http_client=httpx.AsyncClient(verify=settings.CAIR_LLM_SSL_VERIFY),
    ),
)

nbme_agent: Agent[NBMEDeps, CoverageOutput] = Agent(
    _model,
    deps_type=NBMEDeps,
    output_type=CoverageOutput,
    system_prompt=SYSTEM_PROMPT,
)


@nbme_agent.tool
async def retrieve_concept_info(ctx: RunContext[NBMEDeps], feature_num: str) -> str:
    """Retrieve the concept name and synonym variants for a given feature number.

    Args:
        feature_num: The feature number string (e.g. '000', '104').

    Returns:
        The concept text showing all accepted synonym variants, or an error message.
    """
    doc_id = f"{ctx.deps.case_num}_{feature_num}"
    results = ctx.deps.chroma_collection.get(ids=[doc_id])
    if results["documents"]:
        return results["documents"][0]
    return f"Concept {feature_num} (case {ctx.deps.case_num}) not found in knowledge base."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def assess_note(
    note_text: str,
    case_num: int,
    feature_nums: list[str],
    chroma_collection: Any,
) -> list[ConceptVerdict]:
    """Run the NBME coverage agent on a single patient note.

    Args:
        note_text:         The pn_history field from patient_notes.csv.
        case_num:          Integer case number (0–9).
        feature_nums:      Feature number strings to evaluate (e.g. ['000', '001']).
        chroma_collection: ChromaDB collection holding concept embeddings.

    Returns:
        List of ConceptVerdict, one per feature_num.
    """
    deps = NBMEDeps(case_num=case_num, chroma_collection=chroma_collection)

    feature_list = "\n".join(f"- {fn}" for fn in feature_nums)
    prompt = (
        f"Patient history note (case {case_num}):\n"
        f"{note_text}\n\n"
        f"---\n"
        f"Evaluate the following {len(feature_nums)} feature numbers. "
        f"For each, call retrieve_concept_info then determine coverage.\n\n"
        f"Feature numbers:\n{feature_list}"
    )

    result = await nbme_agent.run(prompt, deps=deps)
    return result.output.verdicts


# ---------------------------------------------------------------------------
# Smoke test — python services/agent.py
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import pandas as pd

    DATA_DIR = settings.NBME_DATA_DIR
    notes_df = pd.read_csv(DATA_DIR / "NBME_PN_HISTORY.txt", sep="|")
    features_df = pd.read_csv(DATA_DIR / "NBME_PN_HISTORY_FEATURES.txt", sep="|")

    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    collection = client.get_collection("nbme_concepts")

    # Pick first note from case 201
    sample = notes_df[notes_df["CASE_NUM"] == 201].iloc[0]
    feature_nums = features_df[features_df["CASE_NUM"] == 201]["FEATURE_NUM"].astype(int).astype(str).tolist()

    async def run_test() -> None:
        print(f"Note {sample['PN_NUM']} | Case 201 | {len(feature_nums)} concepts\n")
        print(f"Note text:\n{sample['PN_HISTORY'][:400]}...\n")
        print("=" * 60)

        verdicts = await assess_note(
            note_text=str(sample["PN_HISTORY"]),
            case_num=201,
            feature_nums=feature_nums,
            chroma_collection=collection,
        )

        for v in verdicts:
            status = "PRESENT" if v.present else "absent "
            ev = f" → \"{v.evidence[:70]}\"" if v.evidence else ""
            print(f"  [{v.feature_num}] {status}  {v.concept}{ev}")

    asyncio.run(run_test())
