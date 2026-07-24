"""Naive RAG baseline — single prompt, bulk retrieval.

Retrieves ALL concepts for the case from ChromaDB in one query,
stuffs them into a single prompt with the note, and makes ONE LLM call.

No tool use, no iteration. Contrast with the agentic approach which
calls retrieve_concept_info individually for each concept.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    import django
    django.setup()

from django.conf import settings
from services.agent import ConceptVerdict, CoverageOutput

SYSTEM_PROMPT = """\
You are ClinNoteRAG, an expert medical education AI that evaluates whether a medical \
student's patient history note documents specific clinical concepts.

You will be given:
1. A patient history note
2. A list of clinical concepts with their accepted synonym variants

For EACH concept, follow these steps in order:
1. SCAN — identify every sentence or phrase in the note that could relate to this concept, \
   including paraphrases, abbreviations, clinical shorthand, and synonyms.
2. JUDGE — decide: would a reasonable clinician reading this note conclude that the student \
   documented this concept? Accept natural clinical language even when wording differs from \
   the concept name (e.g. "feels feverish" counts for "Subjective fevers"; \
   "lifted heavy objects" counts for "Recent heavy lifting at work").
3. RETURN:
   - present=True if the note documents this concept in any recognisable form
   - present=False only if there is genuinely no mention or implication of the concept
   - evidence: the most relevant phrase or sentence from the note (null if absent)
   - reason: if present=False, one sentence for the student — either \
     "Not mentioned in the note." OR \
     "Mentioned but incorrect: the note states \'[quote]\' but this concept requires \
     [expected value]." (null if present=True)

Cover ALL concepts — do not skip any.
"""

_model = OpenAIModel(
    settings.CAIR_LLM_MODEL,
    provider=OpenAIProvider(
        base_url=settings.CAIR_LLM_URL,
        api_key=settings.CAIR_LLM_API_KEY,
        http_client=httpx.AsyncClient(verify=settings.CAIR_LLM_SSL_VERIFY),
    ),
)

naive_agent: Agent[None, CoverageOutput] = Agent(
    _model,
    output_type=CoverageOutput,
    system_prompt=SYSTEM_PROMPT,
)


async def assess_note_naive(
    note_text: str,
    case_num: int,
    feature_nums: list[str],
    chroma_collection: Any,
) -> list[ConceptVerdict]:
    """Naive RAG: retrieve all concepts at once, one LLM call."""

    # Fetch all concepts for this case from ChromaDB in one query
    results = chroma_collection.get(
        where={"case_num": int(case_num)},
        include=["documents", "metadatas"],
    )

    # Build a lookup so we can order by feature_num
    doc_map: dict[str, str] = {}
    for doc, meta in zip(results["documents"], results["metadatas"]):
        doc_map[str(meta["feature_num"])] = doc

    # Build context block — only for requested feature_nums
    context_parts = []
    for fn in feature_nums:
        doc = doc_map.get(fn, f"Concept {fn}: (not found)")
        context_parts.append(f"[feature_num={fn}]\n{doc}")

    context = "\n\n".join(context_parts)
    feature_list = "\n".join(f"- {fn}" for fn in feature_nums)

    prompt = (
        f"Patient history note (case {case_num}):\n"
        f"{note_text}\n\n"
        f"---\n"
        f"Concept information (retrieved from knowledge base):\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Evaluate ALL {len(feature_nums)} feature numbers listed below.\n"
        f"Feature numbers:\n{feature_list}"
    )

    result = await naive_agent.run(prompt)
    return result.output.verdicts
