"""No-RAG baseline — zero retrieval, concept names only.

Gives the LLM only the concept names (no synonym expansion, no ChromaDB).
One LLM call per note. Shows what the model can do purely from its
training knowledge without any retrieval support.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
2. A list of clinical concept names to check

For EACH concept, follow these steps:
1. SCAN — identify every sentence or phrase in the note that could relate to this concept, \
   including paraphrases, abbreviations, and clinical shorthand.
2. JUDGE — decide: would a reasonable clinician reading this note conclude that the student \
   documented this concept? Accept natural clinical language even when wording differs \
   from the concept name.
3. RETURN:
   - present=True if the note documents this concept in any recognisable form
   - present=False only if there is genuinely no mention or implication of the concept
   - evidence: the most relevant phrase or sentence from the note (null if absent)
   - reason: if present=False, one sentence — either "Not mentioned in the note." OR \
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

no_rag_agent: Agent[None, CoverageOutput] = Agent(
    _model,
    output_type=CoverageOutput,
    system_prompt=SYSTEM_PROMPT,
)


async def assess_note_no_rag(
    note_text: str,
    case_num: int,
    feature_nums: list[str],
    concept_names: dict[str, str],
) -> list[ConceptVerdict]:
    """No-RAG: concept names only in prompt, zero retrieval."""

    concept_list = "\n".join(
        f"- feature_num={fn}: {concept_names.get(fn, fn)}"
        for fn in feature_nums
    )

    prompt = (
        f"Patient history note (case {case_num}):\n"
        f"{note_text}\n\n"
        f"---\n"
        f"Evaluate ALL {len(feature_nums)} concepts listed below.\n\n"
        f"{concept_list}"
    )

    result = await no_rag_agent.run(prompt)
    return result.output.verdicts
