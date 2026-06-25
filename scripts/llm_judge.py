"""
LLM-as-Judge evaluation.

Selects a stratified sample of (note, concept) pairs, asks an independent
LLM judge to label each, then computes Cohen's kappa against:
  - Expert ground truth labels
  - Our no_rag system's predictions

Run from project root:
  python scripts/llm_judge.py

Requires VPN to reach CAIR LiteLLM endpoint.
Output: scripts/judge_results.csv, prints kappa summary.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()

from django.conf import settings
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

# ── Model setup ────────────────────────────────────────────────────────────────
_model = OpenAIModel(
    settings.CAIR_LLM_MODEL,
    provider=OpenAIProvider(
        base_url=settings.CAIR_LLM_URL,
        api_key=settings.CAIR_LLM_API_KEY,
        http_client=httpx.AsyncClient(verify=settings.CAIR_LLM_SSL_VERIFY),
    ),
)

class JudgeVerdict(BaseModel):
    present: bool
    reasoning: str

judge_agent: Agent[None, JudgeVerdict] = Agent(
    _model,
    output_type=JudgeVerdict,
    system_prompt="""\
You are a medical education expert evaluating whether a clinical concept is \
documented in a student's patient history note.

Given a patient note and one specific clinical concept, determine:
- present=True if the note contains clear textual evidence of this concept
- present=False if the concept is absent or not mentioned
- reasoning: one sentence explaining your decision

Be strict: only mark present=True if there is direct textual evidence in the note.
""",
)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_notes() -> dict[tuple[int, int], str]:
    """Returns {(case_num, pn_num): note_text}."""
    df = pd.read_csv(
        settings.NBME_DATA_DIR / "NBME_PN_HISTORY.txt",
        sep="|",
        quoting=1,        # QUOTE_ALL
    )
    df["CASE_NUM"] = pd.to_numeric(df["CASE_NUM"], errors="coerce")
    df["PN_NUM"]   = pd.to_numeric(df["PN_NUM"],   errors="coerce")
    df.dropna(subset=["CASE_NUM", "PN_NUM"], inplace=True)
    return {
        (int(row.CASE_NUM), int(row.PN_NUM)): str(row.PN_HISTORY)
        for _, row in df.iterrows()
    }


def build_sample(norag_df: pd.DataFrame, notes: dict, n: int = 80) -> pd.DataFrame:
    """
    Stratified sample balanced across ground truth classes:
      - 20 hard negation concepts (No/Not/Lack prefix, ground_truth=1)
      - 15 high-FN concepts (known hard features, ground_truth=1)
      - 15 easy positives (ground_truth=1, predicted=1)
      - 15 true negatives (ground_truth=0, predicted=0)
      - 15 false positives (predicted=1, ground_truth=0)
    Balanced GT distribution ensures valid kappa vs expert labels.
    """
    norag_df = norag_df.copy()
    norag_df["has_note"] = norag_df.apply(
        lambda r: (int(r.case_num), int(r.pn_num)) in notes, axis=1
    )
    pool = norag_df[norag_df.has_note].copy()

    neg_mask = pool.concept.str.lower().str.startswith(
        ("no ", "not ", "lack", "deny", "denies", "without", "absent")
    )
    hard_features = {
        "20911", "20813", "20215", "20209", "20109",
        "20607", "20405", "20510", "20515",
    }

    strata = {
        "neg_hard":  pool[neg_mask & (pool.ground_truth == 1)],
        "high_fn":   pool[pool.feature_num.astype(str).isin(hard_features) & (pool.ground_truth == 1)],
        "easy_pos":  pool[(pool.predicted == 1) & (pool.ground_truth == 1)],
        "true_neg":  pool[(pool.predicted == 0) & (pool.ground_truth == 0)],
        "false_pos": pool[(pool.predicted == 1) & (pool.ground_truth == 0)],
    }
    counts = {"neg_hard": 20, "high_fn": 15, "easy_pos": 15, "true_neg": 15, "false_pos": 15}

    parts = []
    for key, sub in strata.items():
        k = min(counts[key], len(sub))
        parts.append(sub.sample(k, random_state=42))

    sample = pd.concat(parts).drop_duplicates(subset=["pn_num", "feature_num"])
    sample = sample.head(n).reset_index(drop=True)
    pos_rate = (sample.ground_truth == 1).mean()
    print(f"Sample size: {len(sample)}  GT positive rate: {pos_rate:.1%}")
    return sample


# ── Judge ──────────────────────────────────────────────────────────────────────
async def judge_one(note: str, concept: str, feature_num: str) -> JudgeVerdict | None:
    prompt = (
        f"Patient history note:\n{note}\n\n"
        f"---\n"
        f"Clinical concept to evaluate (feature {feature_num}): {concept}\n\n"
        f"Is this concept documented in the note above?"
    )
    try:
        result = await judge_agent.run(prompt)
        return result.output
    except Exception as e:
        print(f"  [error] {feature_num}: {e}")
        return None


async def run_judge(sample: pd.DataFrame, notes: dict) -> list[dict]:
    rows = []
    total = len(sample)
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        note = notes[(int(row.case_num), int(row.pn_num))]
        verdict = await judge_one(note, str(row.concept), str(row.feature_num))
        if verdict is None:
            continue
        rows.append({
            "pn_num":        row.pn_num,
            "case_num":      row.case_num,
            "feature_num":   row.feature_num,
            "concept":       row.concept,
            "ground_truth":  row.ground_truth,
            "pred_norag":    row.predicted,
            "judge_present": int(verdict.present),
            "judge_reason":  verdict.reasoning,
        })
        status = "✓" if verdict.present == bool(row.ground_truth) else "✗"
        print(f"  [{i:2}/{total}] {status} {row.concept[:45]:<45}  judge={'Y' if verdict.present else 'N'}  gt={row.ground_truth}")

    return rows


# ── Kappa ──────────────────────────────────────────────────────────────────────
def cohen_kappa(a, b):
    a, b = np.array(a, dtype=int), np.array(b, dtype=int)
    n  = len(a)
    po = (a == b).sum() / n
    pa = a.mean(); pb = b.mean()
    pe = pa*pb + (1-pa)*(1-pb)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    out_path = Path("scripts/judge_results.csv")

    print("Loading data...")
    norag = pd.read_csv("scripts/results_no_rag.csv")
    for col in ("predicted", "ground_truth", "case_num"):
        norag[col] = pd.to_numeric(norag[col], errors="coerce")
    norag.dropna(subset=["predicted", "ground_truth", "case_num"], inplace=True)
    norag["predicted"]    = norag["predicted"].astype(int)
    norag["ground_truth"] = norag["ground_truth"].astype(int)
    norag["case_num"]     = norag["case_num"].astype(int)

    notes = load_notes()
    print(f"Loaded {len(notes):,} notes")

    sample = build_sample(norag, notes, n=60)

    print(f"\nRunning LLM judge on {len(sample)} items...")
    rows = asyncio.run(run_judge(sample, notes))

    if not rows:
        print("No results — check VPN connection.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    gt  = df.ground_truth.tolist()
    sys = df.pred_norag.tolist()
    jdg = df.judge_present.tolist()

    print("\n=== KAPPA RESULTS ===")
    print(f"  No-RAG system  vs expert labels : κ = {cohen_kappa(sys, gt)}")
    print(f"  LLM Judge      vs expert labels : κ = {cohen_kappa(jdg, gt)}")
    print(f"  No-RAG system  vs LLM Judge     : κ = {cohen_kappa(sys, jdg)}")

    acc_sys = (np.array(sys) == np.array(gt)).mean()
    acc_jdg = (np.array(jdg) == np.array(gt)).mean()
    print(f"\n  No-RAG accuracy on sample : {acc_sys:.4f}")
    print(f"  Judge  accuracy on sample : {acc_jdg:.4f}")

    print("\n=== BREAKDOWN BY CONCEPT TYPE ===")
    neg_mask = df.concept.str.lower().str.startswith(
        ("no ", "not ", "lack", "deny", "denies", "without", "absent")
    )
    for label, mask in [("Negation concepts", neg_mask), ("Positive concepts", ~neg_mask)]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        k_sys = cohen_kappa(sub.pred_norag, sub.ground_truth)
        k_jdg = cohen_kappa(sub.judge_present, sub.ground_truth)
        print(f"  {label} (n={len(sub)}): system κ={k_sys}  judge κ={k_jdg}")


if __name__ == "__main__":
    main()
