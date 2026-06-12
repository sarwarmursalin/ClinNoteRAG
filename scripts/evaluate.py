"""Task 3 — Evaluate the agent against 1,000 expert-annotated NBME notes.

Runs assess_note() on every (pn_num, case_num) pair in train.csv,
compares agent verdicts against expert physician labels, and computes
Precision / Recall / F1 per case and overall.

Output files (written to scripts/):
    results.csv         — one row per (note, concept)
    metrics_summary.csv — P/R/F1 per case + overall

Run:
    python scripts/evaluate.py
"""

import ast
import asyncio
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

import chromadb
import pandas as pd
from django.conf import settings
from sklearn.metrics import precision_recall_fscore_support

from services.agent import ConceptVerdict, assess_note

DATA_DIR = settings.NBME_DATA_DIR
OUT_DIR = Path(__file__).parent
DELAY_SECONDS = 0.3  # pause between notes to avoid API rate limits


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features_df = pd.read_csv(DATA_DIR / "features.csv")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    notes_df = pd.read_csv(DATA_DIR / "patient_notes.csv")
    return features_df, train_df, notes_df


def build_case_feature_map(features_df: pd.DataFrame) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for case_num, group in features_df.groupby("case_num"):
        result[int(case_num)] = group["feature_num"].astype(str).tolist()
    return result


def build_ground_truth(train_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Returns {(pn_num, feature_num): 0_or_1}."""
    gt: dict[tuple[str, str], int] = {}
    for _, row in train_df.iterrows():
        try:
            annotation = ast.literal_eval(str(row["annotation"]))
        except (ValueError, SyntaxError):
            annotation = []
        gt[(str(row["pn_num"]), str(row["feature_num"]))] = 1 if annotation else 0
    return gt


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


async def evaluate_all(
    collection: chromadb.Collection,
    case_feature_map: dict[int, list[str]],
    ground_truth: dict[tuple[str, str], int],
    notes_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[list[dict], list[int], list[int], dict]:
    notes_lookup = {
        (str(row["pn_num"]), int(row["case_num"])): str(row["pn_history"])
        for _, row in notes_df.iterrows()
    }

    pairs = (
        train_df[["pn_num", "case_num"]]
        .drop_duplicates()
        .sort_values(["case_num", "pn_num"])
        .reset_index(drop=True)
    )

    rows: list[dict] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    per_case: dict[int, dict] = defaultdict(lambda: {"y_true": [], "y_pred": []})

    total = len(pairs)
    print(f"Evaluating {total} notes...\n")

    for i, (_, pair_row) in enumerate(pairs.iterrows()):
        pn_num = str(pair_row["pn_num"])
        case_num = int(pair_row["case_num"])
        feature_nums = case_feature_map[case_num]
        note_text = notes_lookup.get((pn_num, case_num), "")

        if not note_text:
            print(f"  WARNING: No note for pn_num={pn_num} case={case_num}")
            continue

        try:
            verdicts = await assess_note(
                note_text=note_text,
                case_num=case_num,
                feature_nums=feature_nums,
                chroma_collection=collection,
            )
        except Exception as exc:
            print(f"  ERROR note {pn_num} case {case_num}: {exc}")
            verdicts = [
                ConceptVerdict(feature_num=fn, concept="", present=False, evidence=None)
                for fn in feature_nums
            ]

        for v in verdicts:
            pred = 1 if v.present else 0
            true = ground_truth.get((pn_num, v.feature_num), 0)

            y_true_all.append(true)
            y_pred_all.append(pred)
            per_case[case_num]["y_true"].append(true)
            per_case[case_num]["y_pred"].append(pred)

            rows.append({
                "pn_num": pn_num,
                "case_num": case_num,
                "feature_num": v.feature_num,
                "concept": v.concept,
                "predicted": pred,
                "ground_truth": true,
                "evidence": v.evidence or "",
            })

        if (i + 1) % 50 == 0:
            p, r, f1, _ = precision_recall_fscore_support(
                y_true_all, y_pred_all, average="binary", zero_division=0
            )
            print(f"  [{i + 1}/{total}] running F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

        if i < total - 1:
            await asyncio.sleep(DELAY_SECONDS)

    return rows, y_true_all, y_pred_all, per_case


# ---------------------------------------------------------------------------
# Metrics printing
# ---------------------------------------------------------------------------


def compute_metrics(y_true: list[int], y_pred: list[int], label: str) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"  {label:<22} P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (n={len(y_true)})")
    return {"label": label, "precision": float(p), "recall": float(r), "f1": float(f1), "n": len(y_true)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    print("ClinNoteRAG — Evaluation Harness")
    print("=" * 50)

    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    collection = client.get_collection("nbme_concepts")

    features_df, train_df, notes_df = load_data()
    case_feature_map = build_case_feature_map(features_df)
    ground_truth = build_ground_truth(train_df)

    start = time.time()
    rows, y_true_all, y_pred_all, per_case = await evaluate_all(
        collection=collection,
        case_feature_map=case_feature_map,
        ground_truth=ground_truth,
        notes_df=notes_df,
        train_df=train_df,
    )
    elapsed = time.time() - start

    print(f"\n{'=' * 50}")
    print("RESULTS")
    print("=" * 50)

    metric_rows = []
    for case_num in sorted(per_case):
        m = compute_metrics(per_case[case_num]["y_true"], per_case[case_num]["y_pred"], f"Case {case_num}")
        metric_rows.append(m)

    print("  " + "-" * 48)
    overall = compute_metrics(y_true_all, y_pred_all, "OVERALL")
    metric_rows.append(overall)

    print(f"\nCompleted in {elapsed:.1f}s")

    results_path = OUT_DIR / "results.csv"
    metrics_path = OUT_DIR / "metrics_summary.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f"Saved: {results_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    asyncio.run(main())
