"""Evaluate ClinNoteRAG against NBME annotated notes.

Supports three strategies for the ablation study:
  agentic_rag  — per-concept tool-use retrieval (default)
  naive_rag    — bulk retrieval, single prompt
  no_rag       — concept names only, zero retrieval

Results saved incrementally after every note so a kill never loses data.
Each strategy writes to its own output files.

Run:
    python -u scripts/evaluate.py                              # agentic RAG, all notes
    python -u scripts/evaluate.py --strategy naive_rag         # naive RAG baseline
    python -u scripts/evaluate.py --strategy no_rag            # no-RAG baseline
    python -u scripts/evaluate.py --limit 100                  # quick test
    python -u scripts/evaluate.py --resume-from-case 206       # resume interrupted run
    python -u scripts/evaluate.py --to-case 205                # stop after case 205
"""

import argparse
import asyncio
import csv
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
from services.naive_rag import assess_note_naive
from services.no_rag import assess_note_no_rag

DATA_DIR = settings.NBME_DATA_DIR
OUT_DIR = Path(__file__).parent
DELAY_SECONDS = 0.3

RESULTS_FIELDS = ["pn_num", "case_num", "feature_num", "concept",
                  "predicted", "ground_truth", "evidence"]


def output_paths(strategy: str):
    return (
        OUT_DIR / f"results_{strategy}.csv",
        OUT_DIR / f"metrics_{strategy}.csv",
    )


def load_data():
    features_df = pd.read_csv(DATA_DIR / "NBME_PN_HISTORY_FEATURES.txt", sep="|")
    annot_df    = pd.read_csv(DATA_DIR / "NBME_PN_HISTORY_ANNOTATIONS.txt", sep="|")
    notes_df    = pd.read_csv(DATA_DIR / "NBME_PN_HISTORY.txt", sep="|")
    return features_df, annot_df, notes_df


def build_case_feature_map(features_df) -> dict[int, list[str]]:
    result = {}
    for case_num, group in features_df.groupby("CASE_NUM"):
        result[int(case_num)] = group["FEATURE_NUM"].astype(int).astype(str).tolist()
    return result


def build_concept_names(features_df) -> dict[int, dict[str, str]]:
    """Returns {case_num: {feature_num: primary_concept_name}}."""
    result = {}
    for case_num, group in features_df.groupby("CASE_NUM"):
        names = {}
        for _, row in group.iterrows():
            fn = str(int(row["FEATURE_NUM"]))
            primary = str(row["FEATURE_TEXT"]).split("-OR-")[0].replace("-", " ").strip()
            names[fn] = primary
        result[int(case_num)] = names
    return result


def build_ground_truth(annot_df) -> set[tuple[str, str]]:
    return set(
        zip(annot_df["PN_NUM"].astype(str), annot_df["FEATURE_NUM"].astype(int).astype(str))
    )


def build_annotated_pairs(annot_df) -> pd.DataFrame:
    return (
        annot_df[["PN_NUM", "CASE_NUM"]]
        .drop_duplicates()
        .sort_values(["CASE_NUM", "PN_NUM"])
        .reset_index(drop=True)
    )


async def evaluate_all(strategy, collection, case_feature_map, concept_names_map,
                       present_set, notes_df, annot_df, results_csv,
                       limit=None, resume_from_case=None, to_case=None):

    notes_lookup = {
        (str(row["PN_NUM"]), int(row["CASE_NUM"])): str(row["PN_HISTORY"])
        for _, row in notes_df.iterrows()
    }

    pairs = build_annotated_pairs(annot_df)
    if resume_from_case:
        pairs = pairs[pairs["CASE_NUM"] >= resume_from_case].reset_index(drop=True)
    if to_case:
        pairs = pairs[pairs["CASE_NUM"] <= to_case].reset_index(drop=True)
    if limit:
        pairs = pairs.head(limit)

    total = len(pairs)
    print(f"Evaluating {total} notes...", flush=True)
    print(f"Results saved live to: {results_csv}\n", flush=True)

    y_true_all, y_pred_all = [], []
    per_case = defaultdict(lambda: {"y_true": [], "y_pred": []})

    file_mode = "a" if resume_from_case else "w"
    results_file = open(results_csv, file_mode, newline="", buffering=1)
    writer = csv.DictWriter(results_file, fieldnames=RESULTS_FIELDS)
    if file_mode == "w":
        writer.writeheader()
    results_file.flush()

    try:
        for i, (_, pair_row) in enumerate(pairs.iterrows()):
            pn_num   = str(int(pair_row["PN_NUM"]))
            case_num = int(pair_row["CASE_NUM"])
            feature_nums = case_feature_map[case_num]
            note_text = notes_lookup.get((pn_num, case_num), "")

            if not note_text:
                print(f"  WARNING: No note for pn_num={pn_num} case={case_num}", flush=True)
                continue

            try:
                if strategy == "agentic_rag":
                    verdicts = await assess_note(
                        note_text=note_text, case_num=case_num,
                        feature_nums=feature_nums, chroma_collection=collection,
                    )
                elif strategy == "naive_rag":
                    verdicts = await assess_note_naive(
                        note_text=note_text, case_num=case_num,
                        feature_nums=feature_nums, chroma_collection=collection,
                    )
                else:  # no_rag
                    verdicts = await assess_note_no_rag(
                        note_text=note_text, case_num=case_num,
                        feature_nums=feature_nums,
                        concept_names=concept_names_map[case_num],
                    )
            except Exception as exc:
                print(f"  ERROR note {pn_num} case {case_num}: {exc}", flush=True)
                verdicts = [
                    ConceptVerdict(feature_num=fn, concept="", present=False, evidence=None)
                    for fn in feature_nums
                ]

            for v in verdicts:
                pred = 1 if v.present else 0
                true = 1 if (pn_num, v.feature_num) in present_set else 0
                y_true_all.append(true)
                y_pred_all.append(pred)
                per_case[case_num]["y_true"].append(true)
                per_case[case_num]["y_pred"].append(pred)
                writer.writerow({
                    "pn_num": pn_num, "case_num": case_num,
                    "feature_num": v.feature_num, "concept": v.concept,
                    "predicted": pred, "ground_truth": true,
                    "evidence": v.evidence or "",
                })

            results_file.flush()

            if (i + 1) % 10 == 0 and y_true_all:
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true_all, y_pred_all, average="binary", zero_division=0
                )
                print(f"  [{i+1:>4}/{total}] F1={f1:.4f}  P={p:.4f}  R={r:.4f}", flush=True)

            if i < total - 1:
                await asyncio.sleep(DELAY_SECONDS)

    finally:
        results_file.close()

    return y_true_all, y_pred_all, per_case


def compute_and_print_metrics(y_true, y_pred, label):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"  {label:<22} P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (n={len(y_true)})", flush=True)
    return {"label": label, "precision": float(p), "recall": float(r),
            "f1": float(f1), "n": len(y_true)}


async def main():
    parser = argparse.ArgumentParser(description="ClinNoteRAG Evaluation")
    parser.add_argument("--strategy", choices=["agentic_rag", "naive_rag", "no_rag"],
                        default="agentic_rag", help="Evaluation strategy (default: agentic_rag)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N notes")
    parser.add_argument("--resume-from-case", type=int, default=None,
                        help="Skip cases before this number and append to existing results")
    parser.add_argument("--to-case", type=int, default=None,
                        help="Stop after this case number (inclusive)")
    args = parser.parse_args()

    results_csv, metrics_csv = output_paths(args.strategy)

    print(f"ClinNoteRAG — Evaluation Harness [{args.strategy}]", flush=True)
    print("=" * 50, flush=True)
    if args.limit:
        print(f"Mode: first {args.limit} notes only", flush=True)
    if args.resume_from_case:
        print(f"Resuming from case {args.resume_from_case}", flush=True)
    if args.to_case:
        print(f"Stopping after case {args.to_case}", flush=True)
    print("", flush=True)

    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    collection = client.get_collection("nbme_concepts")

    features_df, annot_df, notes_df = load_data()
    case_feature_map  = build_case_feature_map(features_df)
    concept_names_map = build_concept_names(features_df)
    present_set       = build_ground_truth(annot_df)

    start = time.time()
    y_true_all, y_pred_all, per_case = await evaluate_all(
        strategy=args.strategy,
        collection=collection,
        case_feature_map=case_feature_map,
        concept_names_map=concept_names_map,
        present_set=present_set,
        notes_df=notes_df,
        annot_df=annot_df,
        results_csv=results_csv,
        limit=args.limit,
        resume_from_case=args.resume_from_case,
        to_case=args.to_case,
    )
    elapsed = time.time() - start

    print(f"\n{'=' * 50}", flush=True)
    print(f"RESULTS [{args.strategy}]", flush=True)
    print("=" * 50, flush=True)

    metric_rows = []
    for case_num in sorted(per_case):
        m = compute_and_print_metrics(
            per_case[case_num]["y_true"], per_case[case_num]["y_pred"], f"Case {case_num}"
        )
        metric_rows.append(m)

    print("  " + "-" * 48, flush=True)
    overall = compute_and_print_metrics(y_true_all, y_pred_all, "OVERALL")
    metric_rows.append(overall)

    print(f"\nCompleted in {elapsed:.1f}s", flush=True)
    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)
    print(f"Saved: {results_csv}", flush=True)
    print(f"Saved: {metrics_csv}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
