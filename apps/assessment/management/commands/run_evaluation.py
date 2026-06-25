"""Django management command: run the ClinNoteRAG evaluation.

Usage:
    python manage.py run_evaluation                              # agentic RAG, all notes
    python manage.py run_evaluation --strategy naive_rag
    python manage.py run_evaluation --strategy no_rag
    python manage.py run_evaluation --limit 50                  # quick test
    python manage.py run_evaluation --resume-from-case 206      # skip earlier cases
    python manage.py run_evaluation --to-case 205               # stop after case 205

Results are saved to the Django DB (EvaluationRun + ConceptVerdict records) so the
web dashboard at /runs/<id>/ shows live data, and also exported to CSV in scripts/.
"""

import asyncio
import csv
import time
from collections import defaultdict
from pathlib import Path

import chromadb
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from sklearn.metrics import precision_recall_fscore_support

from apps.assessment.models import ConceptVerdict as ConceptVerdictModel
from apps.assessment.models import EvaluationRun
from services.agent import ConceptVerdict, assess_note
from services.naive_rag import assess_note_naive
from services.no_rag import assess_note_no_rag

DATA_DIR = settings.NBME_DATA_DIR
OUT_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "scripts"
DELAY_SECONDS = 0.3

RESULTS_FIELDS = [
    "pn_num", "case_num", "feature_num", "concept",
    "predicted", "ground_truth", "evidence",
]


def _load_data():
    features_df = pd.read_csv(DATA_DIR / "NBME_PN_HISTORY_FEATURES.txt", sep="|")
    annot_df    = pd.read_csv(DATA_DIR / "NBME_PN_HISTORY_ANNOTATIONS.txt", sep="|")
    notes_df    = pd.read_csv(DATA_DIR / "NBME_PN_HISTORY.txt", sep="|")
    return features_df, annot_df, notes_df


def _build_case_feature_map(features_df) -> dict[int, list[str]]:
    return {
        int(case_num): group["FEATURE_NUM"].astype(int).astype(str).tolist()
        for case_num, group in features_df.groupby("CASE_NUM")
    }


def _build_concept_names(features_df) -> dict[int, dict[str, str]]:
    result = {}
    for case_num, group in features_df.groupby("CASE_NUM"):
        names = {}
        for _, row in group.iterrows():
            fn = str(int(row["FEATURE_NUM"]))
            primary = str(row["FEATURE_TEXT"]).split("-OR-")[0].replace("-", " ").strip()
            names[fn] = primary
        result[int(case_num)] = names
    return result


def _build_present_set(annot_df) -> set[tuple[str, str]]:
    return set(zip(annot_df["PN_NUM"].astype(str), annot_df["FEATURE_NUM"].astype(int).astype(str)))


def _build_annotated_pairs(annot_df) -> pd.DataFrame:
    return (
        annot_df[["PN_NUM", "CASE_NUM"]]
        .drop_duplicates()
        .sort_values(["CASE_NUM", "PN_NUM"])
        .reset_index(drop=True)
    )


async def _run_evaluation(
    strategy, collection, case_feature_map, concept_names_map,
    present_set, notes_df, annot_df, run_record, results_csv,
    limit=None, resume_from_case=None, to_case=None, stdout=None,
):
    def log(msg):
        if stdout:
            stdout.write(msg)

    notes_lookup = {
        (str(row["PN_NUM"]), int(row["CASE_NUM"])): str(row["PN_HISTORY"])
        for _, row in notes_df.iterrows()
    }

    pairs = _build_annotated_pairs(annot_df)
    if resume_from_case:
        pairs = pairs[pairs["CASE_NUM"] >= resume_from_case].reset_index(drop=True)
    if to_case:
        pairs = pairs[pairs["CASE_NUM"] <= to_case].reset_index(drop=True)
    if limit:
        pairs = pairs.head(limit)

    total = len(pairs)
    log(f"Evaluating {total} notes...")
    log(f"Results CSV: {results_csv}\n")

    y_true_all, y_pred_all = [], []
    per_case = defaultdict(lambda: {"y_true": [], "y_pred": []})
    db_verdicts = []

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
                log(f"  WARNING: No note for pn_num={pn_num} case={case_num}")
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
                else:
                    verdicts = await assess_note_no_rag(
                        note_text=note_text, case_num=case_num,
                        feature_nums=feature_nums,
                        concept_names=concept_names_map[case_num],
                    )
            except Exception as exc:
                log(f"  ERROR note {pn_num} case {case_num}: {exc}")
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

                db_verdicts.append(ConceptVerdictModel(
                    run=run_record,
                    pn_num=pn_num,
                    case_num=case_num,
                    feature_num=v.feature_num,
                    concept=v.concept,
                    predicted=bool(v.present),
                    ground_truth=bool(true),
                    evidence=v.evidence or "",
                ))

            results_file.flush()

            # Flush DB verdicts every 50 notes to avoid huge in-memory batch
            if len(db_verdicts) >= 50 * len(feature_nums):
                ConceptVerdictModel.objects.bulk_create(db_verdicts)
                db_verdicts.clear()

            if (i + 1) % 10 == 0 and y_true_all:
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true_all, y_pred_all, average="binary", zero_division=0
                )
                log(f"  [{i+1:>4}/{total}] F1={f1:.4f}  P={p:.4f}  R={r:.4f}")

            if i < total - 1:
                await asyncio.sleep(DELAY_SECONDS)

    finally:
        results_file.close()
        if db_verdicts:
            ConceptVerdictModel.objects.bulk_create(db_verdicts)

    return y_true_all, y_pred_all, per_case


class Command(BaseCommand):
    help = "Run ClinNoteRAG evaluation and save results to DB + CSV"

    def add_arguments(self, parser):
        parser.add_argument(
            "--strategy",
            choices=["agentic_rag", "naive_rag", "no_rag"],
            default="agentic_rag",
        )
        parser.add_argument("--limit", type=int, default=None,
                            help="Evaluate only first N notes")
        parser.add_argument("--resume-from-case", type=int, default=None,
                            help="Skip cases before this number, append to existing CSV")
        parser.add_argument("--to-case", type=int, default=None,
                            help="Stop after this case number (inclusive)")

    def handle(self, *args, **options):
        strategy = options["strategy"]
        results_csv = OUT_DIR / f"results_{strategy}.csv"
        metrics_csv = OUT_DIR / f"metrics_{strategy}.csv"

        self.stdout.write(f"ClinNoteRAG — Evaluation [{strategy}]")
        self.stdout.write("=" * 50)

        client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        collection = client.get_collection("nbme_concepts")

        features_df, annot_df, notes_df = _load_data()
        case_feature_map  = _build_case_feature_map(features_df)
        concept_names_map = _build_concept_names(features_df)
        present_set       = _build_present_set(annot_df)

        run_record = EvaluationRun.objects.create(
            strategy=strategy,
            llm_model=settings.CAIR_LLM_MODEL,
            notes_evaluated=0,
        )
        self.stdout.write(f"Created EvaluationRun id={run_record.pk}")

        start = time.time()
        y_true_all, y_pred_all, per_case = asyncio.run(_run_evaluation(
            strategy=strategy,
            collection=collection,
            case_feature_map=case_feature_map,
            concept_names_map=concept_names_map,
            present_set=present_set,
            notes_df=notes_df,
            annot_df=annot_df,
            run_record=run_record,
            results_csv=results_csv,
            limit=options["limit"],
            resume_from_case=options["resume_from_case"],
            to_case=options["to_case"],
            stdout=self.stdout,
        ))
        elapsed = time.time() - start

        self.stdout.write(f"\n{'=' * 50}")
        self.stdout.write(f"RESULTS [{strategy}]")
        self.stdout.write("=" * 50)

        metric_rows = []
        for case_num in sorted(per_case):
            yt = per_case[case_num]["y_true"]
            yp = per_case[case_num]["y_pred"]
            p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
            self.stdout.write(f"  Case {case_num:<6} P={p:.4f}  R={r:.4f}  F1={f1:.4f}  (n={len(yt)})")
            metric_rows.append({"label": f"Case {case_num}", "precision": float(p),
                                 "recall": float(r), "f1": float(f1), "n": len(yt)})

        self.stdout.write("  " + "-" * 48)
        p_all, r_all, f1_all, _ = precision_recall_fscore_support(
            y_true_all, y_pred_all, average="binary", zero_division=0
        )
        self.stdout.write(
            f"  {'OVERALL':<22} P={p_all:.4f}  R={r_all:.4f}  F1={f1_all:.4f}  (n={len(y_true_all)})"
        )
        metric_rows.append({"label": "OVERALL", "precision": float(p_all),
                             "recall": float(r_all), "f1": float(f1_all), "n": len(y_true_all)})

        # Update DB record with final metrics
        run_record.precision = float(p_all)
        run_record.recall = float(r_all)
        run_record.f1 = float(f1_all)
        run_record.notes_evaluated = len(set(
            (str(pair_row["PN_NUM"]), int(pair_row["CASE_NUM"]))
            for _, pair_row in _build_annotated_pairs(annot_df).iterrows()
        ))
        run_record.save()

        pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)
        self.stdout.write(f"\nCompleted in {elapsed:.1f}s")
        self.stdout.write(self.style.SUCCESS(f"Saved: {results_csv}"))
        self.stdout.write(self.style.SUCCESS(f"Saved: {metrics_csv}"))
        self.stdout.write(self.style.SUCCESS(f"EvaluationRun id={run_record.pk} updated in DB"))
