"""Django management command: import a results CSV into the Django DB.

Usage:
    python manage.py import_results --csv scripts/results_naive_rag.csv --strategy naive_rag
    python manage.py import_results --csv scripts/results_agentic_rag.csv --strategy agentic_rag
    python manage.py import_results --csv scripts/results_no_rag.csv --strategy no_rag

Rows with non-numeric case_num (garbage from interrupted LLM calls) are skipped
automatically.
"""

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from sklearn.metrics import precision_recall_fscore_support

from apps.assessment.models import ConceptVerdict as ConceptVerdictModel
from apps.assessment.models import EvaluationRun

BATCH_SIZE = 500


class Command(BaseCommand):
    help = "Import a results CSV into the Django DB"

    def add_arguments(self, parser):
        parser.add_argument("--csv", required=True, help="Path to results CSV")
        parser.add_argument(
            "--strategy",
            choices=["agentic_rag", "naive_rag", "no_rag"],
            required=True,
        )
        parser.add_argument(
            "--model",
            default=settings.CAIR_LLM_MODEL,
            help="LLM model name to record (default: from settings)",
        )

    def handle(self, *args, **options):
        csv_path = options["csv"]
        strategy = options["strategy"]

        self.stdout.write(f"Importing {csv_path} as strategy={strategy}")

        df = pd.read_csv(csv_path)
        original_count = len(df)

        # Keep only rows where case_num and pn_num are numeric
        df = df[pd.to_numeric(df["case_num"], errors="coerce").notna()]
        df = df[pd.to_numeric(df["pn_num"], errors="coerce").notna()]
        df["case_num"] = df["case_num"].astype(int)
        df["pn_num"] = df["pn_num"].astype(str)
        df["predicted"] = df["predicted"].astype(int)
        df["ground_truth"] = df["ground_truth"].astype(int)

        skipped = original_count - len(df)
        if skipped:
            self.stdout.write(self.style.WARNING(f"Skipped {skipped} malformed rows"))

        self.stdout.write(f"Importing {len(df)} verdict rows...")

        p, r, f1, _ = precision_recall_fscore_support(
            df["ground_truth"].tolist(), df["predicted"].tolist(),
            average="binary", zero_division=0,
        )
        notes_evaluated = df[["pn_num", "case_num"]].drop_duplicates().shape[0]

        run_record = EvaluationRun.objects.create(
            strategy=strategy,
            llm_model=options["model"],
            notes_evaluated=notes_evaluated,
            precision=float(p),
            recall=float(r),
            f1=float(f1),
        )
        self.stdout.write(f"Created EvaluationRun id={run_record.pk}  F1={f1:.4f}")

        # Bulk insert in batches
        batch = []
        for _, row in df.iterrows():
            batch.append(ConceptVerdictModel(
                run=run_record,
                pn_num=str(row["pn_num"]),
                case_num=int(row["case_num"]),
                feature_num=str(row["feature_num"]),
                concept=str(row.get("concept", "") or ""),
                predicted=bool(int(row["predicted"])),
                ground_truth=bool(int(row["ground_truth"])),
                evidence=str(row.get("evidence", "") or ""),
            ))
            if len(batch) >= BATCH_SIZE:
                ConceptVerdictModel.objects.bulk_create(batch)
                batch.clear()

        if batch:
            ConceptVerdictModel.objects.bulk_create(batch)

        self.stdout.write(self.style.SUCCESS(
            f"Done. EvaluationRun id={run_record.pk}  "
            f"P={p:.4f}  R={r:.4f}  F1={f1:.4f}  "
            f"n_verdicts={len(df)}  n_notes={notes_evaluated}"
        ))
