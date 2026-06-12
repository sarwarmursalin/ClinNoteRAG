from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pandas as pd
from django.conf import settings
from django.shortcuts import render

from .models import ConceptVerdict, EvaluationRun

_CASE_TOPICS = {
    0: "Heart palpitations",
    1: "Abdominal pain",
    2: "Irregular menses",
    3: "Epigastric pain",
    4: "Anxiety / insomnia",
    5: "Panic attacks",
    6: "Chest pain / pleuritic",
    7: "Weight gain / fatigue",
    8: "Sleep disturbance / grief",
    9: "Headache",
}


def _get_chroma():
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    return client.get_collection("nbme_concepts")


def _load_features() -> pd.DataFrame:
    return pd.read_csv(settings.NBME_DATA_DIR / "features.csv")


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def index(request):
    """Dashboard: evaluation runs + case overview."""
    runs = EvaluationRun.objects.all()

    # Case summary from ChromaDB
    cases = []
    try:
        collection = _get_chroma()
        features_df = _load_features()
        for case_num in range(10):
            count = len(features_df[features_df["case_num"] == case_num])
            cases.append({
                "case_num": case_num,
                "topic": _CASE_TOPICS.get(case_num, ""),
                "concept_count": count,
            })
    except Exception:
        cases = []

    return render(request, "assessment/index.html", {
        "runs": runs,
        "cases": cases,
    })


def case_concepts(request, case_num: int):
    """Browse all concepts for a case."""
    try:
        collection = _get_chroma()
        features_df = _load_features()
        case_df = features_df[features_df["case_num"] == case_num].reset_index(drop=True)

        concepts = []
        for _, row in case_df.iterrows():
            fn = str(row["feature_num"])
            result = collection.get(ids=[f"{case_num}_{fn}"])
            doc = result["documents"][0] if result["documents"] else str(row["feature_text"])
            concepts.append({
                "feature_num": fn,
                "raw_text": row["feature_text"],
                "document": doc,
            })
    except Exception as e:
        concepts = []

    return render(request, "assessment/concepts.html", {
        "case_num": case_num,
        "topic": _CASE_TOPICS.get(case_num, ""),
        "concepts": concepts,
    })


def run_detail(request, run_id: int):
    """Show per-case F1 breakdown for one evaluation run."""
    run = EvaluationRun.objects.get(pk=run_id)
    verdicts = ConceptVerdict.objects.filter(run=run)

    # Per-case metrics
    case_stats = {}
    for v in verdicts:
        s = case_stats.setdefault(v.case_num, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        pred, gt = v.predicted, v.ground_truth
        if pred and gt:
            s["tp"] += 1
        elif pred and not gt:
            s["fp"] += 1
        elif not pred and gt:
            s["fn"] += 1
        else:
            s["tn"] += 1

    case_metrics = []
    for case_num in sorted(case_stats):
        s = case_stats[case_num]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        case_metrics.append({
            "case_num": case_num,
            "topic": _CASE_TOPICS.get(case_num, ""),
            "tp": tp, "fp": fp, "fn": fn, "tn": s["tn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
        })

    return render(request, "assessment/run_detail.html", {
        "run": run,
        "case_metrics": case_metrics,
    })
