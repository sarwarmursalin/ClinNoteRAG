from __future__ import annotations

import asyncio
import json
from functools import wraps
from pathlib import Path

import chromadb
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect, render

from services.naive_rag import assess_note_naive

from .forms import CASE_DESCRIPTIONS, LoginForm, NoteEvaluationForm, StudentRegistrationForm
from .models import ConceptVerdict, EvaluationRun, UserProfile

_CASE_TOPICS = {
    201: "Irregular menses (44F)",
    202: "Epigastric discomfort (35M)",
    203: "Headache (20F)",
    204: "Sleep disturbance / grief (67F)",
    205: "Palpitations / heart racing (26F)",
    206: "Anxiety / nervousness (45F)",
    207: "Heavy periods / weight gain (35F)",
    208: "Right lower quadrant pain (20F)",
    209: "Chest pain / pleuritic (17M)",
    210: "Palpitations / heart pounding (17M)",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_chroma():
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    return client.get_collection("nbme_concepts")


def _load_features() -> pd.DataFrame:
    df = pd.read_csv(settings.NBME_DATA_DIR / "NBME_PN_HISTORY_FEATURES.txt", sep="|")
    df["CASE_NUM"]    = df["CASE_NUM"].astype(int)
    df["FEATURE_NUM"] = df["FEATURE_NUM"].astype(int).astype(str)
    return df


def faculty_required(view_fn):
    """Decorator: login required + must have faculty role."""
    @wraps(view_fn)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(f"{settings.LOGIN_URL}?next={request.path}")
        try:
            if not request.user.profile.is_faculty:
                return redirect("evaluate-note")
        except UserProfile.DoesNotExist:
            return redirect("evaluate-note")
        return view_fn(request, *args, **kwargs)
    return _wrapped


def superuser_required(view_fn):
    """Decorator: login required + superuser only (developers/admin)."""
    @wraps(view_fn)
    def _wrapped(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(f"{settings.LOGIN_URL}?next={request.path}")
        if not request.user.is_superuser:
            return redirect("home")
        return view_fn(request, *args, **kwargs)
    return _wrapped


# ── Auth views ───────────────────────────────────────────────────────────────

def login_view(request):
    if request.user.is_authenticated:
        return redirect("home")
    form = LoginForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())
        return redirect(request.GET.get("next") or "home")
    return render(request, "assessment/login.html", {"form": form})


def custom_404(request, exception=None):
    return render(request, "404.html", status=404)


def landing_view(request):
    """Public landing page — visible before login."""
    return render(request, "assessment/landing.html")


def about_view(request):
    """Public about page."""
    cases = [
        {"num": n, "topic": t} for n, t in _CASE_TOPICS.items()
    ]
    return render(request, "assessment/about.html", {"cases": cases})


def register_choice_view(request):
    if request.user.is_authenticated:
        return redirect("home")
    return render(request, "assessment/register_choice.html")


def register_view(request):
    if request.user.is_authenticated:
        return redirect("home")
    form = StudentRegistrationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        d = form.cleaned_data
        user = User.objects.create_user(
            username   = d["username"],
            password   = d["password1"],
            first_name = d["first_name"],
            last_name  = d["last_name"],
            email      = d.get("email", ""),
        )
        UserProfile.objects.create(user=user, role="student", student_id=d["student_id"])
        login(request, user)
        messages.success(request, f"Welcome, {user.first_name}! Your account has been created.")
        return redirect("home")
    return render(request, "assessment/register.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("login")


def home_view(request):
    """Role-based redirect after login."""
    if not request.user.is_authenticated:
        return redirect("login")
    try:
        if request.user.profile.is_faculty:
            return redirect("faculty-dashboard")
    except UserProfile.DoesNotExist:
        pass
    return redirect("evaluate-note")


# ── Faculty views ─────────────────────────────────────────────────────────────

@faculty_required
def faculty_dashboard(request):
    """All student submissions, paginated."""
    submissions = (
        EvaluationRun.objects
        .filter(note_text__gt="")          # student submissions have note text
        .exclude(student_name="")
        .order_by("-created_at")
    )

    submission_rows = []
    for run in submissions:
        verdicts = ConceptVerdict.objects.filter(run=run)
        total   = verdicts.count()
        present = verdicts.filter(predicted=True).count()
        pct     = round(present / total * 100) if total else 0
        submission_rows.append({
            "run": run, "score": present, "total": total, "pct": pct,
            "topic": _CASE_TOPICS.get(run.case_num, ""),
            "level": "high" if pct >= 75 else ("mid" if pct >= 50 else "low"),
        })

    research_runs = EvaluationRun.objects.filter(f1__isnull=False).order_by("-created_at")

    # Summary stats
    # Count distinct authenticated users + distinct legacy student names separately
    auth_students = submissions.filter(user__isnull=False).exclude(user__username="anonymous_student").values("user").distinct().count()
    legacy_students = submissions.filter(user__isnull=True).values("student_name").distinct().count()
    anon_students = submissions.filter(user__username="anonymous_student").values("student_name").distinct().count()
    unique_students = auth_students + legacy_students + anon_students
    all_pcts = [r["pct"] for r in submission_rows]
    avg_score = round(sum(all_pcts) / len(all_pcts)) if all_pcts else 0

    return render(request, "assessment/faculty_dashboard.html", {
        "submission_rows":  submission_rows,
        "research_runs":    research_runs,
        "unique_students":  unique_students,
        "total_submissions": len(submission_rows),
        "avg_score":        avg_score,
    })


@faculty_required
def run_detail(request, run_id: int):
    run      = EvaluationRun.objects.get(pk=run_id)
    verdicts = ConceptVerdict.objects.filter(run=run)

    case_stats = {}
    for v in verdicts:
        s = case_stats.setdefault(v.case_num, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        if v.predicted and v.ground_truth:       s["tp"] += 1
        elif v.predicted and not v.ground_truth: s["fp"] += 1
        elif not v.predicted and v.ground_truth: s["fn"] += 1
        else:                                    s["tn"] += 1

    case_metrics = []
    for case_num in sorted(case_stats):
        s = case_stats[case_num]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p  = tp / (tp + fp) if (tp + fp) else 0
        r  = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        case_metrics.append({
            "case_num": case_num, "topic": _CASE_TOPICS.get(case_num, ""),
            "tp": tp, "fp": fp, "fn": fn, "tn": s["tn"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        })

    return render(request, "assessment/run_detail.html", {"run": run, "case_metrics": case_metrics})


@faculty_required
def case_concepts(request, case_num: int):
    try:
        collection   = _get_chroma()
        features_df  = _load_features()
        case_df      = features_df[features_df["CASE_NUM"] == case_num].reset_index(drop=True)
        concepts = []
        for _, row in case_df.iterrows():
            fn     = str(row["FEATURE_NUM"])
            result = collection.get(ids=[f"{case_num}_{fn}"])
            doc    = result["documents"][0] if result["documents"] else str(row["FEATURE_TEXT"])
            concepts.append({"feature_num": fn, "raw_text": row["FEATURE_TEXT"], "document": doc})
    except Exception:
        concepts = []

    return render(request, "assessment/concepts.html", {
        "case_num": case_num,
        "topic":    _CASE_TOPICS.get(case_num, ""),
        "concepts": concepts,
    })


@superuser_required
def ablation_view(request):
    STRATEGIES = ["no_rag", "naive_rag", "agentic_rag"]
    LABELS     = {"no_rag": "No-RAG", "naive_rag": "Naive RAG", "agentic_rag": "Agentic RAG"}
    COLORS     = {
        "no_rag":      {"bg": "rgba(16,185,129,0.75)",  "border": "rgba(16,185,129,1)"},
        "naive_rag":   {"bg": "rgba(59,130,246,0.75)",  "border": "rgba(59,130,246,1)"},
        "agentic_rag": {"bg": "rgba(245,158,11,0.85)",  "border": "rgba(245,158,11,1)"},
    }

    def _compute(run):
        verdicts   = ConceptVerdict.objects.filter(run=run)
        case_data  = {}
        for v in verdicts:
            s = case_data.setdefault(v.case_num, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            if v.predicted and v.ground_truth:       s["tp"] += 1
            elif v.predicted and not v.ground_truth: s["fp"] += 1
            elif not v.predicted and v.ground_truth: s["fn"] += 1
            else:                                    s["tn"] += 1
        results = {}
        for case_num in sorted(case_data):
            s  = case_data[case_num]
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            p  = tp / (tp + fp) if (tp + fp) else 0
            r  = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            results[case_num] = round(f1, 4)
        return results, run

    strategy_data = {}
    for strat in STRATEGIES:
        qs = EvaluationRun.objects.filter(strategy=strat, f1__isnull=False).order_by("-f1")
        if qs.exists():
            best_run          = qs.first()
            case_f1s, run     = _compute(best_run)
            strategy_data[strat] = {
                "label":   LABELS[strat],
                "run":     run,
                "overall": run.f1 or 0,
                "case_f1": case_f1s,
                "color":   COLORS[strat],
            }

    all_cases = sorted({c for sd in strategy_data.values() for c in sd["case_f1"]}) or list(range(201, 211))
    case_rows = []
    for c in all_cases:
        row = {"case_num": c, "topic": _CASE_TOPICS.get(c, "")}
        best_f1, best_strat = -1, None
        for strat, sd in strategy_data.items():
            f1 = sd["case_f1"].get(c)
            row[strat] = f1
            if f1 is not None and f1 > best_f1:
                best_f1, best_strat = f1, strat
        row["best"] = best_strat
        case_rows.append(row)

    case_labels   = [f"Case {c}" for c in all_cases]
    chart_datasets = [
        {
            "label":           strategy_data[s]["label"],
            "data":            [strategy_data[s]["case_f1"].get(c, 0) for c in all_cases],
            "backgroundColor": strategy_data[s]["color"]["bg"],
            "borderColor":     strategy_data[s]["color"]["border"],
            "borderWidth":     1,
        }
        for s in STRATEGIES if s in strategy_data
    ]

    return render(request, "assessment/ablation.html", {
        "strategy_data": strategy_data,
        "strategies":    STRATEGIES,
        "labels":        LABELS,
        "case_rows":     case_rows,
        "chart_json":    json.dumps({"labels": case_labels, "datasets": chart_datasets}),
    })


# ── Student views ─────────────────────────────────────────────────────────────

@login_required
def evaluate_note_view(request):
    # Pull student info from profile
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = None

    initial = {}
    prefill_run_id = request.GET.get("from_run")
    if prefill_run_id:
        try:
            prev = EvaluationRun.objects.get(pk=prefill_run_id, user=request.user)
            initial = {"note_text": prev.note_text, "case_num": prev.case_num}
        except EvaluationRun.DoesNotExist:
            prefill_run_id = None

    if request.method == "POST":
        form = NoteEvaluationForm(request.POST)
        if form.is_valid():
            case_num  = int(form.cleaned_data["case_num"])
            note_text = form.cleaned_data["note_text"].strip()
            prev_run_id = request.POST.get("prev_run_id", "").strip()

            features_df  = _load_features()
            feature_nums = features_df[features_df["CASE_NUM"] == case_num]["FEATURE_NUM"].tolist()
            collection   = _get_chroma()

            verdicts = asyncio.run(assess_note_naive(
                note_text=note_text, case_num=case_num,
                feature_nums=feature_nums, chroma_collection=collection,
            ))

            run = EvaluationRun.objects.create(
                strategy     = "naive_rag",
                llm_model    = settings.CAIR_LLM_MODEL,
                notes_evaluated = 1,
                user         = request.user,
                student_name = request.user.get_full_name() or request.user.username,
                student_id   = profile.student_id if profile else "",
                case_num     = case_num,
                note_text    = note_text,
                notes        = f"Student submission — Case {case_num}",
            )
            ConceptVerdict.objects.bulk_create([
                ConceptVerdict(
                    run=run, pn_num="student_submission", case_num=case_num,
                    feature_num=v.feature_num, concept=v.concept,
                    predicted=bool(v.present), ground_truth=False, evidence=v.evidence or "",
                )
                for v in verdicts
            ])

            result_url = f"/evaluate/result/{run.pk}/"
            if prev_run_id:
                result_url += f"?prev={prev_run_id}"
            return redirect(result_url)
    else:
        form = NoteEvaluationForm(initial=initial)

    case_cards = [
        {"num": num, "label": label, "description": CASE_DESCRIPTIONS.get(num, "")}
        for num, label in [
            (201, "Case 201 — Irregular menses (44F)"),
            (202, "Case 202 — Epigastric discomfort (35M)"),
            (203, "Case 203 — Headache (20F)"),
            (204, "Case 204 — Sleep disturbance / grief (67F)"),
            (205, "Case 205 — Palpitations / heart racing (26F)"),
            (206, "Case 206 — Anxiety / nervousness (45F)"),
            (207, "Case 207 — Heavy periods / weight gain (35F)"),
            (208, "Case 208 — Right lower quadrant pain (20F)"),
            (209, "Case 209 — Chest pain / pleuritic (17M)"),
            (210, "Case 210 — Palpitations / heart pounding (17M)"),
        ]
    ]

    return render(request, "assessment/evaluate.html", {
        "form":         form,
        "case_cards":   case_cards,
        "prev_run_id":  prefill_run_id or "",
        "profile":      profile,
    })


@login_required
def note_result_view(request, run_id: int):
    run = get_object_or_404(EvaluationRun, pk=run_id)

    # Students can only see their own results
    try:
        if request.user.profile.is_student and run.user != request.user:
            return redirect("evaluate-note")
    except UserProfile.DoesNotExist:
        pass

    verdicts = ConceptVerdict.objects.filter(run=run).order_by("feature_num")
    present  = [v for v in verdicts if v.predicted]
    missing  = [v for v in verdicts if not v.predicted]
    score    = len(present)
    total    = len(verdicts)
    pct      = round(score / total * 100) if total else 0

    if pct >= 75:
        score_level, score_msg, score_color = "high", "Good coverage — most key concepts documented.", "#065f46"
    elif pct >= 50:
        score_level, score_msg, score_color = "mid",  "Moderate coverage — several concepts missing.", "#92400e"
    else:
        score_level, score_msg, score_color = "low",  "Low coverage — many key concepts not documented.", "#991b1b"

    case_num = run.case_num or (verdicts.first().case_num if verdicts.exists() else None)

    synonym_map = {}
    if missing:
        try:
            collection = _get_chroma()
            for v in missing:
                result = collection.get(ids=[f"{case_num}_{v.feature_num}"])
                if result["documents"]:
                    for line in result["documents"][0].split("\n"):
                        if line.startswith("Synonyms:"):
                            synonym_map[v.feature_num] = line.replace("Synonyms:", "").strip()
                            break
        except Exception:
            pass

    prev_data = None
    prev_run_id = request.GET.get("prev")
    if prev_run_id:
        try:
            prev_run   = EvaluationRun.objects.get(pk=prev_run_id)
            prev_v     = ConceptVerdict.objects.filter(run=prev_run)
            prev_score = prev_v.filter(predicted=True).count()
            prev_total = prev_v.count()
            prev_pct   = round(prev_score / prev_total * 100) if prev_total else 0
            prev_data  = {
                "run": prev_run, "score": prev_score, "total": prev_total, "pct": prev_pct,
                "delta_score": score - prev_score, "delta_pct": pct - prev_pct,
            }
        except EvaluationRun.DoesNotExist:
            pass

    return render(request, "assessment/note_result.html", {
        "run": run, "present": present, "missing": missing,
        "score": score, "total": total, "pct": pct,
        "score_level": score_level, "score_msg": score_msg, "score_color": score_color,
        "case_num": case_num, "topic": _CASE_TOPICS.get(case_num, ""),
        "synonym_map": synonym_map, "prev_data": prev_data,
    })


@login_required
def profile_view(request):
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = None

    submissions = EvaluationRun.objects.filter(user=request.user, note_text__gt="")
    total_subs  = submissions.count()
    pcts = []
    for run in submissions:
        v = ConceptVerdict.objects.filter(run=run)
        t = v.count()
        if t:
            pcts.append(round(v.filter(predicted=True).count() / t * 100))
    avg_pct = round(sum(pcts) / len(pcts)) if pcts else 0
    best_pct = max(pcts, default=0)

    return render(request, "assessment/profile.html", {
        "profile":    profile,
        "total_subs": total_subs,
        "avg_pct":    avg_pct,
        "best_pct":   best_pct,
    })


@login_required
def my_results_view(request):
    submissions = EvaluationRun.objects.filter(user=request.user, note_text__gt="").order_by("-created_at")
    runs = []
    for run in submissions:
        verdicts = ConceptVerdict.objects.filter(run=run)
        total    = verdicts.count()
        present  = verdicts.filter(predicted=True).count()
        pct      = round(present / total * 100) if total else 0
        runs.append({
            "run": run, "score": present, "total": total, "pct": pct,
            "topic": _CASE_TOPICS.get(run.case_num, ""),
            "level": "high" if pct >= 75 else ("mid" if pct >= 50 else "low"),
        })

    avg_pct  = round(sum(r["pct"] for r in runs) / len(runs)) if runs else 0
    best_pct = max((r["pct"] for r in runs), default=0)

    return render(request, "assessment/my_results.html", {
        "runs": runs, "avg_pct": avg_pct, "best_pct": best_pct,
    })
