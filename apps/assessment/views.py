from __future__ import annotations

import asyncio
import csv
import json
import re
from collections import defaultdict
from functools import wraps
from pathlib import Path

import chromadb
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from services.naive_rag import assess_note_naive
from services.no_rag import assess_note_no_rag
from services.agent import assess_note as assess_note_agentic

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


_AGE_CONCEPT_RE   = re.compile(r'\b\d+\s*year', re.IGNORECASE)
_AGE_IN_NOTE_RE   = re.compile(r'\b(\d+)[\s\-]*(year|yr|y\.?o\.?|years?\s*old)', re.IGNORECASE)
_GENDER_IN_NOTE_RE = re.compile(
    r'\b(male|female|man|woman|boy|girl|he\b|she\b|his\b|her\b|mr\.?|ms\.?|mrs\.?)\b',
    re.IGNORECASE,
)

_REASON_SKIP_WORDS = {
    "a", "an", "the", "of", "to", "in", "or", "and", "with",
    "no", "not", "is", "was", "are", "be", "by", "on", "at",
    "for", "its", "this", "that", "from",
}


def _friendly_concept(concept: str) -> str:
    """Return a student-facing display name for age/gender concepts."""
    if _AGE_CONCEPT_RE.search(concept):
        return "Patient Age (any age accepted)"
    if concept.strip().lower() in ("male", "female"):
        return "Patient Gender (any gender accepted)"
    return concept


def _enrich_absent_reasons(verdicts, note_text, case_num, collection):
    """Python-side fallback: detect when an absent concept's keywords DO appear
    in the note (wrong value / partial mention) and override the LLM reason with
    the actual sentence so the student understands what went wrong."""
    note_lower = note_text.lower()
    # Split note into sentences for quote extraction
    sentences = [s.strip() for s in re.split(r'[.!?\n]', note_text) if s.strip()]

    for v in verdicts:
        if v.present:
            continue
        try:
            result = collection.get(ids=[f"{case_num}_{v.feature_num}"])
            if not result["documents"]:
                continue
            doc = result["documents"][0]
        except Exception:
            continue

        # Extract meaningful keywords from the concept doc (3+ char, non-stopword)
        raw_words = re.findall(r'\b[a-z]{3,}\b', doc.lower())
        keywords = [w for w in raw_words if w not in _REASON_SKIP_WORDS]

        # Find first sentence in the note that contains any keyword
        matched_sentence = None
        for kw in keywords:
            for sent in sentences:
                if kw in sent.lower():
                    matched_sentence = sent
                    break
            if matched_sentence:
                break

        if matched_sentence:
            v.reason = (
                f'Mentioned but incorrect: your note states "{matched_sentence}" '
                f'— this concept specifically requires "{v.concept}".'
            )
        else:
            v.reason = "Not mentioned in the note."

    return verdicts

def _apply_flex_age_gender(verdicts, note_text):
    """Give credit for any documented age and any documented gender.

    The NBME rubric ties age/gender to the case patient (e.g. '17 year', 'Male').
    For web evaluation we award these if the student documented *any* age or *any*
    gender — the clinical skill being tested is whether they asked, not whether
    they matched a specific demographic.
    """
    has_age    = bool(_AGE_IN_NOTE_RE.search(note_text))
    has_gender = bool(_GENDER_IN_NOTE_RE.search(note_text))

    for v in verdicts:
        name = v.concept.strip().lower()
        if _AGE_CONCEPT_RE.search(name) and has_age and not v.present:
            v.present = True
            m = _AGE_IN_NOTE_RE.search(note_text)
            v.evidence = m.group(0) if m else v.evidence
        elif name in ("male", "female") and has_gender and not v.present:
            v.present = True
            m = _GENDER_IN_NOTE_RE.search(note_text)
            v.evidence = m.group(0) if m else v.evidence
    return verdicts


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
    auth_students = submissions.filter(user__isnull=False).exclude(user__username="anonymous_student").values("user").distinct().count()
    legacy_students = submissions.filter(user__isnull=True).values("student_name").distinct().count()
    anon_students = submissions.filter(user__username="anonymous_student").values("student_name").distinct().count()
    unique_students = auth_students + legacy_students + anon_students
    all_pcts = [r["pct"] for r in submission_rows]
    avg_score = round(sum(all_pcts) / len(all_pcts)) if all_pcts else 0

    # Per-case chart data
    case_buckets = defaultdict(list)
    for r in submission_rows:
        if r["run"].case_num:
            case_buckets[r["run"].case_num].append(r["pct"])
    sorted_cases = sorted(case_buckets.keys())
    case_labels  = [f"Case {c}" for c in sorted_cases]
    case_avgs    = [round(sum(case_buckets[c]) / len(case_buckets[c])) for c in sorted_cases]
    case_counts  = [len(case_buckets[c]) for c in sorted_cases]

    # Most-missed concepts (from ConceptVerdict of student submissions)
    student_run_ids = [r["run"].pk for r in submission_rows]
    missed_counts = defaultdict(lambda: {"count": 0, "total": 0, "case_num": None})
    if student_run_ids:
        verdicts_qs = ConceptVerdict.objects.filter(run_id__in=student_run_ids)
        for v in verdicts_qs:
            key = (v.case_num, v.concept)
            missed_counts[key]["total"] += 1
            missed_counts[key]["case_num"] = v.case_num
            if not v.predicted:
                missed_counts[key]["count"] += 1
    missed_list = [
        {
            "concept":   k[1],
            "case_num":  v["case_num"],
            "miss_pct":  round(v["count"] / v["total"] * 100) if v["total"] else 0,
        }
        for k, v in missed_counts.items() if v["total"] >= 2
    ]
    missed_list.sort(key=lambda x: -x["miss_pct"])
    missed_list = missed_list[:8]

    dash_chart_json = json.dumps({
        "case_labels": case_labels,
        "case_avgs":   case_avgs,
        "case_counts": case_counts,
        "missed":      missed_list,
    }) if submission_rows else None

    return render(request, "assessment/faculty_dashboard.html", {
        "submission_rows":   submission_rows,
        "research_runs":     research_runs,
        "unique_students":   unique_students,
        "total_submissions": len(submission_rows),
        "avg_score":         avg_score,
        "dash_chart_json":   dash_chart_json,
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

    if request.method == "POST":
        form = NoteEvaluationForm(request.POST)
        if form.is_valid():
            case_num  = int(form.cleaned_data["case_num"])
            note_text = form.cleaned_data["note_text"].strip()
            prev_run_id = request.POST.get("prev_run_id", "").strip()

            strategy     = form.cleaned_data["strategy"]
            features_df  = _load_features()
            feature_nums = features_df[features_df["CASE_NUM"] == case_num]["FEATURE_NUM"].tolist()
            collection   = _get_chroma()

            try:
                if strategy == "no_rag":
                    results = collection.get(
                        where={"case_num": int(case_num)},
                        include=["documents", "metadatas"],
                    )
                    concept_names = {
                        str(m["feature_num"]): d.split("\n")[0].replace("Concept:", "").strip()
                        for d, m in zip(results["documents"], results["metadatas"])
                    }
                    verdicts = asyncio.run(assess_note_no_rag(
                        note_text=note_text, case_num=case_num,
                        feature_nums=feature_nums, concept_names=concept_names,
                    ))
                elif strategy == "agentic_rag":
                    verdicts = asyncio.run(assess_note_agentic(
                        note_text=note_text, case_num=case_num,
                        feature_nums=feature_nums, chroma_collection=collection,
                    ))
                else:
                    verdicts = asyncio.run(assess_note_naive(
                        note_text=note_text, case_num=case_num,
                        feature_nums=feature_nums, chroma_collection=collection,
                    ))
                verdicts = _apply_flex_age_gender(verdicts, note_text)
                verdicts = _enrich_absent_reasons(verdicts, note_text, case_num, collection)
            except Exception as e:
                err = str(e)
                vpn_keywords = ("503", "502", "Service Unavailable", "Connection error",
                                "ConnectError", "connection attempt", "unavailable")
                if any(kw.lower() in err.lower() for kw in vpn_keywords):
                    messages.error(request, "The AI server is unreachable. Please make sure you are connected to MUN VPN and try again.")
                else:
                    messages.error(request, f"Evaluation failed: {err[:200]}")
                return render(request, "assessment/evaluate.html", {
                    "form": form, "case_cards": case_cards, "prev_run_id": prefill_run_id or "",
                    "profile": profile,
                })

            run = EvaluationRun.objects.create(
                strategy     = strategy,
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
                    predicted=bool(v.present), ground_truth=False,
                    evidence=v.evidence or "",
                    reason=v.reason or "",
                )
                for v in verdicts
            ])

            result_url = f"/evaluate/result/{run.pk}/"
            if prev_run_id:
                result_url += f"?prev={prev_run_id}"
            return redirect(result_url)
    else:
        form = NoteEvaluationForm(initial=initial)

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
    for v in verdicts:
        v.display_concept = _friendly_concept(v.concept)
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

    # Chart: progress over time (chronological order)
    chron = list(reversed(runs))
    chart_labels   = [r["run"].created_at.strftime("%-d %b") for r in chron]
    chart_values   = [r["pct"] for r in chron]
    chart_tooltips = [f"Case {r['run'].case_num} · {r['run'].created_at.strftime('%b %-d, %Y')}" for r in chron]

    # Per-case best scores
    case_best = defaultdict(lambda: {"best_pct": 0, "count": 0})
    for r in runs:
        cn = r["run"].case_num
        if cn:
            case_best[cn]["count"] += 1
            case_best[cn]["best_pct"] = max(case_best[cn]["best_pct"], r["pct"])
    by_case = [
        {"case_num": cn, "best_pct": v["best_pct"], "count": v["count"]}
        for cn, v in sorted(case_best.items())
    ]

    chart_json = json.dumps({
        "labels":   chart_labels,
        "values":   chart_values,
        "tooltips": chart_tooltips,
        "by_case":  by_case,
    })

    return render(request, "assessment/my_results.html", {
        "runs":             runs,
        "avg_pct":          avg_pct,
        "best_pct":         best_pct,
        "cases_attempted":  len(case_best),
        "chart_json":       chart_json,
    })


@faculty_required
def export_submissions_csv(request):
    """Download all student submissions as CSV."""
    submissions = (
        EvaluationRun.objects
        .filter(note_text__gt="")
        .exclude(student_name="")
        .order_by("-created_at")
    )
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="clinnoterag_submissions.csv"'
    writer = csv.writer(response)
    writer.writerow(["Date", "Student Name", "Student ID", "Case", "Topic", "Score", "Total", "Coverage %"])
    for run in submissions:
        verdicts = ConceptVerdict.objects.filter(run=run)
        total   = verdicts.count()
        present = verdicts.filter(predicted=True).count()
        pct     = round(present / total * 100) if total else 0
        writer.writerow([
            run.created_at.strftime("%Y-%m-%d %H:%M"),
            run.student_name or (run.user.get_full_name() if run.user else ""),
            run.student_id or "",
            run.case_num or "",
            _CASE_TOPICS.get(run.case_num, ""),
            present, total, pct,
        ])
    return response
