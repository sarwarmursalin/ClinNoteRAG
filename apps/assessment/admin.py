from django.contrib import admin
from .models import EvaluationRun, ConceptVerdict


@admin.register(EvaluationRun)
class EvaluationRunAdmin(admin.ModelAdmin):
    list_display = ["strategy", "llm_model", "notes_evaluated", "precision", "recall", "f1", "created_at"]
    list_filter = ["strategy", "llm_model"]
    readonly_fields = ["created_at"]


@admin.register(ConceptVerdict)
class ConceptVerdictAdmin(admin.ModelAdmin):
    list_display = ["pn_num", "case_num", "feature_num", "concept", "predicted", "ground_truth"]
    list_filter = ["run", "case_num", "predicted", "ground_truth"]
    search_fields = ["pn_num", "concept", "evidence"]
