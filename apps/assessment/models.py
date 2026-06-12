from django.db import models


class EvaluationRun(models.Model):
    """One batch evaluation run over the NBME dataset."""

    STRATEGY_CHOICES = [
        ("agentic_rag", "Agentic RAG"),
        ("naive_rag", "Naive RAG"),
        ("no_rag", "No RAG Baseline"),
    ]

    strategy = models.CharField(max_length=20, choices=STRATEGY_CHOICES, default="agentic_rag")
    llm_model = models.CharField(max_length=100)
    notes_evaluated = models.IntegerField(default=0)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1 = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.strategy} | F1={self.f1:.3f} | {self.created_at:%Y-%m-%d %H:%M}"


class ConceptVerdict(models.Model):
    """Per-concept verdict from the agent for one patient note."""

    run = models.ForeignKey(EvaluationRun, on_delete=models.CASCADE, related_name="verdicts")
    pn_num = models.CharField(max_length=20)
    case_num = models.IntegerField()
    feature_num = models.CharField(max_length=10)
    concept = models.CharField(max_length=200)
    predicted = models.BooleanField()
    ground_truth = models.BooleanField()
    evidence = models.TextField(blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["run", "case_num"]),
            models.Index(fields=["pn_num", "feature_num"]),
        ]

    def __str__(self):
        match = "✓" if self.predicted == self.ground_truth else "✗"
        return f"{match} {self.pn_num}/{self.feature_num}: pred={self.predicted} gt={self.ground_truth}"
