from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """Extended profile for both students and faculty."""

    ROLE_CHOICES = [
        ("student", "Student"),
        ("faculty", "Faculty / Grader"),
    ]

    user       = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    role       = models.CharField(max_length=10, choices=ROLE_CHOICES, default="student")
    student_id = models.CharField(max_length=50, blank=True, help_text="Student ID number (students only)")

    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username} ({self.role})"

    @property
    def is_faculty(self):
        return self.role == "faculty"

    @property
    def is_student(self):
        return self.role == "student"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Auto-create a UserProfile whenever a new User is saved (e.g. via Django admin)."""
    if created:
        UserProfile.objects.get_or_create(user=instance)


class EvaluationRun(models.Model):
    """One evaluation run — either a batch research run or a single student submission."""

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
    # Submission fields (student submissions only)
    user       = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="submissions")
    student_name = models.CharField(max_length=200, blank=True)
    student_id = models.CharField(max_length=50, blank=True)
    case_num = models.IntegerField(null=True, blank=True)
    note_text = models.TextField(blank=True)

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
