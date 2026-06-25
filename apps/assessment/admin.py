from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import EvaluationRun, ConceptVerdict, UserProfile


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = "Profile"
    fields = ["role", "student_id"]


class CustomUserAdmin(UserAdmin):
    inlines = [UserProfileInline]


admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)


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
