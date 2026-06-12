from django.shortcuts import render
from .models import EvaluationRun


def index(request):
    runs = EvaluationRun.objects.all()
    return render(request, "assessment/index.html", {"runs": runs})
