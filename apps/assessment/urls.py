from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("case/<int:case_num>/", views.case_concepts, name="case-concepts"),
    path("runs/<int:run_id>/", views.run_detail, name="run-detail"),
]
