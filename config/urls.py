from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

handler404 = "apps.assessment.views.custom_404"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("apps.assessment.urls")),
]
