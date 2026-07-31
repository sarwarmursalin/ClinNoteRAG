from django.urls import path
from . import views

urlpatterns = [
    # Public
    path("",        views.landing_view, name="landing"),
    path("about/",  views.about_view,   name="about"),

    # Auth
    path("login/",           views.login_view,           name="login"),
    path("forgot-password/", views.forgot_password_view, name="forgot-password"),
    path("verify-otp/",      views.verify_otp_view,      name="verify-otp"),
    path("reset-password/",  views.reset_password_view,  name="reset-password"),
    path("signup/",          views.register_choice_view, name="register-choice"),
    path("register/",        views.register_view,        name="register"),
    path("logout/",          views.logout_view,          name="logout"),
    path("home/",            views.home_view,            name="home"),

    # Student
    path("evaluate/",                     views.evaluate_note_view, name="evaluate-note"),
    path("evaluate/result/<int:run_id>/", views.note_result_view,   name="note-result"),
    path("my-results/",                   views.my_results_view,    name="my-results"),
    path("profile/",                      views.profile_view,       name="profile"),

    # Faculty
    path("dashboard/",              views.faculty_dashboard,       name="faculty-dashboard"),
    path("ablation/",               views.ablation_view,           name="ablation"),
    path("runs/<int:run_id>/",      views.run_detail,              name="run-detail"),
    path("case/<int:case_num>/",    views.case_concepts,           name="case-concepts"),
    path("export-submissions/",     views.export_submissions_csv,  name="export-submissions"),
]
