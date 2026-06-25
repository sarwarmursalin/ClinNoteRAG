from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-secret-key-change-in-production")
DEBUG = True
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "apps.assessment",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STATIC_URL = "/static/"

LOGIN_URL          = "/login/"
LOGIN_REDIRECT_URL = "/home/"
LOGOUT_REDIRECT_URL = "/login/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ClinNoteRAG settings
NBME_DATA_DIR = Path(os.getenv(
    "NBME_DATA_DIR",
    str(BASE_DIR.parent / "NBME new data"),
))

CHROMA_DB_PATH = str(BASE_DIR / "chroma_db")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# CAIR LLM (LiteLLM proxy — used when available, free to use)
CAIR_LLM_URL     = os.getenv("CAIR_LLM_URL", "https://django.cair.mun.ca/litellm/v1")
CAIR_LLM_API_KEY = os.getenv("CAIR_LLM_API_KEY", "")
CAIR_LLM_MODEL   = os.getenv("CAIR_LLM_MODEL", "granite-4.1-30b")
CAIR_LLM_SSL_VERIFY = os.getenv("CAIR_LLM_SSL_VERIFY", "False").lower() != "true"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
