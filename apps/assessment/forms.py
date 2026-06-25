from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User

CASE_CHOICES = [
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

CASE_DESCRIPTIONS = {
    201: "A 44-year-old woman presents with irregular menstrual cycles for the past 3 years.",
    202: "A 35-year-old man presents with epigastric discomfort for the past 2 months.",
    203: "A 20-year-old woman presents with a global headache lasting 1–2 days.",
    204: "A 67-year-old woman presents with difficulty sleeping for 3 weeks after her son's death.",
    205: "A 26-year-old woman presents with episodes of heart racing for the past 5 years.",
    206: "A 45-year-old woman presents with feeling anxious and nervous.",
    207: "A 35-year-old woman presents with heavy, irregular periods and weight gain for 6 months.",
    208: "A 20-year-old woman presents with right lower quadrant abdominal pain.",
    209: "A 17-year-old male presents with chest pain that worsens with deep breathing.",
    210: "A 17-year-old male presents with intermittent heart pounding for a few months.",
}

INPUT_CLASS    = "form-input"
TEXTAREA_CLASS = "form-textarea"


class LoginForm(AuthenticationForm):
    """Thin wrapper so we can add CSS classes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update({"class": INPUT_CLASS, "placeholder": "Username", "autofocus": True})
        self.fields["password"].widget.attrs.update({"class": INPUT_CLASS, "placeholder": "Password"})


class StudentRegistrationForm(forms.Form):
    first_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={"class": INPUT_CLASS, "placeholder": "First name"}),
    )
    last_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={"class": INPUT_CLASS, "placeholder": "Last name"}),
    )
    student_id = forms.CharField(
        max_length=50,
        label="Student ID",
        widget=forms.TextInput(attrs={"class": INPUT_CLASS, "placeholder": "e.g. 202481996"}),
    )
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={"class": INPUT_CLASS, "placeholder": "Choose a username"}),
    )
    email = forms.EmailField(
        required=False,
        widget=forms.EmailInput(attrs={"class": INPUT_CLASS, "placeholder": "Email (optional)"}),
    )
    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={"class": INPUT_CLASS, "placeholder": "Password"}),
    )
    password2 = forms.CharField(
        label="Confirm password",
        widget=forms.PasswordInput(attrs={"class": INPUT_CLASS, "placeholder": "Confirm password"}),
    )

    def clean_username(self):
        username = self.cleaned_data["username"]
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("This username is already taken.")
        return username

    def clean(self):
        cleaned = super().clean()
        p1 = cleaned.get("password1")
        p2 = cleaned.get("password2")
        if p1 and p2 and p1 != p2:
            self.add_error("password2", "Passwords do not match.")
        return cleaned


class NoteEvaluationForm(forms.Form):
    case_num = forms.ChoiceField(
        choices=CASE_CHOICES,
        label="Clinical Case",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
    note_text = forms.CharField(
        label="Patient History Note",
        widget=forms.Textarea(attrs={
            "rows": 12,
            "placeholder": "Write or paste your patient history note here…",
            "class": TEXTAREA_CLASS,
        }),
    )
