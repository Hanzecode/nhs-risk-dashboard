"""
LACE index — transparent, rule-based 30-day readmission risk score.

The LACE index (van Walraven et al., CMAJ 2010) is a validated tool used across
the NHS to stratify patients by risk of unplanned readmission or death within
30 days of discharge. It is deliberately *not* a machine-learning model: every
point is auditable and explainable to a clinician, and it needs no training
label — which is exactly why it is the right tool for this dataset, whose
synthetic `readmitted_within_30days` label carries no learnable signal.

    LACE = L + A + C + E
      L — Length of stay (days)
      A — Acuity of admission (acute / emergency)
      C — Comorbidity (Charlson-style burden)
      E — Emergency admissions in the prior period

Score range 0–19. Higher = higher risk. A score of 10+ is the conventional
threshold for "high risk".

This module operates on a pandas DataFrame and adds the four component columns,
the total `lace_score`, and a `risk_tier` (Low / Moderate / High).
"""

from __future__ import annotations

import pandas as pd

# ── ICD-10 reference (the 10 codes present in this dataset) ────────────────────
# Plain-English names for the dashboard, and the ICD-10 chapter each belongs to.
CONDITION_NAMES: dict[str, str] = {
    "E11": "Type 2 diabetes",
    "E78": "High cholesterol",
    "F32": "Depression",
    "G47": "Sleep disorders",
    "I10": "Hypertension",
    "I25": "Ischaemic heart disease",
    "J45": "Asthma",
    "K21": "Acid reflux (GORD)",
    "M54": "Back pain",
    "N39": "Urinary conditions",
}

CHAPTER_NAMES: dict[str, str] = {
    "E": "Endocrine & metabolic",
    "F": "Mental health",
    "G": "Nervous system",
    "I": "Circulatory",
    "J": "Respiratory",
    "K": "Digestive",
    "M": "Musculoskeletal",
    "N": "Genitourinary",
}

# ICD-10 codes that contribute to a Charlson-style comorbidity count.
_CHARLSON_CODES = {"E11": 1, "I25": 1}  # diabetes, chronic ischaemic heart disease

RISK_TIERS = ["Low", "Moderate", "High"]


# ── LACE component scoring ────────────────────────────────────────────────────

def _score_length_of_stay(days: int) -> int:
    """L — points for length of stay (van Walraven bands)."""
    if days < 1:
        return 0
    if days <= 3:
        return int(days)          # 1→1, 2→2, 3→3
    if days <= 6:
        return 4
    if days <= 13:
        return 5
    return 7                       # 14+


def _score_acuity(admission_type: str) -> int:
    """A — 3 points for an acute (emergency/urgent) admission, else 0."""
    return 3 if str(admission_type).strip().lower() in {"emergency", "urgent"} else 0


def _charlson_points(row: pd.Series) -> int:
    """
    C — a Charlson-style comorbidity count approximated from what the dataset
    records: the documented-comorbidity flag plus any Charlson-qualifying
    primary diagnosis. Capped at the LACE maximum of 5.
    """
    pts = int(row.get("has_comorbidity", 0) or 0)
    pts += _CHARLSON_CODES.get(str(row.get("primary_diagnosis_code", "")), 0)
    return min(pts, 5)


def _score_emergency_visits(prior: int) -> int:
    """E — points for emergency admissions in the prior period (capped at 4)."""
    return min(int(prior), 4)


def add_lace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `df` with LACE component columns, the total `lace_score`,
    and a categorical `risk_tier`.

    Expects columns: time_in_hospital, admission_type, has_comorbidity,
    primary_diagnosis_code, num_prior_admissions.
    """
    out = df.copy()

    out["lace_L"] = out["time_in_hospital"].apply(_score_length_of_stay)
    out["lace_A"] = out["admission_type"].apply(_score_acuity)
    out["lace_C"] = out.apply(_charlson_points, axis=1)
    out["lace_E"] = out["num_prior_admissions"].apply(_score_emergency_visits)

    out["lace_score"] = out[["lace_L", "lace_A", "lace_C", "lace_E"]].sum(axis=1)
    out["risk_tier"] = pd.cut(
        out["lace_score"],
        bins=[-1, 4, 9, 100],
        labels=RISK_TIERS,
    )
    return out


# Component metadata for the dashboard's explainability view.
LACE_COMPONENTS = {
    "lace_L": ("Length of stay", "Longer inpatient stays signal more complex, higher-risk episodes."),
    "lace_A": ("Acute admission", "Unplanned emergency/urgent admissions carry higher readmission risk."),
    "lace_C": ("Comorbidity", "Documented comorbidities and chronic conditions raise risk."),
    "lace_E": ("Prior admissions", "A history of emergency admissions is the strongest single predictor."),
}
