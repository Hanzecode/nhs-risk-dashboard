"""
Centralised data loading for all dashboard pages.

The clinical cohort is real (readmission_dataset.csv). Risk is scored with the
transparent, rule-based LACE index (see backend/processing/lace.py) rather than
a trained classifier — the synthetic readmission label carries no learnable
signal, so LACE gives honest, explainable stratification instead.

Uses st.cache_data so the CSV is read and scored once per session.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]                 # project root
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))                 # make processing.lace importable

from processing.lace import (                    # noqa: E402
    add_lace,
    CONDITION_NAMES,
    CHAPTER_NAMES,
    LACE_COMPONENTS,
    RISK_TIERS,
)

CLINICAL_CSV = ROOT / "readmission_dataset.csv"
METRICS_LOG  = BACKEND / "models" / "metrics_log.jsonl"

# ── Colour system (aligned with the NHS theme in .streamlit/config.toml) ──────
TIER_COLOURS = {"Low": "#007F3B", "Moderate": "#ED8B00", "High": "#D5281B"}

CHAPTER_COLOURS = {
    "Circulatory":          "#005EB8",
    "Respiratory":          "#41B6E6",
    "Endocrine & metabolic":"#007F3B",
    "Mental health":        "#330072",
    "Nervous system":       "#8A1538",
    "Digestive":            "#ED8B00",
    "Musculoskeletal":      "#78BE20",
    "Genitourinary":        "#AE2573",
}


@st.cache_data(show_spinner="Loading patient cohort…")
def load_patients() -> pd.DataFrame:
    """Load the real clinical cohort, engineer features, and score with LACE."""
    df = pl.read_csv(CLINICAL_CSV).to_pandas()

    # ICD-10 chapter + plain-English condition name
    df["chapter_letter"] = df["primary_diagnosis_code"].str[0]
    df["ICD10_Chapter"] = df["chapter_letter"].map(CHAPTER_NAMES).fillna("Other")
    df["condition"] = df["primary_diagnosis_code"].map(CONDITION_NAMES).fillna(
        df["primary_diagnosis_code"]
    )

    # Demographic / clinical bands
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 29, 49, 69, 200],
        labels=["Under 30", "30–49", "50–69", "70+"],
    )
    df["polypharmacy_flag"] = (df["num_medications"] >= 5).astype(int)

    # LACE risk score + tier (the "model")
    df = add_lace(df)
    df["risk_tier"] = pd.Categorical(df["risk_tier"], categories=RISK_TIERS, ordered=True)

    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> list[dict]:
    """
    Load the trained-classifier metrics history. Used only on the Methodology
    page to evidence *why* the supervised model was set aside (near-random AUC).
    """
    if not METRICS_LOG.exists():
        return []
    with open(METRICS_LOG) as f:
        return [json.loads(line) for line in f if line.strip()]


def tier_colour(tier: str) -> str:
    return TIER_COLOURS.get(tier, "#768692")
