"""Methodology & validation — how risk is scored, and why LACE over a classifier."""

import altair as alt
import pandas as pd
import streamlit as st

from utils.data import load_patients, load_metrics
from processing.lace import LACE_COMPONENTS, RISK_TIERS

df = load_patients()

st.title("Methodology & validation")
st.caption("How the LACE index works — and the data check behind the choice")

# ── What is LACE ──────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("The LACE index")
    st.markdown(
        "**LACE** is a validated tool (van Walraven et al., *CMAJ* 2010) used "
        "across the NHS to stratify patients by risk of unplanned readmission or "
        "death within 30 days of discharge. It is **rule-based and fully "
        "transparent** — every point is auditable and needs no training label."
    )
    weights = pd.DataFrame([
        {"Component": "L — Length of stay", "Points": "0–7",
         "From": "time_in_hospital"},
        {"Component": "A — Acuity of admission", "Points": "0–3",
         "From": "admission_type (emergency / urgent)"},
        {"Component": "C — Comorbidity", "Points": "0–5",
         "From": "has_comorbidity + Charlson-qualifying diagnosis"},
        {"Component": "E — Emergency admissions (prior)", "Points": "0–4",
         "From": "num_prior_admissions"},
    ])
    st.dataframe(weights, hide_index=True, width="stretch")
    st.markdown(
        "Total score **0–19**. Tiers: "
        ":green-badge[Low 0–4] :orange-badge[Moderate 5–9] :red-badge[High 10+]"
    )

# ── Why not a trained classifier ──────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Why not a machine-learning classifier?")
    metrics = load_metrics()
    auc = metrics[-1]["auc_roc"] if metrics else None

    left, right = st.columns([1, 1.4])
    with left:
        if auc is not None:
            st.metric("Trained model AUC-ROC", auc, border=True,
                      help="0.5 = random guessing, 1.0 = perfect")
        st.markdown(
            "A Gradient Boosting classifier was trained first. It scored "
            "**near-random** — because the synthetic readmission label carries "
            "no learnable signal."
        )

    with right:
        st.caption("Observed readmission rate is flat across every risk tier —")
        st.caption("evidence the label is noise, not a learnable outcome.")
        by_tier = (
            df.groupby("risk_tier", observed=True)["readmitted_within_30days"]
            .mean().mul(100).reindex(RISK_TIERS).reset_index()
        )
        by_tier.columns = ["Tier", "Readmission rate"]
        chart = (
            alt.Chart(by_tier)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Tier:N", sort=RISK_TIERS, axis=alt.Axis(labelAngle=0),
                        title=None),
                y=alt.Y("Readmission rate:Q", title="Readmission rate (%)",
                        scale=alt.Scale(domain=[0, 40])),
                color=alt.value("#768692"),
                tooltip=["Tier", alt.Tooltip("Readmission rate:Q", format=".1f")],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, theme="streamlit")

    st.markdown(
        "So instead of a black-box model that would fabricate confidence from "
        "noise, risk is scored with **LACE** — externally validated in the "
        "clinical literature, transparent to clinicians, and honest about what "
        "this data can and cannot support."
    )

# ── Pipeline ──────────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Pipeline")
    st.markdown(
        "- **Ingest** — validate the clinical CSV schema\n"
        "- **Process** — engineer features and map ICD-10 codes (Polars)\n"
        "- **Score** — compute LACE components and risk tiers\n"
        "- **Serve** — cached loader feeds all four dashboard pages"
    )
    st.caption("Component definitions")
    for key, (name, desc) in LACE_COMPONENTS.items():
        st.markdown(f"**{name}** — {desc}")
