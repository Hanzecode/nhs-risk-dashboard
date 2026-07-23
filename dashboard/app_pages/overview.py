"""Overview — population summary, risk-tier mix, and a live admission feed."""

import altair as alt
import pandas as pd
import streamlit as st

from utils.data import load_patients, TIER_COLOURS
from processing.lace import RISK_TIERS

df = load_patients()

st.title("Population overview")
st.caption("Key metrics across the full patient cohort, scored with the LACE index")

# ── KPI row ───────────────────────────────────────────────────────────────────
total      = len(df)
high_risk  = int((df["risk_tier"] == "High").sum())
mean_lace  = round(float(df["lace_score"].mean()), 1)
readmit    = round(float(df["readmitted_within_30days"].mean()) * 100, 1)
poly_rate  = round(float(df["polypharmacy_flag"].mean()) * 100, 1)
avg_stay   = round(float(df["time_in_hospital"].mean()), 1)

with st.container(horizontal=True):
    st.metric("Total patients", f"{total:,}", border=True)
    st.metric("High-risk patients", f"{high_risk:,}", border=True,
              help="LACE score of 10 or above")
    st.metric("Mean LACE score", mean_lace, border=True, help="Range 0–19")
    st.metric("Observed readmission rate", f"{readmit}%", border=True,
              help="Recorded 30-day readmissions in the cohort")
    st.metric("Polypharmacy rate", f"{poly_rate}%", border=True,
              help="Patients on 5 or more medications")
    st.metric("Avg length of stay", f"{avg_stay} days", border=True)

# ── Risk tier mix + age distribution ──────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    with st.container(border=True):
        st.subheader("Risk tier distribution")
        tier_counts = (
            df["risk_tier"].value_counts().reindex(RISK_TIERS)
            .rename_axis("Tier").reset_index(name="Patients")
        )
        donut = (
            alt.Chart(tier_counts)
            .mark_arc(innerRadius=70, cornerRadius=3)
            .encode(
                theta=alt.Theta("Patients:Q", stack=True),
                color=alt.Color(
                    "Tier:N",
                    scale=alt.Scale(domain=RISK_TIERS,
                                    range=[TIER_COLOURS[t] for t in RISK_TIERS]),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["Tier", "Patients"],
            )
            .properties(height=300)
        )
        st.altair_chart(donut, theme="streamlit")

with col_b:
    with st.container(border=True):
        st.subheader("Admissions by age band")
        age_order = ["Under 30", "30–49", "50–69", "70+"]
        age_counts = (
            df["age_band"].value_counts().reindex(age_order)
            .rename_axis("Age band").reset_index(name="Patients")
        )
        bars = (
            alt.Chart(age_counts)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Age band:N", sort=age_order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Patients:Q"),
                color=alt.value("#005EB8"),
                tooltip=["Age band", "Patients"],
            )
            .properties(height=300)
        )
        st.altair_chart(bars, theme="streamlit")

# ── Live admission feed (real cohort, auto-refreshing fragment) ────────────────
st.subheader("Live admission feed")
st.caption("Samples one real scored patient from the cohort every 5 seconds")

if "feed_rows" not in st.session_state:
    st.session_state.feed_rows = []


@st.fragment(run_every="5s")
def live_feed():
    row = df.sample(1).iloc[0]
    st.session_state.feed_rows.insert(0, {
        "Patient ID":  str(row["patient_id"]),
        "Age":         int(row["age"]),
        "Condition":   str(row["condition"]),
        "LACE score":  int(row["lace_score"]),
        "Risk tier":   str(row["risk_tier"]),
    })
    st.session_state.feed_rows = st.session_state.feed_rows[:12]

    st.dataframe(
        pd.DataFrame(st.session_state.feed_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "LACE score": st.column_config.ProgressColumn(
                "LACE score", min_value=0, max_value=19, format="%d"
            ),
        },
    )


live_feed()
