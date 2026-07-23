"""Conditions — disease and hospital breakdown across the cohort."""

import altair as alt
import streamlit as st

from utils.data import load_patients, CHAPTER_COLOURS

df = load_patients()

st.title("Conditions & hospitals")
st.caption("Admission patterns, length of stay, and risk by diagnosis and site")

# ── Aggregate by condition ────────────────────────────────────────────────────
by_cond = (
    df.groupby("condition")
    .agg(
        Patients=("patient_id", "count"),
        Avg_stay=("time_in_hospital", "mean"),
        Avg_meds=("num_medications", "mean"),
        Avg_lace=("lace_score", "mean"),
        High_risk_pct=("risk_tier", lambda s: (s == "High").mean() * 100),
    )
    .reset_index()
    .sort_values("Patients", ascending=False)
)
for c in ["Avg_stay", "Avg_meds", "Avg_lace", "High_risk_pct"]:
    by_cond[c] = by_cond[c].round(1)

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Admissions by condition")
        chart = (
            alt.Chart(by_cond)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Patients:Q"),
                y=alt.Y("condition:N", sort="-x", title=None),
                color=alt.value("#005EB8"),
                tooltip=["condition", "Patients"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, theme="streamlit")

with col2:
    with st.container(border=True):
        st.subheader("Mean LACE score by condition")
        chart = (
            alt.Chart(by_cond)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Avg_lace:Q", title="Mean LACE score"),
                y=alt.Y("condition:N", sort="-x", title=None),
                color=alt.Color("Avg_lace:Q", scale=alt.Scale(scheme="blues"),
                                legend=None),
                tooltip=["condition", "Avg_lace"],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, theme="streamlit")

# ── By ICD-10 chapter ─────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Admissions by ICD-10 chapter")
    by_chap = (
        df.groupby("ICD10_Chapter")
        .agg(Patients=("patient_id", "count"),
             Avg_stay=("time_in_hospital", "mean"))
        .reset_index()
        .sort_values("Patients", ascending=False)
    )
    by_chap["Avg_stay"] = by_chap["Avg_stay"].round(1)
    chapters = by_chap["ICD10_Chapter"].tolist()
    chart = (
        alt.Chart(by_chap)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("ICD10_Chapter:N", sort="-y", title=None,
                    axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("Patients:Q"),
            color=alt.Color(
                "ICD10_Chapter:N",
                scale=alt.Scale(domain=list(CHAPTER_COLOURS.keys()),
                                range=list(CHAPTER_COLOURS.values())),
                legend=None,
            ),
            tooltip=["ICD10_Chapter", "Patients", "Avg_stay"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, theme="streamlit")

# ── Hospital comparison ───────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Hospital comparison")
    st.caption("Across the 7 sites in the cohort")
    by_hosp = (
        df.groupby("hospital_id")
        .agg(
            Patients=("patient_id", "count"),
            Avg_stay=("time_in_hospital", "mean"),
            Avg_lace=("lace_score", "mean"),
            High_risk_pct=("risk_tier", lambda s: (s == "High").mean() * 100),
        )
        .reset_index()
    )
    by_hosp["hospital"] = "Hospital " + by_hosp["hospital_id"].astype(str)
    for c in ["Avg_stay", "Avg_lace", "High_risk_pct"]:
        by_hosp[c] = by_hosp[c].round(1)

    st.dataframe(
        by_hosp[["hospital", "Patients", "Avg_stay", "Avg_lace", "High_risk_pct"]],
        hide_index=True,
        width="stretch",
        column_config={
            "hospital": "Hospital",
            "Avg_stay": st.column_config.NumberColumn("Avg stay (days)", format="%.1f"),
            "Avg_lace": st.column_config.NumberColumn("Mean LACE", format="%.1f"),
            "High_risk_pct": st.column_config.ProgressColumn(
                "High-risk %", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )

# ── Full condition table ──────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Condition breakdown")
    st.dataframe(
        by_cond,
        hide_index=True,
        width="stretch",
        column_config={
            "condition": "Condition",
            "Patients": st.column_config.NumberColumn("Patients", format="%d"),
            "Avg_stay": st.column_config.NumberColumn("Avg stay (days)", format="%.1f"),
            "Avg_meds": st.column_config.NumberColumn("Avg medications", format="%.1f"),
            "Avg_lace": st.column_config.NumberColumn("Mean LACE", format="%.1f"),
            "High_risk_pct": st.column_config.ProgressColumn(
                "High-risk %", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )
