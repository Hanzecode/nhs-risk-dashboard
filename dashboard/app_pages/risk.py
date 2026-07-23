"""Risk stratification — LACE scores, component breakdown, and patient drill-down."""

import altair as alt
import pandas as pd
import streamlit as st

from utils.data import load_patients, TIER_COLOURS
from processing.lace import RISK_TIERS, LACE_COMPONENTS

df = load_patients()

st.title("Risk stratification")
st.caption("LACE-based readmission risk, with a transparent breakdown of every score")

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.caption("Filters")
    tier_sel = st.pills("Risk tier", RISK_TIERS, selection_mode="multi",
                        default=RISK_TIERS)
    age_sel = st.multiselect(
        "Age band", ["Under 30", "30–49", "50–69", "70+"],
        default=["Under 30", "30–49", "50–69", "70+"],
    )
    chap_sel = st.multiselect(
        "ICD-10 chapter", sorted(df["ICD10_Chapter"].unique()),
        default=sorted(df["ICD10_Chapter"].unique()),
    )

tier_sel = tier_sel or RISK_TIERS
mask = (
    df["risk_tier"].isin(tier_sel)
    & df["age_band"].isin(age_sel)
    & df["ICD10_Chapter"].isin(chap_sel)
)
fdf = df[mask]

# ── KPI row ───────────────────────────────────────────────────────────────────
with st.container(horizontal=True):
    st.metric("Patients in view", f"{len(fdf):,}", border=True)
    st.metric("Mean LACE score",
              round(float(fdf["lace_score"].mean()), 1) if len(fdf) else "—",
              border=True)
    st.metric("High-risk patients",
              int((fdf["risk_tier"] == "High").sum()) if len(fdf) else 0,
              border=True)

col1, col2 = st.columns(2)

# ── LACE score distribution ───────────────────────────────────────────────────
with col1:
    with st.container(border=True):
        st.subheader("LACE score distribution")
        if len(fdf):
            hist = (
                alt.Chart(fdf)
                .mark_bar()
                .encode(
                    x=alt.X("lace_score:Q", bin=alt.Bin(maxbins=19),
                            title="LACE score"),
                    y=alt.Y("count():Q", title="Patients"),
                    color=alt.Color(
                        "risk_tier:N",
                        scale=alt.Scale(domain=RISK_TIERS,
                                        range=[TIER_COLOURS[t] for t in RISK_TIERS]),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                    tooltip=["count()"],
                )
                .properties(height=300)
            )
            st.altair_chart(hist, theme="streamlit")
        else:
            st.info("No patients match the current filters.", icon=":material/info:")

# ── Average LACE component contribution (explainability) ──────────────────────
with col2:
    with st.container(border=True):
        st.subheader("What drives the risk")
        st.caption("Average points contributed by each LACE component")
        if len(fdf):
            comp = pd.DataFrame({
                "Component": [LACE_COMPONENTS[c][0] for c in LACE_COMPONENTS],
                "Points": [round(fdf[c].mean(), 2) for c in LACE_COMPONENTS],
            })
            chart = (
                alt.Chart(comp)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Points:Q", title="Average points"),
                    y=alt.Y("Component:N", sort="-x", title=None),
                    color=alt.value("#005EB8"),
                    tooltip=["Component", "Points"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, theme="streamlit")

# ── Single-patient drill-down ─────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("Explain a patient's score")
    if len(fdf):
        pid = st.selectbox(
            "Select a patient", fdf["patient_id"].tolist(),
            label_visibility="collapsed",
        )
        prow = fdf[fdf["patient_id"] == pid].iloc[0]

        c1, c2 = st.columns([1, 2])
        with c1:
            tier = str(prow["risk_tier"])
            colour = {"Low": "green", "Moderate": "orange", "High": "red"}[tier]
            st.metric("LACE score", int(prow["lace_score"]))
            st.markdown(f":{colour}-badge[{tier} risk]")
            st.caption(
                f"{prow['condition']} · age {int(prow['age'])} · "
                f"{prow['admission_type']} admission"
            )
        with c2:
            breakdown = pd.DataFrame({
                "Component": [LACE_COMPONENTS[c][0] for c in LACE_COMPONENTS],
                "Points": [int(prow[c]) for c in LACE_COMPONENTS],
            })
            chart = (
                alt.Chart(breakdown)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Points:Q", title="Points contributed"),
                    y=alt.Y("Component:N", sort=[LACE_COMPONENTS[c][0] for c in LACE_COMPONENTS],
                            title=None),
                    color=alt.value("#005EB8"),
                    tooltip=["Component", "Points"],
                )
                .properties(height=200)
            )
            st.altair_chart(chart, theme="streamlit")

# ── Patient table + export ────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(f"Patient list — {len(fdf):,} records")
    cols = [
        "patient_id", "age", "age_band", "gender", "condition",
        "admission_type", "time_in_hospital", "num_medications",
        "lace_L", "lace_A", "lace_C", "lace_E", "lace_score", "risk_tier",
    ]
    table = fdf[cols].sort_values("lace_score", ascending=False).head(500)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "patient_id": "ID", "age": "Age", "age_band": "Age band",
            "gender": "Gender", "condition": "Condition",
            "admission_type": "Admission", "time_in_hospital": "Stay (days)",
            "num_medications": "Meds",
            "lace_L": "L", "lace_A": "A", "lace_C": "C", "lace_E": "E",
            "lace_score": st.column_config.ProgressColumn(
                "LACE", min_value=0, max_value=19, format="%d"
            ),
            "risk_tier": "Tier",
        },
    )
    st.download_button(
        "Download filtered cohort (CSV)",
        data=fdf[cols].to_csv(index=False).encode("utf-8"),
        file_name="lace_cohort.csv",
        mime="text/csv",
        icon=":material/download:",
    )
