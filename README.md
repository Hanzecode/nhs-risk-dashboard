# NHS Readmission Risk Dashboard

An interactive Streamlit dashboard for exploring **10,000 synthetic hospital episodes**. It uses a transparent, rule-based implementation of the LACE index to stratify records into Low, Moderate, and High risk tiers. It does not train or serve a machine-learning model, and the synthetic data must not be used to draw clinical conclusions.

## What the project demonstrates

- Loading and transforming a clinical-style CSV with Polars and pandas.
- Applying an explainable LACE-style scoring rule and showing each patient's component scores.
- Exploring cohort patterns by diagnosis, age, hospital, and risk tier in an interactive dashboard.
- Documenting the limitations of synthetic data and a rule-based score.

## Architecture

```text
readmission_dataset.csv
        │
        ▼
Polars load ──► pandas feature preparation
                    ├── Age bands, ICD-10 condition/chapter labels
                    ├── Polypharmacy flag
                    └── LACE component points: L + A + C + E
                              │
                              ▼
                    Score (0–19) + risk tier
                              │
                              ▼
              Cached Streamlit data loader
                 ├── Population overview
                 ├── Conditions & hospitals
                 ├── Risk stratification
                 └── Methodology & validation
```

The dashboard reads `readmission_dataset.csv` from the repository root. The current app does not connect to PostgreSQL; `backend/db/schema.sql` is a reference schema only. The live admission feed is a demonstration that samples an existing scored row from the synthetic cohort every five seconds. It does not ingest real-time admissions.

## Risk scoring

For each episode, the app calculates four components and sums them:

| Component | What it represents | Input column(s) |
|---|---|---|
| **L — Length of stay** | Points based on the number of days in hospital (0–7) | `time_in_hospital` |
| **A — Acuity** | 3 points for Emergency or Urgent admission; otherwise 0 | `admission_type` |
| **C — Comorbidity** | Approximation using the comorbidity flag and selected qualifying primary diagnoses; capped at 5 | `has_comorbidity`, `primary_diagnosis_code` |
| **E — Prior admissions** | Prior-admission count, capped at 4 | `num_prior_admissions` |

`lace_score = L + A + C + E`, with a score range of 0–19. The dashboard groups scores as follows:

| Tier | Score |
|---|---:|
| Low | 0–4 |
| Moderate | 5–9 |
| High | 10+ |

These are **rule-based scores and tiers, not predicted probabilities**. The C component is an approximation because the dataset does not contain a complete comorbidity history. The E input is the dataset's `num_prior_admissions` field; the file does not establish that every prior admission was an emergency admission. For real clinical use, the inputs and scoring implementation would need review against the intended LACE specification and validated patient data.

## Dashboard pages

| Page | What it shows | What it helps explore |
|---|---|---|
| **Population overview** | Cohort KPIs, risk-tier distribution, age bands, and a sample feed | Overall profile of the synthetic cohort |
| **Conditions & hospitals** | Episode counts and mean LACE by condition/chapter, plus a comparison of seven hospital IDs | How the synthetic cohort is distributed across diagnoses and sites |
| **Risk stratification** | Filters, score distribution, average component contributions, patient-level score explanation, and CSV export | How the rules assign points to each record and how selected groups compare |
| **Methodology & validation** | LACE rule summary, historical classifier metric, observed label rates by LACE tier, and pipeline description | Why the project uses rules and what the synthetic data can support |

The methodology page includes a historical Gradient Boosting result from an earlier experiment. That model is not part of the current scoring path. Its near-random AUC helped identify that the synthetic readmission label has little useful relationship with the available features; it is shown as validation history, not as current model performance.

## Data

The repository's `readmission_dataset.csv` contains synthetic records. Fields used by the current dashboard include:

- Identifiers and grouping: `patient_id`, `hospital_id`
- Demographics: `age`, `gender`
- Admission details: `admission_type`, `time_in_hospital`, `discharge_disposition`
- Clinical details: `primary_diagnosis_code`, `has_comorbidity`, `num_prior_admissions`, `num_lab_procedures`, `num_medications`
- Recorded outcome fields for demonstration: `readmitted_within_30days`, `days_to_readmission`

The dashboard derives age bands, ICD-10 chapter and condition labels, `polypharmacy_flag`, LACE component points, total LACE score, and risk tier. All patient records are synthetic; the recorded outcomes and resulting summaries are not evidence of real NHS patterns.

## Run locally

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/Hanzecode/nhs-risk-dashboard.git
cd nhs-risk-dashboard
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository's `main` branch.
3. Set the main file path to `dashboard/app.py`.
4. Deploy. Cloud installs the packages listed in the root `requirements.txt`.

The CSV must be present in the repository for the deployed app to load. This project's CSV is synthetic. Do not commit identifiable or confidential patient data to a public repository.

## Project structure

```text
.
├── dashboard/
│   ├── app.py                    # Streamlit entry point and navigation
│   ├── app_pages/                # Four dashboard pages
│   └── utils/data.py             # Cached loading, feature preparation, LACE scoring
├── backend/
│   ├── processing/lace.py        # LACE rules and diagnosis labels
│   ├── models/metrics_log.jsonl  # Historical classifier metrics for methodology page
│   └── db/schema.sql             # Reference schema; not connected to the app
├── readmission_dataset.csv       # Synthetic cohort used by the dashboard
├── requirements.txt              # Runtime dependencies
└── NHS_Readmission_Dashboard_Guide.docx
```

The `_archive/` directory contains older pipeline files and intermediate data artifacts. They are not imported by the current dashboard.

## Technology

- **Python**, **Polars**, and **pandas** for data loading and preparation
- **Streamlit** for the interactive application, navigation, caching, and filters
- **Altair** for interactive charts
- **LACE-style rules** for transparent score calculation

## Important limitation

This is a portfolio and workflow demonstration using synthetic data, not a clinical decision-support system. The score tiers are descriptive outputs of the implemented rules; they have not been calibrated or validated against real patient outcomes in this dataset. Do not use the dashboard to make decisions about patient care.
