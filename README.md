# NHS Readmission Risk Dashboard

A data science project predicting 30-day hospital readmission risk using
simulated NHS data. Built with Polars, scikit-learn, and Streamlit.

---

## Project structure

```
nhs-risk-dashboard/
├── backend/
│   ├── processing/features.py         # Polars + Pandas feature engineering
│   ├── ml/train.py                    # scikit-learn model + metrics
│   ├── db/schema.sql                  # PostgreSQL schema (optional)
│   └── requirements.txt
├── dashboard/
│   ├── app.py                         # Streamlit entry point
│   ├── pages/
│   │   ├── overview.py                # Page 1 — KPIs + live feed
│   │   ├── diseases.py                # Page 2 — ICD-10 analysis
│   │   ├── risk_patients.py           # Page 3 — patient table + filters
│   │   └── model_perf.py              # Page 4 — AUC, F1, feature importance
│   └── utils/
│       └── data.py                    # cached data loader (shared by all pages)
├── data/                              # place your CSV files here
├── docker-compose.yml
└── .env.example
```

---

## Quickstart

### 1. Place your data files

```
nhs-risk-dashboard/
├── readmission_dataset.csv
└── ID_2019.csv
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Train the model (first time)

```bash
cd backend
python -m ml.train
# Creates: backend/models/readmission_model.pkl
#          backend/models/metrics_log.jsonl
```

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
# Opens automatically at http://localhost:8501
```

That's it — no Node.js, no separate API server, no Docker required to get started.

---

## API endpoints

| Method | Path                   | Dashboard    | Description                        |
|--------|------------------------|--------------|------------------------------------|
| GET    | /api/overview          | Overview     | KPI summary stats                  |
| GET    | /api/diseases          | Diseases     | ICD-10 chapter breakdown           |
| GET    | /api/patients          | Risk patients| Filterable patient table           |
| GET    | /api/risk-distribution | Risk patients| Risk score histogram               |
| GET    | /api/deprivation-risk  | Risk patients| IMD decile vs mean risk            |
| GET    | /api/model-metrics     | Model perf   | AUC, F1, feature importance        |
| WS     | /ws/admissions         | Overview     | Live admission events (WebSocket)  |

Query parameters for `/api/patients`:
- `risk_band` — `Low` | `Medium` | `High`
- `age_band`  — `Under 30` | `30-49` | `50-69` | `70+`
- `limit`     — max rows returned (default 200)

---

## Pipeline

`run_pipeline.py` runs 3 stages sequentially:

```
ingest → process → train
```

| Stage   | Script                  | What it does                              |
|---------|-------------------------|--------------------------------------------|
| ingest  | run_pipeline.py         | Validates CSV columns                     |
| process | processing/features.py  | Cleans data, engineers 8 model features   |
| train   | ml/train.py             | Trains GBM, scores patients, logs metrics |

To run manually:
```bash
python backend/run_pipeline.py
```

---

## Features engineered

| Feature           | Description                                      |
|-------------------|--------------------------------------------------|
| age_band          | Under 30 / 30-49 / 50-69 / 70+                  |
| polypharmacy_flag | 1 if patient on 5+ medications                   |
| charlson_proxy    | Simplified comorbidity severity score            |
| complexity_score  | Composite of lab procedures + meds + comorbidity |
| ICD10_Chapter     | Mapped from primary diagnosis code first letter  |

---

## Dashboards

| Page              | Route       | Key charts                                        |
|-------------------|-------------|---------------------------------------------------|
| Overview          | /           | Live counter, risk bar (WebSocket), recent events |
| Disease analysis  | /diseases   | ICD-10 distribution, avg stay, readmission rate   |
| Patient risk      | /patients   | Risk histogram, deprivation scatter, patient table|
| Model performance | /model      | AUC/F1 over time, feature importance, run history |

---

## Tech stack

| Layer         | Technology                        |
|---------------|-----------------------------------|
| Processing    | Polars + Pandas + NumPy           |
| ML            | scikit-learn (GradientBoosting)   |
| Dashboard     | Streamlit + Plotly                |
| Storage       | PostgreSQL (optional)             |
| Container     | Docker + Docker Compose           |
