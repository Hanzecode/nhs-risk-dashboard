-- NHS Readmission Risk — database schema
-- Optional structured storage for the scored cohort.
-- The dashboard reads the CSV and scores in memory; this schema is the target
-- for persisting results in a deployment. Run: psql -U postgres -d nhs_risk -f schema.sql

CREATE TABLE IF NOT EXISTS patients (
    patient_id               TEXT PRIMARY KEY,

    -- Demographics & admission
    age                      INTEGER,
    age_band                 TEXT,          -- Under 30 / 30-49 / 50-69 / 70+
    gender                   TEXT,
    admission_type           TEXT,          -- Emergency / Urgent / Elective
    hospital_id              INTEGER,

    -- Diagnosis
    primary_diagnosis_code   TEXT,          -- ICD-10 code (e.g. E11)
    icd10_chapter            TEXT,          -- e.g. Endocrine & metabolic
    condition                TEXT,          -- plain-English name (e.g. Type 2 diabetes)

    -- Clinical episode
    time_in_hospital         INTEGER,       -- length of stay, days
    num_prior_admissions     INTEGER,       -- emergency admissions in prior period
    num_medications          INTEGER,
    num_lab_procedures       INTEGER,
    has_comorbidity          BOOLEAN,
    polypharmacy_flag        SMALLINT,      -- 1 if 5+ medications
    discharge_disposition    TEXT,
    insurance_type           TEXT,

    -- LACE risk score (rule-based) — components, total, and tier
    lace_l                   SMALLINT,      -- length of stay points   (0-7)
    lace_a                   SMALLINT,      -- acuity of admission      (0-3)
    lace_c                   SMALLINT,      -- comorbidity              (0-5)
    lace_e                   SMALLINT,      -- prior emergency visits   (0-4)
    lace_score               SMALLINT,      -- total                    (0-19)
    risk_tier                TEXT,          -- Low | Moderate | High

    -- Recorded outcome
    readmitted_within_30days BOOLEAN,
    days_to_readmission      INTEGER,
    scored_at                TIMESTAMP DEFAULT NOW()
);

-- Rolling feed of new admissions shown on the Overview page
CREATE TABLE IF NOT EXISTS admissions_live (
    id           SERIAL PRIMARY KEY,
    patient_id   TEXT,
    age          INTEGER,
    condition    TEXT,
    lace_score   SMALLINT,
    risk_tier    TEXT,
    admitted_at  TIMESTAMP DEFAULT NOW()
);

-- Indexes for dashboard query performance
CREATE INDEX IF NOT EXISTS idx_patients_risk_tier    ON patients(risk_tier);
CREATE INDEX IF NOT EXISTS idx_patients_icd_chapter  ON patients(icd10_chapter);
CREATE INDEX IF NOT EXISTS idx_patients_hospital     ON patients(hospital_id);
CREATE INDEX IF NOT EXISTS idx_admissions_admitted   ON admissions_live(admitted_at DESC);
