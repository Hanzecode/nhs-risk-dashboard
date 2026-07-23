"""
test_pipeline.py — end-to-end tests for the NHS readmission pipeline.

Run with:
    cd nhs-risk-dashboard
    pytest backend/tests/test_pipeline.py -v

Tests use synthetic data generated in-memory — no real CSV files needed.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

# Make backend importable
sys.path.insert(0, str(Path(__file__).parents[1]))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_clinical_df() -> pl.DataFrame:
    """Minimal synthetic clinical dataset (100 rows)."""
    rng = np.random.default_rng(0)
    n = 100
    codes = ["I21", "J18", "A09", "Z99", "I50", "J44", "B20", "I10"]
    return pl.DataFrame({
        "patient_id":               [f"P{i:04d}" for i in range(n)],
        "readmitted_within_30days": rng.integers(0, 2, n).tolist(),
        "time_in_hospital":         rng.integers(1, 15, n).tolist(),
        "num_medications":          rng.integers(0, 20, n).tolist(),
        "primary_diagnosis_code":   rng.choice(codes, n).tolist(),
        "has_comorbidity":          rng.integers(0, 2, n).tolist(),
        "age":                      rng.integers(18, 90, n).tolist(),
        "gender":                   rng.choice(["Male", "Female"], n).tolist(),
        "admission_type":           rng.choice(["Emergency", "Elective"], n).tolist(),
        "num_lab_procedures":       rng.integers(0, 80, n).tolist(),
    })


@pytest.fixture
def sample_imd_df() -> pl.DataFrame:
    """Minimal synthetic IMD dataset."""
    return pl.DataFrame({
        "LSOA_Code": [f"E0100{i:04d}" for i in range(50)],
        "IMD_Decile": list(range(1, 11)) * 5,
    })


# ── charlson.py tests ─────────────────────────────────────────────────────────

class TestCharlson:
    def test_cci_weight_circulatory(self):
        from processing.charlson import icd_to_cci_weight
        assert icd_to_cci_weight("I") == 1

    def test_cci_weight_respiratory(self):
        from processing.charlson import icd_to_cci_weight
        assert icd_to_cci_weight("J") == 2

    def test_cci_weight_infectious(self):
        from processing.charlson import icd_to_cci_weight
        assert icd_to_cci_weight("A") == 6
        assert icd_to_cci_weight("B") == 6

    def test_cci_weight_unknown(self):
        from processing.charlson import icd_to_cci_weight
        assert icd_to_cci_weight("Z") == 0

    def test_age_adjustment_under_50(self):
        from processing.charlson import age_adjustment
        assert age_adjustment(30) == 0
        assert age_adjustment(49) == 0

    def test_age_adjustment_over_70(self):
        from processing.charlson import age_adjustment
        assert age_adjustment(70) == 3
        assert age_adjustment(85) == 4

    def test_calculate_cci_polars_adds_column(self, sample_clinical_df):
        from processing.features import load_clinical
        from processing.charlson import calculate_cci_polars

        # Need Code_Letter column
        df = sample_clinical_df.with_columns(
            pl.col("primary_diagnosis_code").str.slice(0, 1).alias("Code_Letter")
        )
        result = calculate_cci_polars(df, age_adjust=True)
        assert "cci_score" in result.columns

    def test_cci_scores_non_negative(self, sample_clinical_df):
        from processing.charlson import calculate_cci_polars
        df = sample_clinical_df.with_columns(
            pl.col("primary_diagnosis_code").str.slice(0, 1).alias("Code_Letter")
        )
        result = calculate_cci_polars(df)
        assert result["cci_score"].min() >= 0

    def test_cci_risk_category(self):
        from processing.charlson import cci_risk_category
        assert cci_risk_category(0) == "Low (no comorbidities)"
        assert cci_risk_category(2) == "Mild"
        assert cci_risk_category(4) == "Moderate"
        assert cci_risk_category(5) == "Severe"


# ── features.py tests ─────────────────────────────────────────────────────────

class TestFeatures:
    def test_load_clinical_returns_polars(self, tmp_path, sample_clinical_df):
        from processing.features import load_clinical
        csv_path = tmp_path / "clinical.csv"
        sample_clinical_df.write_csv(csv_path)
        result = load_clinical(str(csv_path))
        assert isinstance(result, pl.DataFrame)

    def test_load_clinical_has_icd10_chapter(self, tmp_path, sample_clinical_df):
        from processing.features import load_clinical
        csv_path = tmp_path / "clinical.csv"
        sample_clinical_df.write_csv(csv_path)
        result = load_clinical(str(csv_path))
        assert "ICD10_Chapter" in result.columns

    def test_icd10_chapter_values(self, tmp_path, sample_clinical_df):
        from processing.features import load_clinical
        csv_path = tmp_path / "clinical.csv"
        sample_clinical_df.write_csv(csv_path)
        result = load_clinical(str(csv_path))
        valid = {"Circulatory System", "Respiratory System", "Infectious Diseases", "Other Chapters"}
        assert set(result["ICD10_Chapter"].unique().to_list()).issubset(valid)

    def test_engineer_features_columns(self, sample_clinical_df):
        from processing.features import load_clinical, engineer_features
        import io, csv as csv_mod
        # Use in-memory approach
        df = sample_clinical_df.with_columns(
            pl.col("primary_diagnosis_code").str.slice(0, 1).alias("Code_Letter")
        ).with_columns(
            pl.when(pl.col("Code_Letter") == "I").then(pl.lit("Circulatory System"))
            .when(pl.col("Code_Letter") == "J").then(pl.lit("Respiratory System"))
            .when(pl.col("Code_Letter").is_in(["A", "B"])).then(pl.lit("Infectious Diseases"))
            .otherwise(pl.lit("Other Chapters"))
            .alias("ICD10_Chapter")
        )
        result = engineer_features(df)
        for col in ["age_band", "polypharmacy_flag", "complexity_score", "charlson_proxy", "cci_score"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_age_band_values(self, sample_clinical_df):
        from processing.features import engineer_features
        df = sample_clinical_df.with_columns([
            pl.col("primary_diagnosis_code").str.slice(0, 1).alias("Code_Letter"),
            pl.lit("Other Chapters").alias("ICD10_Chapter"),
        ])
        result = engineer_features(df)
        valid_bands = {"Under 30", "30-49", "50-69", "70+"}
        assert set(result["age_band"].unique().to_list()).issubset(valid_bands)

    def test_polypharmacy_flag_binary(self, sample_clinical_df):
        from processing.features import engineer_features
        df = sample_clinical_df.with_columns([
            pl.col("primary_diagnosis_code").str.slice(0, 1).alias("Code_Letter"),
            pl.lit("Other Chapters").alias("ICD10_Chapter"),
        ])
        result = engineer_features(df)
        assert set(result["polypharmacy_flag"].unique().to_list()).issubset({0, 1})

    def test_build_feature_table_no_imd(self, tmp_path, sample_clinical_df):
        from processing.features import build_feature_table
        csv_path = tmp_path / "clinical.csv"
        sample_clinical_df.write_csv(csv_path)
        result = build_feature_table(clinical_path=str(csv_path), join_imd=False)
        assert len(result) == len(sample_clinical_df)
        assert "charlson_proxy" in result.columns


# ── train.py tests ────────────────────────────────────────────────────────────

class TestTrain:
    def _make_feature_df(self, sample_clinical_df) -> pl.DataFrame:
        """Build a minimal feature DataFrame for training."""
        return sample_clinical_df.with_columns([
            pl.col("primary_diagnosis_code").str.slice(0, 1).alias("Code_Letter"),
            pl.lit("Other Chapters").alias("ICD10_Chapter"),
        ]).with_columns([
            pl.lit("50-69").alias("age_band"),
            (pl.col("num_medications") >= 5).cast(pl.Int8).alias("polypharmacy_flag"),
            (pl.col("has_comorbidity").cast(pl.Int32) * 2).alias("charlson_proxy"),
            (pl.col("has_comorbidity").cast(pl.Int32) * 2).alias("cci_score"),
            (pl.col("num_lab_procedures") / 100.0 + pl.col("num_medications") / 50.0).alias("complexity_score"),
        ])

    def test_train_returns_metrics(self, tmp_path, sample_clinical_df):
        from ml.train import train
        import os
        # Redirect model output to tmp
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            df = self._make_feature_df(sample_clinical_df)
            metrics = train(df)
            assert "auc_roc" in metrics
            assert "f1_score" in metrics
            assert "feature_importance" in metrics
            assert 0 <= metrics["auc_roc"] <= 1
            assert 0 <= metrics["f1_score"] <= 1
        finally:
            os.chdir(original_cwd)

    def test_train_saves_model(self, tmp_path, sample_clinical_df):
        from ml.train import train
        import os
        os.chdir(tmp_path)
        try:
            df = self._make_feature_df(sample_clinical_df)
            train(df)
            assert (tmp_path / "models" / "readmission_model.pkl").exists()
        finally:
            os.chdir(Path(__file__).parents[2])

    def test_train_logs_metrics(self, tmp_path, sample_clinical_df):
        from ml.train import train
        import os
        os.chdir(tmp_path)
        try:
            df = self._make_feature_df(sample_clinical_df)
            train(df)
            log = tmp_path / "models" / "metrics_log.jsonl"
            assert log.exists()
            with open(log) as f:
                row = json.loads(f.readline())
            assert "trained_at" in row
        finally:
            os.chdir(Path(__file__).parents[2])

    def test_score_patients_adds_risk_score(self, tmp_path, sample_clinical_df):
        from ml.train import train, score_patients
        import os
        os.chdir(tmp_path)
        try:
            df = self._make_feature_df(sample_clinical_df)
            train(df)
            scored = score_patients(df)
            assert "risk_score" in scored.columns
            assert "risk_band" in scored.columns
            assert scored["risk_score"].min() >= 0.0
            assert scored["risk_score"].max() <= 1.0
        finally:
            os.chdir(Path(__file__).parents[2])

    def test_risk_bands_valid(self, tmp_path, sample_clinical_df):
        from ml.train import train, score_patients
        import os
        os.chdir(tmp_path)
        try:
            df = self._make_feature_df(sample_clinical_df)
            train(df)
            scored = score_patients(df)
            assert set(scored["risk_band"].unique().to_list()).issubset({"Low", "Medium", "High"})
        finally:
            os.chdir(Path(__file__).parents[2])


# ── Integration test ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self, tmp_path, sample_clinical_df):
        """Runs ingest → process → train end to end with synthetic data."""
        import os
        from processing.features import build_feature_table
        from ml.train import train, score_patients

        # Write synthetic CSV
        csv_path = tmp_path / "readmission_dataset.csv"
        sample_clinical_df.write_csv(csv_path)

        os.chdir(tmp_path)
        try:
            # Process
            df = build_feature_table(clinical_path=str(csv_path), join_imd=False)
            assert len(df) == 100
            assert "charlson_proxy" in df.columns

            # Train
            metrics = train(df)
            assert metrics["auc_roc"] > 0

            # Score
            scored = score_patients(df)
            assert len(scored) == 100
            assert "risk_score" in scored.columns

            print(f"\n  Integration test passed: AUC={metrics['auc_roc']}, F1={metrics['f1_score']}")
        finally:
            os.chdir(Path(__file__).parents[2])
