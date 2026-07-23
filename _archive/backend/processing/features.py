"""
Feature engineering pipeline.
Joins IMD deprivation data with clinical records,
then builds all model-ready columns.
"""

import polars as pl
import numpy as np
from processing.charlson import calculate_cci_polars


def load_imd(path: str = "ID_2019.csv") -> pl.DataFrame:
    return pl.read_csv(path).rename({
        "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)": "IMD_Decile",
        "LSOA code (2011)": "LSOA_Code",
    })[["IMD_Decile", "LSOA_Code"]]


def load_clinical(path: str = "readmission_dataset.csv") -> pl.DataFrame:
    df = pl.read_csv(path)[[
        "readmitted_within_30days",
        "patient_id",
        "time_in_hospital",
        "num_medications",
        "primary_diagnosis_code",
        "has_comorbidity",
        "age",
        "gender",
        "admission_type",
        "num_lab_procedures",
    ]]

    df = df.with_columns(
        pl.col("primary_diagnosis_code").str.slice(0, 1).alias("Code_Letter")
    ).with_columns(
        pl.when(pl.col("Code_Letter") == "I")
        .then(pl.lit("Circulatory System"))
        .when(pl.col("Code_Letter") == "J")
        .then(pl.lit("Respiratory System"))
        .when(pl.col("Code_Letter").is_in(["A", "B"]))
        .then(pl.lit("Infectious Diseases"))
        .otherwise(pl.lit("Other Chapters"))
        .alias("ICD10_Chapter")
    )
    return df


def engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add derived columns used by the ML model and dashboards."""
    df = df.with_columns([
        # Age band
        pl.when(pl.col("age") < 30).then(pl.lit("Under 30"))
        .when(pl.col("age") < 50).then(pl.lit("30-49"))
        .when(pl.col("age") < 70).then(pl.lit("50-69"))
        .otherwise(pl.lit("70+"))
        .alias("age_band"),

        # Polypharmacy flag (5+ medications)
        (pl.col("num_medications") >= 5).cast(pl.Int8).alias("polypharmacy_flag"),

        # Complexity score (normalised composite)
        (
            (pl.col("num_lab_procedures") / 100.0)
            + (pl.col("num_medications") / 50.0)
            + pl.col("has_comorbidity").cast(pl.Float64)
        ).alias("complexity_score"),
    ])

    # Real Charlson Comorbidity Index (age-adjusted)
    df = calculate_cci_polars(df, age_adjust=True)

    # Keep charlson_proxy as alias for backwards compatibility with train.py
    df = df.with_columns(pl.col("cci_score").alias("charlson_proxy"))

    return df


def build_feature_table(
    clinical_path: str = "readmission_dataset.csv",
    imd_path: str = "ID_2019.csv",
    join_imd: bool = False,
) -> pl.DataFrame:
    df = load_clinical(clinical_path)
    df = engineer_features(df)

    if join_imd:
        imd = load_imd(imd_path)
        # Only join if LSOA_Code column exists in clinical data
        if "LSOA_Code" in df.columns:
            df = df.join(imd, on="LSOA_Code", how="left")

    return df


if __name__ == "__main__":
    df = build_feature_table()
    print(df.head())
    print(f"\nRows: {len(df)}, Columns: {df.columns}")
