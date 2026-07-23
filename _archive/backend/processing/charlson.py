"""
Charlson Comorbidity Index (CCI) calculator.

The CCI is a standard clinical score that predicts 10-year mortality risk
based on a patient's combination of chronic conditions. Each condition
carries a weight (1, 2, 3, or 6). The total score is summed and optionally
adjusted for age.

Reference: Charlson ME et al. J Chronic Dis. 1987;40(5):373-83.

In this project we derive a proxy score from ICD-10 code prefixes
available in the readmission dataset. A full implementation would
map every ICD-10 code; here we map the chapters we have.
"""

import polars as pl
import pandas as pd


# ── ICD-10 first-letter → CCI weight mapping ─────────────────────────────────
# Weight 1 conditions
WEIGHT_1 = {"I"}     # Circulatory (myocardial infarction, peripheral vascular)
# Weight 2 conditions
WEIGHT_2 = {"J"}     # Respiratory (COPD, mild liver disease)
# Weight 3 conditions
WEIGHT_3 = set()     # Moderate/severe liver disease — not in this dataset
# Weight 6 conditions
WEIGHT_6 = {"A", "B"}  # Infectious disease (proxy for AIDS/HIV in original CCI)


def icd_to_cci_weight(code_letter: str) -> int:
    """Return CCI weight for a given ICD-10 first letter."""
    if code_letter in WEIGHT_6:
        return 6
    if code_letter in WEIGHT_3:
        return 3
    if code_letter in WEIGHT_2:
        return 2
    if code_letter in WEIGHT_1:
        return 1
    return 0


def age_adjustment(age: int) -> int:
    """
    Add 1 point per decade above 40 years.
    Standard CCI age adjustment (Quan et al. 2011).
    """
    if age < 50:
        return 0
    if age < 60:
        return 1
    if age < 70:
        return 2
    if age < 80:
        return 3
    return 4


def calculate_cci_polars(df: pl.DataFrame, age_adjust: bool = True) -> pl.DataFrame:
    """
    Add a `cci_score` column to a Polars DataFrame.

    Requires columns:
      - Code_Letter  (first character of primary_diagnosis_code)
      - has_comorbidity (0/1)
      - age (integer)

    Returns the DataFrame with an added `cci_score` column.
    """
    # Base score from diagnosis chapter
    df = df.with_columns(
        pl.col("Code_Letter")
        .map_elements(icd_to_cci_weight, return_dtype=pl.Int32)
        .alias("_cci_base")
    )

    # Comorbidity adds 2 points (proxy for one additional major comorbidity)
    df = df.with_columns(
        (pl.col("_cci_base") + pl.col("has_comorbidity").cast(pl.Int32) * 2)
        .alias("_cci_with_comorbidity")
    )

    # Polypharmacy proxy: 8+ medications → +1 (suggests complex disease burden)
    df = df.with_columns(
        (pl.col("_cci_with_comorbidity") + (pl.col("num_medications") > 8).cast(pl.Int32))
        .alias("_cci_with_poly")
    )

    if age_adjust:
        df = df.with_columns(
            pl.col("age")
            .map_elements(age_adjustment, return_dtype=pl.Int32)
            .alias("_age_adj")
        )
        df = df.with_columns(
            (pl.col("_cci_with_poly") + pl.col("_age_adj")).alias("cci_score")
        ).drop(["_cci_base", "_cci_with_comorbidity", "_cci_with_poly", "_age_adj"])
    else:
        df = df.with_columns(
            pl.col("_cci_with_poly").alias("cci_score")
        ).drop(["_cci_base", "_cci_with_comorbidity", "_cci_with_poly"])

    return df


def calculate_cci_pandas(df: pd.DataFrame, age_adjust: bool = True) -> pd.DataFrame:
    """Pandas version — used inside data.py for dashboard scoring."""
    df = df.copy()
    df["_cci_base"] = df["Code_Letter"].apply(icd_to_cci_weight)
    df["_cci_comorbid"] = df["_cci_base"] + df["has_comorbidity"].fillna(0).astype(int) * 2
    df["_cci_poly"] = df["_cci_comorbid"] + (df["num_medications"] > 8).astype(int)

    if age_adjust:
        df["_age_adj"] = df["age"].apply(age_adjustment)
        df["cci_score"] = df["_cci_poly"] + df["_age_adj"]
    else:
        df["cci_score"] = df["_cci_poly"]

    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)
    return df


def cci_risk_category(score: int) -> str:
    """
    Interpret a CCI score as a risk category.
    Based on Charlson's original survival estimates.
    """
    if score == 0:
        return "Low (no comorbidities)"
    if score <= 2:
        return "Mild"
    if score <= 4:
        return "Moderate"
    return "Severe"


if __name__ == "__main__":
    # Quick smoke test
    import polars as pl
    sample = pl.DataFrame({
        "Code_Letter":    ["I", "J", "A", "Z"],
        "has_comorbidity":[1,   0,   1,   0],
        "num_medications":[10,  3,   9,   2],
        "age":            [72,  45,  65,  30],
    })
    result = calculate_cci_polars(sample, age_adjust=True)
    print(result.select(["Code_Letter", "age", "has_comorbidity", "cci_score"]))
