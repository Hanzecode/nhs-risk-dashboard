"""
Train a readmission risk classifier and save the model + metrics.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

BACKEND_ROOT = Path(__file__).parents[1]
MODEL_PATH = BACKEND_ROOT / "models" / "readmission_model.pkl"
METRICS_PATH = BACKEND_ROOT / "models" / "metrics_log.jsonl"


FEATURE_COLS = [
    "time_in_hospital",
    "num_medications",
    "has_comorbidity",
    "age",
    "num_lab_procedures",
    "polypharmacy_flag",
    "charlson_proxy",
    "complexity_score",
]
TARGET_COL = "readmitted_within_30days"


def prepare_xy(df: pl.DataFrame):
    pdf = df.select(FEATURE_COLS + [TARGET_COL]).to_pandas().dropna()
    X = pdf[FEATURE_COLS].values
    y = pdf[TARGET_COL].values
    return X, y


def train(df: pl.DataFrame) -> dict:
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    f1 = round(float(f1_score(y_test, y_pred)), 4)

    importances = pipe.named_steps["clf"].feature_importances_
    feature_importance = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLS, importances)
    }

    metrics = {
        "trained_at": datetime.utcnow().isoformat(),
        "auc_roc": auc,
        "f1_score": f1,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_importance": feature_importance,
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    with open(METRICS_PATH, "a") as f:
        f.write(json.dumps(metrics) + "\n")

    print(f"AUC-ROC: {auc}  |  F1: {f1}")
    return metrics


def score_patients(df: pl.DataFrame) -> pl.DataFrame:
    """Add risk_score column (0-1) to every patient row."""
    with open(MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)

    pdf = df.to_pandas()
    available = [c for c in FEATURE_COLS if c in pdf.columns]
    X = pdf[available].fillna(0).values

    # Pad missing features with zeros if needed
    if X.shape[1] < len(FEATURE_COLS):
        pad = np.zeros((X.shape[0], len(FEATURE_COLS) - X.shape[1]))
        X = np.hstack([X, pad])

    scores = pipe.predict_proba(X)[:, 1]
    pdf["risk_score"] = scores.round(4)

    pdf["risk_band"] = np.where(
        pdf["risk_score"] >= 0.7, "High",
        np.where(pdf["risk_score"] >= 0.4, "Medium", "Low")
    )

    return pl.from_pandas(pdf)


def load_metrics_history() -> list[dict]:
    if not METRICS_PATH.exists():
        return []
    with open(METRICS_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    from processing.features import build_feature_table
    df = build_feature_table()
    metrics = train(df)
    print(json.dumps(metrics, indent=2))
