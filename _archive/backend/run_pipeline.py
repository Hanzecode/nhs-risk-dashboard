"""
run_pipeline.py — run the full pipeline locally.

Usage:
    cd nhs-risk-dashboard
    python backend/run_pipeline.py

Stages:
    1. Ingest   — validate CSVs
    2. Process  — feature engineering, save Parquet
    3. Train    — train model, score patients, log metrics

Flags:
    --stage ingest|process|train   run only one stage (default: all)
    --clinical PATH                override clinical CSV path
    --imd PATH                     override IMD CSV path
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

# ── Load .env before anything else ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[1] / ".env")
except ImportError:
    pass  # dotenv optional — keys can be set as real env vars instead

# ── Make backend/ importable regardless of working directory ──────────────────
BACKEND = Path(__file__).parent
sys.path.insert(0, str(BACKEND))

CLINICAL_DEFAULT = Path(__file__).parents[1] / "readmission_dataset.csv"
IMD_DEFAULT      = Path(__file__).parents[1] / "ID_2019.csv"
DATA_DIR         = Path(__file__).parents[1] / "data"


# ── Stage 1: Ingest ───────────────────────────────────────────────────────────

def stage_ingest(clinical_path: Path, imd_path: Path) -> dict:
    print("\n── Stage 1: Ingest ───────────────────────────────────────────")
    import polars as pl

    if not clinical_path.exists():
        raise FileNotFoundError(
            f"Clinical CSV not found: {clinical_path}\n"
            "Place readmission_dataset.csv in the project root."
        )

    df = pl.read_csv(clinical_path)
    required = {"patient_id", "readmitted_within_30days", "age", "num_medications",
                "primary_diagnosis_code", "has_comorbidity", "time_in_hospital"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"  Clinical CSV : {clinical_path.name}  ({len(df):,} rows, {len(df.columns)} cols)")

    if imd_path.exists():
        imd = pl.read_csv(imd_path)
        print(f"  IMD CSV      : {imd_path.name}  ({len(imd):,} rows)")
    else:
        print("  IMD CSV      : not found — deprivation features will be simulated")

    print("  ✓ Ingest complete")
    return {"clinical_path": str(clinical_path), "imd_path": str(imd_path)}


# ── Stage 2: Process ──────────────────────────────────────────────────────────

def stage_process(clinical_path: str, imd_path: str) -> dict:
    print("\n── Stage 2: Process ──────────────────────────────────────────")
    from processing.features import build_feature_table

    imd_exists = Path(imd_path).exists()
    df = build_feature_table(
        clinical_path=clinical_path,
        imd_path=imd_path,
        join_imd=imd_exists,
    )

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = date.today().isoformat()
    out_path = out_dir / f"features_{run_date}.parquet"
    df.write_parquet(out_path)

    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {len(df.columns)}")
    print(f"  Saved locally: {out_path.relative_to(Path(__file__).parents[1])}")
    print("  ✓ Process complete")
    return {"features_path": str(out_path)}


# ── Stage 3: Train ────────────────────────────────────────────────────────────

def stage_train(features_path: str) -> dict:
    print("\n── Stage 3: Train ────────────────────────────────────────────")
    import polars as pl
    from ml.train import train, score_patients

    df = pl.read_parquet(features_path)
    print(f"  Loaded       : {len(df):,} rows")

    metrics = train(df)
    print(f"  AUC-ROC      : {metrics['auc_roc']}")
    print(f"  F1 score     : {metrics['f1_score']}")
    print(f"  Trained at   : {metrics['trained_at'][:19]}")

    df_scored = score_patients(df)
    out_dir = DATA_DIR / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = date.today().isoformat()
    out_path = out_dir / f"patients_{run_date}.parquet"
    df_scored.write_parquet(out_path)

    band_counts = df_scored["risk_band"].value_counts().sort("risk_band")
    print("  Risk bands   :")
    for row in band_counts.iter_rows(named=True):
        print(f"    {row['risk_band']:8s} : {row['count']:,}")

    print(f"  Saved locally: {out_path.relative_to(Path(__file__).parents[1])}")
    print("  ✓ Train complete")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NHS Readmission Risk Pipeline")
    parser.add_argument("--stage", choices=["ingest", "process", "train"],
                        help="Run only one stage (default: run all three)")
    parser.add_argument("--clinical", default=str(CLINICAL_DEFAULT),
                        help="Path to readmission_dataset.csv")
    parser.add_argument("--imd",      default=str(IMD_DEFAULT),
                        help="Path to ID_2019.csv")
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("  NHS Readmission Risk Pipeline")
    print(f"  Run date  : {date.today().isoformat()}")
    print("=" * 60)

    try:
        ctx = {}

        if args.stage in (None, "ingest"):
            ctx = stage_ingest(
                clinical_path=Path(args.clinical),
                imd_path=Path(args.imd),
            )

        if args.stage in (None, "process"):
            if args.stage == "process":
                ctx = {"clinical_path": args.clinical, "imd_path": args.imd}
            ctx = stage_process(ctx["clinical_path"], ctx["imd_path"])

        if args.stage in (None, "train"):
            if args.stage == "train":
                processed = sorted((DATA_DIR / "processed").glob("*.parquet"))
                if not processed:
                    raise FileNotFoundError(
                        "No processed parquet found. Run --stage process first."
                    )
                ctx = {"features_path": str(processed[-1])}
            stage_train(ctx["features_path"])

        elapsed = round(time.time() - t0, 1)
        print(f"\n{'='*60}")
        print(f"  Pipeline complete in {elapsed}s")
        print(f"  Dashboard : streamlit run dashboard/app.py")
        print(f"{'='*60}\n")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ERROR: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
