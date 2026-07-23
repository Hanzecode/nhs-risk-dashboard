# _archive — inactive files

These files are **not used by the current LACE dashboard**. They were moved here
(not deleted) so the active project is easy to see. Paths below mirror their
original location, so restoring is just a matter of moving a file back.

Nothing in here is imported or read by the running app. The dashboard reads only:
`readmission_dataset.csv` (root), `.streamlit/config.toml`,
`backend/processing/lace.py`, and `backend/models/metrics_log.jsonl`.

## What's here

### `root/`, `backend/ID_2019.csv`, `backend/readmission_dataset.csv`
- **`root/ID_2019.csv`** — UK IMD deprivation dataset. The deprivation-vs-risk
  chart was removed (the clinical data has no geographic key to join on), so this
  is no longer used anywhere.
- **`backend/ID_2019.csv`, `backend/readmission_dataset.csv`** — duplicate copies.
  The active dataset is the single `readmission_dataset.csv` in the project root.

### `backend/` — legacy ML pipeline (superseded by the LACE index)
A Gradient Boosting classifier was built first, scored ~0.56 AUC (near-random,
because the synthetic label has no signal), and was replaced by the transparent
LACE index. This code is kept as evidence of that validation work.
- `run_pipeline.py` — 3-stage ingest/process/train runner
- `ml/train.py` — classifier training + metrics logging
- `processing/features.py`, `processing/charlson.py` — old feature engineering
- `tests/test_pipeline.py` — tests for the old pipeline
- `models/readmission_model.pkl` — the trained classifier (no longer loaded)

> Note: `backend/models/metrics_log.jsonl` was **kept active** — the Methodology
> page reads it to show the classifier's AUC as part of the "why LACE, not ML" story.

### `data/` — regenerable pipeline outputs
Dated Parquet files written by the old pipeline (`processed/` features and
`scored/` patients). The dashboard scores in memory, so these are not needed.

## Restoring something
Move it back to the same relative path, e.g.:

```bash
mv _archive/backend/run_pipeline.py backend/run_pipeline.py
```
