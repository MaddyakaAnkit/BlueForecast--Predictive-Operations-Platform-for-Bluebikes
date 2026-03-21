"""
BlueForecast training runner.

QUICK_CHECK = True  → 5% sample, ~30 sec, validates the full plumbing end-to-end
QUICK_CHECK = False → full 5.8M row training run, ~10–20 min on CPU
"""

import logging
import sys

sys.path.insert(0, "src")
logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

# ── Toggle here ──────────────────────────────────────────────────────────────
QUICK_CHECK = False  # flip to False for the real baseline run
SAMPLE_FRAC = 0.05   # fraction used when QUICK_CHECK=True
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import LabelEncoder

from model_pipeline.data_loader import load_feature_matrix, get_X_y, FEATURE_COLS
from model_pipeline.splitter import temporal_split
from model_pipeline.trainer import run_training_pipeline, DEFAULT_PARAMS

# 1. Load
df, version_hash = load_feature_matrix()

# 2. Encode start_station_id (string like 'C32078') to integer
#    Fit on full df so train/val/test share the same encoding
le = LabelEncoder()
df["start_station_id"] = le.fit_transform(df["start_station_id"])
print(f"Encoded {len(le.classes_)} unique station IDs to integers 0-{len(le.classes_)-1}")

# 3. Split
train_df, val_df, test_df = temporal_split(df)

# 4. Optionally sample for quick check
if QUICK_CHECK:
    train_df = train_df.sample(frac=SAMPLE_FRAC, random_state=42)
    val_df   = val_df.sample(frac=SAMPLE_FRAC, random_state=42)
    print(f"\n[QUICK_CHECK] Sampled {SAMPLE_FRAC*100:.0f}% -> "
          f"train={len(train_df):,} rows | val={len(val_df):,} rows\n")

# 5. Extract X/y
X_train, y_train = get_X_y(train_df)
X_val,   y_val   = get_X_y(val_df)

# 6. For quick check: fewer trees so it finishes fast
params = dict(DEFAULT_PARAMS)
if QUICK_CHECK:
    params["n_estimators"] = 50
    params["early_stopping_rounds"] = 10

# 7. Train + log to MLflow
forecaster, run_id = run_training_pipeline(
    X_train, y_train,
    X_val,   y_val,
    feature_cols=FEATURE_COLS,
    dataset_version_hash=version_hash,
    params=params,
)

print(f"\n{'='*60}")
print(f"Run ID:    {run_id}")
print(f"Mode:      {'QUICK_CHECK (5% sample)' if QUICK_CHECK else 'FULL TRAINING'}")
print(f"MLflow UI: http://localhost:5000  (run: mlflow ui)")
print(f"{'='*60}\n")
