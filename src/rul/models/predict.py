import json

import joblib
import numpy as np
import pandas as pd

from rul.data.preprocess import apply_scaler, drop_columns, preprocess_data
from rul.features.build_features import add_rolling, last_cycle_per_unit

REQUIRED_META_KEYS = [
    "dropped_columns",
    "sensor_cols",
    "feature_cols",
    "window_size",
    "rul_clip",
]


def load_artifacts(model_dir):
    """Load the saved model, scaler and meta information."""
    model = joblib.load(model_dir / "rul_model_v1.joblib")
    scaler = joblib.load(model_dir / "scaler.joblib")
    with open(model_dir / "meta.json") as f:
        meta = json.load(f)
    return model, scaler, meta


def prepare_features(raw_df, scaler, meta):
    """Turn raw C-MAPSS rows into one row of features per engine."""
    missing = [key for key in REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"meta.json is missing keys: {missing}")

    df = preprocess_data(raw_df)
    df = drop_columns(df, meta["dropped_columns"])
    df = apply_scaler(df, scaler, meta["sensor_cols"])
    df = add_rolling(df, meta["sensor_cols"], meta["window_size"])
    return last_cycle_per_unit(df)[meta["feature_cols"]]


def predict_rul(raw_df, model, scaler, meta):
    """Predict the remaining useful life of each engine from its last cycle."""
    X = prepare_features(raw_df, scaler, meta)
    predictions = np.clip(model.predict(X), 0, meta["rul_clip"])
    return pd.Series(predictions, index=X.index, name="predicted_RUL")