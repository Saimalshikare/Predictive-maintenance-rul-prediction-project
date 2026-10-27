import json

import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


def split_engines(df, val_size, random_state):
    """Split whole engines (not rows) into train and validation sets."""
    engine_ids = df["unit"].unique()
    train_units, val_units = train_test_split(
        engine_ids, test_size=val_size, random_state=random_state
    )
    train_df = df[df["unit"].isin(train_units)].copy()
    val_df = df[df["unit"].isin(val_units)].copy()
    return train_df, val_df


def build_model(model_cfg, random_state):
    """Create an untrained Extra Trees model from the config settings."""
    return ExtraTreesRegressor(
        n_estimators=model_cfg["n_estimators"],
        min_samples_leaf=model_cfg["min_samples_leaf"],
        n_jobs=model_cfg["n_jobs"],
        random_state=random_state,
    )


def train_model(X, y, model_cfg, random_state):
    """Build the model and fit it on the given data."""
    model = build_model(model_cfg, random_state)
    model.fit(X, y)
    return model


def save_artifacts(model, scaler, meta, model_dir):
    """Save the model, the scaler and a small JSON file describing them."""
    joblib.dump(model, model_dir / "rul_model_v1.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)