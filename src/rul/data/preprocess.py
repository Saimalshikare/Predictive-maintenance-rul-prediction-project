import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from rul.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLS = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def get_sensor_cols(df):
    """Return sensor columns present in the DataFrame."""
    return [col for col in df.columns if col.startswith("sensor_")]


def find_constant_columns(df):
    """Identify constant operating-setting and sensor columns."""
    cols = [col for col in df.columns if col.startswith(("setting_", "sensor_"))]
    return [col for col in cols if df[col].nunique(dropna=False) == 1]


def preprocess_data(df):
    """
    Validate raw C-MAPSS data and sort it by engine and cycle.

    Run this on RAW data (all 26 columns present), before dropping
    constant columns. It does not scale, drop columns or create RUL.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    out = df.copy()

    if out.empty:
        raise ValueError("Input DataFrame is empty.")

    missing_cols = [col for col in REQUIRED_COLS if col not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if out[REQUIRED_COLS].isnull().any().any():
        missing_counts = out[REQUIRED_COLS].isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(f"Missing values found: {missing_counts}")

    for col in REQUIRED_COLS:
        out[col] = pd.to_numeric(out[col], errors="raise")

    if not np.isfinite(out[REQUIRED_COLS].to_numpy(dtype=float)).all():
        raise ValueError("Infinite or non-finite numeric values found.")

    if (out["unit"] <= 0).any():
        raise ValueError("Engine IDs must be positive.")

    if (out["cycle"] <= 0).any():
        raise ValueError("Cycle numbers must be positive.")

    for col in ("unit", "cycle"):
        if (out[col] % 1 != 0).any():
            raise ValueError(f"{col} must contain whole numbers.")

    if out.duplicated(subset=["unit", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle records found.")

    out = out.sort_values(["unit", "cycle"]).reset_index(drop=True)

    logger.info("Validated data. Shape: %s", out.shape)
    return out


def drop_columns(df, cols):
    """Return a copy of df without the given columns."""
    return df.drop(columns=cols)


def fit_scaler(train_df, sensor_cols):
    """Learn mean and std of each sensor from TRAIN data only."""
    scaler = StandardScaler()
    scaler.fit(train_df[sensor_cols])
    return scaler


def apply_scaler(df, scaler, sensor_cols):
    """Scale sensors with an already fitted scaler (no fitting here)."""
    out = df.copy()
    out[sensor_cols] = scaler.transform(df[sensor_cols])
    return out