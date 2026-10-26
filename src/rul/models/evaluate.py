import numpy as np
from sklearn.metrics import r2_score


def _prepare(y_true, y_pred):
    """Convert inputs to 1-D float arrays and check they are usable."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.size == 0:
        raise ValueError("y_true is empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.size} values, "
            f"y_pred has {y_pred.size}."
        )
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("y_true and y_pred must contain only finite values.")

    return y_true, y_pred


def rmse(y_true, y_pred):
    y_true, y_pred = _prepare(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = _prepare(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true, y_pred):
    """
    PHM08 asymmetric score (lower is better).
    d = predicted - true. Late predictions (d > 0) are punished more.
    """
    y_true, y_pred = _prepare(y_true, y_pred)
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def r2(y_true, y_pred):
    """Supporting metric. RMSE, MAE and NASA are the main RUL metrics."""
    y_true, y_pred = _prepare(y_true, y_pred)
    if y_true.size < 2:
        raise ValueError("R2 needs at least 2 samples.")
    return float(r2_score(y_true, y_pred))


def adjusted_r2(y_true, y_pred, n_features):
    """
    Adjusted R2. n_features must be the real number of input features
    the model used (for example 45). It is only meaningful when the number
    of samples is much larger than the number of features.
    """
    y_true, y_pred = _prepare(y_true, y_pred)
    n = y_true.size
    if n_features < 0:
        raise ValueError("n_features cannot be negative.")
    if n - n_features - 1 <= 0:
        raise ValueError("Need more samples than features for adjusted R2.")
    return float(1 - (1 - r2(y_true, y_pred)) * (n - 1) / (n - n_features - 1))


def evaluate(y_true, y_pred, n_features=None):
    results = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "NASA": nasa_score(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    if n_features is not None:
        results["Adjusted_R2"] = adjusted_r2(y_true, y_pred, n_features)
    return results