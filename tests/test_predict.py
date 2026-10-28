import numpy as np
import pandas as pd
import pytest

from rul.data.preprocess import (
    REQUIRED_COLS,
    apply_scaler,
    drop_columns,
    find_constant_columns,
    fit_scaler,
    get_sensor_cols,
    preprocess_data,
)
from rul.features.build_features import add_rolling, add_rul, feature_columns
from rul.models.predict import load_artifacts, predict_rul
from rul.models.train import save_artifacts, train_model

WINDOW = 5


def make_df(n_units, n_cycles, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, n_cycles + 1):
            rows.append([unit, cycle] + list(rng.normal(size=24)))
    df = pd.DataFrame(rows, columns=REQUIRED_COLS)
    df["setting_3"] = 100.0
    df["sensor_1"] = 518.67
    return df


def build_artifacts(tmp_path):
    train = preprocess_data(make_df(6, 40, seed=0))
    dropped = find_constant_columns(train)
    train = drop_columns(train, dropped)
    sensors = get_sensor_cols(train)
    scaler = fit_scaler(train, sensors)
    train = apply_scaler(train, scaler, sensors)
    train = add_rolling(add_rul(train, 125), sensors, WINDOW)
    features = feature_columns(sensors, WINDOW)

    model_cfg = {"n_estimators": 10, "min_samples_leaf": 2, "n_jobs": 1}
    model = train_model(train[features], train["RUL_clipped"], model_cfg, 42)

    meta = {
        "dropped_columns": dropped,
        "sensor_cols": sensors,
        "feature_cols": features,
        "window_size": WINDOW,
        "rul_clip": 125,
    }
    save_artifacts(model, scaler, meta, tmp_path)
    return load_artifacts(tmp_path)


def test_saved_files_exist(tmp_path):
    build_artifacts(tmp_path)
    for name in ("rul_model_v1.joblib", "scaler.joblib", "meta.json"):
        assert (tmp_path / name).exists()


def test_one_prediction_per_engine(tmp_path):
    model, scaler, meta = build_artifacts(tmp_path)
    new_data = make_df(4, 30, seed=1)
    pred = predict_rul(new_data, model, scaler, meta)
    assert len(pred) == 4
    assert list(pred.index) == [1, 2, 3, 4]


def test_predictions_stay_between_zero_and_clip(tmp_path):
    model, scaler, meta = build_artifacts(tmp_path)
    pred = predict_rul(make_df(4, 30, seed=2), model, scaler, meta)
    assert (pred >= 0).all()
    assert (pred <= meta["rul_clip"]).all()


def test_prediction_does_not_depend_on_row_order(tmp_path):
    model, scaler, meta = build_artifacts(tmp_path)
    new_data = make_df(4, 30, seed=3)
    shuffled = new_data.sample(frac=1, random_state=7)
    a = predict_rul(new_data, model, scaler, meta)
    b = predict_rul(shuffled, model, scaler, meta)
    assert np.allclose(a.values, b.values)


def test_missing_meta_key_raises_error(tmp_path):
    model, scaler, meta = build_artifacts(tmp_path)
    del meta["rul_clip"]
    with pytest.raises(ValueError, match="missing keys"):
        predict_rul(make_df(2, 10, seed=4), model, scaler, meta)