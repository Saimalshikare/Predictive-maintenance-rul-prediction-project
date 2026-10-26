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


def make_df(n_units=3, n_cycles=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, n_cycles + 1):
            rows.append([unit, cycle] + list(rng.normal(size=24)))
    df = pd.DataFrame(rows, columns=REQUIRED_COLS)
    df["setting_3"] = 100.0
    df["sensor_1"] = 518.67
    return df


def test_find_constant_columns():
    df = make_df()
    assert set(find_constant_columns(df)) == {"setting_3", "sensor_1"}


def test_drop_columns_removes_them():
    df = make_df()
    out = drop_columns(df, ["setting_3", "sensor_1"])
    assert "setting_3" not in out.columns
    assert "sensor_1" not in out.columns
    assert out.shape[1] == df.shape[1] - 2


def test_scaler_gives_zero_mean_unit_std_on_train():
    df = make_df(n_units=10, n_cycles=20)
    df = drop_columns(df, find_constant_columns(df))
    sensors = get_sensor_cols(df)
    scaled = apply_scaler(df, fit_scaler(df, sensors), sensors)
    assert np.allclose(scaled[sensors].mean(), 0, atol=1e-9)
    assert np.allclose(scaled[sensors].std(ddof=0), 1, atol=1e-9)


def test_scaler_uses_train_statistics_on_new_data():
    train = make_df(n_units=10, n_cycles=20, seed=0)
    test = make_df(n_units=5, n_cycles=20, seed=1)
    const = find_constant_columns(train)
    train, test = drop_columns(train, const), drop_columns(test, const)
    sensors = get_sensor_cols(train)
    scaler = fit_scaler(train, sensors)
    scaled_test = apply_scaler(test, scaler, sensors)
    assert not np.allclose(scaled_test[sensors].mean(), 0, atol=1e-9)


def test_preprocess_sorts_by_unit_and_cycle():
    df = make_df().sample(frac=1, random_state=1)
    out = preprocess_data(df)
    keys = list(zip(out["unit"], out["cycle"]))
    assert keys == sorted(keys)


def test_preprocess_rejects_missing_values():
    df = make_df()
    df.loc[0, "sensor_2"] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        preprocess_data(df)


def test_preprocess_rejects_duplicate_records():
    df = make_df()
    df = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(ValueError, match="Duplicate"):
        preprocess_data(df)


def test_preprocess_rejects_non_integer_cycle():
    df = make_df()
    df["cycle"] = df["cycle"].astype(float)
    df.loc[0, "cycle"] = 1.5
    with pytest.raises(ValueError, match="whole numbers"):
        preprocess_data(df)


def test_preprocess_rejects_missing_columns():
    df = make_df().drop(columns=["sensor_3"])
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocess_data(df)