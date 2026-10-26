import numpy as np
import pandas as pd

from rul.features.build_features import (
    add_rolling,
    add_rul,
    feature_columns,
    last_cycle_per_unit,
)


def make_df():
    return pd.DataFrame({
        "unit": [1] * 5 + [2] * 3,
        "cycle": list(range(1, 6)) + list(range(1, 4)),
        "sensor_2": np.arange(8, dtype=float),
    })


def test_rul_is_zero_at_last_cycle():
    out = add_rul(make_df())
    last = out.sort_values("cycle").groupby("unit").tail(1)
    assert (last["RUL"] == 0).all()


def test_rul_value_at_first_cycle():
    out = add_rul(make_df())
    first = out[(out["unit"] == 1) & (out["cycle"] == 1)]["RUL"].item()
    assert first == 4


def test_rul_is_never_negative():
    assert (add_rul(make_df())["RUL"] >= 0).all()


def test_rul_clipping_caps_values():
    out = add_rul(make_df(), clip=2)
    assert out["RUL_clipped"].max() == 2
    assert out["RUL"].max() == 4


def test_no_clipped_column_without_clip():
    assert "RUL_clipped" not in add_rul(make_df()).columns


def test_rolling_does_not_mix_engines():
    out = add_rolling(make_df(), ["sensor_2"], window=3)
    first_of_unit2 = out[(out["unit"] == 2) & (out["cycle"] == 1)]["sensor_2_mean3"].item()
    assert first_of_unit2 == 5.0


def test_rolling_mean_uses_only_past_values():
    out = add_rolling(make_df(), ["sensor_2"], window=3)
    value = out[(out["unit"] == 1) & (out["cycle"] == 3)]["sensor_2_mean3"].item()
    assert value == np.mean([0, 1, 2])


def test_rolling_std_has_no_missing_values():
    out = add_rolling(make_df(), ["sensor_2"], window=3)
    assert not out["sensor_2_std3"].isna().any()
    first = out[(out["unit"] == 1) & (out["cycle"] == 1)]["sensor_2_std3"].item()
    assert first == 0


def test_rolling_adds_expected_columns():
    out = add_rolling(make_df(), ["sensor_2"], window=3)
    assert "sensor_2_mean3" in out.columns
    assert "sensor_2_std3" in out.columns


def test_feature_columns_count_and_order():
    sensors = [f"sensor_{i}" for i in range(1, 16)]
    feats = feature_columns(sensors, 30)
    assert len(feats) == 45
    assert feats[:15] == sensors
    assert feats[15] == "sensor_1_mean30"
    assert feats[30] == "sensor_1_std30"


def test_last_cycle_per_unit():
    last = last_cycle_per_unit(make_df())
    assert list(last.index) == [1, 2]
    assert list(last["cycle"]) == [5, 3]