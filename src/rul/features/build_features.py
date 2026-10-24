import pandas as pd


def add_rul(df, clip=None):
    out = df.copy()
    out["RUL"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    if clip is not None:
        out["RUL_clipped"] = out["RUL"].clip(upper=clip)
    return out


def add_rolling(df, cols, window):
    data = df.sort_values(["unit", "cycle"]).copy()
    g = data.groupby("unit")[cols]

    roll_mean = g.transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    ).add_suffix(f"_mean{window}")

    roll_std = g.transform(
        lambda s: s.rolling(window, min_periods=2).std()
    ).fillna(0).add_suffix(f"_std{window}")

    return pd.concat([data, roll_mean, roll_std], axis=1)


def feature_columns(sensor_cols, window):
    return (
        list(sensor_cols)
        + [f"{c}_mean{window}" for c in sensor_cols]
        + [f"{c}_std{window}" for c in sensor_cols]
    )


def last_cycle_per_unit(df):
    data = df.sort_values(["unit", "cycle"])
    return data.groupby("unit").tail(1).set_index("unit")