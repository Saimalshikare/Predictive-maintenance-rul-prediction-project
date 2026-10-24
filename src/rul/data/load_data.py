from pathlib import Path
import pandas as pd

COLUMNS = (
    ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def _read(path):
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS
    return df


def load_train(raw_dir, dataset):
    return _read(Path(raw_dir) / f"train_{dataset}.txt")


def load_test(raw_dir, dataset):
    return _read(Path(raw_dir) / f"test_{dataset}.txt")


def load_rul(raw_dir, dataset):
    s = pd.read_csv(Path(raw_dir) / f"RUL_{dataset}.txt", header=None)[0]
    s.index = range(1, len(s) + 1)
    s.name = "RUL"
    return s