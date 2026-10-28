import argparse

import numpy as np

from rul.data.load_data import load_rul, load_test, load_train
from rul.data.preprocess import (
    apply_scaler,
    drop_columns,
    find_constant_columns,
    fit_scaler,
    get_sensor_cols,
    preprocess_data,
)
from rul.features.build_features import (
    add_rolling,
    add_rul,
    feature_columns,
    last_cycle_per_unit,
)
from rul.models.evaluate import evaluate
from rul.models.train import save_artifacts, split_engines, train_model
from rul.utils.config import load_config, resolve
from rul.utils.logger import get_logger

logger = get_logger(__name__)


def build_train_features(df, sensors, scaler, clip, window):
    """Scale sensors, add RUL labels and rolling features to a train table."""
    df = apply_scaler(df, scaler, sensors)
    return add_rolling(add_rul(df, clip), sensors, window)


def run(dataset=None):
    cfg = load_config()
    dataset = dataset or cfg["data"]["dataset"]
    clip = cfg["features"]["rul_clip"]
    window = cfg["features"]["window_size"]
    seed = cfg["train"]["random_state"]
    val_size = cfg["train"]["val_size"]
    model_cfg = cfg["model"]

    raw_dir = resolve(cfg, "raw_dir")
    model_dir = resolve(cfg, "model_dir")

    if cfg["data"]["n_op_conditions"][dataset] > 1:
        logger.warning(
            "%s has several operating conditions. Per-condition scaling is "
            "not implemented yet, so results may be poor.", dataset
        )

    # 1. Load and validate raw data
    logger.info("Loading %s", dataset)
    train = preprocess_data(load_train(raw_dir, dataset))
    test = preprocess_data(load_test(raw_dir, dataset))
    rul_true = load_rul(raw_dir, dataset)

    # 2. Remove constant columns
    dropped = find_constant_columns(train)
    train = drop_columns(train, dropped)
    test = drop_columns(test, dropped)
    sensors = get_sensor_cols(train)
    features = feature_columns(sensors, window)
    logger.info("Dropped %d constant columns, kept %d sensors", len(dropped), len(sensors))

    # 3. Validation run: 80 engines train, 20 engines validation
    train_part, val_part = split_engines(train, val_size, seed)
    val_scaler = fit_scaler(train_part, sensors)
    train_part = build_train_features(train_part, sensors, val_scaler, clip, window)
    val_part = build_train_features(val_part, sensors, val_scaler, clip, window)

    val_model = train_model(train_part[features], train_part["RUL_clipped"], model_cfg, seed)
    val_pred = np.clip(val_model.predict(val_part[features]), 0, clip)
    val_metrics = evaluate(val_part["RUL_clipped"], val_pred, n_features=len(features))
    logger.info("Validation metrics: %s", {k: round(v, 3) for k, v in val_metrics.items()})

    # 4. Final run: train on all engines, test on the official test set
    scaler = fit_scaler(train, sensors)
    train_all = build_train_features(train, sensors, scaler, clip, window)
    model = train_model(train_all[features], train_all["RUL_clipped"], model_cfg, seed)

    test_scaled = apply_scaler(test, scaler, sensors)
    test_last = last_cycle_per_unit(add_rolling(test_scaled, sensors, window))
    test_pred = np.clip(model.predict(test_last[features]), 0, clip)
    test_metrics = evaluate(rul_true.loc[test_last.index], test_pred)
    logger.info("Test metrics: %s", {k: round(v, 3) for k, v in test_metrics.items()})

    # 5. Save everything the predict step needs
    meta = {
        "dataset": dataset,
        "model": "ExtraTreesRegressor",
        "dropped_columns": dropped,
        "sensor_cols": sensors,
        "feature_cols": features,
        "window_size": window,
        "rul_clip": clip,
        "n_features": len(features),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_artifacts(model, scaler, meta, model_dir)
    logger.info("Saved model, scaler and meta.json to %s", model_dir)
    return meta


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the RUL model.")
    parser.add_argument("--dataset", default=None, help="FD001, FD002, FD003 or FD004")
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()