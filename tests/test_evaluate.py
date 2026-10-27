import pytest

from rul.models.evaluate import adjusted_r2, evaluate, mae, nasa_score, rmse


def test_perfect_prediction_gives_zero_errors():
    y = [10, 20, 30]
    assert rmse(y, y) == 0
    assert mae(y, y) == 0
    assert nasa_score(y, y) == 0


def test_rmse_and_mae_known_values():
    assert mae([10, 20], [12, 18]) == 2
    assert rmse([10, 20], [12, 18]) == 2


def test_late_prediction_is_punished_more_than_early():
    assert nasa_score([50], [60]) > nasa_score([50], [40])


def test_column_shaped_input_is_handled():
    assert rmse([[10], [20]], [12, 18]) == 2


def test_evaluate_returns_expected_keys():
    out = evaluate([10, 20, 30, 40], [12, 18, 33, 38])
    assert set(out) == {"RMSE", "MAE", "R2", "NASA"}


def test_evaluate_adds_adjusted_r2_when_requested():
    out = evaluate([10, 20, 30, 40, 50, 60], [12, 18, 33, 38, 52, 59], n_features=2)
    assert "Adjusted_R2" in out
    assert out["Adjusted_R2"] < out["R2"]


def test_adjusted_r2_needs_more_samples_than_features():
    with pytest.raises(ValueError):
        adjusted_r2([1, 2, 3], [1, 2, 3], n_features=5)


def test_length_mismatch_raises_error():
    with pytest.raises(ValueError, match="Length mismatch"):
        rmse([1, 2, 3], [1, 2])


def test_single_value_is_not_silently_broadcast():
    with pytest.raises(ValueError, match="Length mismatch"):
        rmse([50], [10, 20, 30])


def test_nan_values_raise_error():
    with pytest.raises(ValueError, match="finite"):
        mae([1, 2, 3], [1, float("nan"), 3])


def test_empty_input_raises_error():
    with pytest.raises(ValueError, match="empty"):
        nasa_score([], [])