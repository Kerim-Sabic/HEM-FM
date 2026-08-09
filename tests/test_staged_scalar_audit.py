import numpy as np

from hemfm.staged_scalar_audit import (
    _aggregate_predictions,
    _interval_calibration,
    _risk_coverage,
)


def test_ensemble_uncertainty_combines_aleatoric_and_seed_dispersion() -> None:
    values = np.asarray([10.0, 20.0, 30.0, 40.0])
    predictions = np.asarray(
        [
            [9.0, 19.0, 29.0, 39.0],
            [11.0, 21.0, 31.0, 41.0],
            [10.0, 20.0, 30.0, 40.0],
        ]
    )
    sigmas = np.full_like(predictions, 2.0)
    result = _aggregate_predictions(values, predictions, sigmas)

    expected_dispersion = np.std(predictions, axis=0)
    assert np.allclose(result["prediction"], values)
    assert np.allclose(result["aleatoric_sigma"], 2.0)
    assert np.allclose(result["ensemble_dispersion"], expected_dispersion)
    assert np.allclose(
        result["total_uncertainty"],
        np.sqrt(4.0 + expected_dispersion**2),
    )


def test_interval_calibration_reports_raw_nominal_coverage_error() -> None:
    quantiles = np.linspace(-1.0, 1.0, 101)
    values = quantiles.copy()
    predictions = np.zeros_like(values)
    sigma = np.ones_like(values)

    result = _interval_calibration(values, predictions, sigma)

    assert len(result["intervals"]) == 4
    assert result["intervals"][0]["nominal_coverage"] == 0.50
    assert result["mean_absolute_coverage_error"] >= 0.0
    assert np.isfinite(result["mean_gaussian_nll"])


def test_risk_coverage_accepts_low_uncertainty_cases_first() -> None:
    values = np.zeros(10)
    predictions = np.arange(10, dtype=float)
    risk = np.arange(10, dtype=float)
    failure = predictions >= 9

    curve = _risk_coverage(values, predictions, risk, failure)

    assert curve[0]["coverage"] == 1.0
    assert curve[-1]["coverage"] == 0.5
    assert curve[-1]["mae"] < curve[0]["mae"]
    assert curve[-1]["failure_rate"] < curve[0]["failure_rate"]


def test_aggregate_rejects_nonpositive_sigma() -> None:
    values = np.asarray([1.0, 2.0])
    predictions = np.asarray([[1.0, 2.0], [1.0, 2.0]])
    sigmas = np.asarray([[1.0, 0.0], [1.0, 1.0]])

    try:
        _aggregate_predictions(values, predictions, sigmas)
    except ValueError as exc:
        assert "positive sigmas" in str(exc)
    else:
        raise AssertionError("nonpositive uncertainty must fail closed")
