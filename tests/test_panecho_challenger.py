from __future__ import annotations

import numpy as np

from hemfm.panecho_challenger import _fit_grouped_calibration, _risk_coverage


def test_grouped_calibration_learns_linear_scale_without_patient_leakage() -> None:
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(30), 2)
    raw = rng.normal(size=(60, 3))
    targets = np.column_stack(
        [55 + 7 * raw[:, 0], 110 + 20 * raw[:, 1], 45 + 12 * raw[:, 2]]
    )
    validation_raw = rng.normal(size=(12, 3))
    result = _fit_grouped_calibration(raw, targets, validation_raw, groups, folds=5)

    assert result["group_folds"] == 5
    assert result["oof"].shape == targets.shape
    assert result["validation"].shape == (12, 3)
    assert all(alpha in (0.01, 0.1, 1.0, 10.0, 100.0) for alpha in result["selected_alphas"].values())


def test_risk_coverage_accepts_low_risk_first() -> None:
    target = np.zeros(10)
    prediction = np.arange(10, dtype=float)
    risk = np.arange(10, dtype=float)
    curve = _risk_coverage(target, prediction, risk)
    assert curve[-1]["mae"] < curve[0]["mae"]

