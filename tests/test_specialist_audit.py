from __future__ import annotations

import numpy as np
import pandas as pd

from hemfm.specialist_audit import (
    _canonical_vendor,
    _input_sufficiency_stress,
    _risk_coverage,
    _risk_feature_matrix,
    _safe_auc,
)


def test_safe_auc_requires_two_classes() -> None:
    assert _safe_auc(np.zeros(4), np.arange(4)) is None
    assert _safe_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_risk_coverage_uses_low_uncertainty_first() -> None:
    values = np.zeros(10)
    predictions = np.arange(10, dtype=float)
    uncertainty = np.arange(10, dtype=float)

    curve = _risk_coverage(values, predictions, uncertainty)

    assert [point["coverage"] for point in curve] == [1.0, 0.9, 0.8, 0.7, 0.5]
    assert curve[-1]["mae"] < curve[0]["mae"]


def test_ge_manufacturer_aliases_are_one_vendor() -> None:
    assert _canonical_vendor("GE Healthcare Ultrasound") == "GE"
    assert _canonical_vendor("GE Vingmed Ultrasound") == "GE"


def test_input_sufficiency_stress_detects_predefined_failures() -> None:
    result = _input_sufficiency_stress(5)

    assert result["clean_examples"] == 5
    assert result["synthetic_failures"] == 15
    assert result["auroc"] == 1.0


def test_risk_feature_matrix_is_numeric_and_finite() -> None:
    frame = pd.DataFrame(
        {
            "prediction": [1.0, np.nan],
            "selected_cines": [2, 1],
            "minimum_frames": [8, 7],
            "minimum_axis_pixels": [300, 299],
            "mean_frames": [20, 10],
            "mean_axis_pixels": [512, 256],
            "view_confidence_minimum": [0.8, np.nan],
            "view_confidence_mean": [0.9, 0.4],
            "declared_view_count": [2, 1],
            "calibration_failure_proxy": [False, True],
        }
    )

    matrix = _risk_feature_matrix(frame, "prediction")

    assert matrix.shape == (2, 10)
    assert np.isfinite(matrix).all()

