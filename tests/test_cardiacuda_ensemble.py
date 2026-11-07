from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from hemfm.cardiacuda_ensemble import (
    _abstention_summary,
    _binary_auroc,
    _calibration_threshold,
    _ensemble_prediction,
    _patient_partition,
    _reliability_report,
    _risk_coverage,
    _validate_checkpoint,
)
from hemfm.cardiacuda_segmentation import (
    BASELINE_SCOPE,
    BASELINE_VIEW,
    GLOBAL_CLASSES,
    TRANSFORM_POLICY,
    VIEW_NAMES,
    ViewConditionedUNet,
    _canonical_fingerprint,
    _policy_fingerprint,
    active_class_mask,
)


def _checkpoint(seed: int = 1103, manifest: str = "a" * 64) -> dict:
    settings = {"resolution": 32, "base_channels": 4}
    config = {"scope": BASELINE_SCOPE, "view": BASELINE_VIEW, "settings": settings}
    return {
        "schema_version": 2,
        "scope": BASELINE_SCOPE,
        "sealed_test_accessed": False,
        "seed": seed,
        "manifest_fingerprint": manifest,
        "transform_policy_fingerprint": _policy_fingerprint(TRANSFORM_POLICY),
        "global_classes": GLOBAL_CLASSES,
        "view_names": VIEW_NAMES,
        "trained_views": [BASELINE_VIEW],
        "validation_views": [BASELINE_VIEW],
        "config": config,
        "config_fingerprint": _canonical_fingerprint(config),
        "settings": settings,
        "state_dict": ViewConditionedUNet(base_channels=4).state_dict(),
        "best_validation": {"macro_dice": 0.8},
    }


def test_checkpoint_validation_is_fail_closed_on_safety_and_ancestry() -> None:
    payload = _checkpoint()
    assert _validate_checkpoint(
        payload, expected_seed=1103, expected_manifest_fingerprint="a" * 64
    ) == {"resolution": 32, "base_channels": 4}

    unsafe = copy.deepcopy(payload)
    unsafe["sealed_test_accessed"] = 0
    with pytest.raises(ValueError, match="sealed TEST"):
        _validate_checkpoint(
            unsafe, expected_seed=1103, expected_manifest_fingerprint="a" * 64
        )

    drifted = copy.deepcopy(payload)
    drifted["manifest_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="manifest ancestry"):
        _validate_checkpoint(
            drifted, expected_seed=1103, expected_manifest_fingerprint="a" * 64
        )

    wrong_view = copy.deepcopy(payload)
    wrong_view["trained_views"] = ["PLAX"]
    with pytest.raises(ValueError, match="A4C-only"):
        _validate_checkpoint(
            wrong_view, expected_seed=1103, expected_manifest_fingerprint="a" * 64
        )


def test_patient_partition_is_manifest_bound_and_patient_disjoint() -> None:
    manifest = "1" * 64
    patients = [f"patient-{index}" for index in range(200)]
    first = {patient: _patient_partition(patient, manifest) for patient in patients}
    second = {patient: _patient_partition(patient, manifest) for patient in patients}

    assert first == second
    assert set(first.values()) == {"calibration", "audit"}
    calibration = {patient for patient, value in first.items() if value == "calibration"}
    audit = {patient for patient, value in first.items() if value == "audit"}
    assert not calibration & audit
    assert any(
        _patient_partition(patient, "2" * 64) != first[patient] for patient in patients
    )


class _FixedModel:
    def __init__(self, foreground_logit: float, log_variance: float) -> None:
        self.foreground_logit = foreground_logit
        self.log_variance = log_variance

    def __call__(self, images, _views):
        logits = torch.zeros(
            images.shape[0], len(GLOBAL_CLASSES), *images.shape[-2:], device=images.device
        )
        logits[:, 1] = self.foreground_logit
        variance = torch.full(
            (images.shape[0], 1, *images.shape[-2:]),
            self.log_variance,
            device=images.device,
        )
        return {"logits": logits, "log_variance": variance}


def test_ensemble_prediction_is_finite_and_records_disagreement() -> None:
    images = torch.zeros(1, 1, 4, 4)
    views = torch.zeros(1, dtype=torch.long)
    class_valid = torch.from_numpy(active_class_mask("A4C")).unsqueeze(0)
    agreeing = [_FixedModel(3.0, -2.0), _FixedModel(3.0, -2.0)]
    disagreeing = [_FixedModel(3.0, -2.0), _FixedModel(-3.0, -2.0)]

    probability, uncertainty, _entropy, _aleatoric, epistemic_agree = (
        _ensemble_prediction(agreeing, images, views, class_valid)
    )
    _, uncertainty_disagree, _, _, epistemic_disagree = _ensemble_prediction(
        disagreeing, images, views, class_valid
    )

    assert probability.shape == (1, len(GLOBAL_CLASSES), 4, 4)
    assert torch.isfinite(probability).all()
    assert torch.isfinite(uncertainty).all()
    assert float(epistemic_agree.max()) == pytest.approx(0.0, abs=1e-7)
    assert float(epistemic_disagree.mean()) > float(epistemic_agree.mean())
    assert float(uncertainty_disagree.mean()) > float(uncertainty.mean())


def test_abstention_threshold_uses_only_requested_calibration_coverage() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]

    threshold = _calibration_threshold(values, 0.8)

    assert threshold == pytest.approx(0.4)
    assert sum(value <= threshold for value in values) == 4
    with pytest.raises(ValueError, match="finite"):
        _calibration_threshold([0.1, np.nan], 0.8)


def test_failure_auroc_and_risk_coverage_reward_informative_uncertainty() -> None:
    assert _binary_auroc([False, False, True, True], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert _binary_auroc([False, False], [0.1, 0.2]) is None
    rows = [
        {"patient_id": "a", "macro_dice": 0.95, "mean_uncertainty": 0.1},
        {"patient_id": "b", "macro_dice": 0.90, "mean_uncertainty": 0.2},
        {"patient_id": "c", "macro_dice": 0.70, "mean_uncertainty": 0.8},
        {"patient_id": "d", "macro_dice": 0.60, "mean_uncertainty": 0.9},
    ]

    curve = _risk_coverage(rows)

    assert curve[0]["mean_macro_dice"] == pytest.approx(0.925)
    assert curve[-1]["mean_macro_dice"] == pytest.approx(0.7875)
    assert curve[0]["selective_risk_one_minus_dice"] < curve[-1][
        "selective_risk_one_minus_dice"
    ]


def test_pixel_reliability_ece_uses_population_weights() -> None:
    result = _reliability_report(
        counts=[2, 2],
        confidence_sums=[0.2, 1.8],
        correct_sums=[0.0, 2.0],
    )

    assert result["pixels"] == 4
    assert result["ece"] == pytest.approx(0.1)
    assert result["bins"][0]["accuracy"] == 0.0
    assert result["bins"][1]["accuracy"] == 1.0


def test_abstention_gate_fails_when_selective_risk_worsens() -> None:
    audit = [
        {"patient_id": "a", "macro_dice": 0.90},
        {"patient_id": "b", "macro_dice": 0.80},
        {"patient_id": "c", "macro_dice": 0.70},
    ]
    retained = [audit[1], audit[2]]

    result = _abstention_summary(
        audit, retained, [audit[0]], threshold=0.25
    )

    assert result["improves_selective_risk"] is False
    assert result["gate_status"] == "failed"
    assert result["audit_selective_risk_after"] > result["audit_selective_risk_before"]
