from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .gates import assert_through
from .hashing import sha256_file
from .specialist_audit import FAILURE_AUROC_MIN, PROGRAMME_TARGETS
from .staged_final_scalar import (
    _atomic_json,
    _build_model,
    _cohort,
    _compact_names,
    _load_samples,
    _normalized_videos,
)


CALIBRATION_MEAN_ABSOLUTE_COVERAGE_ERROR_MAX = 0.10
CALIBRATION_90_ABSOLUTE_COVERAGE_ERROR_MAX = 0.10
MIN_VALIDATION_SAMPLES = 50
CENTRAL_GAUSSIAN_INTERVALS = {
    0.50: 0.6744897501960817,
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
}


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        return None
    if not np.isfinite(scores).all():
        return None
    return float(roc_auc_score(labels.astype(int), scores))


def _risk_coverage(
    values: np.ndarray,
    predictions: np.ndarray,
    risk_score: np.ndarray,
    failure: np.ndarray,
) -> list[dict[str, float | int]]:
    order = np.argsort(risk_score, kind="stable")
    rows: list[dict[str, float | int]] = []
    for requested_coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(1, int(round(len(order) * requested_coverage)))
        accepted = order[:count]
        residual = np.abs(predictions[accepted] - values[accepted])
        rows.append(
            {
                "requested_coverage": requested_coverage,
                "coverage": count / len(order),
                "accepted": count,
                "mae": float(residual.mean()),
                "median_ae": float(np.median(residual)),
                "failure_rate": float(failure[accepted].mean()),
            }
        )
    return rows


def _interval_calibration(
    values: np.ndarray,
    predictions: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, Any]:
    safe_sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)
    residual = np.asarray(values, dtype=np.float64) - np.asarray(
        predictions, dtype=np.float64
    )
    intervals: list[dict[str, float]] = []
    for nominal, z_score in CENTRAL_GAUSSIAN_INTERVALS.items():
        empirical = float(np.mean(np.abs(residual) <= z_score * safe_sigma))
        intervals.append(
            {
                "nominal_coverage": nominal,
                "empirical_coverage": empirical,
                "absolute_coverage_error": abs(empirical - nominal),
                "gaussian_z": z_score,
            }
        )
    mean_absolute_error = float(
        np.mean([row["absolute_coverage_error"] for row in intervals])
    )
    coverage_90 = next(
        row for row in intervals if math.isclose(row["nominal_coverage"], 0.90)
    )
    gaussian_nll = 0.5 * (
        np.log(2.0 * math.pi * safe_sigma**2) + (residual / safe_sigma) ** 2
    )
    return {
        "distribution": "raw moment-matched Gaussian central intervals from aleatoric plus ensemble variance; no evaluation-label rescaling",
        "intervals": intervals,
        "mean_absolute_coverage_error": mean_absolute_error,
        "coverage_90_absolute_error": coverage_90["absolute_coverage_error"],
        "mean_gaussian_nll": float(np.mean(gaussian_nll)),
        "mean_absolute_standardized_residual": float(
            np.mean(np.abs(residual) / safe_sigma)
        ),
        "passes_mean_coverage_error": bool(
            mean_absolute_error
            <= CALIBRATION_MEAN_ABSOLUTE_COVERAGE_ERROR_MAX
        ),
        "passes_90_coverage_error": bool(
            coverage_90["absolute_coverage_error"]
            <= CALIBRATION_90_ABSOLUTE_COVERAGE_ERROR_MAX
        ),
    }


def _aggregate_predictions(
    values: np.ndarray,
    seed_predictions: np.ndarray,
    seed_sigmas: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    predictions = np.asarray(seed_predictions, dtype=np.float64)
    sigmas = np.asarray(seed_sigmas, dtype=np.float64)
    if predictions.ndim != 2 or sigmas.shape != predictions.shape:
        raise ValueError("Seed predictions and sigmas must be equally shaped [seed, sample]")
    if predictions.shape[1] != len(values):
        raise ValueError("Prediction sample count does not match validation values")
    if not (
        np.isfinite(values).all()
        and np.isfinite(predictions).all()
        and np.isfinite(sigmas).all()
        and np.all(sigmas > 0)
    ):
        raise ValueError("Matched uncertainty inputs must be finite with positive sigmas")

    prediction = predictions.mean(axis=0)
    aleatoric_sigma = np.sqrt(np.mean(sigmas**2, axis=0))
    ensemble_dispersion = predictions.std(axis=0, ddof=0)
    total_uncertainty = np.sqrt(
        aleatoric_sigma**2 + ensemble_dispersion**2
    )
    absolute_error = np.abs(prediction - values)
    residual_threshold = float(np.quantile(absolute_error, 0.90))
    large_residual = absolute_error >= residual_threshold
    failure_auroc = _safe_auc(large_residual, total_uncertainty)
    risk_coverage = _risk_coverage(
        values, prediction, total_uncertainty, large_residual
    )
    calibration = _interval_calibration(
        values, prediction, total_uncertainty
    )
    return {
        "prediction": prediction,
        "aleatoric_sigma": aleatoric_sigma,
        "ensemble_dispersion": ensemble_dispersion,
        "total_uncertainty": total_uncertainty,
        "absolute_error": absolute_error,
        "large_residual": large_residual,
        "large_residual_threshold": residual_threshold,
        "large_residual_auroc": failure_auroc,
        "risk_coverage": risk_coverage,
        "calibration": calibration,
    }


def _evaluate_checkpoint(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    checkpoint_path: Path,
    *,
    target: str,
    seed: int,
    device: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    if str(checkpoint.get("target")) != target:
        raise RuntimeError(
            f"Checkpoint target mismatch: expected {target}, got {checkpoint.get('target')}"
        )
    if int(checkpoint.get("seed")) != seed:
        raise RuntimeError(
            f"Checkpoint seed mismatch: expected {seed}, got {checkpoint.get('seed')}"
        )
    if checkpoint.get("locked_test_accessed", False):
        raise RuntimeError("A staged checkpoint declares locked-test access")
    model, depth, _ = _build_model(config, device)
    compact_names = _compact_names(model, depth)
    for name, parameter in model.named_parameters():
        if name in compact_names and not name.startswith(
            ("head.", "mean.", "log_variance.")
        ):
            parameter.data = parameter.data.float()
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    checkpoint_keys = set(checkpoint["model"])
    unknown_checkpoint_keys = sorted(checkpoint_keys - set(model.state_dict()))
    missing_compact = sorted(compact_names - checkpoint_keys)
    if unexpected or unknown_checkpoint_keys or missing_compact:
        raise RuntimeError(
            "Compact checkpoint mismatch: "
            f"unexpected={unexpected}, unknown={unknown_checkpoint_keys}, "
            f"missing_compact={missing_compact}"
        )

    target_mean = float(checkpoint["target_mean"])
    target_std = float(checkpoint["target_std"])
    predictions: list[float] = []
    sigmas: list[float] = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            videos = _normalized_videos(sample["videos"], device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mean, log_variance = model(videos)
            predictions.append(float(mean) * target_std + target_mean)
            sigmas.append(
                math.sqrt(math.exp(float(log_variance))) * target_std
            )
    details = {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_mae": float(checkpoint["metrics"]["mae"]),
        "samples": len(samples),
        "locked_test_accessed": False,
    }
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return (
        np.asarray(predictions, dtype=np.float64),
        np.asarray(sigmas, dtype=np.float64),
        details,
    )


def _validate_stored_cohort(
    validation: list[dict[str, Any]], prediction_path: Path
) -> np.ndarray:
    stored = pd.read_csv(
        prediction_path,
        dtype={"sample_id": str, "subject_id": str},
        usecols=["sample_id", "subject_id", "value", "prediction"],
    )
    expected = pd.DataFrame(
        {
            "sample_id": [sample["sample_id"] for sample in validation],
            "subject_id": [sample["subject_id"] for sample in validation],
            "value": [sample["value"] for sample in validation],
            "_validation_order": np.arange(len(validation)),
        }
    )
    if stored[["sample_id", "subject_id"]].duplicated().any():
        raise RuntimeError(f"Duplicate validation identity in {prediction_path}")
    merged = expected.merge(
        stored,
        on=["sample_id", "subject_id"],
        how="outer",
        suffixes=("_expected", "_stored"),
        indicator=True,
        validate="one_to_one",
    )
    values_match = np.allclose(
        merged["value_expected"].to_numpy(dtype=float),
        merged["value_stored"].to_numpy(dtype=float),
        equal_nan=False,
    )
    if not merged["_merge"].eq("both").all() or not values_match:
        raise RuntimeError(
            f"Stored predictions do not match the immutable validation cohort: {prediction_path}"
        )
    # An outer merge sorts join keys in current pandas releases. Restore the
    # immutable validation order before comparing against checkpoint inference.
    merged = merged.sort_values("_validation_order", kind="stable")
    return merged["prediction"].to_numpy(dtype=np.float64)


def run_staged_scalar_uncertainty_audit(
    config: dict[str, Any],
    *,
    target: str,
    device: int,
    mode: str = "full",
) -> dict[str, Any]:
    """Audit staged scalar uncertainty on development validation only.

    This deliberately reconstructs legacy per-seed aleatoric outputs from the
    best checkpoints instead of trusting prediction CSVs that predate sigma
    persistence. It never selects, loads, or scores an internal locked-test row.
    """

    assert_through(config, "G5")
    if mode != "full":
        raise ValueError(
            "Matched promotion evidence is defined only for the full three-seed run"
        )
    if target not in PROGRAMME_TARGETS or target not in config["targets"]:
        raise ValueError(f"Unknown staged scalar target: {target}")
    rows = _cohort(config, target)
    cache_root = Path(config["paths"]["private_root"]) / "staged_scalar_video_cache"
    train, validation = _load_samples(
        rows,
        cache_root,
        dicom_root=Path(config["paths"]["dicom_root"]),
        maximum_train=None,
        maximum_validation=None,
        seed=int(config["splits"]["seed"]),
    )
    train_patients = {sample["subject_id"] for sample in train}
    validation_patients = {sample["subject_id"] for sample in validation}
    patient_disjoint = not (train_patients & validation_patients)
    if not patient_disjoint:
        raise RuntimeError("Patient leakage detected in staged uncertainty audit")

    configured_seeds = [int(seed) for seed in config["training"]["seeds"]]
    seeds = configured_seeds
    root = Path(config["paths"]["run_root"]) / "staged_final" / target
    seed_predictions: list[np.ndarray] = []
    seed_sigmas: list[np.ndarray] = []
    seed_details: list[dict[str, Any]] = []
    for seed in seeds:
        destination = root / f"seed_{seed}_{mode}"
        prediction_path = destination / "validation_predictions_private.csv"
        checkpoint_path = destination / "checkpoint_best.pt"
        stored_prediction = _validate_stored_cohort(validation, prediction_path)
        prediction, sigma, details = _evaluate_checkpoint(
            config,
            validation,
            checkpoint_path,
            target=target,
            seed=seed,
            device=device,
        )
        reproduction_error = float(
            np.max(np.abs(prediction - stored_prediction), initial=0.0)
        )
        if not np.allclose(
            prediction, stored_prediction, rtol=1e-5, atol=1e-2
        ):
            raise RuntimeError(
                "Best-checkpoint inference does not reproduce the stored validation "
                f"predictions for {target} seed {seed}; max_abs={reproduction_error}"
            )
        details["stored_prediction_reproduction_max_abs"] = reproduction_error
        seed_predictions.append(prediction)
        seed_sigmas.append(sigma)
        seed_details.append(details)

    values = np.asarray([sample["value"] for sample in validation], dtype=np.float64)
    predictions = np.stack(seed_predictions)
    sigmas = np.stack(seed_sigmas)
    aggregate = _aggregate_predictions(values, predictions, sigmas)
    ensemble_mae = float(aggregate["absolute_error"].mean())

    baseline_report = json.loads(
        (
            Path(config["paths"]["evidence_root"])
            / "G6"
            / "specialist_holdouts.json"
        ).read_text(encoding="utf-8")
    )
    baseline_mae = float(
        baseline_report["targets"][target]["internal_validation_mae"]
    )
    full_coverage_target = float(PROGRAMME_TARGETS[target])
    calibration = aggregate["calibration"]
    risk_coverage = aggregate["risk_coverage"]
    selective_risk_passed = bool(
        risk_coverage[-1]["mae"] <= risk_coverage[0]["mae"]
        and risk_coverage[-1]["failure_rate"] <= risk_coverage[0]["failure_rate"]
    )
    failure_auroc = aggregate["large_residual_auroc"]
    audit_checks = {
        "development_rows_only": True,
        "patient_disjoint": patient_disjoint,
        "exact_immutable_validation_cohort": True,
        "minimum_validation_samples": len(validation) >= MIN_VALIDATION_SAMPLES,
        "three_seed_full_ensemble": len(seeds) == 3,
        "finite_positive_uncertainty": bool(
            np.isfinite(aggregate["total_uncertainty"]).all()
            and np.all(aggregate["total_uncertainty"] > 0)
        ),
        "locked_test_not_accessed": True,
    }
    clinical_checks = {
        "full_coverage_programme_target_met": ensemble_mae
        <= full_coverage_target,
        "improves_current_g6_baseline": ensemble_mae < baseline_mae,
    }
    failure_checks = {
        "matched_large_residual_auroc_present": failure_auroc is not None,
        "matched_large_residual_auroc_at_least_0_70": bool(
            failure_auroc is not None and failure_auroc >= FAILURE_AUROC_MIN
        ),
        "selective_risk_improves_at_50_percent_coverage": selective_risk_passed,
    }
    calibration_checks = {
        "raw_mean_absolute_coverage_error_at_most_0_10": calibration[
            "passes_mean_coverage_error"
        ],
        "raw_90_interval_coverage_error_at_most_0_10": calibration[
            "passes_90_coverage_error"
        ],
    }
    audit_passed = all(audit_checks.values())
    promotion_eligible = bool(
        audit_passed
        and all(clinical_checks.values())
        and all(failure_checks.values())
        and all(calibration_checks.values())
    )

    private_frame = pd.DataFrame(
        {
            "sample_id": [sample["sample_id"] for sample in validation],
            "subject_id": [sample["subject_id"] for sample in validation],
            "study_id": [sample["study_id"] for sample in validation],
            "target": target,
            "value": values,
        }
    )
    for index, seed in enumerate(seeds):
        private_frame[f"prediction_seed_{seed}"] = predictions[index]
        private_frame[f"aleatoric_sigma_seed_{seed}"] = sigmas[index]
    private_frame["prediction"] = aggregate["prediction"]
    private_frame["aleatoric_sigma"] = aggregate["aleatoric_sigma"]
    private_frame["ensemble_dispersion"] = aggregate["ensemble_dispersion"]
    private_frame["total_uncertainty"] = aggregate["total_uncertainty"]
    private_frame["absolute_error"] = aggregate["absolute_error"]
    private_frame["large_residual"] = aggregate["large_residual"]
    private_frame["risk_rank_percentile"] = (
        private_frame["total_uncertainty"].rank(method="average", pct=True)
    )
    private_path = (
        Path(config["paths"]["private_root"])
        / f"staged_final_{target.lower()}_{mode}_matched_uncertainty_private.csv"
    )
    private_frame.to_csv(private_path, index=False)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": audit_passed,
        "scope": f"{target} development-only matched staged scalar uncertainty and failure audit",
        "mode": mode,
        "target": target,
        "validation_samples": len(validation),
        "validation_patients": len(validation_patients),
        "seeds": seed_details,
        "ensemble_validation_mae": ensemble_mae,
        "full_coverage_programme_target": full_coverage_target,
        "current_g6_baseline_mae": baseline_mae,
        "mae_change_vs_current_g6": ensemble_mae - baseline_mae,
        "audit_checks": audit_checks,
        "clinical_checks": clinical_checks,
        "failure_checks": failure_checks,
        "calibration_checks": calibration_checks,
        "large_residual_definition": "top development-validation absolute-error decile for this endpoint",
        "large_residual_threshold": aggregate["large_residual_threshold"],
        "large_residual_rate": float(aggregate["large_residual"].mean()),
        "matched_large_residual_auroc": failure_auroc,
        "uncertainty_definition": "sqrt(mean seed aleatoric variance + variance of seed predictions)",
        "mean_aleatoric_sigma": float(aggregate["aleatoric_sigma"].mean()),
        "mean_ensemble_dispersion": float(
            aggregate["ensemble_dispersion"].mean()
        ),
        "mean_total_uncertainty": float(aggregate["total_uncertainty"].mean()),
        "risk_coverage": risk_coverage,
        "interval_calibration": calibration,
        "pre_registered_engineering_thresholds": {
            "failure_auroc_min": FAILURE_AUROC_MIN,
            "mean_absolute_interval_coverage_error_max": CALIBRATION_MEAN_ABSOLUTE_COVERAGE_ERROR_MAX,
            "central_90_interval_coverage_error_max": CALIBRATION_90_ABSOLUTE_COVERAGE_ERROR_MAX,
            "minimum_validation_samples": MIN_VALIDATION_SAMPLES,
        },
        "promotion_eligible": promotion_eligible,
        "promotion_rule": "A full three-seed staged route must improve the current G6 baseline, meet the v4 full-coverage programme MAE, and pass matched raw interval-calibration, failure-AUROC, selective-risk, cohort-integrity, and locked-test gates.",
        "private_predictions": str(private_path),
        "claim_limit": "Development evidence only; this audit is not a temporal, train-excluded-vendor, locked external, or prospective claim.",
        "locked_test_accessed": False,
    }
    evidence_path = (
        Path(config["paths"]["evidence_root"])
        / "G6"
        / f"staged_final_{target.lower()}_{mode}_matched_uncertainty.json"
    )
    _atomic_json(evidence_path, report)
    return report
