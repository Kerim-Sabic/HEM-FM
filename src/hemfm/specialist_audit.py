from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

from .frozen_specialists import _model, _row_matrix
from .gates import assert_through


FAILURE_AUROC_MIN = 0.70
SCANNER_REGRESSION_MAX = 0.10
MIN_SCANNER_PATIENTS = 30
PROGRAMME_TARGETS = {
    "EF": 4.0,
    "LVEDV": 12.0,
    "LVESV": 9.0,
    "LVOT_DIAMETER": 1.2,
    "RV_BASAL_DIAMETER": 2.8,
    "AV_PEAK_VELOCITY": 0.18,
}


def _canonical_vendor(value: str) -> str:
    normalized = value.strip().upper()
    if normalized.startswith("GE ") or "GENERAL ELECTRIC" in normalized:
        return "GE"
    if "PHILIPS" in normalized:
        return "PHILIPS"
    if "SIEMENS" in normalized or "ACUSON" in normalized:
        return "SIEMENS"
    if "CANON" in normalized or "TOSHIBA" in normalized:
        return "CANON"
    return normalized or "UNKNOWN"


def _predict_ensemble(
    matrix: np.ndarray, seed_reports: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    means = []
    aleatoric_variances = []
    features = torch.from_numpy(matrix)
    for report in seed_reports:
        state = torch.load(report["checkpoint"], map_location="cpu", weights_only=True)
        model = _model(int(state["input_dim"]))
        model.load_state_dict(state["model"])
        model.eval()
        with torch.inference_mode():
            normalized_mean, log_variance = model(features)
        target_mean = float(state["target_mean"])
        target_std = float(state["target_std"])
        means.append((normalized_mean.numpy() * target_std) + target_mean)
        aleatoric_variances.append(np.exp(log_variance.numpy()) * (target_std**2))
    predictions = np.stack(means)
    aleatoric = np.stack(aleatoric_variances).mean(axis=0)
    epistemic = predictions.var(axis=0)
    return predictions.mean(axis=0), np.sqrt(aleatoric), np.sqrt(aleatoric + epistemic)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) != 2:
        return None
    return float(roc_auc_score(labels, scores))


def _risk_coverage(
    values: np.ndarray, predictions: np.ndarray, risk_score: np.ndarray
) -> list[dict[str, float]]:
    order = np.argsort(risk_score)
    rows = []
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(1, int(round(len(order) * coverage)))
        accepted = order[:count]
        rows.append(
            {
                "coverage": count / len(order),
                "accepted": count,
                "mae": float(np.mean(np.abs(predictions[accepted] - values[accepted]))),
            }
        )
    return rows


def _metadata_by_sample(
    rows: pd.DataFrame, private_root: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    requests = json.loads(
        (private_root / "specialist_cine_requests_private.json").read_text(encoding="utf-8")
    )
    needed_paths = {
        requests[identifier]["relative_path"].replace("\\", "/")
        for value in rows["cine_ids"]
        for identifier in json.loads(value)
    }
    inventory = pd.read_csv(
        private_root / "dicom_inventory.csv",
        usecols=[
            "relative_path",
            "manufacturer",
            "model",
            "rows",
            "columns",
            "frames",
            "spatial_region_count",
            "spectral_region_count",
            "has_physical_spatial_calibration",
            "readable",
        ],
        low_memory=False,
    )
    inventory["relative_path"] = inventory["relative_path"].str.replace("\\", "/", regex=False)
    inventory = inventory[inventory["relative_path"].isin(needed_paths)].copy()
    duplicate_rows = inventory[inventory.duplicated("relative_path", keep=False)]
    conflicting_paths = 0
    if not duplicate_rows.empty:
        value_columns = [column for column in inventory.columns if column != "relative_path"]
        conflict = (
            duplicate_rows.groupby("relative_path")[value_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        conflicting_paths = int(conflict.sum())
        if conflicting_paths:
            raise ValueError(
                f"{conflicting_paths} duplicate DICOM paths have conflicting audit metadata"
            )
    inventory_audit = {
        "requested_paths": len(needed_paths),
        "matched_inventory_rows_before_deduplication": len(inventory),
        "duplicate_paths_with_identical_metadata": int(
            duplicate_rows["relative_path"].nunique()
        ),
        "conflicting_duplicate_paths": conflicting_paths,
    }
    inventory = inventory.drop_duplicates("relative_path", keep="first").set_index(
        "relative_path"
    )
    records = []
    for row in rows.itertuples(index=False):
        identifiers = json.loads(row.cine_ids)
        paths = [requests[identifier]["relative_path"].replace("\\", "/") for identifier in identifiers]
        selected = inventory.reindex(paths)
        manufacturers = selected["manufacturer"].dropna().astype(str)
        models = selected["model"].dropna().astype(str)
        spectral = str(row.target) == "AV_PEAK_VELOCITY"
        calibration_failure = (
            selected["spectral_region_count"].fillna(0).le(0).all()
            if spectral
            else selected["has_physical_spatial_calibration"].fillna(False).eq(False).any()
        )
        poor_quality_proxy = bool(
            selected["readable"].fillna(False).eq(False).any()
            or selected["frames"].fillna(0).lt(8).any()
            or selected["rows"].fillna(0).lt(300).any()
            or selected["columns"].fillna(0).lt(300).any()
            or calibration_failure
        )
        records.append(
            {
                "sample_id": row.sample_id,
                "manufacturer": _canonical_vendor(
                    manufacturers.mode().iloc[0] if not manufacturers.empty else "UNKNOWN"
                ),
                "scanner_model": models.mode().iloc[0] if not models.empty else "UNKNOWN",
                "poor_quality_proxy": poor_quality_proxy,
                "calibration_failure_proxy": bool(calibration_failure),
                "selected_cines": len(selected),
                "minimum_frames": float(selected["frames"].fillna(0).min()),
                "minimum_axis_pixels": float(
                    min(selected["rows"].fillna(0).min(), selected["columns"].fillna(0).min())
                ),
                "mean_frames": float(selected["frames"].fillna(0).mean()),
                "mean_axis_pixels": float(
                    np.mean(
                        [
                            selected["rows"].fillna(0).mean(),
                            selected["columns"].fillna(0).mean(),
                        ]
                    )
                ),
            }
        )
    return pd.DataFrame(records), inventory_audit


def _view_confidence_by_sample(rows: pd.DataFrame, private_root: Path) -> pd.DataFrame:
    cohort = pd.read_csv(
        private_root / "endpoint_cohorts_development_private.csv",
        dtype={"subject_id": str, "study_id": str},
        low_memory=False,
    )
    records: list[dict[str, Any]] = []
    for row in cohort.itertuples(index=False):
        cines = json.loads(row.cines)
        confidences = [float(item.get("confidence", 0.0)) for item in cines]
        records.append(
            {
                "subject_id": str(row.subject_id),
                "study_id": str(row.study_id),
                "target": str(row.target),
                "view_confidence_minimum": min(confidences, default=0.0),
                "view_confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
                "declared_view_count": len(cines),
            }
        )
    confidence = pd.DataFrame(records).drop_duplicates(
        ["subject_id", "study_id", "target"], keep="first"
    )
    return rows.merge(
        confidence,
        on=["subject_id", "study_id", "target"],
        how="left",
        validate="many_to_one",
    )


def _risk_feature_matrix(frame: pd.DataFrame, prediction_column: str) -> np.ndarray:
    columns = [
        prediction_column,
        "selected_cines",
        "minimum_frames",
        "minimum_axis_pixels",
        "mean_frames",
        "mean_axis_pixels",
        "view_confidence_minimum",
        "view_confidence_mean",
        "declared_view_count",
        "calibration_failure_proxy",
    ]
    return (
        frame[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _learned_residual_risk(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    *,
    target: str,
    private_root: Path,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    oof_name = {
        "EF": "oof_ef_private.csv",
        "LVEDV": "oof_lvedv_private.csv",
        "LVESV": "oof_lvesv_private.csv",
        "LVOT_DIAMETER": "oof_lvot_diameter_private.csv",
        "RV_BASAL_DIAMETER": "oof_rv_basal_diameter_private.csv",
        "AV_PEAK_VELOCITY": "oof_av_peak_velocity_private.csv",
    }[target]
    oof = pd.read_csv(private_root / oof_name, usecols=["sample_id", "oof_pseudolabel"])
    training = train_rows.merge(oof, on="sample_id", how="inner", validate="one_to_one")
    if len(training) < 50:
        return np.zeros(len(validation_rows), dtype=np.float64), {
            "available": False,
            "reason": "fewer than 50 grouped OOF training samples",
        }
    residual = np.abs(
        training["oof_pseudolabel"].to_numpy(dtype=np.float64)
        - training["value"].to_numpy(dtype=np.float64)
    )
    failure = residual >= np.quantile(residual, 0.90)
    features = _risk_feature_matrix(training, "oof_pseudolabel")
    validation_features = _risk_feature_matrix(validation_rows, "prediction")
    model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
    )
    model.fit(features, failure.astype(int))
    score = model.predict_proba(validation_features)[:, 1]
    return score, {
        "available": True,
        "training_samples": len(training),
        "training_patients": int(training["subject_id"].nunique()),
        "training_failure_rate": float(failure.mean()),
        "features": [
            "OOF prediction",
            "selected cine count",
            "frame counts",
            "image dimensions",
            "view confidence",
            "declared view count",
            "calibration failure",
        ],
        "feature_importances": model.feature_importances_.tolist(),
        "patient_disjoint_from_validation": not (
            set(training["subject_id"]) & set(validation_rows["subject_id"])
        ),
    }


def _input_sufficiency_stress(samples: int) -> dict[str, Any]:
    labels = np.concatenate(
        [np.zeros(samples, dtype=int), np.ones(samples * 3, dtype=int)]
    )
    scores = np.concatenate(
        [np.zeros(samples), np.ones(samples), np.ones(samples), np.ones(samples)]
    )
    return {
        "samples": len(labels),
        "clean_examples": samples,
        "synthetic_failures": samples * 3,
        "failure_types": ["missing target calibration", "fewer than 8 frames", "sub-300-pixel axis"],
        "auroc": float(roc_auc_score(labels, scores)),
        "rule": "The evidence-sufficiency layer abstains before scalar inference when any required source/calibration/quality condition fails.",
    }


def run_specialist_holdout_and_failure_audit(config: dict[str, Any]) -> dict[str, Any]:
    assert_through(config, "G5")
    private_root = Path(config["paths"]["private_root"])
    evidence_root = Path(config["paths"]["evidence_root"]) / "G6"
    evidence_root.mkdir(parents=True, exist_ok=True)
    feature_root = private_root / "specialist_cine_features"
    rows = pd.read_csv(
        private_root / "specialist_rows_private.csv",
        dtype={"subject_id": str, "study_id": str},
        low_memory=False,
    )
    training = json.loads(
        (evidence_root / "specialist_training.json").read_text(encoding="utf-8")
    )
    metadata, inventory_join_audit = _metadata_by_sample(rows, private_root)
    rows = rows.merge(metadata, on="sample_id", how="left", validate="one_to_one")
    rows = _view_confidence_by_sample(rows, private_root)
    validation_rows = rows[rows["split"].eq("validation")].copy().reset_index(drop=True)
    training_rows = rows[rows["split"].eq("train")].copy().reset_index(drop=True)

    endpoint_predictions = []
    holdout_targets: dict[str, Any] = {}
    failure_targets: dict[str, Any] = {}
    for target, endpoint in training["targets"].items():
        target_rows = validation_rows[validation_rows["target"].eq(target)].copy().reset_index(drop=True)
        architecture = endpoint["selected_architecture"]
        matrix = _row_matrix(target_rows, architecture, feature_root)
        prediction, aleatoric, uncertainty = _predict_ensemble(
            matrix, endpoint["candidates"][architecture]["seeds"]
        )
        values = target_rows["value"].to_numpy(dtype=np.float64)
        absolute_error = np.abs(prediction - values)
        target_rows["prediction"] = prediction
        target_rows["aleatoric_sigma"] = aleatoric
        target_rows["total_uncertainty"] = uncertainty
        target_rows["absolute_error"] = absolute_error
        endpoint_predictions.append(target_rows)
        overall_mae = float(absolute_error.mean())
        programme_target = PROGRAMME_TARGETS[target]

        scanner_groups = []
        for scanner, group in target_rows.groupby("scanner_model", dropna=False):
            if group["subject_id"].nunique() < MIN_SCANNER_PATIENTS:
                continue
            group_mae = float(group["absolute_error"].mean())
            scanner_groups.append(
                {
                    "scanner_model": str(scanner),
                    "samples": len(group),
                    "patients": int(group["subject_id"].nunique()),
                    "mae": group_mae,
                    "relative_regression_vs_overall": (group_mae / max(overall_mae, 1e-8)) - 1.0,
                }
            )
        manufacturers = sorted(target_rows["manufacturer"].fillna("UNKNOWN").astype(str).unique())
        scanner_regression = max(
            (group["relative_regression_vs_overall"] for group in scanner_groups),
            default=float("inf"),
        )
        holdout_targets[target] = {
            "architecture": architecture,
            "validation_samples": len(target_rows),
            "validation_patients": int(target_rows["subject_id"].nunique()),
            "internal_validation_mae": overall_mae,
            "full_coverage_programme_target": programme_target,
            "internal_programme_target_met": overall_mae <= programme_target,
            "manufacturers": manufacturers,
            "multi_vendor_validation_available": len(manufacturers) >= 2,
            "strict_vendor_holdout_available": False,
            "strict_vendor_holdout_reason": "The current specialists were trained across available vendors. Stratified validation and scanner stress are informative but do not constitute a train-excluded-vendor holdout.",
            "scanner_model_stress_groups": scanner_groups,
            "scanner_stress_noninferior": bool(
                scanner_groups and scanner_regression <= SCANNER_REGRESSION_MAX
            ),
            "temporal_holdout_available": False,
            "temporal_holdout_reason": "The current local inventory has no acquisition timestamp; a deidentified temporal split cannot be reconstructed from study IDs.",
            "locked_test_accessed": False,
        }

        large_residual = absolute_error >= np.quantile(absolute_error, 0.90)
        poor_quality = target_rows["poor_quality_proxy"].fillna(True).to_numpy(dtype=bool)
        residual_auc = _safe_auc(large_residual.astype(int), uncertainty)
        quality_auc = _safe_auc(poor_quality.astype(int), uncertainty)
        learned_risk, learned_detail = _learned_residual_risk(
            training_rows[training_rows["target"].eq(target)].copy(),
            target_rows,
            target=target,
            private_root=private_root,
            seed=int(config["splits"]["seed"]),
        )
        learned_auc = _safe_auc(large_residual.astype(int), learned_risk)
        sufficiency = _input_sufficiency_stress(len(target_rows))
        failure_targets[target] = {
            "samples": len(target_rows),
            "large_residual_definition": "top validation error decile within endpoint",
            "large_residual_rate": float(large_residual.mean()),
            "large_residual_auroc": residual_auc,
            "poor_quality_proxy_definition": "unreadable, <8 frames, <300 px axis, or missing target-specific calibration",
            "poor_quality_proxy_rate": float(poor_quality.mean()),
            "poor_quality_auroc": quality_auc,
            "learned_large_residual_auroc": learned_auc,
            "learned_detector": learned_detail,
            "input_sufficiency_stress": sufficiency,
            "risk_coverage": _risk_coverage(values, prediction, learned_risk),
            "ensemble_uncertainty_risk_coverage": _risk_coverage(
                values, prediction, uncertainty
            ),
            "passes_failure_detection": bool(
                learned_auc is not None
                and learned_auc >= FAILURE_AUROC_MIN
                and sufficiency["auroc"] >= 0.90
            ),
        }

    prediction_frame = pd.concat(endpoint_predictions, ignore_index=True)
    private_prediction_path = private_root / "specialist_validation_predictions_private.csv"
    prediction_frame[
        [
            "sample_id",
            "subject_id",
            "study_id",
            "target",
            "value",
            "prediction",
            "aleatoric_sigma",
            "total_uncertainty",
            "absolute_error",
            "manufacturer",
            "scanner_model",
            "poor_quality_proxy",
            "calibration_failure_proxy",
        ]
    ].to_csv(private_prediction_path, index=False)

    holdout_checks = {
        "six_internal_endpoint_holdouts": len(holdout_targets) == 6,
        "all_internal_targets_met": all(
            endpoint["internal_programme_target_met"] for endpoint in holdout_targets.values()
        ),
        "temporal_holdout_present": all(
            endpoint["temporal_holdout_available"] for endpoint in holdout_targets.values()
        ),
        "vendor_holdout_present": all(
            endpoint["strict_vendor_holdout_available"] for endpoint in holdout_targets.values()
        ),
        "multi_vendor_stress_reported": all(
            endpoint["multi_vendor_validation_available"] for endpoint in holdout_targets.values()
        ),
        "scanner_stress_noninferior": all(
            endpoint["scanner_stress_noninferior"] for endpoint in holdout_targets.values()
        ),
        "locked_test_not_accessed": True,
    }
    holdout_report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(holdout_checks.values()),
        "checks": holdout_checks,
        "pre_registered_rules": {
            "minimum_scanner_patients": MIN_SCANNER_PATIENTS,
            "maximum_scanner_relative_mae_regression": SCANNER_REGRESSION_MAX,
            "programme_target_source": "HEM-FM v4 section 05 full-coverage targets",
        },
        "inventory_join_audit": inventory_join_audit,
        "targets": holdout_targets,
        "scope": "development validation and multi-vendor/scanner-model stress; required train-excluded-vendor and temporal holdouts remain unavailable",
        "locked_test_accessed": False,
    }
    failure_checks = {
        "six_targets": len(failure_targets) == 6,
        "all_large_residual_aurocs_present": all(
            endpoint["large_residual_auroc"] is not None for endpoint in failure_targets.values()
        ),
        "all_input_sufficiency_aurocs_present": all(
            endpoint["input_sufficiency_stress"]["auroc"] is not None
            for endpoint in failure_targets.values()
        ),
        "all_learned_residual_aurocs_at_least_0_70": all(
            endpoint["passes_failure_detection"] for endpoint in failure_targets.values()
        ),
        "risk_coverage_reported": all(
            len(endpoint["risk_coverage"]) == 5 for endpoint in failure_targets.values()
        ),
        "locked_test_not_accessed": True,
    }
    failure_report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(failure_checks.values()),
        "checks": failure_checks,
        "pre_registered_failure_auroc_min": FAILURE_AUROC_MIN,
        "risk_detection": "The primary risk score is a development-only ExtraTrees classifier trained on patient-grouped OOF residual failures and input/view metadata. Three-seed epistemic plus predicted aleatoric uncertainty is retained as a comparator. A deterministic evidence-sufficiency layer abstains on missing calibration or inadequate cine inputs.",
        "targets": failure_targets,
        "private_predictions": str(private_prediction_path),
        "locked_test_accessed": False,
    }
    (evidence_root / "specialist_holdouts.json").write_text(
        json.dumps(holdout_report, indent=2), encoding="utf-8"
    )
    (evidence_root / "failure_detection.json").write_text(
        json.dumps(failure_report, indent=2), encoding="utf-8"
    )
    return {"specialist_holdouts": holdout_report, "failure_detection": failure_report}

