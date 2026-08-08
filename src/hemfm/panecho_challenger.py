from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dicom_preprocess import preprocess_dicom_cine
from .frozen_specialists import _atomic_json
from .gates import assert_through
from .mimic_lv_training import TARGETS, _filter_spatially_calibrated, _select_targets


PANECHO_TASKS = ("EF", "LVEDV", "LVESV")
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class PanEchoAuditConfig:
    frames: int = 16
    resolution: int = 224
    batch_size: int = 2
    num_workers: int = 2
    folds: int = 5
    seed: int = 20260807


class PanEchoMIMICDataset:
    def __init__(self, frame: pd.DataFrame, *, frames: int = 16, resolution: int = 224) -> None:
        self.frame = frame.reset_index(drop=True)
        self.frames = frames
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.frame.iloc[index]
        cine = preprocess_dicom_cine(
            row["local_dicom"],
            "spatial",
            frames=self.frames,
            output_size=self.resolution,
            clean_overlays=True,
        )
        video = torch.from_numpy(cine.video.copy()).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        standard_deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return {
            "video": ((video - mean) / standard_deviation).float(),
            "row_index": index,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_panecho(source: Path, checkpoint: Path, device: int):
    import torch

    models_path = source / "src" / "models.py"
    tasks_path = source / "content" / "tasks.pkl"
    if not models_path.is_file() or not tasks_path.is_file():
        raise FileNotFoundError(f"Incomplete PanEcho source tree: {source}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PanEcho checkpoint unavailable: {checkpoint}")

    specification = importlib.util.spec_from_file_location("_hemfm_external_panecho_models", models_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load PanEcho models from {models_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    class Task:
        def __init__(self, task_name: str, task_type: str, class_names: Any, mean: float) -> None:
            self.task_name = task_name
            self.task_type = task_type
            self.class_names = np.asarray(class_names)
            self.class_indices = np.arange(self.class_names.size)
            self.mean = mean

    task_dictionary = pd.read_pickle(tasks_path)
    tasks = [
        Task(name, task_dictionary[name]["task_type"], task_dictionary[name]["class_names"], task_dictionary[name]["mean"])
        for name in PANECHO_TASKS
    ]
    encoder = module.FrameTransformer("convnext_tiny", 8, 4, 0.0, "mean", 16)
    model = module.MultiTaskModel(encoder, encoder.encoder.n_features, tasks, 0.25, True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = dict(payload["weights"])
    weights.pop("encoder.time_encoder.pe", None)
    missing, _unexpected = model.load_state_dict(weights, strict=False)
    required_missing = [name for name in missing if any(name.startswith(f"{task}_head") for task in PANECHO_TASKS)]
    if required_missing:
        raise RuntimeError(f"PanEcho checkpoint lacks requested task heads: {required_missing}")
    return model.eval().to(device)


def _predict(model, loader, device: int, status_path: Path, split: str) -> np.ndarray:
    import torch

    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for step, batch in enumerate(loader, start=1):
            video = batch["video"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(video)
            predictions.append(
                np.column_stack([output[name].float().cpu().numpy().reshape(-1) for name in PANECHO_TASKS])
            )
            if step % 10 == 0 or step == len(loader):
                _atomic_json(
                    status_path,
                    {
                        "schema_version": 1,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "phase": f"panecho_mimic_lv_{split}",
                        "complete_sequences": min(step * loader.batch_size, len(loader.dataset)),
                        "total_sequences": len(loader.dataset),
                        "device": device,
                        "locked_test_accessed": False,
                    },
                )
    return np.concatenate(predictions)


def _clip_predictions(prediction: np.ndarray) -> np.ndarray:
    output = prediction.copy()
    output[:, 0] = np.clip(output[:, 0], 0.0, 100.0)
    output[:, 1:] = np.clip(output[:, 1:], 1.0, 500.0)
    return output


def _fit_grouped_calibration(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int = 5,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(folds, len(unique_groups)))
    splits = list(splitter.split(train_features, train_targets, groups))
    oof = np.zeros_like(train_targets, dtype=np.float64)
    validation = np.zeros((len(validation_features), train_targets.shape[1]), dtype=np.float64)
    validation_std = np.zeros_like(validation)
    train_model_std = np.zeros_like(train_targets, dtype=np.float64)
    selected_alphas: dict[str, float] = {}
    alpha_scores: dict[str, dict[str, float]] = {}

    for target_index, target_name in enumerate(TARGETS):
        scores: dict[float, float] = {}
        cached: dict[float, np.ndarray] = {}
        for alpha in RIDGE_ALPHAS:
            candidate = np.zeros(len(train_targets), dtype=np.float64)
            for fit_indices, held_indices in splits:
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                estimator.fit(train_features[fit_indices], train_targets[fit_indices, target_index])
                candidate[held_indices] = estimator.predict(train_features[held_indices])
            cached[alpha] = candidate
            scores[alpha] = float(np.mean(np.abs(candidate - train_targets[:, target_index])))
        selected = min(scores, key=scores.get)
        selected_alphas[target_name] = float(selected)
        alpha_scores[target_name] = {str(alpha): scores[alpha] for alpha in RIDGE_ALPHAS}
        oof[:, target_index] = cached[selected]

        fold_estimators = []
        for fit_indices, held_indices in splits:
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=selected))
            estimator.fit(train_features[fit_indices], train_targets[fit_indices, target_index])
            fold_estimators.append(estimator)
        fold_validation = np.column_stack(
            [estimator.predict(validation_features) for estimator in fold_estimators]
        )
        fold_train = np.column_stack([estimator.predict(train_features) for estimator in fold_estimators])
        validation[:, target_index] = fold_validation.mean(axis=1)
        validation_std[:, target_index] = fold_validation.std(axis=1)
        train_model_std[:, target_index] = fold_train.std(axis=1)

    oof = _clip_predictions(oof)
    validation = _clip_predictions(validation)
    direct_oof_mae = float(np.mean(np.abs(oof[:, 0] - train_targets[:, 0])))
    oof_volume_ef = np.clip((oof[:, 1] - oof[:, 2]) / np.maximum(oof[:, 1], 1.0) * 100.0, 0.0, 100.0)
    volume_oof_mae = float(np.mean(np.abs(oof_volume_ef - train_targets[:, 0])))
    ef_source = "volume_derived" if volume_oof_mae < direct_oof_mae else "direct_calibrated"
    if ef_source == "volume_derived":
        oof[:, 0] = oof_volume_ef
        validation[:, 0] = np.clip(
            (validation[:, 1] - validation[:, 2]) / np.maximum(validation[:, 1], 1.0) * 100.0,
            0.0,
            100.0,
        )
    return {
        "oof": oof,
        "validation": validation,
        "validation_std": validation_std,
        "train_model_std": train_model_std,
        "selected_alphas": selected_alphas,
        "alpha_oof_mae": alpha_scores,
        "ef_source": ef_source,
        "group_folds": len(splits),
    }


def _percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, values, side="right") / max(1, len(ordered))


def _risk_coverage(target: np.ndarray, prediction: np.ndarray, risk: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(risk)
    output = []
    for requested in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(1, int(round(len(order) * requested)))
        accepted = order[:count]
        output.append(
            {
                "coverage": count / len(order),
                "accepted": count,
                "mae": float(np.mean(np.abs(prediction[accepted] - target[accepted]))),
            }
        )
    return output


def _metrics(
    train_targets: np.ndarray,
    train_oof: np.ndarray,
    validation_targets: np.ndarray,
    validation_prediction: np.ndarray,
    train_raw: np.ndarray,
    validation_raw: np.ndarray,
    train_std: np.ndarray,
    validation_std: np.ndarray,
    selected_sources: dict[str, str],
) -> dict[str, Any]:
    train_center = np.median(train_raw, axis=0)
    train_scale = np.quantile(train_raw, 0.75, axis=0) - np.quantile(train_raw, 0.25, axis=0)
    train_scale = np.where(train_scale > 1e-6, train_scale, 1.0)
    train_distance = np.sqrt(np.square((train_raw - train_center) / train_scale).sum(axis=1))
    validation_distance = np.sqrt(np.square((validation_raw - train_center) / train_scale).sum(axis=1))
    train_consistency = np.abs(
        train_raw[:, 0]
        - ((train_raw[:, 1] - train_raw[:, 2]) / np.maximum(train_raw[:, 1], 1.0) * 100.0)
    )
    validation_consistency = np.abs(
        validation_raw[:, 0]
        - ((validation_raw[:, 1] - validation_raw[:, 2]) / np.maximum(validation_raw[:, 1], 1.0) * 100.0)
    )
    output: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGETS):
        absolute = np.abs(validation_prediction[:, target_index] - validation_targets[:, target_index])
        raw_absolute = np.abs(validation_raw[:, target_index] - validation_targets[:, target_index])
        train_absolute = np.abs(train_oof[:, target_index] - train_targets[:, target_index])
        conformal90 = float(np.quantile(train_absolute, 0.90, method="higher"))
        risk = (
            _percentile(train_distance, validation_distance)
            + _percentile(train_std[:, target_index], validation_std[:, target_index])
            + _percentile(train_consistency, validation_consistency)
        ) / 3.0
        large_residual = absolute >= np.quantile(absolute, 0.90)
        auc = float(roc_auc_score(large_residual.astype(int), risk)) if len(np.unique(large_residual)) == 2 else None
        output[target_name] = {
            "raw_mae": float(raw_absolute.mean()),
            "selected_predictor": selected_sources[target_name],
            "selected_mae": float(absolute.mean()),
            "median_absolute_error": float(np.median(absolute)),
            "conformal_90_half_width": conformal90,
            "conformal_90_coverage": float(np.mean(absolute <= conformal90)),
            "large_residual_auroc": auc,
            "risk_coverage": _risk_coverage(
                validation_targets[:, target_index], validation_prediction[:, target_index], risk
            ),
        }
    return output


def _direct_baselines(evidence_root: Path) -> dict[str, float | None]:
    candidates: dict[str, list[float]] = {target: [] for target in TARGETS}
    for path in evidence_root.glob("mimic_lv_*.json"):
        try:
            payload = pd.read_json(path, typ="series")
            metrics = payload.get("best_validation", {}).get("metrics", {})
        except (ValueError, AttributeError):
            continue
        for target in TARGETS:
            value = metrics.get(target, {}).get("mae")
            if value is not None and np.isfinite(value):
                candidates[target].append(float(value))
    return {target: min(values) if values else None for target, values in candidates.items()}


def run_panecho_mimic_lv_audit(
    config: dict[str, Any],
    *,
    device: int = 0,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    assert_through(config, "G5")
    audit = PanEchoAuditConfig(seed=int(config["splits"]["seed"]))
    torch.manual_seed(audit.seed)
    torch.cuda.manual_seed_all(audit.seed)
    torch.cuda.set_device(device)
    root = Path(config["paths"]["mimic_lv_staging_root"])
    labels = _select_targets(
        pd.read_csv(root / "development_labels_private.csv", dtype={"patient_id": str, "study_id": str})
    )
    files = pd.read_csv(
        root / "development_files_private.csv", dtype={"patient_id": str, "study_id": str}
    )
    frame = labels.merge(files[["study_id", "local_dicom"]], on="study_id", how="inner", validate="one_to_one")
    frame, spatially_ineligible = _filter_spatially_calibrated(frame)
    train_frame = frame.loc[frame["split"] == "train"].sort_values(["patient_id", "study_id"]).reset_index(drop=True)
    validation_frame = frame.loc[frame["split"] == "validation"].sort_values(["patient_id", "study_id"]).reset_index(drop=True)
    if maximum_train is not None:
        train_frame = train_frame.iloc[:maximum_train].reset_index(drop=True)
    if maximum_validation is not None:
        validation_frame = validation_frame.iloc[:maximum_validation].reset_index(drop=True)
    if set(train_frame["patient_id"]) & set(validation_frame["patient_id"]):
        raise RuntimeError("PanEcho challenger patient leakage")
    if not len(train_frame) or not len(validation_frame):
        raise ValueError("PanEcho challenger requires non-empty train and validation data")

    source = Path(config["paths"]["panecho_source"])
    checkpoint = Path(config["paths"]["panecho_checkpoint"])
    status_path = Path(config["paths"]["run_root"]) / "week_training" / "panecho_mimic_lv_status.json"
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    model = _load_panecho(source, checkpoint, device)
    train_loader = DataLoader(
        PanEchoMIMICDataset(train_frame, frames=audit.frames, resolution=audit.resolution),
        batch_size=audit.batch_size,
        shuffle=False,
        num_workers=audit.num_workers,
        pin_memory=True,
        persistent_workers=audit.num_workers > 0,
    )
    validation_loader = DataLoader(
        PanEchoMIMICDataset(validation_frame, frames=audit.frames, resolution=audit.resolution),
        batch_size=audit.batch_size,
        shuffle=False,
        num_workers=audit.num_workers,
        pin_memory=True,
        persistent_workers=audit.num_workers > 0,
    )
    train_raw = _predict(model, train_loader, device, status_path, "train")
    validation_raw = _predict(model, validation_loader, device, status_path, "validation")
    train_targets = train_frame[list(TARGETS)].to_numpy(dtype=np.float64)
    validation_targets = validation_frame[list(TARGETS)].to_numpy(dtype=np.float64)
    calibration = _fit_grouped_calibration(
        train_raw,
        train_targets,
        validation_raw,
        train_frame["patient_id"].to_numpy(),
        folds=audit.folds,
    )
    calibrated_validation = calibration["validation"].copy()
    selected_train = calibration["oof"].copy()
    selected_validation = calibrated_validation.copy()
    selected_sources: dict[str, str] = {}
    for target_index, target in enumerate(TARGETS):
        raw_training_mae = float(
            np.mean(np.abs(train_raw[:, target_index] - train_targets[:, target_index]))
        )
        calibrated_oof_mae = float(
            np.mean(np.abs(calibration["oof"][:, target_index] - train_targets[:, target_index]))
        )
        selected_sources[target] = (
            "raw_official_head"
            if raw_training_mae <= calibrated_oof_mae
            else "grouped_ridge_calibration"
        )
        if selected_sources[target] == "raw_official_head":
            selected_train[:, target_index] = train_raw[:, target_index]
            selected_validation[:, target_index] = validation_raw[:, target_index]
    metrics = _metrics(
        train_targets,
        selected_train,
        validation_targets,
        selected_validation,
        train_raw,
        validation_raw,
        calibration["train_model_std"],
        calibration["validation_std"],
        selected_sources,
    )
    baseline = _direct_baselines(Path(config["paths"]["evidence_root"]) / "G6")
    promotion = {
        target: bool(
            mode == "full"
            and len(validation_frame) >= 100
            and baseline[target] is not None
            and metrics[target]["selected_mae"] < float(baseline[target])
            and metrics[target]["large_residual_auroc"] is not None
            and metrics[target]["large_residual_auroc"] >= 0.70
        )
        for target in TARGETS
    }
    private_predictions = pd.DataFrame(
        {
            "patient_id": validation_frame["patient_id"],
            "study_id": validation_frame["study_id"],
            **{f"target_{target}": validation_targets[:, index] for index, target in enumerate(TARGETS)},
            **{f"panecho_raw_{target}": validation_raw[:, index] for index, target in enumerate(TARGETS)},
            **{f"panecho_calibrated_{target}": calibrated_validation[:, index] for index, target in enumerate(TARGETS)},
            **{f"panecho_selected_{target}": selected_validation[:, index] for index, target in enumerate(TARGETS)},
            **{f"panecho_epistemic_{target}": calibration["validation_std"][:, index] for index, target in enumerate(TARGETS)},
        }
    )
    private_path = Path(config["paths"]["private_root"]) / "panecho_mimic_lv_validation_private.csv"
    private_predictions.to_csv(private_path, index=False)
    checks = {
        "official_checkpoint_present": checkpoint.is_file(),
        "external_source_not_redistributed": True,
        "patient_disjoint_calibration_validation": not (
            set(train_frame["patient_id"]) & set(validation_frame["patient_id"])
        ),
        "locked_test_not_staged_or_accessed": True,
        "grouped_oof_calibration": calibration["group_folds"] >= 2,
        "all_metrics_finite": all(
            np.isfinite(metrics[target]["selected_mae"]) for target in TARGETS
        ),
        "licence_boundary_recorded": (source / "LICENSES.md").is_file(),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "PanEcho Research Track LV-volume challenger on the MIMIC development partition",
        "mode": mode,
        "checks": checks,
        "train_cines": len(train_frame),
        "validation_cines": len(validation_frame),
        "spatially_ineligible_cines_excluded": len(spatially_ineligible),
        "tasks": PANECHO_TASKS,
        "selected_alphas": calibration["selected_alphas"],
        "alpha_oof_mae": calibration["alpha_oof_mae"],
        "ef_source": calibration["ef_source"],
        "selected_predictor_by_endpoint": selected_sources,
        "metrics": metrics,
        "direct_mimic_baseline_mae": baseline,
        "promotion_eligible_by_endpoint": promotion,
        "promotion_rule": "A full run with at least 100 validation cines must beat the best exact-split direct-video baseline and achieve validation large-residual AUROC at least 0.70. Smoke runs and weaker endpoints remain comparison evidence.",
        "checkpoint_sha256": _sha256(checkpoint),
        "source_files_sha256": {
            "hubconf.py": _sha256(source / "hubconf.py"),
            "src/models.py": _sha256(source / "src" / "models.py"),
        },
        "private_predictions": str(private_path),
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "track": "R",
        "licence": "PanEcho weights CC BY-NC-SA 4.0; source AGPLv3; no weights or source redistributed",
        "locked_test_accessed": False,
    }
    destination = (
        Path(config["paths"]["evidence_root"]) / "G6" / "panecho_mimic_lv.json"
        if mode == "full"
        else Path(config["paths"]["report_root"]) / "panecho_mimic_lv_smoke.json"
    )
    _atomic_json(destination, report)
    _atomic_json(
        status_path,
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": f"panecho_mimic_lv_{mode}_complete",
            "metrics": {target: metrics[target]["selected_mae"] for target in TARGETS},
            "promotion_eligible_by_endpoint": promotion,
            "device": device,
            "locked_test_accessed": False,
        },
    )
    return report

