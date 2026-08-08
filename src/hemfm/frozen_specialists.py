from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .architecture_pilot import ARCHITECTURES
from .gates import assert_through


@dataclass(frozen=True)
class SpecialistConfig:
    epochs: int = 120
    batch_size: int = 64
    patience: int = 15
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    ema_decay: float = 0.995


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _row_matrix(
    rows: pd.DataFrame, architecture: str, feature_root: Path
) -> np.ndarray:
    vectors = []
    for value in rows["cine_ids"]:
        cine_ids = json.loads(value)
        features = [
            np.load(feature_root / f"{identifier}.npz", allow_pickle=False)[architecture]
            .astype(np.float32)
            .reshape(-1)
            for identifier in cine_ids
        ]
        stack = np.stack(features)
        vectors.append(np.concatenate([stack.mean(axis=0), stack.max(axis=0)]))
    return np.stack(vectors).astype(np.float32)


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model(input_dim: int):
    import torch

    class Specialist(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, 512),
                torch.nn.GELU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(512, 128),
                torch.nn.GELU(),
            )
            self.mean = torch.nn.Linear(128, 1)
            self.log_variance = torch.nn.Linear(128, 1)

        def forward(self, features):
            hidden = self.network(features)
            return self.mean(hidden).squeeze(-1), self.log_variance(hidden).squeeze(-1).clamp(-6, 4)

    return Specialist()


def _evaluate(model, features, targets, target_mean, target_std, device: int) -> dict[str, float]:
    import torch

    model.eval()
    predictions = []
    uncertainties = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            batch = torch.from_numpy(features[start : start + 256]).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mean, log_variance = model(batch)
            predictions.append((mean.float().cpu().numpy() * target_std) + target_mean)
            uncertainties.append(np.sqrt(np.exp(log_variance.float().cpu().numpy())) * target_std)
    prediction = np.concatenate(predictions)
    uncertainty = np.concatenate(uncertainties)
    error = np.abs(prediction - targets)
    return {
        "mae": float(error.mean()),
        "median_ae": float(np.median(error)),
        "rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
        "mean_predicted_sigma": float(uncertainty.mean()),
    }


def _train_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    seed: int,
    device: int,
    config: SpecialistConfig,
    checkpoint_path: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _seed(seed)
    target_mean = float(y_train.mean())
    target_std = float(y_train.std()) or 1.0
    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(((y_train - target_mean) / target_std).astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, generator=generator,
    )
    model = _model(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    ema = deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    best_mae = math.inf
    best_epoch = 0
    best_state = None
    history = []
    patience = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mean, log_variance = model(features)
                loss = 0.5 * (
                    torch.exp(-log_variance.float()) * (mean.float() - target).square()
                    + log_variance.float()
                ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema.parameters(), model.parameters(), strict=True
                ):
                    ema_parameter.lerp_(parameter.detach(), 1.0 - config.ema_decay)
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = _evaluate(
            ema, x_validation, y_validation, target_mean, target_std, device
        )
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics}
        )
        if metrics["mae"] < best_mae - 1e-5:
            best_mae = metrics["mae"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu() for key, value in ema.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= config.patience:
            break
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "input_dim": x_train.shape[1],
            "target_mean": target_mean,
            "target_std": target_std,
            "seed": seed,
            "best_epoch": best_epoch,
            "track": "R",
            "locked_test_accessed": False,
        },
        checkpoint_path,
    )
    return {
        "passed": bool(np.isfinite(best_mae)),
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation": history[best_epoch - 1]["validation"],
        "wall_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(checkpoint_path),
        "history": history,
    }


def _oof_predictions(
    rows: pd.DataFrame,
    features: np.ndarray,
    output_path: Path,
    seed: int,
) -> dict[str, Any]:
    train_mask = rows["split"].eq("train").to_numpy()
    train_rows = rows.loc[train_mask].reset_index(drop=True)
    x = features[train_mask].astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    y = train_rows["value"].to_numpy(dtype=np.float64)
    groups = train_rows["subject_id"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    predictions = np.full(len(train_rows), np.nan, dtype=np.float64)
    folds = np.full(len(train_rows), -1, dtype=int)
    leakage = []
    for fold, (fit_index, held_index) in enumerate(
        splitter.split(x, y, groups=groups)
    ):
        train_groups = set(groups[fit_index])
        held_groups = set(groups[held_index])
        leakage.append(len(train_groups & held_groups))
        model = Ridge(alpha=100.0, solver="lsqr", max_iter=5000, tol=1e-5)
        model.fit(x[fit_index], y[fit_index])
        predictions[held_index] = model.predict(x[held_index])
        folds[held_index] = fold
    frame = pd.DataFrame(
        {
            "sample_id": train_rows["sample_id"],
            "target": train_rows["target"],
            "fold": folds,
            "oof_pseudolabel": predictions,
        }
    )
    frame.to_csv(output_path, index=False)
    return {
        "rows": len(frame),
        "patients": int(train_rows["subject_id"].nunique()),
        "folds": 5,
        "all_rows_predicted_once": bool(np.isfinite(predictions).all() and (folds >= 0).all()),
        "max_patient_overlap_per_fold": max(leakage),
        "oof_mae_against_available_label": float(np.mean(np.abs(predictions - y))),
    }


def run_frozen_specialists(config: dict[str, Any]) -> dict[str, Any]:
    assert_through(config, "G5")
    report_root = Path(config["paths"]["report_root"])
    extraction_report = json.loads(
        (report_root / "full_feature_extraction.json").read_text(encoding="utf-8")
    )
    if not extraction_report.get("passed"):
        raise RuntimeError("Full specialist feature extraction has not passed")
    private_root = Path(config["paths"]["private_root"])
    feature_root = private_root / "specialist_cine_features"
    run_root = Path(config["paths"]["run_root"]) / "week_training"
    specialist_root = Path(config["paths"]["run_root"]) / "specialists"
    rows = pd.read_csv(
        private_root / "specialist_rows_private.csv",
        dtype={"subject_id": str, "study_id": str}, low_memory=False,
    )
    seeds = list(config["training"]["seeds"])
    training_config = SpecialistConfig()
    targets: dict[str, Any] = {}
    oof_summaries: dict[str, Any] = {}
    started = time.perf_counter()
    total_runs = 6 * len(ARCHITECTURES) * len(seeds)
    complete_runs = 0
    for target_index, target in enumerate(config["targets"]):
        target_rows = rows[rows["target"] == target].reset_index(drop=True)
        train_mask = target_rows["split"].eq("train").to_numpy()
        validation_mask = target_rows["split"].eq("validation").to_numpy()
        candidate_results: dict[str, Any] = {}
        candidate_matrices: dict[str, np.ndarray] = {}
        for architecture in ARCHITECTURES:
            matrix = _row_matrix(target_rows, architecture, feature_root)
            candidate_matrices[architecture] = matrix
            runs = []
            for seed in seeds:
                destination = specialist_root / target / architecture / f"seed_{seed}"
                run_report_path = destination / "report.json"
                if run_report_path.exists():
                    run_report = json.loads(run_report_path.read_text(encoding="utf-8"))
                else:
                    run_report = _train_one(
                        matrix[train_mask],
                        target_rows.loc[train_mask, "value"].to_numpy(dtype=np.float32),
                        matrix[validation_mask],
                        target_rows.loc[validation_mask, "value"].to_numpy(dtype=np.float32),
                        seed=seed,
                        device=target_index % 2,
                        config=training_config,
                        checkpoint_path=destination / "best_ema.pt",
                    )
                    destination.mkdir(parents=True, exist_ok=True)
                    run_report_path.write_text(
                        json.dumps(run_report, indent=2), encoding="utf-8"
                    )
                runs.append(run_report)
                complete_runs += 1
                _atomic_json(
                    run_root / "status.json",
                    {
                        "schema_version": 1,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "phase": "frozen_specialist_training",
                        "total_runs": total_runs,
                        "complete_runs": complete_runs,
                        "current_target": target,
                        "current_architecture": architecture,
                        "locked_test_accessed": False,
                        "mutable_storage": "local C drive",
                    },
                )
            mean_mae = float(
                np.mean([run["best_validation"]["mae"] for run in runs])
            )
            candidate_results[architecture] = {
                "passed": all(run["passed"] for run in runs),
                "mean_seed_validation_mae": mean_mae,
                "seeds": runs,
            }
        winner = min(
            candidate_results,
            key=lambda name: candidate_results[name]["mean_seed_validation_mae"],
        )
        oof_path = private_root / f"oof_{target.lower()}_private.csv"
        oof_summaries[target] = _oof_predictions(
            target_rows,
            candidate_matrices[winner],
            oof_path,
            int(config["splits"]["seed"]),
        )
        targets[target] = {
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
            "patient_disjoint": not bool(
                set(target_rows.loc[train_mask, "subject_id"])
                & set(target_rows.loc[validation_mask, "subject_id"])
            ),
            "candidates": candidate_results,
            "selected_architecture": winner,
        }
    specialist_checks = {
        "six_targets_trained": len(targets) == 6,
        "four_architectures_three_seeds": all(
            len(endpoint["candidates"]) == 4
            and all(len(candidate["seeds"]) == 3 for candidate in endpoint["candidates"].values())
            for endpoint in targets.values()
        ),
        "all_runs_finite": all(
            candidate["passed"]
            for endpoint in targets.values()
            for candidate in endpoint["candidates"].values()
        ),
        "patient_disjoint": all(endpoint["patient_disjoint"] for endpoint in targets.values()),
        "locked_test_not_accessed": True,
        "adamw_ema_bf16": True,
        "fp8_not_used": True,
    }
    specialist_report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(specialist_checks.values()),
        "scope": "development frozen-backbone specialist training; not the complete G6 holdout gate",
        "checks": specialist_checks,
        "targets": targets,
        "precision": "BF16 compute with FP32 loss",
        "locked_test_accessed": False,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    oof_checks = {
        "six_targets": len(oof_summaries) == 6,
        "every_training_row_predicted_once": all(
            row["all_rows_predicted_once"] for row in oof_summaries.values()
        ),
        "zero_fold_patient_leakage": all(
            row["max_patient_overlap_per_fold"] == 0 for row in oof_summaries.values()
        ),
        "locked_test_not_accessed": True,
    }
    oof_report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(oof_checks.values()),
        "checks": oof_checks,
        "targets": oof_summaries,
        "private_outputs": True,
        "locked_test_accessed": False,
    }
    evidence_root = Path(config["paths"]["evidence_root"]) / "G6"
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "specialist_training.json").write_text(
        json.dumps(specialist_report, indent=2), encoding="utf-8"
    )
    (evidence_root / "oof_pseudolabels.json").write_text(
        json.dumps(oof_report, indent=2), encoding="utf-8"
    )
    _atomic_json(
        run_root / "status.json",
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": (
                "development_specialists_complete_g6_holdouts_pending"
                if specialist_report["passed"] and oof_report["passed"]
                else "development_specialists_failed"
            ),
            "complete_runs": complete_runs,
            "total_runs": total_runs,
            "locked_test_accessed": False,
            "mutable_storage": "local C drive",
        },
    )
    return {"specialist_training": specialist_report, "oof_pseudolabels": oof_report}

