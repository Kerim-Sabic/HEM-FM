from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import random
import time
from typing import Any

import numpy as np
import pandas as pd

from .architecture_pilot import ARCHITECTURES
from .frozen_specialists import _atomic_json, _row_matrix
from .gates import assert_through


@dataclass(frozen=True)
class FusionConfig:
    epochs: int = 180
    batch_size: int = 64
    patience: int = 25
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    ema_decay: float = 0.995
    stream_width: int = 192
    fusion_width: int = 384


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fusion_matrix(
    rows: pd.DataFrame, feature_root: Path
) -> tuple[np.ndarray, tuple[int, ...]]:
    matrices = [_row_matrix(rows, name, feature_root) for name in ARCHITECTURES]
    dimensions = tuple(int(matrix.shape[1]) for matrix in matrices)
    return np.concatenate(matrices, axis=1).astype(np.float32), dimensions


def _fusion_model(
    stream_dimensions: tuple[int, ...],
    *,
    stream_width: int = 192,
    fusion_width: int = 384,
):
    import torch

    if len(stream_dimensions) < 2 or any(dimension <= 0 for dimension in stream_dimensions):
        raise ValueError("Fusion requires at least two non-empty backbone streams")

    class GatedFusionSpecialist(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.streams = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.LayerNorm(dimension),
                        torch.nn.Linear(dimension, stream_width),
                        torch.nn.GELU(),
                        torch.nn.Dropout(0.10),
                    )
                    for dimension in stream_dimensions
                ]
            )
            concatenated_width = stream_width * len(stream_dimensions)
            self.router = torch.nn.Sequential(
                torch.nn.LayerNorm(concatenated_width),
                torch.nn.Linear(concatenated_width, stream_width),
                torch.nn.GELU(),
                torch.nn.Linear(stream_width, len(stream_dimensions)),
            )
            self.fusion = torch.nn.Sequential(
                torch.nn.LayerNorm(concatenated_width + stream_width),
                torch.nn.Linear(concatenated_width + stream_width, fusion_width),
                torch.nn.GELU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(fusion_width, stream_width),
                torch.nn.GELU(),
            )
            self.mean = torch.nn.Linear(stream_width, 1)
            self.log_variance = torch.nn.Linear(stream_width, 1)

        def forward(self, features, *, return_gates: bool = False):
            chunks = torch.split(features, stream_dimensions, dim=-1)
            streams = torch.stack(
                [encoder(chunk) for encoder, chunk in zip(self.streams, chunks, strict=True)],
                dim=1,
            )
            concatenated = streams.flatten(1)
            gates = torch.softmax(self.router(concatenated), dim=-1)
            pooled = torch.sum(streams * gates.unsqueeze(-1), dim=1)
            hidden = self.fusion(torch.cat([concatenated, pooled], dim=-1))
            outputs = (
                self.mean(hidden).squeeze(-1),
                self.log_variance(hidden).squeeze(-1).clamp(-6, 4),
            )
            return (*outputs, gates) if return_gates else outputs

    return GatedFusionSpecialist()


def _evaluate(
    model,
    features: np.ndarray,
    targets: np.ndarray,
    target_mean: float,
    target_std: float,
    device: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    predictions = []
    uncertainties = []
    gate_batches = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            batch = torch.from_numpy(features[start : start + 256]).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mean, log_variance, gates = model(batch, return_gates=True)
            predictions.append((mean.float().cpu().numpy() * target_std) + target_mean)
            uncertainties.append(
                np.sqrt(np.exp(log_variance.float().cpu().numpy())) * target_std
            )
            gate_batches.append(gates.float().cpu().numpy())
    prediction = np.concatenate(predictions)
    uncertainty = np.concatenate(uncertainties)
    gates = np.concatenate(gate_batches)
    error = np.abs(prediction - targets)
    return {
        "mae": float(error.mean()),
        "median_ae": float(np.median(error)),
        "rmse": float(np.sqrt(np.mean((prediction - targets) ** 2))),
        "mean_predicted_sigma": float(uncertainty.mean()),
        "mean_stream_gates": {
            name: float(value) for name, value in zip(ARCHITECTURES, gates.mean(axis=0), strict=True)
        },
    }


def _train_seed(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    stream_dimensions: tuple[int, ...],
    *,
    seed: int,
    device: int,
    config: FusionConfig,
    checkpoint_path: Path,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _seed_everything(seed)
    torch.cuda.set_device(device)
    target_mean = float(y_train.mean())
    target_std = float(y_train.std()) or 1.0
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(((y_train - target_mean) / target_std).astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = _fusion_model(
        stream_dimensions,
        stream_width=config.stream_width,
        fusion_width=config.fusion_width,
    ).to(device)
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
    best_metrics: dict[str, Any] = {}
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
            best_metrics = metrics
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
            "stream_architectures": list(ARCHITECTURES),
            "stream_dimensions": list(stream_dimensions),
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
        "best_validation": best_metrics,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(checkpoint_path),
        "history": history,
    }


def train_fusion_target(
    config: dict[str, Any], target: str, device: int
) -> dict[str, Any]:
    assert_through(config, "G5")
    if target not in config["targets"]:
        raise ValueError(f"Unknown target: {target}")
    private_root = Path(config["paths"]["private_root"])
    feature_root = private_root / "specialist_cine_features"
    run_root = Path(config["paths"]["run_root"]) / "week_training"
    fusion_root = Path(config["paths"]["run_root"]) / "fusion_specialists"
    target_root = fusion_root / target
    target_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / f"fusion_{target}_status.json"
    rows = pd.read_csv(
        private_root / "specialist_rows_private.csv",
        dtype={"subject_id": str, "study_id": str},
        low_memory=False,
    )
    rows = rows[rows["target"] == target].reset_index(drop=True)
    train_mask = rows["split"].eq("train").to_numpy()
    validation_mask = rows["split"].eq("validation").to_numpy()
    if not train_mask.any() or not validation_mask.any():
        raise RuntimeError(f"{target} is missing a development split")
    if set(rows.loc[train_mask, "subject_id"]) & set(
        rows.loc[validation_mask, "subject_id"]
    ):
        raise RuntimeError(f"{target} train/validation patient leakage detected")

    _atomic_json(
        status_path,
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "loading_multibackbone_features",
            "target": target,
            "device": device,
            "complete_runs": 0,
            "total_runs": len(config["training"]["seeds"]),
        },
    )
    matrix, stream_dimensions = _fusion_matrix(rows, feature_root)
    training = FusionConfig()
    runs = []
    for index, seed in enumerate(config["training"]["seeds"], start=1):
        destination = target_root / f"seed_{seed}"
        report_path = destination / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = _train_seed(
                matrix[train_mask],
                rows.loc[train_mask, "value"].to_numpy(dtype=np.float32),
                matrix[validation_mask],
                rows.loc[validation_mask, "value"].to_numpy(dtype=np.float32),
                stream_dimensions,
                seed=int(seed),
                device=device,
                config=training,
                checkpoint_path=destination / "best_ema.pt",
            )
            destination.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        runs.append(report)
        _atomic_json(
            status_path,
            {
                "schema_version": 1,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "multibackbone_fusion_training",
                "target": target,
                "device": device,
                "complete_runs": index,
                "total_runs": len(config["training"]["seeds"]),
                "locked_test_accessed": False,
            },
        )

    baseline = json.loads(
        (Path(config["paths"]["evidence_root"]) / "G6" / "specialist_training.json").read_text(
            encoding="utf-8"
        )
    )["targets"][target]
    baseline_mae = min(
        candidate["mean_seed_validation_mae"]
        for candidate in baseline["candidates"].values()
    )
    fusion_mae = float(np.mean([run["best_validation"]["mae"] for run in runs]))
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(run["passed"] for run in runs),
        "scope": "development cached-feature Arm E fusion study; not a G6 holdout",
        "target": target,
        "device": device,
        "architectures": list(ARCHITECTURES),
        "stream_dimensions": list(stream_dimensions),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "patient_disjoint": True,
        "locked_test_accessed": False,
        "baseline_best_mean_seed_validation_mae": float(baseline_mae),
        "fusion_mean_seed_validation_mae": fusion_mae,
        "mae_change_vs_best_single": float(fusion_mae - baseline_mae),
        "promote_over_frozen_single": bool(fusion_mae < baseline_mae),
        "seeds": runs,
    }
    (target_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    final_status = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "multibackbone_fusion_target_complete",
        "target": target,
        "device": device,
        "complete_runs": len(runs),
        "total_runs": len(runs),
        "locked_test_accessed": False,
    }
    _atomic_json(status_path, final_status)
    return report


def _fusion_worker(
    config: dict[str, Any], targets: list[str], device: int, results
) -> None:
    completed = []
    try:
        for target in targets:
            train_fusion_target(config, target, device)
            completed.append(target)
        results.put({"ok": True, "device": device, "targets": completed})
    except Exception as exc:
        results.put(
            {
                "ok": False,
                "device": device,
                "targets": completed,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise


def _completed_fusion_runs(run_root: Path) -> int:
    completed = 0
    for path in run_root.glob("fusion_*_status.json"):
        try:
            completed += int(json.loads(path.read_text(encoding="utf-8"))["complete_runs"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            continue
    return completed


def run_multibackbone_fusion(config: dict[str, Any]) -> dict[str, Any]:
    assert_through(config, "G5")
    targets = list(config["targets"])
    if len(targets) != 6:
        raise RuntimeError("The protocol requires exactly six endpoint specialists")
    run_root = Path(config["paths"]["run_root"]) / "week_training"
    total_runs = len(targets) * len(config["training"]["seeds"])
    assignments = [targets[::2], targets[1::2]]
    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(
            target=_fusion_worker,
            args=(config, assignment, device, results),
            name=f"hemfm-fusion-gpu-{device}",
        )
        for device, assignment in enumerate(assignments)
    ]
    for worker in workers:
        worker.start()
    while any(worker.is_alive() for worker in workers):
        _atomic_json(
            run_root / "status.json",
            {
                "schema_version": 1,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "multibackbone_fusion_training",
                "complete_runs": _completed_fusion_runs(run_root),
                "total_runs": total_runs,
                "gpu_workers": 2,
                "locked_test_accessed": False,
                "mutable_storage": "local C drive",
            },
        )
        time.sleep(2)
    for worker in workers:
        worker.join()
    messages = []
    while True:
        try:
            messages.append(results.get_nowait())
        except queue.Empty:
            break
    failures = [message for message in messages if not message.get("ok")]
    failures.extend(
        {"device": index, "error": f"worker exit code {worker.exitcode}"}
        for index, worker in enumerate(workers)
        if worker.exitcode != 0
    )
    if failures:
        raise RuntimeError(f"Fusion worker failure: {failures}")

    fusion_root = Path(config["paths"]["run_root"]) / "fusion_specialists"
    reports = {
        target: json.loads((fusion_root / target / "report.json").read_text(encoding="utf-8"))
        for target in targets
    }
    checks = {
        "six_targets": len(reports) == 6,
        "three_seeds_each": all(len(report["seeds"]) == 3 for report in reports.values()),
        "patient_disjoint": all(report["patient_disjoint"] for report in reports.values()),
        "all_runs_finite": all(report["passed"] for report in reports.values()),
        "both_gpus_used": {report["device"] for report in reports.values()} == {0, 1},
        "locked_test_not_accessed": all(
            not report["locked_test_accessed"] for report in reports.values()
        ),
    }
    aggregate = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "development cached-feature Arm E fusion study; not a G6 holdout",
        "checks": checks,
        "promoted_targets": [
            target for target, report in reports.items() if report["promote_over_frozen_single"]
        ],
        "targets": reports,
        "locked_test_accessed": False,
    }
    evidence_path = Path(config["paths"]["evidence_root"]) / "G6" / "multibackbone_fusion.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    _atomic_json(
        run_root / "status.json",
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "multibackbone_fusion_complete_g6_holdouts_pending",
            "complete_runs": total_runs,
            "total_runs": total_runs,
            "gpu_workers": 2,
            "locked_test_accessed": False,
            "mutable_storage": "local C drive",
        },
    )
    return aggregate

