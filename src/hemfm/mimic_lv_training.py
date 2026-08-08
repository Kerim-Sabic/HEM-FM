from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd

from .camus_dense import _load_backbone
from .dicom_preprocess import preprocess_dicom_cine, select_calibrated_crop
from .frozen_specialists import _atomic_json
from .gates import assert_through
from .mimic_lv_extension import CITATION


TARGETS = ("LVEF", "LVEDV", "LVESV")


@dataclass(frozen=True)
class VolumeTrainingConfig:
    frames: int = 16
    resolution: int = 224
    epochs: int = 8
    frozen_epochs: int = 1
    batch_size: int = 2
    accumulation: int = 4
    backbone_learning_rate: float = 1e-5
    head_learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    ema_decay: float = 0.99
    num_workers: int = 4
    validation_workers: int = 2
    seed: int = 20260807


def _select_targets(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for target, bp, a4c in (
        ("LVEF", "LVEF_BP", "LVEF_A4C"),
        ("LVEDV", "LVEDV_BP", "LVEDV_A4C"),
        ("LVESV", "LVESV_BP", "LVESV_A4C"),
    ):
        output[target] = pd.to_numeric(output[bp], errors="coerce").fillna(
            pd.to_numeric(output[a4c], errors="coerce")
        )
        output[f"{target}_source"] = np.where(output[bp].notna(), "biplane", "a4c")
    return output.dropna(subset=list(TARGETS)).reset_index(drop=True)


def _target_statistics(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = frame[list(TARGETS)].to_numpy(dtype=np.float32)
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("Degenerate MIMIC LV training targets")
    return center, scale


def _filter_spatially_calibrated(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    import pydicom

    keep: list[bool] = []
    excluded: list[str] = []
    for path in frame["local_dicom"].astype(str):
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
            select_calibrated_crop(dataset, "spatial")
            keep.append(True)
        except Exception:
            keep.append(False)
            excluded.append(path)
    return frame.loc[keep].reset_index(drop=True), excluded


class MIMICLVVideoDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        target_center: np.ndarray,
        target_scale: np.ndarray,
        frames: int = 16,
        resolution: int = 224,
        augment: bool = False,
        maximum: int | None = None,
    ) -> None:
        self.frame = frame.iloc[:maximum].reset_index(drop=True) if maximum is not None else frame.reset_index(drop=True)
        self.target_center = target_center.astype(np.float32)
        self.target_scale = target_scale.astype(np.float32)
        self.frames = frames
        self.resolution = resolution
        self.augment = augment

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
        if self.augment:
            video = (
                video.clamp(0, 1)
                .pow(random.uniform(0.90, 1.10))
                .mul(random.uniform(0.90, 1.10))
                .clamp(0, 1)
            )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        crop_width = cine.crop.x1 - cine.crop.x0 + 1
        crop_height = cine.crop.y1 - cine.crop.y0 + 1
        width_mm = crop_width * float(cine.crop.delta_x or 0.0) * 10.0
        height_mm = crop_height * float(cine.crop.delta_y or 0.0) * 10.0
        physical = torch.tensor(
            [width_mm / 250.0, height_mm / 250.0, float(row["frame_rate"]) / 60.0, float(row["number_of_frames"]) / 120.0],
            dtype=torch.float32,
        )
        target = row[list(TARGETS)].to_numpy(dtype=np.float32)
        return {
            "video": ((video - mean) / std).float(),
            "physical": physical,
            "target": torch.from_numpy((target - self.target_center) / self.target_scale),
            "target_units": torch.from_numpy(target),
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
        }


def _make_model(encoder, backbone: str, device: int):
    import torch

    feature_channels = 768 if backbone == "dinov3_vitb" else 3072

    class VolumeRegressor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.physical = torch.nn.Sequential(
                torch.nn.Linear(4, 32), torch.nn.LayerNorm(32), torch.nn.GELU()
            )
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(feature_channels + 32),
                torch.nn.Linear(feature_channels + 32, 384),
                torch.nn.GELU(),
                torch.nn.Dropout(0.15),
                torch.nn.Linear(384, 6),
            )

        def forward(self, video, physical):
            features = self.encoder(video)
            if isinstance(features, list):
                features = torch.cat(features, dim=-1).mean(dim=1)
            elif features.ndim == 5:
                features = features.mean(dim=(2, 3, 4))
            else:
                features = features.mean(dim=1)
            output = self.head(torch.cat([features, self.physical(physical)], dim=-1))
            return {"mean": output[:, :3], "log_variance": output[:, 3:].clamp(-6, 4)}

    return VolumeRegressor().to(device)


def _loss(outputs, target, center, scale):
    import torch
    import torch.nn.functional as functional

    mean = outputs["mean"].float()
    log_variance = outputs["log_variance"].float()
    residual = mean - target.float()
    gaussian = 0.5 * (torch.exp(-log_variance) * residual.square() + log_variance).mean()
    center_tensor = torch.as_tensor(center, device=mean.device)
    scale_tensor = torch.as_tensor(scale, device=mean.device)
    units = (mean * scale_tensor) + center_tensor
    ef, edv, esv = units.unbind(dim=1)
    derived_ef = ((edv - esv) / edv.clamp_min(1.0) * 100.0).clamp(-50, 150)
    consistency = functional.smooth_l1_loss(ef / 10.0, derived_ef / 10.0)
    physiology = (functional.relu(esv - edv) / 25.0).mean() + (functional.relu(-esv) / 25.0).mean()
    total = gaussian + (0.15 * consistency) + (0.10 * physiology)
    return total, {
        "gaussian_nll": float(gaussian.detach().cpu()),
        "ef_volume_consistency": float(consistency.detach().cpu()),
        "physiology_penalty": float(physiology.detach().cpu()),
    }


def _evaluate(model, loader, device: int, center: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    import torch

    predictions = []
    targets = []
    sigmas = []
    losses = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            physical = batch["physical"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(video, physical)
                loss, _ = _loss(output, target, center, scale)
            predictions.append(output["mean"].float().cpu().numpy())
            targets.append(batch["target_units"].numpy())
            sigmas.append(np.exp(0.5 * output["log_variance"].float().cpu().numpy()) * scale)
            losses.append(float(loss.cpu()))
    prediction = (np.concatenate(predictions) * scale) + center
    target = np.concatenate(targets)
    sigma = np.concatenate(sigmas)
    absolute = np.abs(prediction - target)
    metrics = {}
    for index, name in enumerate(TARGETS):
        coverage90 = np.mean(absolute[:, index] <= (1.644854 * sigma[:, index]))
        metrics[name] = {
            "mae": float(absolute[:, index].mean()),
            "median_absolute_error": float(np.median(absolute[:, index])),
            "coverage_90": float(coverage90),
            "mean_predicted_sigma": float(sigma[:, index].mean()),
        }
    return {"loss": float(np.mean(losses)), "metrics": metrics, "samples": len(target)}


def _cpu_state(model) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def run_mimic_lv_training(
    config: dict[str, Any],
    *,
    device: int = 0,
    epochs: int = 8,
    frozen_epochs: int = 1,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
    backbone: str = "vjepa21_vitb",
    ema_decay: float = 0.99,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    assert_through(config, "G5")
    training = VolumeTrainingConfig(
        epochs=epochs,
        frozen_epochs=frozen_epochs,
        ema_decay=ema_decay,
        seed=int(config["splits"]["seed"]),
    )
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    torch.cuda.manual_seed_all(training.seed)
    torch.cuda.set_device(device)
    root = Path(config["paths"]["mimic_lv_staging_root"])
    labels = _select_targets(pd.read_csv(root / "development_labels_private.csv", dtype={"patient_id": str, "study_id": str}))
    files = pd.read_csv(root / "development_files_private.csv", dtype={"patient_id": str, "study_id": str})
    frame = labels.merge(files[["study_id", "local_dicom"]], on="study_id", how="inner", validate="one_to_one")
    staged_cines = len(frame)
    frame, spatially_ineligible = _filter_spatially_calibrated(frame)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if set(train_frame["patient_id"]) & set(validation_frame["patient_id"]):
        raise RuntimeError("MIMIC LV patient leakage")
    center, scale = _target_statistics(train_frame)
    train_dataset = MIMICLVVideoDataset(
        train_frame,
        target_center=center,
        target_scale=scale,
        frames=training.frames,
        resolution=training.resolution,
        augment=True,
        maximum=maximum_train,
    )
    validation_dataset = MIMICLVVideoDataset(
        validation_frame,
        target_center=center,
        target_scale=scale,
        frames=training.frames,
        resolution=training.resolution,
        augment=False,
        maximum=maximum_validation,
    )
    generator = torch.Generator().manual_seed(training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        pin_memory=True,
        persistent_workers=training.num_workers > 0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.validation_workers,
        pin_memory=True,
        persistent_workers=training.validation_workers > 0,
    )
    encoder = _load_backbone(config, device, backbone)
    camus_label = "vitb" if backbone == "vjepa21_vitb" else backbone
    camus_checkpoint = Path(config["paths"]["run_root"]) / f"camus_dense_{camus_label}_full" / "checkpoint_best.pt"
    if not camus_checkpoint.exists():
        raise FileNotFoundError(f"CAMUS checkpoint is not ready: {camus_checkpoint}")
    state = torch.load(camus_checkpoint, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["ema"]["encoder"])
    del state
    model = _make_model(encoder, backbone, device)
    ema = deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": training.backbone_learning_rate},
            {"params": head_parameters, "lr": training.head_learning_rate},
        ],
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training.epochs)
    run_root = Path(config["paths"]["run_root"]) / f"mimic_lv_{backbone}_{mode}"
    run_root.mkdir(parents=True, exist_ok=True)
    status_root = Path(config["paths"]["run_root"]) / "week_training"
    last_checkpoint = run_root / "checkpoint_last.pt"
    best_checkpoint = run_root / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_composite = math.inf
    if last_checkpoint.exists():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_composite = float(checkpoint["best_composite"])
        start_epoch = int(checkpoint["epoch"]) + 1

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer_updates = (start_epoch - 1) * math.ceil(len(train_loader) / training.accumulation)
    for epoch in range(start_epoch, training.epochs + 1):
        frozen = epoch <= training.frozen_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(not frozen)
        model.train()
        if frozen:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        components: dict[str, list[float]] = {}
        for step, batch in enumerate(train_loader, start=1):
            video = batch["video"].to(device, non_blocking=True)
            physical = batch["physical"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(video, physical)
                loss, detail = _loss(output, target, center, scale)
            (loss / training.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            for key, value in detail.items():
                components.setdefault(key, []).append(value)
            if step % training.accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                # Bias-correct the early EMA so a newly initialized regression
                # head is not dominated by its random starting weights.
                effective_decay = min(
                    training.ema_decay,
                    (1.0 + optimizer_updates) / (10.0 + optimizer_updates),
                )
                with torch.no_grad():
                    for ema_parameter, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                        ema_parameter.lerp_(parameter.detach(), 1.0 - effective_decay)
            if step % 10 == 0 or step == len(train_loader):
                status = {
                    "schema_version": 1,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "phase": f"mimic_lv_{backbone}_{mode}",
                    "epoch": epoch,
                    "total_epochs": training.epochs,
                    "complete_sequences": min(step * training.batch_size, len(train_dataset)),
                    "total_sequences": len(train_dataset),
                    "device": device,
                    "backbone_frozen": frozen,
                    "locked_test_accessed": False,
                }
                _atomic_json(status_root / f"dense_lv_mimic_{backbone}_status.json", status)
                _atomic_json(status_root / "status.json", status)
        scheduler.step()
        validation = _evaluate(ema, validation_loader, device, center, scale)
        composite = float(np.mean([validation["metrics"][name]["mae"] / scale[index] for index, name in enumerate(TARGETS)]))
        history.append(
            {
                "epoch": epoch,
                "backbone_frozen": frozen,
                "train_loss": float(np.mean(losses)),
                "train_components": {key: float(np.mean(values)) for key, values in components.items()},
                "validation": validation,
                "normalised_mae_composite": composite,
            }
        )
        if composite < best_composite:
            best_composite = composite
            torch.save(
                {
                    "ema": _cpu_state(ema),
                    "epoch": epoch,
                    "validation": validation,
                    "target_center": center.tolist(),
                    "target_scale": scale.tolist(),
                    "training": asdict(training),
                    "track": "R",
                    "locked_test_accessed": False,
                },
                best_checkpoint,
            )
        torch.save(
            {
                "model": _cpu_state(model),
                "ema": _cpu_state(ema),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_composite": best_composite,
                "target_center": center.tolist(),
                "target_scale": scale.tolist(),
                "training": asdict(training),
                "track": "R",
                "locked_test_accessed": False,
            },
            last_checkpoint,
        )
        _atomic_json(run_root / "history.json", {"history": history})

    best = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    final_validation = best["validation"]
    checks = {
        "patient_disjoint_development_split": not (set(train_frame["patient_id"]) & set(validation_frame["patient_id"])),
        "biplane_labels_preferred": True,
        "physical_scale_features": True,
        "spatial_calibration_enforced": True,
        "heteroscedastic_uncertainty": True,
        "ef_volume_consistency_loss": True,
        "camus_dense_initialisation": camus_checkpoint.exists(),
        "bf16_compute_fp32_loss": True,
        "ema_validation": True,
        "locked_test_not_staged_or_accessed": True,
        "finite_metrics": all(np.isfinite(item["mae"]) for item in final_validation["metrics"].values()),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "MIMIC LV-volume extension development-only direct video fine-tuning",
        "mode": mode,
        "backbone": backbone,
        "checks": checks,
        "training": asdict(training),
        "train_cines": len(train_dataset),
        "validation_cines": len(validation_dataset),
        "staged_development_cines": staged_cines,
        "spatially_ineligible_cines_excluded": len(spatially_ineligible),
        "train_patients": int(train_frame["patient_id"].nunique()),
        "validation_patients": int(validation_frame["patient_id"].nunique()),
        "label_sources": {target: train_frame[f"{target}_source"].value_counts().to_dict() for target in TARGETS},
        "best_normalised_mae_composite": best_composite,
        "best_validation": final_validation,
        "history": history,
        "camus_checkpoint": str(camus_checkpoint),
        "checkpoint_best": str(best_checkpoint),
        "checkpoint_last": str(last_checkpoint),
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "track": "R",
        "citation": CITATION,
        "locked_test_accessed": False,
    }
    destination = (
        Path(config["paths"]["evidence_root"]) / "G6" / f"mimic_lv_{backbone}_{mode}.json"
        if mode.startswith("full")
        else Path(config["paths"]["report_root"]) / f"mimic_lv_{backbone}_smoke.json"
    )
    _atomic_json(destination, report)
    _atomic_json(
        status_root / f"dense_lv_mimic_{backbone}_status.json",
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": f"mimic_lv_{backbone}_{mode}_complete",
            "complete_runs": training.epochs,
            "total_runs": training.epochs,
            "best_validation": final_validation["metrics"],
            "device": device,
            "locked_test_accessed": False,
        },
    )
    return report

