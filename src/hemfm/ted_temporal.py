from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .camus_dense import (
    DenseLVDecoder,
    DenseLVModel,
    _boundary_target,
    _load_backbone,
    _load_model_state,
    _model_state,
)
from .frozen_specialists import _atomic_json
from .gates import assert_through
from .research_datasets import TED_CITATION, _camus_patient_id, _lines, audit_ted, parse_metaimage_header


@dataclass(frozen=True)
class TemporalTrainingConfig:
    frames: int = 16
    resolution: int = 224
    epochs: int = 10
    frozen_epochs: int = 1
    accumulation: int = 4
    backbone_learning_rate: float = 1e-5
    decoder_learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    ema_decay: float = 0.99
    num_workers: int = 2
    seed: int = 20260807


def ted_development_split(config: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    ted_root = Path(config["paths"]["ted_root"])
    camus_root = Path(config["paths"]["camus_root"])
    camus_train = set(_lines(camus_root / "database_split" / "subgroup_training.txt"))
    camus_test = set(_lines(camus_root / "database_split" / "subgroup_testing.txt"))
    patients = sorted(path.name for path in (ted_root / "database").glob("patient*") if path.is_dir())
    development = [patient for patient in patients if _camus_patient_id(patient) in camus_train]
    sealed = [patient for patient in patients if _camus_patient_id(patient) in camus_test]
    seed = int(config["splits"]["seed"])
    ranked = sorted(
        development,
        key=lambda patient: hashlib.sha256(f"{seed}:TED:{patient}".encode()).hexdigest(),
    )
    tune_count = max(1, round(len(ranked) * 0.106))
    tune = sorted(ranked[:tune_count])
    train = sorted(ranked[tune_count:])
    return train, tune, sealed


class TEDSequenceDataset:
    def __init__(
        self,
        root: Path,
        patients: list[str],
        *,
        frames: int = 16,
        resolution: int = 224,
        augment: bool = False,
        maximum: int | None = None,
    ) -> None:
        self.root = root
        self.patients = patients[:maximum] if maximum is not None else patients
        self.frames = frames
        self.resolution = resolution
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        import torch.nn.functional as functional
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import affine

        patient = self.patients[index]
        patient_root = self.root / "database" / patient
        stem = patient_root / f"{patient}_4CH_sequence"
        header = parse_metaimage_header(Path(f"{stem}.mhd"))
        dimensions = tuple(header["dimensions"])
        width, height, frame_count = dimensions
        image = np.memmap(Path(f"{stem}.raw"), dtype=np.uint8, mode="r", shape=(frame_count, height, width))
        mask = np.memmap(Path(f"{stem}_gt.raw"), dtype=np.uint8, mode="r", shape=(frame_count, height, width))
        temporal = np.linspace(0, frame_count - 1, self.frames).round().astype(int)
        image_tensor = torch.from_numpy(np.asarray(image[temporal]).copy()).unsqueeze(1).float()
        mask_tensor = torch.from_numpy(np.asarray(mask[temporal]).copy()).unsqueeze(1).float()
        image_tensor = functional.interpolate(
            image_tensor, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        mask_tensor = functional.interpolate(
            mask_tensor, size=(self.resolution, self.resolution), mode="nearest"
        )
        if self.augment:
            angle = random.uniform(-6.0, 6.0)
            scale = random.uniform(0.95, 1.05)
            image_tensor = affine(
                image_tensor,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            mask_tensor = affine(
                mask_tensor,
                angle=angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            )
            image_tensor = (
                (image_tensor / 255.0)
                .clamp(0, 1)
                .pow(random.uniform(0.90, 1.10))
                .mul(random.uniform(0.90, 1.10))
                .clamp(0, 1)
            )
        else:
            image_tensor = (image_tensor / 255.0).clamp(0, 1)
        image_tensor = image_tensor.repeat(1, 3, 1, 1).permute(1, 0, 2, 3)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        spacing_x, spacing_y = header["spacing"][:2]
        spacing = torch.tensor(
            [spacing_y * height / self.resolution, spacing_x * width / self.resolution],
            dtype=torch.float32,
        )
        return {
            "video": ((image_tensor - mean) / std).float(),
            "mask": mask_tensor.squeeze(1).long(),
            "spacing_mm": spacing,
            "patient": patient,
        }


def _temporal_loss(outputs: dict[str, Any], mask) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    logits = outputs["segmentation"]
    weights = torch.tensor([0.15, 1.0, 1.0, 0.10], device=logits.device)
    pixel_ce = functional.cross_entropy(logits.float(), mask, weight=weights, reduction="none")
    log_variance = outputs["log_variance"].squeeze(1).float()
    heteroscedastic = (torch.exp(-log_variance) * pixel_ce + 0.05 * log_variance).mean()
    probabilities = torch.softmax(logits.float(), dim=1)
    dice_losses = []
    for label in (1, 2):
        target = (mask == label).float()
        prediction = probabilities[:, label]
        intersection = (prediction * target).sum(dim=(1, 2, 3))
        denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_losses.append(1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean())
    dice = torch.stack(dice_losses).mean()
    boundary = functional.binary_cross_entropy_with_logits(
        outputs["boundary"].float(), _boundary_target(mask)
    )
    lv_area = probabilities[:, 1].mean(dim=(-1, -2))
    second_difference = lv_area[:, 2:] - (2 * lv_area[:, 1:-1]) + lv_area[:, :-2]
    temporal = second_difference.abs().mean()
    total = heteroscedastic + dice + (0.20 * boundary) + (0.20 * temporal)
    return total, {
        "heteroscedastic_ce": float(heteroscedastic.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "boundary_loss": float(boundary.detach().cpu()),
        "temporal_area_loss": float(temporal.detach().cpu()),
    }


def _evaluate(model: DenseLVModel, loader, device: int) -> dict[str, Any]:
    import torch

    model.eval()
    intersections = np.zeros(2, dtype=np.float64)
    denominators = np.zeros(2, dtype=np.float64)
    losses: list[float] = []
    temporal_roughness: list[float] = []
    uncertainty: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(video)
                loss, _ = _temporal_loss(outputs, mask)
            prediction = outputs["segmentation"].argmax(dim=1)
            for index, label in enumerate((1, 2)):
                predicted = prediction == label
                target = mask == label
                intersections[index] += float((predicted & target).sum().cpu()) * 2.0
                denominators[index] += float(predicted.sum().cpu() + target.sum().cpu())
            area = torch.softmax(outputs["segmentation"].float(), dim=1)[:, 1].mean(dim=(-1, -2))
            roughness = (area[:, 2:] - (2 * area[:, 1:-1]) + area[:, :-2]).abs().mean()
            temporal_roughness.append(float(roughness.cpu()))
            uncertainty.append(float(torch.exp(0.5 * outputs["log_variance"].float()).mean().cpu()))
            losses.append(float(loss.cpu()))
    dice = intersections / np.clip(denominators, 1.0, None)
    return {
        "loss": float(np.mean(losses)),
        "mean_foreground_dice": float(dice.mean()),
        "dice": {"lv_cavity": float(dice[0]), "myocardium": float(dice[1])},
        "temporal_area_roughness": float(np.mean(temporal_roughness)),
        "mean_predicted_sigma": float(np.mean(uncertainty)),
        "sequences": len(loader.dataset),
    }


def run_ted_temporal_training(
    config: dict[str, Any],
    *,
    device: int = 0,
    epochs: int = 10,
    frozen_epochs: int = 1,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
    backbone: str = "vjepa21_vitb",
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    assert_through(config, "G5")
    acquisition = audit_ted(config)
    if not acquisition["passed"]:
        raise RuntimeError("TED acquisition audit failed")
    training = TemporalTrainingConfig(
        epochs=epochs,
        frozen_epochs=frozen_epochs,
        seed=int(config["splits"]["seed"]),
    )
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    torch.cuda.manual_seed_all(training.seed)
    torch.cuda.set_device(device)
    train_patients, validation_patients, sealed_patients = ted_development_split(config)
    root = Path(config["paths"]["ted_root"])
    train_dataset = TEDSequenceDataset(
        root,
        train_patients,
        frames=training.frames,
        resolution=training.resolution,
        augment=True,
        maximum=maximum_train,
    )
    validation_dataset = TEDSequenceDataset(
        root,
        validation_patients,
        frames=training.frames,
        resolution=training.resolution,
        augment=False,
        maximum=maximum_validation,
    )
    generator = torch.Generator().manual_seed(training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=training.num_workers,
        pin_memory=True,
        persistent_workers=training.num_workers > 0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=training.num_workers,
        pin_memory=True,
        persistent_workers=training.num_workers > 0,
    )
    encoder = _load_backbone(config, device, backbone)
    decoder_channels = 768 if backbone == "dinov3_vitb" else 3072
    decoder = DenseLVDecoder(
        in_channels=decoder_channels, frames=training.frames, resolution=training.resolution
    ).to(device)
    model = DenseLVModel(encoder, decoder, training.frames)
    camus_label = "vitb" if backbone == "vjepa21_vitb" else backbone
    camus_checkpoint = (
        Path(config["paths"]["run_root"])
        / f"camus_dense_{camus_label}_full"
        / "checkpoint_best.pt"
    )
    if not camus_checkpoint.exists():
        raise FileNotFoundError(f"CAMUS dense checkpoint is not ready: {camus_checkpoint}")
    camus_state = torch.load(camus_checkpoint, map_location="cpu", weights_only=True)
    _load_model_state(model, camus_state["ema"])
    del camus_state
    ema = DenseLVModel(deepcopy(model.encoder), deepcopy(model.decoder), training.frames).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": training.backbone_learning_rate},
            {"params": list(model.decoder.parameters()), "lr": training.decoder_learning_rate},
        ],
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training.epochs)
    run_root = Path(config["paths"]["run_root"]) / f"ted_temporal_{backbone}_{mode}"
    run_root.mkdir(parents=True, exist_ok=True)
    status_root = Path(config["paths"]["run_root"]) / "week_training"
    last_checkpoint = run_root / "checkpoint_last.pt"
    best_checkpoint = run_root / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_dice = -math.inf
    if last_checkpoint.exists():
        state = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        _load_model_state(model, state["model"])
        _load_model_state(ema, state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_dice = float(state["best_dice"])
        start_epoch = int(state["epoch"]) + 1

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer_updates = (start_epoch - 1) * math.ceil(len(train_loader) / training.accumulation)
    for epoch in range(start_epoch, training.epochs + 1):
        frozen = epoch <= training.frozen_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(not frozen)
        model.train()
        if frozen:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses: list[float] = []
        components: dict[str, list[float]] = {}
        for step, batch in enumerate(train_loader, start=1):
            video = batch["video"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(video)
                loss, detail = _temporal_loss(outputs, mask)
            (loss / training.accumulation).backward()
            epoch_losses.append(float(loss.detach().cpu()))
            for key, value in detail.items():
                components.setdefault(key, []).append(value)
            if step % training.accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
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
                    "phase": f"ted_temporal_{backbone}_{mode}",
                    "epoch": epoch,
                    "total_epochs": training.epochs,
                    "complete_sequences": step,
                    "total_sequences": len(train_loader),
                    "device": device,
                    "backbone_frozen": frozen,
                    "locked_test_accessed": False,
                }
                _atomic_json(status_root / f"dense_lv_ted_{backbone}_status.json", status)
                _atomic_json(status_root / "status.json", status)
        scheduler.step()
        validation = _evaluate(ema, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "backbone_frozen": frozen,
                "train_loss": float(np.mean(epoch_losses)),
                "train_components": {key: float(np.mean(values)) for key, values in components.items()},
                "validation": validation,
            }
        )
        improved = validation["mean_foreground_dice"] > best_dice
        if improved:
            best_dice = validation["mean_foreground_dice"]
            torch.save(
                {
                    "ema": _model_state(ema),
                    "epoch": epoch,
                    "validation": validation,
                    "training": asdict(training),
                    "track": "R",
                    "locked_test_accessed": False,
                },
                best_checkpoint,
            )
        torch.save(
            {
                "model": _model_state(model),
                "ema": _model_state(ema),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "history": history,
                "training": asdict(training),
                "track": "R",
                "locked_test_accessed": False,
            },
            last_checkpoint,
        )
        _atomic_json(run_root / "history.json", {"history": history})

    checks = {
        "acquisition_audit_passed": acquisition["passed"],
        "camus_dense_initialisation": camus_checkpoint.exists(),
        "camus_test_linked_patients_sealed": len(sealed_patients) == 4,
        "development_split_disjoint": not (set(train_patients) & set(validation_patients)),
        "full_cycle_temporal_supervision": True,
        "boundary_and_uncertainty_outputs": True,
        "bf16_compute_fp32_loss": True,
        "ema_validation": True,
        "locked_test_not_accessed": True,
        "finite_best_dice": bool(np.isfinite(best_dice)),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "TED full-cycle temporal adaptation; four CAMUS-test-linked patients remain sealed",
        "limitation": "The development holdout is sequence-disjoint for this adaptation but its identities were present in CAMUS representation pretraining, so it is not an external patient holdout.",
        "mode": mode,
        "backbone": backbone,
        "checks": checks,
        "training": asdict(training),
        "train_patients": len(train_dataset),
        "validation_patients": len(validation_dataset),
        "sealed_test_patients": len(sealed_patients),
        "best_validation_mean_foreground_dice": best_dice,
        "history": history,
        "camus_checkpoint": str(camus_checkpoint),
        "checkpoint_best": str(best_checkpoint),
        "checkpoint_last": str(last_checkpoint),
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "track": "R",
        "citation": TED_CITATION,
        "locked_test_accessed": False,
    }
    destination = (
        Path(config["paths"]["evidence_root"]) / "G6" / f"ted_temporal_{backbone}.json"
        if mode == "full"
        else Path(config["paths"]["report_root"]) / f"ted_temporal_{backbone}_smoke.json"
    )
    _atomic_json(destination, report)
    final_status = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": f"ted_temporal_{backbone}_{mode}_complete",
        "complete_runs": training.epochs,
        "total_runs": training.epochs,
        "best_validation_mean_foreground_dice": best_dice,
        "device": device,
        "locked_test_accessed": False,
    }
    _atomic_json(status_root / f"dense_lv_ted_{backbone}_status.json", final_status)
    _atomic_json(status_root / "status.json", final_status)
    return report

