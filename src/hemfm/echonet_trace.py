Exit code: 0
Wall time: 1.2 seconds
Output:
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd

from .echonet_dynamic import DEVELOPMENT_SPLITS, SOURCE_URL
from .frozen_specialists import _atomic_json
from .gates import assert_through
from .staged_final_scalar import (
    StagedScalarConfig,
    _apply_shadow,
    _compact_names,
    _compact_state,
    _load_encoder,
    _optimizer,
    _restore,
    _set_stage,
    stage_for_epoch,
)


class EchoNetTraceDataset:
    """Exact expert-traced ED/ES frames assembled as a 16-frame training clip."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        frames: int = 16,
        augment: bool,
        maximum: int | None = None,
    ) -> None:
        selected = frame[frame["trace_count"].astype(int).gt(0)].copy()
        if maximum is not None:
            selected = selected.iloc[:maximum].copy()
        self.frame = selected.reset_index(drop=True)
        self.frames = frames
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.frame.iloc[index]
        with np.load(Path(row["cache_path"]), allow_pickle=False) as cached:
            traced = cached["trace_frames"].astype(np.float32)
            masks = cached["trace_masks"].astype(np.uint8)
            trace_indices = cached["trace_indices"].astype(np.int32)
        if traced.ndim != 4 or traced.shape[1] != 3 or len(traced) != len(masks):
            raise ValueError(f"Invalid traced-frame cache for {row['FileName']}")
        if not 1 <= len(traced) <= self.frames:
            raise ValueError(f"Unsupported trace count {len(traced)} for {row['FileName']}")
        assignment = np.rint(np.linspace(0, len(traced) - 1, self.frames)).astype(int)
        video = torch.from_numpy(traced[assignment].transpose(1, 0, 2, 3)) / 255.0
        mask = torch.from_numpy(masks[assignment].astype(np.float32))
        if self.augment:
            if random.random() < 0.5:
                video = torch.flip(video, dims=(-1,))
                mask = torch.flip(mask, dims=(-1,))
            gamma = random.uniform(0.85, 1.15)
            gain = random.uniform(0.92, 1.08)
            video = (video.clamp_min(1e-4).pow(gamma) * gain).clamp(0, 1)
            video = (video + torch.randn_like(video) * random.uniform(0.0, 0.01)).clamp(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        padded_indices = np.full(self.frames, -1, dtype=np.int32)
        padded_indices[: len(trace_indices)] = trace_indices
        return {
            "video": ((video - mean) / std).float(),
            "mask": mask,
            "trace_count": len(traced),
            "trace_indices": torch.from_numpy(padded_indices),
            "file_name": str(row["FileName"]),
        }


def _normalization(channels: int):
    import torch

    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return torch.nn.GroupNorm(groups, channels)


def _reshape_tokens(tokens, frames: int):
    temporal = frames // 2
    spatial_tokens = tokens.shape[1] // temporal
    spatial = int(round(math.sqrt(spatial_tokens)))
    if temporal * spatial * spatial != tokens.shape[1]:
        raise ValueError(f"Cannot reshape {tokens.shape[1]} tokens into an EchoNet grid")
    return tokens.transpose(1, 2).reshape(
        tokens.shape[0], tokens.shape[2], temporal, spatial, spatial
    )


def _build_model(config: dict[str, Any], device: int, *, frames: int = 16, resolution: int = 224):
    import torch
    import torch.nn.functional as functional

    encoder, depth, target_modules = _load_encoder(config, device)

    class TraceModel(torch.nn.Module):
        def __init__(self, backbone) -> None:
            super().__init__()
            self.encoder = backbone
            self.frames = frames
            self.resolution = resolution
            self.head = torch.nn.ModuleDict(
                {
                    "project": torch.nn.Conv3d(1024, 128, 1),
                    "low": torch.nn.Sequential(
                        _normalization(128), torch.nn.GELU(), torch.nn.Conv3d(128, 128, 3, padding=1)
                    ),
                    "mid": torch.nn.Sequential(
                        _normalization(128), torch.nn.GELU(), torch.nn.Conv3d(128, 64, 3, padding=1)
                    ),
                    "high": torch.nn.Sequential(
                        _normalization(64), torch.nn.GELU(), torch.nn.Conv3d(64, 32, 3, padding=1)
                    ),
                    "segmentation": torch.nn.Conv3d(32, 1, 1),
                    "boundary": torch.nn.Conv3d(32, 1, 1),
                    "log_variance": torch.nn.Conv3d(32, 1, 1),
                }
            )

        def forward(self, videos):
            tokens = self.encoder(videos)
            if isinstance(tokens, list):
                tokens = tokens[-1]
            features = tokens if tokens.ndim == 5 else _reshape_tokens(tokens, self.frames)
            values = self.head["low"](self.head["project"](features.float()))
            values = functional.interpolate(
                values,
                size=(self.frames, self.resolution // 4, self.resolution // 4),
                mode="trilinear",
                align_corners=False,
            )
            values = self.head["mid"](values)
            values = functional.interpolate(
                values,
                size=(self.frames, self.resolution // 2, self.resolution // 2),
                mode="trilinear",
                align_corners=False,
            )
            values = self.head["high"](values)
            values = functional.interpolate(
                values,
                size=(self.frames, self.resolution, self.resolution),
                mode="trilinear",
                align_corners=False,
            )
            return {
                "segmentation": self.head["segmentation"](values).squeeze(1),
                "boundary": self.head["boundary"](values).squeeze(1),
                "log_variance": self.head["log_variance"](values).squeeze(1).clamp(-6, 4),
            }

    model = TraceModel(encoder).to(device)
    model.head.to(dtype=torch.float32)
    return model, depth, target_modules


def _boundary_target(mask):
    import torch.nn.functional as functional

    foreground = mask.float().flatten(0, 1).unsqueeze(1)
    dilated = functional.max_pool2d(foreground, 3, stride=1, padding=1)
    eroded = -functional.max_pool2d(-foreground, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1).reshape(mask.shape)


def _trace_loss(outputs: dict[str, Any], mask):
    import torch
    import torch.nn.functional as functional

    logits = outputs["segmentation"].float()
    target = mask.float()
    pixel = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    log_variance = outputs["log_variance"].float()
    heteroscedastic = (torch.exp(-log_variance) * pixel + 0.05 * log_variance).mean()
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    boundary = functional.binary_cross_entropy_with_logits(
        outputs["boundary"].float(), _boundary_target(target)
    )
    total = (0.65 * heteroscedastic) + (0.25 * dice) + (0.10 * boundary)
    return total, {
        "heteroscedastic_bce": float(heteroscedastic.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "boundary_loss": float(boundary.detach().cpu()),
    }


def _binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = prediction.astype(bool)
    truth = target.astype(bool)
    intersection = np.logical_and(predicted, truth).sum(dtype=np.float64)
    denominator = predicted.sum(dtype=np.float64) + truth.sum(dtype=np.float64)
    union = np.logical_or(predicted, truth).sum(dtype=np.float64)
    return {
        "dice": float((2.0 * intersection + 1.0) / (denominator + 1.0)),
        "iou": float((intersection + 1.0) / (union + 1.0)),
    }


def _unique_trace_positions(trace_count: int, frames: int) -> np.ndarray:
    assignment = np.rint(np.linspace(0, trace_count - 1, frames)).astype(int)
    return np.asarray([int(np.flatnonzero(assignment == index)[0]) for index in range(trace_count)])


def _evaluate(model, loader, device: int, *, save_path: Path | None = None) -> dict[str, Any]:
    import torch

    predictions = []
    targets = []
    sigmas = []
    keys: list[str] = []
    losses = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True).to(dtype=torch.bfloat16)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(video)
                loss, _ = _trace_loss(outputs, mask)
            probability = torch.sigmoid(outputs["segmentation"].float()).cpu().numpy()
            uncertainty = torch.exp(0.5 * outputs["log_variance"].float()).cpu().numpy()
            truth = mask.cpu().numpy().astype(np.uint8)
            counts = batch["trace_count"].cpu().numpy().astype(int)
            trace_indices = batch["trace_indices"].cpu().numpy()
            for item, count in enumerate(counts):
                positions = _unique_trace_positions(int(count), probability.shape[1])
                predictions.append((probability[item, positions] >= 0.5).astype(np.uint8))
                targets.append(truth[item, positions])
                sigmas.append(uncertainty[item, positions].mean(axis=(1, 2)))
                keys.extend(
                    f"{batch['file_name'][item]}:{int(frame_index)}"
                    for frame_index in trace_indices[item, :count]
                )
            losses.append(float(loss.cpu()))
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    sigma = np.concatenate(sigmas, axis=0).astype(np.float32)
    metrics = _binary_metrics(prediction, target)
    per_frame_dice = [
        _binary_metrics(estimate, truth)["dice"]
        for estimate, truth in zip(prediction, target, strict=True)
    ]
    poor = np.asarray(per_frame_dice) < 0.75
    failure_auroc = None
    if poor.any() and (~poor).any():
        from sklearn.metrics import roc_auc_score

        failure_auroc = float(roc_auc_score(poor.astype(int), sigma))
    metrics.update(
        {
            "loss": float(np.mean(losses)),
            "mean_predicted_sigma": float(sigma.mean()),
            "poor_mask_failure_auroc": failure_auroc,
            "traced_frames": int(len(prediction)),
            "videos": int(len(loader.dataset)),
        }
    )
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = save_path.with_name(f"{save_path.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                keys=np.asarray(keys),
                prediction=prediction,
                target=target,
                sigma=sigma,
            )
        temporary.replace(save_path)
    return metrics


def _frames(config: dict[str, Any], mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    staging_root = (
        Path(config["paths"]["local_staging_root"])
        / "datasets"
        / "research"
        / "echonet-dynamic"
    )
    manifest_path = staging_root / "development_manifest_private.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run echonet-dynamic stage before traced segmentation")
    stage_name = "echonet_dynamic_stage.json" if mode == "full" else "echonet_dynamic_stage_smoke.json"
    stage_report = json.loads(
        (Path(config["paths"]["evidence_root"]) / "G4" / stage_name).read_text(encoding="utf-8")
    )
    if mode == "full" and not stage_report.get("training_ready"):
        raise RuntimeError("Full EchoNet-Dynamic cache is not hash-verified and complete")
    frame = pd.read_csv(manifest_path)
    if set(frame["Split"]) - set(DEVELOPMENT_SPLITS):
        raise RuntimeError("EchoNet-Dynamic trace manifest contains a reserved split")
    if "trace_count" not in frame:
        raise RuntimeError("EchoNet-Dynamic cache manifest predates traced-frame indexing; rerun stage")
    train = frame[frame["Split"].eq("TRAIN") & frame["trace_count"].gt(0)].copy()
    validation = frame[frame["Split"].eq("VAL") & frame["trace_count"].gt(0)].copy()
    if train.empty or validation.empty:
        raise RuntimeError("EchoNet-Dynamic trace cache is missing TRAIN or VAL annotations")
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def _train_seed(
    config: dict[str, Any],
    *,
    seed: int,
    device: int,
    epochs: int,
    maximum_train: int | None,
    maximum_validation: int | None,
    mode: str,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    train_frame, validation_frame = _frames(config, mode)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    training = replace(
        StagedScalarConfig(),
        epochs=epochs,
        frozen_epochs=1,
        peft_epochs=max(1, min(3, epochs - 2)),
        accumulation=4,
        patience=2 if mode == "smoke" else 4,
    )
    train_dataset = EchoNetTraceDataset(
        train_frame, frames=training.frames, augment=True, maximum=maximum_train
    )
    validation_dataset = EchoNetTraceDataset(
        validation_frame, frames=training.frames, augment=False, maximum=maximum_validation
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    model, depth, target_modules = _build_model(config, device)
    compact_names = _compact_names(model, depth)
    for name, parameter in model.named_parameters():
        if name in compact_names and not name.startswith("head."):
            parameter.data = parameter.data.float()
    run_root = Path(config["paths"]["run_root"]) / "echonet_dynamic_trace" / f"seed_{seed}_{mode}"
    run_root.mkdir(parents=True, exist_ok=True)
    best_path = run_root / "checkpoint_best.pt"
    last_path = run_root / "checkpoint_last.pt"
    prediction_path = run_root / "validation_masks_private.npz"
    history: list[dict[str, Any]] = []
    best_dice = -math.inf
    best_epoch = 0
    best_state = None
    shadow = _compact_state(model, compact_names)
    start_epoch = 1
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=False)
        shadow = checkpoint["shadow"]
        best_state = checkpoint.get("best_state")
        best_dice = float(checkpoint.get("best_dice", -math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
    current_stage = None
    optimizer = None
    patience = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    status_path = Path(config["paths"]["run_root"]) / "week_training" / f"echonet_dynamic_trace_status_seed_{seed}.json"
    for epoch in range(start_epoch, training.epochs + 1):
        stage = stage_for_epoch(epoch, training)
        if stage != current_stage:
            _set_stage(model, stage, depth)
            optimizer = _optimizer(model, stage, depth, training)
            current_stage = stage
        assert optimizer is not None
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        components: dict[str, list[float]] = {}
        for step, batch in enumerate(train_loader, start=1):
            video = batch["video"].to(device, non_blocking=True).to(dtype=torch.bfloat16)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, details = _trace_loss(model(video), mask)
            (loss / training.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            for key, value in details.items():
                components.setdefault(key, []).append(value)
            if step % training.accumulation == 0 or step == len(train_loader):
                active = [parameter for parameter in model.parameters() if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(active, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name in shadow and parameter.requires_grad:
                            shadow[name] = shadow[name].to(parameter.device, dtype=parameter.dtype)
                            shadow[name].lerp_(parameter.detach(), 1.0 - training.ema_decay)
            if step % 50 == 0 or step == len(train_loader):
                _atomic_json(
                    status_path,
                    {
                        "schema_version": 1,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "phase": f"echonet_trace_{stage}",
                        "device": device,
                        "seed": seed,
                        "epoch": epoch,
                        "epochs": training.epochs,
                        "complete_sequences": step,
                        "total_sequences": len(train_dataset),
                        "best_validation_dice": None if not np.isfinite(best_dice) else best_dice,
                        "official_test_videos_accessed": False,
                        "locked_test_accessed": False,
                    },
                )
        backup = _apply_shadow(model, shadow)
        validation = _evaluate(model, validation_loader, device)
        _restore(model, backup)
        improved = validation["dice"] > best_dice + 1e-5
        if improved:
            best_dice = float(validation["dice"])
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in shadow.items()}
            patience = 0
            backup = _apply_shadow(model, shadow)
            validation = _evaluate(model, validation_loader, device, save_path=prediction_path)
            _restore(model, backup)
            torch.save(
                {
                    "model": best_state,
                    "seed": seed,
                    "epoch": epoch,
                    "metrics": validation,
                    "source": SOURCE_URL,
                    "official_test_videos_accessed": False,
                    "locked_test_accessed": False,
                },
                best_path,
            )
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "train_loss": float(np.mean(losses)),
                "train_components": {key: float(np.mean(values)) for key, values in components.items()},
                "validation": validation,
            }
        )
        torch.save(
            {
                "model": _compact_state(model, compact_names),
                "shadow": {name: value.detach().cpu() for name, value in shadow.items()},
                "best_state": best_state,
                "best_dice": best_dice,
                "best_epoch": best_epoch,
                "epoch": epoch,
                "history": history,
            },
            last_path,
        )
        if patience >= training.patience and stage == "selective_unfreeze":
            break
    report = {
        "passed": bool(np.isfinite(best_dice) and best_state is not None),
        "seed": seed,
        "device": device,
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"] if best_epoch else {},
        "epochs_completed": len(history),
        "history": history,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "validation_masks": str(prediction_path),
        "target_modules": target_modules,
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "official_test_videos_accessed": False,
        "locked_test_accessed": False,
    }
    _atomic_json(run_root / "report.json", report)
    del model, shadow, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_echonet_trace_seed(
    config: dict[str, Any],
    *,
    seed: int,
    device: int,
    epochs: int = 8,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    assert_through(config, "G5")
    configured = [int(value) for value in config["training"]["seeds"]]
    if seed not in configured:
        raise ValueError(f"Seed {seed} is not one of the prespecified seeds {configured}")
    run_root = Path(config["paths"]["run_root"]) / "echonet_dynamic_trace" / f"seed_{seed}_{mode}"
    destination = run_root / "report.json"
    if destination.exists() and (run_root / "validation_masks_private.npz").exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    return _train_seed(
        config,
        seed=seed,
        device=device,
        epochs=epochs,
        maximum_train=maximum_train,
        maximum_validation=maximum_validation,
        mode=mode,
    )


def run_echonet_trace_training(
    config: dict[str, Any],
    *,
    device: int,
    epochs: int = 8,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    seeds = (
        [int(config["training"]["seeds"][0])]
        if mode == "smoke"
        else [int(value) for value in config["training"]["seeds"]]
    )
    for seed in seeds:
        run_echonet_trace_seed(
            config,
            seed=seed,
            device=device,
            epochs=epochs,
            maximum_train=maximum_train,
            maximum_validation=maximum_validation,
            mode=mode,
        )
    return finalize_echonet_trace(config, mode=mode)


def finalize_echonet_trace(config: dict[str, Any], *, mode: str = "full") -> dict[str, Any]:
    expected = (
        [int(config["training"]["seeds"][0])]
        if mode == "smoke"
        else [int(value) for value in config["training"]["seeds"]]
    )
    root = Path(config["paths"]["run_root"]) / "echonet_dynamic_trace"
    reports = []
    arrays = []
    for seed in expected:
        destination = root / f"seed_{seed}_{mode}"
        report_path = destination / "report.json"
        prediction_path = destination / "validation_masks_private.npz"
        if not report_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"EchoNet traced segmentation seed {seed} is incomplete")
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        with np.load(prediction_path, allow_pickle=False) as cached:
            arrays.append(
                {
                    "keys": cached["keys"].astype(str),
                    "prediction": cached["prediction"].astype(np.uint8),
                    "target": cached["target"].astype(np.uint8),
                }
            )
    reference_keys = arrays[0]["keys"]
    reference_target = arrays[0]["target"]
    if any(not np.array_equal(item["keys"], reference_keys) for item in arrays[1:]):
        raise RuntimeError("EchoNet trace seeds do not cover identical validation frames")
    if any(not np.array_equal(item["target"], reference_target) for item in arrays[1:]):
        raise RuntimeError("EchoNet trace seed targets disagree")
    votes = np.stack([item["prediction"] for item in arrays], axis=0).sum(axis=0)
    ensemble = (votes >= (len(arrays) // 2 + 1)).astype(np.uint8)
    metrics = _binary_metrics(ensemble, reference_target)
    metrics["traced_frames"] = int(len(reference_target))
    checks = {
        "official_train_validation_only": True,
        "three_seed_full_run": mode != "full" or len(reports) == len(expected) == 3,
        "all_runs_finite": all(report["passed"] for report in reports),
        "frozen_peft_selective_schedule": all(
            {row["stage"] for row in report["history"]}
            >= ({"frozen", "peft"} if mode == "smoke" else {"frozen", "peft", "selective_unfreeze"})
            for report in reports
        ),
        "expert_traced_frames_only": True,
        "official_test_not_accessed": True,
        "locked_mimic_test_not_accessed": True,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "EchoNet-Dynamic expert ED/ES LV segmentation transfer challenger",
        "mode": mode,
        "checks": checks,
        "ensemble_validation": metrics,
        "seeds": reports,
        "promotion_eligible": False,
        "promotion_policy": "External A4C trace training is transfer-only until it improves the patient-disjoint MIMIC development route and passes matched failure detection.",
        "official_test_videos_accessed": False,
        "locked_test_accessed": False,
        "track": "R",
    }
    _atomic_json(
        Path(config["paths"]["evidence_root"]) / "G5" / f"echonet_dynamic_trace_{mode}.json",
        report,
    )
    return report

