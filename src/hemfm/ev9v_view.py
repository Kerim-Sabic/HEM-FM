from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd

from .camus_dense import _load_backbone
from .extended_dataset_audit import _exam_key
from .frozen_specialists import _atomic_json
from .gates import assert_through


VIEWS = ("A4C", "A5C", "PASA", "PLHLA", "PMASA", "PMPALA", "PMVLSA", "PPMLSA", "SC4C")


@dataclass(frozen=True)
class ViewTrainingConfig:
    frames: int = 8
    resolution: int = 224
    epochs: int = 8
    frozen_epochs: int = 1
    batch_size: int = 4
    accumulation: int = 4
    backbone_learning_rate: float = 1e-5
    head_learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    ema_decay: float = 0.99
    label_smoothing: float = 0.05
    num_workers: int = 4
    seed: int = 20260807


def _sample_indices(frame_count: int, requested: int, augment: bool, rng: random.Random) -> np.ndarray:
    if frame_count < 1 or requested < 1:
        raise ValueError("frame counts must be positive")
    if not augment:
        return np.rint(np.linspace(0, frame_count - 1, requested)).astype(int)
    edges = np.linspace(0, frame_count, requested + 1)
    selected = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        low = min(frame_count - 1, int(math.floor(left)))
        high = min(frame_count - 1, max(low, int(math.ceil(right)) - 1))
        selected.append(rng.randint(low, high))
    return np.asarray(selected, dtype=int)


def _decode_video(path: str | Path, frames: int, augment: bool) -> np.ndarray:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    if not decoded:
        raise ValueError(f"No frames decoded from {path}")
    indices = _sample_indices(len(decoded), frames, augment, random)
    return np.stack([decoded[int(index)] for index in indices])


class EV9VViewDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        frames: int = 8,
        resolution: int = 224,
        augment: bool = False,
        maximum: int | None = None,
    ) -> None:
        self.frame = frame.iloc[:maximum].reset_index(drop=True) if maximum is not None else frame.reset_index(drop=True)
        self.frames = frames
        self.resolution = resolution
        self.augment = augment
        self.label_to_index = {label: index for index, label in enumerate(VIEWS)}
        unknown = sorted(set(self.frame["view"]) - set(VIEWS))
        if unknown:
            raise ValueError(f"Unknown EV9V views: {unknown}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        import torch.nn.functional as functional

        row = self.frame.iloc[index]
        array = _decode_video(row["local_path"], self.frames, self.augment)
        video = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float() / 255.0
        video = functional.interpolate(
            video, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        if self.augment:
            video = video.clamp(0, 1).pow(random.uniform(0.90, 1.10)).mul(
                random.uniform(0.90, 1.10)
            )
            video = (video + torch.randn_like(video) * random.uniform(0.0, 0.012)).clamp(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return {
            "video": ((video - mean) / std).permute(1, 0, 2, 3).float(),
            "label": self.label_to_index[str(row["view"])],
            "exam": _exam_key(str(row["video_name"])),
            "name": str(row["video_name"]),
        }


def _make_model(encoder, device: int):
    import torch

    class ViewRouter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(768),
                torch.nn.Linear(768, 384),
                torch.nn.GELU(),
                torch.nn.Dropout(0.20),
                torch.nn.Linear(384, len(VIEWS)),
            )

        def forward(self, video):
            features = self.encoder(video)
            if isinstance(features, list):
                features = torch.cat(features, dim=-1).mean(dim=1)
            elif features.ndim == 5:
                features = features.mean(dim=(2, 3, 4))
            else:
                features = features.mean(dim=1)
            return self.head(features)

    return ViewRouter().to(device)


def _classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    ece = 0.0
    for left in np.linspace(0, 1, 11)[:-1]:
        right = left + 0.1
        selected = (confidence > left) & (confidence <= right)
        if selected.any():
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    matrix = confusion_matrix(targets, predictions, labels=np.arange(len(VIEWS)))
    recalls = np.divide(
        np.diag(matrix), matrix.sum(axis=1), out=np.zeros(len(VIEWS), dtype=float), where=matrix.sum(axis=1) > 0
    )
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "expected_calibration_error_10_bin": ece,
        "per_view_recall": {view: float(recalls[index]) for index, view in enumerate(VIEWS)},
        "confusion_matrix": matrix.tolist(),
    }


def _evaluate(model, loader, device: int) -> dict[str, Any]:
    import torch

    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(video)
            loss = torch.nn.functional.cross_entropy(output.float(), target)
            logits.append(output.float().cpu().numpy())
            targets.append(target.cpu().numpy())
            losses.append(float(loss.cpu()))
    all_logits = np.concatenate(logits)
    all_targets = np.concatenate(targets)
    return {
        "loss": float(np.mean(losses)),
        "metrics": _classification_metrics(all_logits, all_targets),
        "videos": len(all_targets),
    }


def _cpu_state(model) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def run_ev9v_view_training(
    config: dict[str, Any],
    *,
    device: int = 0,
    epochs: int = 8,
    frozen_epochs: int = 1,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    assert_through(config, "G5")
    training = ViewTrainingConfig(
        epochs=epochs, frozen_epochs=frozen_epochs, seed=int(config["splits"]["seed"])
    )
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    torch.cuda.manual_seed_all(training.seed)
    torch.cuda.set_device(device)
    root = Path(config["paths"]["ev9v_root"])
    manifest = pd.read_csv(root / "development_manifest.csv")
    train_frame = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    validation_frame = manifest.loc[manifest["split"] == "validation"].reset_index(drop=True)
    train_exams = {_exam_key(value) for value in train_frame["video_name"]}
    validation_exams = {_exam_key(value) for value in validation_frame["video_name"]}
    if train_exams & validation_exams:
        raise RuntimeError("EV9V exam-proxy leakage")
    train_dataset = EV9VViewDataset(
        train_frame, frames=training.frames, resolution=training.resolution, augment=True, maximum=maximum_train
    )
    validation_dataset = EV9VViewDataset(
        validation_frame, frames=training.frames, resolution=training.resolution, augment=False, maximum=maximum_validation
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
        num_workers=training.num_workers,
        pin_memory=True,
        persistent_workers=training.num_workers > 0,
    )
    counts = Counter(train_frame["view"])
    class_weights = np.asarray([1.0 / math.sqrt(counts[view]) for view in VIEWS], dtype=np.float32)
    class_weights /= class_weights.mean()
    class_weights_tensor = torch.from_numpy(class_weights).to(device)
    encoder = _load_backbone(config, device, "dinov3_vitb")
    camus_checkpoint = Path(config["paths"]["run_root"]) / "camus_dense_dinov3_vitb_full" / "checkpoint_best.pt"
    state = torch.load(camus_checkpoint, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["ema"]["encoder"])
    del state
    model = _make_model(encoder, device)
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
    run_root = Path(config["paths"]["run_root"]) / f"ev9v_view_dinov3_{mode}"
    run_root.mkdir(parents=True, exist_ok=True)
    status_root = Path(config["paths"]["run_root"]) / "week_training"
    best_checkpoint = run_root / "checkpoint_best.pt"
    last_checkpoint = run_root / "checkpoint_last.pt"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    start_epoch = 1
    if last_checkpoint.exists():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        best_score = float(checkpoint["best_score"])
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
        for step, batch in enumerate(train_loader, start=1):
            video = batch["video"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(video)
                loss = torch.nn.functional.cross_entropy(
                    logits.float(), target, weight=class_weights_tensor, label_smoothing=training.label_smoothing
                )
            (loss / training.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if step % training.accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                effective_decay = min(training.ema_decay, (1.0 + optimizer_updates) / (10.0 + optimizer_updates))
                with torch.no_grad():
                    for ema_parameter, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                        ema_parameter.lerp_(parameter.detach(), 1.0 - effective_decay)
            if step % 20 == 0 or step == len(train_loader):
                status = {
                    "schema_version": 1,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "phase": f"ev9v_view_dinov3_{mode}",
                    "epoch": epoch,
                    "total_epochs": training.epochs,
                    "complete_sequences": min(step * training.batch_size, len(train_dataset)),
                    "total_sequences": len(train_dataset),
                    "device": device,
                    "backbone_frozen": frozen,
                    "locked_test_accessed": False,
                }
                _atomic_json(status_root / "ev9v_view_dinov3_status.json", status)
                _atomic_json(status_root / "status.json", status)
        scheduler.step()
        validation = _evaluate(ema, validation_loader, device)
        score = float(validation["metrics"]["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "backbone_frozen": frozen,
                "train_loss": float(np.mean(losses)),
                "validation": validation,
            }
        )
        if score > best_score:
            best_score = score
            torch.save(
                {"ema": _cpu_state(ema), "epoch": epoch, "validation": validation, "training": asdict(training), "views": VIEWS, "track": "R", "locked_test_accessed": False},
                best_checkpoint,
            )
        torch.save(
            {"model": _cpu_state(model), "ema": _cpu_state(ema), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history, "best_score": best_score, "training": asdict(training), "views": VIEWS, "track": "R", "locked_test_accessed": False},
            last_checkpoint,
        )
        _atomic_json(run_root / "history.json", {"history": history})
    best = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    validation = best["validation"]
    metrics = validation["metrics"]
    promotion_eligible = metrics["balanced_accuracy"] >= 0.75 and metrics["macro_f1"] >= 0.75
    checks = {
        "official_train_validation_only": True,
        "exam_proxy_disjoint": not (train_exams & validation_exams),
        "test_labels_not_downloaded": not (root / "test_labeled.txt").exists(),
        "all_nine_views_present": set(train_frame["view"]) == set(VIEWS),
        "class_imbalance_weighted": True,
        "camus_dense_initialisation": camus_checkpoint.exists(),
        "ema_validation": True,
        "finite_metrics": all(np.isfinite(metrics[key]) for key in ("accuracy", "balanced_accuracy", "macro_f1", "expected_calibration_error_10_bin")),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "EV9V development-only nine-view routing",
        "mode": mode,
        "checks": checks,
        "promotion_eligible": promotion_eligible,
        "promotion_rule": "balanced accuracy and macro-F1 must both be at least 0.75; scalar heads are never updated from EV9V labels",
        "training": asdict(training),
        "train_videos": len(train_dataset),
        "validation_videos": len(validation_dataset),
        "train_exams": len(train_exams),
        "validation_exams": len(validation_exams),
        "class_counts": dict(sorted(counts.items())),
        "best_validation": validation,
        "history": history,
        "checkpoint_best": str(best_checkpoint),
        "checkpoint_last": str(last_checkpoint),
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "track": "R",
        "licence": "CC BY 4.0",
        "locked_test_accessed": False,
    }
    destination = Path(config["paths"]["evidence_root"]) / "G6" / "ev9v_view_dinov3.json" if mode == "full" else Path(config["paths"]["report_root"]) / "ev9v_view_dinov3_smoke.json"
    _atomic_json(destination, report)
    _atomic_json(
        status_root / "ev9v_view_dinov3_status.json",
        {"schema_version": 1, "updated_utc": datetime.now(timezone.utc).isoformat(), "phase": f"ev9v_view_dinov3_{mode}_complete", "complete_runs": training.epochs, "total_runs": training.epochs, "best_validation": metrics, "promotion_eligible": promotion_eligible, "device": device, "locked_test_accessed": False},
    )
    return report

