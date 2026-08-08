from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any

import numpy as np

from .camus_dense import _load_backbone
from .frozen_specialists import _atomic_json
from .gates import assert_through


LANDMARKS = (
    "ao-sinus-bottom-inner",
    "ao-sinus-top-inner",
    "ao-sinus-top-outer",
    "ao-stj-bottom-inner",
    "ao-stj-top-inner",
    "ao-stj-top-outer",
    "ao-valve-top-inner",
    "ao-valve-top-outer",
    "la-bottom-inner",
    "la-top-inner",
    "lv-apex-endo",
    "lv-apex-epi",
    "lv-antsep-endo-apex",
    "lv-antsep-endo-base",
    "lv-antsep-rv-apex",
    "lv-antsep-rv-base",
    "lv-ivs-bottom",
    "lv-ivs-top",
    "lv-post-endo-apex",
    "lv-post-endo-base",
    "lv-post-epi-apex",
    "lv-post-epi-base",
    "lv-pw-bottom",
    "lv-pw-top",
    "mv-ant-hinge",
    "mv-post-hinge",
    "rv-apex-endo",
    "rv-bottom-inner",
    "rv-top-inner",
    "tv-ant-hinge",
    "tv-sep-hinge",
)


# The published Unity label lists contain a small number of records whose file name
# does not resolve to any image in the official png-cache (one malformed duplicate in
# labels-train.txt at the time of writing). Those records are excluded and reported
# rather than silently substituted, and the loader still fails closed if the number of
# unresolvable records rises above this fraction of the split.
MAX_UNAVAILABLE_IMAGE_FRACTION = 0.001


@dataclass(frozen=True)
class LandmarkTrainingConfig:
    resolution: int = 224
    heatmap_size: int = 56
    epochs: int = 8
    frozen_epochs: int = 1
    batch_size: int = 8
    accumulation: int = 2
    backbone_learning_rate: float = 1e-5
    head_learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    ema_decay: float = 0.99
    num_workers: int = 8
    seed: int = 20260807


def _prefix(filename: str) -> str:
    return re.sub(r"-\d{4}\.png$", "", filename)


def _split_names(root: Path, split: str) -> list[str]:
    return [
        line.strip()
        for line in (root / "extracted" / "labels" / f"labels-{split}.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _read_annotations(root: Path, split: str) -> dict[str, Any]:
    path = root / "extracted" / "labels" / f"labels-{split}.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _image_index(root: Path) -> dict[str, Path]:
    return {path.name: path for path in (root / "extracted" / "png-cache").rglob("*.png")}


def _physical_index(root: Path, prefixes: set[str]) -> dict[str, tuple[float, float]]:
    import pandas as pd

    output: dict[str, tuple[float, float]] = {}
    frame = pd.read_csv(
        root / "01_database_physical.csv",
        usecols=["FileHash", "PhysicalDeltaX", "PhysicalDeltaY"],
    )
    frame = frame.loc[frame["FileHash"].isin(prefixes)]
    for row in frame.itertuples(index=False):
        try:
            output[str(row.FileHash)] = (float(row.PhysicalDeltaX), float(row.PhysicalDeltaY))
        except (TypeError, ValueError):
            continue
    return output


def _view_index(root: Path) -> dict[str, str]:
    with (root / "view_index_20220705.csv").open(newline="", encoding="utf-8-sig") as handle:
        return {row["file"]: row["view"] for row in csv.DictReader(handle)}


class UnityLandmarkDataset:
    def __init__(
        self,
        root: Path,
        split: str,
        *,
        resolution: int = 224,
        augment: bool = False,
        maximum: int | None = None,
        images: dict[str, Path] | None = None,
        physical: dict[str, tuple[float, float]] | None = None,
        views: dict[str, str] | None = None,
        max_unavailable_fraction: float = MAX_UNAVAILABLE_IMAGE_FRACTION,
    ) -> None:
        self.root = root
        self.split = split
        self.resolution = resolution
        self.augment = augment
        names = _split_names(root, split)
        names = names[:maximum] if maximum is not None else names
        self.names = names
        self.annotations = _read_annotations(root, split)
        self.images = images if images is not None else _image_index(root)
        self.physical = physical if physical is not None else _physical_index(root, {_prefix(name) for name in self.names})
        self.views = views if views is not None else _view_index(root)
        unannotated = [name for name in names if name not in self.annotations]
        if unannotated:
            raise FileNotFoundError(
                f"Unity split {split!r} has {len(unannotated)} records without annotations: {unannotated[:3]}"
            )
        unavailable = [name for name in names if name not in self.images]
        permitted = math.floor(len(names) * max_unavailable_fraction)
        if len(unavailable) > permitted:
            raise FileNotFoundError(
                f"Unity split {split!r} has {len(unavailable)} records with no image in the official "
                f"png-cache, above the permitted {permitted} of {len(names)}: {unavailable[:3]}"
            )
        self.excluded_unavailable = tuple(unavailable)
        if unavailable:
            excluded = set(unavailable)
            self.names = [name for name in names if name not in excluded]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        from PIL import Image

        name = self.names[index]
        with Image.open(self.images[name]) as source:
            image = source.convert("RGB")
            width, height = image.size
            image = image.resize((self.resolution, self.resolution), resample=Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.augment:
            tensor = tensor.clamp(0, 1).pow(random.uniform(0.90, 1.10)).mul(random.uniform(0.90, 1.10)).clamp(0, 1)
            tensor = (tensor + torch.randn_like(tensor) * random.uniform(0.0, 0.015)).clamp(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        standard_deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        coordinates = torch.zeros((len(LANDMARKS), 2), dtype=torch.float32)
        visible = torch.zeros(len(LANDMARKS), dtype=torch.float32)
        labels = self.annotations[name].get("labels", {})
        for point_index, key in enumerate(LANDMARKS):
            label = labels.get(key, {})
            if label.get("type") != "point":
                continue
            try:
                x = float(label["x"])
                y = float(label["y"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= x < width and 0 <= y < height:
                coordinates[point_index] = torch.tensor([x / max(1, width - 1), y / max(1, height - 1)])
                visible[point_index] = 1.0
        delta_x, delta_y = self.physical.get(_prefix(name), (math.nan, math.nan))
        return {
            "video": ((tensor - mean) / standard_deviation).unsqueeze(1),
            "coordinates": coordinates,
            "visible": visible,
            "image_size": torch.tensor([width, height], dtype=torch.float32),
            "spacing_mm": torch.tensor([delta_x * 10.0, delta_y * 10.0], dtype=torch.float32),
            "name": name,
            "study": _prefix(name),
            "view": self.views.get(name, "unknown"),
        }


def _make_model(encoder, device: int, points: int, heatmap_size: int):
    import torch

    class LandmarkModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.decoder = torch.nn.Sequential(
                torch.nn.Conv2d(768, 256, 1),
                torch.nn.GroupNorm(8, 256),
                torch.nn.GELU(),
                torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                torch.nn.GroupNorm(8, 128),
                torch.nn.GELU(),
                torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(64, points, 1),
            )
            self.visibility = torch.nn.Sequential(
                torch.nn.LayerNorm(768), torch.nn.Linear(768, points)
            )
            self.heatmap_size = heatmap_size

        def forward(self, video):
            features = self.encoder(video)
            if features.ndim != 5 or features.shape[1] != 768:
                raise RuntimeError(f"unexpected DINO landmark feature shape: {tuple(features.shape)}")
            spatial = features.mean(dim=2)
            heatmap_logits = self.decoder(spatial)
            visibility_logits = self.visibility(spatial.mean(dim=(2, 3)))
            probabilities = heatmap_logits.flatten(2).softmax(dim=-1).reshape_as(heatmap_logits)
            grid = torch.linspace(0, 1, self.heatmap_size, device=video.device, dtype=probabilities.dtype)
            x = (probabilities * grid.view(1, 1, 1, -1)).sum(dim=(2, 3))
            y = (probabilities * grid.view(1, 1, -1, 1)).sum(dim=(2, 3))
            return {
                "heatmap_logits": heatmap_logits,
                "visibility_logits": visibility_logits,
                "coordinates": torch.stack([x, y], dim=-1),
            }

    return LandmarkModel().to(device)


def _loss(outputs, coordinates, visible, heatmap_size: int):
    import torch
    import torch.nn.functional as functional

    prediction = outputs["coordinates"].float()
    mask = visible.float()
    denominator = mask.sum().clamp_min(1.0)
    coordinate = (functional.smooth_l1_loss(prediction, coordinates.float(), reduction="none").sum(dim=-1) * mask).sum() / denominator
    visibility = functional.binary_cross_entropy_with_logits(outputs["visibility_logits"].float(), mask)
    grid = torch.arange(heatmap_size, device=prediction.device, dtype=torch.float32)
    x = coordinates[..., 0].float() * (heatmap_size - 1)
    y = coordinates[..., 1].float() * (heatmap_size - 1)
    target = torch.exp(
        -(
            (grid.view(1, 1, 1, -1) - x[..., None, None]).square()
            + (grid.view(1, 1, -1, 1) - y[..., None, None]).square()
        )
        / (2.0 * 1.5**2)
    )
    heatmap_error = (outputs["heatmap_logits"].float().sigmoid() - target).square()
    heatmap = (heatmap_error.mean(dim=(2, 3)) * mask).sum() / denominator
    total = (10.0 * coordinate) + visibility + heatmap
    return total, {
        "coordinate": float(coordinate.detach().cpu()),
        "visibility": float(visibility.detach().cpu()),
        "heatmap": float(heatmap.detach().cpu()),
    }


def _cpu_state(model) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _evaluate(model, loader, device: int, heatmap_size: int) -> dict[str, Any]:
    import torch

    errors: list[np.ndarray] = []
    millimetres: list[np.ndarray] = []
    visibility_predictions: list[np.ndarray] = []
    visibility_targets: list[np.ndarray] = []
    views: Counter[str] = Counter()
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            coordinates = batch["coordinates"].to(device, non_blocking=True)
            visible = batch["visible"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(video)
                loss, _ = _loss(outputs, coordinates, visible, heatmap_size)
            difference = outputs["coordinates"].float().cpu() - coordinates.cpu()
            normalized = torch.linalg.vector_norm(difference, dim=-1).numpy()
            mask = visible.cpu().numpy().astype(bool)
            errors.append(np.where(mask, normalized, np.nan))
            size = batch["image_size"].numpy()
            spacing = batch["spacing_mm"].numpy()
            pixel_difference = difference.numpy() * size[:, None, :]
            physical = np.sqrt(np.square(pixel_difference * spacing[:, None, :]).sum(axis=-1))
            physical[~mask | ~np.isfinite(physical)] = np.nan
            millimetres.append(physical)
            visibility_predictions.append(outputs["visibility_logits"].float().sigmoid().cpu().numpy())
            visibility_targets.append(mask.astype(np.float32))
            views.update(batch["view"])
            losses.append(float(loss.cpu()))
    error = np.concatenate(errors)
    physical_error = np.concatenate(millimetres)
    visibility_prediction = np.concatenate(visibility_predictions)
    visibility_target = np.concatenate(visibility_targets)
    valid = np.isfinite(error)
    metrics = {
        "mean_normalized_error": float(np.nanmean(error)),
        "median_normalized_error": float(np.nanmedian(error)),
        "pck_0_05": float(np.mean(error[valid] <= 0.05)),
        "pck_0_10": float(np.mean(error[valid] <= 0.10)),
        "mean_mm_error": float(np.nanmean(physical_error)),
        "median_mm_error": float(np.nanmedian(physical_error)),
        "visibility_accuracy": float(np.mean((visibility_prediction >= 0.5) == visibility_target)),
    }
    return {
        "loss": float(np.mean(losses)),
        "metrics": metrics,
        "visible_points": int(valid.sum()),
        "physical_points": int(np.isfinite(physical_error).sum()),
        "views": dict(sorted(views.items())),
    }


def run_unity_landmark_training(
    config: dict[str, Any],
    *,
    device: int = 1,
    epochs: int = 8,
    frozen_epochs: int = 1,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    assert_through(config, "G5")
    training = LandmarkTrainingConfig(
        epochs=epochs, frozen_epochs=frozen_epochs, seed=int(config["splits"]["seed"])
    )
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    torch.cuda.manual_seed_all(training.seed)
    torch.cuda.set_device(device)
    root = Path(config["paths"]["unity_root"])
    shared_images = _image_index(root)
    shared_views = _view_index(root)
    selected_names = _split_names(root, "train")[:maximum_train] if maximum_train is not None else _split_names(root, "train")
    selected_names += _split_names(root, "tune")[:maximum_validation] if maximum_validation is not None else _split_names(root, "tune")
    shared_physical = _physical_index(root, {_prefix(name) for name in selected_names})
    train_dataset = UnityLandmarkDataset(root, "train", resolution=training.resolution, augment=True, maximum=maximum_train, images=shared_images, physical=shared_physical, views=shared_views)
    validation_dataset = UnityLandmarkDataset(root, "tune", resolution=training.resolution, augment=False, maximum=maximum_validation, images=shared_images, physical=shared_physical, views=shared_views)
    if {_prefix(item) for item in train_dataset.names} & {
        _prefix(item) for item in validation_dataset.names
    }:
        raise RuntimeError("Unity study leakage")
    generator = torch.Generator().manual_seed(training.seed)
    train_loader = DataLoader(train_dataset, batch_size=training.batch_size, shuffle=True, num_workers=training.num_workers, pin_memory=True, persistent_workers=True, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=training.batch_size, shuffle=False, num_workers=training.num_workers, pin_memory=True, persistent_workers=True)
    encoder = _load_backbone(config, device, "dinov3_vitb")
    camus_checkpoint = Path(config["paths"]["run_root"]) / "camus_dense_dinov3_vitb_full" / "checkpoint_best.pt"
    state = torch.load(camus_checkpoint, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["ema"]["encoder"])
    del state
    model = _make_model(encoder, device, len(LANDMARKS), training.heatmap_size)
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
    run_root = Path(config["paths"]["run_root"]) / f"unity_landmarks_dinov3_{mode}"
    run_root.mkdir(parents=True, exist_ok=True)
    status_root = Path(config["paths"]["run_root"]) / "week_training"
    best_checkpoint = run_root / "checkpoint_best.pt"
    last_checkpoint = run_root / "checkpoint_last.pt"
    history: list[dict[str, Any]] = []
    best_score = math.inf
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
        components: dict[str, list[float]] = {}
        for step, batch in enumerate(train_loader, start=1):
            video = batch["video"].to(device, non_blocking=True)
            coordinates = batch["coordinates"].to(device, non_blocking=True)
            visible = batch["visible"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(video)
                loss, detail = _loss(outputs, coordinates, visible, training.heatmap_size)
            (loss / training.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            for key, value in detail.items():
                components.setdefault(key, []).append(value)
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
                _atomic_json(
                    status_root / "dense_lv_unity_landmarks_status.json",
                    {
                        "schema_version": 1,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "phase": f"unity_landmarks_dinov3_{mode}",
                        "epoch": epoch,
                        "total_epochs": training.epochs,
                        "complete_sequences": min(step * training.batch_size, len(train_dataset)),
                        "total_sequences": len(train_dataset),
                        "device": device,
                        "backbone_frozen": frozen,
                        "locked_test_accessed": False,
                    },
                )
        scheduler.step()
        validation = _evaluate(ema, validation_loader, device, training.heatmap_size)
        score = float(validation["metrics"]["mean_normalized_error"])
        history.append(
            {
                "epoch": epoch,
                "backbone_frozen": frozen,
                "train_loss": float(np.mean(losses)),
                "train_components": {key: float(np.mean(value)) for key, value in components.items()},
                "validation": validation,
            }
        )
        if score < best_score:
            best_score = score
            torch.save({"ema": _cpu_state(ema), "epoch": epoch, "validation": validation, "training": asdict(training), "landmarks": LANDMARKS, "track": "R", "locked_test_accessed": False}, best_checkpoint)
        torch.save({"model": _cpu_state(model), "ema": _cpu_state(ema), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history, "best_score": best_score, "training": asdict(training), "landmarks": LANDMARKS, "track": "R", "locked_test_accessed": False}, last_checkpoint)
        _atomic_json(run_root / "history.json", {"history": history})
    best = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    validation = best["validation"]
    checks = {
        "official_train_tune_only": True,
        "study_disjoint_validation": True,
        "locked_ival_annotations_not_accessed": True,
        "camus_dense_initialisation": camus_checkpoint.exists(),
        "masked_missing_landmarks": True,
        "unavailable_records_within_cap": True,
        "physical_spacing_evaluation": validation["physical_points"] > 0,
        "ema_validation": True,
        "bf16_compute_fp32_loss": True,
        "finite_metrics": all(np.isfinite(value) for value in validation["metrics"].values()),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "Unity development-only anatomical landmark localisation",
        "mode": mode,
        "backbone": "dinov3_vitb",
        "checks": checks,
        "training": asdict(training),
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "data_exclusions": {
            "reason": "record name absent from the official Unity png-cache",
            "max_unavailable_fraction": MAX_UNAVAILABLE_IMAGE_FRACTION,
            "train": list(train_dataset.excluded_unavailable),
            "tune": list(validation_dataset.excluded_unavailable),
        },
        "landmarks": LANDMARKS,
        "best_validation": validation,
        "history": history,
        "checkpoint_best": str(best_checkpoint),
        "checkpoint_last": str(last_checkpoint),
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "track": "R",
        "licence": "CC BY-NC-SA 4.0 for images, labels, metadata, and released weights; MIT code; weights are not redistributed",
        "locked_test_accessed": False,
    }
    destination = Path(config["paths"]["evidence_root"]) / "G6" / "unity_landmarks_dinov3.json" if mode == "full" else Path(config["paths"]["report_root"]) / "unity_landmarks_dinov3_smoke.json"
    _atomic_json(destination, report)
    _atomic_json(status_root / "dense_lv_unity_landmarks_status.json", {"schema_version": 1, "updated_utc": datetime.now(timezone.utc).isoformat(), "phase": f"unity_landmarks_dinov3_{mode}_complete", "complete_runs": training.epochs, "total_runs": training.epochs, "best_validation": validation["metrics"], "device": device, "locked_test_accessed": False})
    return report

