from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import nibabel as nib
import numpy as np

from .frozen_specialists import _atomic_json
from .gates import assert_through


CAMUS_CITATION = (
    "S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al., "
    '"Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D '
    'Echocardiography," IEEE TMI 38(9), 2019, doi:10.1109/TMI.2019.2900516.'
)


@dataclass(frozen=True)
class DenseTrainingConfig:
    frames: int = 16
    resolution: int = 224
    epochs: int = 18
    frozen_epochs: int = 2
    accumulation: int = 4
    backbone_learning_rate: float = 2e-5
    decoder_learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    ema_decay: float = 0.999
    num_workers: int = 2
    seed: int = 20260807


def _split_ids(split_root: Path, split: str) -> list[str]:
    path = split_root / f"subgroup_{split}.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_camus_acquisition_manifest(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["camus_root"])
    data_root = root / "database_nifti"
    split_root = root / "database_split"
    split_ids = {name: _split_ids(split_root, name) for name in ("training", "validation", "testing")}
    overlaps = {
        "train_validation": len(set(split_ids["training"]) & set(split_ids["validation"])),
        "train_test": len(set(split_ids["training"]) & set(split_ids["testing"])),
        "validation_test": len(set(split_ids["validation"]) & set(split_ids["testing"])),
    }
    available_patients = {path.name for path in data_root.glob("patient*") if path.is_dir()}
    expected_files = [
        data_root / patient / f"{patient}_{view}_half_sequence{suffix}.nii.gz"
        for patient in split_ids["training"] + split_ids["validation"]
        for view in ("2CH", "4CH")
        for suffix in ("", "_gt")
    ]
    download_status = root.parents[4] / "runs" / "week_training" / "dataset_download_status.json"
    status = json.loads(download_status.read_text(encoding="utf-8")) if download_status.exists() else {}
    checks = {
        "five_hundred_patients": len(available_patients) == 500,
        "official_400_50_50_split": [len(split_ids[name]) for name in ("training", "validation", "testing")] == [400, 50, 50],
        "patient_disjoint_splits": all(value == 0 for value in overlaps.values()),
        "all_development_sequences_present": all(path.exists() for path in expected_files),
        "archive_hash_recorded": len(status.get("sha256", "")) == 64,
        "locked_test_not_loaded": True,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "dataset": "CAMUS public database",
        "official_source": "https://www.creatis.insa-lyon.fr/Challenge/camus/",
        "download_sha256": status.get("sha256"),
        "archive_bytes": status.get("downloaded_bytes"),
        "local_root": str(root),
        "patients": {name: len(values) for name, values in split_ids.items()},
        "development_sequences": len(expected_files) // 2,
        "split_overlap": overlaps,
        "citation": CAMUS_CITATION,
        "track": "R",
        "redistribution": False,
        "clinical_use": False,
        "test_policy": "Official test patient identifiers are recorded but their images are not loaded until the analysis plan is frozen.",
        "locked_test_accessed": False,
    }
    destination = Path(config["paths"]["evidence_root"]) / "G4" / "camus_acquisition.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class CamusSequenceDataset:
    def __init__(
        self,
        root: Path,
        split: str,
        *,
        frames: int = 16,
        resolution: int = 224,
        augment: bool = False,
        maximum: int | None = None,
    ) -> None:
        self.root = root
        self.frames = frames
        self.resolution = resolution
        self.augment = augment
        patients = _split_ids(root / "database_split", split)
        self.samples = [(patient, view) for patient in patients for view in ("2CH", "4CH")]
        if maximum is not None:
            self.samples = self.samples[:maximum]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        import torch.nn.functional as functional
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import affine

        patient, view = self.samples[index]
        patient_root = self.root / "database_nifti" / patient
        stem = patient_root / f"{patient}_{view}_half_sequence"
        image_nii = nib.load(f"{stem}.nii.gz")
        mask_nii = nib.load(f"{stem}_gt.nii.gz")
        image = np.asanyarray(image_nii.dataobj).astype(np.float32, copy=False)
        mask = np.asanyarray(mask_nii.dataobj).astype(np.int64, copy=False)
        if image.ndim != 3 or mask.shape != image.shape:
            raise ValueError(f"Invalid CAMUS sequence shape for {patient} {view}: {image.shape}, {mask.shape}")
        if mask.min() < 0 or mask.max() > 3:
            raise ValueError(f"Unexpected CAMUS label for {patient} {view}")
        temporal = np.linspace(0, image.shape[2] - 1, self.frames).round().astype(int)
        image_tensor = torch.from_numpy(image[:, :, temporal].copy()).permute(2, 0, 1).unsqueeze(1)
        mask_tensor = torch.from_numpy(mask[:, :, temporal].copy()).permute(2, 0, 1).unsqueeze(1)
        original_height, original_width = image.shape[:2]
        image_tensor = functional.interpolate(
            image_tensor, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        mask_tensor = functional.interpolate(
            mask_tensor.float(), size=(self.resolution, self.resolution), mode="nearest"
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
            gain = random.uniform(0.90, 1.10)
            gamma = random.uniform(0.90, 1.10)
            image_tensor = ((image_tensor / 255.0).clamp(0, 1).pow(gamma) * gain).clamp(0, 1)
        else:
            image_tensor = (image_tensor / 255.0).clamp(0, 1)
        image_tensor = image_tensor.repeat(1, 3, 1, 1).permute(1, 0, 2, 3)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        standard_deviation = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        image_tensor = (image_tensor - mean) / standard_deviation
        zooms = image_nii.header.get_zooms()
        spacing = torch.tensor(
            [
                float(zooms[0]) * original_height / self.resolution,
                float(zooms[1]) * original_width / self.resolution,
            ],
            dtype=torch.float32,
        )
        return {
            "video": image_tensor.float(),
            "mask": mask_tensor.squeeze(1).long(),
            "spacing_mm": spacing,
            "patient": patient,
            "view": view,
        }


def _load_backbone(config: dict[str, Any], device: int, backbone: str = "vjepa21_vitb"):
    import torch

    if backbone == "dinov3_vitb":
        import timm
        from safetensors.torch import load_file

        vision = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m", pretrained=False, num_classes=0
        )
        incompatible = vision.load_state_dict(
            load_file(config["paths"]["dinov3_checkpoint"], device="cpu"), strict=False
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("DINOv3 checkpoint mismatch")
        if hasattr(vision, "set_grad_checkpointing"):
            vision.set_grad_checkpointing(True)

        class DinoSequenceEncoder(torch.nn.Module):
            def __init__(self, model) -> None:
                super().__init__()
                self.model = model

            def forward(self, video):
                batch, channels, frames, height, width = video.shape
                images = video.permute(0, 2, 1, 3, 4).reshape(
                    batch * frames, channels, height, width
                )
                tokens = self.model.forward_features(images)
                patches = tokens[:, -196:, :]
                return patches.reshape(batch, frames, 14, 14, 768).permute(0, 4, 1, 2, 3)

        return DinoSequenceEncoder(vision).to(device)
    if backbone != "vjepa21_vitb":
        raise ValueError(f"Unsupported dense backbone: {backbone}")

    source = Path(config["paths"]["vjepa21_source"])
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from app.vjepa_2_1.models.vision_transformer import vit_base

    model = vit_base(
        img_size=(224, 224),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        use_rope=True,
        uniform_power=True,
        use_sdpa=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        modality_embedding=True,
        n_registers=0,
        has_cls_first=False,
        use_activation_checkpointing=True,
    )
    model.return_hierarchical = True
    state = torch.load(
        config["paths"]["dense_vitb_checkpoint"], map_location="cpu", weights_only=True, mmap=True
    )
    encoder = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state["encoder"].items()
    }
    incompatible = model.load_state_dict(encoder, strict=False)
    del state, encoder
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"ViT-B checkpoint mismatch: {incompatible.missing_keys}, {incompatible.unexpected_keys}"
        )
    return model.to(device)


def _normalization(channels: int):
    import torch

    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return torch.nn.GroupNorm(groups, channels)


class DenseLVDecoder:
    def __init__(
        self, in_channels: int = 3072, width: int = 192, frames: int = 16, resolution: int = 224
    ) -> None:
        import torch

        self.module = torch.nn.ModuleDict(
            {
                "project": torch.nn.Conv3d(in_channels, width, 1),
                "low": torch.nn.Sequential(
                    _normalization(width), torch.nn.GELU(), torch.nn.Conv3d(width, width, 3, padding=1)
                ),
                "mid": torch.nn.Sequential(
                    _normalization(width), torch.nn.GELU(), torch.nn.Conv3d(width, 128, 3, padding=1)
                ),
                "high": torch.nn.Sequential(
                    _normalization(128), torch.nn.GELU(), torch.nn.Conv3d(128, 64, 3, padding=1)
                ),
                "refine": torch.nn.Sequential(
                    _normalization(64), torch.nn.GELU(), torch.nn.Conv3d(64, 64, 3, padding=1)
                ),
                "segmentation": torch.nn.Conv3d(64, 4, 1),
                "boundary": torch.nn.Conv3d(64, 1, 1),
                "log_variance": torch.nn.Conv3d(64, 1, 1),
            }
        )
        self.frames = frames
        self.resolution = resolution

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    def __call__(self, features):
        import torch.nn.functional as functional

        x = self.module["low"](self.module["project"](features))
        x = functional.interpolate(
            x,
            size=(self.frames, self.resolution // 4, self.resolution // 4),
            mode="trilinear",
            align_corners=False,
        )
        x = self.module["mid"](x)
        x = functional.interpolate(
            x,
            size=(self.frames, self.resolution // 2, self.resolution // 2),
            mode="trilinear",
            align_corners=False,
        )
        x = self.module["high"](x)
        x = functional.interpolate(
            x,
            size=(self.frames, self.resolution, self.resolution),
            mode="trilinear",
            align_corners=False,
        )
        x = self.module["refine"](x)
        return {
            "segmentation": self.module["segmentation"](x),
            "boundary": self.module["boundary"](x),
            "log_variance": self.module["log_variance"](x).clamp(-6, 4),
        }


def _reshape_tokens(tokens, frames: int):
    temporal = frames // 2
    spatial_tokens = tokens.shape[1] // temporal
    spatial = int(round(math.sqrt(spatial_tokens)))
    if temporal * spatial * spatial != tokens.shape[1]:
        raise ValueError(f"Cannot reshape {tokens.shape[1]} tokens into a video grid")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], temporal, spatial, spatial)


def _boundary_target(mask):
    import torch.nn.functional as functional

    foreground = (mask > 0).float().flatten(0, 1).unsqueeze(1)
    dilated = functional.max_pool2d(foreground, 3, stride=1, padding=1)
    eroded = -functional.max_pool2d(-foreground, 3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary.reshape(mask.shape[0], mask.shape[1], 1, mask.shape[2], mask.shape[3]).permute(0, 2, 1, 3, 4)


def _loss(outputs: dict[str, Any], mask) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    logits = outputs["segmentation"]
    class_weights = torch.tensor([0.15, 1.0, 1.0, 0.75], device=logits.device)
    pixel_ce = functional.cross_entropy(logits.float(), mask, weight=class_weights, reduction="none")
    log_variance = outputs["log_variance"].squeeze(1).float()
    heteroscedastic = (torch.exp(-log_variance) * pixel_ce + 0.05 * log_variance).mean()
    probabilities = torch.softmax(logits.float(), dim=1)
    dice_losses = []
    for label in (1, 2, 3):
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
    temporal = (
        (lv_area[:, 2:] - (2 * lv_area[:, 1:-1]) + lv_area[:, :-2]).abs().mean()
        if lv_area.shape[1] > 2
        else logits.new_tensor(0.0)
    )
    total = heteroscedastic + dice + (0.20 * boundary) + (0.10 * temporal)
    return total, {
        "heteroscedastic_ce": float(heteroscedastic.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "boundary_loss": float(boundary.detach().cpu()),
        "temporal_area_loss": float(temporal.detach().cpu()),
    }


class DenseLVModel:
    def __init__(self, encoder, decoder: DenseLVDecoder, frames: int) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.frames = frames

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()

    def to(self, *args, **kwargs):
        self.encoder.to(*args, **kwargs)
        self.decoder.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        self.decoder.train(mode)
        return self

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()
        return self

    def __call__(self, video):
        import torch

        tokens = self.encoder(video)
        if tokens.ndim == 5:
            return self.decoder(tokens)
        if isinstance(tokens, list):
            tokens = torch.cat(tokens, dim=-1)
        return self.decoder(_reshape_tokens(tokens, self.frames))


def _dice_counts(logits, mask) -> tuple[np.ndarray, np.ndarray]:
    prediction = logits.argmax(dim=1)
    intersections = []
    denominators = []
    for label in (1, 2, 3):
        predicted = prediction == label
        target = mask == label
        intersections.append(float((predicted & target).sum().cpu()) * 2.0)
        denominators.append(float(predicted.sum().cpu() + target.sum().cpu()))
    return np.asarray(intersections), np.asarray(denominators)


def _evaluate(model: DenseLVModel, loader, device: int) -> dict[str, Any]:
    import torch

    model.eval()
    intersections = np.zeros(3, dtype=np.float64)
    denominators = np.zeros(3, dtype=np.float64)
    losses = []
    uncertainty = []
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(video)
                loss, _ = _loss(outputs, mask)
            current_intersection, current_denominator = _dice_counts(outputs["segmentation"], mask)
            intersections += current_intersection
            denominators += current_denominator
            losses.append(float(loss.cpu()))
            uncertainty.append(float(torch.exp(0.5 * outputs["log_variance"].float()).mean().cpu()))
    per_class = intersections / np.clip(denominators, 1.0, None)
    return {
        "loss": float(np.mean(losses)),
        "mean_foreground_dice": float(per_class.mean()),
        "dice": {
            "lv_cavity": float(per_class[0]),
            "myocardium": float(per_class[1]),
            "left_atrium": float(per_class[2]),
        },
        "mean_predicted_sigma": float(np.mean(uncertainty)),
        "sequences": len(loader.dataset),
    }


def _cpu_state(module) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def _model_state(model: DenseLVModel) -> dict[str, Any]:
    return {"encoder": _cpu_state(model.encoder), "decoder": _cpu_state(model.decoder)}


def _load_model_state(model: DenseLVModel, state: dict[str, Any]) -> None:
    model.encoder.load_state_dict(state["encoder"])
    model.decoder.load_state_dict(state["decoder"])


def run_camus_dense_pretraining(
    config: dict[str, Any],
    *,
    device: int = 0,
    epochs: int = 18,
    frozen_epochs: int = 2,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
    backbone: str = "vjepa21_vitb",
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    assert_through(config, "G5")
    manifest = write_camus_acquisition_manifest(config)
    if not manifest["passed"]:
        raise RuntimeError("CAMUS acquisition manifest failed")
    training = DenseTrainingConfig(
        epochs=epochs,
        frozen_epochs=frozen_epochs,
        seed=int(config["splits"]["seed"]),
    )
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    torch.cuda.manual_seed_all(training.seed)
    torch.cuda.set_device(device)
    root = Path(config["paths"]["camus_root"])
    train_dataset = CamusSequenceDataset(
        root,
        "training",
        frames=training.frames,
        resolution=training.resolution,
        augment=True,
        maximum=maximum_train,
    )
    validation_dataset = CamusSequenceDataset(
        root,
        "validation",
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
    ema = DenseLVModel(deepcopy(encoder), deepcopy(decoder), training.frames).eval()
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
    run_label = "vitb" if backbone == "vjepa21_vitb" else backbone
    run_root = Path(config["paths"]["run_root"]) / f"camus_dense_{run_label}_{mode}"
    run_root.mkdir(parents=True, exist_ok=True)
    status_root = Path(config["paths"]["run_root"]) / "week_training"
    last_checkpoint = run_root / "checkpoint_last.pt"
    best_checkpoint = run_root / "checkpoint_best.pt"
    history = []
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
    for epoch in range(start_epoch, training.epochs + 1):
        frozen = epoch <= training.frozen_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(not frozen)
        model.train()
        if frozen:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        components: dict[str, list[float]] = {}
        for step, batch in enumerate(train_loader, start=1):
            video = batch["video"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(video)
                loss, detail = _loss(outputs, mask)
            (loss / training.accumulation).backward()
            epoch_losses.append(float(loss.detach().cpu()))
            for key, value in detail.items():
                components.setdefault(key, []).append(value)
            if step % training.accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for ema_parameter, parameter in zip(
                        ema.parameters(), model.parameters(), strict=True
                    ):
                        ema_parameter.lerp_(parameter.detach(), 1.0 - training.ema_decay)
            if step % 20 == 0 or step == len(train_loader):
                status = {
                    "schema_version": 1,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "phase": f"camus_dense_{backbone}_{mode}",
                    "epoch": epoch,
                    "total_epochs": training.epochs,
                    "complete_sequences": step,
                    "total_sequences": len(train_loader),
                    "device": device,
                    "backbone_frozen": frozen,
                    "locked_test_accessed": False,
                }
                _atomic_json(status_root / f"dense_lv_{backbone}_status.json", status)
                _atomic_json(status_root / "status.json", status)
        scheduler.step()
        validation = _evaluate(ema, validation_loader, device)
        epoch_report = {
            "epoch": epoch,
            "backbone_frozen": frozen,
            "train_loss": float(np.mean(epoch_losses)),
            "train_components": {
                key: float(np.mean(values)) for key, values in components.items()
            },
            "validation": validation,
        }
        history.append(epoch_report)
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
        "acquisition_manifest_passed": manifest["passed"],
        "official_development_split_only": True,
        "patient_disjoint_validation": True,
        "protocol_dense_backbone": backbone in {"vjepa21_vitb", "dinov3_vitb"},
        "multiclass_boundary_uncertainty_outputs": True,
        "ema_validation": True,
        "bf16_compute_fp32_loss": True,
        "full_finetune_after_frozen_warmup": training.epochs > training.frozen_epochs,
        "finite_best_dice": bool(np.isfinite(best_dice)),
        "locked_test_not_accessed": True,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "CAMUS development dense LV pretraining; official test remains sealed",
        "mode": mode,
        "backbone": backbone,
        "checks": checks,
        "training": asdict(training),
        "train_sequences": len(train_dataset),
        "validation_sequences": len(validation_dataset),
        "best_validation_mean_foreground_dice": best_dice,
        "history": history,
        "checkpoint_best": str(best_checkpoint),
        "checkpoint_last": str(last_checkpoint),
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "track": "R",
        "citation": CAMUS_CITATION,
        "locked_test_accessed": False,
    }
    destination = (
        Path(config["paths"]["evidence_root"]) / "G6" / f"camus_dense_{backbone}_pretraining.json"
        if mode == "full"
        else Path(config["paths"]["report_root"]) / "camus_dense_smoke.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    final_status = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": f"camus_dense_{backbone}_{mode}_complete",
        "complete_runs": training.epochs,
        "total_runs": training.epochs,
        "best_validation_mean_foreground_dice": best_dice,
        "device": device,
        "locked_test_accessed": False,
    }
    _atomic_json(status_root / f"dense_lv_{backbone}_status.json", final_status)
    _atomic_json(status_root / "status.json", final_status)
    return report

