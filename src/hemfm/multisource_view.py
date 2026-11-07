from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .camus_dense import _load_backbone
from .ev9v_view import _decode_video
from .extended_dataset_audit import _exam_key
from .frozen_specialists import _atomic_json
from .unity_landmarks import _image_index, _prefix, _split_names, _view_index


SCOPE = "unity_ev9v_development_bmode_view_classifier"
CLASSES = (
    "A2C",
    "A3C",
    "A4C",
    "A5C",
    "PLAX",
    "PSAX_UNSPECIFIED",
    "PSAX_MV",
    "PSAX_PM",
    "PSAX_APICAL",
    "PA_LAX",
    "SC4C",
)
UNITY_MAP = {"a2c": "A2C", "a3c": "A3C", "a4c": "A4C", "a5c": "A5C", "plax": "PLAX"}
EV9V_MAP = {
    "A4C": "A4C",
    "A5C": "A5C",
    "PASA": "PSAX_UNSPECIFIED",
    "PLHLA": "PLAX",
    "PMASA": "PSAX_APICAL",
    "PMPALA": "PA_LAX",
    "PMVLSA": "PSAX_MV",
    "PPMLSA": "PSAX_PM",
    "SC4C": "SC4C",
}
SHARED_CLASSES = ("A4C", "A5C", "PLAX")
SPLIT_SALT = "unity-ev9v-view-v1"
CALIBRATION_FRACTION = 0.60
MINIMUM_PARTITION_CLASS_SAMPLES = 5
DEVELOPMENT_GATE = {
    "audit_balanced_accuracy_min": 0.85,
    "audit_macro_f1_min": 0.85,
    "audit_per_class_recall_min": 0.65,
    "audit_ece_max": 0.08,
    "abstention_retained_accuracy_min": 0.95,
    "abstention_coverage_min": 0.50,
}
ABSTENTION_CALIBRATION_ACCURACY_TARGET = 0.97


@dataclass(frozen=True)
class ViewSample:
    source: str
    group: str
    name: str
    label: str
    paths: tuple[Path, ...]
    partition: str


@dataclass(frozen=True)
class TrainingConfig:
    frames: int = 4
    resolution: int = 224
    epochs: int = 6
    frozen_epochs: int = 1
    batch_size: int = 6
    accumulation: int = 2
    backbone_learning_rate: float = 8e-6
    head_learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    ema_decay: float = 0.995
    label_smoothing: float = 0.05
    num_workers: int = 4
    patience: int = 3
    seed: int = 20260811


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_groups(
    rows: Sequence[tuple[str, str]],
    *,
    source: str,
    minimum: int = MINIMUM_PARTITION_CLASS_SAMPLES,
) -> dict[str, str]:
    """Split groups while requiring label coverage; labels never select model outcomes."""
    group_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for group, label in rows:
        group_labels[group][label] += 1
    if not group_labels:
        raise ValueError("View partitioning requires groups")
    labels = sorted({label for _group, label in rows})
    for attempt in range(10_000):
        assignment = {}
        for group in sorted(group_labels):
            digest = hashlib.sha256(
                f"{SPLIT_SALT}:{source}:{attempt}:{group}".encode()
            ).digest()
            value = int.from_bytes(digest[:8], "big") / float(2**64)
            assignment[group] = "calibration" if value < CALIBRATION_FRACTION else "audit"
        counts = {
            partition: Counter(
                label
                for group, label in rows
                if assignment[group] == partition
            )
            for partition in ("calibration", "audit")
        }
        if all(counts[partition][label] >= minimum for partition in counts for label in labels):
            return assignment
    raise ValueError("Unable to create a group-disjoint view split with minimum class coverage")


def _deduplicate_paths(samples: Sequence[ViewSample]) -> tuple[list[ViewSample], list[dict[str, Any]]]:
    by_hash: dict[str, list[tuple[ViewSample, Path]]] = defaultdict(list)
    for sample in samples:
        for path in sample.paths:
            by_hash[_sha256(path)].append((sample, path))
    excluded: set[Path] = set()
    evidence = []
    for digest, entries in sorted(by_hash.items()):
        partitions = {sample.partition for sample, _path in entries}
        labels = {sample.label for sample, _path in entries}
        if len(partitions) > 1:
            raise ValueError("Exact view bytes cross development partitions")
        if len(labels) > 1:
            raise ValueError("Exact view bytes have conflicting labels")
        if len(entries) > 1:
            retained = sorted(path for _sample, path in entries)[0]
            duplicates = sorted(path for _sample, path in entries if path != retained)
            excluded.update(duplicates)
            evidence.append(
                {
                    "sha256": digest,
                    "label": next(iter(labels)),
                    "partition": next(iter(partitions)),
                    "retained": retained.name,
                    "excluded": [path.name for path in duplicates],
                }
            )
    output = []
    for sample in samples:
        paths = tuple(path for path in sample.paths if path not in excluded)
        if paths:
            output.append(
                ViewSample(
                    source=sample.source,
                    group=sample.group,
                    name=sample.name,
                    label=sample.label,
                    paths=paths,
                    partition=sample.partition,
                )
            )
    return output, evidence


def build_development_manifest(unity_root: Path, ev9v_root: Path) -> tuple[list[ViewSample], dict[str, Any]]:
    unity_root = Path(unity_root)
    ev9v_root = Path(ev9v_root)
    unity_images = _image_index(unity_root)
    unity_views = _view_index(unity_root)
    samples: list[ViewSample] = []
    unity_groups: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for split in ("train", "tune"):
        for name in _split_names(unity_root, split):
            if name not in unity_images:
                continue
            raw_view = unity_views.get(name)
            if raw_view not in UNITY_MAP:
                raise ValueError(f"Unknown Unity view label: {raw_view}")
            group = _prefix(name)
            unity_groups[split][group].append(unity_images[name])
    tune_rows = []
    for group, paths in unity_groups["tune"].items():
        labels = {UNITY_MAP[unity_views[path.name]] for path in paths}
        if len(labels) != 1:
            raise ValueError("Unity acquisition contains conflicting view labels")
        tune_rows.append((group, next(iter(labels))))
    unity_assignment = _partition_groups(tune_rows, source="unity")
    for split, groups in unity_groups.items():
        for group, paths in groups.items():
            labels = {UNITY_MAP[unity_views[path.name]] for path in paths}
            if len(labels) != 1:
                raise ValueError("Unity acquisition contains conflicting view labels")
            samples.append(
                ViewSample(
                    source="unity",
                    group=group,
                    name=group,
                    label=next(iter(labels)),
                    paths=tuple(sorted(paths)),
                    partition="train" if split == "train" else unity_assignment[group],
                )
            )

    frame = pd.read_csv(ev9v_root / "development_manifest.csv")
    if set(frame["split"]) != {"train", "validation"}:
        raise ValueError("EV9V development manifest has unexpected split labels")
    if (ev9v_root / "test_labeled.txt").exists():
        raise ValueError("EV9V TEST labels must not enter multi-source development")
    validation_rows = [
        (_exam_key(str(row.video_name)), EV9V_MAP[str(row.view)])
        for row in frame.loc[frame["split"] == "validation"].itertuples()
    ]
    ev9v_assignment = _partition_groups(validation_rows, source="ev9v")
    for row in frame.itertuples():
        raw_view = str(row.view)
        if raw_view not in EV9V_MAP:
            raise ValueError(f"Unknown EV9V view label: {raw_view}")
        path = Path(row.local_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        group = _exam_key(str(row.video_name))
        samples.append(
            ViewSample(
                source="ev9v",
                group=group,
                name=str(row.video_name),
                label=EV9V_MAP[raw_view],
                paths=(path,),
                partition="train" if str(row.split) == "train" else ev9v_assignment[group],
            )
        )

    samples, duplicate_evidence = _deduplicate_paths(samples)
    groups_by_partition = {
        partition: {(sample.source, sample.group) for sample in samples if sample.partition == partition}
        for partition in ("train", "calibration", "audit")
    }
    if any(
        groups_by_partition[left] & groups_by_partition[right]
        for left, right in (("train", "calibration"), ("train", "audit"), ("calibration", "audit"))
    ):
        raise ValueError("View group leakage across development partitions")
    rows = [
        {
            "source": sample.source,
            "group": sample.group,
            "name": sample.name,
            "label": sample.label,
            "partition": sample.partition,
            "files": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sample.paths
            ],
        }
        for sample in sorted(samples, key=lambda value: (value.partition, value.source, value.name))
    ]
    fingerprint = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema_version": 1,
        "scope": SCOPE,
        "fingerprint": fingerprint,
        "samples": len(samples),
        "groups": {partition: len(groups) for partition, groups in groups_by_partition.items()},
        "counts": {
            partition: dict(
                sorted(Counter(sample.label for sample in samples if sample.partition == partition).items())
            )
            for partition in ("train", "calibration", "audit")
        },
        "sources": {
            partition: dict(
                sorted(Counter(sample.source for sample in samples if sample.partition == partition).items())
            )
            for partition in ("train", "calibration", "audit")
        },
        "exact_duplicate_quarantine": duplicate_evidence,
        "unity_ival_pixels_accessed": False,
        "ev9v_test_pixels_accessed": False,
        "locked_test_accessed": False,
    }
    return samples, report


class MultiSourceViewDataset:
    def __init__(self, samples: Sequence[ViewSample], config: TrainingConfig, *, augment: bool) -> None:
        self.samples = tuple(samples)
        self.config = config
        self.augment = augment
        self.label_to_index = {label: index for index, label in enumerate(CLASSES)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch
        import torch.nn.functional as functional
        from PIL import Image

        sample = self.samples[index]
        if sample.source == "ev9v":
            array = _decode_video(
                sample.paths[0], self.config.frames, self.augment
            )
        else:
            path = random.choice(sample.paths) if self.augment else sample.paths[len(sample.paths) // 2]
            with Image.open(path) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
            array = np.repeat(frame[None], self.config.frames, axis=0)
        video = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float() / 255.0
        video = functional.interpolate(
            video,
            size=(self.config.resolution, self.config.resolution),
            mode="bilinear",
            align_corners=False,
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
            "label": self.label_to_index[sample.label],
            "source": sample.source,
            "group": sample.group,
            "name": sample.name,
        }


def _make_model(encoder, device: int):
    import torch

    class Router(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(768),
                torch.nn.Linear(768, 384),
                torch.nn.GELU(),
                torch.nn.Dropout(0.20),
                torch.nn.Linear(384, len(CLASSES)),
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

    return Router().to(device)


def _metrics(logits: np.ndarray, targets: np.ndarray, sources: Sequence[str]) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = prediction == targets
    ece = 0.0
    for lower in np.linspace(0, 1, 16)[:-1]:
        selected = (confidence > lower) & (confidence <= lower + 1 / 15)
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    matrix = confusion_matrix(targets, prediction, labels=np.arange(len(CLASSES)))
    recalls = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(len(CLASSES), dtype=float),
        where=matrix.sum(axis=1) > 0,
    )
    per_source = {}
    source_array = np.asarray(sources)
    for source in sorted(set(sources)):
        selected = source_array == source
        source_targets = targets[selected]
        source_prediction = prediction[selected]
        per_source[source] = {
            "samples": int(selected.sum()),
            "accuracy": float((source_prediction == source_targets).mean()),
            "per_class_recall": {
                label: (
                    float((source_prediction[source_targets == index] == index).mean())
                    if bool((source_targets == index).any())
                    else None
                )
                for index, label in enumerate(CLASSES)
            },
        }
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(targets, prediction)),
        "macro_f1": float(f1_score(targets, prediction, average="macro", zero_division=0)),
        "ece_15_bin": float(ece),
        "per_class_recall": {label: float(recalls[index]) for index, label in enumerate(CLASSES)},
        "confusion_matrix": matrix.tolist(),
        "per_source": per_source,
    }


def _evaluate(model, loader, device: int) -> dict[str, Any]:
    import torch

    logits, targets, sources = [], [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = model(video)
            logits.append(output.float().cpu().numpy())
            targets.append(batch["label"].numpy())
            sources.extend(str(value) for value in batch["source"])
    values = np.concatenate(logits)
    truth = np.concatenate(targets)
    return {
        "logits": values,
        "targets": truth,
        "sources": sources,
        "metrics": _metrics(values, truth, sources),
    }


def _temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        scaled = logits / temperature
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))

    result = minimize_scalar(objective, bounds=(math.log(0.25), math.log(4.0)), method="bounded")
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("Temperature calibration failed")
    return float(math.exp(result.x))


def _abstention_threshold(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= probability.sum(axis=1, keepdims=True)
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == targets
    best = None
    for threshold in sorted(set(float(value) for value in confidence)):
        retained = confidence >= threshold
        if not retained.any():
            continue
        accuracy = float(correct[retained].mean())
        coverage = float(retained.mean())
        if accuracy >= ABSTENTION_CALIBRATION_ACCURACY_TARGET and (
            best is None or coverage > best["calibration_coverage"]
        ):
            best = {
                "threshold": threshold,
                "calibration_coverage": coverage,
                "calibration_retained_accuracy": accuracy,
                "calibration_accuracy_target": ABSTENTION_CALIBRATION_ACCURACY_TARGET,
            }
    return best or {
        "threshold": 1.0,
        "calibration_coverage": 0.0,
        "calibration_retained_accuracy": None,
        "calibration_accuracy_target": ABSTENTION_CALIBRATION_ACCURACY_TARGET,
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    device: int,
    mode: str,
    epochs: int | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    settings = TrainingConfig(
        epochs=epochs if epochs is not None else (1 if mode == "smoke" else 6),
        frozen_epochs=1,
        num_workers=0 if mode == "smoke" else 4,
    )
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    torch.cuda.manual_seed_all(settings.seed)
    torch.cuda.set_device(device)
    samples, manifest = build_development_manifest(
        Path(config["paths"]["unity_root"]), Path(config["paths"]["ev9v_root"])
    )
    if mode == "smoke":
        selected = []
        for partition in ("train", "calibration", "audit"):
            for label in CLASSES:
                selected.extend(
                    [sample for sample in samples if sample.partition == partition and sample.label == label][
                        :1
                    ]
                )
        samples = selected
    by_partition = {
        partition: [sample for sample in samples if sample.partition == partition]
        for partition in ("train", "calibration", "audit")
    }
    for partition, values in by_partition.items():
        if set(sample.label for sample in values) != set(CLASSES):
            raise ValueError(f"{partition} partition lacks one or more unified view classes")
    datasets = {
        partition: MultiSourceViewDataset(values, settings, augment=partition == "train")
        for partition, values in by_partition.items()
    }
    generator = torch.Generator().manual_seed(settings.seed)
    loaders = {
        partition: DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=partition == "train",
            generator=generator if partition == "train" else None,
            num_workers=settings.num_workers,
            pin_memory=True,
            persistent_workers=settings.num_workers > 0,
        )
        for partition, dataset in datasets.items()
    }
    encoder = _load_backbone(dict(config), device, "dinov3_vitb")
    initializer_path = Path(config["paths"]["run_root"]) / "ev9v_view_dinov3_full" / "checkpoint_best.pt"
    initializer = torch.load(initializer_path, map_location="cpu", weights_only=True)
    if initializer.get("locked_test_accessed") is not False:
        raise ValueError("EV9V initializer lacks a sealed TEST attestation")
    encoder_state = {
        name.removeprefix("encoder."): value
        for name, value in initializer["ema"].items()
        if name.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    model = _make_model(encoder, device)
    ema = deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    counts = Counter(sample.label for sample in by_partition["train"])
    class_weights = np.asarray([1.0 / math.sqrt(counts[label]) for label in CLASSES], dtype=np.float32)
    class_weights /= class_weights.mean()
    class_weights_tensor = torch.from_numpy(class_weights).to(device)
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": settings.backbone_learning_rate},
            {"params": head_parameters, "lr": settings.head_learning_rate},
        ],
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite multi-source view evidence: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "manifest.json", manifest)
    best_path = output_dir / "checkpoint_best.pt"
    last_path = output_dir / "checkpoint_last.pt"
    history = []
    best_score = -math.inf
    best_epoch = None
    bad_epochs = 0
    updates = 0
    for epoch in range(1, settings.epochs + 1):
        frozen = epoch <= settings.frozen_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(not frozen)
        model.train()
        if frozen:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(loaders["train"], start=1):
            video = batch["video"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(video)
                loss = torch.nn.functional.cross_entropy(
                    logits.float(),
                    target,
                    weight=class_weights_tensor,
                    label_smoothing=settings.label_smoothing,
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite multi-source view loss")
            (loss / settings.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if step % settings.accumulation == 0 or step == len(loaders["train"]):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                decay = min(settings.ema_decay, (1.0 + updates) / (10.0 + updates))
                with torch.no_grad():
                    for ema_parameter, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
        scheduler.step()
        calibration = _evaluate(ema, loaders["calibration"], device)
        score = float(calibration["metrics"]["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "backbone_frozen": frozen,
                "train_loss": float(np.mean(losses)),
                "calibration_metrics": calibration["metrics"],
            }
        )
        _atomic_json(output_dir / "history.json", history)
        payload = {
            "schema_version": 1,
            "scope": SCOPE,
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "ema": {name: value.detach().cpu() for name, value in ema.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_score": max(best_score, score),
            "settings": asdict(settings),
            "classes": CLASSES,
            "manifest_fingerprint": manifest["fingerprint"],
            "initializer_sha256": _sha256(initializer_path),
            "locked_test_accessed": False,
        }
        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            _atomic_torch_save(payload, best_path)
        else:
            bad_epochs += 1
        _atomic_torch_save({**payload, "bad_epochs": bad_epochs}, last_path)
        if bad_epochs >= settings.patience:
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    ema.load_state_dict(best["ema"])
    calibration = _evaluate(ema, loaders["calibration"], device)
    temperature = _temperature(calibration["logits"], calibration["targets"])
    threshold = _abstention_threshold(calibration["logits"] / temperature, calibration["targets"])
    audit = _evaluate(ema, loaders["audit"], device)
    audit_logits = audit["logits"] / temperature
    calibrated_metrics = _metrics(audit_logits, audit["targets"], audit["sources"])
    shifted = audit_logits - audit_logits.max(axis=1, keepdims=True)
    audit_probability = np.exp(shifted)
    audit_probability /= audit_probability.sum(axis=1, keepdims=True)
    confidence = audit_probability.max(axis=1)
    retained = confidence >= threshold["threshold"]
    prediction = audit_probability.argmax(axis=1)
    retained_accuracy = float((prediction[retained] == audit["targets"][retained]).mean()) if retained.any() else None
    shared_gaps = {}
    for label in SHARED_CLASSES:
        values = []
        for source in ("ev9v", "unity"):
            value = calibrated_metrics["per_source"][source]["per_class_recall"][label]
            if value is not None:
                values.append(float(value))
        shared_gaps[label] = max(values) - min(values) if len(values) == 2 else None
    checks = {
        "audit_balanced_accuracy": calibrated_metrics["balanced_accuracy"] >= DEVELOPMENT_GATE["audit_balanced_accuracy_min"],
        "audit_macro_f1": calibrated_metrics["macro_f1"] >= DEVELOPMENT_GATE["audit_macro_f1_min"],
        "audit_per_class_recall": min(calibrated_metrics["per_class_recall"].values()) >= DEVELOPMENT_GATE["audit_per_class_recall_min"],
        "audit_ece": calibrated_metrics["ece_15_bin"] <= DEVELOPMENT_GATE["audit_ece_max"],
        "abstention_coverage": float(retained.mean()) >= DEVELOPMENT_GATE["abstention_coverage_min"],
        "abstention_retained_accuracy": retained_accuracy is not None and retained_accuracy >= DEVELOPMENT_GATE["abstention_retained_accuracy_min"],
        "locked_test_sealed": True,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "mode": mode,
        "classes": CLASSES,
        "settings": asdict(settings),
        "manifest": manifest,
        "best_epoch": best_epoch,
        "calibration_metrics": calibration["metrics"],
        "temperature": temperature,
        "audit_metrics": calibrated_metrics,
        "shared_source_recall_gap": shared_gaps,
        "abstention": {
            **threshold,
            "audit_coverage": float(retained.mean()),
            "audit_retained_accuracy": retained_accuracy,
            "audit_retained_samples": int(retained.sum()),
            "audit_samples": len(retained),
        },
        "development_gate": DEVELOPMENT_GATE,
        "checks": checks,
        "development_gate_passed": all(checks.values()),
        "automatic_measurement_routing_allowed": False,
        "development_audit_status": "repeated_challenger_audit",
        "development_audit_reuse_count": 2,
        "limitations": [
            "No spectral Doppler, PW Doppler, TDI, M-mode, or unknown/OOD class.",
            "Unity and EV9V are separate acquisition domains; external DICOM validation is required.",
            "Unity-derived weights remain subject to CC BY-NC-SA research restrictions.",
            "The development audit was reused after the first challenger; results are not an untouched confirmation.",
        ],
        "unity_ival_pixels_accessed": False,
        "ev9v_test_pixels_accessed": False,
        "locked_test_accessed": False,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unity+EV9V development B-mode view classifier")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--epochs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    args = _parser().parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = train(
        config,
        args.output_dir,
        device=args.device,
        mode=args.mode,
        epochs=args.epochs,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
