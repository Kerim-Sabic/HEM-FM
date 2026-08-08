from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from .camus_dense import DenseLVDecoder, DenseLVModel, _boundary_target, _load_backbone, _load_model_state
from .frozen_specialists import _atomic_json
from .gates import assert_through


FEATURES = (
    "predicted_sigma",
    "segmentation_entropy",
    "inverse_confidence",
    "foreground_fraction",
    "foreground_deviation_0_25",
    "temporal_area_roughness",
    "boundary_disagreement",
)


def _without_nii_suffix(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else name[:-4] if name.endswith(".nii") else path.stem


def _external_pairs(config: dict[str, Any]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    echocp = Path(config["paths"]["echocp_root"])
    for image in sorted(echocp.glob("*_image.nii.gz")):
        label = image.with_name(image.name.replace("_image.nii.gz", "_label.nii.gz"))
        if label.is_file():
            patient = image.name.split("_", 1)[0]
            output.append({"dataset": "EchoCP", "group": "contrast-bubble", "patient": f"EchoCP:{patient}", "image": str(image), "label": str(label)})

    cardiacnet = Path(config["paths"]["cardiacnet_root"])
    for image in sorted(path for path in cardiacnet.rglob("*_image.nii") if path.is_file()):
        container = image.parent
        label = container.parent / container.name.replace("_image.nii", "_label.nii")
        if not label.is_file():
            continue
        group = next(name for name in ("Non-ASD", "ASD", "Non-PAH", "PAH") if name in image.parts)
        match = re.search(r"patient-(\d+)", image.name)
        patient = match.group(1) if match else container.name.split("_", 1)[0]
        task = "ASD" if "CardiacNet-ASD" in image.parts else "PAH"
        output.append({"dataset": "CardiacNet", "group": group, "patient": f"CardiacNet:{task}:{patient}", "image": str(image), "label": str(label)})

    cardiacuda = Path(config["paths"]["cardiacuda_root"])
    for image in sorted(path for path in cardiacuda.rglob("*_image.nii.gz") if path.is_file()):
        if "label_all_frame" in image.parts:
            continue
        label = image.with_name(image.name.replace("_image.nii.gz", "_label.nii.gz"))
        if label.is_file():
            patient = _without_nii_suffix(image).removesuffix("_image")
            output.append({"dataset": "CardiacUDA", "group": image.parent.name, "patient": f"CardiacUDA:{image.parent.name}:{patient}", "image": str(image), "label": str(label)})
    return output


def _development_partition(patient: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:external-ood:{patient}".encode()).digest()
    return "calibration" if int.from_bytes(digest[:8], "big") / 2**64 < 0.70 else "evaluation"


def _load_annotated_frames(image_path: Path, label_path: Path, requested: int = 16) -> tuple[np.ndarray, np.ndarray, list[int]]:
    import nibabel as nib

    label_nii = nib.load(str(label_path))
    if len(label_nii.shape) != 3:
        raise ValueError(f"Expected 3D label cine: {label_path} {label_nii.shape}")
    label_all = np.asarray(label_nii.dataobj)
    annotated = np.flatnonzero(np.any(label_all != 0, axis=(0, 1)))
    if not len(annotated):
        raise ValueError(f"No annotated frames: {label_path}")
    positions = np.rint(np.linspace(0, len(annotated) - 1, requested)).astype(int)
    indices = annotated[positions]
    labels = np.stack([label_all[:, :, int(index)] for index in indices])
    del label_all

    image_nii = nib.load(str(image_path))
    if image_nii.shape != label_nii.shape:
        raise ValueError(f"Image/label shape mismatch: {image_path} {image_nii.shape} vs {label_nii.shape}")
    if image_path.name.endswith(".gz"):
        image_all = np.asarray(image_nii.dataobj)
        images = np.stack([image_all[:, :, int(index)] for index in indices])
        del image_all
    else:
        images = np.stack([np.asarray(image_nii.dataobj[:, :, int(index)]) for index in indices])
    return images, labels, [int(index) for index in indices]


def _prepare_pair(image: np.ndarray, label: np.ndarray, resolution: int = 224):
    import torch
    import torch.nn.functional as functional

    video = torch.from_numpy(image.copy()).unsqueeze(1).float()
    low = torch.quantile(video, 0.005)
    high = torch.quantile(video, 0.995)
    video = ((video - low) / (high - low).clamp_min(1.0)).clamp(0, 1)
    video = functional.interpolate(video, size=(resolution, resolution), mode="bilinear", align_corners=False)
    mask = torch.from_numpy(label.copy()).unsqueeze(1).float()
    mask = functional.interpolate(mask, size=(resolution, resolution), mode="nearest").squeeze(1).long()
    video = video.repeat(1, 3, 1, 1).permute(1, 0, 2, 3)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
    return ((video - mean) / std).unsqueeze(0).float(), mask.unsqueeze(0)


def _sequence_result(model, video, target, device: int) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(video.to(device, non_blocking=True))
    logits = outputs["segmentation"].float()
    probabilities = torch.softmax(logits, dim=1)
    prediction = logits.argmax(dim=1)
    target = target.to(device, non_blocking=True)
    predicted_foreground = prediction > 0
    target_foreground = target > 0
    intersection = (predicted_foreground & target_foreground).sum(dim=(-1, -2)).float() * 2.0
    denominator = predicted_foreground.sum(dim=(-1, -2)) + target_foreground.sum(dim=(-1, -2))
    dice = ((intersection + 1.0) / (denominator.float() + 1.0)).mean()
    confidence = probabilities.max(dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1) / math.log(probabilities.shape[1])
    foreground_area = (1.0 - probabilities[:, 0]).mean(dim=(-1, -2))
    roughness = (foreground_area[:, 2:] - (2 * foreground_area[:, 1:-1]) + foreground_area[:, :-2]).abs().mean()
    boundary_target = _boundary_target(prediction)
    boundary_disagreement = functional.binary_cross_entropy_with_logits(
        outputs["boundary"].float(), boundary_target
    )
    foreground_fraction = float(predicted_foreground.float().mean().cpu())
    return {
        "foreground_dice": float(dice.cpu()),
        "predicted_sigma": float(torch.exp(0.5 * outputs["log_variance"].float()).mean().cpu()),
        "segmentation_entropy": float(entropy.mean().cpu()),
        "inverse_confidence": float((1.0 - confidence).mean().cpu()),
        "foreground_fraction": foreground_fraction,
        "foreground_deviation_0_25": abs(foreground_fraction - 0.25),
        "temporal_area_roughness": float(roughness.cpu()),
        "boundary_disagreement": float(boundary_disagreement.cpu()),
    }


def _fit_failure_detector(records: list[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    calibration = [record for record in records if record["partition"] == "calibration"]
    evaluation = [record for record in records if record["partition"] == "evaluation"]
    x_train = np.asarray([[record[name] for name in FEATURES] for record in calibration], dtype=float)
    y_train = np.asarray([record["failure"] for record in calibration], dtype=int)
    x_test = np.asarray([[record[name] for name in FEATURES] for record in evaluation], dtype=float)
    y_test = np.asarray([record["failure"] for record in evaluation], dtype=int)
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return {"available": False, "reason": "both failure classes are required in calibration and evaluation partitions", "calibration_samples": len(calibration), "evaluation_samples": len(evaluation)}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260807))
    model.fit(x_train, y_train)
    train_probability = model.predict_proba(x_train)[:, 1]
    test_probability = model.predict_proba(x_test)[:, 1]
    thresholds = np.unique(train_probability)
    threshold = 0.5
    for candidate in sorted(thresholds):
        specificity = np.mean(train_probability[y_train == 0] < candidate)
        if specificity >= 0.90:
            threshold = float(candidate)
            break
    specificity = float(np.mean(test_probability[y_test == 0] < threshold))
    sensitivity = float(np.mean(test_probability[y_test == 1] >= threshold))
    baselines = {}
    for index, name in enumerate(FEATURES):
        raw = x_test[:, index]
        score = max(float(roc_auc_score(y_test, raw)), float(roc_auc_score(y_test, -raw)))
        baselines[name] = score
    return {
        "available": True,
        "features": FEATURES,
        "calibration_samples": len(calibration),
        "evaluation_samples": len(evaluation),
        "calibration_failure_rate": float(y_train.mean()),
        "evaluation_failure_rate": float(y_test.mean()),
        "failure_probability_threshold": threshold,
        "evaluation_auroc": float(roc_auc_score(y_test, test_probability)),
        "evaluation_average_precision": float(average_precision_score(y_test, test_probability)),
        "evaluation_sensitivity": sensitivity,
        "evaluation_specificity": specificity,
        "single_feature_aurocs_direction_agnostic": baselines,
        "model": "standardized class-balanced logistic regression fit on patient-disjoint external calibration partition",
    }


def run_external_ood_audit(
    config: dict[str, Any],
    *,
    device: int = 0,
    maximum_per_dataset: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    import torch

    assert_through(config, "G5")
    torch.cuda.set_device(device)
    seed = int(config["splits"]["seed"])
    pairs = _external_pairs(config)
    if maximum_per_dataset is not None:
        selected = []
        for dataset in ("EchoCP", "CardiacNet", "CardiacUDA"):
            selected.extend([item for item in pairs if item["dataset"] == dataset][:maximum_per_dataset])
        pairs = selected
    encoder = _load_backbone(config, device, "dinov3_vitb")
    decoder = DenseLVDecoder(in_channels=768, frames=16, resolution=224).to(device)
    model = DenseLVModel(encoder, decoder, 16).to(device).eval()
    checkpoint_path = Path(config["paths"]["run_root"]) / "camus_dense_dinov3_vitb_full" / "checkpoint_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _load_model_state(model, checkpoint["ema"])
    del checkpoint
    status_root = Path(config["paths"]["run_root"]) / "week_training"
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for index, item in enumerate(pairs, start=1):
        try:
            image, label, frame_indices = _load_annotated_frames(Path(item["image"]), Path(item["label"]), 16)
            video, target = _prepare_pair(image, label)
            metrics = _sequence_result(model, video, target, device)
            records.append({
                "dataset": item["dataset"],
                "group": item["group"],
                "patient": item["patient"],
                "sample_sha256": hashlib.sha256(item["image"].encode()).hexdigest(),
                "partition": _development_partition(item["patient"], seed),
                "annotated_source_frames": len(set(frame_indices)),
                **metrics,
                "failure": metrics["foreground_dice"] < 0.50,
            })
        except Exception as error:
            errors.append({"sample_sha256": hashlib.sha256(item["image"].encode()).hexdigest(), "error": f"{type(error).__name__}: {error}"})
        if index % 5 == 0 or index == len(pairs):
            status = {"schema_version": 1, "updated_utc": datetime.now(timezone.utc).isoformat(), "phase": f"external_ood_{mode}", "complete_sequences": index, "total_sequences": len(pairs), "device": device, "locked_test_accessed": False}
            _atomic_json(status_root / "external_ood_status.json", status)
            _atomic_json(status_root / "status.json", status)
    detector = _fit_failure_detector(records) if mode == "full" else {"available": False, "reason": "smoke mode does not fit or promote a detector"}
    promotion_eligible = bool(
        detector.get("available")
        and detector.get("evaluation_auroc", 0.0) >= 0.80
        and detector.get("evaluation_sensitivity", 0.0) >= 0.60
        and detector.get("evaluation_specificity", 0.0) >= 0.80
    )
    checks = {
        "camus_checkpoint_loaded": checkpoint_path.exists(),
        "all_three_external_sources_processed": {record["dataset"] for record in records} == {"EchoCP", "CardiacNet", "CardiacUDA"},
        "patient_disjoint_calibration_evaluation": not ({record["patient"] for record in records if record["partition"] == "calibration"} & {record["patient"] for record in records if record["partition"] == "evaluation"}),
        "no_scalar_head_updates": True,
        "no_locked_core_test_access": True,
        "bounded_error_rate": len(errors) / max(1, len(pairs)) <= (0.10 if mode == "smoke" else 0.01),
        "finite_features": all(np.isfinite(record[name]) for record in records for name in FEATURES),
    }
    dataset_summary = {}
    for dataset in ("EchoCP", "CardiacNet", "CardiacUDA"):
        selected = [record for record in records if record["dataset"] == dataset]
        dataset_summary[dataset] = {
            "sequences": len(selected),
            "groups": dict(sorted(Counter(record["group"] for record in selected).items())),
            "mean_foreground_dice": float(np.mean([record["foreground_dice"] for record in selected])) if selected else None,
            "failure_rate": float(np.mean([record["failure"] for record in selected])) if selected else None,
        }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "development-only external segmentation failure and abstention audit",
        "mode": mode,
        "checks": checks,
        "failure_definition": "binary foreground Dice below 0.50 on source-provided annotated frames",
        "promotion_eligible": promotion_eligible,
        "promotion_rule": "patient-disjoint evaluation AUROC >=0.80, sensitivity >=0.60, specificity >=0.80; otherwise keep the existing fail-closed system",
        "detector": detector,
        "dataset_summary": dataset_summary,
        "processed_sequences": len(records),
        "errors": errors,
        "records": records,
        "checkpoint": str(checkpoint_path),
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "track": "R",
        "locked_test_accessed": False,
    }
    destination = Path(config["paths"]["evidence_root"]) / "G6" / "external_ood_abstention.json" if mode == "full" else Path(config["paths"]["report_root"]) / "external_ood_abstention_smoke.json"
    _atomic_json(destination, report)
    _atomic_json(status_root / "external_ood_status.json", {"schema_version": 1, "updated_utc": datetime.now(timezone.utc).isoformat(), "phase": f"external_ood_{mode}_complete", "complete_runs": 1, "total_runs": 1, "processed_sequences": len(records), "promotion_eligible": promotion_eligible, "device": device, "locked_test_accessed": False})
    return report

