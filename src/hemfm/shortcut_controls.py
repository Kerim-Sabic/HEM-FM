from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dicom_preprocess import PreprocessedCine, preprocess_dicom_cine
from .hashing import sha256_file


FEATURE_NAMES = [
    "overlay_fraction",
    "overlay_top_fraction",
    "overlay_bottom_fraction",
    "overlay_left_fraction",
    "overlay_right_fraction",
    "border_mean",
    "border_std",
    "border_bright_fraction",
    "border_saturation",
    "border_temporal_std",
    "masked_mean",
    "masked_std",
    "masked_bright_fraction",
    "burned_in_annotation_yes",
    "dicom_overlay_groups",
]


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def _select_rows(frame: pd.DataFrame, per_split: dict[str, int], seed: int) -> pd.DataFrame:
    selected = []
    for (target, split), group in frame.groupby(["target", "split"], sort=True):
        limit = int(per_split.get(str(split), 0))
        if limit < 1:
            continue
        group = group.drop_duplicates("subject_id").copy()
        group["value"] = pd.to_numeric(group["value"], errors="coerce")
        group = group[group["value"].notna()]
        if len(group) <= limit:
            selected.append(group)
            continue
        # Cover the target range without using the locked test set.  A tiny seeded
        # jitter makes ties deterministic while retaining low/mid/high examples.
        rng = np.random.default_rng(seed + sum(ord(ch) for ch in str(target)) + len(str(split)))
        group["_order"] = group["value"] + rng.normal(0.0, 1e-9, size=len(group))
        group = group.sort_values("_order")
        indices = np.rint(np.linspace(0, len(group) - 1, limit)).astype(int)
        selected.append(group.iloc[indices].drop(columns="_order"))
    if not selected:
        return frame.head(0).copy()
    result = pd.concat(selected, ignore_index=True)
    return result.sort_values(["target", "split", "subject_id"]).reset_index(drop=True)


def _border_mask(height: int, width: int, fraction: float = 0.12) -> np.ndarray:
    thickness_y = max(1, int(round(height * fraction)))
    thickness_x = max(1, int(round(width * fraction)))
    mask = np.zeros((height, width), dtype=bool)
    mask[:thickness_y] = True
    mask[-thickness_y:] = True
    mask[:, :thickness_x] = True
    mask[:, -thickness_x:] = True
    return mask


def overlay_only_features(cine: PreprocessedCine, dicom_path: str | Path) -> np.ndarray:
    video = cine.video.astype(np.float32).transpose(1, 2, 3, 0)  # [T,H,W,C]
    maximum = video.max(axis=-1)
    minimum = video.min(axis=-1)
    mask = cine.overlay_mask.astype(bool)
    height, width = mask.shape
    border = _border_mask(height, width)
    border_values = maximum[:, border]
    border_saturation = (maximum - minimum)[:, border]
    masked_values = maximum[:, mask]
    if masked_values.size == 0:
        masked_mean = masked_std = masked_bright = 0.0
    else:
        masked_mean = float(masked_values.mean() / 255.0)
        masked_std = float(masked_values.std() / 255.0)
        masked_bright = float((masked_values >= 220.0).mean())
    dataset = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=False)
    burned_in = str(getattr(dataset, "BurnedInAnnotation", "")).strip().upper() == "YES"
    overlay_groups = sum(
        1
        for group in range(0x6000, 0x6020, 2)
        if (group, 0x0010) in dataset or (group, 0x3000) in dataset
    )
    top = mask[: max(1, height // 4)]
    bottom = mask[-max(1, height // 4) :]
    left = mask[:, : max(1, width // 4)]
    right = mask[:, -max(1, width // 4) :]
    return np.asarray(
        [
            float(mask.mean()),
            float(top.mean()),
            float(bottom.mean()),
            float(left.mean()),
            float(right.mean()),
            float(border_values.mean() / 255.0),
            float(border_values.std() / 255.0),
            float((border_values >= 220.0).mean()),
            float(border_saturation.mean() / 255.0),
            float(maximum[:, border].std(axis=0).mean() / 255.0),
            masked_mean,
            masked_std,
            masked_bright,
            float(burned_in),
            float(overlay_groups),
        ],
        dtype=np.float64,
    )


def _extract_one(row: dict[str, Any], dicom_root: Path, output_size: int) -> dict[str, Any]:
    cines = json.loads(row["cines"])
    if not cines:
        raise ValueError("Cohort row contains no DICOM cine")
    relative = str(cines[0]["path"])
    path = dicom_root.joinpath(*relative.replace("\\", "/").split("/"))
    calibration_type = "spectral" if row["target"] == "AV_PEAK_VELOCITY" else "spatial"
    requested_frames = 1 if calibration_type == "spectral" else 8
    cine = preprocess_dicom_cine(
        path,
        calibration_type=calibration_type,
        frames=requested_frames,
        output_size=output_size,
        clean_overlays=False,
    )
    return {
        "sample_id": _stable_id(row["subject_id"], row["study_id"], row["target"], relative),
        "subject_id": str(row["subject_id"]),
        "target": str(row["target"]),
        "split": str(row["split"]),
        "value": float(row["value"]),
        "relative_path": str(cines[0]["path"]),
        "calibration_type": calibration_type,
        "features": overlay_only_features(cine, path),
        "video": cine.video,
        "overlay_mask": cine.overlay_mask,
        "overlay_fraction": cine.overlay_fraction,
    }


def _fit_negative_controls(records: list[dict[str, Any]], seed: int) -> tuple[dict[str, Any], bool]:
    targets: dict[str, Any] = {}
    passed = True
    for target in sorted({str(record["target"]) for record in records}):
        train = [record for record in records if record["target"] == target and record["split"] == "train"]
        validation = [record for record in records if record["target"] == target and record["split"] == "validation"]
        if len(train) < 20 or len(validation) < 12:
            targets[target] = {
                "passed": False,
                "error": "Insufficient patient-disjoint samples",
                "train_samples": len(train),
                "validation_samples": len(validation),
            }
            passed = False
            continue
        x_train = np.stack([record["features"] for record in train])
        y_train = np.asarray([record["value"] for record in train], dtype=np.float64)
        x_validation = np.stack([record["features"] for record in validation])
        y_validation = np.asarray([record["value"] for record in validation], dtype=np.float64)
        predictor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        predictor.fit(x_train, y_train)
        prediction = predictor.predict(x_validation)
        r2 = float(r2_score(y_validation, prediction))
        mae = float(mean_absolute_error(y_validation, prediction))
        mean_baseline_mae = float(mean_absolute_error(y_validation, np.full_like(y_validation, y_train.mean())))
        rng = np.random.default_rng(seed + sum(ord(character) for character in target))
        permutation_r2 = []
        for _ in range(32):
            null_predictor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            null_predictor.fit(x_train, rng.permutation(y_train))
            permutation_r2.append(float(r2_score(y_validation, null_predictor.predict(x_validation))))
        target_passed = bool(r2 < 0.10)
        passed = passed and target_passed
        targets[target] = {
            "passed": target_passed,
            "train_samples": len(train),
            "validation_samples": len(validation),
            "patient_disjoint": not bool({r["subject_id"] for r in train} & {r["subject_id"] for r in validation}),
            "overlay_only_r2": r2,
            "overlay_only_mae": mae,
            "train_mean_baseline_mae": mean_baseline_mae,
            "r2_failure_threshold": 0.10,
            "permuted_train_label_r2_p95": float(np.percentile(permutation_r2, 95)),
        }
    return targets, passed


def _load_dinov3(config: dict[str, Any], device: int):
    import timm
    import torch
    from safetensors.torch import load_file

    model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=False, num_classes=0)
    incompatible = model.load_state_dict(load_file(config["paths"]["dinov3_checkpoint"], device="cpu"), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"DINOv3 state mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return model.eval().to(device=device, dtype=torch.bfloat16)


def _embedding_parity(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
    per_target: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    chosen = []
    for target in sorted({str(record["target"]) for record in records}):
        candidates = [record for record in records if record["target"] == target]
        chosen.extend(candidates[:per_target])
    model = _load_dinov3(config, device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    similarities = []
    relative_l2 = []
    changed_fraction = []
    sample_rows = []
    for start in range(0, len(chosen), 8):
        batch_records = chosen[start : start + 8]
        raw_images = []
        clean_images = []
        for record in batch_records:
            video = torch.from_numpy(record["video"]).float()
            image = video[:, video.shape[1] // 2] / 255.0
            mask = torch.from_numpy(record["overlay_mask"]).bool()
            cleaned = image.clone().masked_fill(mask.unsqueeze(0), 0.0)
            raw_images.append(image)
            clean_images.append(cleaned)
            changed_fraction.append(float(mask.float().mean().item()))
        raw = torch.stack(raw_images).to(device)
        cleaned = torch.stack(clean_images).to(device)
        raw = functional.interpolate(raw, size=(256, 256), mode="bilinear", align_corners=False)
        cleaned = functional.interpolate(cleaned, size=(256, 256), mode="bilinear", align_corners=False)
        raw = (raw - mean) / std
        cleaned = (cleaned - mean) / std
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            raw_tokens = model.forward_features(raw.to(dtype=torch.bfloat16))
            clean_tokens = model.forward_features(cleaned.to(dtype=torch.bfloat16))
            raw_embedding = raw_tokens.float().mean(dim=1)
            clean_embedding = clean_tokens.float().mean(dim=1)
        cosine = functional.cosine_similarity(raw_embedding, clean_embedding, dim=1).cpu().numpy()
        delta = ((raw_embedding - clean_embedding).norm(dim=1) / raw_embedding.norm(dim=1).clamp_min(1e-8)).cpu().numpy()
        similarities.extend(float(value) for value in cosine)
        relative_l2.extend(float(value) for value in delta)
        for record, cos_value, delta_value in zip(batch_records, cosine, delta):
            sample_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "target": record["target"],
                    "cosine_similarity": float(cos_value),
                    "relative_l2": float(delta_value),
                    "overlay_fraction": float(record["overlay_fraction"]),
                }
            )
    median_cosine = float(np.median(similarities))
    p05_cosine = float(np.percentile(similarities, 5))
    median_relative_l2 = float(np.median(relative_l2))
    passed = bool(len(chosen) >= 24 and median_cosine >= 0.98 and p05_cosine >= 0.90 and median_relative_l2 <= 0.20)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "control": "raw versus deterministic static-overlay-cleaned DINOv3 representation parity",
        "samples": len(chosen),
        "targets_covered": sorted({record["target"] for record in chosen}),
        "checkpoint_sha256": sha256_file(config["paths"]["dinov3_checkpoint"]),
        "locked_test_accessed": False,
        "median_cosine_similarity": median_cosine,
        "p05_cosine_similarity": p05_cosine,
        "median_relative_l2": median_relative_l2,
        "median_changed_pixel_fraction": float(np.median(changed_fraction)),
        "thresholds": {"median_cosine_min": 0.98, "p05_cosine_min": 0.90, "median_relative_l2_max": 0.20},
        "sample_results": sample_rows,
    }


def run_shortcut_controls(
    config: dict[str, Any],
    max_train_per_target: int = 60,
    max_validation_per_target: int = 30,
    workers: int = 8,
    device: int = 1,
) -> dict[str, Any]:
    private_root = Path(config["paths"]["private_root"])
    cohort = pd.read_csv(
        private_root / "endpoint_cohorts_development_private.csv",
        dtype={"subject_id": str, "study_id": str},
        low_memory=False,
    )
    if not set(cohort["split"]).issubset({"train", "validation"}) or cohort["test_locked"].fillna(False).astype(bool).any():
        raise RuntimeError("Locked test rows were present in the development manifest")
    selected = _select_rows(
        cohort,
        {"train": max_train_per_target, "validation": max_validation_per_target},
        int(config["splits"]["seed"]),
    )
    dicom_root = Path(config["paths"]["dicom_root"])
    records: list[dict[str, Any]] = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {
            executor.submit(_extract_one, row._asdict(), dicom_root, 128): row
            for row in selected.itertuples(index=False)
        }
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                records.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "sample_id": _stable_id(row.subject_id, row.study_id, row.target),
                        "target": row.target,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    cache_rows = []
    for record in records:
        cache_rows.append(
            {
                "sample_id": record["sample_id"],
                "subject_id": record["subject_id"],
                "target": record["target"],
                "split": record["split"],
                "value": record["value"],
                "relative_path": record["relative_path"],
                **{name: float(value) for name, value in zip(FEATURE_NAMES, record["features"])},
            }
        )
    pd.DataFrame(cache_rows).to_csv(private_root / "shortcut_control_features_private.csv", index=False)
    targets, shortcut_passed = _fit_negative_controls(records, int(config["splits"]["seed"]))
    control_report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(shortcut_passed and not errors),
        "control": "overlay/caliper/border-only ridge negative control",
        "feature_names": FEATURE_NAMES,
        "selected_rows": int(len(selected)),
        "successfully_decoded": len(records),
        "decode_errors": errors,
        "locked_test_accessed": False,
        "split_contract": "preassigned patient-disjoint train and validation only",
        "targets": targets,
    }
    clean_report = _embedding_parity(records, config, device=device, per_target=8)
    evidence_root = Path(config["paths"]["evidence_root"]) / "G2"
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "caliper_shortcut_controls.json").write_text(
        json.dumps(control_report, indent=2), encoding="utf-8"
    )
    (evidence_root / "clean_frame_negative_control.json").write_text(
        json.dumps(clean_report, indent=2), encoding="utf-8"
    )
    return {"caliper_shortcut_controls": control_report, "clean_frame_negative_control": clean_report}

