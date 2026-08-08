from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dicom_preprocess import preprocess_dicom_cine
from .hashing import sha256_file
from .shortcut_controls import _select_rows


ARCHITECTURES = [
    "functional_vjepa2_vitl",
    "dense_vjepa21_vitb",
    "dense_vjepa21_vitl",
    "dinov3_vitb",
]


def _sample_id(row: dict[str, Any]) -> str:
    payload = f"{row['subject_id']}|{row['study_id']}|{row['target']}|{row['cines']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _load_video_encoder(config: dict[str, Any], kind: str, device: int):
    import torch

    if kind == "functional_vjepa2_vitl":
        source = Path(config["paths"]["echojepa_source"])
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from src.models.vision_transformer import vit_large

        model = vit_large(
            img_size=(224, 224), patch_size=16, num_frames=16, tubelet_size=2,
            use_rope=True, uniform_power=True, use_sdpa=True,
            use_activation_checkpointing=False,
        )
        checkpoint = Path(config["paths"]["functional_checkpoint"])
    else:
        source = Path(config["paths"]["vjepa21_source"])
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from app.vjepa_2_1.models.vision_transformer import vit_base, vit_large

        constructor = vit_base if kind == "dense_vjepa21_vitb" else vit_large
        model = constructor(
            img_size=(224, 224), patch_size=16, num_frames=16, tubelet_size=2,
            use_rope=True, uniform_power=True, use_sdpa=True, img_temporal_dim_size=1,
            interpolate_rope=True, modality_embedding=True, n_registers=0,
            has_cls_first=False, use_activation_checkpointing=False,
        )
        model.return_hierarchical = True
        checkpoint = Path(config["paths"]["dense_vitb_checkpoint" if kind.endswith("vitb") else "dense_vitl_checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    encoder = {key.replace("module.", "").replace("backbone.", ""): value for key, value in state["encoder"].items()}
    incompatible = model.load_state_dict(encoder, strict=False)
    missing = [key for key in incompatible.missing_keys if key != "pos_embed"]
    unexpected = [key for key in incompatible.unexpected_keys if key != "pos_embed"]
    del state, encoder
    if missing or unexpected:
        raise RuntimeError(f"{kind} checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval().to(device=device, dtype=torch.bfloat16)


def _load_dino(config: dict[str, Any], device: int):
    import timm
    import torch
    from safetensors.torch import load_file

    model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=False, num_classes=0)
    incompatible = model.load_state_dict(load_file(config["paths"]["dinov3_checkpoint"], device="cpu"), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("DINOv3 checkpoint mismatch")
    return model.eval().to(device=device, dtype=torch.bfloat16)


def _mean_tokens(output):
    if isinstance(output, list):
        output = output[-1]
    return output.float().mean(dim=1)


def _embed_row(
    row: dict[str, Any], config: dict[str, Any], models: dict[str, Any], devices: dict[str, int]
) -> dict[str, np.ndarray]:
    import torch
    import torch.nn.functional as functional

    target = str(row["target"])
    calibration_type = "spectral" if target == "AV_PEAK_VELOCITY" else "spatial"
    root = Path(config["paths"]["dicom_root"])
    decoded = []
    for item in json.loads(row["cines"]):
        path = root.joinpath(*str(item["path"]).replace("\\", "/").split("/"))
        decoded.append(
            preprocess_dicom_cine(
                path, calibration_type=calibration_type, frames=16,
                output_size=224, clean_overlays=True,
            ).video
        )
    if not decoded:
        raise ValueError("No cines in selected cohort row")
    videos = torch.from_numpy(np.stack(decoded)).float() / 255.0
    mean_video = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std_video = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    videos = (videos - mean_video) / std_video
    result: dict[str, np.ndarray] = {}
    for name in ("functional_vjepa2_vitl", "dense_vjepa21_vitb", "dense_vjepa21_vitl"):
        device = devices[name]
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            embedding = _mean_tokens(
                models[name](videos.to(device=device, dtype=torch.bfloat16))
            ).mean(dim=0)
        result[name] = embedding.cpu().numpy().astype(np.float16)
    middle = videos[:, :, videos.shape[2] // 2]
    middle = functional.interpolate(middle, size=(256, 256), mode="bilinear", align_corners=False)
    device = devices["dinov3_vitb"]
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        embedding = _mean_tokens(
            models["dinov3_vitb"].forward_features(middle.to(device=device, dtype=torch.bfloat16))
        ).mean(dim=0)
    result["dinov3_vitb"] = embedding.cpu().numpy().astype(np.float16)
    return result


def _bootstrap_mae(
    y_true: np.ndarray, y_pred: np.ndarray, seed: int, trials: int = 500
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(trials):
        index = rng.integers(0, len(y_true), size=len(y_true))
        values.append(float(mean_absolute_error(y_true[index], y_pred[index])))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _evaluate(
    samples: list[dict[str, Any]], seed: int
) -> tuple[dict[str, Any], dict[str, str]]:
    metrics: dict[str, Any] = {}
    winners: dict[str, str] = {}
    targets = sorted({sample["target"] for sample in samples})
    for target in targets:
        target_samples = [sample for sample in samples if sample["target"] == target]
        train = [sample for sample in target_samples if sample["split"] == "train"]
        validation = [sample for sample in target_samples if sample["split"] == "validation"]
        y_train = np.asarray([sample["value"] for sample in train], dtype=np.float64)
        y_validation = np.asarray([sample["value"] for sample in validation], dtype=np.float64)
        baseline_prediction = np.full_like(y_validation, np.median(y_train))
        baseline_mae = float(mean_absolute_error(y_validation, baseline_prediction))
        endpoint = {
            "train_samples": len(train),
            "validation_samples": len(validation),
            "patient_disjoint": not bool(
                {sample["subject_id"] for sample in train}
                & {sample["subject_id"] for sample in validation}
            ),
            "median_baseline_mae": baseline_mae,
            "candidates": {},
        }
        for candidate in ARCHITECTURES:
            x_train = np.stack([sample[candidate] for sample in train]).astype(np.float32)
            x_validation = np.stack([sample[candidate] for sample in validation]).astype(np.float32)
            folds = KFold(n_splits=4, shuffle=True, random_state=seed)
            search = GridSearchCV(
                Pipeline(
                    [("scale", StandardScaler()), ("ridge", Ridge(solver="lsqr", max_iter=5000))]
                ),
                {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
                scoring="neg_mean_absolute_error", cv=folds, n_jobs=1,
            )
            search.fit(x_train, y_train)
            prediction = search.predict(x_validation)
            correlation = spearmanr(y_validation, prediction).statistic
            candidate_mae = float(mean_absolute_error(y_validation, prediction))
            endpoint["candidates"][candidate] = {
                "mae": candidate_mae,
                "mae_95ci": _bootstrap_mae(
                    y_validation, prediction, seed + len(candidate) + len(target)
                ),
                "normalized_mae_vs_median": candidate_mae / max(baseline_mae, 1e-8),
                "r2": float(r2_score(y_validation, prediction)),
                "spearman": float(correlation) if np.isfinite(correlation) else None,
                "train_only_selected_alpha": float(search.best_params_["ridge__alpha"]),
                "feature_dimension": int(x_train.shape[1]),
            }
        winners[target] = min(
            endpoint["candidates"], key=lambda name: endpoint["candidates"][name]["mae"]
        )
        endpoint["pilot_winner"] = winners[target]
        metrics[target] = endpoint
    return metrics, winners


def run_architecture_ladder(
    config: dict[str, Any], train_per_target: int = 32,
    validation_per_target: int = 16,
) -> dict[str, Any]:
    started = time.perf_counter()
    cohort = pd.read_csv(
        Path(config["paths"]["private_root"])
        / "endpoint_cohorts_development_private.csv",
        dtype={"subject_id": str, "study_id": str}, low_memory=False,
    )
    if (
        not set(cohort["split"]).issubset({"train", "validation"})
        or cohort["test_locked"].fillna(False).astype(bool).any()
    ):
        raise RuntimeError("Architecture pilot received a locked test row")
    selected = _select_rows(
        cohort,
        {"train": train_per_target, "validation": validation_per_target},
        int(config["splits"]["seed"]) + 17,
    )
    cache_root = Path(config["paths"]["private_root"]) / "architecture_ladder_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    pending = []
    samples: list[dict[str, Any]] = []
    for row_tuple in selected.itertuples(index=False):
        row = row_tuple._asdict()
        identifier = _sample_id(row)
        path = cache_root / f"{identifier}.npz"
        if path.exists():
            payload = np.load(path, allow_pickle=False)
            samples.append(
                {
                    "sample_id": identifier, "subject_id": str(row["subject_id"]),
                    "target": str(row["target"]), "split": str(row["split"]),
                    "value": float(row["value"]),
                    **{name: payload[name] for name in ARCHITECTURES},
                }
            )
        else:
            pending.append((row, identifier, path))
    errors = []
    model_metadata = {}
    if pending:
        devices = {
            "functional_vjepa2_vitl": 0, "dense_vjepa21_vitb": 1,
            "dense_vjepa21_vitl": 0, "dinov3_vitb": 1,
        }
        models = {
            "functional_vjepa2_vitl": _load_video_encoder(
                config, "functional_vjepa2_vitl", 0
            ),
            "dense_vjepa21_vitb": _load_video_encoder(
                config, "dense_vjepa21_vitb", 1
            ),
            "dense_vjepa21_vitl": _load_video_encoder(
                config, "dense_vjepa21_vitl", 0
            ),
            "dinov3_vitb": _load_dino(config, 1),
        }
        model_metadata = {
            name: {
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "device": devices[name],
            }
            for name, model in models.items()
        }
        for row, identifier, path in pending:
            try:
                features = _embed_row(row, config, models, devices)
                np.savez_compressed(path, **features)
                samples.append(
                    {
                        "sample_id": identifier,
                        "subject_id": str(row["subject_id"]),
                        "target": str(row["target"]),
                        "split": str(row["split"]),
                        "value": float(row["value"]),
                        **features,
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "sample_id": identifier, "target": row["target"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    metrics, winners = (
        _evaluate(samples, int(config["splits"]["seed"])) if samples else ({}, {})
    )
    expected = int(len(selected))
    checks = {
        "all_selected_samples_embedded": len(samples) == expected,
        "zero_embedding_errors": not errors,
        "four_candidates_compared": (
            all(
                set(endpoint["candidates"]) == set(ARCHITECTURES)
                for endpoint in metrics.values()
            )
            and len(metrics) == 6
        ),
        "patient_disjoint": all(
            endpoint["patient_disjoint"] for endpoint in metrics.values()
        ),
        "minimum_pilot_sample_size": all(
            endpoint["train_samples"] >= 24 and endpoint["validation_samples"] >= 12
            for endpoint in metrics.values()
        ) and len(metrics) == 6,
        "all_embeddings_finite": all(
            np.isfinite(sample[name]).all()
            for sample in samples
            for name in ARCHITECTURES
        ),
        "locked_test_not_accessed": True,
        "bf16_inference": True,
    }
    checkpoint_paths = {
        "functional_vjepa2_vitl": config["paths"]["functional_checkpoint"],
        "dense_vjepa21_vitb": config["paths"]["dense_vitb_checkpoint"],
        "dense_vjepa21_vitl": config["paths"]["dense_vitl_checkpoint"],
        "dinov3_vitb": config["paths"]["dinov3_checkpoint"],
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "track": "R", "locked_test_accessed": False,
        "selected_rows": expected, "embedded_rows": len(samples),
        "errors": errors, "cache_root": str(cache_root),
        "resumable_per_sample_cache": True,
        "precision": "BF16 inference; FP32 linear probes",
        "protocol_role": {
            "functional_vjepa2_vitl": "functional primary",
            "dense_vjepa21_vitb": "ViT-B search model",
            "dense_vjepa21_vitl": "ViT-L final dense candidate",
            "dinov3_vitb": "mandatory image challenger",
        },
        "checkpoint_sha256": {
            name: sha256_file(path) for name, path in checkpoint_paths.items()
        },
        "models": model_metadata,
        "targets": metrics,
        "pilot_winners": winners,
        "selection_note": "Pilot winners inform G5; they do not access the locked test set or replace multi-seed specialist selection.",
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    destination = (
        Path(config["paths"]["evidence_root"]) / "G4" / "architecture_ladder.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

