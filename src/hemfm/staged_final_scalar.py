from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from .dicom_preprocess import preprocess_dicom_cine
from .gates import assert_through
from .hashing import sha256_file


@dataclass(frozen=True)
class StagedScalarConfig:
    epochs: int = 10
    frozen_epochs: int = 1
    peft_epochs: int = 4
    accumulation: int = 8
    head_learning_rate: float = 2e-4
    adapter_learning_rate: float = 8e-5
    selective_learning_rate: float = 2e-6
    weight_decay: float = 1e-2
    ema_decay: float = 0.995
    patience: int = 4
    frames: int = 16
    resolution: int = 224


def stage_for_epoch(epoch: int, config: StagedScalarConfig) -> str:
    if epoch <= config.frozen_epochs:
        return "frozen"
    if epoch <= config.frozen_epochs + config.peft_epochs:
        return "peft"
    return "selective_unfreeze"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _cohort(config: dict[str, Any], target: str | None = None) -> pd.DataFrame:
    path = Path(config["paths"]["private_root"]) / "endpoint_cohorts_development_private.csv"
    rows = pd.read_csv(
        path, dtype={"subject_id": str, "study_id": str}, low_memory=False
    )
    rows = rows[rows["split"].isin(["train", "validation"])].copy()
    if target is not None:
        rows = rows[rows["target"] == target].copy()
    if rows.empty:
        raise RuntimeError(f"No development rows found for target={target!r}")
    if rows["test_locked"].fillna(False).astype(bool).any():
        raise RuntimeError("The staged final cohort contains a locked test row")
    train_patients = set(rows.loc[rows["split"].eq("train"), "subject_id"])
    validation_patients = set(
        rows.loc[rows["split"].eq("validation"), "subject_id"]
    )
    if train_patients & validation_patients:
        raise RuntimeError("Patient leakage detected in the staged final cohort")
    return rows.reset_index(drop=True)


def _calibration(target: str) -> str:
    return "spectral" if target == "AV_PEAK_VELOCITY" else "spatial"


def _cache_path(cache_root: Path, relative_path: str, calibration: str) -> Path:
    payload = f"{relative_path.replace(chr(92), '/')}|{calibration}|16|224"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return cache_root / digest[:2] / f"{digest}.npz"


def _cine_jobs(rows: pd.DataFrame, cache_root: Path) -> list[dict[str, str]]:
    jobs: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows.itertuples(index=False):
        calibration = _calibration(str(row.target))
        for cine in json.loads(row.cines):
            relative_path = str(cine["path"])
            key = (relative_path, calibration)
            jobs[key] = {
                "relative_path": relative_path,
                "calibration": calibration,
                "cache_path": str(_cache_path(cache_root, relative_path, calibration)),
            }
    return list(jobs.values())


def _write_cached_cine(job: dict[str, str], dicom_root: Path) -> dict[str, Any]:
    destination = Path(job["cache_path"])
    if destination.exists():
        return {"cached": True, "created": False, "bytes": destination.stat().st_size}
    source = dicom_root.joinpath(
        *job["relative_path"].replace("\\", "/").split("/")
    )
    result = preprocess_dicom_cine(
        source,
        calibration_type=job["calibration"],
        frames=16,
        output_size=224,
        clean_overlays=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, video=result.video.astype(np.uint8))
    temporary.replace(destination)
    return {"cached": True, "created": True, "bytes": destination.stat().st_size}


def stage_staged_scalar_cache(
    config: dict[str, Any], *, workers: int = 12, limit: int | None = None
) -> dict[str, Any]:
    assert_through(config, "G5")
    rows = _cohort(config)
    cache_root = Path(config["paths"]["private_root"]) / "staged_scalar_video_cache"
    jobs = _cine_jobs(rows, cache_root)
    if limit is not None:
        jobs = jobs[:limit]
    status_path = Path(config["paths"]["run_root"]) / "staged_final" / "cache_status.json"
    complete = 0
    created = 0
    errors: list[dict[str, str]] = []
    bytes_total = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _write_cached_cine, job, Path(config["paths"]["dicom_root"])
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                created += int(result["created"])
                bytes_total += int(result["bytes"])
            except Exception as exc:
                errors.append(
                    {
                        "cache_key": Path(job["cache_path"]).stem,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            complete += 1
            if complete == len(jobs) or complete % 25 == 0:
                status = {
                    "schema_version": 1,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "phase": "staging_local_video_cache",
                    "complete": complete,
                    "total": len(jobs),
                    "created": created,
                    "errors": len(errors),
                    "locked_test_accessed": False,
                }
                _atomic_json(status_path, status)
                print(json.dumps(status), flush=True)
    checks = {
        "development_rows_only": True,
        "patient_disjoint": True,
        "all_requested_cines_cached": complete == len(jobs) and not errors,
        "local_mutable_cache": True,
        "locked_test_not_accessed": True,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "development_rows": len(rows),
        "unique_cines": len(jobs),
        "created_this_run": created,
        "cache_bytes": bytes_total,
        "errors": errors[:100],
        "wall_seconds": round(time.perf_counter() - started, 3),
        "locked_test_accessed": False,
    }
    evidence = Path(config["paths"]["evidence_root"]) / "G5" / "staged_scalar_cache.json"
    _atomic_json(evidence, report)
    return report


def _sample_cache_paths(
    row: Any, cache_root: Path
) -> list[Path]:
    calibration = _calibration(str(row.target))
    return [
        _cache_path(cache_root, str(cine["path"]), calibration)
        for cine in json.loads(row.cines)
    ]


def _load_samples(
    rows: pd.DataFrame,
    cache_root: Path,
    *,
    dicom_root: Path,
    maximum_train: int | None,
    maximum_validation: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    for split, maximum in (
        ("train", maximum_train),
        ("validation", maximum_validation),
    ):
        subset = rows[rows["split"].eq(split)].copy()
        if maximum is not None and len(subset) > maximum:
            subset = subset.sample(n=maximum, random_state=seed).sort_index()
        selected.append(subset)
    frame = pd.concat(selected).sort_index()
    samples: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        paths = _sample_cache_paths(row, cache_root)
        calibration = _calibration(str(row.target))
        for cine, path in zip(json.loads(row.cines), paths, strict=True):
            if not path.exists():
                _write_cached_cine(
                    {
                        "relative_path": str(cine["path"]),
                        "calibration": calibration,
                        "cache_path": str(path),
                    },
                    dicom_root,
                )
        videos = np.stack(
            [np.load(path, allow_pickle=False)["video"] for path in paths]
        ).astype(np.uint8)
        samples.append(
            {
                "sample_id": hashlib.sha256(
                    f"{row.subject_id}|{row.study_id}|{row.target}|{row.cines}".encode("utf-8")
                ).hexdigest()[:20],
                "subject_id": str(row.subject_id),
                "study_id": str(row.study_id),
                "split": str(row.split),
                "value": float(row.value),
                "videos": videos,
            }
        )
    return (
        [sample for sample in samples if sample["split"] == "train"],
        [sample for sample in samples if sample["split"] == "validation"],
    )


def _load_encoder(config: dict[str, Any], device: int):
    import torch
    from peft import LoraConfig, get_peft_model

    source = Path(config["paths"]["echojepa_source"])
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from src.models.vision_transformer import vit_large

    encoder = vit_large(
        img_size=(224, 224),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        use_rope=True,
        uniform_power=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
    )
    checkpoint = Path(config["paths"]["functional_checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weights = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state["encoder"].items()
    }
    incompatible = encoder.load_state_dict(weights, strict=False)
    missing = [key for key in incompatible.missing_keys if key != "pos_embed"]
    unexpected = [key for key in incompatible.unexpected_keys if key != "pos_embed"]
    del state, weights
    if missing or unexpected:
        raise RuntimeError(
            f"Functional ViT-L checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    depth = len(encoder.blocks)
    target_modules = [
        f"blocks.{block}.{module}"
        for block in range(depth - 4, depth)
        for module in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    ]
    encoder = get_peft_model(
        encoder,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            use_dora=True,
        ),
    )
    return encoder.to(device=device, dtype=torch.bfloat16), depth, target_modules


def _build_model(config: dict[str, Any], device: int):
    import torch

    encoder, depth, target_modules = _load_encoder(config, device)

    class ScalarModel(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.encoder = backbone
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(1024),
                torch.nn.Linear(1024, 256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.15),
            )
            self.mean = torch.nn.Linear(256, 1)
            self.log_variance = torch.nn.Linear(256, 1)

        def forward(self, videos):
            tokens = self.encoder(videos)
            if isinstance(tokens, list):
                tokens = tokens[-1]
            features = tokens.float().mean(dim=1).mean(dim=0, keepdim=True)
            hidden = self.head(features)
            return (
                self.mean(hidden).squeeze(),
                self.log_variance(hidden).squeeze().clamp(-6.0, 4.0),
            )

    model = ScalarModel(encoder).to(device)
    model.head.to(dtype=torch.float32)
    model.mean.to(dtype=torch.float32)
    model.log_variance.to(dtype=torch.float32)
    return model, depth, target_modules


def _compact_names(model, depth: int) -> set[str]:
    final_block = f".blocks.{depth - 1}."
    names = set()
    for name, _ in model.named_parameters():
        if (
            name.startswith(("head.", "mean.", "log_variance."))
            or "lora_" in name
            or final_block in name
            or name.endswith(".norm.weight")
            or name.endswith(".norm.bias")
        ):
            names.add(name)
    return names


def _set_stage(model, stage: str, depth: int) -> None:
    final_block = f".blocks.{depth - 1}."
    for name, parameter in model.named_parameters():
        active = name.startswith(("head.", "mean.", "log_variance."))
        if stage in {"peft", "selective_unfreeze"} and "lora_" in name:
            active = True
        if stage == "selective_unfreeze" and (
            final_block in name
            or name.endswith(".norm.weight")
            or name.endswith(".norm.bias")
        ):
            active = True
        parameter.requires_grad_(active)


def _optimizer(model, stage: str, depth: int, training: StagedScalarConfig):
    import torch

    final_block = f".blocks.{depth - 1}."
    groups: dict[str, list[Any]] = {"head": [], "adapter": [], "selective": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("head.", "mean.", "log_variance.")):
            groups["head"].append(parameter)
        elif "lora_" in name:
            groups["adapter"].append(parameter)
        elif final_block in name or name.endswith((".norm.weight", ".norm.bias")):
            groups["selective"].append(parameter)
    parameter_groups = []
    for name, learning_rate in (
        ("head", training.head_learning_rate),
        ("adapter", training.adapter_learning_rate),
        ("selective", training.selective_learning_rate),
    ):
        if groups[name]:
            parameter_groups.append({"params": groups[name], "lr": learning_rate})
    if not parameter_groups:
        raise RuntimeError(f"No trainable parameters for stage {stage}")
    return torch.optim.AdamW(
        parameter_groups, weight_decay=training.weight_decay
    )


def _normalized_videos(array: np.ndarray, device: int):
    import torch

    videos = torch.from_numpy(array).to(device=device, dtype=torch.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
    return ((videos - mean) / std).to(dtype=torch.bfloat16)


def _apply_shadow(model, shadow: dict[str, Any]) -> dict[str, Any]:
    backup = {}
    with __import__("torch").no_grad():
        for name, parameter in model.named_parameters():
            if name in shadow:
                backup[name] = parameter.detach().clone()
                parameter.copy_(shadow[name])
    return backup


def _restore(model, backup: dict[str, Any]) -> None:
    with __import__("torch").no_grad():
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.copy_(backup[name])


def _evaluate(
    model,
    samples: list[dict[str, Any]],
    target_mean: float,
    target_std: float,
    device: int,
) -> tuple[dict[str, float], np.ndarray]:
    import torch

    predictions = []
    sigmas = []
    targets = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            videos = _normalized_videos(sample["videos"], device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mean, log_variance = model(videos)
            predictions.append(float(mean) * target_std + target_mean)
            sigmas.append(math.sqrt(math.exp(float(log_variance))) * target_std)
            targets.append(sample["value"])
    prediction = np.asarray(predictions)
    target = np.asarray(targets)
    error = np.abs(prediction - target)
    return (
        {
            "mae": float(error.mean()),
            "median_ae": float(np.median(error)),
            "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
            "mean_predicted_sigma": float(np.mean(sigmas)),
        },
        prediction,
    )


def _compact_state(model, names: set[str]) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in names
    }


def _train_seed(
    config: dict[str, Any],
    target: str,
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    *,
    seed: int,
    device: int,
    training: StagedScalarConfig,
    destination: Path,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, depth, target_modules = _build_model(config, device)
    compact_names = _compact_names(model, depth)
    for name, parameter in model.named_parameters():
        if name in compact_names and not name.startswith(("head.", "mean.", "log_variance.")):
            parameter.data = parameter.data.float()
    target_mean = float(np.mean([sample["value"] for sample in train]))
    target_std = float(np.std([sample["value"] for sample in train])) or 1.0
    destination.mkdir(parents=True, exist_ok=True)
    last_path = destination / "checkpoint_last.pt"
    best_path = destination / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []
    best_mae = math.inf
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    shadow = _compact_state(model, compact_names)
    start_epoch = 1
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=False)
        shadow = checkpoint["shadow"]
        best_state = checkpoint.get("best_state")
        best_mae = float(checkpoint.get("best_mae", math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
    patience = 0
    current_stage = None
    optimizer = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, training.epochs + 1):
        stage = stage_for_epoch(epoch, training)
        if stage != current_stage:
            _set_stage(model, stage, depth)
            optimizer = _optimizer(model, stage, depth, training)
            current_stage = stage
        assert optimizer is not None
        model.train()
        order = list(range(len(train)))
        random.Random(seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, index in enumerate(order, start=1):
            sample = train[index]
            videos = _normalized_videos(sample["videos"], device)
            target_value = torch.tensor(
                (sample["value"] - target_mean) / target_std,
                device=device,
                dtype=torch.float32,
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mean, log_variance = model(videos)
                nll = 0.5 * (
                    torch.exp(-log_variance.float())
                    * (mean.float() - target_value).square()
                    + log_variance.float()
                )
                loss = 0.7 * functional.smooth_l1_loss(mean.float(), target_value) + 0.3 * nll
            (loss / training.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if step % training.accumulation == 0 or step == len(order):
                active = [parameter for parameter in model.parameters() if parameter.requires_grad]
                torch.nn.utils.clip_grad_norm_(active, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name in shadow and parameter.requires_grad:
                            shadow[name] = shadow[name].to(parameter.device, dtype=parameter.dtype)
                            shadow[name].lerp_(parameter.detach(), 1.0 - training.ema_decay)
        backup = _apply_shadow(model, shadow)
        metrics, prediction = _evaluate(
            model, validation, target_mean, target_std, device
        )
        _restore(model, backup)
        improved = metrics["mae"] < best_mae - 1e-5
        if improved:
            best_mae = metrics["mae"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in shadow.items()}
            patience = 0
            torch.save(
                {
                    "model": best_state,
                    "target": target,
                    "seed": seed,
                    "epoch": epoch,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "metrics": metrics,
                    "track": "R",
                    "locked_test_accessed": False,
                },
                best_path,
            )
            pd.DataFrame(
                {
                    "sample_id": [sample["sample_id"] for sample in validation],
                    "subject_id": [sample["subject_id"] for sample in validation],
                    "target": target,
                    "seed": seed,
                    "value": [sample["value"] for sample in validation],
                    "prediction": prediction,
                }
            ).to_csv(destination / "validation_predictions_private.csv", index=False)
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "train_loss": float(np.mean(losses)),
                "validation": metrics,
            }
        )
        torch.save(
            {
                "model": _compact_state(model, compact_names),
                "shadow": {name: value.detach().cpu() for name, value in shadow.items()},
                "best_state": best_state,
                "best_mae": best_mae,
                "best_epoch": best_epoch,
                "epoch": epoch,
                "history": history,
                "target_mean": target_mean,
                "target_std": target_std,
            },
            last_path,
        )
        status = {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": stage,
            "target": target,
            "seed": seed,
            "device": device,
            "epoch": epoch,
            "epochs": training.epochs,
            "best_validation_mae": best_mae,
            "locked_test_accessed": False,
        }
        _atomic_json(destination / "status.json", status)
        print(json.dumps(status), flush=True)
        if patience >= training.patience and stage == "selective_unfreeze":
            break
    report = {
        "passed": bool(np.isfinite(best_mae) and best_state is not None),
        "target": target,
        "seed": seed,
        "device": device,
        "best_epoch": best_epoch,
        "best_validation_mae": best_mae,
        "epochs_completed": len(history),
        "history": history,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "target_modules": target_modules,
        "trainable_compact_parameters": int(
            sum(parameter.numel() for name, parameter in model.named_parameters() if name in compact_names)
        ),
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "locked_test_accessed": False,
    }
    _atomic_json(destination / "report.json", report)
    del model, shadow, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_staged_scalar_target(
    config: dict[str, Any],
    *,
    target: str,
    device: int,
    epochs: int = 10,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    assert_through(config, "G5")
    if target not in config["targets"]:
        raise ValueError(f"Unknown target: {target}")
    rows = _cohort(config, target)
    cache_root = Path(config["paths"]["private_root"]) / "staged_scalar_video_cache"
    train, validation = _load_samples(
        rows,
        cache_root,
        dicom_root=Path(config["paths"]["dicom_root"]),
        maximum_train=maximum_train,
        maximum_validation=maximum_validation,
        seed=int(config["splits"]["seed"]),
    )
    if not train or not validation:
        raise RuntimeError(f"{target} is missing a development split")
    if {sample["subject_id"] for sample in train} & {
        sample["subject_id"] for sample in validation
    }:
        raise RuntimeError(f"{target} staged training has patient leakage")
    training = StagedScalarConfig(
        epochs=epochs,
        frozen_epochs=1,
        peft_epochs=max(1, min(4, epochs - 2)),
        patience=2 if mode == "smoke" else 4,
    )
    root = Path(config["paths"]["run_root"]) / "staged_final" / target
    seeds = [int(config["training"]["seeds"][0])] if mode == "smoke" else [
        int(seed) for seed in config["training"]["seeds"]
    ]
    runs = []
    for seed in seeds:
        destination = root / f"seed_{seed}_{mode}"
        report_path = destination / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = _train_seed(
                config,
                target,
                train,
                validation,
                seed=seed,
                device=device,
                training=training,
                destination=destination,
            )
        runs.append(report)
    prediction_frames = [
        pd.read_csv(root / f"seed_{seed}_{mode}" / "validation_predictions_private.csv")
        for seed in seeds
    ]
    merged = prediction_frames[0][["sample_id", "subject_id", "target", "value"]].copy()
    for index, frame in enumerate(prediction_frames):
        merged[f"prediction_seed_{seeds[index]}"] = frame["prediction"].to_numpy()
    prediction_columns = [column for column in merged if column.startswith("prediction_seed_")]
    merged["prediction"] = merged[prediction_columns].mean(axis=1)
    ensemble_mae = float(np.mean(np.abs(merged["prediction"] - merged["value"])))
    merged.to_csv(root / f"validation_ensemble_{mode}_private.csv", index=False)
    baseline = json.loads(
        (Path(config["paths"]["evidence_root"]) / "G6" / "specialist_holdouts.json").read_text(
            encoding="utf-8"
        )
    )["targets"][target]["internal_validation_mae"]
    checks = {
        "functional_vitl_checkpoint_verified": sha256_file(config["paths"]["functional_checkpoint"])
        == config["assets"]["functional_checkpoint_sha256"],
        "patient_disjoint": True,
        "three_seed_full_run": mode != "full" or len(runs) == 3,
        "frozen_peft_selective_schedule": all(
            {row["stage"] for row in run["history"]}
            >= ({"frozen", "peft"} if mode == "smoke" else {"frozen", "peft", "selective_unfreeze"})
            for run in runs
        ),
        "all_runs_finite": all(run["passed"] for run in runs),
        "locked_test_not_accessed": True,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": f"{target} development-only staged ViT-L scalar challenger",
        "mode": mode,
        "checks": checks,
        "training": training.__dict__,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "seeds": runs,
        "ensemble_validation_mae": ensemble_mae,
        "current_g6_baseline_mae": float(baseline),
        "mae_change_vs_current_g6": ensemble_mae - float(baseline),
        "promotion_eligible": bool(ensemble_mae < float(baseline)),
        "track": "R",
        "locked_test_accessed": False,
    }
    evidence = (
        Path(config["paths"]["evidence_root"])
        / "G6"
        / f"staged_final_{target.lower()}_{mode}.json"
    )
    _atomic_json(evidence, report)
    return report

