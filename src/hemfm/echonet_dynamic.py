from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import gc
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import threading
import time
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from .frozen_specialists import _atomic_json
from .gates import assert_through
from .hashing import sha256_file
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


SOURCE_URL = "https://echonet.github.io/dynamic/"
DATASET_ROOT = "EchoNet-Dynamic"
FILE_LIST_MEMBER = f"{DATASET_ROOT}/FileList.csv"
TRACINGS_MEMBER = f"{DATASET_ROOT}/VolumeTracings.csv"
TARGETS = ("EF", "ESV", "EDV")
DEVELOPMENT_SPLITS = ("TRAIN", "VAL")
ALL_SPLITS = (*DEVELOPMENT_SPLITS, "TEST")
_THREAD_LOCAL = threading.local()


def _archive_member(file_name: str) -> str:
    name = str(file_name).strip()
    if not name or Path(name).name != name or any(part == ".." for part in PurePosixPath(name).parts):
        raise ValueError(f"Unsafe EchoNet-Dynamic filename: {file_name!r}")
    if name.lower().endswith(".avi"):
        name = name[:-4]
    return f"{DATASET_ROOT}/Videos/{name}.avi"


def _sample_indices(frame_count: int, requested: int) -> np.ndarray:
    if frame_count < 1 or requested < 1:
        raise ValueError("frame counts must be positive")
    return np.rint(np.linspace(0, frame_count - 1, requested)).astype(int)


def _validated_manifest(frame: pd.DataFrame, members: set[str]) -> pd.DataFrame:
    required = [
        "FileName",
        "EF",
        "ESV",
        "EDV",
        "FrameHeight",
        "FrameWidth",
        "FPS",
        "NumberOfFrames",
        "Split",
    ]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"EchoNet-Dynamic FileList is missing columns: {missing_columns}")
    output = frame[required].copy()
    output["FileName"] = output["FileName"].astype(str).str.strip()
    output["Split"] = output["Split"].astype(str).str.strip().str.upper()
    if output["FileName"].duplicated().any():
        raise ValueError("EchoNet-Dynamic FileList contains duplicate filenames")
    unknown_splits = sorted(set(output["Split"]) - set(ALL_SPLITS))
    if unknown_splits:
        raise ValueError(f"EchoNet-Dynamic contains unknown splits: {unknown_splits}")
    development = output["Split"].isin(DEVELOPMENT_SPLITS)
    for column in TARGETS:
        output.loc[development, column] = pd.to_numeric(
            output.loc[development, column], errors="coerce"
        )
    for column in ("FrameHeight", "FrameWidth", "FPS", "NumberOfFrames"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if not np.isfinite(output.loc[development, list(TARGETS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("EchoNet-Dynamic contains missing or non-finite development targets")
    if (output[["FrameHeight", "FrameWidth", "FPS", "NumberOfFrames"]] <= 0).any().any():
        raise ValueError("EchoNet-Dynamic contains invalid video metadata")
    output["archive_member"] = output["FileName"].map(_archive_member)
    missing_videos = sorted(set(output["archive_member"]) - members)
    if missing_videos:
        raise ValueError(f"EchoNet-Dynamic archive is missing {len(missing_videos)} declared videos")
    return output.sort_values(["Split", "FileName"]).reset_index(drop=True)


def _read_archive(archive_path: Path) -> tuple[pd.DataFrame, set[str], int]:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError("EchoNet-Dynamic archive contains duplicate member paths")
        members = set(names)
        for required in (FILE_LIST_MEMBER, TRACINGS_MEMBER):
            if required not in members:
                raise ValueError(f"EchoNet-Dynamic archive is missing {required}")
        with archive.open(FILE_LIST_MEMBER) as handle:
            frame = pd.read_csv(handle)
        uncompressed_bytes = sum(item.file_size for item in infos)
    return _validated_manifest(frame, members), members, uncompressed_bytes


def _read_tracings(archive_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(TRACINGS_MEMBER) as handle:
            tracings = pd.read_csv(handle)
    required = ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
    missing = sorted(set(required) - set(tracings.columns))
    if missing:
        raise ValueError(f"EchoNet-Dynamic VolumeTracings is missing columns: {missing}")
    output = tracings[required].copy()
    output["FileName"] = (
        output["FileName"].astype(str).str.strip().str.replace(r"\.avi$", "", regex=True)
    )
    for column in ("X1", "Y1", "X2", "Y2", "Frame"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if output.isna().any().any() or not np.isfinite(
        output[["X1", "Y1", "X2", "Y2", "Frame"]].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("EchoNet-Dynamic volume tracings contain non-finite values")
    output["Frame"] = output["Frame"].astype(int)
    return output


def _rasterize_trace(rows: pd.DataFrame, width: int, height: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    left = list(zip(rows["X1"].astype(float), rows["Y1"].astype(float), strict=True))
    right = list(zip(rows["X2"].astype(float), rows["Y2"].astype(float), strict=True))
    polygon = [*left, *reversed(right)]
    if len(polygon) < 6:
        raise ValueError("EchoNet-Dynamic trace contains fewer than three boundary pairs")
    image = Image.new("L", (width, height), color=0)
    ImageDraw.Draw(image).polygon(polygon, fill=1)
    mask = np.asarray(image, dtype=np.uint8)
    if not mask.any():
        raise ValueError("EchoNet-Dynamic trace rasterized to an empty mask")
    return mask


def audit_echonet_dynamic_archive(
    config: dict[str, Any], archive_path: str | Path, *, compute_hash: bool = False
) -> dict[str, Any]:
    assert_through(config, "G4")
    source = Path(archive_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest, members, uncompressed_bytes = _read_archive(source)
    counts = {split: int(manifest["Split"].eq(split).sum()) for split in ALL_SPLITS}
    checks = {
        "official_metadata_present": True,
        "official_volume_tracings_present": TRACINGS_MEMBER in members,
        "all_declared_videos_present": len(manifest) == sum(counts.values()),
        "splits_recognized": set(manifest["Split"]) == set(ALL_SPLITS),
        "development_targets_finite": bool(
            np.isfinite(
                manifest.loc[manifest["Split"].isin(DEVELOPMENT_SPLITS), list(TARGETS)].to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "official_test_reserved": counts["TEST"] > 0,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "EchoNet-Dynamic archive structural and split audit",
        "source": SOURCE_URL,
        "licence": "Stanford research-use agreement; research use only unless separately licensed",
        "archive_bytes": source.stat().st_size,
        "archive_sha256": sha256_file(source) if compute_hash else None,
        "uncompressed_bytes": uncompressed_bytes,
        "records": len(manifest),
        "split_counts": counts,
        "checks": checks,
        "training_policy": "Only official TRAIN is optimized; VAL is used for development selection; official TEST is neither decoded nor trained on.",
        "official_test_videos_accessed": False,
        "locked_mimic_test_accessed": False,
        "track": "R",
    }
    destination = Path(config["paths"]["evidence_root"]) / "G4" / "echonet_dynamic_archive.json"
    _atomic_json(destination, report)
    return report


def _thread_archive(source: Path) -> zipfile.ZipFile:
    current = getattr(_THREAD_LOCAL, "archive", None)
    current_source = getattr(_THREAD_LOCAL, "source", None)
    if current is None or current_source != str(source):
        if current is not None:
            current.close()
        current = zipfile.ZipFile(source)
        _THREAD_LOCAL.archive = current
        _THREAD_LOCAL.source = str(source)
    return current


def _decode_member(
    source: Path,
    member: str,
    destination: Path,
    *,
    frames: int,
    resolution: int,
    tracing_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if destination.exists():
        try:
            with np.load(destination, allow_pickle=False) as cached:
                video = cached["video"]
                cache_version = int(cached.get("schema_version", np.asarray(1)))
                traced = cached.get("trace_frames")
                masks = cached.get("trace_masks")
            tracing_expected = tracing_rows is not None and not tracing_rows.empty
            tracing_valid = (
                not tracing_expected
                or (
                    traced is not None
                    and masks is not None
                    and traced.ndim == 4
                    and traced.shape[0] == masks.shape[0]
                    and masks.shape[1:] == (resolution, resolution)
                )
            )
            if (
                cache_version >= 2
                and video.shape == (3, frames, resolution, resolution)
                and video.dtype == np.uint8
                and tracing_valid
            ):
                return {
                    "created": False,
                    "bytes": destination.stat().st_size,
                    "trace_frames": int(0 if masks is None else masks.shape[0]),
                }
        except Exception:
            pass
    import av

    archive = _thread_archive(source)
    with archive.open(member) as handle:
        payload = handle.read()
    decoded = []
    with av.open(io.BytesIO(payload)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            decoded.append(frame)
    if not decoded:
        raise ValueError(f"No frames decoded from {member}")
    indices = _sample_indices(len(decoded), frames)
    selected = np.stack(
        [
            decoded[int(index)]
            .reformat(width=resolution, height=resolution, format="rgb24")
            .to_ndarray()
            for index in indices
        ]
    ).transpose(3, 0, 1, 2)
    trace_frames = np.empty((0, 3, resolution, resolution), dtype=np.uint8)
    trace_masks = np.empty((0, resolution, resolution), dtype=np.uint8)
    trace_indices = np.empty((0,), dtype=np.int32)
    if tracing_rows is not None and not tracing_rows.empty:
        frame_arrays = []
        mask_arrays = []
        indices = []
        for frame_index, group in tracing_rows.groupby("Frame", sort=True):
            frame_index = int(frame_index)
            if frame_index < 0 or frame_index >= len(decoded):
                raise ValueError(
                    f"Trace frame {frame_index} is outside decoded range 0..{len(decoded) - 1}"
                )
            source_height = int(decoded[frame_index].height)
            source_width = int(decoded[frame_index].width)
            mask = _rasterize_trace(group, source_width, source_height)
            from PIL import Image

            resized_mask = np.asarray(
                Image.fromarray(mask).resize((resolution, resolution), resample=Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
            frame_array = (
                decoded[frame_index]
                .reformat(width=resolution, height=resolution, format="rgb24")
                .to_ndarray()
                .transpose(2, 0, 1)
            )
            frame_arrays.append(frame_array)
            mask_arrays.append(resized_mask)
            indices.append(frame_index)
        trace_frames = np.stack(frame_arrays).astype(np.uint8)
        trace_masks = np.stack(mask_arrays).astype(np.uint8)
        trace_indices = np.asarray(indices, dtype=np.int32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(2, dtype=np.int16),
            video=selected.astype(np.uint8),
            trace_frames=trace_frames,
            trace_masks=trace_masks,
            trace_indices=trace_indices,
        )
    temporary.replace(destination)
    return {
        "created": True,
        "bytes": destination.stat().st_size,
        "trace_frames": int(trace_masks.shape[0]),
    }


def stage_echonet_dynamic(
    config: dict[str, Any],
    archive_path: str | Path,
    *,
    workers: int = 8,
    frames: int = 16,
    resolution: int = 224,
    limit: int | None = None,
) -> dict[str, Any]:
    source = Path(archive_path)
    full_stage = limit is None
    audit = audit_echonet_dynamic_archive(config, source, compute_hash=full_stage)
    if not audit["passed"]:
        raise RuntimeError("EchoNet-Dynamic archive audit failed")
    manifest, _, _ = _read_archive(source)
    tracings = _read_tracings(source)
    development = manifest[manifest["Split"].isin(DEVELOPMENT_SPLITS)].copy()
    if limit is not None:
        development = development.groupby("Split", group_keys=False).head(limit).copy()
    staging_root = (
        Path(config["paths"]["local_staging_root"])
        / "datasets"
        / "research"
        / "echonet-dynamic"
    )
    cache_root = staging_root / f"cache_{frames}x{resolution}"
    expected_raw = len(development) * 3 * frames * resolution * resolution
    free_bytes = shutil.disk_usage(staging_root.parent).free
    if free_bytes < expected_raw + 10 * 1024**3:
        raise RuntimeError(
            f"Insufficient local disk for EchoNet-Dynamic cache: free={free_bytes}, required={expected_raw + 10 * 1024**3}"
        )
    status_path = Path(config["paths"]["run_root"]) / "week_training" / "dataset_download_status_echonet_dynamic.json"
    tracing_groups = {
        str(file_name): group.copy()
        for file_name, group in tracings.groupby("FileName", sort=False)
        if file_name in set(development["FileName"])
    }
    jobs = []
    for row in development.itertuples(index=False):
        destination = cache_root / str(row.Split).lower() / f"{row.FileName}.npz"
        jobs.append((row.archive_member, destination, tracing_groups.get(str(row.FileName))))
    completed = 0
    created = 0
    bytes_total = 0
    traced_frames = 0
    errors: list[dict[str, str]] = []
    started = time.perf_counter()

    def write_status() -> None:
        _atomic_json(
            status_path,
            {
                "schema_version": 1,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": f"EchoNet-Dynamic development cache ({'full' if full_stage else 'smoke'})",
                "phase": "complete" if completed == len(jobs) and not errors else "decode_and_cache",
                "complete": completed,
                "total": len(jobs),
                "errors": len(errors),
                "official_test_videos_accessed": False,
                "locked_mimic_test_accessed": False,
            },
        )

    write_status()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _decode_member,
                source,
                member,
                destination,
                frames=frames,
                resolution=resolution,
                tracing_rows=tracing_rows,
            ): (member, destination)
            for member, destination, tracing_rows in jobs
        }
        for future in as_completed(futures):
            member, _ = futures[future]
            try:
                result = future.result()
                created += int(result["created"])
                bytes_total += int(result["bytes"])
                traced_frames += int(result["trace_frames"])
            except Exception as error:
                errors.append({"member": member, "error": f"{type(error).__name__}: {error}"})
            completed += 1
            if completed % 100 == 0 or completed == len(jobs):
                write_status()
                print(f"EchoNet-Dynamic cache {completed}/{len(jobs)} errors={len(errors)}", flush=True)
    development["cache_path"] = [
        str(cache_root / str(row.Split).lower() / f"{row.FileName}.npz")
        for row in development.itertuples(index=False)
    ]
    staging_root.mkdir(parents=True, exist_ok=True)
    manifest_path = staging_root / "development_manifest_private.csv"
    development.drop(columns=["archive_member"]).to_csv(manifest_path, index=False)
    counts = {split: int(development["Split"].eq(split).sum()) for split in DEVELOPMENT_SPLITS}
    stage_passed = not errors and completed == len(jobs)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": stage_passed,
        "scope": "EchoNet-Dynamic official TRAIN/VAL local cache",
        "mode": "full" if full_stage else "smoke",
        "training_ready": bool(stage_passed and full_stage and audit["archive_sha256"]),
        "source": SOURCE_URL,
        "archive_sha256": audit["archive_sha256"],
        "development_records": len(development),
        "split_counts": counts,
        "created_this_run": created,
        "cache_bytes": bytes_total,
        "trace_annotated_records": len(tracing_groups),
        "trace_frames_cached": traced_frames,
        "trace_cache_contents": "exact ED/ES RGB frames, rasterized LV masks, and original frame indices",
        "frames": frames,
        "resolution": resolution,
        "errors": errors[:100],
        "wall_seconds": round(time.perf_counter() - started, 3),
        "official_test_records_reserved": audit["split_counts"]["TEST"],
        "official_test_videos_accessed": False,
        "locked_mimic_test_accessed": False,
        "track": "R",
    }
    evidence_name = "echonet_dynamic_stage.json" if full_stage else "echonet_dynamic_stage_smoke.json"
    _atomic_json(Path(config["paths"]["evidence_root"]) / "G4" / evidence_name, report)
    write_status()
    return report


class EchoNetDynamicCacheDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        target_center: np.ndarray,
        target_scale: np…1321 tokens truncated…prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    sigma = np.concatenate(sigmas)
    metrics = {}
    for index, target_name in enumerate(TARGETS):
        error = np.abs(prediction[:, index] - target[:, index])
        metrics[target_name] = {
            "mae": float(error.mean()),
            "median_absolute_error": float(np.median(error)),
            "rmse": float(np.sqrt(np.mean((prediction[:, index] - target[:, index]) ** 2))),
            "mean_predicted_sigma": float(sigma[:, index].mean()),
        }
    return metrics, prediction, sigma, target, names


def _train_transfer_seed(
    config: dict[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    seed: int,
    device: int,
    training: StagedScalarConfig,
    destination: Path,
    maximum_train: int | None,
    maximum_validation: int | None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_dataset = EchoNetDynamicCacheDataset(
        train_frame,
        target_center=center,
        target_scale=scale,
        augment=True,
        maximum=maximum_train,
    )
    validation_dataset = EchoNetDynamicCacheDataset(
        validation_frame,
        target_center=center,
        target_scale=scale,
        augment=False,
        maximum=maximum_validation,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    model, depth, target_modules = _build_transfer_model(config, device)
    compact_names = _compact_names(model, depth)
    for name, parameter in model.named_parameters():
        if name in compact_names and not name.startswith(("head.", "mean.", "log_variance.")):
            parameter.data = parameter.data.float()
    destination.mkdir(parents=True, exist_ok=True)
    last_path = destination / "checkpoint_last.pt"
    best_path = destination / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    shadow = _compact_state(model, compact_names)
    start_epoch = 1
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=False)
        shadow = checkpoint["shadow"]
        best_state = checkpoint.get("best_state")
        best_score = float(checkpoint.get("best_score", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
    current_stage = None
    optimizer = None
    patience = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    status_path = Path(config["paths"]["run_root"]) / "week_training" / "echonet_dynamic_transfer_status.json"
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
        for step, batch in enumerate(train_loader, start=1):
            videos = batch["video"].to(device, non_blocking=True).to(dtype=torch.bfloat16)
            target = batch["target"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = _transfer_loss(model(videos), target, center, scale)
            (loss / training.accumulation).backward()
            losses.append(float(loss.detach().cpu()))
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
                        "phase": f"echonet_dynamic_{stage}",
                        "device": device,
                        "seed": seed,
                        "epoch": epoch,
                        "total_epochs": training.epochs,
                        "complete_sequences": min(step * 2, len(train_dataset)),
                        "total_sequences": len(train_dataset),
                        "official_test_videos_accessed": False,
                        "locked_test_accessed": False,
                    },
                )
        backup = _apply_shadow(model, shadow)
        metrics, prediction, sigma, target_units, names = _evaluate_transfer(
            model, validation_loader, device, center, scale
        )
        _restore(model, backup)
        score = float(np.mean([metrics[target]["mae"] / scale[index] for index, target in enumerate(TARGETS)]))
        improved = score < best_score - 1e-5
        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in shadow.items()}
            patience = 0
            torch.save(
                {
                    "model": best_state,
                    "seed": seed,
                    "epoch": epoch,
                    "target_center": center,
                    "target_scale": scale,
                    "metrics": metrics,
                    "source": SOURCE_URL,
                    "official_test_videos_accessed": False,
                    "locked_test_accessed": False,
                },
                best_path,
            )
            predictions = {"FileName": names}
            for index, target_name in enumerate(TARGETS):
                predictions[f"value_{target_name}"] = target_units[:, index]
                predictions[f"prediction_{target_name}"] = prediction[:, index]
                predictions[f"sigma_{target_name}"] = sigma[:, index]
            pd.DataFrame(predictions).to_csv(
                destination / "validation_predictions_private.csv", index=False
            )
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "stage": stage,
                "train_loss": float(np.mean(losses)),
                "validation": metrics,
                "selection_score": score,
            }
        )
        torch.save(
            {
                "model": _compact_state(model, compact_names),
                "shadow": {name: value.detach().cpu() for name, value in shadow.items()},
                "best_state": best_state,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "epoch": epoch,
                "history": history,
            },
            last_path,
        )
        if patience >= training.patience and stage == "selective_unfreeze":
            break
    report = {
        "passed": bool(np.isfinite(best_score) and best_state is not None),
        "seed": seed,
        "device": device,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_validation": history[best_epoch - 1]["validation"] if best_epoch else {},
        "epochs_completed": len(history),
        "history": history,
        "checkpoint_best": str(best_path),
        "checkpoint_last": str(last_path),
        "target_modules": target_modules,
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "official_test_videos_accessed": False,
        "locked_test_accessed": False,
    }
    _atomic_json(destination / "report.json", report)
    del model, shadow, best_state
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_echonet_dynamic_transfer(
    config: dict[str, Any],
    *,
    device: int,
    epochs: int = 10,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    assert_through(config, "G5")
    staging_root = (
        Path(config["paths"]["local_staging_root"])
        / "datasets"
        / "research"
        / "echonet-dynamic"
    )
    manifest_path = staging_root / "development_manifest_private.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run echonet-dynamic stage before transfer training")
    stage_name = "echonet_dynamic_stage.json" if mode == "full" else "echonet_dynamic_stage_smoke.json"
    stage_report = json.loads(
        (Path(config["paths"]["evidence_root"]) / "G4" / stage_name).read_text(encoding="utf-8")
    )
    if mode == "full" and not stage_report.get("training_ready"):
        raise RuntimeError("Full EchoNet-Dynamic cache is not hash-verified and complete")
    frame = pd.read_csv(manifest_path)
    if set(frame["Split"]) - set(DEVELOPMENT_SPLITS):
        raise RuntimeError("EchoNet-Dynamic cache manifest contains a reserved split")
    train_frame = frame[frame["Split"].eq("TRAIN")].copy().reset_index(drop=True)
    validation_frame = frame[frame["Split"].eq("VAL")].copy().reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise RuntimeError("EchoNet-Dynamic cache is missing TRAIN or VAL")
    center, scale = _target_statistics(train_frame)
    training = replace(
        StagedScalarConfig(),
        epochs=epochs,
        frozen_epochs=1,
        peft_epochs=max(1, min(4, epochs - 2)),
        accumulation=4,
        patience=2 if mode == "smoke" else 4,
    )
    seeds = [int(config["training"]["seeds"][0])] if mode == "smoke" else [
        int(seed) for seed in config["training"]["seeds"]
    ]
    root = Path(config["paths"]["run_root"]) / "echonet_dynamic_transfer"
    runs = []
    for seed in seeds:
        destination = root / f"seed_{seed}_{mode}"
        report_path = destination / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = _train_transfer_seed(
                config,
                train_frame,
                validation_frame,
                center=center,
                scale=scale,
                seed=seed,
                device=device,
                training=training,
                destination=destination,
                maximum_train=maximum_train,
                maximum_validation=maximum_validation,
            )
        runs.append(report)
    return finalize_echonet_dynamic_transfer(config, mode=mode)


def run_echonet_dynamic_transfer_seed(
    config: dict[str, Any],
    *,
    seed: int,
    device: int,
    epochs: int = 10,
    maximum_train: int | None = None,
    maximum_validation: int | None = None,
    mode: str = "full",
) -> dict[str, Any]:
    assert_through(config, "G5")
    configured_seeds = [int(value) for value in config["training"]["seeds"]]
    if seed not in configured_seeds:
        raise ValueError(f"Seed {seed} is not one of the prespecified seeds {configured_seeds}")
    staging_root = (
        Path(config["paths"]["local_staging_root"])
        / "datasets"
        / "research"
        / "echonet-dynamic"
    )
    manifest_path = staging_root / "development_manifest_private.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run echonet-dynamic stage before transfer training")
    stage_name = "echonet_dynamic_stage.json" if mode == "full" else "echonet_dynamic_stage_smoke.json"
    stage_report = json.loads(
        (Path(config["paths"]["evidence_root"]) / "G4" / stage_name).read_text(encoding="utf-8")
    )
    if mode == "full" and not stage_report.get("training_ready"):
        raise RuntimeError("Full EchoNet-Dynamic cache is not hash-verified and complete")
    frame = pd.read_csv(manifest_path)
    if set(frame["Split"]) - set(DEVELOPMENT_SPLITS):
        raise RuntimeError("EchoNet-Dynamic cache manifest contains a reserved split")
    train_frame = frame[frame["Split"].eq("TRAIN")].copy().reset_index(drop=True)
    validation_frame = frame[frame["Split"].eq("VAL")].copy().reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise RuntimeError("EchoNet-Dynamic cache is missing TRAIN or VAL")
    center, scale = _target_statistics(train_frame)
    training = replace(
        StagedScalarConfig(),
        epochs=epochs,
        frozen_epochs=1,
        peft_epochs=max(1, min(4, epochs - 2)),
        accumulation=4,
        patience=2 if mode == "smoke" else 4,
    )
    destination = (
        Path(config["paths"]["run_root"])
        / "echonet_dynamic_transfer"
        / f"seed_{seed}_{mode}"
    )
    report_path = destination / "report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return _train_transfer_seed(
        config,
        train_frame,
        validation_frame,
        center=center,
        scale=scale,
        seed=seed,
        device=device,
        training=training,
        destination=destination,
        maximum_train=maximum_train,
        maximum_validation=maximum_validation,
    )


def finalize_echonet_dynamic_transfer(
    config: dict[str, Any], *, mode: str = "full"
) -> dict[str, Any]:
    expected_seeds = (
        [int(config["training"]["seeds"][0])]
        if mode == "smoke"
        else [int(seed) for seed in config["training"]["seeds"]]
    )
    root = Path(config["paths"]["run_root"]) / "echonet_dynamic_transfer"
    runs = []
    prediction_frames = []
    for seed in expected_seeds:
        destination = root / f"seed_{seed}_{mode}"
        report_path = destination / "report.json"
        prediction_path = destination / "validation_predictions_private.csv"
        if not report_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"EchoNet-Dynamic seed {seed} is incomplete")
        runs.append(json.loads(report_path.read_text(encoding="utf-8")))
        prediction_frames.append(pd.read_csv(prediction_path))
    merged = prediction_frames[0][
        ["FileName", *[f"value_{target}" for target in TARGETS]]
    ].copy()
    for seed, frame in zip(expected_seeds, prediction_frames, strict=True):
        selected = frame[
            ["FileName", *[f"prediction_{target}" for target in TARGETS], *[f"sigma_{target}" for target in TARGETS]]
        ].copy()
        selected = selected.rename(
            columns={
                **{f"prediction_{target}": f"prediction_{target}_seed_{seed}" for target in TARGETS},
                **{f"sigma_{target}": f"sigma_{target}_seed_{seed}" for target in TARGETS},
            }
        )
        merged = merged.merge(selected, on="FileName", how="inner", validate="one_to_one")
    if len(merged) != len(prediction_frames[0]):
        raise RuntimeError("EchoNet-Dynamic seed predictions do not cover identical validation records")
    ensemble_metrics = {}
    for target in TARGETS:
        prediction_columns = [f"prediction_{target}_seed_{seed}" for seed in expected_seeds]
        sigma_columns = [f"sigma_{target}_seed_{seed}" for seed in expected_seeds]
        predictions = merged[prediction_columns].to_numpy(dtype=np.float64)
        sigmas = merged[sigma_columns].to_numpy(dtype=np.float64)
        merged[f"prediction_{target}"] = predictions.mean(axis=1)
        merged[f"sigma_{target}"] = np.sqrt(
            np.mean(sigmas**2, axis=1) + np.var(predictions, axis=1)
        )
        values = merged[f"value_{target}"].to_numpy(dtype=np.float64)
        estimate = merged[f"prediction_{target}"].to_numpy(dtype=np.float64)
        absolute_error = np.abs(estimate - values)
        ensemble_metrics[target] = {
            "mae": float(absolute_error.mean()),
            "median_absolute_error": float(np.median(absolute_error)),
            "rmse": float(np.sqrt(np.mean((estimate - values) ** 2))),
            "mean_predicted_sigma": float(merged[f"sigma_{target}"].mean()),
        }
    merged.to_csv(root / f"validation_ensemble_{mode}_private.csv", index=False)
    checks = {
        "official_train_validation_only": True,
        "three_seed_full_run": mode != "full" or len(runs) == len(expected_seeds) == 3,
        "all_runs_finite": all(run["passed"] for run in runs),
        "frozen_peft_selective_schedule": all(
            {row["stage"] for row in run["history"]}
            >= ({"frozen", "peft"} if mode == "smoke" else {"frozen", "peft", "selective_unfreeze"})
            for run in runs
        ),
        "official_test_not_accessed": True,
        "locked_mimic_test_not_accessed": True,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "EchoNet-Dynamic EF/ESV/EDV transfer pretraining",
        "mode": mode,
        "checks": checks,
        "training": {
            "epochs": max((run.get("epochs_completed", 0) for run in runs), default=0),
            "schedule": ["frozen", "peft", "selective_unfreeze"],
            "precision": "BF16 activations with FP32 trainable parameters",
        },
        "validation_records": len(merged),
        "seeds": runs,
        "ensemble_validation": ensemble_metrics,
        "promotion_eligible": False,
        "promotion_policy": "This external A4C model is a transfer initializer only. It must beat the MIMIC patient-disjoint development route after MIMIC fine-tuning before promotion.",
        "official_test_videos_accessed": False,
        "locked_test_accessed": False,
        "track": "R",
    }
    evidence_name = f"echonet_dynamic_transfer_{mode}.json"
    _atomic_json(Path(config["paths"]["evidence_root"]) / "G5" / evidence_name, report)
    return report
