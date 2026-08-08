from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from .dicom_preprocess import preprocess_dicom_cine
from .hashing import sha256_file
from .shortcut_controls import _select_rows


VARIANTS = {
    "lora_adamw": {"adapter": "lora", "optimizer": "adamw"},
    "lora_muon": {"adapter": "lora", "optimizer": "muon"},
    "rslora_adamw": {"adapter": "rslora", "optimizer": "adamw"},
    "dora_adamw": {"adapter": "dora", "optimizer": "adamw"},
    "pissa_adamw": {"adapter": "pissa", "optimizer": "adamw"},
}


def _identifier(row: dict[str, Any]) -> str:
    value = f"{row['subject_id']}|{row['study_id']}|{row['target']}|{row['cines']}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _target_modules() -> list[str]:
    return [
        f"blocks.{block}.{module}"
        for block in (10, 11)
        for module in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    ]


def _prepare_video_cache(
    rows: pd.DataFrame, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    cache_root = Path(config["paths"]["private_root"]) / "g5_video_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    dicom_root = Path(config["paths"]["dicom_root"])
    samples = []
    errors = []
    for row_tuple in rows.itertuples(index=False):
        row = row_tuple._asdict()
        identifier = _identifier(row)
        cache_path = cache_root / f"{identifier}.npz"
        try:
            if cache_path.exists():
                videos = np.load(cache_path, allow_pickle=False)["videos"]
            else:
                videos = []
                for cine in json.loads(row["cines"]):
                    path = dicom_root.joinpath(
                        *str(cine["path"]).replace("\\", "/").split("/")
                    )
                    videos.append(
                        preprocess_dicom_cine(
                            path, calibration_type="spatial", frames=16,
                            output_size=224, clean_overlays=True,
                        ).video
                    )
                videos = np.stack(videos).astype(np.uint8)
                np.savez_compressed(cache_path, videos=videos)
            samples.append(
                {
                    "sample_id": identifier,
                    "subject_id": str(row["subject_id"]),
                    "split": str(row["split"]),
                    "value": float(row["value"]),
                    "videos": videos,
                }
            )
        except Exception as exc:
            errors.append(
                {"sample_id": identifier, "error": f"{type(exc).__name__}: {exc}"}
            )
    return samples, errors


def _load_backbone(config: dict[str, Any], adapter: str, device: int):
    import torch
    from peft import LoraConfig, get_peft_model

    source = Path(config["paths"]["vjepa21_source"])
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from app.vjepa_2_1.models.vision_transformer import vit_base

    model = vit_base(
        img_size=(224, 224), patch_size=16, num_frames=16, tubelet_size=2,
        use_rope=True, uniform_power=True, use_sdpa=True, img_temporal_dim_size=1,
        interpolate_rope=True, modality_embedding=True, n_registers=0,
        has_cls_first=False, use_activation_checkpointing=False,
    )
    model.return_hierarchical = True
    checkpoint = Path(config["paths"]["dense_vitb_checkpoint"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
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
    config_kwargs: dict[str, Any] = {
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.05,
        "target_modules": _target_modules(),
        "bias": "none",
    }
    if adapter == "rslora":
        config_kwargs["use_rslora"] = True
    elif adapter == "dora":
        config_kwargs["use_dora"] = True
    elif adapter == "pissa":
        config_kwargs["init_lora_weights"] = "pissa_niter_2"
    model = get_peft_model(model, LoraConfig(**config_kwargs))
    model = model.to(device=device, dtype=torch.bfloat16)
    # Keep FP32 master adapter weights while autocast performs BF16 tensor-core
    # compute, matching the mixed-precision contract used for the probe head.
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model


def _normalized_videos(array: np.ndarray, device: int):
    import torch

    videos = torch.from_numpy(array).to(device=device, dtype=torch.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
    return ((videos - mean) / std).to(dtype=torch.bfloat16)


def _make_optimizer(model, kind: str):
    import torch

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if kind == "adamw":
        return [torch.optim.AdamW(trainable, lr=5e-4, weight_decay=1e-2)]
    matrices = [parameter for parameter in trainable if parameter.ndim == 2]
    auxiliary = [parameter for parameter in trainable if parameter.ndim != 2]
    optimizers = [
        torch.optim.Muon(
            matrices, lr=5e-4, weight_decay=1e-2,
            momentum=0.95, adjust_lr_fn="match_rms_adamw",
        )
    ]
    if auxiliary:
        optimizers.append(torch.optim.AdamW(auxiliary, lr=5e-4, weight_decay=1e-2))
    return optimizers


def _ema_update(shadow: dict[str, Any], model, decay: float = 0.98) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            shadow[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)


def _evaluate(model, samples: list[dict[str, Any]], target_mean: float, target_std: float, device: int) -> float:
    import torch

    predictions = []
    values = []
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            videos = _normalized_videos(sample["videos"], device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(videos).float().item() * target_std + target_mean
            predictions.append(prediction)
            values.append(sample["value"])
    return float(np.mean(np.abs(np.asarray(predictions) - np.asarray(values))))


def _train_variant(
    variant_name: str,
    variant: dict[str, str],
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    config: dict[str, Any],
    epochs: int,
    seed: int,
    device: int,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backbone = _load_backbone(config, variant["adapter"], device)

    class Regressor(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(3072, 1)

        def forward(self, videos):
            tokens = self.encoder(videos)
            if isinstance(tokens, list):
                tokens = tokens[-1]
            features = tokens.float().mean(dim=1).mean(dim=0, keepdim=True)
            return self.head(features).squeeze()

    model = Regressor(backbone).to(device)
    model.head.to(dtype=torch.float32)
    optimizers = _make_optimizer(model, variant["optimizer"])
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    shadow = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    target_mean = float(np.mean([sample["value"] for sample in train]))
    target_std = float(np.std([sample["value"] for sample in train])) or 1.0
    history = []
    best_mae = float("inf")
    best_epoch = 0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    accumulation = 4
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.Random(seed + epoch).shuffle(order)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, index in enumerate(order, start=1):
            sample = train[index]
            videos = _normalized_videos(sample["videos"], device)
            target = torch.tensor(
                (sample["value"] - target_mean) / target_std,
                device=device, dtype=torch.float32,
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(videos)
                loss = functional.smooth_l1_loss(prediction, target)
            (loss / accumulation).backward()
            losses.append(float(loss.detach().cpu()))
            if step % accumulation == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                _ema_update(shadow, model)
        backup = {name: parameter.detach().clone() for name, parameter in trainable.items()}
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(shadow[name])
        validation_mae = _evaluate(model, validation, target_mean, target_std, device)
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(backup[name])
        if validation_mae < best_mae:
            best_mae = validation_mae
            best_epoch = epoch
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "ema_validation_mae": validation_mae}
        )
    report = {
        "passed": bool(np.isfinite(best_mae)),
        "adapter": variant["adapter"],
        "optimizer": variant["optimizer"],
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable.values())),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_ema_validation_mae": best_mae,
        "best_epoch": best_epoch,
        "epochs": epochs,
        "history": history,
        "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "seed": seed,
    }
    del model, backbone, optimizers, shadow
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_optimizer_peft_pilot(
    config: dict[str, Any],
    train_count: int = 24,
    validation_count: int = 12,
    epochs: int = 3,
    device: int = 0,
) -> dict[str, Any]:
    cohort = pd.read_csv(
        Path(config["paths"]["private_root"]) / "endpoint_cohorts_development_private.csv",
        dtype={"subject_id": str, "study_id": str}, low_memory=False,
    )
    cohort = cohort[cohort["target"] == "EF"].copy()
    if not set(cohort["split"]).issubset({"train", "validation"}) or cohort["test_locked"].fillna(False).astype(bool).any():
        raise RuntimeError("G5 pilot received a locked test row")
    selected = _select_rows(
        cohort, {"train": train_count, "validation": validation_count},
        int(config["splits"]["seed"]) + 29,
    )
    samples, errors = _prepare_video_cache(selected, config)
    train = [sample for sample in samples if sample["split"] == "train"]
    validation = [sample for sample in samples if sample["split"] == "validation"]
    results = {}
    for offset, (name, variant) in enumerate(VARIANTS.items()):
        results[name] = _train_variant(
            name, variant, train, validation, config, epochs,
            int(config["splits"]["seed"]) + 100 + offset, device,
        )
    checks = {
        "all_variants_completed": set(results) == set(VARIANTS) and all(row["passed"] for row in results.values()),
        "patient_disjoint": not bool({sample["subject_id"] for sample in train} & {sample["subject_id"] for sample in validation}),
        "minimum_samples": len(train) >= 24 and len(validation) >= 12,
        "zero_decode_errors": not errors,
        "locked_test_not_accessed": True,
        "bf16_encoder_training": True,
        "ema_evaluation": True,
        "fp8_not_used": True,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "representative_endpoint": "EF",
        "backbone": "EchoJEPA V-JEPA2.1 ViT-B",
        "checkpoint_sha256": sha256_file(config["paths"]["dense_vitb_checkpoint"]),
        "target_modules": _target_modules(),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "decode_errors": errors,
        "locked_test_accessed": False,
        "results": results,
    }
    evidence_root = Path(config["paths"]["evidence_root"]) / "G5"
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "optimizer_peft_pilot.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def write_decision_ledger(config: dict[str, Any], pilot: dict[str, Any]) -> dict[str, Any]:
    architecture = json.loads(
        (Path(config["paths"]["evidence_root"]) / "G4" / "architecture_ladder.json").read_text(encoding="utf-8")
    )
    results = pilot["results"]
    adamw_adapters = [name for name, row in results.items() if row["optimizer"] == "adamw"]
    adapter_winner = min(adamw_adapters, key=lambda name: results[name]["best_ema_validation_mae"])
    optimizer_candidates = ["lora_adamw", "lora_muon"]
    optimizer_winner = min(
        optimizer_candidates, key=lambda name: results[name]["best_ema_validation_mae"]
    )
    ledger = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(pilot["passed"] and architecture["passed"]),
        "locked_test_accessed": False,
        "decisions": {
            "architecture_by_endpoint_for_specialist_search": architecture["pilot_winners"],
            "search_backbone": "dense_vjepa21_vitb",
            "final_backbone_policy": "ViT-L candidate after ViT-B search; retain task-specific functional/DINO challengers",
            "peft_initialization": results[adapter_winner]["adapter"],
            "optimizer": results[optimizer_winner]["optimizer"],
            "ema": True,
            "precision": "BF16",
            "fp8": "forbidden until separate BF16 parity evidence",
            "schedule": ["frozen", "peft", "selective_unfreeze"],
        },
        "selection_evidence": {
            "adapter_winner": adapter_winner,
            "adapter_validation_mae": results[adapter_winner]["best_ema_validation_mae"],
            "optimizer_winner": optimizer_winner,
            "optimizer_validation_mae": results[optimizer_winner]["best_ema_validation_mae"],
        },
        "revalidation_rule": "Re-evaluate the representative G5 choice per endpoint during specialist validation; never tune on locked test.",
    }
    destination = Path(config["paths"]["evidence_root"]) / "G5" / "decision_ledger.json"
    destination.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger

