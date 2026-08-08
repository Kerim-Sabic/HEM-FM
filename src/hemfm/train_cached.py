from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["ef_value", "lvedv_value", "lvesv_value"]
TARGET_NAMES = ["EF", "LVEDV", "LVESV"]


@dataclass(frozen=True)
class CachedPilotConfig:
    seed: int = 1103
    device: int = 0
    epochs: int = 120
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    patience: int = 30
    consistency_weight: float = 0.1
    ema_decay: float = 0.995


class CachedStudyDataset:
    def __init__(self, feature_root: str | Path, split: str, preload: bool = True):
        import torch

        self.torch = torch
        root = Path(feature_root)
        source_tokens = np.load(root / "tokens_fp16.npy", mmap_mode="r")
        source_structured = np.load(root / "structured_fp32.npy", mmap_mode="r")
        self.completed = np.load(root / "completed_uint8.npy", mmap_mode="r")
        index = pd.read_csv(root / "study_index.csv")
        index = index[(index["split"] == split) & (self.completed[index["feature_index"].to_numpy(dtype=int)] == 1)].copy()
        index = index[index[TARGET_COLUMNS].notna().any(axis=1)].reset_index(drop=True)
        self.feature_indices = index["feature_index"].to_numpy(dtype=int)
        self.targets = index[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        self.patient_hashes = index["patient_hash"].astype(str).to_numpy()
        self.study_hashes = index["study_hash"].astype(str).to_numpy()
        self.preloaded = preload
        if preload:
            self.tokens = np.array(source_tokens[self.feature_indices], copy=True)
            self.structured = np.array(source_structured[self.feature_indices], copy=True)
        else:
            self.tokens = source_tokens
            self.structured = source_structured

    def __len__(self) -> int:
        return len(self.feature_indices)

    def __getitem__(self, index: int):
        feature_index = index if self.preloaded else self.feature_indices[index]
        tokens = self.torch.from_numpy(np.asarray(self.tokens[feature_index], dtype=np.float32).copy())
        structured = self.torch.from_numpy(np.asarray(self.structured[feature_index], dtype=np.float32).reshape(-1).copy())
        target_array = self.targets[index]
        mask = np.isfinite(target_array)
        target = self.torch.from_numpy(np.nan_to_num(target_array, nan=0.0))
        return tokens, structured, target, self.torch.from_numpy(mask)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _normalization(dataset: CachedStudyDataset) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(dataset.targets, axis=0).astype(np.float32)
    scales = np.nanstd(dataset.targets, axis=0).astype(np.float32)
    scales = np.maximum(scales, np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
    return means, scales


def _make_model(structured_dim: int, token_dim: int = 1024):
    import torch

    class AttentiveTwoViewRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(token_dim)
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(token_dim, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 1),
            )
            self.structured = torch.nn.Sequential(
                torch.nn.LayerNorm(structured_dim),
                torch.nn.Linear(structured_dim, 64),
                torch.nn.GELU(),
            )
            self.trunk = torch.nn.Sequential(
                torch.nn.LayerNorm(token_dim * 2 + 64),
                torch.nn.Linear(token_dim * 2 + 64, 512),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(512, 256),
                torch.nn.GELU(),
            )
            self.mean = torch.nn.Linear(256, 3)
            self.log_variance = torch.nn.Linear(256, 3)

        def forward(self, tokens, structured):
            pooled = []
            for view in range(tokens.shape[1]):
                view_tokens = tokens[:, view]
                scores = self.attention(self.norm(view_tokens)).squeeze(-1)
                weights = torch.softmax(scores.float(), dim=1).to(view_tokens.dtype)
                pooled.append(torch.sum(view_tokens * weights.unsqueeze(-1), dim=1))
            features = torch.cat([*pooled, self.structured(structured)], dim=-1)
            hidden = self.trunk(features)
            return self.mean(hidden), self.log_variance(hidden).clamp(-6.0, 4.0)

    return AttentiveTwoViewRegressor()


def _loss(mean, log_variance, target, mask, target_mean, target_scale, consistency_weight: float):
    import torch

    standardized = (target - target_mean) / target_scale
    per_target = 0.5 * (torch.exp(-log_variance) * (mean - standardized).square() + log_variance)
    observed = mask.to(per_target.dtype)
    nll = (per_target * observed).sum() / observed.sum().clamp_min(1.0)
    physical = mean * target_scale + target_mean
    derived_ef = 100.0 * (physical[:, 1] - physical[:, 2]) / physical[:, 1].clamp_min(1.0)
    consistency = torch.nn.functional.smooth_l1_loss(physical[:, 0], derived_ef)
    return nll + consistency_weight * consistency / target_scale[0]


def _evaluate(model, loader, target_mean, target_scale, device: int) -> dict[str, Any]:
    import torch

    model.eval()
    predictions = []
    targets = []
    masks = []
    with torch.inference_mode():
        for tokens, structured, target, mask in loader:
            tokens = tokens.to(device, non_blocking=True)
            structured = structured.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mean, _ = model(tokens, structured)
            predictions.append((mean.float() * target_scale + target_mean).cpu().numpy())
            targets.append(target.numpy())
            masks.append(mask.numpy())
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    mask = np.concatenate(masks).astype(bool)
    metrics: dict[str, Any] = {}
    normalized = []
    for index, name in enumerate(TARGET_NAMES):
        errors = np.abs(prediction[:, index] - target[:, index])[mask[:, index]]
        mae = float(np.mean(errors)) if len(errors) else math.nan
        metrics[name] = {"mae": mae, "n": int(len(errors))}
        normalized.append(mae / float(target_scale[index]))
    metrics["mean_normalized_mae"] = float(np.mean(normalized))
    return metrics


def run_cached_functional_pilot(feature_root: str | Path, output_root: str | Path, config: CachedPilotConfig) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    _seed_everything(config.seed)
    started = time.perf_counter()
    train = CachedStudyDataset(feature_root, "train")
    validation = CachedStudyDataset(feature_root, "validation")
    target_mean_np, target_scale_np = _normalization(train)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = _make_model(int(np.prod(train.structured.shape[1:]))).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    target_mean = torch.as_tensor(target_mean_np, device=config.device)
    target_scale = torch.as_tensor(target_scale_np, device=config.device)
    ema = deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    best_score = math.inf
    best_epoch = -1
    best_state = None
    history = []
    patience = 0
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for tokens, structured, target, mask in train_loader:
            tokens = tokens.to(config.device, non_blocking=True)
            structured = structured.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)
            mask = mask.to(config.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mean, log_variance = model(tokens, structured)
                loss = _loss(mean, log_variance, target, mask, target_mean, target_scale, config.consistency_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for ema_parameter, parameter in zip(ema.parameters(), model.parameters(), strict=True):
                    ema_parameter.lerp_(parameter.detach(), 1.0 - config.ema_decay)
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = _evaluate(ema, validation_loader, target_mean, target_scale, config.device)
        score = metrics["mean_normalized_mae"]
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation": metrics})
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu() for key, value in ema.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= config.patience:
            break
    destination = Path(output_root) / f"functional_seed_{config.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "best_ema.pt"
    torch.save(
        {
            "model": best_state,
            "target_mean": target_mean_np,
            "target_scale": target_scale_np,
            "config": asdict(config),
            "best_epoch": best_epoch,
            "test_accessed": False,
            "track": "R",
        },
        checkpoint_path,
    )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "candidate": "B_functional_vjepa2_vitl_frozen_attentive",
        "track": "R",
        "sources": ["MIMIC-IV-ECHO", "EchoJEPA-MIMIC"],
        "config": asdict(config),
        "train_studies": len(train),
        "validation_studies": len(validation),
        "test_accessed": False,
        "best_epoch": best_epoch,
        "best_validation": history[best_epoch - 1]["validation"],
        "target_mean": dict(zip(TARGET_NAMES, target_mean_np.tolist(), strict=True)),
        "target_scale": dict(zip(TARGET_NAMES, target_scale_np.tolist(), strict=True)),
        "epochs_completed": len(history),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

