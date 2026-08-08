from __future__ import annotations

from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

import psutil

from .hashing import sha256_file


def gpu_smoke(config: dict[str, Any], include_checkpoint_hash: bool = True) -> dict[str, Any]:
    import torch

    floor = config["hardware_floor"]
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            left = torch.randn((2048, 2048), device=index, dtype=torch.bfloat16)
            right = torch.randn((2048, 2048), device=index, dtype=torch.bfloat16)
            result = left @ right
            torch.cuda.synchronize(index)
            finite = bool(torch.isfinite(result).all().item())
        devices.append(
            {
                "index": index,
                "name": props.name,
                "memory_mib": round(props.total_memory / 1024**2),
                "compute_capability": [props.major, props.minor],
                "bf16_matmul_finite": finite,
            }
        )
    supported_arches = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    checks = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": len(devices) >= floor["gpu_count"],
        "gpu_names": all(floor["gpu_name_contains"] in d["name"] for d in devices[: floor["gpu_count"]]),
        "gpu_memory": all(d["memory_mib"] >= floor["memory_mib_each_min"] for d in devices[: floor["gpu_count"]]),
        "compute_capability": all(tuple(d["compute_capability"]) >= tuple(floor["compute_capability_min"]) for d in devices[: floor["gpu_count"]]),
        "bf16": all(d["bf16_matmul_finite"] for d in devices[: floor["gpu_count"]]),
        "wheel_contains_sm120": "sm_120" in supported_arches,
        "system_ram": psutil.virtual_memory().total / 1024**3 >= floor["system_ram_gib_min"],
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "supported_arches": supported_arches,
        "devices": devices,
        "system_ram_gib": round(psutil.virtual_memory().total / 1024**3, 1),
    }
    if include_checkpoint_hash:
        checkpoint = Path(config["paths"]["functional_checkpoint"])
        report["functional_checkpoint"] = {
            "exists": checkpoint.exists(),
            "sha256": sha256_file(checkpoint) if checkpoint.exists() else None,
            "expected_sha256": config["assets"]["functional_checkpoint_sha256"],
        }
        checks["functional_checkpoint_hash"] = (
            report["functional_checkpoint"]["sha256"] == report["functional_checkpoint"]["expected_sha256"]
        )
        report["passed"] = all(checks.values())
    return report


def write_report(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def echojepa_checkpoint_smoke(config: dict[str, Any], device: int = 0) -> dict[str, Any]:
    import torch

    source = Path(config["paths"]["echojepa_source"])
    checkpoint = Path(config["paths"]["functional_checkpoint"])
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "checkpoint": str(checkpoint),
        "device": device,
        "passed": False,
    }
    try:
        from src.models.vision_transformer import vit_large

        model = vit_large(
            img_size=(config["training"]["resolution"], config["training"]["resolution"]),
            patch_size=16,
            num_frames=config["training"]["frames"],
            tubelet_size=2,
            use_rope=True,
            uniform_power=True,
            use_sdpa=True,
            use_activation_checkpointing=False,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        encoder_state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state["encoder"].items()
        }
        incompatible = model.load_state_dict(encoder_state, strict=False)
        missing = [key for key in incompatible.missing_keys if key != "pos_embed"]
        unexpected = [key for key in incompatible.unexpected_keys if key != "pos_embed"]
        del state, encoder_state
        model = model.eval().to(device=device, dtype=torch.bfloat16)
        sample = torch.randn(
            (1, 3, config["training"]["frames"], config["training"]["resolution"], config["training"]["resolution"]),
            device=device,
            dtype=torch.bfloat16,
        )
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            output = model(sample)
        torch.cuda.synchronize(device)
        if isinstance(output, list):
            output = output[-1]
        report.update(
            {
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "input_shape": list(sample.shape),
                "output_shape": list(output.shape),
                "output_finite": bool(torch.isfinite(output).all().item()),
                "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
        )
        report["passed"] = bool(
            not missing
            and not unexpected
            and report["output_shape"] == [1, 1568, 1024]
            and report["output_finite"]
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["wall_seconds"] = round(time.perf_counter() - started, 3)
    return report


def dinov3_checkpoint_smoke(config: dict[str, Any], device: int = 1) -> dict[str, Any]:
    import timm
    import torch
    from safetensors.torch import load_file

    checkpoint = Path(config["paths"]["dinov3_checkpoint"])
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "DINOv3_ViT-B/16_LVD-1689M",
        "track": "R",
        "license": "dinov3-license",
        "checkpoint": str(checkpoint),
        "passed": False,
    }
    try:
        model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=False, num_classes=0)
        state = load_file(checkpoint, device="cpu")
        incompatible = model.load_state_dict(state, strict=False)
        model = model.eval().to(device=device, dtype=torch.bfloat16)
        sample = torch.randn((1, 3, 256, 256), device=device, dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model.forward_features(sample)
        torch.cuda.synchronize(device)
        report.update(
            {
                "checkpoint_sha256": sha256_file(checkpoint),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                "input_shape": list(sample.shape),
                "output_shape": list(output.shape),
                "output_finite": bool(torch.isfinite(output).all().item()),
                "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
        )
        report["passed"] = bool(
            not report["missing_keys"]
            and not report["unexpected_keys"]
            and report["output_shape"] == [1, 261, 768]
            and report["output_finite"]
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["wall_seconds"] = round(time.perf_counter() - started, 3)
    return report


def vjepa21_checkpoint_smoke(config: dict[str, Any], device: int = 1) -> dict[str, Any]:
    import torch

    source = Path(config["paths"]["vjepa21_source"])
    checkpoint = Path(config["paths"]["dense_vitl_checkpoint"])
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "EchoJEPA_V-JEPA2.1_ViT-L_hierarchical",
        "track": "R",
        "source": str(source),
        "source_revision": config["assets"]["vjepa21_source_revision"],
        "checkpoint": str(checkpoint),
        "passed": False,
    }
    if not checkpoint.exists():
        report["error"] = "Dense ViT-L checkpoint is not yet present"
        return report
    try:
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from app.vjepa_2_1.models.vision_transformer import vit_large

        model = vit_large(
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
            use_activation_checkpointing=False,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        encoder_state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state["encoder"].items()
        }
        incompatible = model.load_state_dict(encoder_state, strict=False)
        del state, encoder_state
        model.return_hierarchical = True
        model = model.eval().to(device=device, dtype=torch.bfloat16)
        sample = torch.randn((1, 3, 16, 224, 224), device=device, dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(sample)
        torch.cuda.synchronize(device)
        report.update(
            {
                "checkpoint_sha256": sha256_file(checkpoint),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                "input_shape": list(sample.shape),
                "hierarchical_output_shape": list(output.shape),
                "hierarchical_layers": list(model.hierarchical_layers),
                "output_finite": bool(torch.isfinite(output).all().item()),
                "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
        )
        report["passed"] = bool(
            not report["missing_keys"]
            and not report["unexpected_keys"]
            and report["hierarchical_output_shape"] == [1, 1568, 4096]
            and report["output_finite"]
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["wall_seconds"] = round(time.perf_counter() - started, 3)
    return report


def vjepa21_vitb_checkpoint_smoke(config: dict[str, Any], device: int = 0) -> dict[str, Any]:
    import torch

    source = Path(config["paths"]["vjepa21_source"])
    checkpoint = Path(config["paths"]["dense_vitb_checkpoint"])
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": "EchoJEPA_V-JEPA2.1_ViT-B_hierarchical_search",
        "track": "R",
        "source_revision": config["assets"]["vjepa21_source_revision"],
        "checkpoint": str(checkpoint),
        "passed": False,
    }
    if not checkpoint.exists():
        report["error"] = "Dense ViT-B checkpoint is not yet present"
        return report
    try:
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
            use_activation_checkpointing=False,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        encoder_state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state["encoder"].items()
        }
        incompatible = model.load_state_dict(encoder_state, strict=False)
        del state, encoder_state
        model.return_hierarchical = True
        model = model.eval().to(device=device, dtype=torch.bfloat16)
        sample = torch.randn((1, 3, 16, 224, 224), device=device, dtype=torch.bfloat16)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(sample)
        torch.cuda.synchronize(device)
        report.update(
            {
                "checkpoint_sha256": sha256_file(checkpoint),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                "hierarchical_output_shape": list(output.shape),
                "hierarchical_layers": list(model.hierarchical_layers),
                "output_finite": bool(torch.isfinite(output).all().item()),
                "peak_gpu_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
        )
        report["passed"] = bool(
            not report["missing_keys"]
            and not report["unexpected_keys"]
            and report["hierarchical_output_shape"] == [1, 1568, 3072]
            and report["output_finite"]
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["wall_seconds"] = round(time.perf_counter() - started, 3)
    return report

