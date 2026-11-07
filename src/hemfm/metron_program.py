from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import psutil
import yaml


@dataclass(frozen=True)
class Stage:
    stage_id: str
    name: str
    gpu_hours: int
    weeks: float
    gate: str


STAGES = (
    Stage("S0", "Ingest, calibration, geometry, splits and caches", 300, 4, "G0"),
    Stage("S0.5", "Reader-noise model and soft targets", 20, 1, "G0.5"),
    Stage("S1", "Domain-adaptive self-supervision", 1100, 7, "G1"),
    Stage("S2", "Classification stack", 150, 1, "G2"),
    Stage("S3", "Segmentation arms and mandatory CNN baseline", 650, 5, "G3"),
    Stage("S4", "Hybrid landmarks, cascade and mining", 600, 4, "G4"),
    Stage("S5", "Doppler heads", 250, 2, "G5"),
    Stage("S6", "Motion and biomechanics", 550, 4, "G6"),
    Stage("S7", "Semi-supervised expansion", 650, 5, "G7"),
    Stage("S8", "Joint tuning, calibration and abstention", 380, 3, "G8"),
    Stage("S9", "Distillation and packaging", 200, 2, "G9"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=8)
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        result.append(
            {"index": int(fields[0]), "name": fields[1], "memory_mib": int(fields[2])}
        )
    return result


def _active_training_processes() -> list[dict[str, Any]]:
    markers = ("train_", "training", "echonet_trace", "cardiacuda_segmentation")
    rows = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
        lower = command.lower()
        if "hemfm" not in lower or not any(marker in lower for marker in markers):
            continue
        if "metron_program" in lower:
            continue
        rows.append(
            {
                "pid": int(process.info["pid"]),
                "name": str(process.info.get("name") or "process"),
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=8,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def run_stage0_readiness(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.resolve().parents[1]
    runtime_root = Path(config["paths"]["runtime_root"]).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    status_root = runtime_root / "status"
    evidence_root = runtime_root / "evidence" / "S0"
    status_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)

    spec_paths = {name: Path(path) for name, path in config["specifications"].items()}
    specification_hashes = {
        name: sha256_file(path) for name, path in spec_paths.items() if path.is_file()
    }
    gpus = _gpu_inventory()
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(runtime_root)
    active_training = _active_training_processes()

    foundation = _read_json(Path(config["legacy_evidence"]["foundation_status"]))
    trace_reports = [
        _read_json(Path(path)) for path in config["legacy_evidence"]["trace_reports"]
    ]
    trace_reports_present = bool(trace_reports) and all(trace_reports)
    test_sealed = (
        foundation.get("locked_test_pixels_accessed") is False
        and foundation.get("pixel_data_read") is False
        and trace_reports_present
        and all(report.get("locked_test_accessed") is False for report in trace_reports)
        and all(report.get("official_test_videos_accessed") is False for report in trace_reports)
    )

    required = config["requirements"]
    checks = {
        "specifications_readable": len(specification_hashes) == len(spec_paths),
        "specifications_hash_bound": all(
            len(value) == 64 for value in specification_hashes.values()
        ),
        "gpu_count": len(gpus) >= int(required["gpu_count_min"]),
        "gpu_model": bool(gpus)
        and all(required["gpu_name_contains"] in gpu["name"] for gpu in gpus),
        "gpu_memory": bool(gpus)
        and all(
            gpu["memory_mib"] >= int(required["gpu_memory_mib_each_min"])
            for gpu in gpus
        ),
        "system_ram": memory.total / 1024**3 >= float(required["system_ram_gib_min"]),
        "local_disk_capacity": disk.total / 1024**3
        >= float(required["local_disk_total_gib_min"]),
        "local_disk_free": disk.free / 1024**3
        >= float(required["local_disk_free_gib_min"]),
        "legacy_foundation_complete": foundation.get("phase") == "complete",
        "locked_test_pixels_sealed": test_sealed,
        "no_conflicting_training_process": not active_training,
        "multix_role_is_nonconflicting": config["reference_standard"].get(
            "unitylv_multix_role"
        )
        in {"sealed_evaluation_only", "patient_disjoint_development_partition"},
        "clinical_claims_disabled": config["safety"].get("clinical_use") is False,
        "public_weight_release_disabled": config["safety"].get(
            "publish_weights"
        )
        is False,
    }
    blocking_checks = tuple(config["stage0"]["blocking_checks"])
    blockers = [name for name in blocking_checks if checks.get(name) is not True]
    ready = not blockers
    now = datetime.now(timezone.utc).isoformat()

    stages = []
    for index, stage in enumerate(STAGES):
        if index == 0:
            state = "ready" if ready else "blocked"
        else:
            state = "locked"
        stages.append({**asdict(stage), "state": state})

    report = {
        "schema_version": 1,
        "protocol": "METRON-v8.0",
        "updated_utc": now,
        "phase": "s0_ready" if ready else "s0_blocked",
        "stage": "S0",
        "ready_to_launch": ready,
        "checks": checks,
        "blockers": blockers,
        "locked_test_pixels_accessed": False,
        "pixel_data_read": False,
        "gpu_training_started": False,
        "hardware": {
            "gpus": gpus,
            "ram_total_gib": round(memory.total / 1024**3, 1),
            "disk_total_gib": round(disk.total / 1024**3, 1),
            "disk_free_gib": round(disk.free / 1024**3, 1),
        },
        "active_training_processes": active_training,
        "specification_hashes": specification_hashes,
        "repo_revision": _git_revision(repo_root),
        "stages": stages,
        "programme": {
            "gpu_hours": sum(stage.gpu_hours for stage in STAGES),
            "measurement_playbooks": 26,
            "model_family": ["METRON-L", "METRON-B", "METRON-Edge"],
        },
        "next_action": (
            "launch_s0_inventory_and_baseline_reproduction"
            if ready
            else "resolve_blocking_checks_before_pixel_decode_or_training"
        ),
    }
    (status_root / "status.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (evidence_root / "readiness.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lock = {
        "schema_version": 1,
        "created_utc": now,
        "protocol": "METRON-v8.0",
        "specification_hashes": specification_hashes,
        "repo_revision": report["repo_revision"],
        "unitylv_multix_role": config["reference_standard"]["unitylv_multix_role"],
        "locked_test_pixels_accessed": False,
    }
    (evidence_root / "specification_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="METRON v8 fail-closed programme runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("action", choices=["stage0-readiness"])
    args = parser.parse_args()
    if args.action == "stage0-readiness":
        result = run_stage0_readiness(args.config.resolve())
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["ready_to_launch"] else 2)


if __name__ == "__main__":
    main()
