from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else repository_root() / "configs" / "protocol.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("protocol") != "HEM-FM-v4.0":
        raise ValueError("Refusing to run with an unrecognised protocol")
    assert_local_mutable_paths(config)
    return config


def ensure_run_directories(config: dict[str, Any]) -> None:
    for key in ("run_root", "private_root", "report_root", "evidence_root"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def assert_local_mutable_paths(config: dict[str, Any]) -> None:
    local_drive = repository_root().drive.upper()
    mutable = (
        "local_staging_root",
        "echojepa_source",
        "vjepa21_source",
        "functional_checkpoint",
        "dense_vitl_checkpoint",
        "dense_vitb_checkpoint",
        "dinov3_checkpoint",
        "camus_root",
        "ted_root",
        "unity_root",
        "mimic_lv_staging_root",
        "ev9v_root",
        "cardiacuda_root",
        "hmc_qu_root",
        "echocp_root",
        "cardiacnet_root",
        "echoxflow_demo_root",
        "prior_feature_root",
        "run_root",
        "private_root",
        "report_root",
        "evidence_root",
    )
    violations = []
    for key in mutable:
        candidate = Path(config["paths"][key])
        if not candidate.is_absolute() or candidate.drive.upper() != local_drive:
            violations.append(f"{key}={candidate}")
    if violations:
        raise ValueError(
            "Mutable HEM-FM artifacts must stay on this PC's local drive "
            f"({local_drive}); refused: " + ", ".join(violations)
        )

