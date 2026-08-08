from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file


def write_licence_provenance(config: dict[str, Any]) -> dict[str, Any]:
    echo_source = Path(config["paths"]["echojepa_source"])
    vjepa_source = Path(config["paths"]["vjepa21_source"])
    local_licenses = {
        "echojepa_apache_2": echo_source / "APACHE-LICENSE",
        "vjepa2_mit": vjepa_source / "LICENSE",
        "vjepa2_apache_2": vjepa_source / "APACHE-LICENSE",
    }
    checkpoints = {
        "functional_vjepa2_vitl": Path(config["paths"]["functional_checkpoint"]),
        "dense_vjepa21_vitb": Path(config["paths"]["dense_vitb_checkpoint"]),
        "dense_vjepa21_vitl": Path(config["paths"]["dense_vitl_checkpoint"]),
        "dinov3_vitb": Path(config["paths"]["dinov3_checkpoint"]),
    }
    license_files = {
        name: {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None}
        for name, path in local_licenses.items()
    }
    checkpoint_rows = {
        name: {
            "path": str(path), "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "track": "R", "commercial_inheritance_allowed": False,
        }
        for name, path in checkpoints.items()
    }
    checks = {
        "all_local_license_files_present": all(row["exists"] for row in license_files.values()),
        "all_checkpoint_assets_present": all(row["exists"] for row in checkpoint_rows.values()),
        "all_model_assets_research_track": all(row["track"] == "R" for row in checkpoint_rows.values()),
        "commercial_inheritance_blocked": all(not row["commercial_inheritance_allowed"] for row in checkpoint_rows.values()),
        "mimic_source_is_read_only_network_path": str(config["paths"]["dicom_root"]).upper().startswith("Z:\\"),
        "all_mutable_roots_are_local": all(
            str(config["paths"][key]).upper().startswith("C:\\")
            for key in ("local_staging_root", "run_root", "private_root", "report_root", "evidence_root")
        ),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "policy": {
            "default_track": "R", "track_c_import": "blocked",
            "reason": "MIMIC-derived data, embeddings, thresholds, pseudo-labels, and weights remain Research Track artifacts",
            "clinical_use": False, "redistribution_of_restricted_data": False,
        },
        "dataset": {
            "name": "MIMIC-IV-ECHO v1.0",
            "license": "PhysioNet Credentialed Health Data License 1.5.0",
            "license_url": "https://physionet.org/content/mimic-iv-echo/view-license/1.0/",
            "permitted_pipeline_purpose": "scientific research",
            "raw_source": str(config["paths"]["dicom_root"]), "raw_source_mode": "read-only",
            "copied_into_outputs": False,
        },
        "software_and_weights": {
            "EchoJEPA": {
                "source_url": "https://github.com/bowang-lab/EchoJEPA",
                "source_revision": config["assets"]["echojepa_source_revision"],
                "code_license": "Apache-2.0",
                "checkpoint_provenance": "MIMIC and VideoMix22M lineage; Research Track only",
            },
            "V-JEPA2.1": {
                "source_url": "https://github.com/facebookresearch/vjepa2",
                "source_revision": config["assets"]["vjepa21_source_revision"],
                "repository_licenses_present": ["MIT", "Apache-2.0"],
                "checkpoint_provenance": "EchoJEPA MIMIC lineage; Research Track only",
            },
            "DINOv3": {
                "source_url": "https://github.com/facebookresearch/dinov3",
                "model_id": "timm/vit_base_patch16_dinov3.lvd1689m",
                "license": "DINOv3 License (last updated 2025-08-19)",
                "license_url": "https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md",
                "track": "R",
            },
        },
        "local_license_files": license_files,
        "checkpoint_assets": checkpoint_rows,
        "legal_interpretation": "No legal conclusion is asserted; fail-closed research provenance policy is applied.",
    }
    destination = Path(config["paths"]["evidence_root"]) / "G4" / "licence_provenance.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

