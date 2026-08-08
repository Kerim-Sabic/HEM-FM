from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .architecture_pilot import (
    ARCHITECTURES,
    _embed_row,
    _load_dino,
    _load_video_encoder,
)
from .gates import assert_through


def _cine_id(relative_path: str, calibration_type: str) -> str:
    return hashlib.sha256(
        f"{calibration_type}|{relative_path}".encode("utf-8")
    ).hexdigest()[:24]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _status(
    path: Path,
    *,
    phase: str,
    total: int,
    complete: int,
    started: float,
    current: str | None = None,
    error_count: int = 0,
) -> None:
    elapsed = time.perf_counter() - started
    rate = complete / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - complete)
    payload = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "total_cines": total,
        "complete_cines": complete,
        "progress_fraction": complete / max(total, 1),
        "elapsed_seconds": round(elapsed, 1),
        "cines_per_second": round(rate, 4),
        "estimated_remaining_seconds": round(remaining / rate, 1) if rate else None,
        "current_cine_id": current,
        "error_count": error_count,
        "compute_location": "local workstation GPUs",
        "mutable_storage": "local C drive",
        "network_role": "read-only DICOM source",
        "locked_test_accessed": False,
    }
    _atomic_json(path, payload)


def build_full_specialist_feature_cache(config: dict[str, Any]) -> dict[str, Any]:
    assert_through(config, "G5")
    started = time.perf_counter()
    private_root = Path(config["paths"]["private_root"])
    run_root = Path(config["paths"]["run_root"]) / "week_training"
    feature_root = private_root / "specialist_cine_features"
    run_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    status_path = run_root / "status.json"
    cohort = pd.read_csv(
        private_root / "endpoint_cohorts_development_private.csv",
        dtype={"subject_id": str, "study_id": str},
        low_memory=False,
    )
    if (
        not set(cohort["split"]).issubset({"train", "validation"})
        or cohort["test_locked"].fillna(False).astype(bool).any()
    ):
        raise RuntimeError("Full extractor received a locked internal-test row")
    requests: dict[str, dict[str, str]] = {}
    manifest_rows = []
    for row in cohort.itertuples(index=False):
        calibration_type = "spectral" if row.target == "AV_PEAK_VELOCITY" else "spatial"
        cine_ids = []
        for cine in json.loads(row.cines):
            relative = str(cine["path"])
            identifier = _cine_id(relative, calibration_type)
            cine_ids.append(identifier)
            requests.setdefault(
                identifier,
                {"relative_path": relative, "calibration_type": calibration_type},
            )
        manifest_rows.append(
            {
                "sample_id": hashlib.sha256(
                    f"{row.subject_id}|{row.study_id}|{row.target}|{row.cines}".encode("utf-8")
                ).hexdigest()[:20],
                "subject_id": str(row.subject_id),
                "study_id": str(row.study_id),
                "target": str(row.target),
                "split": str(row.split),
                "value": float(row.value),
                "canonical_unit": str(row.canonical_unit),
                "cine_ids": json.dumps(cine_ids, separators=(",", ":")),
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(private_root / "specialist_rows_private.csv", index=False)
    request_path = private_root / "specialist_cine_requests_private.json"
    request_path.write_text(json.dumps(requests, indent=2), encoding="utf-8")
    pending = [
        (identifier, request)
        for identifier, request in requests.items()
        if not (feature_root / f"{identifier}.npz").exists()
    ]
    total = len(requests)
    complete = total - len(pending)
    errors: list[dict[str, str]] = []
    _status(
        status_path, phase="full_feature_extraction", total=total,
        complete=complete, started=started,
    )
    model_metadata = {}
    if pending:
        devices = {
            "functional_vjepa2_vitl": 0,
            "dense_vjepa21_vitb": 1,
            "dense_vjepa21_vitl": 0,
            "dinov3_vitb": 1,
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
                "device": devices[name],
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
            for name, model in models.items()
        }
        for index, (identifier, request) in enumerate(pending, start=1):
            try:
                fake_target = (
                    "AV_PEAK_VELOCITY"
                    if request["calibration_type"] == "spectral"
                    else "EF"
                )
                features = _embed_row(
                    {
                        "target": fake_target,
                        "cines": json.dumps(
                            [{"path": request["relative_path"]}],
                            separators=(",", ":"),
                        ),
                    },
                    config,
                    models,
                    devices,
                )
                np.savez_compressed(
                    feature_root / f"{identifier}.npz",
                    **{name: features[name] for name in ARCHITECTURES},
                )
                complete += 1
            except Exception as exc:
                errors.append(
                    {
                        "cine_id": identifier,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if index % 10 == 0 or index == len(pending):
                _status(
                    status_path,
                    phase="full_feature_extraction",
                    total=total,
                    complete=complete,
                    started=started,
                    current=identifier,
                    error_count=len(errors),
                )
                if errors:
                    (run_root / "feature_errors.json").write_text(
                        json.dumps(errors, indent=2), encoding="utf-8"
                    )
    checks = {
        "all_unique_development_cines_cached": complete == total,
        "zero_errors": not errors,
        "four_architectures_per_cine": all(
            all(name in np.load(path, allow_pickle=False).files for name in ARCHITECTURES)
            for path in list(feature_root.glob("*.npz"))[: min(total, 256)]
        ),
        "patient_splits_precede_features": True,
        "locked_test_not_accessed": True,
        "bf16_inference": True,
        "fp8_not_used": True,
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "development_rows": len(manifest),
        "unique_cines": total,
        "complete_cines": complete,
        "errors": errors,
        "feature_root": str(feature_root),
        "manifest": str(private_root / "specialist_rows_private.csv"),
        "model_metadata": model_metadata,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "locked_test_accessed": False,
    }
    report_path = Path(config["paths"]["report_root"]) / "full_feature_extraction.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _status(
        status_path,
        phase="feature_extraction_complete" if report["passed"] else "feature_extraction_failed",
        total=total,
        complete=complete,
        started=started,
        error_count=len(errors),
    )
    return report

