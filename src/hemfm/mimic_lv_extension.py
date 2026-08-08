from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd

from .frozen_specialists import _atomic_json
from .hashing import sha256_file
from .splits import patient_split


SOURCE = "https://physionet.org/content/mimic-iv-echo-ext-lvvol-a4c/1.0.0/"
CITATION = (
    "K. Ekambaram, A. Arnab, P. Herbst, and R. Theart, "
    "MIMIC-IV-ECHO-Ext-LVVOLUMES-A4C-ROI, version 1.0.0, PhysioNet, 2026, "
    "doi:10.13026/713s-z339."
)


def _load_metadata(path: Path, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"patient_id": str, "study_id": str})
    required = {
        "patient_id",
        "study_id",
        "parent_dicom_path",
        "LVEDV_A4C",
        "LVESV_A4C",
        "LVEF_A4C",
        "LVEDV_BP",
        "LVESV_BP",
        "LVEF_BP",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MIMIC LV extension is missing columns: {missing}")
    frame["split"] = frame["patient_id"].map(lambda value: patient_split(value, seed))
    return frame


def _status(
    path: Path,
    *,
    phase: str,
    copied_bytes: int,
    total_bytes: int,
    complete_files: int,
    total_files: int,
    started: float,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-6)
    rate = copied_bytes / elapsed
    remaining = max(0, total_bytes - copied_bytes)
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "name": "MIMIC-LV development cines",
            "phase": phase,
            "downloaded_bytes": copied_bytes,
            "total_bytes": total_bytes,
            "progress_fraction": copied_bytes / max(1, total_bytes),
            "complete_files": complete_files,
            "total_files": total_files,
            "bytes_per_second": round(rate, 1),
            "estimated_remaining_seconds": round(remaining / rate, 1) if rate else None,
            "source_url": "credentialed read-only SMB mirror of PhysioNet",
            "track": "R",
            "mutable_storage": "local C drive",
            "locked_test_accessed": False,
        },
    )


def stage_mimic_lv_extension(
    config: dict[str, Any],
    *,
    source_root: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(source_root) if source_root else Path(config["paths"]["corpus_root"])
    metadata = Path(metadata_path) if metadata_path else Path(config["paths"]["lv_extension"])
    destination = Path(config["paths"]["mimic_lv_staging_root"])
    dicom_destination = destination / "dicom"
    destination.mkdir(parents=True, exist_ok=True)
    dicom_destination.mkdir(parents=True, exist_ok=True)
    frame = _load_metadata(metadata, int(config["splits"]["seed"]))
    development = frame.loc[frame["split"].isin(["train", "validation"])].copy()
    locked = frame.loc[frame["split"] == "internal_test"].copy()
    source_paths = [source / Path(value.replace("/", "\\")) for value in development["parent_dicom_path"]]
    missing_sources = [str(path) for path in source_paths if not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"{len(missing_sources)} extension DICOMs are missing; first: {missing_sources[0]}")
    total_bytes = sum(path.stat().st_size for path in source_paths)
    status_path = Path(config["paths"]["run_root"]) / "week_training" / "dataset_download_status_mimic_lv.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    copied_bytes = 0
    records: list[dict[str, Any]] = []
    for index, (row, source_path) in enumerate(zip(development.itertuples(index=False), source_paths, strict=True), start=1):
        target = dicom_destination / f"{row.study_id}.dcm"
        expected_size = source_path.stat().st_size
        if not target.exists() or target.stat().st_size != expected_size:
            temporary = target.with_name(f"{target.name}.partial")
            with source_path.open("rb") as source_handle, temporary.open("wb") as target_handle:
                shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
            if temporary.stat().st_size != expected_size:
                raise IOError(f"Incomplete local copy for {row.study_id}")
            temporary.replace(target)
        copied_bytes += expected_size
        records.append(
            {
                "patient_id": row.patient_id,
                "study_id": row.study_id,
                "split": row.split,
                "local_dicom": str(target),
                "bytes": expected_size,
                "sha256": sha256_file(target),
            }
        )
        if index % 10 == 0 or index == len(development):
            _status(
                status_path,
                phase="staging" if index < len(development) else "complete",
                copied_bytes=copied_bytes,
                total_bytes=total_bytes,
                complete_files=index,
                total_files=len(development),
                started=started,
            )

    development.to_csv(destination / "development_labels_private.csv", index=False)
    pd.DataFrame(records).to_csv(destination / "development_files_private.csv", index=False)
    ef_recomputed = (frame["LVEDV_A4C"] - frame["LVESV_A4C"]) / frame["LVEDV_A4C"] * 100.0
    checks = {
        "official_row_count": len(frame) == 1064,
        "csv_patient_count_is_809": frame["patient_id"].nunique() == 809,
        "source_page_patient_count_discrepancy_documented": True,
        "patient_disjoint_split": bool(frame.groupby("patient_id")["split"].nunique().max() == 1),
        "development_only_staged": set(development["split"]) <= {"train", "validation"},
        "locked_test_not_staged": not set(locked["study_id"]) & {record["study_id"] for record in records},
        "all_development_files_local": len(records) == len(development) and all(Path(record["local_dicom"]).exists() for record in records),
        "local_hashes_recorded": all(len(record["sha256"]) == 64 for record in records),
        "ef_formula_consistent": bool(float(np.nanmax(np.abs(ef_recomputed - frame["LVEF_A4C"]))) < 1e-8),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "dataset": "MIMIC-IV-ECHO-Ext-LVVOLUMES-A4C-ROI v1.0.0",
        "official_source": SOURCE,
        "rows": len(frame),
        "patients": int(frame["patient_id"].nunique()),
        "source_documented_patients": 806,
        "patient_count_discrepancy": int(frame["patient_id"].nunique()) - 806,
        "splits": {name: int(count) for name, count in frame.groupby("split").size().items()},
        "split_patients": {name: int(count) for name, count in frame.groupby("split")["patient_id"].nunique().items()},
        "staged_development_cines": len(records),
        "staged_bytes": copied_bytes,
        "locked_test_cines": len(locked),
        "local_root": str(destination),
        "citation": CITATION,
        "licence": "PhysioNet Credentialed Health Data License 1.5.0",
        "track": "R",
        "redistribution": False,
        "clinical_use": False,
        "locked_test_accessed": False,
    }
    _atomic_json(Path(config["paths"]["evidence_root"]) / "G4" / "mimic_lv_extension.json", report)
    return report

