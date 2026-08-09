from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom

from .calibration import RegionCalibration, round_trip_error, select_region, spectral_velocity_mps
from .hashing import sha256_file


CALIBRATION_EVIDENCE_SCHEMA_VERSION = 2


def _validate_inventory_binding(
    inventory_csv: str | Path,
    inventory_summary_json: str | Path,
    *,
    expected_split_manifest_sha256: str,
    expected_record_list_sha256: str,
) -> dict[str, Any]:
    """Validate the exact v2 inventory lineage before sampling any DICOM."""
    inventory_path = Path(inventory_csv)
    summary_path = Path(inventory_summary_json)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Calibration requires readable v2 inventory evidence") from exc
    enriched_fields = set(summary.get("enriched_fields", []) or [])
    required_enriched_fields = {
        "image_type",
        "frames",
        "region_type",
        "vendor",
        "software",
        "measurement_screen_layout",
        "measurement_screen_layout_provenance",
    }
    required_fields = {
        "schema_v2_inventory": summary.get("schema_version") == 2,
        "inventory_declared_passed": summary.get("passed") is True,
        "inventory_complete": (
            summary.get("complete") is True
            and summary.get("scan_target_complete") is True
            and summary.get("status") == "complete"
        ),
        "metadata_only_inventory": (
            summary.get("metadata_only") is True
            and summary.get("pixel_data_read") is False
        ),
        "locked_test_pixels_sealed": summary.get("locked_test_pixels_accessed") is False,
        "required_enrichment_present": required_enriched_fields.issubset(enriched_fields),
        "layout_is_candidate_prior": (
            summary.get("measurement_screen_layout_provenance")
            == "metadata_only_candidate_prior_not_annotation_truth"
            and summary.get("measurement_screen_layout_is_annotation_truth") is False
        ),
        "manifest_hash_matches": (
            summary.get("split_manifest_sha256") == expected_split_manifest_sha256
        ),
        "record_list_hash_matches": (
            summary.get("record_list_sha256") == expected_record_list_sha256
        ),
    }
    if failed := [name for name, passed in required_fields.items() if not passed]:
        raise RuntimeError(
            "Calibration refused stale or unbound inventory evidence: " + ", ".join(failed)
        )

    summary_inventory_path = Path(str(summary.get("legacy_csv", "")))
    artifact_specs = {
        "inventory_csv": (
            inventory_path,
            summary_inventory_path,
            "legacy_csv_sha256",
        ),
        "instance_parquet": (
            Path(str(summary.get("instance_parquet", ""))),
            Path(str(summary.get("instance_parquet", ""))),
            "instance_parquet_sha256",
        ),
        "region_parquet": (
            Path(str(summary.get("region_parquet", ""))),
            Path(str(summary.get("region_parquet", ""))),
            "region_parquet_sha256",
        ),
    }
    hashes: dict[str, str] = {}
    artifact_checks: dict[str, bool] = {}
    for name, (actual_path, declared_path, hash_field) in artifact_specs.items():
        declared_hash = str(summary.get(hash_field, ""))
        path_matches = actual_path.resolve() == declared_path.resolve()
        exists = actual_path.is_file()
        actual_hash = sha256_file(actual_path) if exists else ""
        artifact_checks[f"{name}_path_matches"] = path_matches
        artifact_checks[f"{name}_exists"] = exists
        artifact_checks[f"{name}_hash_matches"] = bool(
            len(declared_hash) == 64 and actual_hash == declared_hash
        )
        hashes[f"{name}_sha256"] = actual_hash
    if failed := [name for name, passed in artifact_checks.items() if not passed]:
        raise RuntimeError(
            "Calibration refused changed inventory artifacts: " + ", ".join(failed)
        )
    return {
        "checks": {**required_fields, **artifact_checks},
        "inventory_schema_version": 2,
        "inventory_summary_sha256": sha256_file(summary_path),
        "inventory_csv_sha256": hashes["inventory_csv_sha256"],
        "instance_parquet_sha256": hashes["instance_parquet_sha256"],
        "region_parquet_sha256": hashes["region_parquet_sha256"],
        "record_list_sha256": expected_record_list_sha256,
        "split_manifest_sha256": expected_split_manifest_sha256,
        "metadata_only": True,
        "pixel_data_read": False,
        "locked_test_pixels_accessed": False,
    }


def _spatial_regions(dataset: pydicom.Dataset) -> list[RegionCalibration]:
    result = []
    sequence = getattr(dataset, "SequenceOfUltrasoundRegions", []) or []
    for item in sequence:
        try:
            if int(item.PhysicalUnitsXDirection) != 0x0003 or int(item.PhysicalUnitsYDirection) != 0x0003:
                continue
            result.append(
                RegionCalibration(
                    x0=int(item.RegionLocationMinX0),
                    y0=int(item.RegionLocationMinY0),
                    x1=int(item.RegionLocationMaxX1),
                    y1=int(item.RegionLocationMaxY1),
                    delta_x=float(item.PhysicalDeltaX),
                    delta_y=float(item.PhysicalDeltaY),
                    x_unit="cm",
                    y_unit="cm",
                    region_type=str(getattr(item, "RegionDataType", "unknown")),
                )
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return result


def _spectral_checks(dataset: pydicom.Dataset) -> int:
    checks = 0
    sequence = getattr(dataset, "SequenceOfUltrasoundRegions", []) or []
    for item in sequence:
        try:
            x_unit = int(item.PhysicalUnitsXDirection)
            y_unit = int(item.PhysicalUnitsYDirection)
            if x_unit == 0x0007:
                value = spectral_velocity_mps(float(item.PhysicalDeltaX), "cm/s", 0.0, 10.0)
                checks += int(np.isfinite(value) and value != 0)
            if y_unit == 0x0007:
                value = spectral_velocity_mps(float(item.PhysicalDeltaY), "cm/s", 0.0, 10.0)
                checks += int(np.isfinite(value) and value != 0)
        except (AttributeError, TypeError, ValueError):
            continue
    return checks


def validate_real_calibration(
    inventory_csv: str | Path,
    dicom_root: str | Path,
    output_json: str | Path,
    max_files: int = 512,
    *,
    inventory_summary_json: str | Path,
    expected_split_manifest_sha256: str,
    expected_record_list_sha256: str,
) -> dict[str, Any]:
    if max_files < 1:
        raise ValueError("max_files must be positive")
    binding = _validate_inventory_binding(
        inventory_csv,
        inventory_summary_json,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
        expected_record_list_sha256=expected_record_list_sha256,
    )
    inventory = pd.read_csv(inventory_csv, low_memory=False)
    candidates = inventory[
        inventory["readable"].fillna(False).astype(bool)
        & inventory["has_physical_spatial_calibration"].fillna(False).astype(bool)
    ].copy()
    multi = candidates[candidates["spatial_region_count"].fillna(0) > 1]
    spectral = candidates[candidates["spectral_region_count"].fillna(0) > 0]
    ordinary = candidates.drop(index=multi.index.union(spectral.index), errors="ignore")
    sample = pd.concat(
        [multi.head(max_files // 3), spectral.head(max_files // 3), ordinary.head(max_files)],
        ignore_index=True,
    ).drop_duplicates(subset=["relative_path"]).head(max_files)
    errors = []
    max_error = 0.0
    regions_checked = 0
    multi_region_files = 0
    point_specific_selections = 0
    spectral_checks = 0
    for row in sample.itertuples(index=False):
        path = Path(dicom_root) / str(row.relative_path).replace("/", "\\")
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
            regions = _spatial_regions(dataset)
            if not regions:
                raise ValueError("Inventory declared calibration but no valid two-length-axis region was reconstructed")
            for region in regions:
                points = np.asarray(
                    [
                        [region.x0, region.y0],
                        [region.x1, region.y1],
                        [(region.x0 + region.x1) / 2.0, (region.y0 + region.y1) / 2.0],
                    ],
                    dtype=float,
                )
                max_error = max(max_error, round_trip_error(region, points))
                regions_checked += 1
            if len(regions) > 1:
                multi_region_files += 1
                smallest = min(regions, key=lambda region: region.area)
                center = ((smallest.x0 + smallest.x1) / 2.0, (smallest.y0 + smallest.y1) / 2.0)
                selected = select_region(regions, center)
                point_specific_selections += int(selected == smallest)
            spectral_checks += _spectral_checks(dataset)
        except Exception as exc:
            errors.append({"relative_path": str(row.relative_path), "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
    report = {
        "schema_version": CALIBRATION_EVIDENCE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(
            len(sample) >= 100
            and not errors
            and max_error < 1e-6
            and regions_checked >= len(sample)
            and multi_region_files > 0
            and point_specific_selections == multi_region_files
            and spectral_checks > 0
        ),
        "files_checked": int(len(sample)),
        "regions_checked": regions_checked,
        "multi_region_files": multi_region_files,
        "point_specific_selections": point_specific_selections,
        "spectral_axis_checks": spectral_checks,
        "max_round_trip_error_pixels": max_error,
        "errors": errors[:50],
        "inventory_binding": binding,
        "inventory_summary_sha256": binding["inventory_summary_sha256"],
        "inventory_csv_sha256": binding["inventory_csv_sha256"],
        "instance_parquet_sha256": binding["instance_parquet_sha256"],
        "region_parquet_sha256": binding["region_parquet_sha256"],
        "record_list_sha256": binding["record_list_sha256"],
        "split_manifest_sha256": binding["split_manifest_sha256"],
        "metadata_only": True,
        "pixel_data_read": False,
        "locked_test_pixels_accessed": False,
    }
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
