from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom

from .calibration import RegionCalibration, round_trip_error, select_region, spectral_velocity_mps


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
) -> dict[str, Any]:
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
        "schema_version": 1,
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
    }
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

