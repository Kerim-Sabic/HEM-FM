from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pydicom


UNIT_CODES = {
    0x0003: "cm",
    0x0004: "s",
    0x0005: "Hz",
    0x0006: "dB",
    0x0007: "cm/s",
    0x0008: "cm2",
    0x0009: "cm2/s",
    0x000A: "cm3",
    0x000B: "cm3/s",
    0x000C: "degrees",
}


@dataclass
class DicomRow:
    record_index: int
    subject_id: str
    study_id: str
    relative_path: str
    readable: bool
    modality: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    rows: int | None = None
    columns: int | None = None
    frames: int | None = None
    photometric: str | None = None
    region_count: int = 0
    spatial_region_count: int = 0
    spectral_region_count: int = 0
    multi_region: bool = False
    has_physical_spatial_calibration: bool = False
    error: str | None = None


def _value(item: Any, default: Any = None) -> Any:
    return getattr(item, "value", default) if item is not None else default


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _regions(dataset: pydicom.Dataset) -> tuple[int, int, int]:
    sequence = getattr(dataset, "SequenceOfUltrasoundRegions", []) or []
    spatial = 0
    spectral = 0
    for region in sequence:
        x_code = _as_int(getattr(region, "PhysicalUnitsXDirection", None))
        y_code = _as_int(getattr(region, "PhysicalUnitsYDirection", None))
        x_unit = UNIT_CODES.get(x_code)
        y_unit = UNIT_CODES.get(y_code)
        dx = getattr(region, "PhysicalDeltaX", None)
        dy = getattr(region, "PhysicalDeltaY", None)
        if x_unit == "cm" and y_unit == "cm" and dx and dy:
            spatial += 1
        if ({x_unit, y_unit} & {"cm/s"}) and dx and dy:
            spectral += 1
    return len(sequence), spatial, spectral


def inspect_record(record: tuple[int, str, str, str], dicom_root: Path) -> DicomRow:
    index, subject_id, study_id, relative_path = record
    path = dicom_root / relative_path.replace("/", "\\")
    row = DicomRow(index, str(subject_id), str(study_id), relative_path, False)
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        region_count, spatial_count, spectral_count = _regions(dataset)
        row.readable = True
        row.modality = str(dataset.get("Modality", "")) or None
        row.manufacturer = str(dataset.get("Manufacturer", "")) or None
        row.model = str(dataset.get("ManufacturerModelName", "")) or None
        row.rows = _as_int(dataset.get("Rows"))
        row.columns = _as_int(dataset.get("Columns"))
        row.frames = _as_int(dataset.get("NumberOfFrames")) or 1
        row.photometric = str(dataset.get("PhotometricInterpretation", "")) or None
        row.region_count = region_count
        row.spatial_region_count = spatial_count
        row.spectral_region_count = spectral_count
        row.multi_region = region_count > 1
        row.has_physical_spatial_calibration = spatial_count > 0
    except Exception as exc:  # inventory must preserve failures rather than abort a corpus scan
        row.error = f"{type(exc).__name__}: {str(exc)[:240]}"
    return row


def _read_records(path: Path, start: int = 0, limit: int | None = None) -> list[tuple[int, str, str, str]]:
    frame = pd.read_csv(path, dtype=str)
    required = {"subject_id", "study_id", "dicom_filepath"}
    if missing := required.difference(frame.columns):
        raise KeyError(f"Record list is missing: {sorted(missing)}")
    if limit is not None:
        frame = frame.iloc[start : start + limit]
    else:
        frame = frame.iloc[start:]
    return [
        (int(index), row.subject_id, row.study_id, row.dicom_filepath)
        for index, row in frame.iterrows()
    ]


def run_dicom_inventory(
    record_list: str | Path,
    dicom_root: str | Path,
    output_csv: str | Path,
    summary_json: str | Path,
    workers: int = 16,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    if resume and output_path.exists():
        with output_path.open("r", encoding="utf-8", newline="") as handle:
            completed = max(0, sum(1 for _ in handle) - 1)
    records = _read_records(Path(record_list), start=completed, limit=limit)
    fieldnames = list(DicomRow.__dataclass_fields__)
    mode = "a" if completed else "w"
    counters: Counter[str] = Counter()
    manufacturer_counts: Counter[str] = Counter()
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not completed:
            writer.writeheader()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for row in executor.map(lambda record: inspect_record(record, Path(dicom_root)), records, chunksize=32):
                writer.writerow(asdict(row))
                counters["rows"] += 1
                counters["readable"] += int(row.readable)
                counters["calibrated"] += int(row.has_physical_spatial_calibration)
                counters["multi_region"] += int(row.multi_region)
                counters["spectral"] += int(row.spectral_region_count > 0)
                if row.manufacturer:
                    manufacturer_counts[row.manufacturer] += 1
                if counters["rows"] % 1000 == 0:
                    handle.flush()
    all_rows = pd.read_csv(output_path, low_memory=False)
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": limit is None and len(all_rows) == sum(1 for _ in Path(record_list).open("r", encoding="utf-8")) - 1,
        "records": int(len(all_rows)),
        "readable": int(all_rows["readable"].fillna(False).astype(bool).sum()),
        "spatially_calibrated": int(all_rows["has_physical_spatial_calibration"].fillna(False).astype(bool).sum()),
        "multi_region": int(all_rows["multi_region"].fillna(False).astype(bool).sum()),
        "spectral_regions": int((all_rows["spectral_region_count"].fillna(0) > 0).sum()),
        "manufacturers": all_rows["manufacturer"].fillna("UNKNOWN").value_counts().head(50).astype(int).to_dict(),
        "errors": int(all_rows["error"].notna().sum()),
    }
    Path(summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_label_inventory(measurements_path: str | Path, study_list_path: str | Path, targets: dict[str, Any]) -> dict:
    wanted = {name for target in targets.values() for name in target["measurements"]}
    measurements = pd.read_csv(
        measurements_path,
        compression="gzip",
        usecols=["subject_id", "measurement_id", "measurement", "result", "unit"],
        dtype=str,
        low_memory=False,
    )
    measurements = measurements[measurements["measurement"].isin(wanted)].copy()
    measurements["numeric_result"] = pd.to_numeric(measurements["result"], errors="coerce")
    studies = pd.read_csv(study_list_path, dtype={"subject_id": str, "study_id": str, "measurement_id": str})
    measurements["subject_id"] = measurements["subject_id"].astype(str)
    measurements["measurement_id"] = measurements["measurement_id"].astype(str)
    linked = measurements.merge(studies[["subject_id", "study_id", "measurement_id"]], on=["subject_id", "measurement_id"], how="left")
    output: dict[str, Any] = {"schema_version": 1, "targets": {}}
    for target_name, target in targets.items():
        subset = linked[linked["measurement"].isin(target["measurements"]) & linked["numeric_result"].notna()]
        linked_subset = subset[subset["study_id"].notna()]
        output["targets"][target_name] = {
            "numeric_rows": int(len(subset)),
            "patients_total_measurement_table": int(subset["subject_id"].nunique()),
            "linked_numeric_rows": int(len(linked_subset)),
            "linked_patients": int(linked_subset["subject_id"].nunique()),
            "linked_studies": int(linked_subset["study_id"].nunique()),
            "source_measurements": subset["measurement"].value_counts().astype(int).to_dict(),
            "reported_units": subset["unit"].fillna("UNKNOWN").value_counts().astype(int).to_dict(),
        }
    return output

