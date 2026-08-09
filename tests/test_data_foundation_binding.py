from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import pytest

from hemfm.calibration_validation import (
    _validate_inventory_binding,
    validate_real_calibration,
)
from hemfm.cli import _assert_mimic_development_split_ready
from hemfm.splits import build_hashed_split_manifest, load_hashed_split_assignments


ENRICHED_FIELDS = [
    "image_type",
    "frames",
    "region_type",
    "vendor",
    "software",
    "measurement_screen_layout",
    "measurement_screen_layout_provenance",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_records(path: Path, patients: int = 80) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "study_id", "dicom_filepath"])
        for patient in range(patients):
            writer.writerow(
                [f"patient-{patient}", f"study-{patient}", f"files/{patient}.dcm"]
            )


def _write_inventory_evidence(
    evidence_path: Path,
    private: Path,
    *,
    manifest_sha256: str,
    record_list_sha256: str,
) -> dict:
    private.mkdir(parents=True, exist_ok=True)
    inventory_csv = private / "dicom_inventory.csv"
    instance_parquet = private / "dicom_inventory_instances.parquet"
    region_parquet = private / "dicom_inventory_regions.parquet"
    inventory_csv.write_text("record_index,subject_id\n0,patient-0\n", encoding="utf-8")
    instance_parquet.write_bytes(b"synthetic-instance-parquet")
    region_parquet.write_bytes(b"synthetic-region-parquet")
    payload = {
        "schema_version": 2,
        "passed": True,
        "status": "complete",
        "complete": True,
        "scan_target_complete": True,
        "metadata_only": True,
        "pixel_data_read": False,
        "locked_test_pixels_accessed": False,
        "enriched_fields": ENRICHED_FIELDS,
        "measurement_screen_layout_version": "hemfm.metadata_layout.v1",
        "measurement_screen_layout_provenance": (
            "metadata_only_candidate_prior_not_annotation_truth"
        ),
        "measurement_screen_layout_is_annotation_truth": False,
        "split_manifest_preceded_decoding": True,
        "split_manifest_sha256": manifest_sha256,
        "record_list_sha256": record_list_sha256,
        "execution_checks": {"synthetic_complete": True},
        "legacy_csv": str(inventory_csv),
        "legacy_csv_sha256": _sha256(inventory_csv),
        "instance_parquet": str(instance_parquet),
        "instance_parquet_sha256": _sha256(instance_parquet),
        "region_parquet": str(region_parquet),
        "region_parquet_sha256": _sha256(region_parquet),
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_calibration_binding_rejects_changed_v2_inventory_artifact(tmp_path: Path) -> None:
    summary_path = tmp_path / "evidence" / "dicom_inventory_summary.json"
    manifest_sha256 = "a" * 64
    record_list_sha256 = "b" * 64
    summary = _write_inventory_evidence(
        summary_path,
        tmp_path / "private",
        manifest_sha256=manifest_sha256,
        record_list_sha256=record_list_sha256,
    )
    binding = _validate_inventory_binding(
        summary["legacy_csv"],
        summary_path,
        expected_split_manifest_sha256=manifest_sha256,
        expected_record_list_sha256=record_list_sha256,
    )
    assert all(binding["checks"].values())
    assert binding["metadata_only"] is True
    assert binding["pixel_data_read"] is False
    assert binding["locked_test_pixels_accessed"] is False

    Path(summary["region_parquet"]).write_bytes(b"changed-after-inventory")
    with pytest.raises(RuntimeError, match="changed inventory artifacts"):
        _validate_inventory_binding(
            summary["legacy_csv"],
            summary_path,
            expected_split_manifest_sha256=manifest_sha256,
            expected_record_list_sha256=record_list_sha256,
        )


def test_calibration_report_remains_metadata_only_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_path = tmp_path / "evidence" / "dicom_inventory_summary.json"
    manifest_sha256 = "a" * 64
    record_list_sha256 = "b" * 64
    summary = _write_inventory_evidence(
        summary_path,
        tmp_path / "private",
        manifest_sha256=manifest_sha256,
        record_list_sha256=record_list_sha256,
    )
    inventory_path = Path(summary["legacy_csv"])
    pd.DataFrame(
        {
            "relative_path": [f"synthetic/{index}.dcm" for index in range(100)],
            "readable": True,
            "has_physical_spatial_calibration": True,
            "spatial_region_count": ([2] * 33) + ([1] * 67),
            "spectral_region_count": ([1] * 33) + ([0] * 67),
        }
    ).to_csv(inventory_path, index=False)
    summary["legacy_csv_sha256"] = _sha256(inventory_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    large = Dataset()
    large.PhysicalUnitsXDirection = 3
    large.PhysicalUnitsYDirection = 3
    large.RegionLocationMinX0 = 0
    large.RegionLocationMinY0 = 0
    large.RegionLocationMaxX1 = 100
    large.RegionLocationMaxY1 = 100
    large.PhysicalDeltaX = 0.02
    large.PhysicalDeltaY = 0.02
    large.RegionDataType = 1
    small = Dataset()
    small.PhysicalUnitsXDirection = 3
    small.PhysicalUnitsYDirection = 3
    small.RegionLocationMinX0 = 10
    small.RegionLocationMinY0 = 10
    small.RegionLocationMaxX1 = 20
    small.RegionLocationMaxY1 = 20
    small.PhysicalDeltaX = 0.01
    small.PhysicalDeltaY = 0.01
    small.RegionDataType = 1
    spectral = Dataset()
    spectral.PhysicalUnitsXDirection = 4
    spectral.PhysicalUnitsYDirection = 7
    spectral.PhysicalDeltaX = 0.01
    spectral.PhysicalDeltaY = -0.5
    dataset = Dataset()
    dataset.SequenceOfUltrasoundRegions = Sequence([large, small, spectral])
    calls = []

    def metadata_only_dcmread(path, *, stop_before_pixels, force):
        calls.append((path, stop_before_pixels, force))
        assert stop_before_pixels is True
        return dataset

    monkeypatch.setattr(pydicom, "dcmread", metadata_only_dcmread)
    report = validate_real_calibration(
        inventory_path,
        tmp_path / "dicom",
        tmp_path / "calibration.json",
        max_files=100,
        inventory_summary_json=summary_path,
        expected_split_manifest_sha256=manifest_sha256,
        expected_record_list_sha256=record_list_sha256,
    )
    assert report["passed"]
    assert len(calls) == 100
    assert report["schema_version"] == 2
    assert report["metadata_only"] is True
    assert report["pixel_data_read"] is False
    assert report["locked_test_pixels_accessed"] is False
    assert report["inventory_csv_sha256"] == summary["legacy_csv_sha256"]


def test_mimic_training_preflight_requires_matching_v2_foundation(tmp_path: Path) -> None:
    record_list = tmp_path / "records.csv"
    private = tmp_path / "private"
    evidence = tmp_path / "evidence"
    _write_records(record_list)
    split_report = build_hashed_split_manifest(
        record_list,
        private,
        seed=20260807,
        train=0.60,
        validation=0.20,
    )
    assignments, _ = load_hashed_split_assignments(
        record_list,
        private,
        seed=20260807,
        train=0.60,
        validation=0.20,
    )
    development = assignments[assignments["split"].isin(["train", "validation"])].copy()
    development["test_locked"] = development["sealed_internal_test"]
    development["split_manifest_sha256"] = split_report["manifest_sha256"]
    development.to_csv(private / "endpoint_cohorts_development_private.csv", index=False)

    g0 = evidence / "G0"
    g1 = evidence / "G1"
    g0.mkdir(parents=True)
    (g0 / "runtime.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    g1.mkdir(parents=True)
    split_evidence_path = g1 / "split_audit.json"
    split_evidence_path.write_text(json.dumps(split_report), encoding="utf-8")
    inventory_summary_path = g1 / "dicom_inventory_summary.json"
    inventory = _write_inventory_evidence(
        inventory_summary_path,
        private,
        manifest_sha256=split_report["manifest_sha256"],
        record_list_sha256=split_report["record_list_sha256"],
    )
    calibration = {
        "schema_version": 2,
        "passed": True,
        "metadata_only": True,
        "pixel_data_read": False,
        "locked_test_pixels_accessed": False,
        "inventory_summary_sha256": _sha256(inventory_summary_path),
        "inventory_csv_sha256": inventory["legacy_csv_sha256"],
        "instance_parquet_sha256": inventory["instance_parquet_sha256"],
        "region_parquet_sha256": inventory["region_parquet_sha256"],
        "record_list_sha256": split_report["record_list_sha256"],
        "split_manifest_sha256": split_report["manifest_sha256"],
        "inventory_binding": {"checks": {"all_exact": True}},
    }
    calibration_path = g1 / "calibration_tests.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    config = {
        "paths": {
            "private_root": str(private),
            "record_list": str(record_list),
            "evidence_root": str(evidence),
        },
        "splits": {"seed": 20260807, "train": 0.60, "validation": 0.20},
        "gates": {
            "G0": {"execution_required_evidence": ["runtime.json"]},
            "G1": {
                "execution_required_evidence": [
                    "dicom_inventory_summary.json",
                    "calibration_tests.json",
                    "split_audit.json",
                ]
            },
        },
    }
    report = _assert_mimic_development_split_ready(config)
    assert report["passed"]
    assert all(report["checks"].values())

    calibration["split_manifest_sha256"] = "f" * 64
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or mismatched data-foundation evidence"):
        _assert_mimic_development_split_ready(config)
