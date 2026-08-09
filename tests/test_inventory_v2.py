from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import pytest

from hemfm.inventory import run_dicom_inventory


def _region(*, spectral: bool = False) -> Dataset:
    item = Dataset()
    item.RegionSpatialFormat = 3 if spectral else 1
    item.RegionDataType = 3 if spectral else 1
    item.RegionFlags = 2
    item.RegionLocationMinX0 = 4
    item.RegionLocationMinY0 = 5
    item.RegionLocationMaxX1 = 100
    item.RegionLocationMaxY1 = 120
    item.ReferencePixelX0 = -2
    item.ReferencePixelY0 = 40
    item.ReferencePixelPhysicalValueX = 0.0
    item.ReferencePixelPhysicalValueY = 0.0
    item.PhysicalUnitsXDirection = 4 if spectral else 3
    item.PhysicalUnitsYDirection = 7 if spectral else 3
    item.PhysicalDeltaX = 0.01 if spectral else 0.02
    item.PhysicalDeltaY = -0.5 if spectral else 0.02
    item.TransducerFrequency = 2500
    item.PulseRepetitionFrequency = 4000
    item.DopplerCorrectionAngle = 10.0
    item.SteeringAngle = -5.0
    item.ActiveImageAreaOverlayGroup = 0x6000
    return item


def _write_dicom(path: Path, *, detailed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "US"
    dataset.Rows = 128
    dataset.Columns = 160
    dataset.NumberOfFrames = 12
    dataset.SamplesPerPixel = 3
    dataset.PhotometricInterpretation = "YBR_FULL_422"
    dataset.PlanarConfiguration = 0
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.Manufacturer = "Synthetic Vendor"
    dataset.ManufacturerModelName = "Synthetic Model"
    dataset.SoftwareVersions = ["1.2", "build-9"]
    dataset.TransducerData = "P4-2"
    dataset.ImageType = ["DERIVED", "SECONDARY", "ECHO"]
    dataset.ConversionType = "WSD"
    dataset.BurnedInAnnotation = "NO"
    dataset.AcquisitionDateTime = "20260809112233.123+0200"
    dataset.FrameTime = 40.0
    dataset.FrameTimeVector = [40.0, 42.0, 41.0]
    dataset.CineRate = 25
    dataset.RecommendedDisplayFrameRate = 24
    dataset.FrameIncrementPointer = [pydicom.tag.Tag(0x00181063)]
    dataset.SequenceOfUltrasoundRegions = Sequence(
        [_region(), _region(spectral=True)] if detailed else [_region()]
    )
    if detailed:
        dataset.add_new((0x6000, 0x0010), "US", 128)
        dataset.GraphicAnnotationSequence = Sequence([Dataset()])
    pydicom.dcmwrite(path, dataset, enforce_file_format=True)


def _record_list(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "study_id", "acquisition_datetime", "dicom_filepath"])
        for subject, study, relative in rows:
            writer.writerow([subject, study, "2026-08-09 11:22:33", relative])


def test_inventory_writes_legacy_csv_and_instance_region_parquets(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicom"
    _write_dicom(dicom_root / "a.dcm", detailed=True)
    _write_dicom(dicom_root / "b.dcm", detailed=False)
    records = tmp_path / "records.csv"
    _record_list(records, [("p1", "s1", "a.dcm"), ("p2", "s2", "b.dcm")])
    output = tmp_path / "private" / "dicom_inventory.csv"
    summary = run_dicom_inventory(
        records,
        dicom_root,
        output,
        tmp_path / "summary.json",
        workers=2,
    )

    assert summary["complete"]
    assert summary["records"] == 2
    assert summary["spatially_calibrated"] == 2
    assert summary["spectral_regions"] == 1
    assert output.exists()
    legacy = pd.read_csv(output)
    assert {"relative_path", "spatial_region_count", "manufacturer"}.issubset(legacy.columns)

    instances = pd.read_parquet(output.with_name("dicom_inventory_instances.parquet"))
    regions = pd.read_parquet(output.with_name("dicom_inventory_regions.parquet"))
    detailed = instances.loc[instances["relative_path"] == "a.dcm"].iloc[0]
    assert detailed["image_type_has_secondary"]
    assert detailed["is_secondary_capture_sop_class"]
    assert detailed["has_overlay"]
    assert detailed["has_graphic_annotation"]
    assert detailed["photometric_conversion_path"] == "YBR_FULL_422->upsample_422->RGB"
    assert detailed["frame_time_vector_median_ms"] == pytest.approx(41.0)
    assert detailed["effective_frame_rate_fps"] == pytest.approx(25.0)
    assert detailed["acquisition_datetime"].startswith("2026-08-09T11:22:33.123")
    assert detailed["measurement_screen_layout"] == (
        "purpose=spectral_doppler+two_d_tissue;capture=secondary;"
        "temporal=multiframe;annotation=overlay_and_graphics;"
        "domain=vendor_and_software"
    )
    assert detailed["measurement_screen_layout_version"] == "hemfm.metadata_layout.v1"
    assert (
        detailed["measurement_screen_layout_provenance"]
        == "metadata_only_candidate_prior_not_annotation_truth"
    )
    assert detailed["measurement_screen_layout_metadata_only"]
    assert not detailed["measurement_screen_layout_is_annotation_truth"]
    assert len(regions) == 3
    spectral = regions[regions["region_spatial_format"] == "spectral"].iloc[0]
    assert spectral["physical_units_x"] == "s"
    assert spectral["physical_units_y"] == "cm/s"
    assert spectral["physical_delta_y"] == pytest.approx(-0.5)
    assert spectral["reference_pixel_y0"] == 40
    assert spectral["is_spectral_velocity_calibrated"]
    assert summary["missing_sop_instance_uids"] == 0
    assert summary["duplicate_sop_instance_uids"] == 0
    assert summary["cross_patient_duplicate_sop_instance_uids"] == 0
    assert summary["metadata_only"] is True
    assert summary["pixel_data_read"] is False
    assert summary["locked_test_pixels_accessed"] is False
    assert {"image_type", "frames", "region_type", "vendor", "software"}.issubset(
        summary["enriched_fields"]
    )
    assert len(summary["legacy_csv_sha256"]) == 64
    assert summary["measurement_screen_layout_version"] == "hemfm.metadata_layout.v1"
    assert summary["measurement_screen_layout_is_annotation_truth"] is False
    assert sum(summary["measurement_screen_layouts"].values()) == 2


def test_inventory_surfaces_cross_patient_duplicate_sop_instances(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicom"
    _write_dicom(dicom_root / "a.dcm", detailed=True)
    shutil.copyfile(dicom_root / "a.dcm", dicom_root / "b.dcm")
    records = tmp_path / "records.csv"
    _record_list(records, [("p1", "s1", "a.dcm"), ("p2", "s2", "b.dcm")])
    summary = run_dicom_inventory(
        records,
        dicom_root,
        tmp_path / "dicom_inventory.csv",
        tmp_path / "summary.json",
        workers=1,
    )
    assert summary["duplicate_sop_instance_uids"] == 1
    assert summary["cross_patient_duplicate_sop_instance_uids"] == 1


def test_inventory_resume_repairs_csv_only_from_validated_journal(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicom"
    _write_dicom(dicom_root / "a.dcm", detailed=True)
    _write_dicom(dicom_root / "b.dcm", detailed=False)
    records = tmp_path / "records.csv"
    _record_list(records, [("p1", "s1", "a.dcm"), ("p2", "s2", "b.dcm")])
    output = tmp_path / "dicom_inventory.csv"
    summary_path = tmp_path / "summary.json"
    run_dicom_inventory(records, dicom_root, output, summary_path, workers=1)

    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    repaired = run_dicom_inventory(
        records, dicom_root, output, summary_path, workers=1, resume=True
    )
    assert repaired["records"] == 2
    assert len(pd.read_csv(output)) == 2


def test_inventory_resume_repairs_csv_payload_corruption_after_valid_identity(
    tmp_path: Path,
) -> None:
    dicom_root = tmp_path / "dicom"
    _write_dicom(dicom_root / "a.dcm", detailed=True)
    records = tmp_path / "records.csv"
    _record_list(records, [("p1", "s1", "a.dcm")])
    output = tmp_path / "dicom_inventory.csv"
    summary_path = tmp_path / "summary.json"
    run_dicom_inventory(records, dicom_root, output, summary_path, workers=1)

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    # Simulate a crash/torn write after the identity columns: the row still
    # points at the expected source instance, but its metadata is no longer the
    # transaction committed to the journal.
    rows[0]["manufacturer"] = "CORRUPTED_AFTER_IDENTITY"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    repaired = run_dicom_inventory(
        records,
        dicom_root,
        output,
        summary_path,
        workers=1,
        resume=True,
    )
    assert repaired["manufacturers"] == {"Synthetic Vendor": 1}
    assert pd.read_csv(output).iloc[0]["manufacturer"] == "Synthetic Vendor"


def test_inventory_resume_fails_when_source_or_transaction_state_changed(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicom"
    _write_dicom(dicom_root / "a.dcm", detailed=True)
    records = tmp_path / "records.csv"
    _record_list(records, [("p1", "s1", "a.dcm")])
    output = tmp_path / "dicom_inventory.csv"
    summary_path = tmp_path / "summary.json"
    run_dicom_inventory(records, dicom_root, output, summary_path, workers=1)

    _record_list(records, [("p1", "s1", "a.dcm"), ("p2", "s2", "missing.dcm")])
    with pytest.raises(RuntimeError, match="checkpoint/source mismatch"):
        run_dicom_inventory(records, dicom_root, output, summary_path, workers=1, resume=True)


def test_inventory_resume_rejects_parseable_journal_schema_corruption(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicom"
    _write_dicom(dicom_root / "a.dcm", detailed=True)
    records = tmp_path / "records.csv"
    _record_list(records, [("p1", "s1", "a.dcm")])
    output = tmp_path / "dicom_inventory.csv"
    summary_path = tmp_path / "summary.json"
    run_dicom_inventory(records, dicom_root, output, summary_path, workers=1)

    journal = output.with_name(".dicom_inventory.journal.jsonl")
    payload = json.loads(journal.read_text(encoding="utf-8"))
    payload["instance"].pop("manufacturer")
    journal.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="journal schema mismatch"):
        run_dicom_inventory(
            records,
            dicom_root,
            output,
            summary_path,
            workers=1,
            resume=True,
        )


def test_inventory_refuses_legacy_csv_without_transaction_artifacts(tmp_path: Path) -> None:
    records = tmp_path / "records.csv"
    _record_list(records, [("p1", "s1", "a.dcm")])
    output = tmp_path / "dicom_inventory.csv"
    output.write_text("record_index,subject_id\n0,p1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="transaction artifacts are missing"):
        run_dicom_inventory(
            records,
            tmp_path,
            output,
            tmp_path / "summary.json",
            workers=1,
            resume=True,
        )


def test_inventory_rejects_a_source_path_that_escapes_dicom_root(tmp_path: Path) -> None:
    records = tmp_path / "records.csv"
    _record_list(records, [("p1", "s1", "../outside.dcm")])
    with pytest.raises(ValueError, match="remain under dicom_root"):
        run_dicom_inventory(
            records,
            tmp_path / "dicom",
            tmp_path / "dicom_inventory.csv",
            tmp_path / "summary.json",
            workers=1,
        )
