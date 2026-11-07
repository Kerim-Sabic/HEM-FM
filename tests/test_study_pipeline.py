from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hemfm.study_pipeline import (
    StudyDicom,
    contour_traces,
    discover_single_study,
    study_reference,
    supported_measurement_routes,
    temporal_probability_filter,
    temporally_tracked_masks,
)


def _write_metadata_dicom(
    path: Path, *, patient_id: str, study_uid: str, sop_uid: str
) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.Modality = "US"
    dataset.PatientID = patient_id
    dataset.StudyInstanceUID = study_uid
    dataset.NumberOfFrames = "20"
    dataset.save_as(path, enforce_file_format=True)


def test_temporal_filter_and_tracking_preserve_chamber_identity() -> None:
    probability = np.zeros((3, 3, 12, 12), dtype=np.float32)
    probability[:, 0] = 0.05
    probability[:, 1, 2:7, 2:7] = 0.90
    probability[:, 2, 7:10, 7:10] = 0.85
    probability[1, 1, 10:12, 0:2] = 0.99
    probability /= probability.sum(axis=1, keepdims=True)

    filtered = temporal_probability_filter(probability)
    masks = temporally_tracked_masks(filtered)

    assert masks.shape == (3, 12, 12)
    assert np.all(masks[:, 3:6, 3:6] == 1)
    assert not np.any(masks[1, 10:12, 0:2] == 1)
    assert np.allclose(filtered.sum(axis=1), 1.0)


def test_contours_are_frame_indexed_xy_traces() -> None:
    mask = np.zeros((2, 8, 8), dtype=np.uint8)
    mask[:, 2:6, 2:6] = 1

    traces = contour_traces(mask, ("background", "LV"))

    assert [item["frame_index"] for item in traces] == [0, 1]
    assert traces[0]["chambers"]["LV"]
    assert len(traces[0]["chambers"]["LV"][0][0]) == 2


def test_measurement_routes_require_accepted_complete_views() -> None:
    classifications = [
        {"path": "a4c.dcm", "predicted_view": "A4C", "confidence": 0.98,
         "accepted_by_development_confidence_policy": True},
        {"path": "a2c-low.dcm", "predicted_view": "A2C", "confidence": 0.80,
         "accepted_by_development_confidence_policy": False},
        {"path": "plax.dcm", "predicted_view": "PLAX", "confidence": 0.99,
         "accepted_by_development_confidence_policy": True},
    ]

    routes = supported_measurement_routes(classifications)

    assert "EF" not in routes
    assert routes["RV_BASAL_DIAMETER"][0]["path"] == "a4c.dcm"
    assert routes["LVOT_DIAMETER"][0]["view"] == "Parasternal_Long"


def test_study_reference_is_stable_and_contains_no_identifier() -> None:
    record = StudyDicom(Path("cine.dcm"), "patient-secret", "1.2.3", "4.5.6", 30)
    first = study_reference([record])
    second = study_reference([record])

    assert first == second
    assert "patient" not in first


def test_temporal_filter_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="positive odd"):
        temporal_probability_filter(np.ones((2, 2, 4, 4), dtype=np.float32), window=2)


def test_discovery_fails_closed_on_mixed_patients(tmp_path: Path) -> None:
    _write_metadata_dicom(
        tmp_path / "first.dcm", patient_id="one", study_uid="1.2.3", sop_uid="1.2.3.1"
    )
    _write_metadata_dicom(
        tmp_path / "second.dcm", patient_id="two", study_uid="1.2.3", sop_uid="1.2.3.2"
    )

    with pytest.raises(ValueError, match="more than one DICOM PatientID"):
        discover_single_study(tmp_path)


def test_discovery_accepts_one_study_and_skips_non_dicom(tmp_path: Path) -> None:
    _write_metadata_dicom(
        tmp_path / "cine.dcm", patient_id="one", study_uid="1.2.3", sop_uid="1.2.3.1"
    )
    (tmp_path / "notes.txt").write_text("not dicom", encoding="utf-8")

    records, skipped = discover_single_study(tmp_path)

    assert len(records) == 1
    assert records[0].frames == 20
    assert skipped[0]["file"] == "notes.txt"
