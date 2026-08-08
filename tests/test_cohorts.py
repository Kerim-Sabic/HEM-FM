import json

import pandas as pd

from hemfm.cohorts import _study_cines, _study_spectral_cines


def _views():
    return pd.DataFrame(
        [
            {"subject_id": "1", "study_id": "10", "dicom_path": "a", "view": "A4C", "confidence": 0.9, "readable": True, "frames": 16, "has_physical_spatial_calibration": True, "spectral_region_count": 0, "manufacturer": "GE", "model": "E95"},
            {"subject_id": "1", "study_id": "10", "dicom_path": "b", "view": "A2C", "confidence": 0.8, "readable": True, "frames": 16, "has_physical_spatial_calibration": True, "spectral_region_count": 0, "manufacturer": "GE", "model": "E95"},
            {"subject_id": "2", "study_id": "20", "dicom_path": "c", "view": "A4C", "confidence": 0.95, "readable": True, "frames": 16, "has_physical_spatial_calibration": True, "spectral_region_count": 0, "manufacturer": "GE", "model": "E90"},
            {"subject_id": "3", "study_id": "30", "dicom_path": "d", "view": "Apical_Doppler", "confidence": 0.92, "readable": True, "frames": 16, "has_physical_spatial_calibration": False, "spectral_region_count": 1, "manufacturer": "GE", "model": "E95"},
        ]
    )


def test_biplane_requires_both_views():
    result = _study_cines(_views(), {"required_views": ["A4C", "A2C"], "calibration": "spatial"})
    assert list(result["study_id"]) == ["10"]
    assert {item["view"] for item in json.loads(result.iloc[0]["cines"])} == {"A4C", "A2C"}


def test_doppler_uses_velocity_calibrated_frame_near_anchor():
    views = _views()
    views.loc[views["study_id"] == "30", "acquisition_datetime"] = "2026-01-01 12:00:00"
    inventory = pd.DataFrame(
        [
            {"subject_id": "3", "study_id": "30", "relative_path": "spectral", "readable": True, "spectral_region_count": 1, "manufacturer": "GE", "model": "E95"}
        ]
    )
    records = pd.DataFrame(
        [
            {"subject_id": "3", "study_id": "30", "dicom_filepath": "spectral", "acquisition_datetime": "2026-01-01 12:00:30"}
        ]
    )
    result = _study_spectral_cines(
        views,
        inventory,
        records,
        {"anchor_views": ["Apical_Doppler"], "anchor_max_gap_seconds": 180},
    )
    assert list(result["study_id"]) == ["30"]
    cine = json.loads(result.iloc[0]["cines"])[0]
    assert cine["path"] == "spectral"
    assert cine["anchor_gap_seconds"] == 30.0

