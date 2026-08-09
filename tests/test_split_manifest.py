from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from hemfm.splits import (
    audit_development_cohort_alignment,
    audit_hashed_split_manifest,
    build_hashed_split_manifest,
    load_hashed_split_assignments,
    split_artifact_paths,
)


def _write_records(path: Path, patient_count: int = 80) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "study_id", "acquisition_datetime", "dicom_filepath"])
        for patient in range(patient_count):
            for study in range(2):
                writer.writerow(
                    [
                        f"raw-patient-{patient}",
                        f"raw-study-{patient}-{study}",
                        "2026-01-01 00:00:00",
                        f"files/raw-patient-{patient}/study-{study}/instance.dcm",
                    ]
                )


def test_private_manifest_is_hashed_patient_disjoint_immutable_and_sealed(tmp_path: Path) -> None:
    record_list = tmp_path / "records.csv"
    private = tmp_path / "private"
    _write_records(record_list)
    report = build_hashed_split_manifest(
        record_list,
        private,
        seed=20260807,
        train=0.60,
        validation=0.20,
    )
    paths = split_artifact_paths(private)
    manifest = pd.read_parquet(paths["manifest"])

    assert report["passed"]
    assert report["immutable"]
    assert report["raw_identifiers_present"] is False
    assert paths["secret"].read_bytes() and len(paths["secret"].read_bytes()) == 32
    assert set(manifest.columns) == {
        "patient_hash",
        "study_hash",
        "split",
        "sealed_internal_test",
        "instance_count",
        "source_instances_sha256",
    }
    assert manifest.groupby("patient_hash")["split"].nunique().max() == 1
    assert (manifest["sealed_internal_test"] == manifest["split"].eq("internal_test")).all()
    assert manifest["patient_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert not any("raw-patient" in str(value) for value in manifest.astype(str).to_numpy().ravel())

    artifact_hash = hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()
    reused = audit_hashed_split_manifest(
        record_list,
        private,
        seed=20260807,
        train=0.60,
        validation=0.20,
    )
    assert reused["reused"]
    assert hashlib.sha256(paths["manifest"].read_bytes()).hexdigest() == artifact_hash

    _write_records(record_list, patient_count=81)
    with pytest.raises(RuntimeError, match="immutable split manifest"):
        build_hashed_split_manifest(
            record_list,
            private,
            seed=20260807,
            train=0.60,
            validation=0.20,
        )


def test_manifest_deduplicates_only_exact_full_metadata_rows(tmp_path: Path) -> None:
    record_list = tmp_path / "records.csv"
    _write_records(record_list)
    source = pd.read_csv(record_list, dtype=str, keep_default_na=False)
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    source.to_csv(record_list, index=False)
    private = tmp_path / "private"

    report = build_hashed_split_manifest(record_list, private, seed=20260807)
    manifest = pd.read_parquet(split_artifact_paths(private)["manifest"])

    assert report["passed"]
    assert report["instances"] == 160
    assert int(manifest["instance_count"].sum()) == 160
    assert report["source_normalization"] == {
        "policy": "drop_only_exact_full_metadata_rows_v1",
        "source_rows": 161,
        "normalized_rows": 160,
        "exact_duplicate_rows_removed": 1,
        "canonical_paths_unique": True,
        "path_aliases_accepted": False,
        "conflicting_metadata_accepted": False,
        "conflicting_owners_accepted": False,
    }


@pytest.mark.parametrize(
    ("second_subject", "second_study"),
    [("patient-b", "study-b"), ("patient-a", "study-b")],
)
def test_manifest_rejects_conflicting_owner_for_same_source_path_before_secret(
    tmp_path: Path, second_subject: str, second_study: str
) -> None:
    record_list = tmp_path / "records.csv"
    with record_list.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "study_id", "dicom_filepath"])
        writer.writerow(["patient-a", "study-a", "same.dcm"])
        writer.writerow([second_subject, second_study, "same.dcm"])
    private = tmp_path / "private"
    with pytest.raises(ValueError, match="conflicting patient/study owners"):
        build_hashed_split_manifest(record_list, private, seed=1)
    assert not split_artifact_paths(private)["secret"].exists()


def test_manifest_rejects_windows_path_aliases_and_traversal(tmp_path: Path) -> None:
    record_list = tmp_path / "records.csv"
    with record_list.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "study_id", "dicom_filepath"])
        writer.writerow(["patient-a", "study-a", "folder/A.dcm"])
        writer.writerow(["patient-a", "study-a", r"folder\a.dcm"])
    private = tmp_path / "private"
    with pytest.raises(ValueError, match="path alias spellings"):
        build_hashed_split_manifest(record_list, private, seed=1)
    assert not split_artifact_paths(private)["secret"].exists()

    with record_list.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "study_id", "dicom_filepath"])
        writer.writerow(["patient-a", "study-a", "../outside.dcm"])
    with pytest.raises(ValueError, match="remain under dicom_root"):
        build_hashed_split_manifest(record_list, private, seed=1)


def test_manifest_rejects_same_owner_path_with_conflicting_metadata(tmp_path: Path) -> None:
    record_list = tmp_path / "records.csv"
    with record_list.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["subject_id", "study_id", "acquisition_datetime", "dicom_filepath"]
        )
        writer.writerow(["patient-a", "study-a", "2026-01-01", "same.dcm"])
        writer.writerow(["patient-a", "study-a", "2026-01-02", "same.dcm"])
    private = tmp_path / "private"
    with pytest.raises(ValueError, match="conflicting metadata"):
        build_hashed_split_manifest(record_list, private, seed=1)
    assert not split_artifact_paths(private)["secret"].exists()


def test_manifest_rejects_a_study_owned_by_multiple_patients(tmp_path: Path) -> None:
    record_list = tmp_path / "records.csv"
    with record_list.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "study_id", "dicom_filepath"])
        writer.writerow(["patient-a", "shared-study", "a.dcm"])
        writer.writerow(["patient-b", "shared-study", "b.dcm"])
    with pytest.raises(ValueError, match="multiple patients"):
        build_hashed_split_manifest(record_list, tmp_path / "private", seed=1)


def test_hmac_secret_is_written_as_exact_binary_bytes_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_list = tmp_path / "records.csv"
    _write_records(record_list)
    secret = b"A" * 15 + b"\n" + b"B" * 16
    monkeypatch.setattr("hemfm.splits.secrets.token_bytes", lambda size: secret)
    private = tmp_path / "private"
    build_hashed_split_manifest(record_list, private, seed=20260807)
    assert split_artifact_paths(private)["secret"].read_bytes() == secret


def test_manifest_audit_rejects_a_reopened_internal_test_seal(tmp_path: Path) -> None:
    record_list = tmp_path / "records.csv"
    private = tmp_path / "private"
    _write_records(record_list)
    build_hashed_split_manifest(record_list, private, seed=20260807)
    seal_path = split_artifact_paths(private)["internal_test_seal"]
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["state"] = "open"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cryptographic seal is invalid"):
        audit_hashed_split_manifest(record_list, private, seed=20260807)


def test_development_cohort_consumes_manifest_and_rejects_split_drift(tmp_path: Path) -> None:
    record_list = tmp_path / "records.csv"
    private = tmp_path / "private"
    _write_records(record_list)
    built = build_hashed_split_manifest(
        record_list,
        private,
        seed=20260807,
        train=0.60,
        validation=0.20,
    )
    assignments, resolved = load_hashed_split_assignments(
        record_list,
        private,
        seed=20260807,
        train=0.60,
        validation=0.20,
    )
    assert resolved["manifest_sha256"] == built["manifest_sha256"]

    development = assignments[assignments["split"].isin(["train", "validation"])].head(12).copy()
    development["test_locked"] = development["sealed_internal_test"]
    development["split_manifest_sha256"] = built["manifest_sha256"]
    cohort_path = private / "endpoint_cohorts_development_private.csv"
    development.to_csv(cohort_path, index=False)
    report = audit_development_cohort_alignment(
        cohort_path,
        record_list,
        private,
        seed=20260807,
        train=0.60,
        validation=0.20,
    )
    assert report["passed"]
    assert report["manifest_alignment_exact"]

    tampered = development.copy()
    first = tampered.index[0]
    tampered.loc[first, "split"] = (
        "validation" if tampered.loc[first, "split"] == "train" else "train"
    )
    tampered.to_csv(cohort_path, index=False)
    with pytest.raises(RuntimeError, match="disagrees with the immutable split manifest"):
        audit_development_cohort_alignment(
            cohort_path,
            record_list,
            private,
            seed=20260807,
            train=0.60,
            validation=0.20,
        )
