from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path, PureWindowsPath
import secrets
from typing import Any

import pandas as pd


SPLIT_MANIFEST_SCHEMA_VERSION = 2
HMAC_KEY_BYTES = 32
INTERNAL_TEST_POLICY = (
    "evaluation-only; never use for fitting, model selection, calibration or threshold selection"
)
FORBIDDEN_MANIFEST_COLUMNS = {
    "subject_id",
    "study_id",
    "dicom_filepath",
    "relative_path",
    "patient_id",
}
MANIFEST_COLUMNS = [
    "patient_hash",
    "study_hash",
    "split",
    "sealed_internal_test",
    "instance_count",
    "source_instances_sha256",
]
RECORD_LIST_REQUIRED_COLUMNS = {"subject_id", "study_id", "dicom_filepath"}
SOURCE_NORMALIZATION_POLICY = "drop_only_exact_full_metadata_rows_v1"


def canonical_relative_dicom_path(value: object) -> str:
    """Return a Windows-filesystem identity or reject an escaping path."""
    text = str(value).strip().replace("/", "\\")
    candidate = PureWindowsPath(text)
    if not text or candidate.drive or candidate.root or ".." in candidate.parts:
        raise ValueError(f"DICOM path must be relative and remain under dicom_root: {value!r}")
    parts = [part for part in candidate.parts if part not in {"", "."}]
    if not parts or any(
        ":" in part
        or part.endswith((" ", "."))
        or any(ord(character) < 32 for character in part)
        for part in parts
    ):
        raise ValueError(f"DICOM path is not a safe canonical Windows path: {value!r}")
    # The configured corpus is addressed through a Windows drive, whose normal
    # identity semantics are separator- and case-insensitive.
    return "/".join(parts).casefold()


def patient_split(subject_id: object, seed: int, train: float = 0.70, validation: float = 0.15) -> str:
    if not (0.0 < train < 1.0 and 0.0 < validation < 1.0 and train + validation < 1.0):
        raise ValueError("Split fractions must be positive and leave a non-empty internal-test fraction")
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "internal_test"


def assign_patient_splits(frame: pd.DataFrame, seed: int, group_key: str = "subject_id") -> pd.DataFrame:
    if group_key not in frame:
        raise KeyError(f"Missing patient grouping key: {group_key}")
    result = frame.copy()
    result["split"] = result[group_key].map(lambda value: patient_split(value, seed))
    return result


def audit_no_patient_leakage(frame: pd.DataFrame, patient_key: str = "subject_id") -> dict:
    required = {patient_key, "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing split columns: {sorted(missing)}")
    counts = frame.groupby(patient_key, dropna=False)["split"].nunique()
    leaking = counts[counts > 1]
    report = {
        "schema_version": 1,
        "passed": leaking.empty,
        "rows": int(len(frame)),
        "patients": int(frame[patient_key].nunique(dropna=False)),
        "leaking_patients": int(len(leaking)),
        "split_patients": frame.groupby("split")[patient_key].nunique().astype(int).to_dict(),
    }
    if not report["passed"]:
        raise AssertionError(f"Patient leakage detected for {len(leaking)} patients")
    return report


def audit_existing_index(path: str | Path) -> dict:
    """Backward-compatible audit for the pre-v4 feature index."""
    frame = pd.read_csv(path, usecols=["patient_hash", "split"])
    return audit_no_patient_leakage(frame, patient_key="patient_hash")


def split_artifact_paths(private_root: str | Path) -> dict[str, Path]:
    root = Path(private_root)
    return {
        "secret": root / "split_manifest_hmac.key",
        "manifest": root / "patient_study_split_manifest_private.parquet",
        "metadata": root / "patient_study_split_manifest_private.meta.json",
        "internal_test_seal": root / "internal_test_split_seal_private.json",
    }


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hmac_identifier(secret: bytes, domain: str, value: str) -> str:
    message = b"HEM-FM-v4\x00" + domain.encode("ascii") + b"\x00" + value.encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _create_or_read_secret(path: Path) -> tuple[bytes, bool]:
    if path.exists():
        secret = path.read_bytes()
        if len(secret) != HMAC_KEY_BYTES:
            raise RuntimeError(
                f"Refusing split-manifest operation: persisted HMAC secret must be {HMAC_KEY_BYTES} bytes"
            )
        return secret, False
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(HMAC_KEY_BYTES)
    # Windows file descriptors default to text mode unless O_BINARY is set;
    # without it, a random 0x0A byte in the secret is expanded to CRLF and the
    # persisted key no longer has the required length.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(secret)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:  # pragma: no cover - defensive OS failure guard
                raise OSError("Failed to persist the complete HMAC secret")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows ACLs, rather than POSIX mode bits, are authoritative. The file
        # remains under the configured local private root.
        pass
    return secret, True


def read_normalized_record_list(
    record_list: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a record list while preserving all scientifically distinct rows.

    The official MIMIC record list contains a small number of field-for-field
    metadata rows. Re-reading the same path twice adds no independent instance
    and can create false SOP-duplicate evidence, so only rows identical across
    *every* source column are collapsed. Path aliases, changed metadata, and
    conflicting patient/study ownership remain fatal.
    """
    record_path = Path(record_list)
    frame = pd.read_csv(
        record_path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    if missing := RECORD_LIST_REQUIRED_COLUMNS.difference(frame.columns):
        raise KeyError(f"Record list is missing: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Record list is empty")

    source_rows = int(len(frame))
    exact_duplicate = frame.duplicated(keep="first")
    exact_duplicate_rows_removed = int(exact_duplicate.sum())
    frame = frame.loc[~exact_duplicate].copy().reset_index(drop=True)
    raw_path_spellings = frame["dicom_filepath"].copy()

    for column in sorted(RECORD_LIST_REQUIRED_COLUMNS):
        frame[column] = frame[column].str.strip()
        if (frame[column] == "").any():
            raise ValueError(f"Record list contains empty values in {column}")
    canonical_paths = frame["dicom_filepath"].map(canonical_relative_dicom_path)
    if canonical_paths.duplicated().any():
        duplicate_rows = frame.loc[canonical_paths.duplicated(keep=False)].copy()
        duplicate_rows["_canonical_path"] = canonical_paths.loc[duplicate_rows.index]
        duplicate_rows["_raw_path_spelling"] = raw_path_spellings.loc[duplicate_rows.index]
        distinct_owners = (
            duplicate_rows[
                ["_canonical_path", "subject_id", "study_id"]
            ]
            .drop_duplicates()
            .groupby("_canonical_path", sort=False)
            .size()
        )
        if (distinct_owners > 1).any():
            raise ValueError(
                "Record list assigns a canonical DICOM path to conflicting patient/study owners"
            )
        path_spellings = duplicate_rows.groupby("_canonical_path", sort=False)[
            "_raw_path_spelling"
        ].nunique()
        if (path_spellings > 1).any():
            raise ValueError(
                "Record list contains DICOM path alias spellings under Windows path semantics"
            )
        raise ValueError(
            "Record list repeats a canonical DICOM path with conflicting metadata"
        )
    owners = frame.groupby("study_id", sort=False)["subject_id"].nunique()
    if (owners > 1).any():
        raise ValueError("At least one study_id is associated with multiple patients")
    normalization = {
        "policy": SOURCE_NORMALIZATION_POLICY,
        "source_rows": source_rows,
        "normalized_rows": int(len(frame)),
        "exact_duplicate_rows_removed": exact_duplicate_rows_removed,
        "canonical_paths_unique": True,
        "path_aliases_accepted": False,
        "conflicting_metadata_accepted": False,
        "conflicting_owners_accepted": False,
    }
    return frame, normalization


def _read_source_frame(record_list: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, normalization = read_normalized_record_list(record_list)
    return frame[["subject_id", "study_id", "dicom_filepath"]].copy(), normalization


def _source_instances_digest(paths: pd.Series) -> str:
    canonical = "\n".join(sorted(canonical_relative_dicom_path(path) for path in paths))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _expected_manifest(
    source: pd.DataFrame,
    secret: bytes,
    seed: int,
    train: float,
    validation: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = source.groupby(["subject_id", "study_id"], sort=True, dropna=False)
    for (subject_id, study_id), study in grouped:
        patient_hash = _hmac_identifier(secret, "patient", str(subject_id))
        study_hash = _hmac_identifier(secret, "study", f"{subject_id}\x00{study_id}")
        # Preserve the protocol's established seeded patient assignment so
        # existing downstream cohort builders remain exactly aligned. Only the
        # persisted identifier is replaced by its domain-separated HMAC.
        split = patient_split(subject_id, seed, train=train, validation=validation)
        rows.append(
            {
                "patient_hash": patient_hash,
                "study_hash": study_hash,
                "split": split,
                "sealed_internal_test": split == "internal_test",
                "instance_count": int(len(study)),
                "source_instances_sha256": _source_instances_digest(study["dicom_filepath"]),
            }
        )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(
        ["patient_hash", "study_hash"], kind="stable", ignore_index=True
    )
    if manifest["study_hash"].duplicated().any():
        raise RuntimeError("Hashed study collision detected")
    return manifest


def _manifest_audit(manifest: pd.DataFrame) -> dict[str, Any]:
    missing = set(MANIFEST_COLUMNS).difference(manifest.columns)
    forbidden = FORBIDDEN_MANIFEST_COLUMNS.intersection(manifest.columns)
    if missing or forbidden:
        raise RuntimeError(
            f"Split manifest schema violation; missing={sorted(missing)}, forbidden={sorted(forbidden)}"
        )
    if set(manifest.columns) != set(MANIFEST_COLUMNS):
        raise RuntimeError("Split manifest contains unexpected columns")
    if manifest[list(MANIFEST_COLUMNS)].isna().any().any():
        raise RuntimeError("Split manifest contains null values")
    allowed = {"train", "validation", "internal_test"}
    if not set(manifest["split"]).issubset(allowed):
        raise RuntimeError("Split manifest contains an unknown split")
    sealed = manifest["sealed_internal_test"].astype(bool)
    if not (sealed == manifest["split"].eq("internal_test")).all():
        raise RuntimeError("Internal-test seal flags do not exactly match internal-test membership")
    if (pd.to_numeric(manifest["instance_count"], errors="coerce") < 1).any():
        raise RuntimeError("Split manifest contains a study with no source instances")
    if manifest["study_hash"].duplicated().any():
        raise RuntimeError("Split manifest contains duplicate study hashes")
    leakage = audit_no_patient_leakage(manifest, patient_key="patient_hash")
    test = manifest[manifest["split"] == "internal_test"]
    membership = "\n".join(sorted(test["study_hash"].astype(str)))
    split_studies = manifest["split"].value_counts().astype(int).to_dict()
    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "passed": bool(leakage["passed"] and set(split_studies) == allowed),
        "raw_identifiers_present": False,
        "studies": int(len(manifest)),
        "patients": int(manifest["patient_hash"].nunique()),
        "instances": int(pd.to_numeric(manifest["instance_count"]).sum()),
        "split_studies": split_studies,
        "split_patients": manifest.groupby("split")["patient_hash"].nunique().astype(int).to_dict(),
        "leaking_patients": leakage["leaking_patients"],
        "sealed_internal_test_studies": int(len(test)),
        "internal_test_membership_sha256": hashlib.sha256(membership.encode("ascii")).hexdigest(),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, engine="pyarrow", compression="zstd", index=False)
    os.replace(temporary, path)


def _validate_existing_artifacts(
    *,
    paths: dict[str, Path],
    expected: pd.DataFrame,
    record_list_sha256: str,
    secret_sha256: str,
    seed: int,
    train: float,
    validation: float,
    source_normalization: dict[str, Any],
) -> dict[str, Any]:
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    checks = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "record_list_sha256": record_list_sha256,
        "hmac_secret_sha256": secret_sha256,
        "hmac_algorithm": "HMAC-SHA256",
        "seed": seed,
        "train_fraction": train,
        "validation_fraction": validation,
        "internal_test_fraction": 1.0 - train - validation,
        "immutable": True,
        "internal_test_state": "sealed",
        "raw_identifiers_persisted": False,
        "source_normalization": source_normalization,
    }
    mismatch = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in checks.items()
        if metadata.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            "Refusing to replace immutable split manifest: source/configuration mismatch: "
            + json.dumps(mismatch, sort_keys=True)
        )
    manifest_sha256 = _sha256_file(paths["manifest"])
    if manifest_sha256 != metadata.get("manifest_sha256"):
        raise RuntimeError("Immutable split manifest file hash does not match its metadata")
    actual = pd.read_parquet(paths["manifest"], engine="pyarrow")
    _manifest_audit(actual)
    try:
        pd.testing.assert_frame_equal(
            actual[MANIFEST_COLUMNS].reset_index(drop=True),
            expected[MANIFEST_COLUMNS].reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise RuntimeError("Immutable split manifest content does not match the full source list") from exc
    audit = _manifest_audit(actual)
    seal = json.loads(paths["internal_test_seal"].read_text(encoding="utf-8"))
    expected_seal = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "state": "sealed",
        "policy": INTERNAL_TEST_POLICY,
        "manifest_sha256": manifest_sha256,
        "internal_test_studies": audit["sealed_internal_test_studies"],
        "internal_test_membership_sha256": audit["internal_test_membership_sha256"],
    }
    seal_mismatch = {
        key: {"expected": value, "actual": seal.get(key)}
        for key, value in expected_seal.items()
        if seal.get(key) != value
    }
    if seal_mismatch:
        raise RuntimeError(
            "Internal-test cryptographic seal is invalid: "
            + json.dumps(seal_mismatch, sort_keys=True)
        )
    return {**audit, **metadata, "reused": True, "immutable": True}


def build_hashed_split_manifest(
    record_list: str | Path,
    private_root: str | Path,
    seed: int,
    train: float = 0.70,
    validation: float = 0.15,
) -> dict[str, Any]:
    """Create or exactly validate the private, immutable pre-decoding split.

    Raw identifiers are used only in memory. The persisted Parquet contains
    domain-separated HMAC identifiers and per-study source-membership hashes.
    Once present, no source, key, seed, threshold or membership change is
    accepted as an in-place update.
    """
    # Validate fractions before creating any persistent secret.
    patient_split("fraction-validation", seed, train=train, validation=validation)
    source_path = Path(record_list)
    source, source_normalization = _read_source_frame(source_path)
    record_list_sha256 = _sha256_file(source_path)
    paths = split_artifact_paths(private_root)
    Path(private_root).mkdir(parents=True, exist_ok=True)

    non_secret = [paths["manifest"], paths["metadata"], paths["internal_test_seal"]]
    present = [path.exists() for path in non_secret]
    if any(present) and not all(present):
        raise RuntimeError("Refusing split-manifest operation: immutable artifact set is incomplete")
    if any(present) and not paths["secret"].exists():
        raise RuntimeError("Refusing split-manifest operation: HMAC secret is missing")

    secret, secret_created = _create_or_read_secret(paths["secret"])
    secret_sha256 = hashlib.sha256(secret).hexdigest()
    expected = _expected_manifest(source, secret, seed, train, validation)
    audit = _manifest_audit(expected)
    if not audit["passed"]:
        raise RuntimeError(
            "Split manifest failed closed: an empty train/validation/internal-test partition was produced"
        )

    if all(present):
        return _validate_existing_artifacts(
            paths=paths,
            expected=expected,
            record_list_sha256=record_list_sha256,
            secret_sha256=secret_sha256,
            seed=seed,
            train=train,
            validation=validation,
            source_normalization=source_normalization,
        )

    _write_parquet_atomic(expected, paths["manifest"])
    manifest_sha256 = _sha256_file(paths["manifest"])
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "created_utc": now,
        "immutable": True,
        "record_list": str(source_path.resolve()),
        "record_list_sha256": record_list_sha256,
        "record_list_header_sha256": hashlib.sha256(
            ",".join(source.columns).encode("utf-8")
        ).hexdigest(),
        "hmac_algorithm": "HMAC-SHA256",
        "hmac_secret_sha256": secret_sha256,
        "seed": seed,
        "train_fraction": train,
        "validation_fraction": validation,
        "internal_test_fraction": 1.0 - train - validation,
        "manifest_sha256": manifest_sha256,
        "manifest_path": str(paths["manifest"]),
        "internal_test_state": "sealed",
        "raw_identifiers_persisted": False,
        "source_normalization": source_normalization,
    }
    seal = {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "created_utc": now,
        "state": "sealed",
        "policy": INTERNAL_TEST_POLICY,
        "manifest_sha256": manifest_sha256,
        "internal_test_studies": audit["sealed_internal_test_studies"],
        "internal_test_membership_sha256": audit["internal_test_membership_sha256"],
    }
    _write_json_atomic(paths["metadata"], metadata)
    _write_json_atomic(paths["internal_test_seal"], seal)
    return {
        **audit,
        **metadata,
        "reused": False,
        "secret_created": secret_created,
        "secret_path": str(paths["secret"]),
        "metadata_path": str(paths["metadata"]),
        "internal_test_seal_path": str(paths["internal_test_seal"]),
    }


def audit_hashed_split_manifest(
    record_list: str | Path,
    private_root: str | Path,
    seed: int,
    train: float = 0.70,
    validation: float = 0.15,
) -> dict[str, Any]:
    paths = split_artifact_paths(private_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Immutable split artifacts are missing: " + ", ".join(missing))
    # build_hashed_split_manifest becomes a strict read/validation operation when
    # the complete artifact set exists; it never mutates those artifacts.
    return build_hashed_split_manifest(
        record_list,
        private_root,
        seed,
        train=train,
        validation=validation,
    )


def load_hashed_split_assignments(
    record_list: str | Path,
    private_root: str | Path,
    seed: int,
    train: float = 0.70,
    validation: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve the validated private manifest back to raw in-memory keys.

    Raw identifiers never enter the immutable Parquet. They are joined to its
    HMAC identifiers only in memory so downstream cohort builders consume the
    frozen manifest membership instead of independently recomputing a split.
    """
    report = audit_hashed_split_manifest(
        record_list,
        private_root,
        seed,
        train=train,
        validation=validation,
    )
    paths = split_artifact_paths(private_root)
    secret = paths["secret"].read_bytes()
    source, _ = _read_source_frame(Path(record_list))
    studies = (
        source[["subject_id", "study_id"]]
        .drop_duplicates()
        .sort_values(["subject_id", "study_id"], kind="stable", ignore_index=True)
    )
    studies["patient_hash"] = studies["subject_id"].map(
        lambda value: _hmac_identifier(secret, "patient", str(value))
    )
    studies["study_hash"] = studies.apply(
        lambda row: _hmac_identifier(
            secret, "study", f"{row['subject_id']}\x00{row['study_id']}"
        ),
        axis=1,
    )
    manifest = pd.read_parquet(paths["manifest"], engine="pyarrow")
    resolved = studies.merge(
        manifest[
            [
                "patient_hash",
                "study_hash",
                "split",
                "sealed_internal_test",
            ]
        ],
        on=["patient_hash", "study_hash"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if len(resolved) != len(manifest) or not resolved["_merge"].eq("both").all():
        raise RuntimeError(
            "Immutable split manifest could not be resolved exactly to the full source study list"
        )
    resolved = resolved.drop(columns="_merge")
    if resolved[["split", "sealed_internal_test"]].isna().any().any():
        raise RuntimeError("Immutable split manifest resolution produced missing assignments")
    return resolved, report


def audit_development_cohort_alignment(
    cohort_path: str | Path,
    record_list: str | Path,
    private_root: str | Path,
    seed: int,
    train: float = 0.70,
    validation: float = 0.15,
) -> dict[str, Any]:
    """Fail closed unless every development row matches the frozen manifest."""
    assignments, manifest_report = load_hashed_split_assignments(
        record_list,
        private_root,
        seed,
        train=train,
        validation=validation,
    )
    cohort = pd.read_csv(
        cohort_path,
        dtype={"subject_id": str, "study_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    required = {"subject_id", "study_id", "split", "test_locked"}
    if missing := required.difference(cohort.columns):
        raise RuntimeError(f"Development cohort is missing split columns: {sorted(missing)}")
    if cohort.empty:
        raise RuntimeError("Development cohort is empty")
    for column in ("subject_id", "study_id", "split"):
        cohort[column] = cohort[column].astype(str).str.strip()
        if cohort[column].eq("").any():
            raise RuntimeError(f"Development cohort contains an empty {column}")

    def strict_boolean(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        return None

    locked = cohort["test_locked"].map(strict_boolean)
    if locked.isna().any():
        raise RuntimeError("Development cohort contains an invalid test_locked value")
    cohort["test_locked"] = locked.astype(bool)

    expected = assignments.rename(
        columns={
            "split": "manifest_split",
            "sealed_internal_test": "manifest_test_locked",
        }
    )
    checked = cohort.merge(
        expected[
            [
                "subject_id",
                "study_id",
                "manifest_split",
                "manifest_test_locked",
            ]
        ],
        on=["subject_id", "study_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not checked["_merge"].eq("both").all():
        raise RuntimeError("Development cohort contains a study absent from the immutable manifest")
    split_mismatch = checked["split"].ne(checked["manifest_split"])
    seal_mismatch = checked["test_locked"].ne(checked["manifest_test_locked"].astype(bool))
    if split_mismatch.any() or seal_mismatch.any():
        raise RuntimeError(
            "Development cohort split/seal disagrees with the immutable split manifest"
        )
    if not set(checked["split"]).issubset({"train", "validation"}):
        raise RuntimeError("Development cohort contains an internal-test row")
    if checked["test_locked"].any():
        raise RuntimeError("Development cohort contains a sealed internal-test row")
    if "split_manifest_sha256" in checked.columns:
        embedded = checked["split_manifest_sha256"].astype(str).str.strip()
        if not embedded.eq(str(manifest_report["manifest_sha256"])).all():
            raise RuntimeError("Development cohort is bound to a different split manifest")

    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "passed": True,
        "rows": int(len(checked)),
        "patients": int(checked["subject_id"].nunique()),
        "studies": int(checked[["subject_id", "study_id"]].drop_duplicates().shape[0]),
        "split_rows": checked["split"].value_counts().astype(int).to_dict(),
        "manifest_sha256": manifest_report["manifest_sha256"],
        "manifest_record_list_sha256": manifest_report["record_list_sha256"],
        "manifest_alignment_exact": True,
        "sealed_internal_test_rows": 0,
    }


def write_audit(report: dict, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
