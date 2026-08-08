from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def patient_split(subject_id: object, seed: int, train: float = 0.70, validation: float = 0.15) -> str:
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
    frame = pd.read_csv(path, usecols=["patient_hash", "split"])
    return audit_no_patient_leakage(frame, patient_key="patient_hash")


def write_audit(report: dict, destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

