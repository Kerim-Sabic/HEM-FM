from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .splits import audit_no_patient_leakage, patient_split


def _load_views(paths: list[str], inventory: pd.DataFrame, confidence_min: float) -> pd.DataFrame:
    frames = []
    columns = ["dicom_path", "subject_id", "study_id", "acquisition_datetime", "view", "confidence", "error"]
    for path in paths:
        frame = pd.read_csv(path, usecols=columns, dtype={"subject_id": str, "study_id": str}, low_memory=False)
        frames.append(frame)
    views = pd.concat(frames, ignore_index=True)
    views["confidence"] = pd.to_numeric(views["confidence"], errors="coerce")
    views = views[
        views["error"].isna()
        & views["view"].notna()
        & (views["confidence"] >= confidence_min)
    ].copy()
    views["dicom_path"] = views["dicom_path"].str.replace("\\", "/", regex=False)
    inventory = inventory.rename(columns={"relative_path": "dicom_path"}).copy()
    inventory["dicom_path"] = inventory["dicom_path"].astype(str).str.replace("\\", "/", regex=False)
    keep = [
        "dicom_path",
        "readable",
        "has_physical_spatial_calibration",
        "spectral_region_count",
        "frames",
        "manufacturer",
        "model",
    ]
    views = views.merge(inventory[keep], on="dicom_path", how="inner")
    views = views[views["readable"].fillna(False).astype(bool) & (views["frames"].fillna(0) >= 2)]
    views = views.sort_values("confidence", ascending=False).drop_duplicates("dicom_path")
    return views


def _linked_labels(config: dict[str, Any]) -> pd.DataFrame:
    wanted = {measurement for target in config["targets"].values() for measurement in target["measurements"]}
    measurements = pd.read_csv(
        config["paths"]["measurements"],
        compression="gzip",
        usecols=["subject_id", "measurement_id", "measurement", "result", "unit"],
        dtype=str,
        low_memory=False,
    )
    measurements = measurements[measurements["measurement"].isin(wanted)].copy()
    measurements["value"] = pd.to_numeric(measurements["result"], errors="coerce")
    measurements = measurements[measurements["value"].notna()]
    studies = pd.read_csv(
        config["paths"]["study_list"],
        dtype={"subject_id": str, "study_id": str, "measurement_id": str},
        low_memory=False,
    )
    linked = measurements.merge(
        studies[["subject_id", "study_id", "measurement_id"]],
        on=["subject_id", "measurement_id"],
        how="inner",
    )
    target_frames = []
    for target_name, target in config["targets"].items():
        subset = linked[linked["measurement"].isin(target["measurements"])].copy()
        priorities = {name: index for index, name in enumerate(target["measurements"])}
        subset["priority"] = subset["measurement"].map(priorities)
        subset = (
            subset.groupby(["subject_id", "study_id", "measurement", "priority", "unit"], dropna=False, as_index=False)["value"]
            .median()
            .sort_values("priority")
            .drop_duplicates(["subject_id", "study_id"], keep="first")
        )
        scale = float(target.get("scale", 1.0))
        subset["value"] = subset["value"] * scale
        subset["target"] = target_name
        subset["canonical_unit"] = target["unit"]
        target_frames.append(subset)
    return pd.concat(target_frames, ignore_index=True)


def _study_cines(views: pd.DataFrame, target: dict[str, Any]) -> pd.DataFrame:
    if target["calibration"] == "spatial":
        eligible = views[views["has_physical_spatial_calibration"].fillna(False).astype(bool)].copy()
    elif target["calibration"] == "spectral":
        eligible = views[views["spectral_region_count"].fillna(0) > 0].copy()
    else:
        raise ValueError(f"Unsupported calibration contract: {target['calibration']}")
    allowed = target.get("required_views", target.get("any_views", []))
    eligible = eligible[eligible["view"].isin(allowed)]
    eligible = eligible.sort_values("confidence", ascending=False).drop_duplicates(["subject_id", "study_id", "view"])
    rows = []
    for (subject_id, study_id), group in eligible.groupby(["subject_id", "study_id"], sort=False):
        found = set(group["view"])
        required = set(target.get("required_views", []))
        if required and not required.issubset(found):
            continue
        if not required and group.empty:
            continue
        selected = group.sort_values("confidence", ascending=False)
        rows.append(
            {
                "subject_id": str(subject_id),
                "study_id": str(study_id),
                "cines": json.dumps(
                    [
                        {
                            "view": row.view,
                            "path": row.dicom_path,
                            "confidence": float(row.confidence),
                            "manufacturer": row.manufacturer,
                            "model": row.model,
                        }
                        for row in selected.itertuples(index=False)
                    ],
                    separators=(",", ":"),
                ),
                "views": "+".join(sorted(found)),
            }
        )
    return pd.DataFrame(rows)


def _study_spectral_cines(
    views: pd.DataFrame,
    inventory: pd.DataFrame,
    records: pd.DataFrame,
    target: dict[str, Any],
) -> pd.DataFrame:
    anchors = views[views["view"].isin(target["anchor_views"])].copy()
    anchors["acquisition_datetime"] = pd.to_datetime(anchors["acquisition_datetime"], errors="coerce")
    anchors = anchors[anchors["acquisition_datetime"].notna()]
    spectral = inventory[
        inventory["readable"].fillna(False).astype(bool)
        & (inventory["spectral_region_count"].fillna(0) > 0)
    ].copy()
    spectral = spectral.merge(
        records[["subject_id", "study_id", "dicom_filepath", "acquisition_datetime"]],
        left_on=["subject_id", "study_id", "relative_path"],
        right_on=["subject_id", "study_id", "dicom_filepath"],
        how="left",
    )
    spectral["acquisition_datetime"] = pd.to_datetime(spectral["acquisition_datetime"], errors="coerce")
    spectral = spectral[spectral["acquisition_datetime"].notna()]
    max_gap = float(target.get("anchor_max_gap_seconds", 180))
    rows = []
    anchor_groups = {key: group for key, group in anchors.groupby(["subject_id", "study_id"], sort=False)}
    for (subject_id, study_id), candidates in spectral.groupby(["subject_id", "study_id"], sort=False):
        anchor_group = anchor_groups.get((str(subject_id), str(study_id)))
        if anchor_group is None:
            continue
        selected = []
        for candidate in candidates.itertuples(index=False):
            gaps = (anchor_group["acquisition_datetime"] - candidate.acquisition_datetime).abs().dt.total_seconds()
            closest_index = gaps.idxmin()
            gap = float(gaps.loc[closest_index])
            if gap > max_gap:
                continue
            anchor = anchor_group.loc[closest_index]
            selected.append(
                {
                    "view": "Spectral_Doppler",
                    "path": candidate.relative_path,
                    "confidence": float(anchor.confidence),
                    "anchor_view": anchor["view"],
                    "anchor_gap_seconds": gap,
                    "manufacturer": candidate.manufacturer,
                    "model": candidate.model,
                    "score": float(anchor.confidence) * (1.0 - gap / max_gap),
                }
            )
        if not selected:
            continue
        selected = sorted(selected, key=lambda item: item["score"], reverse=True)[:2]
        for item in selected:
            item.pop("score")
        rows.append(
            {
                "subject_id": str(subject_id),
                "study_id": str(study_id),
                "cines": json.dumps(selected, separators=(",", ":")),
                "views": "+".join(sorted({item["anchor_view"] for item in selected})),
            }
        )
    return pd.DataFrame(rows)


def build_endpoint_cohorts(config: dict[str, Any], confidence_min: float = 0.75) -> dict[str, Any]:
    inventory_path = Path(config["paths"]["private_root"]) / "dicom_inventory.csv"
    inventory = pd.read_csv(inventory_path, low_memory=False)
    inventory["subject_id"] = inventory["subject_id"].astype(str)
    inventory["study_id"] = inventory["study_id"].astype(str)
    views = _load_views(config["paths"]["view_prediction_sources"], inventory, confidence_min)
    records = pd.read_csv(
        config["paths"]["record_list"],
        dtype={"subject_id": str, "study_id": str},
        low_memory=False,
    )
    records["dicom_filepath"] = records["dicom_filepath"].str.replace("\\", "/", regex=False)
    labels = _linked_labels(config)
    private_root = Path(config["paths"]["private_root"])
    report_root = Path(config["paths"]["report_root"])
    private_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    target_summary: dict[str, Any] = {}
    seed = int(config["splits"]["seed"])
    for target_name, target in config["targets"].items():
        cines = (
            _study_spectral_cines(views, inventory, records, target)
            if target["calibration"] == "spectral"
            else _study_cines(views, target)
        )
        target_labels = labels[labels["target"] == target_name]
        if cines.empty:
            cohort = target_labels.head(0).copy()
            cohort["cines"] = pd.Series(dtype=str)
            cohort["views"] = pd.Series(dtype=str)
        else:
            cohort = target_labels.merge(cines, on=["subject_id", "study_id"], how="inner")
        cohort["split"] = cohort["subject_id"].map(lambda value: patient_split(value, seed))
        cohort["track"] = "R"
        cohort["test_locked"] = cohort["split"].eq("internal_test")
        cohort["source"] = "MIMIC-IV-ECHO"
        all_rows.append(cohort)
        target_summary[target_name] = {
            "rows": int(len(cohort)),
            "patients": int(cohort["subject_id"].nunique()),
            "studies": int(cohort["study_id"].nunique()),
            "split_rows": cohort["split"].value_counts().astype(int).to_dict(),
            "required_views": target.get("required_views"),
            "anchor_views": target.get("anchor_views"),
            "calibration": target["calibration"],
            "unit": target["unit"],
        }
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(private_root / "endpoint_cohorts_private.csv", index=False)
    development = combined[combined["split"].isin(["train", "validation"])].copy()
    development.to_csv(private_root / "endpoint_cohorts_development_private.csv", index=False)
    locked = combined[combined["split"] == "internal_test"].copy()
    locked.to_csv(private_root / "endpoint_cohorts_locked_test_private.csv", index=False)
    split_audit = audit_no_patient_leakage(combined[["subject_id", "split"]].drop_duplicates())
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(split_audit["passed"] and len(combined) > 0),
        "confidence_min": confidence_min,
        "view_prediction_rows_eligible": int(len(views)),
        "targets": target_summary,
        "all_rows": int(len(combined)),
        "all_patients": int(combined["subject_id"].nunique()),
        "split_audit": split_audit,
        "test_labels_exported_to_development_manifest": False,
        "track": "R",
    }
    (report_root / "endpoint_cohort_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

