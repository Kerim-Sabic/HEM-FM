from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .frozen_specialists import _atomic_json
from .extended_dataset_audit import audit_extended_research_datasets


TED_CITATION = (
    "N. Painchaud et al., Echocardiography segmentation with enforced temporal "
    "consistency, IEEE TMI 41(10), 2022, doi:10.1109/TMI.2022.3173669."
)
UNITY_SOURCE = "https://data.unityimaging.net/"
TED_SOURCE = "https://humanheart-project.creatis.insa-lyon.fr/ted.html"


def _lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _read_status(run_root: Path, suffix: str) -> dict[str, Any]:
    path = run_root / "week_training" / f"dataset_download_status{suffix}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_metaimage_header(path: Path) -> dict[str, Any]:
    raw: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        raw[key.strip()] = value.strip()
    dimensions = tuple(int(value) for value in raw.get("DimSize", "").split())
    spacing = tuple(float(value) for value in raw.get("ElementSpacing", "").split())
    return {
        "dimensions": dimensions,
        "spacing": spacing,
        "element_type": raw.get("ElementType"),
        "data_file": raw.get("ElementDataFile"),
    }


def _read_ted_info(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _camus_patient_id(ted_patient_id: str) -> str:
    number = int(ted_patient_id.removeprefix("patient"))
    return f"patient{number:04d}"


def audit_ted(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["ted_root"])
    database = root / "database"
    camus_root = Path(config["paths"]["camus_root"])
    camus_splits = {
        name: set(_lines(camus_root / "database_split" / f"subgroup_{name}.txt"))
        for name in ("training", "validation", "testing")
    }
    patient_dirs = sorted(path for path in database.glob("patient*") if path.is_dir())
    mapped: dict[str, list[str]] = {name: [] for name in camus_splits}
    unmapped: list[str] = []
    metadata_counts: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    sexes: Counter[str] = Counter()
    frame_counts: list[int] = []
    spacings: list[tuple[float, ...]] = []
    raw_size_mismatches: list[str] = []
    missing_files: list[str] = []
    development_label_values: set[int] = set()

    for patient_dir in patient_dirs:
        patient = patient_dir.name
        camus_id = _camus_patient_id(patient)
        split = next((name for name, ids in camus_splits.items() if camus_id in ids), None)
        if split is None:
            unmapped.append(patient)
        else:
            mapped[split].append(patient)

        stem = f"{patient}_4CH_sequence"
        expected = [
            patient_dir / f"{patient}_4CH_info.cfg",
            patient_dir / f"{stem}.mhd",
            patient_dir / f"{stem}.raw",
            patient_dir / f"{stem}_gt.mhd",
            patient_dir / f"{stem}_gt.raw",
        ]
        missing_files.extend(str(path) for path in expected if not path.exists())
        if any(not path.exists() for path in expected):
            continue

        info = _read_ted_info(expected[0])
        metadata_counts.update(info.keys())
        qualities[info.get("ImageQuality", "missing")] += 1
        sexes[info.get("Sex", "missing")] += 1
        header = parse_metaimage_header(expected[1])
        gt_header = parse_metaimage_header(expected[3])
        frame_counts.append(int(header["dimensions"][-1]))
        spacings.append(tuple(header["spacing"]))
        expected_bytes = int(np.prod(header["dimensions"]))
        expected_gt_bytes = int(np.prod(gt_header["dimensions"]))
        if expected[2].stat().st_size != expected_bytes:
            raw_size_mismatches.append(str(expected[2]))
        if expected[4].stat().st_size != expected_gt_bytes:
            raw_size_mismatches.append(str(expected[4]))
        # Do not load the four CAMUS-linked test masks before the analysis freeze.
        if split == "training":
            mask = np.memmap(expected[4], dtype=np.uint8, mode="r")
            development_label_values.update(int(value) for value in np.unique(mask))

    status = _read_status(Path(config["paths"]["run_root"]), "_ted")
    checks = {
        "ninety_eight_patients": len(patient_dirs) == 98,
        "five_files_per_patient": not missing_files,
        "metaimage_raw_sizes_match": not raw_size_mismatches,
        "metadata_complete": all(metadata_counts[key] == 98 for key in ("ED", "ES", "NbFrame", "ImageQuality", "EF")),
        "labels_are_background_lv_myo": development_label_values <= {0, 1, 2} and {1, 2} <= development_label_values,
        "all_patients_mapped_to_camus_identity": not unmapped,
        "camus_test_patients_sealed": len(mapped["testing"]) == 4,
        "archive_hash_recorded": len(status.get("sha256", "")) == 64,
        "licence_and_citation_present": (database / "LICENSE_TERMS.md").exists() and (database / "MANDATORY_CITATION.md").exists(),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "dataset": "TED temporal echocardiography dataset",
        "official_source": TED_SOURCE,
        "local_root": str(root),
        "download_sha256": status.get("sha256"),
        "archive_bytes": status.get("downloaded_bytes"),
        "patients": len(patient_dirs),
        "camus_identity_mapping": {name: len(values) for name, values in mapped.items()},
        "development_patients": len(mapped["training"]),
        "sealed_test_patients": len(mapped["testing"]),
        "frames": {"minimum": min(frame_counts), "maximum": max(frame_counts)},
        "spacings": {"unique": len(set(spacings))},
        "development_label_values": sorted(development_label_values),
        "image_quality": dict(sorted(qualities.items())),
        "sex": dict(sorted(sexes.items())),
        "missing_files": missing_files,
        "raw_size_mismatches": raw_size_mismatches,
        "unmapped_patients": unmapped,
        "track": "R",
        "licence": "CC BY-NC-SA 4.0 plus dataset-specific non-commercial research terms",
        "redistribution": False,
        "clinical_use": False,
        "citation": TED_CITATION,
        "test_policy": "The four TED patients linked to the official CAMUS test split are file-size audited only; their pixel and mask contents remain sealed.",
        "locked_test_accessed": False,
    }
    destination = Path(config["paths"]["evidence_root"]) / "G4" / "ted_acquisition.json"
    _atomic_json(destination, report)
    return report


def _unity_split_names(labels_root: Path) -> dict[str, set[str]]:
    return {
        name: set(_lines(labels_root / f"labels-{name}.txt"))
        for name in ("train", "tune", "ival")
    }


def _unity_prefix(filename: str) -> str:
    return re.sub(r"-\d{4}\.png$", "", filename)


def _annotation_counts(path: Path, usable_names: set[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    state_counts: Counter[str] = Counter()
    key_counts: Counter[str] = Counter()
    for filename, row in payload.items():
        if filename not in usable_names:
            continue
        for key, label in row.get("labels", {}).items():
            state = str(label.get("type", "missing"))
            state_counts[state] += 1
            if state not in {"off", "blurred", "missing"} and str(label.get("x", "")).strip() and str(label.get("y", "")).strip():
                key_counts[key] += 1
    return {
        "label_states": dict(sorted(state_counts.items())),
        "supervised_key_counts": dict(sorted(key_counts.items())),
        "supervised_annotations": sum(key_counts.values()),
    }


def audit_unity(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["unity_root"])
    extracted = root / "extracted"
    labels_root = extracted / "labels"
    image_root = extracted / "png-cache"
    image_names = {path.name for path in image_root.rglob("*.png")}
    splits = _unity_split_names(labels_root)
    overlap = {
        "train_tune": len(splits["train"] & splits["tune"]),
        "train_ival": len(splits["train"] & splits["ival"]),
        "tune_ival": len(splits["tune"] & splits["ival"]),
    }
    all_label_names = set().union(*splits.values())
    missing_images = sorted(all_label_names - image_names)
    unlabelled_images = sorted(image_names - all_label_names)
    usable = {
        name: values & image_names
        for name, values in splits.items()
    }
    development = usable["train"] | usable["tune"]
    annotation = {
        "train": _annotation_counts(labels_root / "labels-train.json", usable["train"]),
        "tune": _annotation_counts(labels_root / "labels-tune.json", usable["tune"]),
    }

    views: dict[str, str] = {}
    with (root / "view_index_20220705.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["file"] in development:
                views.setdefault(row["file"], row["view"])
    development_prefixes = {_unity_prefix(name) for name in development}
    physical_prefixes: set[str] = set()
    with (root / "01_database_physical.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["FileHash"] in development_prefixes:
                physical_prefixes.add(row["FileHash"])

    statuses = {
        name: _read_status(Path(config["paths"]["run_root"]), suffix)
        for name, suffix in {
            "images": "_unity_images",
            "labels": "_unity_labels",
            "views": "_unity_views",
            "physical": "_unity_physical",
        }.items()
    }
    checks = {
        "official_image_count": len(image_names) == 7522,
        "official_split_counts": [len(splits[name]) for name in ("train", "tune", "ival")] == [5254, 1179, 1090],
        "split_membership_disjoint": all(value == 0 for value in overlap.values()),
        "single_documented_missing_training_image": len(missing_images) == 1 and missing_images[0] in splits["train"],
        "all_images_labelled": not unlabelled_images,
        "all_development_views_available": len(views) == len(development),
        "physical_metadata_high_coverage": len(physical_prefixes) / max(1, len(development_prefixes)) >= 0.98,
        "development_annotations_loaded": annotation["train"]["supervised_annotations"] > 0 and annotation["tune"]["supervised_annotations"] > 0,
        "locked_ival_annotations_not_loaded": True,
        "archive_hashes_recorded": all(len(status.get("sha256", "")) == 64 for status in statuses.values()),
        "licence_present": (labels_root / "LICENSE.txt").exists(),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "dataset": "Unity Imaging collaborative echo annotations",
        "official_source": UNITY_SOURCE,
        "local_root": str(root),
        "download_sha256": {name: status.get("sha256") for name, status in statuses.items()},
        "download_bytes": {name: status.get("downloaded_bytes") for name, status in statuses.items()},
        "images": len(image_names),
        "declared_split_counts": {name: len(values) for name, values in splits.items()},
        "usable_split_counts": {name: len(values) for name, values in usable.items()},
        "split_overlap": overlap,
        "missing_images": missing_images,
        "unlabelled_images": unlabelled_images,
        "development_views": dict(sorted(Counter(views.values()).items())),
        "development_physical_metadata": {
            "studies_with_spacing": len(physical_prefixes),
            "studies_total": len(development_prefixes),
            "coverage": len(physical_prefixes) / max(1, len(development_prefixes)),
        },
        "development_annotations": annotation,
        "track": "R",
        "licence": "CC BY-NC-SA 4.0 for images, labels, metadata, and released weights; MIT code",
        "redistribution": False,
        "clinical_use": False,
        "test_policy": "Only the ival filename list is inventoried; ival annotation JSON remains sealed until the prespecified analysis is frozen.",
        "locked_test_accessed": False,
    }
    destination = Path(config["paths"]["evidence_root"]) / "G4" / "unity_acquisition.json"
    _atomic_json(destination, report)
    return report


def audit_research_datasets(config: dict[str, Any]) -> dict[str, Any]:
    ted = audit_ted(config)
    unity = audit_unity(config)
    extended = audit_extended_research_datasets(config)
    return {"ted": ted, "unity": unity, "extended": extended, "passed": ted["passed"] and unity["passed"] and extended["passed"]}

