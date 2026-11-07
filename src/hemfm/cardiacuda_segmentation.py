from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np


SOURCE_URL = "https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset"
AUTHOR_CODE_URL = "https://github.com/xmed-lab/GraphEcho/blob/main/datasets/cardiac_uda.py"
DATASET_LICENSE = "Apache-2.0 (official Kaggle dataset metadata)"
SEALED_DIRECTORY = "label_all_frame"
ANNOTATED_MASK_SUM_THRESHOLD = 100.0
BASELINE_VIEW = "A4C"
BASELINE_SCOPE = "standalone_a4c_frame_segmentation_baseline"
IGNORE_INDEX = -100
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOUNDARY_TOLERANCE_PIXELS = 2.0
HD95_METRIC_NAME = "hd95_common_grid_pixels"
BOUNDARY_METRIC_NAME = f"boundary_dice_{BOUNDARY_TOLERANCE_PIXELS:g}px_common_grid"

# The public CardiacUDC extraction has two known unusable labelled A4C volumes:
# one has no author-qualified frame and one gzip stream is truncated. They may
# be quarantined only while exact artifact hashes and exact errors remain fixed.
APPROVED_DEVELOPMENT_QUARANTINE: dict[str, dict[str, str]] = {
    "Site_R_126:patient-60:A4C": {
        "error": "no frames with raw mask sum > 100",
        "image_file_sha256": "06b1c72fbb877af416810e4e7903432bcf50fbed81e66cbdf2e60a2b6f21e783",
        "label_file_sha256": "b618a91911e44306eded00719bf8521454c68b460c20c81911ea1c5837fc8675",
    },
    "Site_R_126:patient-73:A4C": {
        "error": "EOFError: Compressed file ended before the end-of-stream marker was reached",
        "image_file_sha256": "4008dc1fe08bdad3481b3bbf74ece1a431cacb8a2dbac2ff570673adfc1c5ec1",
        "label_file_sha256": "a21e4dc87fbe17e44ec42c047caf62500f85f12f2fdf486a22504c684d375666",
    }
}

# The verified local extraction contains one A4C acquisition twice under two
# patient identifiers with different annotation volumes. Treating both rows as
# independent validation patients would inflate the denominator. The second
# annotation is excluded only while the exact compressed artifacts and decoded
# image/label identities below all match. The retained annotation is fixed; no
# validation metric or model output is consulted.
APPROVED_DUPLICATE_ANNOTATION_QUARANTINE: dict[str, dict[str, str]] = {
    "Site_R_126:patient-90:A4C": {
        "retained_key": "Site_R_126:patient-89:A4C",
        "image_file_sha256": "93cd38b723934a88b512bfc621a41db7cb33d51441fcd21f4a2ed7a2286b23b4",
        "label_file_sha256": "1540e8c6cb83b8f48e5edcbd790e36f9288fc00f90873a6e9bd482fffc9fb2d8",
        "decoded_image_sha256": "827f88087c4b70483bb3bb4cf711bde66f45755c1e50b6bca0e23883141b026b",
        "decoded_label_sha256": "5e6e3e3bd7131c81c312220ffce8ab95311b9d94c163efd05b5085698869f02d",
        "retained_image_file_sha256": "f53f101160ed25056de7251ee16d572385187aa07cfcbd4e769991ea05305d2d",
        "retained_label_file_sha256": "f59240fca41cbfc85e731af9ebae5f32e305f1c1bf50d6b6650dc1705e2c49d4",
        "retained_decoded_label_sha256": "d9e83eee92cd84754850eb191ce1a979e778145f8f7c2c64679cf56eae0739c1",
    }
}

APPROVED_UNLABELED_INVENTORIES: dict[str, dict[str, Any]] = {
    "ae53989afcd2cf4ac978121f79814641161634497467ab12d1f8d18dd46bd518": {
        "entry_count": 70,
        "cohort": "Site_R_73",
        "basis": "verified local CardiacUDC research extraction; filename-and-size metadata only",
    }
}

APPROVED_DEVELOPMENT_FILE_INVENTORIES: dict[str, dict[str, Any]] = {
    "b868d9c2594c4a6d02b72368953b5304a60a785cede3e3c09101c4e183ffa386": {
        "entry_count": 638,
        "dataset": "CardiacUDC verified local research extraction",
        "basis": "all non-sealed immediate files bound by relative filename and size",
    }
}

TRANSFORM_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "scope": BASELINE_SCOPE,
    "normalization": {
        "operation": "finite_percentile_minmax",
        "percentiles": [1.0, 99.0],
        "fallback": "finite_minmax_then_zeros_if_constant",
        "clip_range": [0.0, 1.0],
        "storage_quantization": "round(value*255)_uint8",
    },
    "preprocessing": {
        "operation": "aspect_preserving_resize_then_symmetric_pad",
        "image_interpolation": "bilinear_align_corners_false",
        "mask_interpolation": "nearest",
        "image_pad_value": 0.0,
        "mask_pad_value": IGNORE_INDEX,
        "invertible_geometry_record_required": True,
    },
    "augmentations": [
        {
            "name": "horizontal_flip",
            "enabled": False,
            "reason": "unproven_for_this_four_chamber_challenger; requires a preregistered cross-domain ablation",
            "parity_flag": False,
        },
        {
            "name": "intensity_gamma_gain_gaussian_noise",
            "enabled_for_training": True,
            "gamma": [0.9, 1.1],
            "gain": [0.9, 1.1],
            "noise_standard_deviation": [0.0, 0.015],
            "geometric": False,
        },
    ],
    "unsupported_capabilities": [
        "foundation_backbone",
        "temporal_modeling",
        "physical_measurement",
        "clinical_use",
    ],
}

LOSS_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "name": "v4_dense_mask_challenger_without_hd95_surrogate",
    "terms": {
        "dice": 1.0,
        "focal": 0.5,
        "boundary": 0.25,
    },
    "focal_gamma": 2.0,
    "uncertainty_weighting": "heteroscedastic_focal_challenger",
}

GLOBAL_CLASSES = ("background", "LV", "RV", "LA", "RA", "PA")
GLOBAL_CLASS_TO_INDEX = {name: index for index, name in enumerate(GLOBAL_CLASSES)}


@dataclass(frozen=True)
class ViewSchema:
    name: str
    filename_code: int | None
    structures: tuple[str, ...]
    raw_to_global: tuple[tuple[int, int], ...]
    aliases: tuple[str, ...] = ()

    @property
    def active_global_indices(self) -> tuple[int, ...]:
        return (0, *(GLOBAL_CLASS_TO_INDEX[name] for name in self.structures))

    @property
    def raw_mapping(self) -> dict[int, int]:
        return dict(self.raw_to_global)


# Numeric mappings follow the CardiacUDA authors' public loader. CAMUS A2C is
# intentionally not included here: its 0/1/2/3 ontology contains myocardium and
# needs a source-specific adapter rather than a superficially compatible map.
VIEW_SCHEMAS: dict[str, ViewSchema] = {
    "A4C": ViewSchema(
        name="A4C",
        filename_code=4,
        structures=("LV", "RV", "LA", "RA"),
        raw_to_global=((1, 1), (2, 3), (3, 4), (4, 2)),
        aliases=("4CH", "AP4"),
    ),
    "PLAX": ViewSchema(
        name="PLAX",
        filename_code=1,
        structures=("LV", "RV"),
        raw_to_global=((1, 1), (2, 2)),
        aliases=("LVLA", "PSLAX"),
    ),
    "PA": ViewSchema(
        name="PA",
        filename_code=2,
        structures=("PA",),
        raw_to_global=((1, 5),),
        aliases=("PALA", "PA-LAX"),
    ),
    "PSAX": ViewSchema(
        name="PSAX",
        filename_code=3,
        structures=("LV", "RV"),
        raw_to_global=((1, 1), (2, 2)),
        aliases=("LVSA", "SAX"),
    ),
}
VIEW_NAMES = tuple(VIEW_SCHEMAS)
VIEW_TO_INDEX = {name: index for index, name in enumerate(VIEW_NAMES)}
CODE_TO_VIEW = {
    schema.filename_code: name
    for name, schema in VIEW_SCHEMAS.items()
    if schema.filename_code is not None
}
_IMAGE_PATTERN = re.compile(r"^(?P<patient>.+)-(?P<view>[1-4])_image\.nii\.gz$")


class SealedTestError(RuntimeError):
    pass


def _path_is_sealed(path: Path) -> bool:
    marker = SEALED_DIRECTORY.casefold()
    candidate = Path(path)
    paths = (candidate, candidate.resolve(strict=False))
    return any(marker in {part.casefold() for part in value.parts} for value in paths)


def _assert_not_sealed_path(path: Path) -> None:
    if _path_is_sealed(path):
        raise SealedTestError(f"Locked CardiacUDC content cannot be decoded: {path}")


@dataclass(frozen=True)
class VolumeRecord:
    image_path: Path
    label_path: Path | None
    cohort: str
    domain: str
    patient_id: str
    view: str
    sealed: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["image_path"] = str(self.image_path)
        result["label_path"] = str(self.label_path) if self.label_path is not None else None
        return result


@dataclass(frozen=True)
class FrameRecord:
    volume: VolumeRecord
    frame_index: int

    @property
    def key(self) -> str:
        return f"{self.volume.patient_id}:{self.volume.view}:{self.frame_index}"


@dataclass(frozen=True)
class SpatialTransform:
    """Recorded pixel-centre transform for reversible letterbox preprocessing."""

    source_height: int
    source_width: int
    output_height: int
    output_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    scale_y: float
    scale_x: float
    forward_matrix: tuple[tuple[float, float, float], ...]
    inverse_matrix: tuple[tuple[float, float, float], ...]
    coordinate_convention: str = "zero_based_pixel_centres"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpatialTransform":
        required = {
            "source_height",
            "source_width",
            "output_height",
            "output_width",
            "resized_height",
            "resized_width",
            "pad_top",
            "pad_bottom",
            "pad_left",
            "pad_right",
            "scale_y",
            "scale_x",
            "forward_matrix",
            "inverse_matrix",
            "coordinate_convention",
        }
        if set(value) != required:
            raise ValueError(
                "Spatial transform has unknown or missing fields: "
                f"expected={sorted(required)} actual={sorted(value)}"
            )
        transform = cls(
            source_height=int(value["source_height"]),
            source_width=int(value["source_width"]),
            output_height=int(value["output_height"]),
            output_width=int(value["output_width"]),
            resized_height=int(value["resized_height"]),
            resized_width=int(value["resized_width"]),
            pad_top=int(value["pad_top"]),
            pad_bottom=int(value["pad_bottom"]),
            pad_left=int(value["pad_left"]),
            pad_right=int(value["pad_right"]),
            scale_y=float(value["scale_y"]),
            scale_x=float(value["scale_x"]),
            forward_matrix=tuple(tuple(float(item) for item in row) for row in value["forward_matrix"]),
            inverse_matrix=tuple(tuple(float(item) for item in row) for row in value["inverse_matrix"]),
            coordinate_convention=str(value["coordinate_convention"]),
        )
        transform.validate()
        return transform

    def validate(self) -> None:
        if min(
            self.source_height,
            self.source_width,
            self.output_height,
            self.output_width,
            self.resized_height,
            self.resized_width,
        ) <= 0:
            raise ValueError("Spatial transform dimensions must be positive")
        if self.coordinate_convention != "zero_based_pixel_centres":
            raise ValueError("Unsupported spatial-coordinate convention")
        if min(self.pad_top, self.pad_bottom, self.pad_left, self.pad_right) < 0:
            raise ValueError("Spatial transform padding cannot be negative")
        if self.resized_height + self.pad_top + self.pad_bottom != self.output_height:
            raise ValueError("Vertical letterbox geometry is inconsistent")
        if self.resized_width + self.pad_left + self.pad_right != self.output_width:
            raise ValueError("Horizontal letterbox geometry is inconsistent")
        if not math.isclose(self.scale_y, self.resized_height / self.source_height, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Recorded vertical scale is inconsistent")
        if not math.isclose(self.scale_x, self.resized_width / self.source_width, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Recorded horizontal scale is inconsistent")
        forward = np.asarray(self.forward_matrix, dtype=np.float64)
        inverse = np.asarray(self.inverse_matrix, dtype=np.float64)
        if forward.shape != (3, 3) or inverse.shape != (3, 3):
            raise ValueError("Spatial transform matrices must be 3x3")
        offset_x = self.pad_left + 0.5 * self.scale_x - 0.5
        offset_y = self.pad_top + 0.5 * self.scale_y - 0.5
        expected_forward = np.asarray(
            (
                (self.scale_x, 0.0, offset_x),
                (0.0, self.scale_y, offset_y),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        if not np.allclose(forward, expected_forward, rtol=0, atol=1e-12):
            raise ValueError("Spatial forward matrix does not encode the recorded scale and padding")
        expected_inverse = np.linalg.inv(expected_forward)
        if not np.allclose(inverse, expected_inverse, rtol=0, atol=1e-12):
            raise ValueError("Spatial inverse matrix does not invert the recorded geometry")
        if not np.allclose(forward @ inverse, np.eye(3), rtol=0, atol=1e-9):
            raise ValueError("Spatial transform matrices are not inverses")


@dataclass(frozen=True)
class ManifestBundle:
    train_volumes: tuple[VolumeRecord, ...]
    validation_volumes: tuple[VolumeRecord, ...]
    train_frames: tuple[FrameRecord, ...]
    validation_frames: tuple[FrameRecord, ...]
    sealed_volumes: tuple[VolumeRecord, ...]
    unlabeled_volumes: tuple[VolumeRecord, ...]
    errors: tuple[dict[str, Any], ...]
    validation_domain: str
    fingerprint: str
    content_audit: Mapping[str, Any]
    development_file_inventory: Mapping[str, Any]
    unlabeled_inventory: Mapping[str, Any]
    sealed_inventory: Mapping[str, Any]

    def report(self) -> dict[str, Any]:
        train_patients = {item.patient_id for item in self.train_volumes}
        validation_patients = {item.patient_id for item in self.validation_volumes}
        train_domains = {item.domain for item in self.train_volumes}
        validation_domains = {item.domain for item in self.validation_volumes}
        present_views = sorted({item.view for item in (*self.train_volumes, *self.validation_volumes)})
        return {
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_url": SOURCE_URL,
            "author_label_code": AUTHOR_CODE_URL,
            "license": DATASET_LICENSE,
            "license_evidence": {
                "kind": "official Kaggle dataset metadata",
                "source": SOURCE_URL,
                "local_archive_contains_license_file": False,
            },
            "track": "R",
            "development_file_inventory": dict(self.development_file_inventory),
            "baseline_scope": BASELINE_SCOPE,
            "baseline_view": BASELINE_VIEW,
            "capability_statement": (
                "Standalone single-frame A4C chamber segmentation baseline; "
                "not a foundation model, temporal specialist, measurement system, or clinical device."
            ),
            "local_views_present": present_views,
            "view_schemas": {
                name: {
                    "structures": list(schema.structures),
                    "raw_to_global": {str(raw): GLOBAL_CLASSES[global_index] for raw, global_index in schema.raw_to_global},
                }
                for name, schema in VIEW_SCHEMAS.items()
            },
            "train": {
                "volumes": len(self.train_volumes),
                "annotated_frames": len(self.train_frames),
                "patients": len(train_patients),
                "domains": sorted(train_domains),
            },
            "validation": {
                "volumes": len(self.validation_volumes),
                "annotated_frames": len(self.validation_frames),
                "patients": len(validation_patients),
                "domains": sorted(validation_domains),
            },
            "sealed_test": {
                "directory": SEALED_DIRECTORY,
                "volumes": len(self.sealed_volumes),
                "decoded": False,
                "policy": "enumerated by filename only; excluded from audit, training, model selection, and validation",
                "inventory": dict(self.sealed_inventory),
            },
            "unlabeled_volumes": len(self.unlabeled_volumes),
            "unlabeled_inventory": dict(self.unlabeled_inventory),
            "invalid_volumes": list(self.errors),
            "checks": {
                "patient_disjoint": not bool(train_patients & validation_patients),
                "domain_disjoint": not bool(train_domains & validation_domains),
                "sealed_test_not_decoded": True,
                "missing_structures_are_ignored_not_background": True,
                "only_annotated_frames_used": True,
                "exact_development_content_disjoint": self.content_audit.get("cross_partition_duplicates") == 0,
                "all_development_quarantines_explicitly_approved": all(
                    bool(error.get("approved_quarantine")) for error in self.errors
                ),
                "complete_development_file_inventory_approved": bool(
                    self.development_file_inventory.get("approved")
                ),
            },
            "annotated_frame_criterion": {
                "rule": "sum(raw_mask_values) > threshold",
                "threshold": ANNOTATED_MASK_SUM_THRESHOLD,
                "source": AUTHOR_CODE_URL,
            },
            "validation_domain": self.validation_domain,
            "fingerprint": self.fingerprint,
            "content_deduplication": dict(self.content_audit),
            "transform_policy": TRANSFORM_POLICY,
            "loss_policy": LOSS_POLICY,
            "cardiacnet_policy": "optional source remains outside this manifest; its separate audit must prove completeness and CardiacUDC deduplication before any development use",
        }


def canonical_view(value: str) -> str:
    candidate = value.upper().strip()
    for name, schema in VIEW_SCHEMAS.items():
        if candidate == name or candidate in schema.aliases:
            return name
    raise ValueError(f"Unsupported view {value!r}; expected one of {', '.join(VIEW_NAMES)}")


def active_class_mask(view: str) -> np.ndarray:
    schema = VIEW_SCHEMAS[canonical_view(view)]
    result = np.zeros(len(GLOBAL_CLASSES), dtype=bool)
    result[list(schema.active_global_indices)] = True
    return result


def remap_mask(raw_mask: np.ndarray, view: str, *, ignore_index: int = -100) -> np.ndarray:
    schema = VIEW_SCHEMAS[canonical_view(view)]
    raw = np.asarray(raw_mask)
    output = np.full(raw.shape, ignore_index, dtype=np.int64)
    output[raw == 0] = 0
    for raw_value, global_index in schema.raw_to_global:
        output[raw == raw_value] = global_index
    output[raw == 255] = ignore_index
    unexpected = sorted(int(value) for value in np.unique(raw[output == ignore_index]) if value != 255)
    if unexpected:
        raise ValueError(f"Unexpected {schema.name} label values: {unexpected}")
    return output


def _record_from_image(path: Path, *, sealed: bool) -> VolumeRecord | None:
    match = _IMAGE_PATTERN.match(path.name)
    if match is None:
        return None
    view = CODE_TO_VIEW[int(match.group("view"))]
    cohort = path.parent.name
    if cohort.startswith("Site_"):
        domain_match = re.match(r"^Site_([A-Za-z]+)", cohort)
        domain = f"Site_{domain_match.group(1)}" if domain_match else cohort
    else:
        domain = "sealed"
    patient = match.group("patient")
    # Numeric identifiers restart across cohort folders and refer to different
    # files, so the cohort is part of the public-record pseudonym.
    patient_id = f"{cohort}:{patient}"
    label = path.with_name(path.name.replace("_image.nii.gz", "_label.nii.gz"))
    return VolumeRecord(
        image_path=path,
        label_path=label if label.exists() else None,
        cohort=cohort,
        domain=domain,
        patient_id=patient_id,
        view=view,
        sealed=sealed,
    )


def discover_volumes(root: Path) -> tuple[list[VolumeRecord], list[VolumeRecord], list[VolumeRecord]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"CardiacUDC root not found: {root}")
    labelled: list[VolumeRecord] = []
    unlabeled: list[VolumeRecord] = []
    sealed: list[VolumeRecord] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        is_sealed = directory.name == SEALED_DIRECTORY
        images = sorted(directory.glob("*_image.nii.gz"))
        labels = sorted(directory.glob("*_label.nii.gz"))
        expected_label_paths: set[Path] = set()
        for image in images:
            record = _record_from_image(image, sealed=is_sealed)
            if record is None:
                if is_sealed:
                    # Locked content is never decoded, but malformed filename
                    # metadata must not silently disappear from its inventory.
                    raise SealedTestError(f"Malformed locked-test image filename: {image.name}")
                raise ValueError(f"Malformed CardiacUDC development image filename: {image.name}")
            expected_label_paths.add(
                image.with_name(image.name.replace("_image.nii.gz", "_label.nii.gz"))
            )
            if is_sealed:
                sealed.append(record)
            elif record.label_path is None:
                unlabeled.append(record)
            else:
                labelled.append(record)
        orphan_labels = [label for label in labels if label not in expected_label_paths]
        if orphan_labels:
            if is_sealed:
                raise SealedTestError(
                    f"Locked-test label filename has no enumerated image: {orphan_labels[0].name}"
                )
            raise ValueError(
                f"CardiacUDC development label has no paired image: {orphan_labels[0].name}"
            )
    return labelled, unlabeled, sealed


def _normalise_domain(value: str) -> str:
    candidate = value.strip()
    if candidate in {"G", "R"}:
        return f"Site_{candidate}"
    return candidate


def split_domain_disjoint(
    records: Sequence[VolumeRecord], validation_domain: str | None = None
) -> tuple[list[VolumeRecord], list[VolumeRecord], str]:
    if any(record.sealed for record in records):
        raise SealedTestError("Sealed CardiacUDC records cannot enter development splitting")
    domains = sorted({record.domain for record in records})
    if len(domains) < 2:
        raise ValueError("A domain-disjoint split requires at least two labelled domains")
    selected_domain = _normalise_domain(validation_domain) if validation_domain else domains[-1]
    if selected_domain not in domains:
        raise ValueError(f"Validation domain {selected_domain!r} is absent; available domains: {domains}")
    train = sorted((record for record in records if record.domain != selected_domain), key=lambda item: item.patient_id)
    validation = sorted((record for record in records if record.domain == selected_domain), key=lambda item: item.patient_id)
    if not train or not validation:
        raise ValueError("Both domain-disjoint development partitions must be non-empty")
    train_patients = {record.patient_id for record in train}
    validation_patients = {record.patient_id for record in validation}
    if train_patients & validation_patients:
        raise ValueError("Patient leakage detected across CardiacUDC partitions")
    if {record.domain for record in train} & {record.domain for record in validation}:
        raise ValueError("Domain leakage detected across CardiacUDC partitions")
    return train, validation, selected_domain


def _inspect_volume(record: VolumeRecord) -> tuple[list[int], str | None]:
    if record.sealed or _path_is_sealed(record.image_path) or (
        record.label_path is not None and _path_is_sealed(record.label_path)
    ):
        raise SealedTestError("Sealed CardiacUDC test content cannot be inspected during development")
    if record.label_path is None:
        return [], "missing label"
    try:
        image_nii = nib.load(str(record.image_path))
        label_nii = nib.load(str(record.label_path))
        if len(image_nii.shape) != 3 or image_nii.shape != label_nii.shape:
            return [], f"shape mismatch image={image_nii.shape} label={label_nii.shape}"
        label = np.asanyarray(label_nii.dataobj)
        remap_mask(label, record.view)
        mask_values = np.where(label == 255, 0, label)
        mask_sum = mask_values.sum(axis=(0, 1), dtype=np.float64)
        annotated = np.flatnonzero(mask_sum > ANNOTATED_MASK_SUM_THRESHOLD).astype(int).tolist()
        if not annotated:
            return [], f"no frames with raw mask sum > {ANNOTATED_MASK_SUM_THRESHOLD:g}"
        return annotated, None
    except Exception as error:  # Corrupt public archives must fail closed per volume.
        return [], f"{type(error).__name__}: {error}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _approved_quarantine_evidence(
    record: VolumeRecord, error: str
) -> dict[str, Any] | None:
    key = f"{record.patient_id}:{record.view}"
    expected = APPROVED_DEVELOPMENT_QUARANTINE.get(key)
    if expected is None or record.label_path is None or error != expected["error"]:
        return None
    _assert_not_sealed_path(record.image_path)
    _assert_not_sealed_path(record.label_path)
    observed = {
        "error": error,
        "image_file_sha256": _file_sha256(record.image_path),
        "label_file_sha256": _file_sha256(record.label_path),
    }
    if observed != expected:
        return None
    return {
        "allowlist_key": key,
        "policy": "exact_error_and_compressed_artifact_hash_match",
        **observed,
    }


def _array_content_identity(path: Path) -> dict[str, Any]:
    _assert_not_sealed_path(path)
    values = np.asanyarray(nib.load(str(path)).dataobj)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"Development NIfTI has non-numeric data: {path}")
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(canonical.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(memoryview(canonical))
    return {
        "shape": list(canonical.shape),
        "source_dtype": str(values.dtype),
        "canonical_dtype": "little_endian_float64",
        "sha256": digest.hexdigest(),
    }


def _development_volume_identity(record: VolumeRecord) -> dict[str, Any]:
    if record.sealed or _path_is_sealed(record.image_path) or record.label_path is None or _path_is_sealed(record.label_path):
        raise SealedTestError("Sealed or unlabeled content cannot enter the development content audit")
    image = _array_content_identity(record.image_path)
    label = _array_content_identity(record.label_path)
    combined = hashlib.sha256(
        json.dumps({"image": image, "label": label}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"image": image, "label": label, "combined_sha256": combined}


def audit_development_content_disjoint(
    train: Sequence[VolumeRecord], validation: Sequence[VolumeRecord]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Decode development volumes only and reject exact image leakage across partitions."""

    identities: dict[str, dict[str, Any]] = {}

    def volume_key(record: VolumeRecord) -> str:
        return f"{record.patient_id}:{record.view}"

    for record in (*train, *validation):
        key = volume_key(record)
        if key in identities:
            raise ValueError(f"Duplicate development volume key: {key}")
        identities[key] = _development_volume_identity(record)

    def digest_to_keys(records: Sequence[VolumeRecord]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for record in records:
            key = volume_key(record)
            digest = str(identities[key]["image"]["sha256"])
            result.setdefault(digest, []).append(key)
        return result

    train_hashes = digest_to_keys(train)
    validation_hashes = digest_to_keys(validation)
    overlap = sorted(set(train_hashes) & set(validation_hashes))
    if overlap:
        digest = overlap[0]
        raise ValueError(
            "Exact decoded image-content leakage across development partitions: "
            f"sha256={digest[:16]} train={train_hashes[digest]} validation={validation_hashes[digest]}"
        )
    train_duplicates = sum(max(0, len(keys) - 1) for keys in train_hashes.values())
    validation_duplicates = sum(max(0, len(keys) - 1) for keys in validation_hashes.values())
    if train_duplicates or validation_duplicates:
        duplicated = {
            "train": {digest: keys for digest, keys in train_hashes.items() if len(keys) > 1},
            "validation": {
                digest: keys for digest, keys in validation_hashes.items() if len(keys) > 1
            },
        }
        raise ValueError(
            "Exact decoded image-content duplicates within a patient-independent partition: "
            f"{duplicated}"
        )
    return (
        {
            "algorithm": "sha256(shape || canonical_little_endian_float64_decoded_values)",
            "source_dtype_recorded_in_manifest_fingerprint": True,
            "train_unique_images": len(train_hashes),
            "validation_unique_images": len(validation_hashes),
            "train_within_partition_duplicate_images": train_duplicates,
            "validation_within_partition_duplicate_images": validation_duplicates,
            "cross_partition_duplicates": 0,
            "near_duplicate_audit_complete": False,
            "promotion_note": "exact-content gate only; perceptual near-duplicate audit remains required before promotion",
            "sealed_directory_decoded": False,
        },
        identities,
    )


def _apply_approved_duplicate_annotation_quarantine(
    records: Sequence[VolumeRecord],
) -> tuple[list[VolumeRecord], list[dict[str, Any]]]:
    """Drop only byte- and decoded-identity-bound alternate annotations."""

    selected = list(records)
    by_key = {f"{record.patient_id}:{record.view}": record for record in selected}
    evidence: list[dict[str, Any]] = []
    for excluded_key, expected in APPROVED_DUPLICATE_ANNOTATION_QUARANTINE.items():
        excluded = by_key.get(excluded_key)
        if excluded is None:
            continue
        retained_key = expected["retained_key"]
        retained = by_key.get(retained_key)
        if retained is None:
            raise ValueError(
                "Approved duplicate-annotation quarantine is incomplete: "
                f"{excluded_key} exists without retained acquisition {retained_key}"
            )
        if excluded.domain != retained.domain or excluded.view != retained.view:
            raise ValueError("Duplicate-annotation quarantine crossed a domain or view")
        if excluded.label_path is None or retained.label_path is None:
            raise ValueError("Duplicate-annotation quarantine requires two labelled volumes")
        excluded_identity = _development_volume_identity(excluded)
        retained_identity = _development_volume_identity(retained)
        observed = {
            "retained_key": retained_key,
            "image_file_sha256": _file_sha256(excluded.image_path),
            "label_file_sha256": _file_sha256(excluded.label_path),
            "decoded_image_sha256": str(excluded_identity["image"]["sha256"]),
            "decoded_label_sha256": str(excluded_identity["label"]["sha256"]),
            "retained_image_file_sha256": _file_sha256(retained.image_path),
            "retained_label_file_sha256": _file_sha256(retained.label_path),
            "retained_decoded_label_sha256": str(retained_identity["label"]["sha256"]),
        }
        if observed != expected:
            raise ValueError(
                "Approved duplicate-annotation quarantine evidence changed for "
                f"{excluded_key}"
            )
        if excluded_identity["image"]["sha256"] != retained_identity["image"]["sha256"]:
            raise ValueError("Approved duplicate annotations no longer share exact image content")
        selected.remove(excluded)
        evidence.append(
            {
                "patient_id": excluded.patient_id,
                "path": str(excluded.label_path),
                "error": "exact image duplicated under a second patient identifier with a distinct annotation",
                "approved_quarantine": True,
                "quarantine_evidence": {
                    "allowlist_key": excluded_key,
                    "policy": "exact_compressed_and_decoded_identity_bound_alternate_annotation",
                    **observed,
                },
            }
        )
    return selected, evidence


def _development_file_inventory(root: Path) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    paths = sorted(
        path
        for directory in root.iterdir()
        if directory.is_dir() and directory.name != SEALED_DIRECTORY
        for path in directory.rglob("*")
        if path.is_file()
    )
    entries = [
        {
            "relative_path": path.resolve(strict=True).relative_to(root).as_posix(),
            "size_bytes": int(path.stat().st_size),
        }
        for path in paths
    ]
    fingerprint = _canonical_fingerprint({"entries": entries})
    approval = APPROVED_DEVELOPMENT_FILE_INVENTORIES.get(fingerprint)
    if approval is not None and int(approval["entry_count"]) != len(entries):
        raise ValueError("Approved development-file inventory count is inconsistent")
    return {
        "schema_version": 1,
        "evidence_level": "all_nonsealed_relative_filenames_and_sizes",
        "pixel_bytes_read": False,
        "entry_count": len(entries),
        "fingerprint": fingerprint,
        "approved": approval is not None,
        "approval": approval,
        "entries": entries,
    }


def _unlabeled_filename_inventory(
    root: Path, records: Sequence[VolumeRecord]
) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item.image_path)):
        if record.sealed or record.label_path is not None or _path_is_sealed(record.image_path):
            raise ValueError("Unlabeled inventory received an ineligible record")
        resolved = record.image_path.resolve(strict=True)
        entries.append(
            {
                "relative_path": resolved.relative_to(root).as_posix(),
                "size_bytes": int(resolved.stat().st_size),
            }
        )
    fingerprint = _canonical_fingerprint({"entries": entries})
    if entries:
        approved = APPROVED_UNLABELED_INVENTORIES.get(fingerprint)
        if approved is None or int(approved["entry_count"]) != len(entries):
            raise ValueError(
                "Unexpected missing development label(s): unlabeled filename inventory is not approved "
                f"(count={len(entries)} fingerprint={fingerprint})"
            )
    else:
        approved = {"entry_count": 0, "basis": "no unlabeled development images"}
    return {
        "schema_version": 1,
        "evidence_level": "filename_and_size_metadata_only",
        "pixel_bytes_read": False,
        "entry_count": len(entries),
        "fingerprint": fingerprint,
        "approved": True,
        "approval": approved,
        "entries": entries,
    }


def _sealed_filename_inventory(root: Path, records: Sequence[VolumeRecord]) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: str(item.image_path)):
        if not record.sealed or not _path_is_sealed(record.image_path):
            raise SealedTestError("Sealed filename inventory received a non-sealed record")
        paths = [record.image_path]
        if record.label_path is not None:
            paths.append(record.label_path)
        for path in paths:
            resolved = Path(path).resolve(strict=True)
            relative = resolved.relative_to(root)
            entries.append(
                {
                    "relative_path": relative.as_posix(),
                    "size_bytes": int(resolved.stat().st_size),
                }
            )
    fingerprint = _canonical_fingerprint({"entries": entries})
    return {
        "schema_version": 1,
        "evidence_level": "filename_and_size_metadata_only",
        "pixel_bytes_read": False,
        "entry_count": len(entries),
        "fingerprint": fingerprint,
        "entries": entries,
    }


def _fingerprint(
    records: Iterable[FrameRecord],
    identities: Mapping[str, Mapping[str, Any]],
    quarantine_errors: Sequence[Mapping[str, Any]],
    development_file_inventory: Mapping[str, Any],
    unlabeled_inventory: Mapping[str, Any],
    sealed_inventory: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.key):
        digest.update(record.key.encode("utf-8"))
        digest.update(b"\0")
        volume_key = f"{record.volume.patient_id}:{record.volume.view}"
        identity = identities.get(volume_key)
        if identity is None:
            raise ValueError(f"Missing audited content identity for {volume_key}")
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
    digest.update(
        json.dumps(
            list(quarantine_errors), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(
        json.dumps(
            development_file_inventory, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(
        json.dumps(
            unlabeled_inventory, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(
        json.dumps(sealed_inventory, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def build_manifest(
    root: Path,
    *,
    validation_domain: str | None = "Site_R",
    strict: bool = False,
    maximum_volumes_per_split: int | None = None,
    maximum_frames_per_volume: int | None = None,
) -> ManifestBundle:
    labelled, unlabeled, sealed = discover_volumes(root)
    # This module is deliberately a narrow single-frame A4C research baseline.
    # Other schemas remain available for ontology validation, but they cannot
    # silently broaden the model's declared capability.
    labelled = [record for record in labelled if record.view == BASELINE_VIEW]
    unlabeled = [record for record in unlabeled if record.view == BASELINE_VIEW]
    train, validation, selected_domain = split_domain_disjoint(labelled, validation_domain)
    if maximum_volumes_per_split is not None:
        train = train[:maximum_volumes_per_split]
        validation = validation[:maximum_volumes_per_split]

    errors: list[dict[str, Any]] = []

    def inspect_partition(records: Sequence[VolumeRecord]) -> tuple[list[VolumeRecord], list[FrameRecord]]:
        valid_volumes: list[VolumeRecord] = []
        frames: list[FrameRecord] = []
        for record in records:
            annotated, error = _inspect_volume(record)
            if error is not None:
                quarantine = _approved_quarantine_evidence(record, error)
                errors.append(
                    {
                        "patient_id": record.patient_id,
                        "path": str(record.label_path),
                        "error": error,
                        "approved_quarantine": quarantine is not None,
                        "quarantine_evidence": quarantine,
                    }
                )
                continue
            if maximum_frames_per_volume is not None:
                annotated = annotated[:maximum_frames_per_volume]
            valid_volumes.append(record)
            frames.extend(FrameRecord(record, index) for index in annotated)
        return valid_volumes, frames

    train_valid, train_frames = inspect_partition(train)
    validation_valid, validation_frames = inspect_partition(validation)
    train_valid, train_duplicate_quarantine = _apply_approved_duplicate_annotation_quarantine(
        train_valid
    )
    validation_valid, validation_duplicate_quarantine = (
        _apply_approved_duplicate_annotation_quarantine(validation_valid)
    )
    duplicate_quarantine = [
        *train_duplicate_quarantine,
        *validation_duplicate_quarantine,
    ]
    if duplicate_quarantine:
        excluded_keys = {
            f"{item['patient_id']}:A4C" for item in duplicate_quarantine
        }
        train_frames = [
            frame
            for frame in train_frames
            if f"{frame.volume.patient_id}:{frame.volume.view}" not in excluded_keys
        ]
        validation_frames = [
            frame
            for frame in validation_frames
            if f"{frame.volume.patient_id}:{frame.volume.view}" not in excluded_keys
        ]
        errors.extend(duplicate_quarantine)
    unapproved_errors = [error for error in errors if not error["approved_quarantine"]]
    if unapproved_errors:
        raise ValueError(
            "CardiacUDC manifest contains an unapproved corrupt/invalid development volume: "
            f"{unapproved_errors[0]}"
        )
    if strict and errors:
        raise ValueError(f"CardiacUDC manifest has {len(errors)} invalid volume(s): {errors[0]}")
    if not train_frames or not validation_frames:
        raise ValueError("No valid annotated frames remain in one or both development partitions")
    development_file_inventory = _development_file_inventory(root)
    unlabeled_inventory = _unlabeled_filename_inventory(root, unlabeled)
    content_audit, identities = audit_development_content_disjoint(train_valid, validation_valid)
    sealed_inventory = _sealed_filename_inventory(root, sealed)
    fingerprint = _fingerprint(
        (*train_frames, *validation_frames),
        identities,
        errors,
        development_file_inventory,
        unlabeled_inventory,
        sealed_inventory,
    )
    return ManifestBundle(
        train_volumes=tuple(train_valid),
        validation_volumes=tuple(validation_valid),
        train_frames=tuple(train_frames),
        validation_frames=tuple(validation_frames),
        sealed_volumes=tuple(sealed),
        unlabeled_volumes=tuple(unlabeled),
        errors=tuple(errors),
        validation_domain=selected_domain,
        fingerprint=fingerprint,
        content_audit=content_audit,
        development_file_inventory=development_file_inventory,
        unlabeled_inventory=unlabeled_inventory,
        sealed_inventory=sealed_inventory,
    )


def _nifti_digest(path: Path) -> tuple[tuple[int, ...], str]:
    _assert_not_sealed_path(path)
    values = np.ascontiguousarray(np.asanyarray(nib.load(str(path)).dataobj))
    return values.shape, hashlib.sha256(memoryview(values)).hexdigest()


def audit_optional_cardiacnet(
    root: Path | None,
    *,
    cardiacuda_root: Path | None = None,
    maximum_duplicate_checks: int = 3,
) -> dict[str, Any]:
    if root is None:
        return {"present": False, "eligible_for_segmentation": False, "reason": "not requested"}
    root = Path(root)
    if not root.is_dir():
        return {"present": False, "eligible_for_segmentation": False, "reason": "root not found"}
    files = list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
    labels = sorted(path for path in root.rglob("*_label.nii*") if path.is_file())
    pairs: list[tuple[Path, Path]] = []
    for label in labels:
        prefix = label.name.split("_label.nii", 1)[0]
        containers = [path for path in label.parent.glob(f"{prefix}_image.nii*") if path.is_dir()]
        images = [
            nested
            for container in containers
            for nested in container.rglob("*_image.nii*")
            if nested.is_file()
        ]
        if len(images) == 1:
            pairs.append((images[0], label))
    license_files = [
        path for path in root.rglob("*") if path.is_file() and path.name.lower() in {"license", "license.txt", "license.md"}
    ]
    verified_duplicates: list[dict[str, str]] = []
    if cardiacuda_root is not None and Path(cardiacuda_root).is_dir():
        for image, _label in pairs[:maximum_duplicate_checks]:
            candidate_name = image.name + ".gz" if image.name.endswith(".nii") else image.name
            candidates = sorted(Path(cardiacuda_root).rglob(candidate_name))
            try:
                image_digest = _nifti_digest(image)
            except Exception:
                continue
            for candidate in candidates:
                try:
                    if image_digest == _nifti_digest(candidate):
                        verified_duplicates.append(
                            {"cardiacnet_image": str(image), "cardiacuda_image": str(candidate)}
                        )
                        break
                except Exception:
                    continue
    complete = len(pairs) == 750 and len(labels) == 750
    return {
        "present": True,
        "nifti_files": len(files),
        "pixel_mask_pairs": len(pairs),
        "unpaired_labels": len(labels) - len(pairs),
        "official_expected_videos": 750,
        "complete": complete,
        "view": "A4C",
        "structures": ["LV", "RV", "LA", "RA"],
        "license": "CC-BY-4.0 (official Kaggle dataset metadata)",
        "license_source": "https://www.kaggle.com/datasets/xiaoweixumedicalai/abnormcardiacechovideos",
        "license_files": [str(path) for path in license_files],
        "verified_cardiacudc_duplicates": verified_duplicates,
        "eligible_for_segmentation": False,
        "reason": (
            "excluded to prevent patient leakage: the local extraction is incomplete and contains byte-identical CardiacUDC videos"
            if verified_duplicates
            else "excluded pending complete extraction and patient-level deduplication against CardiacUDC"
        ),
    }


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _worker_seed(worker_id: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _policy_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def make_spatial_transform(source_shape: Sequence[int], output_size: int) -> SpatialTransform:
    if len(source_shape) != 2:
        raise ValueError(f"Expected an HxW source shape, got {tuple(source_shape)}")
    source_height, source_width = (int(value) for value in source_shape)
    output_size = int(output_size)
    if min(source_height, source_width, output_size) <= 0:
        raise ValueError("Source and output dimensions must be positive")
    intended_scale = min(output_size / source_height, output_size / source_width)
    resized_height = min(output_size, max(1, int(round(source_height * intended_scale))))
    resized_width = min(output_size, max(1, int(round(source_width * intended_scale))))
    pad_top = (output_size - resized_height) // 2
    pad_bottom = output_size - resized_height - pad_top
    pad_left = (output_size - resized_width) // 2
    pad_right = output_size - resized_width - pad_left
    scale_y = resized_height / source_height
    scale_x = resized_width / source_width
    # align_corners=False maps pixel centres as (x + .5) * scale - .5.
    offset_x = pad_left + 0.5 * scale_x - 0.5
    offset_y = pad_top + 0.5 * scale_y - 0.5
    forward = np.asarray(
        ((scale_x, 0.0, offset_x), (0.0, scale_y, offset_y), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    inverse = np.linalg.inv(forward)
    transform = SpatialTransform(
        source_height=source_height,
        source_width=source_width,
        output_height=output_size,
        output_width=output_size,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
        scale_y=scale_y,
        scale_x=scale_x,
        forward_matrix=tuple(tuple(float(item) for item in row) for row in forward),
        inverse_matrix=tuple(tuple(float(item) for item in row) for row in inverse),
    )
    transform.validate()
    return transform


def apply_spatial_transform(
    image: np.ndarray,
    target: np.ndarray,
    output_size: int,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[np.ndarray, np.ndarray, SpatialTransform]:
    import torch
    import torch.nn.functional as functional

    image = np.asarray(image, dtype=np.float32)
    target = np.asarray(target, dtype=np.int64)
    if image.ndim != 2 or target.shape != image.shape:
        raise ValueError(f"Letterbox input must be paired HxW arrays, got {image.shape} and {target.shape}")
    transform = make_spatial_transform(image.shape, output_size)
    image_tensor = torch.from_numpy(image.copy()).unsqueeze(0).unsqueeze(0)
    target_tensor = torch.from_numpy(target.copy()).unsqueeze(0).unsqueeze(0).float()
    image_resized = functional.interpolate(
        image_tensor,
        size=(transform.resized_height, transform.resized_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    target_resized = functional.interpolate(
        target_tensor,
        size=(transform.resized_height, transform.resized_width),
        mode="nearest",
    ).squeeze(0).squeeze(0).long()
    image_output = torch.full(
        (transform.output_height, transform.output_width), 0.0, dtype=image_resized.dtype
    )
    target_output = torch.full(
        (transform.output_height, transform.output_width), int(ignore_index), dtype=torch.long
    )
    rows = slice(transform.pad_top, transform.pad_top + transform.resized_height)
    columns = slice(transform.pad_left, transform.pad_left + transform.resized_width)
    image_output[rows, columns] = image_resized
    target_output[rows, columns] = target_resized
    return image_output.cpu().numpy(), target_output.cpu().numpy(), transform


def invert_spatial_transform(
    values: np.ndarray,
    transform: SpatialTransform | Mapping[str, Any],
    *,
    mode: str,
) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    if not isinstance(transform, SpatialTransform):
        transform = SpatialTransform.from_dict(transform)
    else:
        transform.validate()
    array = np.asarray(values)
    if array.shape != (transform.output_height, transform.output_width):
        raise ValueError(
            f"Cannot invert array shape {array.shape}; expected "
            f"{(transform.output_height, transform.output_width)}"
        )
    rows = slice(transform.pad_top, transform.pad_top + transform.resized_height)
    columns = slice(transform.pad_left, transform.pad_left + transform.resized_width)
    cropped = array[rows, columns]
    tensor = torch.from_numpy(np.ascontiguousarray(cropped)).unsqueeze(0).unsqueeze(0).float()
    if mode == "nearest":
        restored = functional.interpolate(
            tensor, size=(transform.source_height, transform.source_width), mode="nearest"
        )
    elif mode == "bilinear":
        restored = functional.interpolate(
            tensor,
            size=(transform.source_height, transform.source_width),
            mode="bilinear",
            align_corners=False,
        )
    else:
        raise ValueError("Inverse interpolation mode must be 'nearest' or 'bilinear'")
    result = restored.squeeze(0).squeeze(0).cpu().numpy()
    if mode == "nearest" and np.issubdtype(array.dtype, np.integer):
        result = np.rint(result).astype(array.dtype, copy=False)
    return result


def validate_transform_record(record: Mapping[str, Any]) -> None:
    if set(record) != {
        "schema_version",
        "scope",
        "sample_key",
        "normalization",
        "spatial",
        "augmentations",
    }:
        raise ValueError("Transform record has unknown or missing top-level fields")
    if record["schema_version"] != 1 or record["scope"] != BASELINE_SCOPE:
        raise ValueError("Transform record schema or scope mismatch")
    if not isinstance(record["sample_key"], str) or not record["sample_key"]:
        raise ValueError("Transform record requires a non-empty sample key")
    normalization = record["normalization"]
    if set(normalization) != {
        "operation",
        "lower_percentile",
        "upper_percentile",
        "lower_value",
        "upper_value",
        "clip_range",
        "storage_quantization",
    }:
        raise ValueError("Normalization audit entry is incomplete")
    if (
        normalization["operation"] != "finite_percentile_minmax"
        or float(normalization["lower_percentile"]) != 1.0
        or float(normalization["upper_percentile"]) != 99.0
        or normalization["clip_range"] != [0.0, 1.0]
        or normalization["storage_quantization"] != "round(value*255)_uint8"
    ):
        raise ValueError("Normalization audit entry does not match the approved policy")
    lower_value = float(normalization["lower_value"])
    upper_value = float(normalization["upper_value"])
    if not math.isfinite(lower_value) or not math.isfinite(upper_value) or upper_value < lower_value:
        raise ValueError("Normalization bounds are invalid")
    SpatialTransform.from_dict(record["spatial"])
    augmentations = record["augmentations"]
    if not isinstance(augmentations, list) or [item.get("name") for item in augmentations] != [
        "horizontal_flip",
        "intensity_gamma_gain_gaussian_noise",
    ]:
        raise ValueError("Transform record augmentation chain is not the approved ordered policy")
    flip, intensity = augmentations
    if set(flip) != {"name", "enabled", "applied", "parity_flag"}:
        raise ValueError("Horizontal-flip audit entry is incomplete")
    if flip["enabled"] is not False or flip["applied"] is not False or flip["parity_flag"] is not False:
        raise ValueError("Unpromoted horizontal flip must remain disabled and unapplied")
    if set(intensity) != {
        "name",
        "enabled",
        "applied",
        "gamma",
        "gain",
        "noise_standard_deviation",
        "noise_seed",
    }:
        raise ValueError("Intensity augmentation audit entry is incomplete")
    enabled = intensity["enabled"]
    applied = intensity["applied"]
    if not isinstance(enabled, bool) or not isinstance(applied, bool) or applied != enabled:
        raise ValueError("Intensity augmentation enabled/applied state is inconsistent")
    gamma = float(intensity["gamma"])
    gain = float(intensity["gain"])
    noise = float(intensity["noise_standard_deviation"])
    seed = int(intensity["noise_seed"])
    if enabled:
        if not (0.9 <= gamma <= 1.1 and 0.9 <= gain <= 1.1 and 0.0 <= noise <= 0.015):
            raise ValueError("Intensity augmentation parameters are outside the declared policy")
        if not (0 <= seed < 2**31):
            raise ValueError("Intensity augmentation seed is outside the declared range")
    elif (gamma, gain, noise, seed) != (1.0, 1.0, 0.0, 0):
        raise ValueError("Disabled intensity augmentation must record identity parameters")


class CardiacUDCFrameDataset:
    def __init__(
        self,
        records: Sequence[FrameRecord],
        *,
        resolution: int = 256,
        augment: bool = False,
        cache_size: int = 4,
        preload: bool = False,
    ) -> None:
        if any(
            record.volume.sealed
            or _path_is_sealed(record.volume.image_path)
            or (
                record.volume.label_path is not None
                and _path_is_sealed(record.volume.label_path)
            )
            for record in records
        ):
            raise SealedTestError("Sealed CardiacUDC frames cannot enter a development dataset")
        self.records = tuple(records)
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._preloaded: list[
            tuple[np.ndarray, np.ndarray, SpatialTransform, dict[str, Any]]
        ] | None = None
        if preload:
            self._preloaded = [self._prepare_frame(record) for record in self.records]
            self._cache.clear()

    def __len__(self) -> int:
        return len(self.records)

    def _volume(self, record: VolumeRecord) -> tuple[np.ndarray, np.ndarray]:
        if (
            record.sealed
            or _path_is_sealed(record.image_path)
            or record.label_path is None
            or _path_is_sealed(record.label_path)
        ):
            raise SealedTestError(
                "Sealed or unlabeled CardiacUDC content cannot cross a decode boundary"
            )
        key = str(record.image_path)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        image = np.asanyarray(nib.load(str(record.image_path)).dataobj).astype(np.float32, copy=False)
        raw = np.asanyarray(nib.load(str(record.label_path)).dataobj)
        if image.shape != raw.shape or image.ndim != 3:
            raise ValueError(f"Invalid paired volume for {record.patient_id}: {image.shape}, {raw.shape}")
        value = (image, raw)
        if self.cache_size:
            self._cache[key] = value
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value

    @staticmethod
    def _normalise_image_with_record(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        finite = image[np.isfinite(image)]
        if not len(finite):
            raise ValueError("Ultrasound frame has no finite pixels")
        if len(finite) != image.size:
            raise ValueError("Ultrasound frame contains non-finite pixels")
        lower, upper = np.percentile(finite, (1.0, 99.0))
        if upper <= lower:
            lower, upper = float(finite.min()), float(finite.max())
        record = {
            "operation": "finite_percentile_minmax",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "lower_value": float(lower),
            "upper_value": float(upper),
            "clip_range": [0.0, 1.0],
            "storage_quantization": "round(value*255)_uint8",
        }
        if upper <= lower:
            return np.zeros_like(image, dtype=np.float32), record
        return (
            np.clip((image - lower) / (upper - lower), 0.0, 1.0).astype(np.float32),
            record,
        )

    @staticmethod
    def _normalise_image(image: np.ndarray) -> np.ndarray:
        return CardiacUDCFrameDataset._normalise_image_with_record(image)[0]

    def _prepare_frame(
        self, frame_record: FrameRecord
    ) -> tuple[np.ndarray, np.ndarray, SpatialTransform, dict[str, Any]]:
        volume = frame_record.volume
        image_volume, raw_volume = self._volume(volume)
        image, normalization = self._normalise_image_with_record(
            image_volume[:, :, frame_record.frame_index]
        )
        target = remap_mask(raw_volume[:, :, frame_record.frame_index], volume.view)
        transformed_image, transformed_target, transform = apply_spatial_transform(
            image, target, self.resolution
        )
        cached_image = np.rint(np.clip(transformed_image, 0, 1) * 255.0).astype(np.uint8)
        cached_target = transformed_target.astype(np.int16, copy=False)
        return cached_image, cached_target, transform, normalization

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        frame_record = self.records[index]
        volume = frame_record.volume
        cached_image, cached_target, transform, normalization = (
            self._preloaded[index] if self._preloaded is not None else self._prepare_frame(frame_record)
        )
        image_tensor = torch.from_numpy(cached_image.copy()).unsqueeze(0).float() / 255.0
        target_tensor = torch.from_numpy(cached_target.copy()).long()
        if self.augment:
            # A worker-seeded global draw chooses a complete local augmentation
            # stream. Recording it makes the exact noise field reproducible.
            noise_seed = int(torch.randint(0, 2**31, (1,)).item())
            augmentation_generator = torch.Generator().manual_seed(noise_seed)
            gamma = 0.9 + 0.2 * float(torch.rand((), generator=augmentation_generator))
            gain = 0.9 + 0.2 * float(torch.rand((), generator=augmentation_generator))
            noise = 0.015 * float(torch.rand((), generator=augmentation_generator))
            image_tensor = (image_tensor.clamp_min(1e-5).pow(gamma) * gain).clamp(0, 1)
            sampled_noise = torch.randn(
                image_tensor.shape,
                generator=augmentation_generator,
                dtype=image_tensor.dtype,
            )
            image_tensor = (image_tensor + sampled_noise * noise).clamp(0, 1)
        else:
            noise_seed = 0
            gamma = 1.0
            gain = 1.0
            noise = 0.0
        transform_record = {
            "schema_version": 1,
            "scope": BASELINE_SCOPE,
            "sample_key": frame_record.key,
            "normalization": normalization,
            "spatial": transform.to_dict(),
            "augmentations": [
                {
                    "name": "horizontal_flip",
                    "enabled": False,
                    "applied": False,
                    "parity_flag": False,
                },
                {
                    "name": "intensity_gamma_gain_gaussian_noise",
                    "enabled": self.augment,
                    "applied": self.augment,
                    "gamma": gamma,
                    "gain": gain,
                    "noise_standard_deviation": noise,
                    "noise_seed": noise_seed,
                },
            ],
        }
        validate_transform_record(transform_record)
        class_valid = torch.from_numpy(active_class_mask(volume.view))
        return {
            "image": image_tensor.float(),
            "target": target_tensor,
            "view_id": torch.tensor(VIEW_TO_INDEX[volume.view], dtype=torch.long),
            "class_valid": class_valid,
            "patient_id": volume.patient_id,
            "domain": volume.domain,
            "frame_index": frame_record.frame_index,
            "transform_json": json.dumps(
                transform_record, sort_keys=True, separators=(",", ":")
            ),
        }


def _group_norm(channels: int):
    import torch

    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return torch.nn.GroupNorm(groups, channels)


class _ConditionedBlock:
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int) -> None:
        import torch

        self.module = torch.nn.ModuleDict(
            {
                "conv1": torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                "norm1": _group_norm(out_channels),
                "conv2": torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                "norm2": _group_norm(out_channels),
                "film": torch.nn.Linear(condition_dim, out_channels * 2),
                "skip": torch.nn.Conv2d(in_channels, out_channels, 1, bias=False)
                if in_channels != out_channels
                else torch.nn.Identity(),
            }
        )

    def __call__(self, values, condition):
        import torch.nn.functional as functional

        residual = self.module["skip"](values)
        values = functional.silu(self.module["norm1"](self.module["conv1"](values)))
        values = self.module["norm2"](self.module["conv2"](values))
        scale, bias = self.module["film"](condition).chunk(2, dim=1)
        values = values * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        return functional.silu(values + residual)


class ViewConditionedUNet:
    """Small research U-Net; the training entrypoint is intentionally A4C-only."""

    def __init__(self, base_channels: int = 32, condition_dim: int = 64) -> None:
        import torch

        widths = [base_channels * multiplier for multiplier in (1, 2, 4, 8, 16)]
        self.module = torch.nn.Module()
        self.module.view_embedding = torch.nn.Embedding(len(VIEW_NAMES), condition_dim)
        self.module.encoders = torch.nn.ModuleList()
        self._encoders: list[_ConditionedBlock] = []
        for index, width in enumerate(widths):
            in_channels = 1 if index == 0 else width
            block = _ConditionedBlock(in_channels, width, condition_dim)
            self.module.encoders.append(block.module)
            self._encoders.append(block)
        self.module.down = torch.nn.ModuleList(
            [torch.nn.Conv2d(widths[index], widths[index + 1], 3, stride=2, padding=1) for index in range(4)]
        )
        decoder_pairs = [(widths[4], widths[3]), (widths[3], widths[2]), (widths[2], widths[1]), (widths[1], widths[0])]
        self.module.up = torch.nn.ModuleList(
            [torch.nn.ConvTranspose2d(source, target, 2, stride=2) for source, target in decoder_pairs]
        )
        self.module.decoders = torch.nn.ModuleList()
        self._decoders: list[_ConditionedBlock] = []
        for _source, target in decoder_pairs:
            block = _ConditionedBlock(target * 2, target, condition_dim)
            self.module.decoders.append(block.module)
            self._decoders.append(block)
        self.module.segmentation = torch.nn.Conv2d(widths[0], len(GLOBAL_CLASSES), 1)
        self.module.log_variance = torch.nn.Conv2d(widths[0], 1, 1)
        self.base_channels = base_channels
        self.condition_dim = condition_dim

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.module.load_state_dict(state_dict, strict=strict)

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, images, view_ids):
        condition = self.module.view_embedding(view_ids)
        skips = []
        values = images
        for index, block in enumerate(self._encoders):
            values = block(values, condition)
            skips.append(values)
            if index < len(self.module.down):
                values = self.module.down[index](values)
        for index, block in enumerate(self._decoders):
            values = self.module.up[index](values)
            skip = skips[-2 - index]
            if values.shape[-2:] != skip.shape[-2:]:
                import torch.nn.functional as functional

                values = functional.interpolate(values, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            values = block(__import__("torch").cat((values, skip), dim=1), condition)
        return {
            "logits": self.module.segmentation(values),
            "log_variance": self.module.log_variance(values).clamp(-6.0, 4.0),
        }


def masked_logits(logits, class_valid):
    import torch

    if class_valid.ndim != 2 or class_valid.shape != logits.shape[:2]:
        raise ValueError(f"class_valid must be BxC, got {class_valid.shape} for logits {logits.shape}")
    return torch.where(class_valid[:, :, None, None], logits, torch.full_like(logits, -1.0e4))


def _boundary_map(values):
    import torch.nn.functional as functional

    flat = values.flatten(0, 1).unsqueeze(1)
    dilated = functional.max_pool2d(flat, 3, stride=1, padding=1)
    eroded = -functional.max_pool2d(-flat, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1).reshape(values.shape)


def segmentation_loss(
    outputs: dict[str, Any],
    target,
    class_valid,
    *,
    ignore_index: int = IGNORE_INDEX,
    focal_weight: float = 0.5,
    dice_weight: float = 1.0,
    boundary_weight: float = 0.25,
    focal_gamma: float = 2.0,
):
    import torch
    import torch.nn.functional as functional

    logits = outputs["logits"].float()
    valid_logits = masked_logits(logits, class_valid.bool())
    valid_pixels = target.ne(ignore_index)
    safe_target = target.masked_fill(~valid_pixels, 0)
    target_is_valid = class_valid.gather(1, safe_target.flatten(1)).reshape_as(safe_target)
    if not bool((target_is_valid | ~valid_pixels).all()):
        raise ValueError("Target contains a structure that is invalid for its declared view")
    if not bool(valid_pixels.any()):
        raise ValueError("Segmentation batch contains no labelled pixels")
    per_pixel_ce = functional.cross_entropy(valid_logits, target, ignore_index=ignore_index, reduction="none")
    focal_factor = (1.0 - torch.exp(-per_pixel_ce)).pow(float(focal_gamma))
    per_pixel_focal = focal_factor * per_pixel_ce
    log_variance = outputs["log_variance"].float().squeeze(1)
    heteroscedastic = torch.exp(-log_variance) * per_pixel_focal + 0.05 * log_variance
    focal_loss = heteroscedastic[valid_pixels].mean()

    probabilities = functional.softmax(valid_logits, dim=1)
    one_hot = functional.one_hot(safe_target, num_classes=len(GLOBAL_CLASSES)).permute(0, 3, 1, 2).float()
    pixel_mask = valid_pixels[:, None].float()
    active_foreground = class_valid[:, 1:, None, None].float()
    prediction_fg = probabilities[:, 1:] * pixel_mask
    target_fg = one_hot[:, 1:] * pixel_mask
    intersection = (prediction_fg * target_fg).sum(dim=(2, 3))
    denominator = prediction_fg.sum(dim=(2, 3)) + target_fg.sum(dim=(2, 3))
    dice_per_class = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    active_count = active_foreground[:, :, 0, 0].sum().clamp_min(1.0)
    dice_loss = (dice_per_class * active_foreground[:, :, 0, 0]).sum() / active_count

    predicted_boundary = _boundary_map(prediction_fg)
    target_boundary = _boundary_map(target_fg)
    # This boundary objective operates on probabilities after morphology and
    # therefore cannot be rewritten as BCEWithLogits. CUDA autocast explicitly
    # rejects probability-space BCE even when its tensors were cast to float32;
    # keep this small numerically sensitive term in an FP32-disabled region.
    with torch.autocast(device_type=predicted_boundary.device.type, enabled=False):
        boundary_error = functional.binary_cross_entropy(
            predicted_boundary.float().clamp(1e-5, 1 - 1e-5),
            target_boundary.float(),
            reduction="none",
        ).mean(dim=(2, 3))
    boundary_loss = (boundary_error * active_foreground[:, :, 0, 0]).sum() / active_count
    total = focal_weight * focal_loss + dice_weight * dice_loss + boundary_weight * boundary_loss
    return total, {
        "heteroscedastic_focal": focal_loss.detach(),
        "dice_loss": dice_loss.detach(),
        "boundary_loss": boundary_loss.detach(),
    }


def predictive_uncertainty(outputs: dict[str, Any], class_valid):
    """Return an uncalibrated challenger signal, not clinical confidence."""
    import torch
    import torch.nn.functional as functional

    valid = class_valid.bool()
    probabilities = functional.softmax(masked_logits(outputs["logits"].float(), valid), dim=1)
    entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=1)
    class_count = valid.sum(dim=1).float().clamp_min(2.0)
    entropy = entropy / class_count.log()[:, None, None]
    aleatoric = torch.sigmoid(outputs["log_variance"].float().squeeze(1))
    return (0.5 * entropy + 0.5 * aleatoric).clamp(0, 1)


def _contour(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_dilation, binary_erosion

    value = mask.astype(bool)
    return np.logical_xor(binary_dilation(value), binary_erosion(value))


def export_contour_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    output_path: Path,
    *,
    view: str,
) -> Path:
    from PIL import Image, ImageDraw

    view = canonical_view(view)
    image = CardiacUDCFrameDataset._normalise_image(np.asarray(image, dtype=np.float32))
    prediction = np.asarray(prediction, dtype=np.int64)
    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    if image.shape != prediction.shape or image.shape != uncertainty.shape:
        raise ValueError("Overlay image, prediction, and uncertainty must have identical HxW shapes")
    gray = np.rint(image * 255.0).astype(np.uint8)
    canvas = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    heat = np.clip(uncertainty, 0, 1)[..., None]
    canvas = canvas * (1.0 - 0.28 * heat) + np.array([255.0, 40.0, 180.0]) * (0.28 * heat)
    colors = {
        "LV": np.array([0, 230, 255], dtype=np.float32),
        "RV": np.array([255, 175, 0], dtype=np.float32),
        "LA": np.array([80, 255, 100], dtype=np.float32),
        "RA": np.array([255, 80, 80], dtype=np.float32),
        "PA": np.array([175, 90, 255], dtype=np.float32),
    }
    for structure in VIEW_SCHEMAS[view].structures:
        index = GLOBAL_CLASS_TO_INDEX[structure]
        edge = _contour(prediction == index)
        canvas[edge] = colors[structure]
    legend_height = 28
    rendered = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    final = Image.new("RGB", (rendered.width, rendered.height + legend_height), "black")
    final.paste(rendered, (0, 0))
    draw = ImageDraw.Draw(final)
    cursor = 8
    for structure in VIEW_SCHEMAS[view].structures:
        color = tuple(int(value) for value in colors[structure])
        draw.line((cursor, rendered.height + 13, cursor + 15, rendered.height + 13), fill=color, width=3)
        draw.text((cursor + 19, rendered.height + 6), structure, fill="white")
        cursor += 54
    draw.text((cursor + 4, rendered.height + 6), "magenta = uncalibrated uncertainty", fill=(255, 100, 210))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path, format="PNG", optimize=True)
    return output_path


@dataclass(frozen=True)
class TrainingSettings:
    mode: str
    resolution: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    base_channels: int
    num_workers: int
    patience: int
    ema_decay: float
    maximum_volumes_per_split: int | None
    maximum_frames_per_volume: int | None


def settings_for_mode(mode: str) -> TrainingSettings:
    if mode == "smoke":
        return TrainingSettings(
            mode="smoke",
            resolution=96,
            epochs=1,
            batch_size=2,
            learning_rate=3e-4,
            weight_decay=1e-2,
            base_channels=8,
            num_workers=0,
            patience=1,
            ema_decay=0.99,
            maximum_volumes_per_split=2,
            maximum_frames_per_volume=1,
        )
    if mode == "full":
        return TrainingSettings(
            mode="full",
            resolution=256,
            epochs=80,
            batch_size=8,
            learning_rate=2e-4,
            weight_decay=1e-2,
            base_channels=32,
            num_workers=2,
            patience=12,
            ema_decay=0.999,
            maximum_volumes_per_split=None,
            maximum_frames_per_volume=None,
        )
    raise ValueError("mode must be 'smoke' or 'full'")


def _update_ema(ema: ViewConditionedUNet, model: ViewConditionedUNet, decay: float) -> None:
    import torch

    with torch.no_grad():
        current = model.state_dict()
        for name, value in ema.state_dict().items():
            value.mul_(decay).add_(current[name].detach(), alpha=1.0 - decay)


def _new_patient_class_accumulator() -> dict[str, Any]:
    return {
        "intersection": 0,
        "denominator": 0,
        "truth_pixels": 0,
        "surface_distances": [],
        "matched_boundary": 0,
        "boundary_denominator": 0,
    }


def _surface_statistics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    tolerance_pixels: float,
) -> tuple[np.ndarray, int, int]:
    from scipy.ndimage import distance_transform_edt

    predicted_boundary = _contour(prediction)
    truth_boundary = _contour(truth)
    predicted_count = int(predicted_boundary.sum())
    truth_count = int(truth_boundary.sum())
    denominator = predicted_count + truth_count
    if denominator == 0:
        return np.empty(0, dtype=np.float64), 0, 0
    diagonal = float(math.hypot(*prediction.shape))
    if predicted_count == 0 or truth_count == 0:
        nonempty_count = max(predicted_count, truth_count, 1)
        return np.full(nonempty_count, diagonal, dtype=np.float64), 0, denominator
    distance_to_truth = distance_transform_edt(~truth_boundary)
    distance_to_prediction = distance_transform_edt(~predicted_boundary)
    predicted_distances = distance_to_truth[predicted_boundary]
    truth_distances = distance_to_prediction[truth_boundary]
    distances = np.concatenate((predicted_distances, truth_distances)).astype(np.float64, copy=False)
    matched = int((predicted_distances <= tolerance_pixels).sum()) + int(
        (truth_distances <= tolerance_pixels).sum()
    )
    return distances, matched, denominator


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Patient bootstrap requires a non-empty finite one-dimensional metric")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid bootstrap configuration")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(replicates, len(array)))
    draws = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, (tail, 1.0 - tail))
    return {
        "estimate": float(array.mean()),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "replicates": replicates,
        "unit": "patient",
        "patients": len(array),
    }


def _evaluate(
    model: ViewConditionedUNet,
    loader,
    device: str,
    *,
    bootstrap_seed: int = 0,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    import torch

    model.eval()
    patient_totals: dict[str, dict[str, dict[str, Any]]] = {}
    uncertainty_sum = 0.0
    uncertainty_count = 0
    common_grid_shape: tuple[int, int] | None = None
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            views = batch["view_id"].to(device)
            class_valid = batch["class_valid"].to(device)
            outputs = model(images, views)
            _assert_nested_finite(outputs, "validation outputs")
            probabilities = torch.softmax(masked_logits(outputs["logits"].float(), class_valid), dim=1)
            prediction = probabilities.argmax(dim=1)
            valid_pixels = targets.ne(-100)
            uncertainty = predictive_uncertainty(outputs, class_valid)
            if not bool(torch.isfinite(uncertainty).all()):
                raise FloatingPointError("Non-finite predictive uncertainty during validation")
            uncertainty_sum += float(uncertainty[valid_pixels].sum().cpu())
            uncertainty_count += int(valid_pixels.sum().cpu())
            batch_grid_shape = (int(targets.shape[-2]), int(targets.shape[-1]))
            if common_grid_shape is None:
                common_grid_shape = batch_grid_shape
            elif common_grid_shape != batch_grid_shape:
                raise ValueError(
                    "Patient-level boundary metrics require one declared common evaluation grid"
                )
            for sample_index in range(images.shape[0]):
                if "patient_id" not in batch:
                    raise ValueError(
                        "Patient-level evaluation requires explicit patient identifiers"
                    )
                value = batch["patient_id"]
                if isinstance(value, str):
                    if images.shape[0] != 1:
                        raise ValueError("A scalar patient identifier cannot describe a batch")
                    patient_id = value
                else:
                    if len(value) != images.shape[0]:
                        raise ValueError("Patient identifier count differs from evaluation batch")
                    patient_id = str(value[sample_index])
                if not patient_id:
                    raise ValueError("Patient-level evaluation received a blank identifier")
                predicted_array = prediction[sample_index].detach().cpu().numpy().astype(np.int16)
                target_array = targets[sample_index].detach().cpu().numpy().astype(np.int16)
                if "transform_json" in batch:
                    raw_transform = batch["transform_json"]
                    serialized = raw_transform[sample_index] if not isinstance(raw_transform, str) else raw_transform
                    transform_record = json.loads(serialized)
                    validate_transform_record(transform_record)
                    transform = SpatialTransform.from_dict(transform_record["spatial"])
                    if (transform.output_height, transform.output_width) != batch_grid_shape:
                        raise ValueError(
                            "Transform evidence output geometry differs from the evaluation grid"
                        )
                valid_array = target_array != IGNORE_INDEX
                class_totals = patient_totals.setdefault(patient_id, {})
                for class_index, name in enumerate(GLOBAL_CLASSES[1:], start=1):
                    if not bool(class_valid[sample_index, class_index]):
                        continue
                    totals = class_totals.setdefault(name, _new_patient_class_accumulator())
                    predicted_mask = predicted_array == class_index
                    truth_mask = target_array == class_index
                    predicted_mask &= valid_array
                    truth_mask &= valid_array
                    totals["intersection"] += int((predicted_mask & truth_mask).sum())
                    totals["denominator"] += int(predicted_mask.sum() + truth_mask.sum())
                    totals["truth_pixels"] += int(truth_mask.sum())
                    distances, matched, boundary_denominator = _surface_statistics(
                        predicted_mask,
                        truth_mask,
                        tolerance_pixels=BOUNDARY_TOLERANCE_PIXELS,
                    )
                    if len(distances):
                        totals["surface_distances"].append(distances)
                    totals["matched_boundary"] += matched
                    totals["boundary_denominator"] += boundary_denominator

    patient_metrics: dict[str, dict[str, dict[str, float]]] = {}
    undefined_empty_pairs = 0
    undefined_reference_false_positive_pixels = 0
    for patient_id, classes in patient_totals.items():
        patient_metrics[patient_id] = {}
        for name, totals in classes.items():
            denominator = int(totals["denominator"])
            if int(totals["truth_pixels"]) == 0:
                # A chamber absent from the reference is undefined for its
                # per-class geometry metric regardless of model prediction.
                # This keeps the bootstrap denominator model-independent.
                undefined_empty_pairs += 1
                undefined_reference_false_positive_pixels += denominator
                continue
            if denominator <= 0:
                raise ValueError("Positive reference support has an invalid Dice denominator")
            dice = 2.0 * totals["intersection"] / denominator
            boundary_denominator = int(totals["boundary_denominator"])
            boundary_dice = (
                1.0
                if boundary_denominator == 0
                else totals["matched_boundary"] / boundary_denominator
            )
            distance_chunks = totals["surface_distances"]
            hd95 = (
                float(np.quantile(np.concatenate(distance_chunks), 0.95))
                if distance_chunks
                else 0.0
            )
            patient_metrics[patient_id][name] = {
                "dice": float(dice),
                HD95_METRIC_NAME: hd95,
                BOUNDARY_METRIC_NAME: float(boundary_dice),
                "truth_pixels": float(totals["truth_pixels"]),
            }

    per_class: dict[str, dict[str, Any]] = {}
    dice: dict[str, float] = {}
    hd95: dict[str, float] = {}
    boundary: dict[str, float] = {}
    metric_names = (
        "dice",
        HD95_METRIC_NAME,
        BOUNDARY_METRIC_NAME,
    )
    for class_offset, name in enumerate(GLOBAL_CLASSES[1:]):
        available = [values[name] for values in patient_metrics.values() if name in values]
        if not available:
            continue
        summaries: dict[str, Any] = {}
        for metric_offset, metric_name in enumerate(metric_names):
            summaries[metric_name] = _bootstrap_mean_ci(
                [item[metric_name] for item in available],
                replicates=bootstrap_replicates,
                confidence=BOOTSTRAP_CONFIDENCE,
                seed=int(bootstrap_seed) + class_offset * 101 + metric_offset,
            )
        per_class[name] = summaries
        dice[name] = summaries["dice"]["estimate"]
        hd95[name] = summaries[HD95_METRIC_NAME]["estimate"]
        boundary[name] = summaries[BOUNDARY_METRIC_NAME]["estimate"]
    return {
        "dice": dice,
        "macro_dice": float(np.mean(list(dice.values()))) if dice else 0.0,
        HD95_METRIC_NAME: hd95,
        BOUNDARY_METRIC_NAME: boundary,
        "patient_level": {
            "patient_count": len(patient_metrics),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_replicates": int(bootstrap_replicates),
            "confidence": BOOTSTRAP_CONFIDENCE,
            "per_class": per_class,
            "aggregation": "frame sufficient statistics are combined within patient before metrics",
            "common_grid_shape": list(common_grid_shape) if common_grid_shape else None,
            "hd95_units": "common letterboxed evaluation-grid pixels; no physical-spacing claim",
            "boundary_tolerance_pixels": BOUNDARY_TOLERANCE_PIXELS,
            "undefined_reference_absent_class_patient_pairs_excluded": undefined_empty_pairs,
            "false_positive_pixels_on_reference_absent_pairs": (
                undefined_reference_false_positive_pixels
            ),
        },
        "mean_uncertainty": uncertainty_sum / uncertainty_count if uncertainty_count else math.nan,
    }


def _canonical_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _runtime_identity(device: str) -> dict[str, Any]:
    import scipy
    import torch

    requested = str(device)
    use_cuda = requested.startswith("cuda")
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "nibabel": nib.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "requested_device": requested,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "module_sha256": _file_sha256(Path(__file__)),
    }
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
        device_index = torch.device(requested).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas_workspace not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "Exact deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
        try:
            driver_result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            driver_versions = sorted(
                {line.strip() for line in driver_result.stdout.splitlines() if line.strip()}
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("Could not bind CUDA driver version into run ancestry") from error
        if not driver_versions:
            raise RuntimeError("CUDA driver query returned no version")
        identity["device"] = {
            "index": int(device_index),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": int(properties.total_memory),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "amp_dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
            "driver_versions": driver_versions,
        }
    else:
        identity["device"] = {"type": "cpu", "amp_dtype": None}
    return identity


def _training_config(
    settings: TrainingSettings,
    *,
    seed: int,
    validation_domain: str,
    device: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "scope": BASELINE_SCOPE,
        "view": BASELINE_VIEW,
        "settings": asdict(settings),
        "seed": int(seed),
        "validation_domain": _normalise_domain(validation_domain),
        "runtime_identity": _runtime_identity(device),
        "transform_policy": TRANSFORM_POLICY,
        "transform_policy_fingerprint": _policy_fingerprint(TRANSFORM_POLICY),
        "loss_policy": LOSS_POLICY,
        "bootstrap": {
            "unit": "patient",
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence": BOOTSTRAP_CONFIDENCE,
            "seed": int(seed),
        },
    }


def _capture_rng_state(*, include_cuda: bool) -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if include_cuda else None,
    }


def _restore_rng_state(state: Mapping[str, Any], *, include_cuda: bool) -> None:
    import torch

    required = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
    if set(state) != required:
        raise ValueError("Resume checkpoint RNG payload is incomplete or has unknown fields")
    if include_cuda and state["torch_cuda_all"] is None:
        raise ValueError("CUDA resume requested but checkpoint has no CUDA RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if include_cuda:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_content_addressed_best(
    payload: Mapping[str, Any], output_dir: Path, epoch: int
) -> tuple[Path, str]:
    import torch

    temporary = Path(output_dir) / (
        f".best_epoch_{int(epoch):03d}.candidate.{os.getpid()}.tmp"
    )
    try:
        if temporary.exists():
            raise FileExistsError(f"Stale best-checkpoint transaction exists: {temporary}")
        torch.save(dict(payload), temporary)
        digest = _file_sha256(temporary)
        destination = Path(output_dir) / f"best_epoch_{int(epoch):03d}_{digest[:12]}.pt"
        if destination.exists():
            if _file_sha256(destination) != digest:
                raise ValueError("Content-addressed best-checkpoint collision")
            temporary.unlink()
        else:
            temporary.replace(destination)
        return destination, digest
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(path: Path, payload: Any) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_nested_finite(value: Any, label: str, path: str = "root") -> None:
    import torch

    tensors_by_device: dict[Any, list[tuple[str, Any]]] = {}

    def collect(item: Any, item_path: str) -> None:
        if isinstance(item, torch.Tensor):
            if item.is_floating_point() or item.is_complex():
                tensors_by_device.setdefault(item.device, []).append((item_path, item))
            return
        if isinstance(item, np.ndarray):
            if np.issubdtype(item.dtype, np.inexact) and not np.isfinite(item).all():
                raise FloatingPointError(f"Non-finite array in {label} at {item_path}")
            return
        if isinstance(item, (float, np.floating)):
            if not math.isfinite(float(item)):
                raise FloatingPointError(f"Non-finite scalar in {label} at {item_path}")
            return
        if isinstance(item, Mapping):
            for key, nested in item.items():
                collect(nested, f"{item_path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                collect(nested, f"{item_path}[{index}]")

    collect(value, path)
    for tensors in tensors_by_device.values():
        checks = torch.stack([torch.isfinite(tensor).all() for _path, tensor in tensors])
        if not bool(checks.all()):
            for tensor_path, tensor in tensors:
                if not bool(torch.isfinite(tensor).all()):
                    raise FloatingPointError(
                        f"Non-finite tensor in {label} at {tensor_path}"
                    )


def _validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    manifest_fingerprint: str,
    config_fingerprint: str,
) -> None:
    required = {
        "schema_version",
        "scope",
        "epoch_completed",
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
        "loader_generator_states",
        "best_validation_macro_dice",
        "bad_epochs",
        "history",
        "manifest_fingerprint",
        "config_fingerprint",
        "config",
        "transform_log_state",
        "best_artifact",
        "terminal_state",
        "sealed_test_accessed",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint is incomplete: missing {sorted(missing)}")
    if checkpoint["schema_version"] != 2 or checkpoint["scope"] != BASELINE_SCOPE:
        raise ValueError("Resume checkpoint schema or standalone A4C scope mismatch")
    if checkpoint["manifest_fingerprint"] != manifest_fingerprint:
        raise ValueError("Refusing resume: audited development manifest fingerprint changed")
    if checkpoint["config_fingerprint"] != config_fingerprint:
        raise ValueError("Refusing resume: training configuration fingerprint changed")
    if not isinstance(checkpoint["config"], Mapping) or _canonical_fingerprint(
        checkpoint["config"]
    ) != config_fingerprint:
        raise ValueError("Refusing resume: embedded training configuration is corrupt")
    if checkpoint["sealed_test_accessed"] is not False:
        raise SealedTestError(
            "Refusing resume from a checkpoint that does not attest the sealed test remained untouched"
        )
    generators = checkpoint["loader_generator_states"]
    if not isinstance(generators, Mapping) or set(generators) != {"train", "validation"}:
        raise ValueError("Resume checkpoint loader-generator state is incomplete")
    epoch = int(checkpoint["epoch_completed"])
    history = checkpoint["history"]
    if epoch < 0 or not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Resume checkpoint epoch/history state is inconsistent")
    best = float(checkpoint["best_validation_macro_dice"])
    if epoch > 0 and not math.isfinite(best):
        raise ValueError("Resume checkpoint best validation metric is not finite")
    if int(checkpoint["bad_epochs"]) < 0:
        raise ValueError("Resume checkpoint bad-epoch counter is invalid")
    terminal_state = checkpoint["terminal_state"]
    if terminal_state not in {"epoch_complete", "early_stopped", "horizon_complete"}:
        raise ValueError("Resume checkpoint terminal state is invalid")
    try:
        configured_epochs = int(checkpoint["config"]["settings"]["epochs"])
        configured_patience = int(checkpoint["config"]["settings"]["patience"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Resume checkpoint terminal configuration is incomplete") from error
    if terminal_state == "early_stopped" and int(checkpoint["bad_epochs"]) < configured_patience:
        raise ValueError("Early-stopped checkpoint has an inconsistent bad-epoch counter")
    if terminal_state == "horizon_complete" and epoch != configured_epochs:
        raise ValueError("Horizon-complete checkpoint has an inconsistent epoch")
    if terminal_state == "epoch_complete" and (
        epoch >= configured_epochs or int(checkpoint["bad_epochs"]) >= configured_patience
    ):
        raise ValueError("Non-terminal checkpoint encodes a terminal training condition")
    if not isinstance(checkpoint["transform_log_state"], Mapping):
        raise ValueError("Resume checkpoint transform-log state is not a mapping")
    _validate_transform_log_state(checkpoint["transform_log_state"])
    best_artifact = checkpoint["best_artifact"]
    if not isinstance(best_artifact, Mapping) or set(best_artifact) != {
        "filename",
        "sha256",
        "score",
    }:
        raise ValueError("Resume checkpoint best-artifact anchor is incomplete")
    if (
        Path(str(best_artifact["filename"])).name != best_artifact["filename"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(best_artifact["sha256"]))
        or not math.isclose(float(best_artifact["score"]), best, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("Resume checkpoint best-artifact anchor is invalid")
    for key in (
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
    ):
        if not isinstance(checkpoint[key], Mapping):
            raise ValueError(f"Resume checkpoint {key} is not a mapping")
    for key in (
        "model_state_dict",
        "ema_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "history",
    ):
        _assert_nested_finite(checkpoint[key], f"resume checkpoint {key}")


def _validate_transform_log_envelope(
    envelope: Mapping[str, Any],
    *,
    manifest_fingerprint: str,
    config_fingerprint: str,
) -> None:
    if set(envelope) != {
        "schema_version",
        "manifest_fingerprint",
        "config_fingerprint",
        "epoch",
        "batch",
        "sample",
    }:
        raise ValueError("Augmentation log envelope has unknown or missing fields")
    if envelope["schema_version"] != 1:
        raise ValueError("Augmentation log schema mismatch")
    if envelope["manifest_fingerprint"] != manifest_fingerprint:
        raise ValueError("Augmentation log manifest fingerprint mismatch")
    if envelope["config_fingerprint"] != config_fingerprint:
        raise ValueError("Augmentation log config fingerprint mismatch")
    if int(envelope["epoch"]) <= 0 or int(envelope["batch"]) < 0:
        raise ValueError("Augmentation log epoch/batch is invalid")
    validate_transform_record(envelope["sample"])


def _validate_transform_log_state(state: Mapping[str, Any]) -> None:
    if set(state) != {"schema_version", "line_count", "byte_count", "sha256"}:
        raise ValueError("Transform-log checkpoint state is incomplete")
    if state["schema_version"] != 1:
        raise ValueError("Transform-log checkpoint schema mismatch")
    if int(state["line_count"]) < 0 or int(state["byte_count"]) < 0:
        raise ValueError("Transform-log checkpoint counters are invalid")
    sha256 = str(state["sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Transform-log checkpoint digest is invalid")


class _TransformLogWriter:
    def __init__(
        self,
        path: Path,
        *,
        manifest_fingerprint: str,
        config_fingerprint: str,
        expected_state: Mapping[str, Any] | None,
    ) -> None:
        self.path = Path(path)
        self.manifest_fingerprint = manifest_fingerprint
        self.config_fingerprint = config_fingerprint
        self.recovered_aborted_tail: Path | None = None
        if expected_state is None:
            self.handle = self.path.open("xb")
            prefix = b""
            self.line_count = 0
            self.byte_count = 0
        else:
            _validate_transform_log_state(expected_state)
            if not self.path.is_file():
                raise ValueError("Refusing resume: augmentation JSONL evidence is missing")
            payload = self.path.read_bytes()
            expected_bytes = int(expected_state["byte_count"])
            if len(payload) < expected_bytes:
                raise ValueError("Refusing resume: augmentation log is shorter than its checkpoint anchor")
            prefix = payload[:expected_bytes]
            if hashlib.sha256(prefix).hexdigest() != expected_state["sha256"]:
                raise ValueError("Refusing resume: committed augmentation-log prefix changed")
            lines = prefix.splitlines(keepends=True)
            if len(lines) != int(expected_state["line_count"]) or any(
                not line.endswith(b"\n") for line in lines
            ):
                raise ValueError("Refusing resume: augmentation-log checkpoint counters changed")
            for line_number, line in enumerate(lines, start=1):
                try:
                    envelope = json.loads(line.decode("utf-8"))
                    _validate_transform_log_envelope(
                        envelope,
                        manifest_fingerprint=manifest_fingerprint,
                        config_fingerprint=config_fingerprint,
                    )
                except Exception as error:
                    raise ValueError(
                        f"Refusing resume: invalid committed augmentation log line {line_number}: {error}"
                    ) from error
            tail = payload[expected_bytes:]
            if tail:
                tail_digest = hashlib.sha256(tail).hexdigest()
                archive = self.path.with_name(
                    f"{self.path.stem}.aborted-{tail_digest[:12]}{self.path.suffix}"
                )
                if archive.exists():
                    if archive.read_bytes() != tail:
                        raise ValueError("Refusing resume: aborted-log archive name collision")
                else:
                    temporary_archive = archive.with_name(archive.name + ".tmp")
                    try:
                        temporary_archive.write_bytes(tail)
                        temporary_archive.replace(archive)
                    finally:
                        temporary_archive.unlink(missing_ok=True)
                temporary_prefix = self.path.with_name(self.path.name + ".reconcile.tmp")
                try:
                    temporary_prefix.write_bytes(prefix)
                    temporary_prefix.replace(self.path)
                finally:
                    temporary_prefix.unlink(missing_ok=True)
                self.recovered_aborted_tail = archive
            self.handle = self.path.open("ab")
            self.line_count = int(expected_state["line_count"])
            self.byte_count = expected_bytes
        self.digest = hashlib.sha256(prefix)

    def write_batch(
        self,
        serialized_records: Sequence[str],
        *,
        epoch: int,
        batch_index: int,
    ) -> None:
        for serialized in serialized_records:
            sample = json.loads(serialized)
            validate_transform_record(sample)
            envelope = {
                "schema_version": 1,
                "manifest_fingerprint": self.manifest_fingerprint,
                "config_fingerprint": self.config_fingerprint,
                "epoch": int(epoch),
                "batch": int(batch_index),
                "sample": sample,
            }
            _validate_transform_log_envelope(
                envelope,
                manifest_fingerprint=self.manifest_fingerprint,
                config_fingerprint=self.config_fingerprint,
            )
            encoded = (
                json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            self.handle.write(encoded)
            self.digest.update(encoded)
            self.line_count += 1
            self.byte_count += len(encoded)
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def state(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "line_count": self.line_count,
            "byte_count": self.byte_count,
            "sha256": self.digest.hexdigest(),
        }

    def close(self) -> None:
        self.handle.close()


def train(
    root: Path,
    output_dir: Path,
    *,
    mode: str,
    seed: int,
    device: str,
    validation_domain: str = "Site_R",
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    settings = settings_for_mode(mode)
    seed_everything(seed)
    manifest = build_manifest(
        root,
        validation_domain=validation_domain,
        maximum_volumes_per_split=settings.maximum_volumes_per_split,
        maximum_frames_per_volume=settings.maximum_frames_per_volume,
    )
    if not bool(manifest.development_file_inventory.get("approved")):
        raise ValueError(
            "Training is blocked: complete CardiacUDC development-file inventory is not approved "
            f"(fingerprint={manifest.development_file_inventory.get('fingerprint')})"
        )
    config = _training_config(
        settings,
        seed=seed,
        validation_domain=manifest.validation_domain,
        device=device,
    )
    config_fingerprint = _canonical_fingerprint(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    transform_manifest_path = output_dir / "transform_manifest.json"
    augmentation_log_path = output_dir / "augmentation_log.jsonl"
    history_path = output_dir / "history.json"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    transform_manifest = {
        "schema_version": 1,
        "scope": BASELINE_SCOPE,
        "manifest_fingerprint": manifest.fingerprint,
        "config_fingerprint": config_fingerprint,
        "transform_policy": TRANSFORM_POLICY,
        "transform_policy_fingerprint": _policy_fingerprint(TRANSFORM_POLICY),
        "log_format": (
            "one validated JSON object per sample use; each checkpoint anchors exact bytes/count/hash; "
            "post-anchor crash tails are content-addressed and archived before replay"
        ),
    }
    resume_payload: Mapping[str, Any] | None = None
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        resume_payload = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(resume_payload, Mapping):
            raise ValueError("Resume checkpoint payload is not a mapping")
        _validate_resume_checkpoint(
            resume_payload,
            manifest_fingerprint=manifest.fingerprint,
            config_fingerprint=config_fingerprint,
        )
        if not manifest_path.is_file() or not transform_manifest_path.is_file():
            raise ValueError("Refusing resume: run or transform manifest evidence is missing")
        recorded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if recorded_manifest.get("fingerprint") != manifest.fingerprint:
            raise ValueError("Refusing resume: on-disk run manifest fingerprint changed")
        if json.loads(transform_manifest_path.read_text(encoding="utf-8")) != transform_manifest:
            raise ValueError("Refusing resume: transform manifest changed")
    else:
        collisions = [
            path
            for path in (
                manifest_path,
                transform_manifest_path,
                augmentation_log_path,
                history_path,
                best_path,
                last_path,
            )
            if path.exists()
        ]
        collisions.extend(sorted(output_dir.glob("best_epoch_*.pt")))
        if collisions:
            raise FileExistsError(
                "Refusing to overwrite existing run evidence without an explicit resume checkpoint: "
                + ", ".join(str(path) for path in collisions)
            )
        _atomic_json_write(manifest_path, manifest.report())
        _atomic_json_write(transform_manifest_path, transform_manifest)

    train_dataset = CardiacUDCFrameDataset(
        manifest.train_frames,
        resolution=settings.resolution,
        augment=True,
        cache_size=4,
        preload=True,
    )
    validation_dataset = CardiacUDCFrameDataset(
        manifest.validation_frames,
        resolution=settings.resolution,
        augment=False,
        cache_size=2,
        preload=True,
    )
    train_generator = torch.Generator().manual_seed(seed)
    validation_generator = torch.Generator().manual_seed(seed + 1)
    loader_options = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "worker_init_fn": _worker_seed,
        # Respawning workers each epoch makes their complete random stream a
        # function of the checkpointed DataLoader generator state.
        "persistent_workers": False,
        "pin_memory": device.startswith("cuda"),
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=train_generator, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=validation_generator,
        **loader_options,
    )
    model = ViewConditionedUNet(base_channels=settings.base_channels).to(device)
    ema = ViewConditionedUNet(base_channels=settings.base_channels).to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, settings.epochs))
    use_cuda = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and amp_dtype == torch.float16)
    best = -math.inf
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    start_epoch = 0
    terminal_state = "epoch_complete"
    best_artifact: dict[str, Any] | None = None
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        ema.load_state_dict(resume_payload["ema_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload["scaler_state_dict"])
        train_generator.set_state(resume_payload["loader_generator_states"]["train"])
        validation_generator.set_state(resume_payload["loader_generator_states"]["validation"])
        best = float(resume_payload["best_validation_macro_dice"])
        bad_epochs = int(resume_payload["bad_epochs"])
        history = list(resume_payload["history"])
        start_epoch = int(resume_payload["epoch_completed"])
        terminal_state = str(resume_payload["terminal_state"])
        if start_epoch > settings.epochs:
            raise ValueError("Refusing resume: checkpoint epoch exceeds configured training horizon")
        best_artifact = dict(resume_payload["best_artifact"])
        anchored_best_path = output_dir / str(best_artifact["filename"])
        if (
            not anchored_best_path.is_file()
            or _file_sha256(anchored_best_path) != best_artifact["sha256"]
        ):
            raise ValueError("Refusing resume: immutable best-checkpoint anchor changed")
        recorded_best = torch.load(anchored_best_path, map_location="cpu", weights_only=False)
        if (
            recorded_best.get("schema_version") != 2
            or recorded_best.get("scope") != BASELINE_SCOPE
            or recorded_best.get("manifest_fingerprint") != manifest.fingerprint
            or recorded_best.get("config_fingerprint") != config_fingerprint
            or recorded_best.get("transform_policy_fingerprint")
            != _policy_fingerprint(TRANSFORM_POLICY)
            or recorded_best.get("sealed_test_accessed") is not False
        ):
            raise ValueError("Refusing resume: best inference checkpoint ancestry changed")
        _assert_nested_finite(recorded_best["state_dict"], "best inference checkpoint")
        recorded_best_score = float(recorded_best["best_validation"]["macro_dice"])
        if not math.isclose(recorded_best_score, best, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Refusing resume: best inference checkpoint score changed")
        if not best_path.is_file() or _file_sha256(best_path) != best_artifact["sha256"]:
            _atomic_copy(anchored_best_path, best_path)
        _restore_rng_state(resume_payload["rng_state"], include_cuda=use_cuda)

    transform_log_writer = _TransformLogWriter(
        augmentation_log_path,
        manifest_fingerprint=manifest.fingerprint,
        config_fingerprint=config_fingerprint,
        expected_state=(
            resume_payload["transform_log_state"] if resume_payload is not None else None
        ),
    )
    try:
        epoch_iterator = (
            range(start_epoch, settings.epochs)
            if terminal_state == "epoch_complete"
            else range(start_epoch, start_epoch)
        )
        for epoch in epoch_iterator:
            model.train()
            running = 0.0
            batches = 0
            for batch_index, batch in enumerate(train_loader):
                serialized_records = batch.get("transform_json")
                if not isinstance(serialized_records, (list, tuple)):
                    raise ValueError("Training batch lacks collated structured transform records")
                transform_log_writer.write_batch(
                    serialized_records,
                    epoch=epoch + 1,
                    batch_index=batch_index,
                )
                optimizer.zero_grad(set_to_none=True)
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["target"].to(device, non_blocking=True)
                views = batch["view_id"].to(device, non_blocking=True)
                class_valid = batch["class_valid"].to(device, non_blocking=True)
                if not bool(torch.isfinite(images).all()):
                    raise FloatingPointError("Non-finite training input")
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                    outputs = model(images, views)
                    loss, _parts = segmentation_loss(outputs, targets, class_valid)
                _assert_nested_finite(outputs, "training outputs")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite segmentation loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()), 1.0, error_if_nonfinite=True
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError("Non-finite segmentation gradient norm")
                scaler.step(optimizer)
                scaler.update()
                _assert_nested_finite(model.state_dict(), "post-optimizer model")
                _update_ema(ema, model, settings.ema_decay)
                _assert_nested_finite(ema.state_dict(), "EMA model")
                running += float(loss.detach().cpu())
                batches += 1
            scheduler.step()
            metrics = _evaluate(
                ema,
                validation_loader,
                device,
                bootstrap_seed=seed,
                bootstrap_replicates=BOOTSTRAP_REPLICATES,
            )
            row = {
                "epoch": epoch + 1,
                "train_loss": running / max(1, batches),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **metrics,
            }
            history.append(row)
            _atomic_json_write(history_path, history)
            if metrics["macro_dice"] > best:
                best = metrics["macro_dice"]
                bad_epochs = 0
                best_payload = {
                    "schema_version": 2,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": BASELINE_SCOPE,
                    "capability_statement": (
                        "Standalone single-frame A4C research baseline; no foundation or temporal claim"
                    ),
                    "state_dict": ema.state_dict(),
                    "settings": asdict(settings),
                    "seed": seed,
                    "validation_domain": manifest.validation_domain,
                    "manifest_fingerprint": manifest.fingerprint,
                    "config_fingerprint": config_fingerprint,
                    "config": config,
                    "transform_policy_fingerprint": _policy_fingerprint(TRANSFORM_POLICY),
                    "global_classes": GLOBAL_CLASSES,
                    "view_names": VIEW_NAMES,
                    "trained_views": [BASELINE_VIEW],
                    "validation_views": [BASELINE_VIEW],
                    "best_validation": metrics,
                    "metric_unit": "patient",
                    "uncertainty_status": "uncalibrated challenger signal; no coverage guarantee",
                    "calibration_audit_complete": False,
                    "sealed_test_accessed": False,
                }
                _assert_nested_finite(best_payload, "best inference payload")
                immutable_best_path, immutable_best_sha256 = _save_content_addressed_best(
                    best_payload, output_dir, epoch + 1
                )
                best_artifact = {
                    "filename": immutable_best_path.name,
                    "sha256": immutable_best_sha256,
                    "score": float(best),
                }
                _atomic_copy(immutable_best_path, best_path)
            else:
                bad_epochs += 1
            if best_artifact is None:
                raise RuntimeError("No finite best inference artifact exists after evaluation")
            if bad_epochs >= settings.patience:
                terminal_state = "early_stopped"
            elif epoch + 1 >= settings.epochs:
                terminal_state = "horizon_complete"
            else:
                terminal_state = "epoch_complete"
            _assert_nested_finite(model.state_dict(), "checkpoint model")
            _assert_nested_finite(ema.state_dict(), "checkpoint EMA")
            _assert_nested_finite(optimizer.state_dict(), "checkpoint optimizer")
            _assert_nested_finite(scheduler.state_dict(), "checkpoint scheduler")
            _assert_nested_finite(scaler.state_dict(), "checkpoint scaler")
            _atomic_torch_save(
                {
                    "schema_version": 2,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": BASELINE_SCOPE,
                    "epoch_completed": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "rng_state": _capture_rng_state(include_cuda=use_cuda),
                    "loader_generator_states": {
                        "train": train_generator.get_state(),
                        "validation": validation_generator.get_state(),
                    },
                    "best_validation_macro_dice": best,
                    "bad_epochs": bad_epochs,
                    "history": history,
                    "manifest_fingerprint": manifest.fingerprint,
                    "config_fingerprint": config_fingerprint,
                    "config": config,
                    "transform_log_state": transform_log_writer.state(),
                    "best_artifact": best_artifact,
                    "terminal_state": terminal_state,
                    "sealed_test_accessed": False,
                },
                last_path,
            )
            if terminal_state != "epoch_complete":
                break
    finally:
        transform_log_writer.close()

    if best_artifact is None:
        raise RuntimeError("Training ended without an anchored best inference checkpoint")
    anchored_checkpoint = output_dir / str(best_artifact["filename"])
    report = {
        "schema_version": 2,
        "scope": BASELINE_SCOPE,
        "capability_statement": (
            "Standalone single-frame A4C chamber segmentation research baseline; "
            "not a foundation model, temporal specialist, measurement system, or clinical device."
        ),
        "mode": mode,
        "seed": seed,
        "checkpoint": str(anchored_checkpoint),
        "checkpoint_alias": str(best_path),
        "resumable_checkpoint": str(last_path),
        "resumed_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "epochs_completed": len(history),
        "best_macro_dice": best,
        "manifest_fingerprint": manifest.fingerprint,
        "config_fingerprint": config_fingerprint,
        "validation_domain": manifest.validation_domain,
        "trained_views": [BASELINE_VIEW],
        "validation_views": [BASELINE_VIEW],
        "evaluation_unit": "patient",
        "transform_manifest": str(transform_manifest_path),
        "augmentation_log": str(augmentation_log_path),
        "transform_log_state": transform_log_writer.state(),
        "recovered_aborted_augmentation_tail": (
            str(transform_log_writer.recovered_aborted_tail)
            if transform_log_writer.recovered_aborted_tail is not None
            else None
        ),
        "terminal_state": terminal_state,
        "uncertainty_status": "uncalibrated challenger signal; no coverage guarantee",
        "calibration_audit_complete": False,
        "sealed_test_accessed": False,
        "promotion_status": "challenger_only; locked-test evaluation and multi-seed agreement required",
    }
    _atomic_json_write(output_dir / "report.json", report)
    return report


def _infer_view_from_path(path: Path) -> str:
    match = _IMAGE_PATTERN.match(path.name)
    if match is None:
        raise ValueError("Cannot infer CardiacUDC view from filename; pass --view")
    return CODE_TO_VIEW[int(match.group("view"))]


def export_checkpoint_overlay(
    checkpoint_path: Path,
    image_path: Path,
    output_path: Path,
    *,
    frame_index: int,
    view: str | None,
    device: str,
) -> Path:
    import torch

    image_path = Path(image_path)
    _assert_not_sealed_path(image_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 2 or checkpoint.get("scope") != BASELINE_SCOPE:
        raise ValueError("Checkpoint does not attest the standalone A4C baseline scope")
    if checkpoint.get("sealed_test_accessed") is not False:
        raise SealedTestError("Checkpoint does not attest that locked test pixels remained sealed")
    if checkpoint.get("transform_policy_fingerprint") != _policy_fingerprint(TRANSFORM_POLICY):
        raise ValueError("Checkpoint preprocessing policy does not match this implementation")
    checkpoint_config = checkpoint.get("config")
    if (
        not isinstance(checkpoint_config, Mapping)
        or checkpoint.get("config_fingerprint") != _canonical_fingerprint(checkpoint_config)
    ):
        raise ValueError("Checkpoint configuration ancestry is incomplete or corrupt")
    if tuple(checkpoint.get("global_classes", ())) != GLOBAL_CLASSES or tuple(
        checkpoint.get("view_names", ())
    ) != VIEW_NAMES:
        raise ValueError("Checkpoint ontology does not match this implementation")
    if not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError("Checkpoint model state is missing")
    _assert_nested_finite(checkpoint["state_dict"], "overlay checkpoint")
    settings = checkpoint["settings"]
    if (
        checkpoint_config.get("scope") != BASELINE_SCOPE
        or checkpoint_config.get("view") != BASELINE_VIEW
        or checkpoint_config.get("settings") != settings
    ):
        raise ValueError("Checkpoint settings are not bound to its configuration ancestry")
    model = ViewConditionedUNet(base_channels=int(settings["base_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    selected_view = canonical_view(view) if view else _infer_view_from_path(image_path)
    trained_views = checkpoint.get("trained_views")
    if not isinstance(trained_views, (list, tuple)) or tuple(trained_views) != (BASELINE_VIEW,):
        raise ValueError("Checkpoint does not record exact standalone A4C view support")
    if selected_view not in trained_views:
        raise ValueError(
            f"Checkpoint was not trained for {selected_view}; supported views: {', '.join(trained_views)}"
        )
    if selected_view != BASELINE_VIEW:
        raise ValueError(f"This standalone challenger only supports {BASELINE_VIEW}")
    volume = np.asanyarray(nib.load(str(image_path)).dataobj).astype(np.float32, copy=False)
    if volume.ndim != 3 or not 0 <= frame_index < volume.shape[2]:
        raise ValueError(f"Invalid frame {frame_index} for volume shape {volume.shape}")
    original = volume[:, :, frame_index]
    normalised = CardiacUDCFrameDataset._normalise_image(original)
    transformed, _dummy_target, transform = apply_spatial_transform(
        normalised,
        np.zeros_like(normalised, dtype=np.int16),
        int(settings["resolution"]),
    )
    transformed = np.rint(np.clip(transformed, 0, 1) * 255.0).astype(np.uint8).astype(
        np.float32
    ) / 255.0
    image_tensor = torch.from_numpy(transformed.copy()).unsqueeze(0).unsqueeze(0).to(device)
    view_id = torch.tensor([VIEW_TO_INDEX[selected_view]], device=device)
    class_valid = torch.from_numpy(active_class_mask(selected_view)).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = model(image_tensor, view_id)
        _assert_nested_finite(outputs, "overlay outputs")
        prediction = (
            masked_logits(outputs["logits"].float(), class_valid)
            .argmax(dim=1)
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.int16)
        )
        uncertainty = predictive_uncertainty(outputs, class_valid).squeeze(0).cpu().numpy()
        prediction = invert_spatial_transform(prediction, transform, mode="nearest")
        uncertainty = invert_spatial_transform(uncertainty, transform, mode="bilinear")
    return export_contour_overlay(original, prediction, uncertainty, output_path, view=selected_view)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone single-frame CardiacUDC A4C chamber-segmentation challenger"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.add_argument("--validation-domain", default="Site_R")
    audit.add_argument("--strict", action="store_true")
    audit.add_argument("--cardiacnet-root", type=Path)
    training = subparsers.add_parser("train")
    training.add_argument("--root", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--mode", choices=("smoke", "full"), required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--device", default="cuda:0")
    training.add_argument("--validation-domain", default="Site_R")
    training.add_argument("--resume", type=Path)
    overlay = subparsers.add_parser("overlay")
    overlay.add_argument("--checkpoint", type=Path, required=True)
    overlay.add_argument("--image", type=Path, required=True)
    overlay.add_argument("--output", type=Path, required=True)
    overlay.add_argument("--frame", type=int, required=True)
    overlay.add_argument("--view")
    overlay.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        manifest = build_manifest(args.root, validation_domain=args.validation_domain, strict=args.strict)
        report = manifest.report()
        report["cardiacnet"] = audit_optional_cardiacnet(
            args.cardiacnet_root, cardiacuda_root=args.root
        )
        rendered = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered)
        return (
            0
            if not report["invalid_volumes"]
            and report["development_file_inventory"]["approved"]
            else 2
        )
    if args.command == "train":
        report = train(
            args.root,
            args.output_dir,
            mode=args.mode,
            seed=args.seed,
            device=args.device,
            validation_domain=args.validation_domain,
            resume_checkpoint=args.resume,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.command == "overlay":
        result = export_checkpoint_overlay(
            args.checkpoint,
            args.image,
            args.output,
            frame_index=args.frame,
            view=args.view,
            device=args.device,
        )
        print(result)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
