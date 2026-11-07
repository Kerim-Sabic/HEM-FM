from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class StudyDicom:
    path: Path
    patient_id: str
    study_uid: str
    sop_uid: str
    frames: int

    @property
    def reference(self) -> str:
        value = self.sop_uid or str(self.path.resolve())
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def discover_single_study(root: Path) -> tuple[list[StudyDicom], list[dict[str, str]]]:
    import pydicom

    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    records: list[StudyDicom] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except Exception as exc:
            skipped.append({"file": path.name, "reason": f"not_dicom: {type(exc).__name__}"})
            continue
        if str(getattr(dataset, "Modality", "")).upper() != "US":
            skipped.append({"file": path.name, "reason": "not_ultrasound"})
            continue
        frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
        if frames < 2:
            skipped.append({"file": path.name, "reason": "not_multiframe_cine"})
            continue
        patient_id = str(getattr(dataset, "PatientID", "")).strip()
        study_uid = str(getattr(dataset, "StudyInstanceUID", "")).strip()
        if not patient_id or not study_uid:
            raise ValueError(f"Ultrasound cine lacks PatientID or StudyInstanceUID: {path.name}")
        records.append(
            StudyDicom(
                path=path,
                patient_id=patient_id,
                study_uid=study_uid,
                sop_uid=str(getattr(dataset, "SOPInstanceUID", "")).strip(),
                frames=frames,
            )
        )
    if not records:
        raise ValueError("No multi-frame ultrasound DICOM cines were found")
    if len({record.patient_id for record in records}) != 1:
        raise ValueError("Study folder contains more than one DICOM PatientID")
    if len({record.study_uid for record in records}) != 1:
        raise ValueError("Study folder contains more than one StudyInstanceUID")
    return records, skipped


def study_reference(records: Sequence[StudyDicom]) -> str:
    if not records:
        raise ValueError("At least one study record is required")
    identity = f"{records[0].patient_id}\0{records[0].study_uid}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def temporal_probability_filter(probability: np.ndarray, window: int = 3) -> np.ndarray:
    from scipy.ndimage import median_filter

    values = np.asarray(probability, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("Probability must have shape TxCxHxW with at least two classes")
    if window < 1 or window % 2 == 0:
        raise ValueError("Temporal filter window must be a positive odd number")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Probability contains invalid values")
    filtered = median_filter(values, size=(window, 1, 1, 1), mode="nearest")
    denominator = filtered.sum(axis=1, keepdims=True)
    if (denominator <= 0).any():
        raise ValueError("Probability has a zero-sum pixel")
    return (filtered / denominator).astype(np.float32)


def temporally_tracked_masks(probability: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_dilation, label

    values = np.asarray(probability, dtype=np.float32)
    raw = values.argmax(axis=1)
    frames, classes, height, width = values.shape
    output = np.zeros((frames, height, width), dtype=np.uint8)
    previous = [None] * classes
    for frame_index in range(frames):
        selected = np.zeros((classes, height, width), dtype=bool)
        selected[0] = True
        for class_index in range(1, classes):
            components, count = label(raw[frame_index] == class_index)
            if count == 0:
                previous[class_index] = None
                continue
            candidates = [components == index for index in range(1, count + 1)]
            prior = previous[class_index]
            if prior is None:
                chosen = max(candidates, key=lambda value: int(value.sum()))
            else:
                expanded = binary_dilation(prior, iterations=2)
                chosen = max(
                    candidates,
                    key=lambda value: (
                        int((value & expanded).sum()),
                        int(value.sum()),
                    ),
                )
            selected[class_index] = chosen
            previous[class_index] = chosen
        restricted = np.where(selected, values[frame_index], -1.0)
        output[frame_index] = restricted.argmax(axis=0).astype(np.uint8)
    return output


def contour_traces(mask: np.ndarray, classes: Sequence[str]) -> list[dict[str, Any]]:
    from scipy.ndimage import binary_erosion, label

    values = np.asarray(mask)
    if values.ndim != 3:
        raise ValueError("Mask must have shape TxHxW")
    traces = []
    for frame_index, frame in enumerate(values):
        frame_traces: dict[str, list[list[list[float]]]] = {}
        for class_index, name in enumerate(classes):
            if class_index == 0:
                continue
            components, count = label(frame == class_index)
            contours = []
            for component_index in range(1, count + 1):
                component = components == component_index
                boundary = component & ~binary_erosion(component)
                points = np.argwhere(boundary)
                if not len(points):
                    continue
                center = points.mean(axis=0)
                angles = np.arctan2(points[:, 0] - center[0], points[:, 1] - center[1])
                ordered = points[np.argsort(angles)]
                contours.append(
                    [[round(float(column), 2), round(float(row), 2)] for row, column in ordered]
                )
            frame_traces[str(name)] = sorted(contours, key=len, reverse=True)
        traces.append({"frame_index": frame_index, "chambers": frame_traces})
    return traces


def temporal_quality(
    probability: np.ndarray, mask: np.ndarray, uncertainty: np.ndarray, classes: Sequence[str]
) -> dict[str, Any]:
    values = np.asarray(probability)
    labels = np.asarray(mask)
    uncertainty_values = np.asarray(uncertainty)
    per_frame = []
    consecutive_iou: dict[str, list[float]] = {str(name): [] for name in classes[1:]}
    for frame_index, frame in enumerate(labels):
        chamber_values = {}
        for class_index, name in enumerate(classes[1:], start=1):
            region = frame == class_index
            chamber_values[str(name)] = {
                "pixels": int(region.sum()),
                "mean_probability": float(values[frame_index, class_index][region].mean())
                if region.any()
                else None,
                "mean_uncertainty": float(uncertainty_values[frame_index][region].mean())
                if region.any()
                else None,
            }
            if frame_index:
                previous = labels[frame_index - 1] == class_index
                union = region | previous
                if union.any():
                    consecutive_iou[str(name)].append(float((region & previous).sum() / union.sum()))
        per_frame.append({"frame_index": frame_index, "chambers": chamber_values})
    return {
        "per_frame": per_frame,
        "mean_consecutive_iou": {
            name: float(np.mean(items)) if items else None
            for name, items in consecutive_iou.items()
        },
    }


def supported_measurement_routes(
    classifications: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    accepted = [
        item
        for item in classifications
        if item.get("accepted_by_development_confidence_policy") is True
    ]
    best: dict[str, Mapping[str, Any]] = {}
    for item in accepted:
        view = str(item["predicted_view"])
        if view not in best or float(item["confidence"]) > float(best[view]["confidence"]):
            best[view] = item
    routes: dict[str, list[dict[str, str]]] = {}
    if "A4C" in best and "A2C" in best:
        biplane = [
            {"path": str(best["A4C"]["path"]), "view": "A4C"},
            {"path": str(best["A2C"]["path"]), "view": "A2C"},
        ]
        for target in ("EF", "LVEDV", "LVESV"):
            routes[target] = list(biplane)
    if "A4C" in best:
        routes["RV_BASAL_DIAMETER"] = [
            {"path": str(best["A4C"]["path"]), "view": "A4C"}
        ]
    if "PLAX" in best:
        routes["LVOT_DIAMETER"] = [
            {"path": str(best["PLAX"]["path"]), "view": "Parasternal_Long"}
        ]
    return routes
