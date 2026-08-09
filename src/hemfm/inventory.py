from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pydicom
from pydicom.valuerep import DT

from .splits import canonical_relative_dicom_path, read_normalized_record_list


# DICOM PS3.3 C.8.5.5.1.15. Keep the numeric code as well as this display
# label in the region table so that unknown/future codes never get collapsed.
UNIT_CODES = {
    0x0000: "none",
    0x0001: "percent",
    0x0002: "dB",
    0x0003: "cm",
    0x0004: "s",
    0x0005: "Hz",
    0x0006: "dB/s",
    0x0007: "cm/s",
    0x0008: "cm2",
    0x0009: "cm2/s",
    0x000A: "cm3",
    0x000B: "cm3/s",
    0x000C: "degrees",
}

REGION_SPATIAL_FORMATS = {
    0x0000: "none",
    0x0001: "2D",
    0x0002: "M-mode",
    0x0003: "spectral",
    0x0004: "waveform",
    0x0005: "graphics",
}

REGION_DATA_TYPES = {
    0x0000: "none",
    0x0001: "tissue",
    0x0002: "color_flow",
    0x0003: "PW_spectral_Doppler",
    0x0004: "CW_spectral_Doppler",
    0x0005: "Doppler_mean_trace",
    0x0006: "Doppler_mode_trace",
    0x0007: "Doppler_max_trace",
    0x0008: "volume_trace",
    0x000A: "ECG_trace",
    0x000B: "pulse_trace",
    0x000C: "phonocardiogram_trace",
    0x000D: "gray_bar",
    0x000E: "color_bar",
    0x000F: "integrated_backscatter",
    0x0010: "area_trace",
    0x0011: "d_area_dt",
    0x0012: "other_physiological_trace",
}

INVENTORY_SCHEMA_VERSION = 2
PARQUET_COMPRESSION = "zstd"
MEASUREMENT_SCREEN_LAYOUT_VERSION = "hemfm.metadata_layout.v1"
MEASUREMENT_SCREEN_LAYOUT_PROVENANCE = (
    "metadata_only_candidate_prior_not_annotation_truth"
)
INVENTORY_ENRICHED_FIELDS = [
    "image_type",
    "frames",
    "region_type",
    "region_spatial_format",
    "region_data_type",
    "physical_units_x",
    "physical_units_y",
    "physical_delta_x",
    "physical_delta_y",
    "reference_pixel",
    "region_bounds",
    "frame_time",
    "cine_rate",
    "overlay_planes",
    "graphic_annotations",
    "photometric_conversion_path",
    "vendor",
    "manufacturer",
    "model",
    "software",
    "software_versions",
    "transducer",
    "measurement_screen_layout",
    "measurement_screen_layout_version",
    "measurement_screen_layout_provenance",
]


@dataclass
class DicomRow:
    # The first fields and the legacy calibration fields intentionally retain
    # their names so all existing CSV consumers continue to work.
    record_index: int
    subject_id: str
    study_id: str
    relative_path: str
    readable: bool
    modality: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    rows: int | None = None
    columns: int | None = None
    frames: int | None = None
    photometric: str | None = None
    region_count: int = 0
    spatial_region_count: int = 0
    spectral_region_count: int = 0
    multi_region: bool = False
    has_physical_spatial_calibration: bool = False
    error: str | None = None
    sop_class_uid: str | None = None
    sop_instance_uid: str | None = None
    transfer_syntax_uid: str | None = None
    transfer_syntax_compressed: bool | None = None
    image_type: str | None = None
    image_type_has_secondary: bool = False
    is_secondary_capture_sop_class: bool = False
    conversion_type: str | None = None
    burned_in_annotation: str | None = None
    frame_time_ms: float | None = None
    frame_time_vector_count: int = 0
    frame_time_vector_median_ms: float | None = None
    cine_rate_fps: float | None = None
    recommended_display_frame_rate_fps: float | None = None
    effective_frame_rate_fps: float | None = None
    frame_increment_pointer: str | None = None
    has_overlay: bool = False
    overlay_groups: str | None = None
    has_graphic_annotation: bool = False
    photometric_conversion_path: str | None = None
    software_versions: str | None = None
    transducer_data: str | None = None
    transducer_type: str | None = None
    acquisition_datetime: str | None = None
    acquisition_datetime_source: str | None = None
    samples_per_pixel: int | None = None
    planar_configuration: int | None = None
    bits_allocated: int | None = None
    bits_stored: int | None = None
    pixel_representation: int | None = None
    measurement_screen_layout: str | None = None
    measurement_screen_layout_version: str = MEASUREMENT_SCREEN_LAYOUT_VERSION
    measurement_screen_layout_provenance: str = MEASUREMENT_SCREEN_LAYOUT_PROVENANCE
    measurement_screen_layout_metadata_only: bool = True
    measurement_screen_layout_is_annotation_truth: bool = False


@dataclass
class RegionRow:
    record_index: int
    region_index: int
    relative_path: str
    region_spatial_format_code: int | None = None
    region_spatial_format: str | None = None
    region_data_type_code: int | None = None
    region_data_type: str | None = None
    region_flags: int | None = None
    x0: int | None = None
    y0: int | None = None
    x1: int | None = None
    y1: int | None = None
    reference_pixel_x0: int | None = None
    reference_pixel_y0: int | None = None
    reference_physical_value_x: float | None = None
    reference_physical_value_y: float | None = None
    physical_units_x_code: int | None = None
    physical_units_y_code: int | None = None
    physical_units_x: str | None = None
    physical_units_y: str | None = None
    physical_delta_x: float | None = None
    physical_delta_y: float | None = None
    is_spatially_calibrated: bool = False
    is_spectral_velocity_calibrated: bool = False
    transducer_frequency_khz: float | None = None
    pulse_repetition_frequency_hz: float | None = None
    doppler_correction_angle_degrees: float | None = None
    steering_angle_degrees: float | None = None
    active_image_area_overlay_group: int | None = None


@dataclass
class DicomInspection:
    instance: DicomRow
    regions: list[RegionRow]


def _measurement_screen_layout_candidate(
    row: DicomRow, regions: list[RegionRow]
) -> str:
    """Derive a deterministic sampling prior from header metadata only."""
    purposes: set[str] = set()
    for region in regions:
        spatial_format = region.region_spatial_format_code
        data_type = region.region_data_type_code
        if spatial_format == 0x0003 or data_type in {0x0003, 0x0004}:
            purposes.add("spectral_doppler")
        elif data_type in {0x0005, 0x0006, 0x0007}:
            purposes.add("doppler_trace")
        elif spatial_format == 0x0002:
            purposes.add("m_mode")
        elif data_type == 0x0002:
            purposes.add("color_flow")
        elif spatial_format == 0x0001 or data_type == 0x0001:
            purposes.add("two_d_tissue")
        elif spatial_format == 0x0005:
            purposes.add("graphics")
        else:
            purposes.add("other_region")
    purpose = "+".join(sorted(purposes)) if purposes else "no_us_region"

    image_tokens = {
        token.strip().upper()
        for token in (row.image_type or "").split("\\")
        if token.strip()
    }
    if "SECONDARY" in image_tokens:
        capture = "secondary"
    elif "DERIVED" in image_tokens:
        capture = "derived"
    elif image_tokens:
        capture = "primary_or_original"
    else:
        capture = "image_type_unknown"
    temporal = "multiframe" if (row.frames or 0) > 1 else "single_frame"
    if row.has_overlay and row.has_graphic_annotation:
        annotation_metadata = "overlay_and_graphics"
    elif row.has_overlay:
        annotation_metadata = "overlay"
    elif row.has_graphic_annotation:
        annotation_metadata = "graphics"
    else:
        annotation_metadata = "no_vector_annotation_metadata"
    vendor_present = bool(row.manufacturer or row.model)
    software_present = bool(row.software_versions)
    if vendor_present and software_present:
        domain_metadata = "vendor_and_software"
    elif vendor_present:
        domain_metadata = "vendor_only"
    elif software_present:
        domain_metadata = "software_only"
    else:
        domain_metadata = "vendor_software_unknown"
    return (
        f"purpose={purpose};capture={capture};temporal={temporal};"
        f"annotation={annotation_metadata};domain={domain_metadata}"
    )


def _value(item: Any, default: Any = None) -> Any:
    return getattr(item, "value", default) if item is not None else default


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _nonzero_finite(value: float | None) -> bool:
    return value is not None and value != 0.0 and math.isfinite(value)


def _joined(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (str, bytes)):
        return str(value)
    try:
        return "\\".join(str(item) for item in value)
    except TypeError:
        return str(value)


def _numeric_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        candidates: Iterable[Any] = str(value).split("\\")
    else:
        try:
            candidates = list(value)
        except TypeError:
            candidates = [value]
    return [number for item in candidates if (number := _as_float(item)) is not None]


def _photometric_conversion_path(photometric: str | None) -> str | None:
    if not photometric:
        return None
    value = photometric.upper().strip()
    paths = {
        "MONOCHROME1": "MONOCHROME1->invert->grayscale",
        "MONOCHROME2": "MONOCHROME2->grayscale",
        "PALETTE COLOR": "PALETTE_COLOR->palette_LUT->RGB",
        "RGB": "RGB->RGB",
        "ARGB": "ARGB->drop_alpha->RGB",
        "CMYK": "CMYK->RGB",
        "HSV": "HSV->RGB",
        "YBR_FULL": "YBR_FULL->RGB",
        "YBR_FULL_422": "YBR_FULL_422->upsample_422->RGB",
        "YBR_PARTIAL_422": "YBR_PARTIAL_422->upsample_422->RGB",
        "YBR_PARTIAL_420": "YBR_PARTIAL_420->upsample_420->RGB",
        "YBR_ICT": "YBR_ICT->inverse_ICT->RGB",
        "YBR_RCT": "YBR_RCT->inverse_RCT->RGB",
    }
    return paths.get(value, f"UNSUPPORTED:{value}")


def _acquisition_datetime(dataset: pydicom.Dataset) -> tuple[str | None, str | None]:
    candidates = [
        ("AcquisitionDateTime", dataset.get("AcquisitionDateTime")),
        (
            "AcquisitionDate+AcquisitionTime",
            f"{dataset.get('AcquisitionDate', '')}{dataset.get('AcquisitionTime', '')}",
        ),
        ("ContentDate+ContentTime", f"{dataset.get('ContentDate', '')}{dataset.get('ContentTime', '')}"),
        ("SeriesDate+SeriesTime", f"{dataset.get('SeriesDate', '')}{dataset.get('SeriesTime', '')}"),
        ("StudyDate+StudyTime", f"{dataset.get('StudyDate', '')}{dataset.get('StudyTime', '')}"),
    ]
    for source, raw in candidates:
        text = str(raw or "").strip()
        if len(text) < 8:
            continue
        try:
            return DT(text).isoformat(), source
        except (TypeError, ValueError):
            # Preserve a present but non-conforming value for audit; downstream
            # temporal logic must inspect the source and parseability explicitly.
            return text, f"{source}:unparsed"
    return None, None


def _overlay_metadata(dataset: pydicom.Dataset) -> tuple[bool, str | None, bool]:
    groups = sorted(
        {
            int(tag.group)
            for tag in dataset.keys()
            if 0x6000 <= int(tag.group) <= 0x601E and int(tag.group) % 2 == 0
        }
    )
    has_graphics = bool(
        dataset.get("GraphicAnnotationSequence")
        or dataset.get("GraphicLayerSequence")
        or dataset.get("GraphicObjectSequence")
        or dataset.get("TextObjectSequence")
    )
    return bool(groups), ",".join(f"{group:04X}" for group in groups) or None, has_graphics


def _is_secondary_capture_sop_class(uid: str | None) -> bool:
    # Secondary Capture plus the single-bit/gray/color multi-frame SC family.
    return bool(uid and (uid == "1.2.840.10008.5.1.4.1.1.7" or uid.startswith("1.2.840.10008.5.1.4.1.1.7.")))


def _extract_regions(
    dataset: pydicom.Dataset, record_index: int, relative_path: str
) -> list[RegionRow]:
    output: list[RegionRow] = []
    sequence = getattr(dataset, "SequenceOfUltrasoundRegions", []) or []
    for region_index, region in enumerate(sequence):
        spatial_code = _as_int(getattr(region, "RegionSpatialFormat", None))
        data_code = _as_int(getattr(region, "RegionDataType", None))
        x_code = _as_int(getattr(region, "PhysicalUnitsXDirection", None))
        y_code = _as_int(getattr(region, "PhysicalUnitsYDirection", None))
        x_unit = UNIT_CODES.get(x_code)
        y_unit = UNIT_CODES.get(y_code)
        dx = _as_float(getattr(region, "PhysicalDeltaX", None))
        dy = _as_float(getattr(region, "PhysicalDeltaY", None))
        spatial = x_unit == "cm" and y_unit == "cm" and _nonzero_finite(dx) and _nonzero_finite(dy)
        spectral = (
            (x_unit == "cm/s" and _nonzero_finite(dx))
            or (y_unit == "cm/s" and _nonzero_finite(dy))
        )
        output.append(
            RegionRow(
                record_index=record_index,
                region_index=region_index,
                relative_path=relative_path,
                region_spatial_format_code=spatial_code,
                region_spatial_format=REGION_SPATIAL_FORMATS.get(spatial_code),
                region_data_type_code=data_code,
                region_data_type=REGION_DATA_TYPES.get(data_code),
                region_flags=_as_int(getattr(region, "RegionFlags", None)),
                x0=_as_int(getattr(region, "RegionLocationMinX0", None)),
                y0=_as_int(getattr(region, "RegionLocationMinY0", None)),
                x1=_as_int(getattr(region, "RegionLocationMaxX1", None)),
                y1=_as_int(getattr(region, "RegionLocationMaxY1", None)),
                reference_pixel_x0=_as_int(getattr(region, "ReferencePixelX0", None)),
                reference_pixel_y0=_as_int(getattr(region, "ReferencePixelY0", None)),
                reference_physical_value_x=_as_float(
                    getattr(region, "ReferencePixelPhysicalValueX", None)
                ),
                reference_physical_value_y=_as_float(
                    getattr(region, "ReferencePixelPhysicalValueY", None)
                ),
                physical_units_x_code=x_code,
                physical_units_y_code=y_code,
                physical_units_x=x_unit,
                physical_units_y=y_unit,
                physical_delta_x=dx,
                physical_delta_y=dy,
                is_spatially_calibrated=spatial,
                is_spectral_velocity_calibrated=spectral,
                transducer_frequency_khz=_as_float(getattr(region, "TransducerFrequency", None)),
                pulse_repetition_frequency_hz=_as_float(
                    getattr(region, "PulseRepetitionFrequency", None)
                ),
                doppler_correction_angle_degrees=_as_float(
                    getattr(region, "DopplerCorrectionAngle", None)
                ),
                steering_angle_degrees=_as_float(getattr(region, "SteeringAngle", None)),
                active_image_area_overlay_group=_as_int(
                    getattr(region, "ActiveImageAreaOverlayGroup", None)
                ),
            )
        )
    return output


def inspect_record_detailed(
    record: tuple[int, str, str, str], dicom_root: Path
) -> DicomInspection:
    index, subject_id, study_id, relative_path = record
    # Validate even when this low-level helper is called directly so a crafted
    # record cannot escape the configured corpus root.
    canonical_relative_dicom_path(relative_path)
    path = dicom_root / relative_path.replace("/", "\\")
    row = DicomRow(index, str(subject_id), str(study_id), relative_path, False)
    regions: list[RegionRow] = []
    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        regions = _extract_regions(dataset, index, relative_path)
        row.readable = True
        row.modality = str(dataset.get("Modality", "")) or None
        row.manufacturer = str(dataset.get("Manufacturer", "")) or None
        row.model = str(dataset.get("ManufacturerModelName", "")) or None
        row.rows = _as_int(dataset.get("Rows"))
        row.columns = _as_int(dataset.get("Columns"))
        row.frames = _as_int(dataset.get("NumberOfFrames")) or 1
        row.photometric = str(dataset.get("PhotometricInterpretation", "")) or None
        row.region_count = len(regions)
        row.spatial_region_count = sum(item.is_spatially_calibrated for item in regions)
        row.spectral_region_count = sum(item.is_spectral_velocity_calibrated for item in regions)
        row.multi_region = len(regions) > 1
        row.has_physical_spatial_calibration = row.spatial_region_count > 0

        row.sop_class_uid = str(dataset.get("SOPClassUID", "")) or None
        row.sop_instance_uid = str(dataset.get("SOPInstanceUID", "")) or None
        transfer_syntax = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
        row.transfer_syntax_uid = str(transfer_syntax) if transfer_syntax else None
        row.transfer_syntax_compressed = (
            bool(transfer_syntax.is_compressed) if transfer_syntax is not None else None
        )
        row.image_type = _joined(dataset.get("ImageType"))
        row.image_type_has_secondary = "SECONDARY" in {
            item.upper() for item in (row.image_type or "").split("\\") if item
        }
        row.is_secondary_capture_sop_class = _is_secondary_capture_sop_class(row.sop_class_uid)
        row.conversion_type = str(dataset.get("ConversionType", "")) or None
        row.burned_in_annotation = str(dataset.get("BurnedInAnnotation", "")) or None

        row.frame_time_ms = _as_float(dataset.get("FrameTime"))
        frame_times = _numeric_vector(dataset.get("FrameTimeVector"))
        row.frame_time_vector_count = len(frame_times)
        if frame_times:
            ordered = sorted(frame_times)
            midpoint = len(ordered) // 2
            row.frame_time_vector_median_ms = (
                ordered[midpoint]
                if len(ordered) % 2
                else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
            )
        row.cine_rate_fps = _as_float(dataset.get("CineRate"))
        row.recommended_display_frame_rate_fps = _as_float(
            dataset.get("RecommendedDisplayFrameRate")
        )
        if row.cine_rate_fps and row.cine_rate_fps > 0:
            row.effective_frame_rate_fps = row.cine_rate_fps
        elif row.frame_time_ms and row.frame_time_ms > 0:
            row.effective_frame_rate_fps = 1000.0 / row.frame_time_ms
        elif row.frame_time_vector_median_ms and row.frame_time_vector_median_ms > 0:
            row.effective_frame_rate_fps = 1000.0 / row.frame_time_vector_median_ms
        row.frame_increment_pointer = _joined(dataset.get("FrameIncrementPointer"))

        row.has_overlay, row.overlay_groups, row.has_graphic_annotation = _overlay_metadata(dataset)
        row.photometric_conversion_path = _photometric_conversion_path(row.photometric)
        row.software_versions = _joined(dataset.get("SoftwareVersions"))
        row.transducer_data = _joined(dataset.get("TransducerData"))
        row.transducer_type = _joined(dataset.get("TransducerType"))
        row.acquisition_datetime, row.acquisition_datetime_source = _acquisition_datetime(dataset)
        row.samples_per_pixel = _as_int(dataset.get("SamplesPerPixel"))
        row.planar_configuration = _as_int(dataset.get("PlanarConfiguration"))
        row.bits_allocated = _as_int(dataset.get("BitsAllocated"))
        row.bits_stored = _as_int(dataset.get("BitsStored"))
        row.pixel_representation = _as_int(dataset.get("PixelRepresentation"))
        row.measurement_screen_layout = _measurement_screen_layout_candidate(row, regions)
    except Exception as exc:  # preserve failures rather than abort a corpus scan
        row.error = f"{type(exc).__name__}: {str(exc)[:240]}"
    return DicomInspection(instance=row, regions=regions)


def inspect_record(record: tuple[int, str, str, str], dicom_root: Path) -> DicomRow:
    """Backward-compatible instance-only inspection helper."""
    return inspect_record_detailed(record, dicom_root).instance


def _read_records(
    path: Path,
) -> tuple[list[tuple[int, str, str, str]], dict[str, Any]]:
    frame, normalization = read_normalized_record_list(path)
    records = [
        (int(position), str(row.subject_id), str(row.study_id), str(row.dicom_filepath))
        for position, row in enumerate(frame.itertuples(index=False))
    ]
    return records, normalization


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_fingerprint() -> str:
    payload = {
        "instance": list(DicomRow.__dataclass_fields__),
        "region": list(RegionRow.__dataclass_fields__),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _derived_inventory_paths(output_csv: Path) -> dict[str, Path]:
    return {
        "instance_parquet": output_csv.with_name(f"{output_csv.stem}_instances.parquet"),
        "region_parquet": output_csv.with_name(f"{output_csv.stem}_regions.parquet"),
        "checkpoint": output_csv.with_name(f".{output_csv.stem}.checkpoint.json"),
        "journal": output_csv.with_name(f".{output_csv.stem}.journal.jsonl"),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _expected_checkpoint(
    *,
    record_list: Path,
    dicom_root: Path,
    output_csv: Path,
    record_count: int,
    scan_target: int,
    source_sha256: str,
    source_normalization: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "record_list": str(record_list.resolve()),
        "record_list_sha256": source_sha256,
        "record_count": record_count,
        "scan_target": scan_target,
        "dicom_root": str(dicom_root.resolve()),
        "output_csv": str(output_csv.resolve()),
        "schema_sha256": _schema_fingerprint(),
        "source_normalization": source_normalization,
    }


def _validate_checkpoint(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "Refusing unsafe inventory resume: checkpoint/source mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _read_journal(path: Path, source: list[tuple[int, str, str, str]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    instance_fields = set(DicomRow.__dataclass_fields__)
    region_fields = set(RegionRow.__dataclass_fields__)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"Refusing unsafe resume: blank journal line {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Refusing unsafe resume: corrupt journal line {line_number}: {exc}"
                ) from exc
            if line_number > len(source):
                raise RuntimeError("Refusing unsafe resume: journal is longer than the record list")
            expected_index, expected_subject, expected_study, expected_path = source[line_number - 1]
            instance = payload.get("instance", {})
            regions = payload.get("regions")
            if (
                set(payload) != {"instance", "regions"}
                or not isinstance(instance, dict)
                or set(instance) != instance_fields
                or not isinstance(regions, list)
            ):
                raise RuntimeError(
                    f"Refusing unsafe resume: journal schema mismatch at row {line_number - 1}"
                )
            identity = (
                _as_int(instance.get("record_index")),
                str(instance.get("subject_id", "")),
                str(instance.get("study_id", "")),
                str(instance.get("relative_path", "")),
            )
            if identity != (expected_index, expected_subject, expected_study, expected_path):
                raise RuntimeError(
                    f"Refusing unsafe resume: journal/source prefix mismatch at row {line_number - 1}"
                )
            for region_position, region in enumerate(regions):
                if not isinstance(region, dict) or set(region) != region_fields:
                    raise RuntimeError(
                        f"Refusing unsafe resume: region journal schema mismatch at row {line_number - 1}"
                    )
                region_identity = (
                    _as_int(region.get("record_index")),
                    _as_int(region.get("region_index")),
                    str(region.get("relative_path", "")),
                )
                if region_identity != (expected_index, region_position, expected_path):
                    raise RuntimeError(
                        f"Refusing unsafe resume: region/source mismatch at row {line_number - 1}"
                    )
            entries.append(payload)
    return entries


def _validate_or_repair_csv_from_journal(
    output_path: Path,
    journal_entries: list[dict[str, Any]],
    source: list[tuple[int, str, str, str]],
) -> None:
    fieldnames = list(DicomRow.__dataclass_fields__)
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fieldnames:
            raise RuntimeError("Refusing unsafe resume: legacy CSV schema does not match this inventory version")
        csv_rows = list(reader)
    if len(csv_rows) > len(journal_entries):
        raise RuntimeError("Refusing unsafe resume: CSV is ahead of the transaction journal")
    rebuild_from_journal = len(csv_rows) < len(journal_entries)
    for position, row in enumerate(csv_rows):
        expected = source[position]
        identity = (
            _as_int(row.get("record_index")),
            str(row.get("subject_id", "")),
            str(row.get("study_id", "")),
            str(row.get("relative_path", "")),
        )
        if identity != expected:
            rebuild_from_journal = True
        # Identity equality alone is insufficient: a crash while writing a CSV
        # row may persist the first four columns but truncate or corrupt later
        # metadata. The durable journal is the transaction source of truth, so
        # every already-visible CSV value must match its canonical rendering.
        expected_row = {
            name: "" if journal_entries[position]["instance"].get(name) is None
            else str(journal_entries[position]["instance"].get(name))
            for name in fieldnames
        }
        if set(row) != set(expected_row) or row != expected_row:
            rebuild_from_journal = True
    if rebuild_from_journal:
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.repair.tmp")
        # A process crash may leave this PID-scoped temporary behind; replacing
        # that exact repair temporary is safe because no live process can share
        # the current PID.
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for payload in journal_entries:
                writer.writerow(payload["instance"])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)


def _arrow_schemas() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("DICOM inventory Parquet output requires the pyarrow dependency") from exc

    instance_integers = {
        "record_index", "rows", "columns", "frames", "frame_time_vector_count",
        "samples_per_pixel", "planar_configuration", "bits_allocated", "bits_stored",
        "pixel_representation",
    }
    instance_floats = {
        "frame_time_ms", "frame_time_vector_median_ms", "cine_rate_fps",
        "recommended_display_frame_rate_fps", "effective_frame_rate_fps",
    }
    instance_booleans = {
        "readable", "multi_region", "has_physical_spatial_calibration",
        "transfer_syntax_compressed", "image_type_has_secondary",
        "is_secondary_capture_sop_class", "has_overlay", "has_graphic_annotation",
        "measurement_screen_layout_metadata_only",
        "measurement_screen_layout_is_annotation_truth",
    }
    # Region counts are integers despite being adjacent to the legacy flags.
    instance_integers.update({"region_count", "spatial_region_count", "spectral_region_count"})
    instance_fields = []
    for name in DicomRow.__dataclass_fields__:
        if name in instance_integers:
            data_type = pa.int64()
        elif name in instance_floats:
            data_type = pa.float64()
        elif name in instance_booleans:
            data_type = pa.bool_()
        else:
            data_type = pa.string()
        instance_fields.append(pa.field(name, data_type, nullable=True))

    region_integers = {
        "record_index", "region_index", "region_spatial_format_code",
        "region_data_type_code", "region_flags", "x0", "y0", "x1", "y1",
        "reference_pixel_x0", "reference_pixel_y0", "physical_units_x_code",
        "physical_units_y_code", "active_image_area_overlay_group",
    }
    region_booleans = {"is_spatially_calibrated", "is_spectral_velocity_calibrated"}
    region_fields = []
    for name in RegionRow.__dataclass_fields__:
        if name in region_integers:
            data_type = pa.int64()
        elif name in region_booleans:
            data_type = pa.bool_()
        elif name in {"relative_path", "region_spatial_format", "region_data_type", "physical_units_x", "physical_units_y"}:
            data_type = pa.string()
        else:
            data_type = pa.float64()
        region_fields.append(pa.field(name, data_type, nullable=True))
    return pa.schema(instance_fields), pa.schema(region_fields)


def _materialize_parquets(
    entries: Iterable[dict[str, Any]], instance_path: Path, region_path: Path
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    instance_schema, region_schema = _arrow_schemas()
    instance_temp = instance_path.with_name(f".{instance_path.name}.{os.getpid()}.tmp")
    region_temp = region_path.with_name(f".{region_path.name}.{os.getpid()}.tmp")
    instance_writer = pq.ParquetWriter(instance_temp, instance_schema, compression=PARQUET_COMPRESSION)
    region_writer = pq.ParquetWriter(region_temp, region_schema, compression=PARQUET_COMPRESSION)
    try:
        instance_batch: list[dict[str, Any]] = []
        region_batch: list[dict[str, Any]] = []
        for payload in entries:
            instance_batch.append(payload["instance"])
            region_batch.extend(payload.get("regions", []))
            if len(instance_batch) >= 10_000:
                instance_writer.write_table(pa.Table.from_pylist(instance_batch, schema=instance_schema))
                instance_batch.clear()
            if len(region_batch) >= 25_000:
                region_writer.write_table(pa.Table.from_pylist(region_batch, schema=region_schema))
                region_batch.clear()
        if instance_batch:
            instance_writer.write_table(pa.Table.from_pylist(instance_batch, schema=instance_schema))
        if region_batch:
            region_writer.write_table(pa.Table.from_pylist(region_batch, schema=region_schema))
    finally:
        instance_writer.close()
        region_writer.close()
    os.replace(instance_temp, instance_path)
    os.replace(region_temp, region_path)


def _inventory_summary_from_csv(output_path: Path) -> dict[str, Any]:
    total = readable = calibrated = multi_region = spectral = errors = 0
    manufacturers: Counter[str] = Counter()
    measurement_screen_layouts: Counter[str] = Counter()
    sop_owners: dict[str, str] = {}
    duplicate_sop_uids: set[str] = set()
    cross_patient_duplicate_sop_uids: set[str] = set()
    missing_sop_uids = 0
    columns = [
        "readable", "has_physical_spatial_calibration", "multi_region",
        "spectral_region_count", "manufacturer", "error", "subject_id",
        "sop_instance_uid", "measurement_screen_layout",
    ]
    for frame in pd.read_csv(
        output_path,
        usecols=columns,
        dtype={"subject_id": str, "sop_instance_uid": str},
        chunksize=50_000,
        low_memory=False,
    ):
        total += len(frame)
        readable += int(frame["readable"].fillna(False).astype(bool).sum())
        calibrated += int(frame["has_physical_spatial_calibration"].fillna(False).astype(bool).sum())
        multi_region += int(frame["multi_region"].fillna(False).astype(bool).sum())
        spectral += int((frame["spectral_region_count"].fillna(0) > 0).sum())
        errors += int(frame["error"].notna().sum())
        manufacturers.update(frame["manufacturer"].fillna("UNKNOWN").astype(str).tolist())
        measurement_screen_layouts.update(
            frame["measurement_screen_layout"].fillna("UNAVAILABLE").astype(str).tolist()
        )
        for subject_id, uid in zip(frame["subject_id"], frame["sop_instance_uid"]):
            if pd.isna(uid) or not str(uid).strip():
                missing_sop_uids += 1
                continue
            uid_text = str(uid).strip()
            subject_text = str(subject_id).strip()
            if uid_text in sop_owners:
                duplicate_sop_uids.add(uid_text)
                if sop_owners[uid_text] != subject_text:
                    cross_patient_duplicate_sop_uids.add(uid_text)
            else:
                sop_owners[uid_text] = subject_text
    return {
        "records": total,
        "readable": readable,
        "spatially_calibrated": calibrated,
        "multi_region": multi_region,
        "spectral_regions": spectral,
        "manufacturers": dict(manufacturers.most_common(50)),
        "measurement_screen_layout_version": MEASUREMENT_SCREEN_LAYOUT_VERSION,
        "measurement_screen_layout_provenance": MEASUREMENT_SCREEN_LAYOUT_PROVENANCE,
        "measurement_screen_layout_is_annotation_truth": False,
        "measurement_screen_layouts": dict(
            measurement_screen_layouts.most_common(100)
        ),
        "errors": errors,
        "missing_sop_instance_uids": missing_sop_uids,
        "duplicate_sop_instance_uids": len(duplicate_sop_uids),
        "cross_patient_duplicate_sop_instance_uids": len(
            cross_patient_duplicate_sop_uids
        ),
    }


def run_dicom_inventory(
    record_list: str | Path,
    dicom_root: str | Path,
    output_csv: str | Path,
    summary_json: str | Path,
    workers: int = 16,
    limit: int | None = None,
    resume: bool = False,
    expected_record_list_sha256: str | None = None,
) -> dict[str, Any]:
    """Inventory DICOM headers with resumable CSV/journal transactions.

    The CSV remains the compatibility surface. The two Zstandard-compressed
    Parquet artifacts are authoritative for new code. A resume is accepted only
    when its source bytes, schema, root, target length, journal and CSV prefix all
    match; otherwise the function refuses to guess or silently rescan.
    """
    if workers < 1:
        raise ValueError("workers must be at least one")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")

    record_path = Path(record_list)
    root_path = Path(dicom_root)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = _derived_inventory_paths(output_path)
    source_sha256 = _sha256_file(record_path)
    if expected_record_list_sha256 and source_sha256 != expected_record_list_sha256:
        raise RuntimeError("Record-list bytes do not match the immutable split-manifest source hash")
    records, source_normalization = _read_records(record_path)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Configured DICOM root is unavailable or not a directory: {root_path}")
    scan_target = min(len(records), limit if limit is not None else len(records))
    expected_checkpoint = _expected_checkpoint(
        record_list=record_path,
        dicom_root=root_path,
        output_csv=output_path,
        record_count=len(records),
        scan_target=scan_target,
        source_sha256=source_sha256,
        source_normalization=source_normalization,
    )
    artifact_set = [output_path, paths["checkpoint"], paths["journal"]]

    if resume:
        missing = [str(path) for path in artifact_set if not path.exists()]
        if missing:
            raise RuntimeError(
                "Refusing unsafe inventory resume: required transaction artifacts are missing: "
                + ", ".join(missing)
            )
        checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
        _validate_checkpoint(checkpoint, expected_checkpoint)
        entries = _read_journal(paths["journal"], records)
        if len(entries) > scan_target:
            raise RuntimeError("Refusing unsafe resume: journal exceeds the configured scan target")
        _validate_or_repair_csv_from_journal(output_path, entries, records)
    else:
        existing = [str(path) for path in [*artifact_set, paths["instance_parquet"], paths["region_parquet"]] if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite an existing inventory; use --resume only with a matching v2 checkpoint: "
                + ", ".join(existing)
            )
        _write_json_atomic(paths["checkpoint"], expected_checkpoint)
        with output_path.open("x", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(DicomRow.__dataclass_fields__)).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        paths["journal"].touch(exist_ok=False)
        entries = []

    completed = len(entries)
    # Replace any stale prior evidence before decoding starts. If the process is
    # interrupted, the durable public-facing state remains explicitly incomplete
    # rather than leaving an older green summary in place.
    _write_json_atomic(
        Path(summary_json),
        {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "complete": False,
            "scan_target_complete": False,
            "records_completed": completed,
            "record_list_records": len(records),
            "record_list_source_rows": source_normalization["source_rows"],
            "source_normalization": source_normalization,
            "scan_target": scan_target,
            "record_list_sha256": source_sha256,
            "schema_sha256": _schema_fingerprint(),
            "enriched_fields": INVENTORY_ENRICHED_FIELDS,
            "metadata_only": True,
            "pixel_data_read": False,
            "locked_test_pixels_accessed": False,
        },
    )
    remaining = records[completed:scan_target]
    fieldnames = list(DicomRow.__dataclass_fields__)
    if remaining:
        with (
            paths["journal"].open("a", encoding="utf-8", newline="\n") as journal,
            output_path.open("a", encoding="utf-8", newline="") as csv_handle,
        ):
            writer = csv.DictWriter(csv_handle, fieldnames=fieldnames)
            transaction_batch: list[dict[str, Any]] = []

            def commit_batch() -> None:
                if not transaction_batch:
                    return
                # Commit the complete journal batch durably before exposing any
                # of the matching compatibility CSV rows. A crash can therefore
                # leave only journal-ahead state, which resume can replay safely.
                for item in transaction_batch:
                    journal.write(json.dumps(item, separators=(",", ":")) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                for item in transaction_batch:
                    writer.writerow(item["instance"])
                csv_handle.flush()
                os.fsync(csv_handle.fileno())
                entries.extend(transaction_batch)
                transaction_batch.clear()

            with ThreadPoolExecutor(max_workers=workers) as executor:
                iterator = executor.map(
                    lambda record: inspect_record_detailed(record, root_path),
                    remaining,
                    chunksize=32,
                )
                for offset, inspection in enumerate(iterator, start=1):
                    payload = {
                        "instance": asdict(inspection.instance),
                        "regions": [asdict(region) for region in inspection.regions],
                    }
                    transaction_batch.append(payload)
                    if offset % 256 == 0:
                        commit_batch()
            commit_batch()

    if _sha256_file(record_path) != source_sha256:
        raise RuntimeError("Record list changed during inventory; refusing to finalize derived artifacts")
    entries = _read_journal(paths["journal"], records)
    _validate_or_repair_csv_from_journal(output_path, entries, records)
    if len(entries) != scan_target:
        raise RuntimeError("Inventory stopped before its configured scan target")
    _materialize_parquets(entries, paths["instance_parquet"], paths["region_parquet"])

    counts = _inventory_summary_from_csv(output_path)
    summary = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if scan_target == len(records) else "limited_complete",
        "complete": scan_target == len(records) and counts["records"] == len(records),
        "scan_target_complete": counts["records"] == scan_target,
        "record_list_records": len(records),
        "record_list_source_rows": source_normalization["source_rows"],
        "source_normalization": source_normalization,
        "record_list_sha256": source_sha256,
        "schema_sha256": _schema_fingerprint(),
        "enriched_fields": INVENTORY_ENRICHED_FIELDS,
        "metadata_only": True,
        "pixel_data_read": False,
        "locked_test_pixels_accessed": False,
        **counts,
        "legacy_csv": str(output_path),
        "legacy_csv_sha256": _sha256_file(output_path),
        "instance_parquet": str(paths["instance_parquet"]),
        "instance_parquet_sha256": _sha256_file(paths["instance_parquet"]),
        "region_parquet": str(paths["region_parquet"]),
        "region_parquet_sha256": _sha256_file(paths["region_parquet"]),
        "resume_checkpoint": str(paths["checkpoint"]),
    }
    _write_json_atomic(Path(summary_json), summary)
    return summary


def run_label_inventory(measurements_path: str | Path, study_list_path: str | Path, targets: dict[str, Any]) -> dict:
    wanted = {name for target in targets.values() for name in target["measurements"]}
    measurements = pd.read_csv(
        measurements_path,
        compression="gzip",
        usecols=["subject_id", "measurement_id", "measurement", "result", "unit"],
        dtype=str,
        low_memory=False,
    )
    measurements = measurements[measurements["measurement"].isin(wanted)].copy()
    measurements["numeric_result"] = pd.to_numeric(measurements["result"], errors="coerce")
    studies = pd.read_csv(study_list_path, dtype={"subject_id": str, "study_id": str, "measurement_id": str})
    measurements["subject_id"] = measurements["subject_id"].astype(str)
    measurements["measurement_id"] = measurements["measurement_id"].astype(str)
    linked = measurements.merge(studies[["subject_id", "study_id", "measurement_id"]], on=["subject_id", "measurement_id"], how="left")
    output: dict[str, Any] = {"schema_version": 1, "targets": {}}
    for target_name, target in targets.items():
        subset = linked[linked["measurement"].isin(target["measurements"]) & linked["numeric_result"].notna()]
        linked_subset = subset[subset["study_id"].notna()]
        output["targets"][target_name] = {
            "numeric_rows": int(len(subset)),
            "patients_total_measurement_table": int(subset["subject_id"].nunique()),
            "linked_numeric_rows": int(len(linked_subset)),
            "linked_patients": int(linked_subset["subject_id"].nunique()),
            "linked_studies": int(linked_subset["study_id"].nunique()),
            "source_measurements": subset["measurement"].value_counts().astype(int).to_dict(),
            "reported_units": subset["unit"].fillna("UNKNOWN").value_counts().astype(int).to_dict(),
        }
    return output
