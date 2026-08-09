from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .calibration import LENGTH_UNITS_TO_MM, VELOCITY_UNITS_TO_MPS, RegionCalibration
from .hashing import private_hash


VOCABULARY_SCHEMA_VERSION = 1
PILOT_SCHEMA_VERSION = 2
VOCABULARY_VERSION = "hemfm-caliper-v1.0.0"
SAMPLE_HASH_SCHEME = "hmac-sha256-v1"
ARTIFACT_HASH_SCHEME = "sha256"
REVIEWER_ID_SCHEME = "hmac-sha256-v1"
SELECTION_SCHEME = "deterministic-hmac-round-robin-v1"
GEOMETRY_VERIFIER_VERSION = "hemfm-caliper-geometry-v1"

_RAW_IDENTIFIER_COLUMNS = {
    "subject_id",
    "study_id",
    "relative_path",
    "dicom_filepath",
    "patient_id",
    "mrn",
    "reviewer_name",
    "reviewer_email",
    "accession_number",
    "patient_name",
    "filepath",
    "file_path",
}
_TEST_SPLITS = {"test", "internal_test", "locked_test", "external", "external_test"}
_DEVELOPMENT_SPLITS = {"train", "validation", "development"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VocabularyEntry:
    key: str
    aliases: tuple[str, ...]
    kind: Literal["length", "velocity"]
    canonical_unit: Literal["mm", "m/s"]
    accepted_units: tuple[str, ...]
    plausible_min: float
    plausible_max: float
    audit_abs_error_max: float


@dataclass(frozen=True)
class MeasurementVocabulary:
    version: str
    entries: tuple[VocabularyEntry, ...]

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("Measurement vocabulary must have a version")
        keys = [entry.key for entry in self.entries]
        if len(keys) != len(set(keys)):
            raise ValueError("Measurement vocabulary contains duplicate canonical keys")
        aliases = [alias.casefold() for entry in self.entries for alias in entry.aliases]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Measurement vocabulary contains ambiguous aliases")

    def entry(self, key: str) -> VocabularyEntry:
        for entry in self.entries:
            if entry.key == key:
                return entry
        raise KeyError(f"Unknown canonical measurement key: {key}")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VOCABULARY_SCHEMA_VERSION,
            "vocabulary_version": self.version,
            "entries": [asdict(entry) for entry in self.entries],
        }


DEFAULT_VOCABULARY = MeasurementVocabulary(
    version=VOCABULARY_VERSION,
    entries=(
        VocabularyEntry(
            "LV_END_DIASTOLIC_DIAMETER",
            ("LVIDd", "LVIDd 2D", "LVEDD", "LV EDD"),
            "length",
            "mm",
            ("mm", "cm"),
            15.0,
            90.0,
            1.0,
        ),
        VocabularyEntry(
            "LV_END_SYSTOLIC_DIAMETER",
            ("LVIDs", "LVIDs 2D", "LVESD", "LV ESD"),
            "length",
            "mm",
            ("mm", "cm"),
            8.0,
            75.0,
            1.0,
        ),
        VocabularyEntry(
            "LVOT_DIAMETER",
            ("LVOT Diam", "LVOT Diameter", "LVOTd"),
            "length",
            "mm",
            ("mm", "cm"),
            8.0,
            40.0,
            0.8,
        ),
        VocabularyEntry(
            "RV_BASAL_DIAMETER",
            ("RVD1", "RV Basal", "RV Base"),
            "length",
            "mm",
            ("mm", "cm"),
            10.0,
            80.0,
            1.2,
        ),
        VocabularyEntry(
            "LA_DIAMETER",
            ("LA Diam", "LA Diameter", "LAD"),
            "length",
            "mm",
            ("mm", "cm"),
            10.0,
            90.0,
            1.0,
        ),
        VocabularyEntry(
            "AV_PEAK_VELOCITY",
            ("AV Vmax", "Ao Vmax", "Aortic Vmax", "AV Max Vel"),
            "velocity",
            "m/s",
            ("m/s", "cm/s", "mm/s"),
            0.2,
            8.0,
            0.05,
        ),
    ),
)


@dataclass(frozen=True)
class NormalizedMeasurement:
    key: str
    value: float
    unit: str
    source_value: float
    source_unit: str
    vocabulary_version: str


@dataclass(frozen=True)
class GlyphEndpoints:
    sample_hash: str
    measurement_key: str
    annotated_frame_hash: str
    parent_cine_hash: str
    region_id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    glyph_family: str
    endpoint_confidence: float
    line_association_score: float
    glyph_bank_version: str
    sample_hash_scheme: str = SAMPLE_HASH_SCHEME
    artifact_hash_scheme: str = ARTIFACT_HASH_SCHEME

    def __post_init__(self) -> None:
        coordinates = (self.start_x, self.start_y, self.end_x, self.end_y)
        if not all(math.isfinite(float(value)) for value in coordinates):
            raise ValueError("Glyph endpoints must be finite")
        if (
            not self.annotated_frame_hash
            or not self.parent_cine_hash
            or not self.region_id
            or not self.glyph_family
            or not self.glyph_bank_version
        ):
            raise ValueError("Glyph evidence requires annotated-frame, parent-cine, region, and glyph-bank identifiers")
        for label, value in (
            ("region_id", self.region_id),
            ("glyph_family", self.glyph_family),
            ("glyph_bank_version", self.glyph_bank_version),
        ):
            text = str(value).strip()
            if "\\" in text or "/" in text or re.match(r"^[A-Za-z]:", text):
                raise ValueError(f"{label} must be a sanitized token, not a path")
        if self.sample_hash_scheme != SAMPLE_HASH_SCHEME:
            raise ValueError(f"Sample hashes must declare {SAMPLE_HASH_SCHEME}")
        if self.artifact_hash_scheme != ARTIFACT_HASH_SCHEME:
            raise ValueError(f"Frame hashes must declare {ARTIFACT_HASH_SCHEME}")
        for label, value in (
            ("sample_hash", self.sample_hash),
            ("annotated_frame_hash", self.annotated_frame_hash),
            ("parent_cine_hash", self.parent_cine_hash),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{label} must be a 64-character SHA-256 digest")
        for label, value in (
            ("endpoint_confidence", self.endpoint_confidence),
            ("line_association_score", self.line_association_score),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{label} must be finite and lie in [0, 1]")


@dataclass(frozen=True)
class CalibratedRegion:
    region_id: str
    purpose: Literal["spatial_bmode", "spectral_doppler"]
    calibration: RegionCalibration


@dataclass(frozen=True)
class PilotThresholds:
    p1_min_instances: int = 500_000
    p2_count: int = 200
    p2_min_measurement_types: int = 2
    p2_min_glyph_families: int = 2
    p3_count: int = 2_000
    p3_yield_min: float = 0.10
    p4_evidence_coverage_min: float = 1.0
    p4_clean_recovery_min: float = 0.70
    p5_count: int = 200
    p5_precision_min: float = 0.98
    reviewer_agreement_min: float = 0.95
    active_group_min_population: int = 20
    min_audits_per_active_group: int = 5
    group_precision_min: float = 0.95
    group_p95_normalized_error_max: float = 1.0
    physical_relative_tolerance: float = 0.03
    length_abs_tolerance_mm: float = 0.8
    velocity_abs_tolerance_mps: float = 0.05
    min_endpoint_confidence: float = 0.90
    min_line_association_score: float = 0.85
    min_tissue_match_score: float = 0.98
    clean_probe_min_pairs: int = 50
    clean_probe_max_relative_gap: float = 0.10
    p2_selection_seed: int = 202604
    p3_selection_seed: int = 202605
    p5_selection_seed: int = 202606

    def __post_init__(self) -> None:
        counts = (
            self.p1_min_instances,
            self.p2_count,
            self.p2_min_measurement_types,
            self.p2_min_glyph_families,
            self.p3_count,
            self.p5_count,
            self.clean_probe_min_pairs,
        )
        if any(value < 1 for value in counts):
            raise ValueError("Pilot counts must be positive")
        probabilities = (
            self.p3_yield_min,
            self.p4_evidence_coverage_min,
            self.p4_clean_recovery_min,
            self.p5_precision_min,
            self.reviewer_agreement_min,
            self.group_precision_min,
            self.min_endpoint_confidence,
            self.min_line_association_score,
            self.min_tissue_match_score,
            self.clean_probe_max_relative_gap,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("Rate and confidence thresholds must lie in [0, 1]")
        if self.group_p95_normalized_error_max <= 0:
            raise ValueError("Normalized error threshold must be positive")
        if any(seed < 1 for seed in (self.p2_selection_seed, self.p3_selection_seed, self.p5_selection_seed)):
            raise ValueError("Selection seeds must be positive")


def _is_sha256(value: object) -> bool:
    return bool(_SHA256_RE.fullmatch(str(value).strip().casefold()))


def _sha256_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold().map(_is_sha256)


def _normalise_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


_RAW_IDENTIFIER_COLUMN_KEYS = {_normalise_column_name(value) for value in _RAW_IDENTIFIER_COLUMNS}


def _forbidden_identifier_columns(frame: pd.DataFrame, *, allow_reviewer_id: bool = False) -> set[str]:
    forbidden = set(_RAW_IDENTIFIER_COLUMN_KEYS)
    if not allow_reviewer_id:
        forbidden.add(_normalise_column_name("reviewer_id"))
    return {str(column) for column in frame.columns if _normalise_column_name(column) in forbidden}


def _strict_bool_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    truthy = {"1", "true", "yes"}
    falsy = {"0", "false", "no"}

    def parse(value: object) -> tuple[bool, bool]:
        if isinstance(value, (bool, np.bool_)):
            return bool(value), True
        if pd.isna(value):
            return False, False
        text = str(value).strip().casefold()
        if text in truthy:
            return True, True
        if text in falsy:
            return False, True
        return False, False

    parsed = series.map(parse)
    values = parsed.map(lambda item: item[0]).astype(bool)
    valid = parsed.map(lambda item: item[1]).astype(bool)
    return values, valid


def _partition_contract(frame: pd.DataFrame) -> tuple[bool, pd.Series, str | None]:
    """Validate an explicit development/test marker; absence or ambiguity is unsafe."""

    locked = pd.Series(False, index=frame.index, dtype=bool)
    evidence_present = False
    valid = pd.Series(True, index=frame.index, dtype=bool)
    errors: list[str] = []
    if "split" in frame:
        evidence_present = True
        values = frame["split"].fillna("").astype(str).str.strip().str.casefold()
        allowed = _DEVELOPMENT_SPLITS | _TEST_SPLITS
        split_valid = values.isin(allowed)
        valid &= split_valid
        locked |= values.isin(_TEST_SPLITS)
        if not bool(split_valid.all()):
            errors.append("split contains empty or unrecognized values")
    for column in ("test_locked", "sealed_internal_test"):
        if column in frame:
            evidence_present = True
            parsed, parsed_valid = _strict_bool_series(frame[column])
            valid &= parsed_valid
            locked |= parsed
            if not bool(parsed_valid.all()):
                errors.append(f"{column} contains null or ambiguous values")
    if not evidence_present:
        return False, locked, "missing split/test-lock evidence"
    return bool(valid.all()), locked, "; ".join(errors) if errors else None


def _safe_json_mapping(value: object) -> bool:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(decoded, Mapping):
        return False

    def safe(node: Any) -> bool:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if _normalise_column_name(key) in _RAW_IDENTIFIER_COLUMN_KEYS:
                    return False
                if not safe(child):
                    return False
            return True
        if isinstance(node, (list, tuple)):
            return all(safe(child) for child in node)
        if isinstance(node, str):
            text = node.strip()
            return not (
                "\\" in text
                or bool(re.match(r"^[A-Za-z]:", text))
                or text.casefold().startswith(("file:", "dicom:"))
                or text.startswith(("/", "./", "../"))
                or bool(re.search(r"/(?:[^/]+/)+[^/]+$", text))
                or text.casefold().endswith((".dcm", ".dicom", ".nii", ".nii.gz"))
            )
        if isinstance(node, float):
            return math.isfinite(node)
        return node is None or isinstance(node, (bool, int))

    return safe(decoded)


def _looks_like_filesystem_path(value: object) -> bool:
    text = str(value).strip()
    lower = text.casefold()
    return bool(
        re.match(r"^[A-Za-z]:[/\\]", text)
        or text.startswith(("\\\\", "/", "./", "../"))
        or lower.startswith("file:")
        or lower.endswith((".dcm", ".dicom", ".nii", ".nii.gz", ".csv", ".parquet", ".json"))
    )


def _selection_manifest_digest(
    frame: pd.DataFrame,
    *,
    stage: str,
    seed: int,
    strata: Sequence[str],
) -> str:
    ordered = frame.sort_values("selection_rank")
    rows = []
    for row in ordered[["sample_hash", *strata]].itertuples(index=False, name=None):
        rows.append([str(value) for value in row])
    payload = {
        "schema": SELECTION_SCHEME,
        "stage": stage,
        "seed": int(seed),
        "strata": list(strata),
        "rows": rows,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _attach_selection_provenance(
    frame: pd.DataFrame,
    *,
    stage: str,
    seed: int,
    strata: Sequence[str],
) -> pd.DataFrame:
    result = frame.copy()
    result["selection_stage"] = stage
    result["selection_scheme"] = SELECTION_SCHEME
    result["selection_seed"] = int(seed)
    digest = _selection_manifest_digest(result, stage=stage, seed=seed, strata=strata)
    result["selection_manifest_sha256"] = digest
    return result


def _selection_provenance_valid(
    frame: pd.DataFrame,
    *,
    stage: str,
    seed: int,
    strata: Sequence[str],
) -> bool:
    required = {
        "selection_rank",
        "selection_stage",
        "selection_scheme",
        "selection_seed",
        "selection_manifest_sha256",
        "sample_hash",
        *strata,
    }
    if not required.issubset(frame.columns) or len(frame) == 0:
        return False
    ranks = pd.to_numeric(frame["selection_rank"], errors="coerce")
    if set(ranks.dropna().astype(int)) != set(range(1, len(frame) + 1)) or ranks.isna().any():
        return False
    if not frame["selection_stage"].astype(str).eq(stage).all():
        return False
    if not frame["selection_scheme"].astype(str).eq(SELECTION_SCHEME).all():
        return False
    if not pd.to_numeric(frame["selection_seed"], errors="coerce").eq(seed).all():
        return False
    supplied = frame["selection_manifest_sha256"].fillna("").astype(str).str.casefold()
    if supplied.nunique() != 1 or not supplied.map(_is_sha256).all():
        return False
    expected = _selection_manifest_digest(frame, stage=stage, seed=seed, strata=strata)
    return bool(supplied.eq(expected).all())


def _normalise_unit_text(unit: str) -> str:
    text = str(unit).strip().casefold()
    text = text.replace("−", "-").replace("⁻", "-").replace("·", "/")
    text = re.sub(r"\s+", "", text)
    aliases = {
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",
        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",
        "m/sec": "m/s",
        "m/s-1": "m/s",
        "ms-1": "m/s",
        "cm/sec": "cm/s",
        "cm/s-1": "cm/s",
        "cms-1": "cm/s",
        "mm/sec": "mm/s",
        "mm/s-1": "mm/s",
        "mms-1": "mm/s",
    }
    return aliases.get(text, text)


def _to_canonical(value: float, unit: str, entry: VocabularyEntry) -> float:
    unit = _normalise_unit_text(unit)
    if unit not in entry.accepted_units:
        raise ValueError(f"Unit {unit!r} is not allowed for {entry.key}")
    if entry.kind == "length":
        factor = LENGTH_UNITS_TO_MM.get(unit)
    else:
        factor = VELOCITY_UNITS_TO_MPS.get(unit)
    if factor is None:
        raise ValueError(f"No canonical conversion is defined for {unit!r}")
    result = float(value) * factor
    if not math.isfinite(result):
        raise ValueError("Measurement value must be finite")
    if not entry.plausible_min <= result <= entry.plausible_max:
        raise ValueError(
            f"Canonical value {result:g} {entry.canonical_unit} is outside the predeclared range for {entry.key}"
        )
    return result


def parse_ocr_measurements(
    text: str,
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> list[NormalizedMeasurement]:
    """Parse explicit key/value/unit triples; missing or ambiguous units are never inferred."""

    cleaned = str(text).replace("\u00a0", " ").replace("\u2212", "-")
    cleaned = re.sub(r"(?<=\d),(?=\d)", ".", cleaned)
    unit_pattern = r"(?:mm\s*/\s*s|cm\s*/\s*s|m\s*/\s*s|mm\s*/?\s*sec|cm\s*/?\s*sec|m\s*/?\s*sec|mm|cm)"
    matches: list[tuple[int, NormalizedMeasurement]] = []
    for entry in vocabulary.entries:
        aliases = sorted(entry.aliases, key=len, reverse=True)
        alias_pattern = "|".join(re.escape(alias).replace(r"\ ", r"\s+") for alias in aliases)
        pattern = re.compile(
            rf"(?<![A-Za-z0-9])(?:{alias_pattern})(?![A-Za-z0-9])\s*(?:[:=\-]\s*)?"
            rf"(?P<value>[+]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?P<unit>{unit_pattern})(?![A-Za-z])",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(cleaned):
            source_value = float(match.group("value"))
            source_unit = _normalise_unit_text(match.group("unit"))
            canonical = _to_canonical(source_value, source_unit, entry)
            matches.append(
                (
                    match.start(),
                    NormalizedMeasurement(
                        key=entry.key,
                        value=canonical,
                        unit=entry.canonical_unit,
                        source_value=source_value,
                        source_unit=source_unit,
                        vocabulary_version=vocabulary.version,
                    ),
                )
            )
    matches.sort(key=lambda item: item[0])
    return [measurement for _, measurement in matches]


def parse_ocr_measurement(
    text: str,
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> NormalizedMeasurement:
    measurements = parse_ocr_measurements(text, vocabulary)
    if len(measurements) != 1:
        raise ValueError(f"Expected one unambiguous key/value/unit triple, found {len(measurements)}")
    return measurements[0]


def _is_test_row(frame: pd.DataFrame) -> pd.Series:
    _, locked, _ = _partition_contract(frame)
    return locked


def _first_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> pd.Series:
    for candidate in candidates:
        if candidate in frame:
            return frame[candidate]
    raise KeyError(f"Enriched inventory is missing {label}; expected one of {list(candidates)}")


def _truthy_layout(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() not in {"", "0", "false", "no", "none", "unknown", "not_detected"}


def _region_label(frame: pd.DataFrame) -> pd.Series:
    for column in ("region_types", "region_type"):
        if column in frame:
            return frame[column].fillna("UNKNOWN").astype(str)
    required = {"spatial_region_count", "spectral_region_count"}
    if not required.issubset(frame.columns):
        raise KeyError("Enriched inventory needs region_types or spatial/spectral region counts")
    spatial = pd.to_numeric(frame["spatial_region_count"], errors="coerce").fillna(0).astype(int)
    spectral = pd.to_numeric(frame["spectral_region_count"], errors="coerce").fillna(0).astype(int)
    labels = np.select(
        [
            (spatial > 0) & (spectral > 0),
            spatial > 0,
            spectral > 0,
        ],
        ["multi_spatial_spectral", "spatial", "spectral"],
        default="uncalibrated",
    )
    return pd.Series(labels, index=frame.index, dtype=str)


def _sample_hashes(frame: pd.DataFrame, privacy_secret: str) -> pd.Series:
    if len(privacy_secret) < 16:
        raise ValueError("A private HMAC secret of at least 16 characters is required")
    if "sample_hash" in frame:
        values = frame["sample_hash"].fillna("").astype(str)
        return values.map(lambda value: private_hash(f"upstream-sample-hash|{value}", privacy_secret))
    raw_columns = [column for column in ("subject_id", "study_id", "relative_path", "dicom_filepath") if column in frame]
    if not raw_columns:
        raise KeyError("Inventory needs sample_hash or private source identifiers")
    return frame[raw_columns].fillna("").astype(str).agg("|".join, axis=1).map(
        lambda value: private_hash(f"private-source-identifiers|{value}", privacy_secret)
    )


def _prepare_candidates(inventory: pd.DataFrame, privacy_secret: str) -> pd.DataFrame:
    frame = inventory.copy()
    partition_valid, locked, partition_error = _partition_contract(frame)
    if not partition_valid:
        raise ValueError(f"Locked-test exclusion failed closed: {partition_error}")
    image_type = _first_column(frame, ("image_type", "ImageType"), "ImageType")
    frames = _first_column(frame, ("frames", "number_of_frames", "NumberOfFrames"), "NumberOfFrames")
    vendor = _first_column(frame, ("vendor", "manufacturer", "Manufacturer"), "vendor")
    software = _first_column(
        frame,
        ("software", "software_version", "software_versions", "SoftwareVersions"),
        "software version",
    )
    layout = _first_column(
        frame,
        ("measurement_screen_layout", "known_measurement_layout", "layout_family"),
        "known measurement-screen layout",
    )
    numeric_frames = pd.to_numeric(frames, errors="coerce")
    readable = _safe_bool(frame["readable"]) if "readable" in frame else pd.Series(True, index=frame.index)
    eligible = (
        ~locked
        & readable
        & numeric_frames.ge(1)
        & image_type.fillna("").astype(str).str.strip().ne("")
        & vendor.fillna("").astype(str).str.strip().ne("")
        & software.fillna("").astype(str).str.strip().ne("")
        & layout.map(_truthy_layout)
    )
    frame = frame.loc[eligible].copy()
    numeric_frames = numeric_frames.loc[eligible].astype(int)
    safe = pd.DataFrame(index=frame.index)
    safe["sample_hash"] = _sample_hashes(frame, privacy_secret)
    safe["sample_hash_scheme"] = SAMPLE_HASH_SCHEME
    if "record_index" in frame:
        safe["inventory_record_index"] = pd.to_numeric(frame["record_index"], errors="coerce").astype("Int64")
    else:
        safe["inventory_record_index"] = frame.index.astype(int)
    safe["split"] = frame["split"].astype(str) if "split" in frame else "development"
    safe["image_type"] = image_type.loc[eligible].fillna("UNKNOWN").astype(str)
    safe["frames"] = numeric_frames
    safe["frame_bucket"] = pd.cut(
        numeric_frames,
        bins=[0, 1, 15, 63, np.inf],
        labels=["single", "2-15", "16-63", "64+"],
    ).astype(str)
    safe["region_type"] = _region_label(frame).fillna("UNKNOWN").astype(str)
    safe["vendor"] = vendor.loc[eligible].fillna("UNKNOWN").astype(str)
    safe["software"] = software.loc[eligible].fillna("UNKNOWN").astype(str)
    safe["layout_family"] = layout.loc[eligible].astype(str)
    safe["test_locked"] = False
    return safe.drop_duplicates("sample_hash").reset_index(drop=True)


def _round_robin_stratified(
    frame: pd.DataFrame,
    target: int,
    seed: int,
    strata: Sequence[str] = ("image_type", "frame_bucket", "region_type", "vendor", "software", "layout_family"),
) -> pd.DataFrame:
    strata = list(strata)
    if len(frame) < target:
        raise ValueError(f"Only {len(frame)} eligible development candidates are available; {target} are required")
    missing = [column for column in strata if column not in frame]
    if missing:
        raise KeyError(f"Stratified selection is missing columns: {missing}")
    work = frame.copy()
    work["_tie"] = work["sample_hash"].map(lambda value: private_hash(f"{seed}|{value}", str(seed) * 8))
    groups: list[list[int]] = []
    for _, group in work.sort_values("_tie").groupby(strata, sort=True, dropna=False):
        groups.append(group.index.tolist())
    groups.sort(key=lambda indices: private_hash(f"{seed}|{work.loc[indices[0], '_tie']}", str(seed) * 8))
    selected: list[int] = []
    depth = 0
    while len(selected) < target:
        added = False
        for indices in groups:
            if depth < len(indices):
                selected.append(indices[depth])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        depth += 1
    result = work.loc[selected].drop(columns="_tie").reset_index(drop=True)
    result["selection_rank"] = np.arange(1, len(result) + 1)
    return result


def select_p2_candidates(
    inventory: pd.DataFrame,
    privacy_secret: str,
    *,
    target: int = 200,
    seed: int = 202604,
) -> pd.DataFrame:
    """Create the private P2 locator manifest without exposing source identifiers."""

    strata = ["image_type", "frame_bucket", "region_type", "vendor", "software", "layout_family"]
    result = _round_robin_stratified(_prepare_candidates(inventory, privacy_secret), target, seed)
    result = _attach_selection_provenance(result, stage="P2", seed=seed, strata=strata)
    result["manual_review_complete"] = False
    result["annotation_rendering_reviewed"] = False
    result["annotation_present"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["measurement_key"] = ""
    result["glyph_family"] = ""
    return result


def build_p3_manifest(
    inventory: pd.DataFrame,
    privacy_secret: str,
    *,
    target: int = 2_000,
    seed: int = 202605,
) -> pd.DataFrame:
    strata = ["image_type", "frame_bucket", "region_type", "vendor", "software", "layout_family"]
    result = _round_robin_stratified(_prepare_candidates(inventory, privacy_secret), target, seed)
    result = _attach_selection_provenance(result, stage="P3", seed=seed, strata=strata)
    result["vocabulary_version"] = VOCABULARY_VERSION
    result["ocr_parse_valid"] = False
    result["glyph_localized"] = False
    result["physical_verified"] = False
    result["annotated_frame_trainable"] = False
    result["accepted"] = False
    return result


def verify_glyph_geometry(
    glyph: GlyphEndpoints,
    ocr_measurement: NormalizedMeasurement,
    study_value: float,
    study_unit: str,
    regions: Iterable[CalibratedRegion],
    *,
    thresholds: PilotThresholds = PilotThresholds(),
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> dict[str, Any]:
    """Verify a glyph only through its explicitly linked ultrasound region."""

    reasons: list[str] = []
    ocr_valid = True
    if not math.isfinite(float(ocr_measurement.value)):
        reasons.append("OCR value must be finite")
        ocr_valid = False
    if glyph.measurement_key != ocr_measurement.key:
        reasons.append("glyph/OCR measurement-key mismatch")
    if ocr_measurement.vocabulary_version != vocabulary.version:
        reasons.append("OCR vocabulary version mismatch")
        ocr_valid = False
    if glyph.endpoint_confidence < thresholds.min_endpoint_confidence:
        reasons.append("endpoint confidence below threshold")
    if glyph.line_association_score < thresholds.min_line_association_score:
        reasons.append("line/ridge association below threshold")
    try:
        entry = vocabulary.entry(glyph.measurement_key)
    except KeyError as exc:
        reasons.append(str(exc))
        entry = None
        ocr_valid = False
    if entry is not None and ocr_measurement.unit != entry.canonical_unit:
        reasons.append("OCR canonical unit does not match the versioned vocabulary")
        ocr_valid = False
    matching = [region for region in regions if region.region_id == glyph.region_id]
    if len(matching) != 1:
        reasons.append("glyph must resolve to exactly one ultrasound region")
        region = None
    else:
        region = matching[0]
        if not region.calibration.contains(glyph.start_x, glyph.start_y) or not region.calibration.contains(
            glyph.end_x, glyph.end_y
        ):
            reasons.append("both glyph endpoints must lie inside the linked ultrasound region")

    physical_value: float | None = None
    canonical_study: float | None = None
    if entry is not None and region is not None and not reasons:
        calibration = region.calibration
        try:
            if entry.kind == "length":
                if region.purpose != "spatial_bmode":
                    raise ValueError("length glyph is not linked to a spatial B-mode region")
                start = calibration.image_to_physical(glyph.start_x, glyph.start_y)
                end = calibration.image_to_physical(glyph.end_x, glyph.end_y)
                physical_value = float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
            else:
                if region.purpose != "spectral_doppler":
                    raise ValueError("velocity glyph is not linked to a spectral-Doppler region")
                axes = [
                    (calibration.x_unit, calibration.delta_x, glyph.start_x, glyph.end_x),
                    (calibration.y_unit, calibration.delta_y, glyph.start_y, glyph.end_y),
                ]
                velocity_axes = [axis for axis in axes if axis[0] in VELOCITY_UNITS_TO_MPS]
                if len(velocity_axes) != 1:
                    raise ValueError("spectral region must define exactly one velocity axis")
                unit, delta, start_pixel, end_pixel = velocity_axes[0]
                physical_value = abs((end_pixel - start_pixel) * delta * VELOCITY_UNITS_TO_MPS[unit])
            canonical_study = _to_canonical(float(study_value), study_unit, entry)
        except (TypeError, ValueError) as exc:
            reasons.append(str(exc))

    absolute_tolerance: float | None = None
    ocr_error: float | None = None
    study_error: float | None = None
    if entry is not None and physical_value is not None and canonical_study is not None:
        absolute_tolerance = (
            thresholds.length_abs_tolerance_mm if entry.kind == "length" else thresholds.velocity_abs_tolerance_mps
        )
        ocr_error = abs(physical_value - ocr_measurement.value)
        study_error = abs(physical_value - canonical_study)
        ocr_limit = max(absolute_tolerance, thresholds.physical_relative_tolerance * abs(ocr_measurement.value))
        study_limit = max(absolute_tolerance, thresholds.physical_relative_tolerance * abs(canonical_study))
        if ocr_error > ocr_limit:
            reasons.append("pixel geometry disagrees with OCR beyond tolerance")
        if study_error > study_limit:
            reasons.append("pixel geometry disagrees with study measurement beyond tolerance")
    geometry_json = json.dumps(
        {
            "start_x": float(glyph.start_x),
            "start_y": float(glyph.start_y),
            "end_x": float(glyph.end_x),
            "end_y": float(glyph.end_y),
            "region_id": glyph.region_id,
            "glyph_family": glyph.glyph_family,
            "calibration": asdict(region.calibration) if region is not None else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": PILOT_SCHEMA_VERSION,
        "sample_hash": glyph.sample_hash,
        "sample_hash_scheme": glyph.sample_hash_scheme,
        "annotated_frame_hash": glyph.annotated_frame_hash,
        "parent_cine_hash": glyph.parent_cine_hash,
        "artifact_hash_scheme": glyph.artifact_hash_scheme,
        "measurement_key": glyph.measurement_key,
        "vocabulary_version": vocabulary.version,
        "glyph_bank_version": glyph.glyph_bank_version,
        "region_id": glyph.region_id,
        "region_purpose": region.purpose if region is not None else None,
        "geometry_verifier_version": GEOMETRY_VERIFIER_VERSION,
        "geometry_json": geometry_json,
        "endpoint_confidence": float(glyph.endpoint_confidence),
        "line_association_score": float(glyph.line_association_score),
        "physical_value": physical_value,
        "canonical_unit": entry.canonical_unit if entry is not None else None,
        "ocr_value": ocr_measurement.value,
        "study_value": canonical_study,
        "ocr_absolute_error": ocr_error,
        "study_absolute_error": study_error,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": thresholds.physical_relative_tolerance,
        "accepted": not reasons,
        "ocr_parse_valid": ocr_valid,
        "glyph_localized": bool(
            math.isfinite(float(glyph.endpoint_confidence))
            and math.isfinite(float(glyph.line_association_score))
            and glyph.endpoint_confidence >= thresholds.min_endpoint_confidence
            and glyph.line_association_score >= thresholds.min_line_association_score
        ),
        "physical_verified": not reasons,
        "annotated_frame_trainable": False,
        "rejection_reasons": reasons,
    }


def _safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).map(
        lambda value: bool(value)
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().casefold() in {"1", "true", "yes", "pass", "passed"}
    )


def evaluate_p1(summary: Mapping[str, Any], thresholds: PilotThresholds = PilotThresholds()) -> dict[str, Any]:
    records = int(summary.get("records", 0) or 0)
    source_records = int(summary.get("record_list_records", 0) or 0)
    fields = set(summary.get("enriched_fields", summary.get("columns", [])) or [])
    field_groups = (
        {"image_type", "ImageType"},
        {"frames", "number_of_frames", "NumberOfFrames"},
        {"region_types", "region_type", "spatial_region_count"},
        {"vendor", "manufacturer", "Manufacturer"},
        {"software", "software_version", "software_versions", "SoftwareVersions"},
    )
    enrichment_complete = all(bool(fields & group) for group in field_groups)
    complete = (
        summary.get("complete") is True
        and summary.get("scan_target_complete") is True
        and str(summary.get("status", "")).casefold() == "complete"
        and records == source_records
    )
    metadata_only = summary.get("metadata_only") is True and summary.get("pixel_data_read") is False
    test_rows_sealed = bool(summary.get("locked_test_pixels_accessed") is False)
    digest_evidence = all(
        _is_sha256(summary.get(key, ""))
        for key in ("record_list_sha256", "instance_parquet_sha256", "region_parquet_sha256")
    )
    checks = {
        "complete_full_inventory": complete,
        "minimum_instance_count": records >= thresholds.p1_min_instances,
        "source_count_reconciled": records == source_records and source_records > 0,
        "required_enrichment_fields": enrichment_complete,
        "metadata_only_scan": metadata_only,
        "artifact_digests_recorded": digest_evidence,
        "locked_test_pixels_not_accessed": test_rows_sealed,
    }
    return {
        "passed": all(checks.values()),
        "records": records,
        "required_minimum": thresholds.p1_min_instances,
        "checks": checks,
        "remaining_work": None
        if all(checks.values())
        else [name for name, passed in checks.items() if not passed],
    }


def evaluate_p2(reviewed: pd.DataFrame, thresholds: PilotThresholds = PilotThresholds()) -> dict[str, Any]:
    required = {
        "sample_hash",
        "image_type",
        "frame_bucket",
        "region_type",
        "vendor",
        "software",
        "layout_family",
        "sample_hash_scheme",
        "manual_review_complete",
        "annotation_rendering_reviewed",
        "annotation_present",
        "measurement_key",
        "glyph_family",
    }
    missing = required.difference(reviewed.columns)
    if missing:
        return {"passed": False, "error": f"P2 review is missing columns: {sorted(missing)}"}
    partition_valid, locked, partition_error = _partition_contract(reviewed)
    non_test = partition_valid and not bool(locked.any())
    forbidden_columns = _forbidden_identifier_columns(reviewed)
    hash_provenance = bool(
        _sha256_series(reviewed["sample_hash"]).all()
        and reviewed["sample_hash_scheme"].astype(str).eq(SAMPLE_HASH_SCHEME).all()
    )
    strata = ["image_type", "frame_bucket", "region_type", "vendor", "software", "layout_family"]
    selection_valid = _selection_provenance_valid(
        reviewed,
        stage="P2",
        seed=thresholds.p2_selection_seed,
        strata=strata,
    )
    all_reviewed = bool(_safe_bool(reviewed["manual_review_complete"]).all())
    all_rendering = bool(_safe_bool(reviewed["annotation_rendering_reviewed"]).all())
    annotation_present = reviewed["annotation_present"]
    presence_recorded = bool(annotation_present.notna().all())
    positive = _safe_bool(annotation_present)
    typed = reviewed["measurement_key"].fillna("").astype(str).str.strip().ne("")
    glyphs = reviewed["glyph_family"].fillna("").astype(str).str.strip().ne("")
    checks = {
        "exact_predeclared_count": len(reviewed) == thresholds.p2_count,
        "unique_examples": reviewed["sample_hash"].nunique() == len(reviewed),
        "identifier_columns_excluded": not forbidden_columns,
        "pseudonymous_hash_provenance": hash_provenance,
        "locked_test_excluded": non_test,
        "partition_evidence_valid": partition_valid,
        "deterministic_stratified_selection": selection_valid,
        "manual_review_complete": all_reviewed,
        "annotation_rendering_assessed": all_rendering and presence_recorded,
        "positive_annotations_observed": bool(positive.any()),
        "measurement_mix_recorded_for_positives": bool(typed[positive].all()),
        "glyph_families_recorded_for_positives": bool(glyphs[positive].all()),
        "minimum_measurement_mix": reviewed.loc[positive, "measurement_key"].nunique()
        >= thresholds.p2_min_measurement_types,
        "minimum_glyph_family_mix": reviewed.loc[positive, "glyph_family"].nunique()
        >= thresholds.p2_min_glyph_families,
    }
    return {
        "passed": all(checks.values()),
        "examples": int(len(reviewed)),
        "annotation_positive": int(positive.sum()),
        "candidate_positive_yield": float(positive.mean()) if len(positive) else 0.0,
        "checks": checks,
        "remaining_work": None
        if all(checks.values())
        else [name for name, passed in checks.items() if not passed],
        "partition_error": partition_error,
        "forbidden_identifier_columns": sorted(forbidden_columns),
        "strata": {
            column: reviewed[column].fillna("UNKNOWN").astype(str).value_counts().astype(int).to_dict()
            for column in ("image_type", "frame_bucket", "region_type", "vendor", "software")
        },
    }


def _empty_rejection_reasons(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        decoded = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(decoded, list) and len(decoded) == 0


def _documented_rejection_reasons(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return len(value) > 0 and all(bool(str(item).strip()) for item in value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(decoded, list) and len(decoded) > 0 and all(bool(str(item).strip()) for item in decoded)


def _p3_claimed_acceptance(frame: pd.DataFrame) -> pd.Series:
    required = ("ocr_parse_valid", "glyph_localized", "physical_verified")
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"P3 results are missing columns: {missing}")
    accepted = pd.Series(True, index=frame.index)
    for column in required:
        accepted &= _safe_bool(frame[column])
    if "accepted" in frame:
        accepted &= _safe_bool(frame["accepted"])
    return accepted


def _geometry_json_physical_value(value: object, purpose: str) -> float | None:
    try:
        payload = json.loads(str(value))
        calibration_payload = payload["calibration"]
        calibration = RegionCalibration(
            x0=int(calibration_payload["x0"]),
            y0=int(calibration_payload["y0"]),
            x1=int(calibration_payload["x1"]),
            y1=int(calibration_payload["y1"]),
            delta_x=float(calibration_payload["delta_x"]),
            delta_y=float(calibration_payload["delta_y"]),
            x_unit=str(calibration_payload["x_unit"]),
            y_unit=str(calibration_payload["y_unit"]),
            region_type=str(calibration_payload.get("region_type", "unknown")),
        )
        start_x = float(payload["start_x"])
        start_y = float(payload["start_y"])
        end_x = float(payload["end_x"])
        end_y = float(payload["end_y"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not all(math.isfinite(item) for item in (start_x, start_y, end_x, end_y)):
        return None
    if not calibration.contains(start_x, start_y) or not calibration.contains(end_x, end_y):
        return None
    try:
        if purpose == "spatial_bmode":
            start = calibration.image_to_physical(start_x, start_y)
            end = calibration.image_to_physical(end_x, end_y)
            return float(np.linalg.norm(np.asarray(end) - np.asarray(start)))
        if purpose == "spectral_doppler":
            axes = [
                (calibration.x_unit, calibration.delta_x, start_x, end_x),
                (calibration.y_unit, calibration.delta_y, start_y, end_y),
            ]
            velocity_axes = [axis for axis in axes if axis[0] in VELOCITY_UNITS_TO_MPS]
            if len(velocity_axes) != 1:
                return None
            unit, delta, start_pixel, end_pixel = velocity_axes[0]
            return abs((end_pixel - start_pixel) * delta * VELOCITY_UNITS_TO_MPS[unit])
    except (TypeError, ValueError):
        return None
    return None


def _p3_geometry_verifiable(
    frame: pd.DataFrame,
    thresholds: PilotThresholds,
    vocabulary: MeasurementVocabulary,
) -> pd.Series:
    required = {
        "sample_hash",
        "sample_hash_scheme",
        "annotated_frame_hash",
        "parent_cine_hash",
        "artifact_hash_scheme",
        "measurement_key",
        "vocabulary_version",
        "glyph_bank_version",
        "geometry_verifier_version",
        "geometry_json",
        "region_id",
        "region_purpose",
        "endpoint_confidence",
        "line_association_score",
        "physical_value",
        "canonical_unit",
        "ocr_value",
        "study_value",
        "ocr_absolute_error",
        "study_absolute_error",
        "absolute_tolerance",
        "relative_tolerance",
        "rejection_reasons",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"P3 results are missing verifiable geometry columns: {sorted(missing)}")

    valid = pd.Series(True, index=frame.index, dtype=bool)
    valid &= _sha256_series(frame["sample_hash"])
    valid &= frame["sample_hash_scheme"].astype(str).eq(SAMPLE_HASH_SCHEME)
    valid &= _sha256_series(frame["annotated_frame_hash"])
    valid &= _sha256_series(frame["parent_cine_hash"])
    valid &= frame["artifact_hash_scheme"].astype(str).eq(ARTIFACT_HASH_SCHEME)
    valid &= frame["vocabulary_version"].astype(str).eq(vocabulary.version)
    valid &= frame["glyph_bank_version"].fillna("").astype(str).str.strip().ne("")
    valid &= frame["geometry_verifier_version"].astype(str).eq(GEOMETRY_VERIFIER_VERSION)
    valid &= frame["geometry_json"].map(_safe_json_mapping)
    valid &= frame["rejection_reasons"].map(_empty_rejection_reasons)

    numeric_columns = (
        "endpoint_confidence",
        "line_association_score",
        "physical_value",
        "ocr_value",
        "study_value",
        "ocr_absolute_error",
        "study_absolute_error",
        "absolute_tolerance",
        "relative_tolerance",
    )
    numeric = {column: pd.to_numeric(frame[column], errors="coerce") for column in numeric_columns}
    for values in numeric.values():
        valid &= values.map(math.isfinite)
    valid &= numeric["endpoint_confidence"].ge(thresholds.min_endpoint_confidence)
    valid &= numeric["endpoint_confidence"].le(1.0)
    valid &= numeric["line_association_score"].ge(thresholds.min_line_association_score)
    valid &= numeric["line_association_score"].le(1.0)
    valid &= numeric["absolute_tolerance"].gt(0)
    valid &= np.isclose(
        numeric["relative_tolerance"],
        thresholds.physical_relative_tolerance,
        rtol=0.0,
        atol=1e-12,
    )
    valid &= np.isclose(
        numeric["ocr_absolute_error"],
        (numeric["physical_value"] - numeric["ocr_value"]).abs(),
        rtol=1e-7,
        atol=1e-9,
    )
    valid &= np.isclose(
        numeric["study_absolute_error"],
        (numeric["physical_value"] - numeric["study_value"]).abs(),
        rtol=1e-7,
        atol=1e-9,
    )
    ocr_limit = np.maximum(
        numeric["absolute_tolerance"],
        thresholds.physical_relative_tolerance * numeric["ocr_value"].abs(),
    )
    study_limit = np.maximum(
        numeric["absolute_tolerance"],
        thresholds.physical_relative_tolerance * numeric["study_value"].abs(),
    )
    valid &= numeric["ocr_absolute_error"].le(ocr_limit + 1e-12)
    valid &= numeric["study_absolute_error"].le(study_limit + 1e-12)

    entries = {entry.key: entry for entry in vocabulary.entries}
    for index, key in frame["measurement_key"].astype(str).items():
        entry = entries.get(key)
        if entry is None:
            valid.loc[index] = False
            continue
        expected_purpose = "spatial_bmode" if entry.kind == "length" else "spectral_doppler"
        expected_tolerance = (
            thresholds.length_abs_tolerance_mm if entry.kind == "length" else thresholds.velocity_abs_tolerance_mps
        )
        if str(frame.at[index, "canonical_unit"]) != entry.canonical_unit:
            valid.loc[index] = False
        if str(frame.at[index, "region_purpose"]) != expected_purpose:
            valid.loc[index] = False
        try:
            geometry_region_id = str(json.loads(str(frame.at[index, "geometry_json"]))["region_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            geometry_region_id = ""
        if geometry_region_id != str(frame.at[index, "region_id"]):
            valid.loc[index] = False
        recomputed_physical = _geometry_json_physical_value(frame.at[index, "geometry_json"], expected_purpose)
        if recomputed_physical is None or not math.isclose(
            recomputed_physical,
            float(numeric["physical_value"].loc[index]),
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            valid.loc[index] = False
        if not math.isclose(float(numeric["absolute_tolerance"].loc[index]), expected_tolerance, rel_tol=0.0, abs_tol=1e-12):
            valid.loc[index] = False
        if not entry.plausible_min <= float(numeric["physical_value"].loc[index]) <= entry.plausible_max:
            valid.loc[index] = False
    return valid


def _p3_acceptance(
    frame: pd.DataFrame,
    thresholds: PilotThresholds = PilotThresholds(),
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> pd.Series:
    claimed = _p3_claimed_acceptance(frame)
    verifiable = _p3_geometry_verifiable(frame, thresholds, vocabulary)
    return claimed & verifiable


def _group_yield(frame: pd.DataFrame, accepted: pd.Series, column: str) -> dict[str, Any]:
    rows = []
    for value, group in frame.assign(_accepted=accepted).groupby(column, dropna=False, sort=True):
        rows.append(
            {
                "group": str(value),
                "candidates": int(len(group)),
                "accepted": int(group["_accepted"].sum()),
                "yield": float(group["_accepted"].mean()),
            }
        )
    return {row["group"]: {key: value for key, value in row.items() if key != "group"} for row in rows}


def evaluate_p3(
    results: pd.DataFrame,
    thresholds: PilotThresholds = PilotThresholds(),
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> dict[str, Any]:
    try:
        claimed = _p3_claimed_acceptance(results)
        verifiable = _p3_geometry_verifiable(results, thresholds, vocabulary)
        accepted = claimed & verifiable
    except KeyError as exc:
        return {"passed": False, "error": str(exc)}
    leakage = (
        _safe_bool(results["annotated_frame_trainable"]).any()
        if "annotated_frame_trainable" in results
        else True
    )
    partition_valid, locked_rows, partition_error = _partition_contract(results)
    locked = bool(locked_rows.any())
    prototype_yield = float(accepted.mean()) if len(results) else 0.0
    source_columns = {
        "sample_hash",
        "sample_hash_scheme",
        "annotated_frame_hash",
        "parent_cine_hash",
        "artifact_hash_scheme",
        "glyph_bank_version",
        "vocabulary_version",
    }
    source_linked = source_columns.issubset(results.columns)
    versioned = bool(
        source_linked
        and _sha256_series(results["sample_hash"]).all()
        and results["sample_hash_scheme"].astype(str).eq(SAMPLE_HASH_SCHEME).all()
        and results["vocabulary_version"].fillna("").astype(str).eq(vocabulary.version).all()
        and results["glyph_bank_version"].fillna("").astype(str).str.strip().ne("").all()
        and _sha256_series(results["annotated_frame_hash"]).all()
        and _sha256_series(results["parent_cine_hash"]).all()
        and results["artifact_hash_scheme"].astype(str).eq(ARTIFACT_HASH_SCHEME).all()
    )
    forbidden_columns = _forbidden_identifier_columns(results)
    selection_strata = ["image_type", "frame_bucket", "region_type", "vendor", "software", "layout_family"]
    selection_valid = _selection_provenance_valid(
        results,
        stage="P3",
        seed=thresholds.p3_selection_seed,
        strata=selection_strata,
    )
    rejection_documented = (
        results["rejection_reasons"].map(_documented_rejection_reasons)
        if "rejection_reasons" in results
        else pd.Series(False, index=results.index)
    )
    checks = {
        "exact_predeclared_count": len(results) == thresholds.p3_count,
        "unique_examples": "sample_hash" in results and results["sample_hash"].nunique() == len(results),
        "identifier_columns_excluded": not forbidden_columns,
        "locked_test_excluded": partition_valid and not locked,
        "partition_evidence_valid": partition_valid,
        "annotated_frames_never_trainable": not bool(leakage),
        "source_linked_versioned_evidence": versioned,
        "deterministic_stratified_selection": selection_valid,
        "all_acceptance_claims_geometry_verified": bool((~claimed | verifiable).all()),
        "every_candidate_has_verifiable_acceptance_or_documented_rejection": bool(
            (accepted | (~claimed & rejection_documented)).all()
        ),
        "minimum_prototype_yield": prototype_yield >= thresholds.p3_yield_min,
    }
    group_columns = [column for column in ("measurement_key", "vendor", "software") if column in results]
    return {
        "passed": all(checks.values()),
        "candidates": int(len(results)),
        "accepted": int(accepted.sum()),
        "yield": prototype_yield,
        "minimum_yield": thresholds.p3_yield_min,
        "precision_status": "requires_P5_blinded_two_reviewer_audit",
        "checks": checks,
        "remaining_work": None
        if all(checks.values())
        else [name for name, passed in checks.items() if not passed],
        "partition_error": partition_error,
        "forbidden_identifier_columns": sorted(forbidden_columns),
        "unverifiable_acceptance_claims": int((claimed & ~verifiable).sum()),
        "by_group": {column: _group_yield(results, accepted, column) for column in group_columns},
    }


_SHORTCUT_PROBE_DIGEST_FIELDS = (
    "complete",
    "predeclared",
    "probe_protocol_version",
    "metric_name",
    "metric_direction",
    "paired_samples",
    "annotated_metric",
    "clean_metric",
    "relative_performance_gap",
    "clean_only_probe_passed",
    "probe_split",
    "locked_test_accessed",
    "annotated_frames_used_for_fitting",
)


def shortcut_probe_evidence_digest(evidence: Mapping[str, Any]) -> str:
    payload = {key: evidence.get(key) for key in _SHORTCUT_PROBE_DIGEST_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _evaluate_shortcut_probe(
    evidence: Mapping[str, Any] | None,
    thresholds: PilotThresholds,
) -> dict[str, Any]:
    required = {
        "complete",
        "predeclared",
        "probe_protocol_version",
        "metric_name",
        "metric_direction",
        "evidence_sha256",
        "paired_samples",
        "annotated_metric",
        "clean_metric",
        "relative_performance_gap",
        "clean_only_probe_passed",
        "probe_split",
        "locked_test_accessed",
        "annotated_frames_used_for_fitting",
    }
    if evidence is None:
        return {
            "passed": False,
            "error": "P4 requires recorded annotated-vs-clean shortcut-probe evidence",
            "remaining_work": "Run the predeclared paired clean-only probe and attach its digest-backed evidence.",
        }
    missing = required.difference(evidence)
    if missing:
        return {
            "passed": False,
            "error": f"Shortcut-probe evidence is missing fields: {sorted(missing)}",
            "remaining_work": "Complete the predeclared paired clean-only probe evidence record.",
        }
    try:
        annotated = float(evidence["annotated_metric"])
        clean = float(evidence["clean_metric"])
        reported_gap = float(evidence["relative_performance_gap"])
        paired = int(evidence["paired_samples"])
    except (TypeError, ValueError):
        return {"passed": False, "error": "Shortcut-probe metrics/count are not numeric"}
    finite = all(math.isfinite(value) for value in (annotated, clean, reported_gap))
    denominator = max(abs(clean), 1e-12)
    computed_gap = abs(annotated - clean) / denominator if finite else math.inf
    split = str(evidence["probe_split"]).strip().casefold()
    metric_direction = str(evidence["metric_direction"]).strip().casefold()
    checks = {
        "complete": evidence["complete"] is True,
        "predeclared": evidence["predeclared"] is True,
        "protocol_version_recorded": bool(str(evidence["probe_protocol_version"]).strip()),
        "metric_contract_recorded": bool(str(evidence["metric_name"]).strip())
        and metric_direction in {"higher_is_better", "lower_is_better"},
        "digest_recorded": _is_sha256(evidence["evidence_sha256"]),
        "digest_matches_evidence": finite
        and _is_sha256(evidence["evidence_sha256"])
        and str(evidence["evidence_sha256"]).casefold() == shortcut_probe_evidence_digest(evidence),
        "minimum_paired_samples": paired >= thresholds.clean_probe_min_pairs,
        "metrics_finite": finite,
        "reported_gap_reconciled": finite
        and math.isclose(reported_gap, computed_gap, rel_tol=1e-7, abs_tol=1e-9),
        "gap_below_predeclared_limit": finite
        and computed_gap <= thresholds.clean_probe_max_relative_gap,
        "probe_passed": evidence["clean_only_probe_passed"] is True,
        "held_out_validation_probe_split": split == "validation",
        "locked_test_not_accessed": evidence["locked_test_accessed"] is False,
        "annotated_frames_not_used_for_fitting": evidence["annotated_frames_used_for_fitting"] is False,
    }
    return {
        "passed": all(checks.values()),
        "paired_samples": paired,
        "annotated_metric": annotated,
        "clean_metric": clean,
        "relative_performance_gap": computed_gap,
        "maximum_relative_gap": thresholds.clean_probe_max_relative_gap,
        "probe_protocol_version": str(evidence["probe_protocol_version"]),
        "metric_name": str(evidence["metric_name"]),
        "metric_direction": metric_direction,
        "evidence_sha256": str(evidence["evidence_sha256"]).casefold(),
        "checks": checks,
        "remaining_work": None if all(checks.values()) else "Repair or rerun the paired clean-only shortcut probe.",
    }


def evaluate_p4_clean_recovery(
    p3_results: pd.DataFrame,
    recovery_evidence: pd.DataFrame,
    thresholds: PilotThresholds = PilotThresholds(),
    shortcut_probe_evidence: Mapping[str, Any] | None = None,
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> tuple[dict[str, Any], pd.DataFrame]:
    try:
        accepted = p3_results.loc[_p3_acceptance(p3_results, thresholds, vocabulary)].copy()
    except KeyError as exc:
        return {"passed": False, "error": str(exc)}, pd.DataFrame()
    missing_source = {"annotated_frame_hash", "parent_cine_hash"}.difference(accepted.columns)
    if missing_source:
        report = {"passed": False, "error": f"P3 accepted rows are missing source-link columns: {sorted(missing_source)}"}
        return report, pd.DataFrame()
    required = {
        "sample_hash",
        "sample_hash_scheme",
        "annotated_frame_hash",
        "recovered_frame_hash",
        "parent_cine_hash",
        "artifact_hash_scheme",
        "split",
        "tissue_match_score",
        "annotation_pixels_remaining",
        "overlay_detector_passed",
        "clean_only_probe_eligible",
    }
    missing = required.difference(recovery_evidence.columns)
    if missing:
        report = {"passed": False, "error": f"P4 evidence is missing columns: {sorted(missing)}"}
        return report, pd.DataFrame()
    evidence = recovery_evidence.copy()
    forbidden_columns = _forbidden_identifier_columns(evidence)
    p3_partition_valid, p3_locked, p3_partition_error = _partition_contract(accepted)
    evidence_partition_valid, evidence_locked, evidence_partition_error = _partition_contract(evidence)
    digest_evidence_valid = bool(
        _sha256_series(evidence["sample_hash"]).all()
        and evidence["sample_hash_scheme"].astype(str).eq(SAMPLE_HASH_SCHEME).all()
        and _sha256_series(evidence["annotated_frame_hash"]).all()
        and _sha256_series(evidence["recovered_frame_hash"]).all()
        and _sha256_series(evidence["parent_cine_hash"]).all()
        and evidence["artifact_hash_scheme"].astype(str).eq(ARTIFACT_HASH_SCHEME).all()
    )
    duplicate_evidence = bool(evidence["sample_hash"].duplicated().any())
    duplicate_clean_hashes = bool(evidence["recovered_frame_hash"].duplicated().any())
    merged = accepted.merge(evidence, on="sample_hash", how="left", suffixes=("", "_recovery"), indicator=True)
    distinct = merged["annotated_frame_hash"].fillna("").astype(str).ne(
        merged["recovered_frame_hash"].fillna("").astype(str)
    )
    hashes_present = (
        merged["annotated_frame_hash"].fillna("").astype(str).str.len().gt(0)
        & merged["recovered_frame_hash"].fillna("").astype(str).str.len().gt(0)
        & merged["parent_cine_hash"].fillna("").astype(str).str.len().gt(0)
    )
    all_annotated_hashes = set(p3_results["annotated_frame_hash"].fillna("").astype(str).str.casefold())
    recovered_is_never_annotated = ~merged["recovered_frame_hash"].fillna("").astype(str).str.casefold().isin(
        all_annotated_hashes
    )
    parent_match = pd.Series(True, index=merged.index)
    if "parent_cine_hash_recovery" in merged:
        parent_match = merged["parent_cine_hash"].fillna("").astype(str).eq(
            merged["parent_cine_hash_recovery"].fillna("").astype(str)
        )
    annotated_source_match = pd.Series(True, index=merged.index)
    if "annotated_frame_hash_recovery" in merged:
        annotated_source_match = merged["annotated_frame_hash"].fillna("").astype(str).eq(
            merged["annotated_frame_hash_recovery"].fillna("").astype(str)
        )
    clean = (
        merged["_merge"].eq("both")
        & distinct
        & recovered_is_never_annotated
        & hashes_present
        & parent_match
        & annotated_source_match
        & pd.to_numeric(merged["tissue_match_score"], errors="coerce").ge(thresholds.min_tissue_match_score)
        & pd.to_numeric(merged["annotation_pixels_remaining"], errors="coerce").eq(0)
        & _safe_bool(merged["overlay_detector_passed"])
        & _safe_bool(merged["clean_only_probe_eligible"])
    )
    coverage = float(merged["_merge"].eq("both").mean()) if len(merged) else 0.0
    recovery_rate = float(clean.mean()) if len(merged) else 0.0
    shortcut_probe = _evaluate_shortcut_probe(shortcut_probe_evidence, thresholds)
    locked = bool(p3_locked.any() or evidence_locked.any())
    checks = {
        "all_accepted_have_evidence": coverage >= thresholds.p4_evidence_coverage_min,
        "evidence_unique_per_example": not duplicate_evidence,
        "clean_hash_unique_per_example": not duplicate_clean_hashes,
        "minimum_clean_recovery_rate": recovery_rate >= thresholds.p4_clean_recovery_min,
        "annotated_and_clean_hashes_distinct": bool(distinct[clean].all()),
        "clean_hash_never_matches_any_annotated_hash": bool(recovered_is_never_annotated[clean].all()),
        "annotated_source_and_parent_links_match": bool(parent_match.all() and annotated_source_match.all()),
        "zero_annotation_pixels": bool(
            pd.to_numeric(merged.loc[clean, "annotation_pixels_remaining"], errors="coerce").eq(0).all()
        ),
        "identifier_columns_excluded": not forbidden_columns,
        "digest_provenance_valid": digest_evidence_valid,
        "partition_evidence_valid": p3_partition_valid and evidence_partition_valid,
        "locked_test_excluded": not locked,
        "annotated_vs_clean_shortcut_probe_passed": shortcut_probe.get("passed") is True,
    }
    safe_columns = [
        column
        for column in (
            "sample_hash",
            "sample_hash_scheme",
            "split",
            "measurement_key",
            "physical_value",
            "canonical_unit",
            "geometry_json",
            "parent_cine_hash",
            "artifact_hash_scheme",
            "recovered_frame_hash",
        )
        if column in merged
    ]
    stage_passed = all(checks.values())
    training = merged.loc[clean, safe_columns].copy() if stage_passed else pd.DataFrame(columns=safe_columns)
    if "recovered_frame_hash" in training:
        training = training.rename(columns={"recovered_frame_hash": "clean_frame_hash"})
    training["source_is_clean_recovery"] = True
    training["annotated_frame_trainable"] = False
    report = {
        "passed": stage_passed,
        "accepted_prototypes": int(len(accepted)),
        "evidence_rows": int(len(evidence)),
        "usable_clean_frames": int(clean.sum()),
        "evidence_coverage": coverage,
        "clean_recovery_rate": recovery_rate,
        "checks": checks,
        "remaining_work": None
        if stage_passed
        else [name for name, passed in checks.items() if not passed],
        "partition_errors": {
            "p3": p3_partition_error,
            "recovery": evidence_partition_error,
        },
        "forbidden_identifier_columns": sorted(forbidden_columns),
        "shortcut_probe": shortcut_probe,
        "training_contract": "Only recovered clean-frame hashes are emitted; annotated-frame hashes are excluded.",
    }
    return report, training.reset_index(drop=True)


def select_p5_audit_candidates(
    p3_results: pd.DataFrame,
    clean_training_manifest: pd.DataFrame,
    *,
    thresholds: PilotThresholds = PilotThresholds(),
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> pd.DataFrame:
    """Deterministically select the exact stratified P5 audit packet."""

    accepted = p3_results.loc[_p3_acceptance(p3_results, thresholds, vocabulary)].copy()
    clean_hashes = set(clean_training_manifest.get("sample_hash", pd.Series(dtype=str)).astype(str))
    eligible = accepted[accepted["sample_hash"].astype(str).isin(clean_hashes)].copy()
    partition_valid, locked, partition_error = _partition_contract(eligible)
    if not partition_valid or bool(locked.any()):
        raise ValueError(f"P5 selection partition evidence is unsafe: {partition_error or 'locked rows present'}")
    if not _sha256_series(eligible["sample_hash"]).all():
        raise ValueError("P5 selection contains non-digest sample identifiers")
    if not eligible["sample_hash_scheme"].astype(str).eq(SAMPLE_HASH_SCHEME).all():
        raise ValueError("P5 selection lacks HMAC sample-hash provenance")
    strata = ["measurement_key", "vendor", "software"]
    selected = _round_robin_stratified(
        eligible,
        thresholds.p5_count,
        thresholds.p5_selection_seed,
        strata,
    )
    return _attach_selection_provenance(
        selected,
        stage="P5",
        seed=thresholds.p5_selection_seed,
        strata=strata,
    )


def _cohen_kappa(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0:
        return 0.0
    observed = float(np.mean(left == right))
    left_true = float(np.mean(left))
    right_true = float(np.mean(right))
    expected = left_true * right_true + (1.0 - left_true) * (1.0 - right_true)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def _p5_group_report(
    universe: pd.DataFrame,
    sample_audit: pd.DataFrame,
    dimension: str,
    thresholds: PilotThresholds,
    vocabulary: MeasurementVocabulary,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    accepted = _p3_acceptance(universe, thresholds, vocabulary)
    with_acceptance = universe.assign(_accepted=accepted)
    for value, group in with_acceptance.groupby(dimension, dropna=False, sort=True):
        label = str(value)
        audit = sample_audit[sample_audit[dimension].fillna("UNKNOWN").astype(str) == label]
        accepted_population = int(group["_accepted"].sum())
        required = accepted_population >= thresholds.active_group_min_population
        precision = float(audit["sample_correct"].mean()) if len(audit) else None
        decision_agreement = float(audit["decision_agreement"].mean()) if len(audit) else None
        value_agreement = float(audit["value_agreement"].mean()) if len(audit) else None
        p95_normalized_error = (
            float(np.percentile(audit["normalized_absolute_error"], 95)) if len(audit) else None
        )
        audit_enough = len(audit) >= thresholds.min_audits_per_active_group
        precision_enough = precision is not None and precision >= thresholds.group_precision_min
        agreement_enough = bool(
            decision_agreement is not None
            and value_agreement is not None
            and decision_agreement >= thresholds.reviewer_agreement_min
            and value_agreement >= thresholds.reviewer_agreement_min
        )
        error_small_enough = bool(
            p95_normalized_error is not None
            and p95_normalized_error <= thresholds.group_p95_normalized_error_max
        )
        output[label] = {
            "candidates": int(len(group)),
            "accepted": accepted_population,
            "yield": float(group["_accepted"].mean()),
            "audited": int(len(audit)),
            "precision": precision,
            "decision_agreement": decision_agreement,
            "value_agreement": value_agreement,
            "mean_absolute_error": float(audit["sample_absolute_error"].mean()) if len(audit) else None,
            "mean_normalized_absolute_error": float(audit["normalized_absolute_error"].mean())
            if len(audit)
            else None,
            "p95_normalized_absolute_error": p95_normalized_error,
            "required_for_gate": required,
            "passed": bool(
                not required
                or (audit_enough and precision_enough and agreement_enough and error_small_enough)
            ),
        }
    return output


def evaluate_p5_blinded_audit(
    p3_results: pd.DataFrame,
    clean_training_manifest: pd.DataFrame,
    audits: pd.DataFrame,
    *,
    thresholds: PilotThresholds = PilotThresholds(),
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> dict[str, Any]:
    required = {
        "sample_hash",
        "sample_hash_scheme",
        "reviewer_id",
        "reviewer_id_scheme",
        "blinded",
        "independent",
        "source_answer_visible",
        "blinding_evidence_hash",
        "reviewer_correct",
        "convention_match",
        "reviewed_value",
        "split",
        "selection_rank",
        "selection_stage",
        "selection_scheme",
        "selection_seed",
        "selection_manifest_sha256",
    }
    missing = required.difference(audits.columns)
    if missing:
        return {"passed": False, "error": f"P5 audit is missing columns: {sorted(missing)}"}
    forbidden_columns = _forbidden_identifier_columns(audits, allow_reviewer_id=True)
    if forbidden_columns:
        return {
            "passed": False,
            "error": f"P5 audit contains forbidden raw identifier columns: {sorted(forbidden_columns)}",
        }
    required_p3 = {
        "sample_hash",
        "measurement_key",
        "vendor",
        "software",
        "physical_value",
        "canonical_unit",
    }
    missing_p3 = required_p3.difference(p3_results.columns)
    if missing_p3:
        return {"passed": False, "error": f"P3 results are missing P5 source columns: {sorted(missing_p3)}"}
    eligible_hashes = set(clean_training_manifest.get("sample_hash", pd.Series(dtype=str)).astype(str))
    try:
        accepted = p3_results.loc[_p3_acceptance(p3_results, thresholds, vocabulary)].copy()
        expected_selection = select_p5_audit_candidates(
            p3_results,
            clean_training_manifest,
            thresholds=thresholds,
            vocabulary=vocabulary,
        )
    except (KeyError, ValueError) as exc:
        return {
            "passed": False,
            "error": f"P5 predeclared selection could not be verified: {exc}",
            "remaining_work": "Repair P3/P4 evidence and regenerate the deterministic P5 audit packet.",
        }
    accepted = accepted[accepted["sample_hash"].astype(str).isin(eligible_hashes)].copy()
    audit = audits.copy()
    audit["sample_hash"] = audit["sample_hash"].astype(str)
    audit["reviewer_id"] = audit["reviewer_id"].astype(str)
    audit["_correct"], correct_valid = _strict_bool_series(audit["reviewer_correct"])
    audit["_convention"], convention_valid = _strict_bool_series(audit["convention_match"])
    audit["_blinded"], blinded_valid = _strict_bool_series(audit["blinded"])
    audit["_independent"], independent_valid = _strict_bool_series(audit["independent"])
    answer_visible, answer_visible_valid = _strict_bool_series(audit["source_answer_visible"])
    audit["_answer_hidden"] = ~answer_visible
    categorical_evidence_valid = bool(
        correct_valid.all()
        and convention_valid.all()
        and blinded_valid.all()
        and independent_valid.all()
        and answer_visible_valid.all()
    )
    audit["_reviewed_value"] = pd.to_numeric(audit["reviewed_value"], errors="coerce")
    duplicate_reviews = bool(audit.duplicated(["sample_hash", "reviewer_id"]).any())
    global_reviewers = sorted(audit["reviewer_id"].unique())
    pseudonymous_reviewers = bool(
        audit["reviewer_id"].map(_is_sha256).all()
        and audit["reviewer_id_scheme"].astype(str).eq(REVIEWER_ID_SCHEME).all()
    )
    hash_provenance = bool(
        audit["sample_hash"].map(_is_sha256).all()
        and audit["sample_hash_scheme"].astype(str).eq(SAMPLE_HASH_SCHEME).all()
        and audit["blinding_evidence_hash"].map(_is_sha256).all()
    )
    blinding_packets = audit.groupby("reviewer_id")["blinding_evidence_hash"].nunique()
    distinct_blinding_packets = bool(
        len(blinding_packets) == 2
        and blinding_packets.eq(1).all()
        and audit["blinding_evidence_hash"].nunique() == 2
    )
    partition_valid, audit_locked, partition_error = _partition_contract(audit)
    review_counts = audit.groupby("sample_hash")["reviewer_id"].nunique()
    audit_samples = set(audit["sample_hash"])
    sample_count = len(audit_samples)
    all_eligible = audit_samples.issubset(set(accepted["sample_hash"].astype(str)))
    expected_by_hash = expected_selection.set_index("sample_hash")[
        [
            "selection_rank",
            "selection_stage",
            "selection_scheme",
            "selection_seed",
            "selection_manifest_sha256",
        ]
    ]
    observed_selection = audit.set_index("sample_hash")
    selection_exact = set(observed_selection.index.unique()) == set(expected_by_hash.index)
    if selection_exact:
        for sample_hash, group in observed_selection.groupby(level=0, sort=False):
            expected_row = expected_by_hash.loc[sample_hash]
            if not all(group[column].astype(str).eq(str(expected_row[column])).all() for column in expected_by_hash.columns):
                selection_exact = False
                break

    accepted_columns = [
        column
        for column in ("sample_hash", "measurement_key", "vendor", "software", "physical_value", "canonical_unit")
        if column in accepted
    ]
    joined = audit.merge(accepted[accepted_columns].drop_duplicates("sample_hash"), on="sample_hash", how="left")
    joined["_absolute_error"] = (
        joined["_reviewed_value"] - pd.to_numeric(joined.get("physical_value"), errors="coerce")
    ).abs()

    sample_rows = []
    for sample_hash, group in joined.groupby("sample_hash", sort=True):
        decisions = (group["_correct"] & group["_convention"]).to_numpy(dtype=bool)
        values = group["_reviewed_value"].to_numpy(dtype=float)
        measurement_key = str(group["measurement_key"].iloc[0]) if "measurement_key" in group else ""
        try:
            error_tolerance = vocabulary.entry(measurement_key).audit_abs_error_max
        except KeyError:
            error_tolerance = 0.0
        value_agreement = bool(len(values) == 2 and np.isfinite(values).all() and abs(values[0] - values[1]) <= error_tolerance)
        row = {
            "sample_hash": sample_hash,
            "measurement_key": measurement_key,
            "vendor": str(group["vendor"].iloc[0]) if "vendor" in group else "UNKNOWN",
            "software": str(group["software"].iloc[0]) if "software" in group else "UNKNOWN",
            "sample_correct": bool(len(decisions) == 2 and decisions.all()),
            "decision_agreement": bool(len(decisions) == 2 and decisions[0] == decisions[1]),
            "value_agreement": value_agreement,
            "sample_absolute_error": float(group["_absolute_error"].mean()),
            "error_tolerance": error_tolerance,
        }
        sample_rows.append(row)
    sample_audit = pd.DataFrame(sample_rows)
    if sample_audit.empty:
        sample_audit = pd.DataFrame(
            columns=[
                "sample_hash",
                "measurement_key",
                "vendor",
                "software",
                "sample_correct",
                "decision_agreement",
                "value_agreement",
                "sample_absolute_error",
                "error_tolerance",
            ]
        )
    sample_audit["normalized_absolute_error"] = pd.to_numeric(
        sample_audit["sample_absolute_error"], errors="coerce"
    ) / pd.to_numeric(sample_audit["error_tolerance"], errors="coerce").replace(0, np.nan)
    precision = float(sample_audit["sample_correct"].mean()) if len(sample_audit) else 0.0
    decision_agreement = float(sample_audit["decision_agreement"].mean()) if len(sample_audit) else 0.0
    value_agreement = float(sample_audit["value_agreement"].mean()) if len(sample_audit) else 0.0

    reviewer_pivot = joined.pivot_table(
        index="sample_hash",
        columns="reviewer_id",
        values="_correct",
        aggfunc="first",
    )
    if len(global_reviewers) == 2 and set(global_reviewers).issubset(reviewer_pivot.columns):
        complete = reviewer_pivot.dropna(subset=global_reviewers)
        kappa = _cohen_kappa(
            complete[global_reviewers[0]].to_numpy(dtype=bool),
            complete[global_reviewers[1]].to_numpy(dtype=bool),
        )
    else:
        kappa = 0.0

    error_by_measurement: dict[str, Any] = {}
    measurement_error_pass = True
    for key, group in sample_audit.groupby("measurement_key", sort=True):
        tolerance = vocabulary.entry(str(key)).audit_abs_error_max if str(key) in {entry.key for entry in vocabulary.entries} else 0.0
        p95 = float(np.percentile(group["sample_absolute_error"], 95)) if len(group) else math.inf
        passed = bool(p95 <= tolerance)
        measurement_error_pass &= passed
        error_by_measurement[str(key)] = {
            "audited": int(len(group)),
            "mean_absolute_error": float(group["sample_absolute_error"].mean()),
            "p95_absolute_error": p95,
            "canonical_unit": vocabulary.entry(str(key)).canonical_unit if tolerance else None,
            "p95_error_max": tolerance,
            "passed": passed,
        }

    group_reports = {
        dimension: _p5_group_report(p3_results, sample_audit, dimension, thresholds, vocabulary)
        for dimension in ("measurement_key", "vendor", "software")
        if dimension in p3_results
    }
    group_conditions_pass = all(
        row["passed"] for dimension_report in group_reports.values() for row in dimension_report.values()
    )
    checks = {
        "exact_predeclared_sample_count": sample_count == thresholds.p5_count,
        "exactly_two_global_reviewers": len(global_reviewers) == 2,
        "reviewer_ids_are_pseudonymous": pseudonymous_reviewers,
        "identifier_digest_provenance": hash_provenance,
        "distinct_digest_backed_blinding_packets": distinct_blinding_packets,
        "exactly_two_reviews_per_sample": bool(len(review_counts) == sample_count and review_counts.eq(2).all()),
        "no_duplicate_reviewer_rows": not duplicate_reviews,
        "categorical_review_evidence_unambiguous": categorical_evidence_valid,
        "all_reviews_blinded": bool(audit["_blinded"].all()),
        "all_reviews_independent": bool(audit["_independent"].all()),
        "source_answer_hidden_from_reviewers": bool(audit["_answer_hidden"].all()),
        "predeclared_stratified_sample_exact": selection_exact,
        "all_samples_accepted_and_clean": all_eligible,
        "review_values_complete": bool(audit["_reviewed_value"].notna().all()),
        "minimum_precision": precision >= thresholds.p5_precision_min,
        "minimum_decision_agreement": decision_agreement >= thresholds.reviewer_agreement_min,
        "minimum_value_agreement": value_agreement >= thresholds.reviewer_agreement_min,
        "measurement_error_within_tolerance": measurement_error_pass,
        "active_strata_precision_and_coverage": group_conditions_pass,
        "partition_evidence_valid": partition_valid,
        "locked_test_excluded": partition_valid and not bool(audit_locked.any()),
    }
    return {
        "passed": all(checks.values()),
        "audited_samples": sample_count,
        "review_rows": int(len(audit)),
        "reviewer_count": len(global_reviewers),
        "precision": precision,
        "decision_agreement": decision_agreement,
        "value_agreement": value_agreement,
        "cohen_kappa_correctness": kappa,
        "mean_absolute_error": float(sample_audit["sample_absolute_error"].mean()) if len(sample_audit) else None,
        "p95_absolute_error": float(np.percentile(sample_audit["sample_absolute_error"], 95))
        if len(sample_audit)
        else None,
        "checks": checks,
        "remaining_work": None
        if all(checks.values())
        else [name for name, passed in checks.items() if not passed],
        "partition_error": partition_error,
        "selection_scheme": SELECTION_SCHEME,
        "selection_seed": thresholds.p5_selection_seed,
        "error_by_measurement": error_by_measurement,
        "by_group": group_reports,
    }


def evaluate_caliper_pilot(
    p1_summary: Mapping[str, Any],
    p2_reviewed: pd.DataFrame,
    p3_results: pd.DataFrame,
    recovery_evidence: pd.DataFrame,
    p5_audits: pd.DataFrame,
    *,
    shortcut_probe_evidence: Mapping[str, Any] | None = None,
    thresholds: PilotThresholds = PilotThresholds(),
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> tuple[dict[str, Any], pd.DataFrame]:
    p1 = evaluate_p1(p1_summary, thresholds)
    p2 = evaluate_p2(p2_reviewed, thresholds)
    p3 = evaluate_p3(p3_results, thresholds, vocabulary)
    if p3.get("passed") is True:
        p4, clean_training = evaluate_p4_clean_recovery(
            p3_results,
            recovery_evidence,
            thresholds,
            shortcut_probe_evidence,
            vocabulary,
        )
    else:
        p4, clean_training = (
            {
                "passed": False,
                "error": "P4 is blocked until P3 verifiable geometry evidence passes",
                "remaining_work": "Repair P3 evidence before attempting clean-frame recovery promotion.",
            },
            pd.DataFrame(),
        )
    p5 = evaluate_p5_blinded_audit(
        p3_results,
        clean_training,
        p5_audits,
        thresholds=thresholds,
        vocabulary=vocabulary,
    )
    stages = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5}
    passed = all(stage.get("passed") is True for stage in stages.values())
    remaining = [name for name, stage in stages.items() if stage.get("passed") is not True]
    partition_states = [_partition_contract(frame) for frame in (p2_reviewed, p3_results, recovery_evidence, p5_audits)]
    locked_test_accessed = bool(p1_summary.get("locked_test_pixels_accessed") is True) or any(
        bool(locked.any()) for _, locked, _ in partition_states
    )
    report = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "HEM-FM v4.0 section 8 caliper P1-P5",
        "vocabulary_version": vocabulary.version,
        "passed": passed,
        "promotable": passed,
        "gold_mined_weight_selection_allowed": passed,
        "annotated_frames_trainable": False,
        "locked_test_accessed": locked_test_accessed,
        "all_partition_evidence_valid": all(valid for valid, _, _ in partition_states),
        "thresholds": asdict(thresholds),
        "stages": stages,
        "remaining_stages": remaining,
        "promotion_contract": (
            "Mined geometry may receive a Gold-mined weight only after every P1-P5 check passes. "
            "Annotated measurement frames are never trainable."
        ),
        "remaining_work": {
            name: stage.get("remaining_work", stage.get("error"))
            for name, stage in stages.items()
            if stage.get("passed") is not True
        },
    }
    return report, clean_training if passed else pd.DataFrame()


def _assert_public_safe(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normalise_column_name(key)
            if normalized in _RAW_IDENTIFIER_COLUMN_KEYS or normalized == _normalise_column_name("reviewer_id"):
                raise ValueError(f"Public evidence contains a private identifier at {path}.{key}")
            key_text = str(key)
            if _looks_like_filesystem_path(key_text):
                raise ValueError(f"Public evidence contains a filesystem path in a key at {path}")
            _assert_public_safe(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_public_safe(child, f"{path}[{index}]")
    elif isinstance(value, str):
        text = value.strip()
        if _looks_like_filesystem_path(text):
            raise ValueError(f"Public evidence contains a filesystem path at {path}")


def write_caliper_pilot_artifacts(
    output_root: str | Path,
    report: Mapping[str, Any],
    vocabulary: MeasurementVocabulary = DEFAULT_VOCABULARY,
) -> dict[str, str]:
    """Write only aggregate, identifier-free evidence suitable for a public repository."""

    _assert_public_safe(report)
    required_report_fields = {
        "schema_version",
        "protocol",
        "vocabulary_version",
        "passed",
        "promotable",
        "thresholds",
        "stages",
        "promotion_contract",
    }
    missing = required_report_fields.difference(report)
    if missing:
        raise ValueError(f"Public caliper evidence is not a complete pilot report: {sorted(missing)}")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "caliper_pilot_go_no_go.json"
    vocabulary_path = root / "caliper_measurement_vocabulary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    vocabulary_path.write_text(json.dumps(vocabulary.to_public_dict(), indent=2), encoding="utf-8")
    return {"go_no_go": str(report_path), "vocabulary": str(vocabulary_path)}
