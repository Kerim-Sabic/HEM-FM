from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import yaml

from .hashing import sha256_file


MANIFEST_NAME = "completion_manifest.json"
KINDS = ("transfer", "trace")
MODES = ("full",)
PREDICTION_NAMES = {
    "transfer": "validation_predictions_private.csv",
    "trace": "validation_masks_private.npz",
}
ARTIFACT_NAMES = {
    "report": "report.json",
    "best": "checkpoint_best.pt",
    "last": "checkpoint_last.pt",
}
REQUIRED_STAGES = ("frozen", "peft", "selective_unfreeze")
MATCHED_AUDIT_CHECKS = (
    "development_rows_only",
    "patient_disjoint",
    "exact_immutable_validation_cohort",
    "minimum_validation_samples",
    "three_seed_full_ensemble",
    "finite_positive_uncertainty",
    "locked_test_not_accessed",
)


class CompletionError(RuntimeError):
    """A fail-closed completion-evidence validation error."""


def _reject_json_constant(value: str) -> None:
    raise CompletionError(f"JSON contains non-finite constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CompletionError(f"JSON contains duplicate key {key!r}")
        output[key] = value
    return output


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompletionError(f"missing {label}: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompletionError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompletionError(f"{label} must be a JSON object: {path}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CompletionError(f"missing config: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CompletionError(f"invalid config: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CompletionError("config must be a mapping")
    return value


def _require_exact_bool(
    value: dict[str, Any], key: str, expected: bool, *, label: str
) -> None:
    if key not in value or type(value[key]) is not bool or value[key] is not expected:
        rendered = "true" if expected else "false"
        raise CompletionError(f"{label}.{key} must be JSON boolean {rendered}")


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise CompletionError(f"{label} must be a JSON integer")
    if minimum is not None and value < minimum:
        raise CompletionError(f"{label} must be at least {minimum}")
    return value


def _parse_utc(value: str, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise CompletionError(f"{label} must be a non-empty ISO UTC timestamp")
    normalized = value.strip()
    if normalized.endswith(("Z", "z")):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CompletionError(f"{label} is not a valid ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise CompletionError(f"{label} must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def _assert_fresh(path: Path, not_before: datetime, *, label: str) -> None:
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError as exc:
        raise CompletionError(f"cannot stat {label}: {path}: {exc}") from exc
    if modified < not_before:
        raise CompletionError(
            f"stale {label}: modified {modified.isoformat()} before {not_before.isoformat()}"
        )


def _validate_hex(value: Any, length: int, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise CompletionError(
            f"{label} must be a {length}-character hexadecimal string"
        )
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        raise CompletionError(f"{label} must be hexadecimal")
    return lowered


def _config_identity(config: dict[str, Any]) -> dict[str, Any]:
    protocol = config.get("protocol")
    if protocol != "HEM-FM-v4.0":
        raise CompletionError("config.protocol must be HEM-FM-v4.0")
    assets = config.get("assets")
    if not isinstance(assets, dict):
        raise CompletionError("config.assets must be a mapping")
    revision = _validate_hex(
        assets.get("echojepa_source_revision"),
        40,
        label="config.assets.echojepa_source_revision",
    )
    checkpoint = _validate_hex(
        assets.get("functional_checkpoint_sha256"),
        64,
        label="config.assets.functional_checkpoint_sha256",
    )
    return {
        "protocol": protocol,
        "assets": {
            "echojepa_source_revision": revision,
            "functional_checkpoint_sha256": checkpoint,
        },
    }


def _file_record(path: Path, *, relative_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise CompletionError(f"missing required artifact: {path}")
    if path.is_symlink():
        raise CompletionError(f"required artifact must not be a symlink: {path}")
    try:
        before = path.stat()
        if before.st_size <= 0:
            raise CompletionError(f"required artifact is empty: {path}")
        digest = sha256_file(path)
        after = path.stat()
    except OSError as exc:
        raise CompletionError(f"cannot hash required artifact {path}: {exc}") from exc
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise CompletionError(f"required artifact changed while hashing: {path}")
    return {
        "path": relative_path,
        "size_bytes": before.st_size,
        "sha256": digest,
    }


def _is_link_or_junction(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())
    except OSError as exc:
        raise CompletionError(f"cannot inspect path linkage: {path}: {exc}") from exc


def _reject_link(path: Path, *, label: str) -> None:
    if _is_link_or_junction(path):
        raise CompletionError(f"{label} must not be a symlink or junction: {path}")


def _configured_path(
    config: dict[str, Any],
    key: str,
    *,
    label: str,
    kind: str,
) -> Path:
    paths = config.get("paths")
    raw = paths.get(key) if isinstance(paths, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        raise CompletionError(f"config.paths.{key} must be a non-empty path string")
    unresolved = Path(raw)
    if not unresolved.is_absolute():
        raise CompletionError(f"config.paths.{key} must be absolute")
    _reject_link(unresolved, label=label)
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise CompletionError(f"cannot resolve {label}: {unresolved}: {exc}") from exc
    if kind == "file" and not resolved.is_file():
        raise CompletionError(f"{label} must be a regular file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise CompletionError(f"{label} must be a directory: {resolved}")
    return resolved


def _directory_file_paths(
    root: Path,
    *,
    label: str,
    excluded_directories: set[str],
    excluded_suffixes: set[str],
) -> list[Path]:
    _reject_link(root, label=label)
    if not root.is_dir():
        raise CompletionError(f"{label} is not a directory: {root}")
    paths: list[Path] = []
    try:
        for current_name, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            current = Path(current_name)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                path = current / name
                _reject_link(path, label=f"{label} directory")
                if name not in excluded_directories:
                    kept_directories.append(name)
            directory_names[:] = kept_directories
            for name in sorted(file_names):
                path = current / name
                _reject_link(path, label=f"{label} file")
                if name == ".git" or path.suffix.lower() in excluded_suffixes:
                    continue
                if not path.is_file():
                    raise CompletionError(f"{label} contains a non-regular file: {path}")
                paths.append(path)
    except OSError as exc:
        raise CompletionError(f"cannot enumerate {label}: {root}: {exc}") from exc
    paths.sort(key=lambda path: path.relative_to(root).as_posix())
    if not paths:
        raise CompletionError(f"{label} contains no bindable files: {root}")
    return paths


def _content_tree_record(
    root: Path,
    *,
    label: str,
    recorded_root: str,
    excluded_directories: set[str] | None = None,
    excluded_suffixes: set[str] | None = None,
) -> dict[str, Any]:
    excluded_directories = excluded_directories or set()
    excluded_suffixes = excluded_suffixes or set()
    paths = _directory_file_paths(
        root,
        label=label,
        excluded_directories=excluded_directories,
        excluded_suffixes=excluded_suffixes,
    )
    relative_paths = [path.relative_to(root).as_posix() for path in paths]
    digest = hashlib.sha256()
    digest.update(b"HEMFM-CONTENT-TREE-SHA256-V1\0")
    total_size = 0
    for path, relative in zip(paths, relative_paths, strict=True):
        try:
            before = path.stat()
            encoded = relative.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(before.st_size.to_bytes(8, "big"))
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
            after = path.stat()
        except OSError as exc:
            raise CompletionError(f"cannot hash {label} file {path}: {exc}") from exc
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CompletionError(f"{label} file changed while hashing: {path}")
        total_size += before.st_size
    final_paths = _directory_file_paths(
        root,
        label=label,
        excluded_directories=excluded_directories,
        excluded_suffixes=excluded_suffixes,
    )
    if [path.relative_to(root).as_posix() for path in final_paths] != relative_paths:
        raise CompletionError(f"{label} membership changed while hashing")
    return {
        "algorithm": "sha256-content-tree-v1",
        "root": recorded_root,
        "file_count": len(paths),
        "size_bytes": total_size,
        "sha256": digest.hexdigest(),
    }


def _git_output(checkout: Path, arguments: list[str], *, label: str) -> bytes:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={checkout.as_posix()}",
                *arguments,
            ],
            cwd=checkout,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CompletionError(f"cannot inspect {label}: {exc}") from exc
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise CompletionError(f"cannot inspect {label}: {message or 'git failed'}")
    return result.stdout


def _git_checkout_state(checkout: Path) -> dict[str, Any]:
    head = _validate_hex(
        _git_output(checkout, ["rev-parse", "HEAD"], label="EchoJEPA HEAD")
        .decode("ascii")
        .strip(),
        40,
        label="EchoJEPA HEAD",
    )
    head_tree = _validate_hex(
        _git_output(
            checkout, ["rev-parse", "HEAD^{tree}"], label="EchoJEPA HEAD tree"
        )
        .decode("ascii")
        .strip(),
        40,
        label="EchoJEPA HEAD tree",
    )
    status = _git_output(
        checkout,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        label="EchoJEPA worktree status",
    )
    return {
        "head_revision": head,
        "head_tree": head_tree,
        "dirty": bool(status),
        "status_size_bytes": len(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
    }


def _functional_checkpoint_record(
    config: dict[str, Any], *, expected_sha256: str
) -> dict[str, Any]:
    checkpoint = _configured_path(
        config,
        "functional_checkpoint",
        label="functional checkpoint",
        kind="file",
    )
    record = _file_record(checkpoint, relative_path="functional_checkpoint")
    if record["sha256"] != expected_sha256:
        raise CompletionError(
            "functional checkpoint SHA-256 does not match "
            "config.assets.functional_checkpoint_sha256"
        )
    return record


def _echojepa_checkout_record(
    config: dict[str, Any], *, expected_revision: str
) -> dict[str, Any]:
    checkout = _configured_path(
        config,
        "echojepa_source",
        label="EchoJEPA checkout",
        kind="directory",
    )
    top_level = Path(
        _git_output(
            checkout, ["rev-parse", "--show-toplevel"], label="EchoJEPA checkout root"
        )
        .decode("utf-8")
        .strip()
    ).resolve()
    if os.path.normcase(str(top_level)) != os.path.normcase(str(checkout)):
        raise CompletionError(
            "config.paths.echojepa_source must be the EchoJEPA Git checkout root"
        )
    before = _git_checkout_state(checkout)
    if before["head_revision"] != expected_revision:
        raise CompletionError(
            "EchoJEPA HEAD does not match config.assets.echojepa_source_revision"
        )
    content_tree = _content_tree_record(
        checkout,
        label="EchoJEPA checkout",
        recorded_root="echojepa_source",
        excluded_directories={".git", "__pycache__"},
        excluded_suffixes={".pyc", ".pyo"},
    )
    after = _git_checkout_state(checkout)
    if before != after:
        raise CompletionError("EchoJEPA Git state changed while hashing")
    return {
        **before,
        "expected_revision": expected_revision,
        "content_tree": content_tree,
    }


def _source_paths(repository: Path) -> tuple[Path, list[Path]]:
    source_root = repository / "src" / "hemfm"
    if not source_root.is_dir():
        raise CompletionError(f"repository source tree is missing: {source_root}")
    paths = sorted(
        (
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and "__pycache__" not in path.parts
            and path.suffix.lower() not in {".pyc", ".pyo"}
        ),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not paths:
        raise CompletionError(f"repository source tree is empty: {source_root}")
    return source_root, paths


def _source_tree_record(repository: Path) -> dict[str, Any]:
    source_root, paths = _source_paths(repository)
    relative_paths = [path.relative_to(source_root).as_posix() for path in paths]
    digest = hashlib.sha256()
    digest.update(b"HEMFM-SOURCE-TREE-SHA256-V1\0")
    for path, relative in zip(paths, relative_paths, strict=True):
        if path.is_symlink():
            raise CompletionError(f"source file became a symlink while hashing: {path}")
        try:
            before = path.stat()
            name = relative.encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(before.st_size.to_bytes(8, "big"))
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
            after = path.stat()
        except OSError as exc:
            raise CompletionError(f"cannot hash source file {path}: {exc}") from exc
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise CompletionError(f"source file changed while hashing: {path}")
    _, final_paths = _source_paths(repository)
    final_relative = [path.relative_to(source_root).as_posix() for path in final_paths]
    if final_relative != relative_paths:
        raise CompletionError("repository source-tree membership changed while hashing")
    return {
        "algorithm": "sha256-tree-v1",
        "root": "src/hemfm",
        "file_count": len(paths),
        "sha256": digest.hexdigest(),
    }


def _git_head(repository: Path) -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "rev-parse",
                "--verify",
                "HEAD",
            ],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = result.stdout.strip().lower()
    if result.returncode != 0 or len(head) != 40:
        return None
    if any(character not in "0123456789abcdef" for character in head):
        return None
    return head


def _stage_evidence_path(config: dict[str, Any], *, mode: str) -> Path:
    paths = config.get("paths")
    if not isinstance(paths, dict) or not isinstance(paths.get("evidence_root"), str):
        raise CompletionError("config.paths.evidence_root must be a path string")
    name = (
        "echonet_dynamic_stage.json"
        if mode == "full"
        else "echonet_dynamic_stage_smoke.json"
    )
    return Path(paths["evidence_root"]) / "G4" / name


def _stage_stable_binding(report: dict[str, Any], *, mode: str) -> dict[str, Any]:
    _require_exact_bool(report, "passed", True, label="EchoNet stage evidence")
    _require_exact_bool(report, "training_ready", True, label="EchoNet stage evidence")
    _require_exact_bool(
        report,
        "official_test_videos_accessed",
        False,
        label="EchoNet stage evidence",
    )
    _require_exact_bool(
        report,
        "locked_mimic_test_accessed",
        False,
        label="EchoNet stage evidence",
    )
    if report.get("mode") != mode:
        raise CompletionError("EchoNet stage evidence mode mismatch")
    archive_sha256 = _validate_hex(
        report.get("archive_sha256"), 64, label="EchoNet stage evidence.archive_sha256"
    )
    development_records = _require_int(
        report.get("development_records"),
        label="EchoNet stage evidence.development_records",
        minimum=1,
    )
    split_counts = report.get("split_counts")
    if not isinstance(split_counts, dict) or set(split_counts) != {"TRAIN", "VAL"}:
        raise CompletionError(
            "EchoNet stage evidence.split_counts must contain TRAIN and VAL"
        )
    stable_split_counts = {
        split: _require_int(
            split_counts[split],
            label=f"EchoNet stage evidence.split_counts.{split}",
            minimum=1,
        )
        for split in ("TRAIN", "VAL")
    }
    if sum(stable_split_counts.values()) != development_records:
        raise CompletionError(
            "EchoNet stage split counts do not match development_records"
        )
    errors = report.get("errors")
    if not isinstance(errors, list) or errors:
        raise CompletionError("EchoNet stage evidence.errors must be an empty array")
    source = report.get("source")
    trace_contents = report.get("trace_cache_contents")
    track = report.get("track")
    if not isinstance(source, str) or not source:
        raise CompletionError(
            "EchoNet stage evidence.source must be a non-empty string"
        )
    if not isinstance(trace_contents, str) or not trace_contents:
        raise CompletionError(
            "EchoNet stage evidence.trace_cache_contents must be a non-empty string"
        )
    if track != "R":
        raise CompletionError("EchoNet stage evidence.track must be R")
    return {
        "passed": True,
        "mode": mode,
        "training_ready": True,
        "source": source,
        "archive_sha256": archive_sha256,
        "development_records": development_records,
        "split_counts": stable_split_counts,
        "frames": _require_int(
            report.get("frames"), label="EchoNet stage evidence.frames", minimum=1
        ),
        "resolution": _require_int(
            report.get("resolution"),
            label="EchoNet stage evidence.resolution",
            minimum=1,
        ),
        "trace_annotated_records": _require_int(
            report.get("trace_annotated_records"),
            label="EchoNet stage evidence.trace_annotated_records",
            minimum=1,
        ),
        "trace_frames_cached": _require_int(
            report.get("trace_frames_cached"),
            label="EchoNet stage evidence.trace_frames_cached",
            minimum=1,
        ),
        "trace_cache_contents": trace_contents,
        "official_test_records_reserved": _require_int(
            report.get("official_test_records_reserved"),
            label="EchoNet stage evidence.official_test_records_reserved",
            minimum=1,
        ),
        "official_test_videos_accessed": False,
        "locked_mimic_test_accessed": False,
        "errors": [],
        "track": track,
    }


def _development_manifest_path(config: dict[str, Any]) -> Path:
    staging_root = _configured_path(
        config,
        "local_staging_root",
        label="local staging root",
        kind="directory",
    )
    return (
        staging_root
        / "datasets"
        / "research"
        / "echonet-dynamic"
        / "development_manifest_private.csv"
    )


def _path_within_root(
    raw_path: Path, root: Path, *, label: str
) -> tuple[Path, Path]:
    if not raw_path.is_absolute():
        raise CompletionError(f"{label} must be an absolute path")
    if ".." in raw_path.parts:
        raise CompletionError(f"{label} must not contain parent traversal")
    raw_absolute = Path(os.path.abspath(raw_path))
    root_absolute = Path(os.path.abspath(root))
    try:
        common = Path(os.path.commonpath([str(raw_absolute), str(root_absolute)]))
    except ValueError as exc:
        raise CompletionError(f"{label} is outside the staged cache root") from exc
    if os.path.normcase(str(common)) != os.path.normcase(str(root_absolute)):
        raise CompletionError(f"{label} escapes the staged cache root")
    relative_text = os.path.relpath(raw_absolute, root_absolute)
    relative = Path(relative_text)
    cursor = root_absolute
    _reject_link(cursor, label="staged cache root")
    for part in relative.parts:
        cursor = cursor / part
        _reject_link(cursor, label=label)
    try:
        resolved = raw_absolute.resolve(strict=True)
        resolved_relative = resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise CompletionError(f"{label} escapes or is missing from the staged cache") from exc
    if not resolved.is_file():
        raise CompletionError(f"{label} must be a regular file: {resolved}")
    return resolved, resolved_relative


def _referenced_cache_record(
    config: dict[str, Any], *, stable_stage: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = _development_manifest_path(config)
    _reject_link(manifest_path, label="EchoNet development manifest")
    manifest_before = _file_record(
        manifest_path, relative_path="development_manifest_private.csv"
    )
    cache_root = manifest_path.parent / "cache_16x224"
    _reject_link(cache_root, label="staged EchoNet cache root")
    if not cache_root.is_dir():
        raise CompletionError(f"staged EchoNet cache root is missing: {cache_root}")

    references: list[tuple[str, Path]] = []
    seen_paths: set[str] = set()
    seen_file_names: set[str] = set()
    seen_file_identities: set[tuple[int, int]] = set()
    split_counts = {"TRAIN": 0, "VAL": 0}
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            if not header:
                raise CompletionError("EchoNet development manifest has no header")
            if len(header) != len(set(header)):
                raise CompletionError(
                    "EchoNet development manifest has duplicate columns"
                )
            required = {"FileName", "Split", "cache_path"}
            missing = sorted(required - set(header))
            if missing:
                raise CompletionError(
                    "EchoNet development manifest is missing columns: "
                    + ", ".join(missing)
                )
            for row_number, row in enumerate(reader, start=2):
                file_name = row["FileName"].strip()
                if (
                    not file_name
                    or Path(file_name).name != file_name
                    or file_name.casefold() in seen_file_names
                ):
                    raise CompletionError(
                        f"EchoNet development manifest row {row_number} has an invalid or duplicate FileName"
                    )
                seen_file_names.add(file_name.casefold())
                split = row["Split"].strip().upper()
                if split not in split_counts:
                    raise CompletionError(
                        f"EchoNet development manifest row {row_number} has invalid Split"
                    )
                raw_cache = row["cache_path"].strip()
                if not raw_cache:
                    raise CompletionError(
                        f"EchoNet development manifest row {row_number} has an empty cache_path"
                    )
                cache_path, relative = _path_within_root(
                    Path(raw_cache),
                    cache_root,
                    label=f"EchoNet cache reference at row {row_number}",
                )
                if cache_path.suffix.lower() != ".npz":
                    raise CompletionError(
                        f"EchoNet cache reference at row {row_number} must be an NPZ"
                    )
                if cache_path.stem.casefold() != file_name.casefold():
                    raise CompletionError(
                        f"EchoNet cache reference at row {row_number} does not match FileName"
                    )
                if not relative.parts or relative.parts[0].casefold() != split.casefold():
                    raise CompletionError(
                        f"EchoNet cache reference at row {row_number} does not match Split"
                    )
                normalized = os.path.normcase(str(cache_path))
                if normalized in seen_paths:
                    raise CompletionError(
                        f"EchoNet development manifest duplicates cache reference {cache_path}"
                    )
                seen_paths.add(normalized)
                stat_result = cache_path.stat()
                file_identity = (stat_result.st_dev, stat_result.st_ino)
                if stat_result.st_ino and file_identity in seen_file_identities:
                    raise CompletionError(
                        "EchoNet development manifest references the same cache file identity more than once"
                    )
                if stat_result.st_ino:
                    seen_file_identities.add(file_identity)
                references.append((relative.as_posix(), cache_path))
                split_counts[split] += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CompletionError(
            f"cannot read EchoNet development manifest {manifest_path}: {exc}"
        ) from exc

    expected_records = stable_stage["development_records"]
    if len(references) != expected_records:
        raise CompletionError(
            "EchoNet development manifest row count does not match stage evidence"
        )
    if split_counts != stable_stage["split_counts"]:
        raise CompletionError(
            "EchoNet development manifest split counts do not match stage evidence"
        )
    references.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    digest.update(b"HEMFM-REFERENCED-ECHONET-CACHE-SHA256-V1\0")
    total_size = 0
    for relative, path in references:
        _reject_link(path, label="referenced EchoNet cache artifact")
        try:
            before = path.stat()
            if before.st_size <= 0:
                raise CompletionError(f"referenced EchoNet cache is empty: {path}")
            encoded = relative.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            digest.update(before.st_size.to_bytes(8, "big"))
            with path.open("rb") as handle:
                while block := handle.read(1024 * 1024):
                    digest.update(block)
            after = path.stat()
        except OSError as exc:
            raise CompletionError(
                f"cannot hash referenced EchoNet cache {path}: {exc}"
            ) from exc
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CompletionError(
                f"referenced EchoNet cache changed while hashing: {path}"
            )
        total_size += before.st_size
    manifest_after = _file_record(
        manifest_path, relative_path="development_manifest_private.csv"
    )
    if manifest_before != manifest_after:
        raise CompletionError(
            "EchoNet development manifest changed while hashing referenced caches"
        )
    return manifest_before, {
        "algorithm": "sha256-referenced-cache-v1",
        "root": "cache_16x224",
        "file_count": len(references),
        "size_bytes": total_size,
        "split_counts": split_counts,
        "sha256": digest.hexdigest(),
    }


def _provenance_record(
    *, config_path: Path, config: dict[str, Any], repository: Path, mode: str
) -> dict[str, Any]:
    identity = _config_identity(config)
    stage_path = _stage_evidence_path(config, mode=mode)
    stage_report = _read_json(stage_path, label="EchoNet stage evidence")
    stable_stage = _stage_stable_binding(stage_report, mode=mode)
    development_manifest, referenced_cache = _referenced_cache_record(
        config, stable_stage=stable_stage
    )
    stage_record = {
        "observation": _file_record(stage_path, relative_path=f"G4/{stage_path.name}"),
        "stable_binding": stable_stage,
        "development_manifest": development_manifest,
        "referenced_cache": referenced_cache,
    }
    config_record = _file_record(config_path, relative_path=config_path.name)
    return {
        "config": config_record,
        "source_tree": _source_tree_record(repository),
        "functional_checkpoint": _functional_checkpoint_record(
            config,
            expected_sha256=identity["assets"]["functional_checkpoint_sha256"],
        ),
        "echojepa_checkout": _echojepa_checkout_record(
            config,
            expected_revision=identity["assets"]["echojepa_source_revision"],
        ),
        "echonet_stage_evidence": stage_record,
        **identity,
        "git_head": _git_head(repository),
    }


def _validate_file_binding(
    value: Any, *, label: str, expected_path: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "size_bytes", "sha256"}:
        raise CompletionError(f"{label} must be a complete file record")
    if value.get("path") != expected_path:
        raise CompletionError(f"{label}.path must be {expected_path!r}")
    _require_int(value.get("size_bytes"), label=f"{label}.size_bytes", minimum=1)
    _validate_hex(value.get("sha256"), 64, label=f"{label}.sha256")
    return value


def _validate_tree_binding(
    value: Any, *, label: str, algorithm: str, root: str
) -> dict[str, Any]:
    required = {"algorithm", "root", "file_count", "size_bytes", "sha256"}
    if not isinstance(value, dict) or set(value) != required:
        raise CompletionError(f"{label} must be a complete content-tree record")
    if value.get("algorithm") != algorithm or value.get("root") != root:
        raise CompletionError(f"{label} has the wrong algorithm or root")
    _require_int(value.get("file_count"), label=f"{label}.file_count", minimum=1)
    _require_int(value.get("size_bytes"), label=f"{label}.size_bytes", minimum=0)
    _validate_hex(value.get("sha256"), 64, label=f"{label}.sha256")
    return value


def _validate_echojepa_binding(value: Any, *, label: str) -> dict[str, Any]:
    required = {
        "head_revision",
        "head_tree",
        "dirty",
        "status_size_bytes",
        "status_sha256",
        "expected_revision",
        "content_tree",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise CompletionError(f"{label} must be a complete EchoJEPA checkout record")
    head = _validate_hex(value.get("head_revision"), 40, label=f"{label}.head_revision")
    expected = _validate_hex(
        value.get("expected_revision"), 40, label=f"{label}.expected_revision"
    )
    if head != expected:
        raise CompletionError(f"{label} HEAD does not match its expected revision")
    _validate_hex(value.get("head_tree"), 40, label=f"{label}.head_tree")
    _require_exact_bool(value, "dirty", value.get("dirty"), label=label)
    _require_int(
        value.get("status_size_bytes"),
        label=f"{label}.status_size_bytes",
        minimum=0,
    )
    _validate_hex(value.get("status_sha256"), 64, label=f"{label}.status_sha256")
    _validate_tree_binding(
        value.get("content_tree"),
        label=f"{label}.content_tree",
        algorithm="sha256-content-tree-v1",
        root="echojepa_source",
    )
    return value


def _validate_cache_binding(value: Any, *, label: str) -> dict[str, Any]:
    required = {
        "algorithm",
        "root",
        "file_count",
        "size_bytes",
        "split_counts",
        "sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise CompletionError(f"{label} must be a complete referenced-cache record")
    if value.get("algorithm") != "sha256-referenced-cache-v1":
        raise CompletionError(f"{label} has the wrong algorithm")
    if value.get("root") != "cache_16x224":
        raise CompletionError(f"{label} has the wrong root")
    count = _require_int(
        value.get("file_count"), label=f"{label}.file_count", minimum=1
    )
    _require_int(value.get("size_bytes"), label=f"{label}.size_bytes", minimum=1)
    split_counts = value.get("split_counts")
    if not isinstance(split_counts, dict) or set(split_counts) != {"TRAIN", "VAL"}:
        raise CompletionError(f"{label}.split_counts must contain TRAIN and VAL")
    normalized_counts = {
        split: _require_int(
            split_counts[split], label=f"{label}.split_counts.{split}", minimum=1
        )
        for split in ("TRAIN", "VAL")
    }
    if sum(normalized_counts.values()) != count:
        raise CompletionError(f"{label}.split_counts do not match file_count")
    _validate_hex(value.get("sha256"), 64, label=f"{label}.sha256")
    return value


def _stable_provenance_view(provenance: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise CompletionError(f"{label} must be an object")
    stage = provenance.get("echonet_stage_evidence")
    if not isinstance(stage, dict):
        raise CompletionError(f"{label}.echonet_stage_evidence must be an object")
    observation = stage.get("observation")
    if not isinstance(observation, dict):
        raise CompletionError(
            f"{label}.echonet_stage_evidence.observation must be an object"
        )
    _validate_hex(
        observation.get("sha256"),
        64,
        label=f"{label}.echonet_stage_evidence.observation.sha256",
    )
    _require_int(
        observation.get("size_bytes"),
        label=f"{label}.echonet_stage_evidence.observation.size_bytes",
        minimum=1,
    )
    git_head = provenance.get("git_head")
    if git_head is not None:
        _validate_hex(git_head, 40, label=f"{label}.git_head")
    functional_checkpoint = _validate_file_binding(
        provenance.get("functional_checkpoint"),
        label=f"{label}.functional_checkpoint",
        expected_path="functional_checkpoint",
    )
    echojepa_checkout = _validate_echojepa_binding(
        provenance.get("echojepa_checkout"),
        label=f"{label}.echojepa_checkout",
    )
    referenced_cache = _validate_cache_binding(
        stage.get("referenced_cache"),
        label=f"{label}.echonet_stage_evidence.referenced_cache",
    )
    return {
        "config": provenance.get("config"),
        "source_tree": provenance.get("source_tree"),
        "functional_checkpoint": functional_checkpoint,
        "echojepa_checkout": echojepa_checkout,
        "echonet_stage_stable_binding": stage.get("stable_binding"),
        "development_manifest": stage.get("development_manifest"),
        "referenced_cache": referenced_cache,
        "protocol": provenance.get("protocol"),
        "assets": provenance.get("assets"),
    }


def _read_provenance_snapshot(path: Path) -> dict[str, Any]:
    snapshot = _read_json(path, label="launch provenance snapshot")
    if (
        _require_int(
            snapshot.get("schema_version"), label="launch provenance.schema_version"
        )
        != 1
    ):
        raise CompletionError("unsupported launch provenance schema")
    _parse_utc(snapshot.get("created_utc"), label="launch provenance.created_utc")
    _stable_provenance_view(
        snapshot.get("provenance"), label="launch provenance.provenance"
    )
    return snapshot


def _assert_snapshot_matches_current(
    snapshot: dict[str, Any], current: dict[str, Any]
) -> None:
    recorded_view = _stable_provenance_view(
        snapshot.get("provenance"), label="launch provenance.provenance"
    )
    current_view = _stable_provenance_view(current, label="current provenance")
    if recorded_view != current_view:
        raise CompletionError(
            "launch provenance does not match current config/source/cache"
        )


def snapshot_provenance(
    *, config_path: str | Path, repository: str | Path, output: str | Path
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    repository_path = Path(repository).resolve()
    config = _load_config(config_file)
    output_path = Path(output).resolve()
    snapshot = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": _provenance_record(
            config_path=config_file,
            config=config,
            repository=repository_path,
            mode="full",
        ),
    }
    _atomic_json(output_path, snapshot)
    return {
        "ok": True,
        "action": "snapshot-provenance",
        "output": str(output_path),
        "snapshot_sha256": sha256_file(output_path),
    }


def verify_provenance(
    *, config_path: str | Path, repository: str | Path, input_path: str | Path
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    repository_path = Path(repository).resolve()
    snapshot_path = Path(input_path).resolve()
    config = _load_config(config_file)
    snapshot = _read_provenance_snapshot(snapshot_path)
    current = _provenance_record(
        config_path=config_file,
        config=config,
        repository=repository_path,
        mode="full",
    )
    _assert_snapshot_matches_current(snapshot, current)
    return {
        "ok": True,
        "action": "verify-provenance",
        "input": str(snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
    }


def _seed_directory(artifact_root: Path, *, seed: int, mode: str) -> Path:
    return artifact_root / f"seed_{seed}_{mode}"


def _artifact_records(seed_directory: Path, *, kind: str) -> dict[str, Any]:
    names = {**ARTIFACT_NAMES, "prediction": PREDICTION_NAMES[kind]}
    return {
        key: _file_record(seed_directory / name, relative_path=name)
        for key, name in names.items()
    }


def _validate_seed_report(
    report: dict[str, Any], *, seed: int, mode: str, requested_epochs: int
) -> dict[str, Any]:
    _require_exact_bool(report, "passed", True, label="report")
    _require_exact_bool(report, "official_test_videos_accessed", False, label="report")
    _require_exact_bool(report, "locked_test_accessed", False, label="report")
    if _require_int(report.get("seed"), label="report.seed") != seed:
        raise CompletionError(f"report.seed does not match requested seed {seed}")
    if "mode" in report and report["mode"] != mode:
        raise CompletionError(f"report.mode does not match requested mode {mode}")

    completed = _require_int(
        report.get("epochs_completed"), label="report.epochs_completed", minimum=1
    )
    if completed > requested_epochs:
        raise CompletionError("report.epochs_completed exceeds requested epochs")
    history = report.get("history")
    if not isinstance(history, list) or len(history) != completed:
        raise CompletionError(
            "report.history length must equal report.epochs_completed"
        )
    stages: list[str] = []
    previous_stage_index = -1
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, dict):
            raise CompletionError(
                f"report.history[{expected_epoch - 1}] must be an object"
            )
        epoch = _require_int(
            row.get("epoch"), label=f"report.history[{expected_epoch - 1}].epoch"
        )
        if epoch != expected_epoch:
            raise CompletionError(
                "report.history epochs must be contiguous and one-based"
            )
        stage = row.get("stage")
        if stage not in REQUIRED_STAGES:
            raise CompletionError(f"report history contains invalid stage {stage!r}")
        stage_index = REQUIRED_STAGES.index(stage)
        if stage_index < previous_stage_index:
            raise CompletionError("report history training stages regress")
        previous_stage_index = stage_index
        stages.append(stage)
    covered_stages = tuple(dict.fromkeys(stages))
    if covered_stages != REQUIRED_STAGES:
        raise CompletionError(
            "report history must cover frozen, peft, and selective_unfreeze in order"
        )

    best_epoch = _require_int(
        report.get("best_epoch"), label="report.best_epoch", minimum=1
    )
    if best_epoch > completed:
        raise CompletionError("report.best_epoch exceeds completed epochs")
    if completed < requested_epochs and completed - best_epoch < 4:
        raise CompletionError(
            "report stopped before requested epochs without four post-best epochs"
        )
    return {
        "passed": True,
        "official_test_videos_accessed": False,
        "locked_test_accessed": False,
        "epochs_completed": completed,
        "best_epoch": best_epoch,
        "stages": list(covered_stages),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def record_seed_completion(
    *,
    kind: str,
    artifact_root: str | Path,
    seed: int,
    mode: str,
    requested_epochs: int,
    config_path: str | Path,
    repository: str | Path,
    not_before: str | datetime,
    launch_provenance: str | Path,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise CompletionError(f"unsupported artifact kind: {kind}")
    if mode not in MODES:
        raise CompletionError(f"unsupported mode: {mode}")
    if type(seed) is not int:
        raise CompletionError("seed must be an integer")
    if type(requested_epochs) is not int or requested_epochs < 1:
        raise CompletionError("requested_epochs must be a positive integer")
    threshold = (
        _parse_utc(not_before, label="not-before")
        if isinstance(not_before, str)
        else not_before.astimezone(timezone.utc)
    )
    config_file = Path(config_path).resolve()
    repository_path = Path(repository).resolve()
    launch_path = Path(launch_provenance).resolve()
    config = _load_config(config_file)
    launch_snapshot = _read_provenance_snapshot(launch_path)
    current_provenance = _provenance_record(
        config_path=config_file,
        config=config,
        repository=repository_path,
        mode=mode,
    )
    _assert_snapshot_matches_current(launch_snapshot, current_provenance)
    seed_directory = _seed_directory(
        Path(artifact_root).resolve(), seed=seed, mode=mode
    )
    report_path = seed_directory / ARTIFACT_NAMES["report"]
    _assert_fresh(report_path, threshold, label="seed report")
    report = _read_json(report_path, label="seed report")
    summary = _validate_seed_report(
        report, seed=seed, mode=mode, requested_epochs=requested_epochs
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "seed": seed,
        "mode": mode,
        "requested_epochs": requested_epochs,
        "report_summary": summary,
        "artifacts": _artifact_records(seed_directory, kind=kind),
        "provenance": current_provenance,
        "launch_provenance": launch_snapshot,
        "launch_provenance_file": _file_record(
            launch_path, relative_path=str(launch_path)
        ),
    }
    manifest_path = seed_directory / MANIFEST_NAME
    _atomic_json(manifest_path, manifest)
    return {
        "ok": True,
        "action": "record-seed",
        "kind": kind,
        "seed": seed,
        "mode": mode,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def verify_seed_completion(
    *,
    kind: str,
    artifact_root: str | Path,
    seed: int,
    mode: str,
    requested_epochs: int,
    config_path: str | Path,
    repository: str | Path,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise CompletionError(f"unsupported artifact kind: {kind}")
    if mode not in MODES:
        raise CompletionError(f"unsupported mode: {mode}")
    if type(seed) is not int:
        raise CompletionError("seed must be an integer")
    if type(requested_epochs) is not int or requested_epochs < 1:
        raise CompletionError("requested_epochs must be a positive integer")
    config_file = Path(config_path).resolve()
    repository_path = Path(repository).resolve()
    config = _load_config(config_file)
    seed_directory = _seed_directory(
        Path(artifact_root).resolve(), seed=seed, mode=mode
    )
    manifest_path = seed_directory / MANIFEST_NAME
    manifest = _read_json(manifest_path, label="completion manifest")

    if (
        _require_int(manifest.get("schema_version"), label="manifest.schema_version")
        != 1
    ):
        raise CompletionError("unsupported completion manifest schema")
    _parse_utc(manifest.get("created_utc"), label="manifest.created_utc")
    expected_identity = {
        "kind": kind,
        "seed": seed,
        "mode": mode,
        "requested_epochs": requested_epochs,
    }
    for key, expected in expected_identity.items():
        value = manifest.get(key)
        if key in {"seed", "requested_epochs"}:
            _require_int(value, label=f"manifest.{key}")
        if value != expected:
            raise CompletionError(f"manifest.{key} does not match current request")

    report = _read_json(seed_directory / ARTIFACT_NAMES["report"], label="seed report")
    current_summary = _validate_seed_report(
        report, seed=seed, mode=mode, requested_epochs=requested_epochs
    )
    if manifest.get("report_summary") != current_summary:
        raise CompletionError("completion manifest report summary mismatch")
    current_artifacts = _artifact_records(seed_directory, kind=kind)
    if manifest.get("artifacts") != current_artifacts:
        raise CompletionError("completion manifest artifact hash or size mismatch")
    current_provenance = _provenance_record(
        config_path=config_file,
        config=config,
        repository=repository_path,
        mode=mode,
    )
    recorded_view = _stable_provenance_view(
        manifest.get("provenance"), label="completion manifest.provenance"
    )
    current_view = _stable_provenance_view(
        current_provenance, label="current provenance"
    )
    if recorded_view != current_view:
        raise CompletionError("completion manifest provenance mismatch")
    launch_snapshot = manifest.get("launch_provenance")
    if not isinstance(launch_snapshot, dict):
        raise CompletionError("completion manifest.launch_provenance must be an object")
    _assert_snapshot_matches_current(launch_snapshot, current_provenance)
    launch_file_record = manifest.get("launch_provenance_file")
    if not isinstance(launch_file_record, dict):
        raise CompletionError(
            "completion manifest.launch_provenance_file must be an object"
        )
    launch_file_value = launch_file_record.get("path")
    if not isinstance(launch_file_value, str) or not launch_file_value:
        raise CompletionError(
            "completion manifest launch provenance path must be a string"
        )
    launch_path = Path(launch_file_value)
    current_launch_file = _file_record(launch_path, relative_path=launch_file_value)
    if current_launch_file != launch_file_record:
        raise CompletionError("completion manifest launch provenance file mismatch")
    external_snapshot = _read_provenance_snapshot(launch_path)
    if external_snapshot != launch_snapshot:
        raise CompletionError(
            "embedded launch provenance does not match its source file"
        )
    return {
        "ok": True,
        "action": "verify-seed",
        "kind": kind,
        "seed": seed,
        "mode": mode,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _configured_seeds(config: dict[str, Any]) -> list[int]:
    training = config.get("training")
    values = training.get("seeds") if isinstance(training, dict) else None
    if not isinstance(values, list) or len(values) != 3:
        raise CompletionError("config.training.seeds must contain exactly three seeds")
    seeds = [_require_int(value, label="config.training.seeds[]") for value in values]
    if len(set(seeds)) != 3:
        raise CompletionError("config.training.seeds must be unique")
    return seeds


def _matched_evidence_path(evidence_root: Path, *, target: str, mode: str) -> Path:
    g6 = (
        evidence_root if evidence_root.name.casefold() == "g6" else evidence_root / "G6"
    )
    return g6 / f"staged_final_{target.lower()}_{mode}_matched_uncertainty.json"


def _resolve_private_csv(report: dict[str, Any], config: dict[str, Any]) -> Path:
    raw = report.get("private_predictions")
    if not isinstance(raw, str) or not raw.strip():
        raise CompletionError("matched audit.private_predictions must be a path string")
    path = Path(raw)
    paths = config.get("paths")
    private_root_value = paths.get("private_root") if isinstance(paths, dict) else None
    if not isinstance(private_root_value, str) or not private_root_value:
        raise CompletionError("config.paths.private_root must be a path string")
    private_root = Path(private_root_value).resolve()
    resolved = path.resolve() if path.is_absolute() else (private_root / path).resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise CompletionError(
            "matched audit private CSV escapes config.paths.private_root"
        ) from exc
    if resolved.suffix.lower() != ".csv":
        raise CompletionError(
            "matched audit private prediction reference must be a CSV"
        )
    return resolved


def _verify_matched_csv(
    path: Path, *, target: str, seeds: list[int], expected_rows: int
) -> int:
    if not path.is_file() or path.is_symlink():
        raise CompletionError(f"missing referenced private CSV: {path}")
    numeric_columns = {
        "value",
        "prediction",
        "aleatoric_sigma",
        "ensemble_dispersion",
        "total_uncertainty",
        "absolute_error",
        "risk_rank_percentile",
        *{f"prediction_seed_{seed}" for seed in seeds},
        *{f"aleatoric_sigma_seed_{seed}" for seed in seeds},
    }
    required = {
        "sample_id",
        "subject_id",
        "study_id",
        "target",
        "large_residual",
        *numeric_columns,
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            if not header:
                raise CompletionError("referenced private CSV has no header")
            if len(header) != len(set(header)):
                raise CompletionError("referenced private CSV has duplicate columns")
            missing = sorted(required - set(header))
            if missing:
                raise CompletionError(
                    f"referenced private CSV is missing columns: {', '.join(missing)}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CompletionError(
            f"cannot read referenced private CSV {path}: {exc}"
        ) from exc
    if len(rows) != expected_rows:
        raise CompletionError(
            f"referenced private CSV row count {len(rows)} does not match {expected_rows}"
        )
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        if row["target"] != target:
            raise CompletionError(
                f"referenced private CSV row {index} has the wrong target"
            )
        identity = (row["sample_id"].strip(), row["subject_id"].strip())
        if not all((identity[0], identity[1], row["study_id"].strip())):
            raise CompletionError(
                f"referenced private CSV row {index} has an empty identity"
            )
        if identity in identities:
            raise CompletionError(
                f"referenced private CSV row {index} duplicates an identity"
            )
        identities.add(identity)
        for column in numeric_columns:
            try:
                number = float(row[column])
            except (TypeError, ValueError) as exc:
                raise CompletionError(
                    f"referenced private CSV row {index} has invalid {column}"
                ) from exc
            if not math.isfinite(number):
                raise CompletionError(
                    f"referenced private CSV row {index} has non-finite {column}"
                )
        if row["large_residual"] not in {"True", "False", "true", "false"}:
            raise CompletionError(
                f"referenced private CSV row {index} has invalid large_residual"
            )
    return len(rows)


def verify_matched_audit(
    *,
    evidence_root: str | Path,
    target: str,
    mode: str,
    config_path: str | Path,
    repository: str | Path,
    not_before: str | datetime,
) -> dict[str, Any]:
    if mode not in MODES:
        raise CompletionError(f"unsupported mode: {mode}")
    threshold = (
        _parse_utc(not_before, label="not-before")
        if isinstance(not_before, str)
        else not_before.astimezone(timezone.utc)
    )
    config_file = Path(config_path).resolve()
    repository_path = Path(repository).resolve()
    config = _load_config(config_file)
    _config_identity(config)
    targets = config.get("targets")
    if not isinstance(targets, dict) or target not in targets:
        raise CompletionError(f"target is not configured: {target}")
    seeds = _configured_seeds(config)
    evidence_path = _matched_evidence_path(
        Path(evidence_root).resolve(), target=target, mode=mode
    )
    _assert_fresh(evidence_path, threshold, label="matched audit evidence")
    report = _read_json(evidence_path, label="matched audit evidence")
    created = _parse_utc(report.get("created_utc"), label="matched audit.created_utc")
    if created < threshold:
        raise CompletionError("matched audit.created_utc predates not-before")
    _require_exact_bool(report, "passed", True, label="matched audit")
    _require_exact_bool(report, "locked_test_accessed", False, label="matched audit")
    if report.get("target") != target:
        raise CompletionError("matched audit target does not match the request")
    if report.get("mode") != mode:
        raise CompletionError("matched audit mode does not match the request")

    audit_checks = report.get("audit_checks")
    if not isinstance(audit_checks, dict):
        raise CompletionError("matched audit.audit_checks must be an object")
    for key in MATCHED_AUDIT_CHECKS:
        _require_exact_bool(audit_checks, key, True, label="matched audit.audit_checks")
    for key, value in audit_checks.items():
        if type(value) is not bool or value is not True:
            raise CompletionError(
                f"matched audit.audit_checks.{key} must be JSON boolean true"
            )

    seed_rows = report.get("seeds")
    if not isinstance(seed_rows, list) or len(seed_rows) != 3:
        raise CompletionError("matched audit must contain exactly three seed records")
    reported_seeds: list[int] = []
    for index, row in enumerate(seed_rows):
        if not isinstance(row, dict):
            raise CompletionError(f"matched audit.seeds[{index}] must be an object")
        reported_seeds.append(
            _require_int(row.get("seed"), label=f"matched audit.seeds[{index}].seed")
        )
        _require_exact_bool(
            row,
            "locked_test_accessed",
            False,
            label=f"matched audit.seeds[{index}]",
        )
    if reported_seeds != seeds:
        raise CompletionError("matched audit seeds do not match configured seed order")

    validation_samples = _require_int(
        report.get("validation_samples"),
        label="matched audit.validation_samples",
        minimum=1,
    )
    private_path = _resolve_private_csv(report, config)
    _assert_fresh(private_path, threshold, label="matched audit private CSV")
    rows = _verify_matched_csv(
        private_path,
        target=target,
        seeds=seeds,
        expected_rows=validation_samples,
    )
    return {
        "ok": True,
        "action": "verify-matched-audit",
        "target": target,
        "mode": mode,
        "seeds": seeds,
        "rows": rows,
        "evidence": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),
        "private_csv": str(private_path),
        "private_csv_sha256": sha256_file(private_path),
        "config_sha256": sha256_file(config_file),
        "source_tree_sha256": _source_tree_record(repository_path)["sha256"],
        "git_head": _git_head(repository_path),
    }


def _add_seed_arguments(parser: argparse.ArgumentParser, *, record: bool) -> None:
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--requested-epochs", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    if record:
        parser.add_argument("--not-before", required=True)
        parser.add_argument("--launch-provenance", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hemfm.worker_completion",
        description="Fail-closed HEM-FM worker completion evidence",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot-provenance")
    snapshot.add_argument("--config", type=Path, required=True)
    snapshot.add_argument("--repository", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    provenance = commands.add_parser("verify-provenance")
    provenance.add_argument("--config", type=Path, required=True)
    provenance.add_argument("--repository", type=Path, required=True)
    provenance.add_argument("--input", type=Path, required=True)
    record = commands.add_parser("record-seed")
    _add_seed_arguments(record, record=True)
    verify = commands.add_parser("verify-seed")
    _add_seed_arguments(verify, record=False)
    matched = commands.add_parser("verify-matched-audit")
    matched.add_argument("--evidence-root", type=Path, required=True)
    matched.add_argument("--target", required=True)
    matched.add_argument("--mode", choices=MODES, required=True)
    matched.add_argument("--config", type=Path, required=True)
    matched.add_argument("--repository", type=Path, required=True)
    matched.add_argument("--not-before", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "snapshot-provenance":
            result = snapshot_provenance(
                config_path=args.config,
                repository=args.repository,
                output=args.output,
            )
        elif args.command == "verify-provenance":
            result = verify_provenance(
                config_path=args.config,
                repository=args.repository,
                input_path=args.input,
            )
        elif args.command == "record-seed":
            result = record_seed_completion(
                kind=args.kind,
                artifact_root=args.artifact_root,
                seed=args.seed,
                mode=args.mode,
                requested_epochs=args.requested_epochs,
                config_path=args.config,
                repository=args.repository,
                not_before=args.not_before,
                launch_provenance=args.launch_provenance,
            )
        elif args.command == "verify-seed":
            result = verify_seed_completion(
                kind=args.kind,
                artifact_root=args.artifact_root,
                seed=args.seed,
                mode=args.mode,
                requested_epochs=args.requested_epochs,
                config_path=args.config,
                repository=args.repository,
            )
        else:
            result = verify_matched_audit(
                evidence_root=args.evidence_root,
                target=args.target,
                mode=args.mode,
                config_path=args.config,
                repository=args.repository,
                not_before=args.not_before,
            )
    except (CompletionError, OSError, ValueError) as exc:
        message = " ".join(str(exc).splitlines())
        print(f"worker-completion: {message}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
