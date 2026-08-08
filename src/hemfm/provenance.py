from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable
import json

from .hashing import sha256_file


class Track(StrEnum):
    RESEARCH = "R"
    COMMERCIAL = "C"


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    sha256: str
    track: Track
    sources: tuple[str, ...]
    numerical_ancestors: tuple[str, ...]


def record_artifact(
    path: str | Path,
    track: Track,
    sources: Iterable[str],
    numerical_ancestors: Iterable[str] = (),
) -> ArtifactRecord:
    source_tuple = tuple(sorted(set(sources)))
    ancestor_tuple = tuple(sorted(set(numerical_ancestors)))
    if track is Track.COMMERCIAL and (source_tuple or ancestor_tuple):
        raise PermissionError(
            "Commercial Track artifacts may not inherit research data, weights, "
            "pseudo-labels, thresholds, or other numerical artifacts"
        )
    return ArtifactRecord(str(Path(path)), sha256_file(path), track, source_tuple, ancestor_tuple)


def write_manifest(records: Iterable[ArtifactRecord], destination: str | Path) -> None:
    payload = {"schema_version": 1, "artifacts": [asdict(record) for record in records]}
    Path(destination).write_text(json.dumps(payload, indent=2), encoding="utf-8")

