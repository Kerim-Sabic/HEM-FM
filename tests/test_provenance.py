from pathlib import Path
import pytest

from hemfm.provenance import Track, record_artifact


def test_research_artifact_records_source(tmp_path: Path):
    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"research")
    record = record_artifact(artifact, Track.RESEARCH, ["MIMIC-IV-ECHO"])
    assert record.track == Track.RESEARCH
    assert record.sources == ("MIMIC-IV-ECHO",)


def test_commercial_artifact_cannot_inherit_research(tmp_path: Path):
    artifact = tmp_path / "weights.pt"
    artifact.write_bytes(b"research")
    with pytest.raises(PermissionError):
        record_artifact(artifact, Track.COMMERCIAL, ["MIMIC-IV-ECHO"])

