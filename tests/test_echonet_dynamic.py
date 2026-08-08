from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hemfm.echonet_dynamic import (
    EchoNetDynamicCacheDataset,
    _archive_member,
    _sample_indices,
    _target_statistics,
    _validated_manifest,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "FileName": "study_train",
                "EF": 55.0,
                "ESV": 40.0,
                "EDV": 90.0,
                "FrameHeight": 112,
                "FrameWidth": 112,
                "FPS": 50,
                "NumberOfFrames": 100,
                "Split": "TRAIN",
            },
            {
                "FileName": "study_val",
                "EF": 45.0,
                "ESV": 60.0,
                "EDV": 110.0,
                "FrameHeight": 112,
                "FrameWidth": 112,
                "FPS": 50,
                "NumberOfFrames": 120,
                "Split": "VAL",
            },
            {
                "FileName": "study_test",
                "EF": 50.0,
                "ESV": 50.0,
                "EDV": 100.0,
                "FrameHeight": 112,
                "FrameWidth": 112,
                "FPS": 50,
                "NumberOfFrames": 110,
                "Split": "TEST",
            },
        ]
    )


def test_manifest_requires_every_declared_video_and_preserves_official_splits() -> None:
    frame = _frame()
    members = {_archive_member(name) for name in frame["FileName"]}
    validated = _validated_manifest(frame, members)
    assert set(validated["Split"]) == {"TRAIN", "VAL", "TEST"}
    assert len(validated) == 3


def test_manifest_fails_closed_for_missing_or_duplicate_video() -> None:
    frame = _frame()
    members = {_archive_member(name) for name in frame["FileName"]}
    with pytest.raises(ValueError, match="missing 1"):
        _validated_manifest(frame, members - {_archive_member("study_val")})
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate filenames"):
        _validated_manifest(duplicated, members)


def test_archive_member_rejects_path_traversal() -> None:
    assert _archive_member("example") == "EchoNet-Dynamic/Videos/example.avi"
    with pytest.raises(ValueError, match="Unsafe"):
        _archive_member(str(Path("..") / "escape"))


def test_sampling_is_deterministic_and_spans_the_full_cine() -> None:
    indices = _sample_indices(101, 16)
    assert indices[0] == 0
    assert indices[-1] == 100
    assert np.all(np.diff(indices) > 0)


def test_reserved_test_target_is_not_required_for_development_audit() -> None:
    frame = _frame()
    frame.loc[frame["Split"].eq("TEST"), ["EF", "ESV", "EDV"]] = np.nan
    members = {_archive_member(name) for name in frame["FileName"]}
    validated = _validated_manifest(frame, members)
    assert validated["Split"].eq("TEST").sum() == 1


def test_cache_dataset_returns_normalized_multitask_sample(tmp_path: Path) -> None:
    cache = tmp_path / "sample.npz"
    np.savez_compressed(cache, video=np.full((3, 16, 8, 8), 127, dtype=np.uint8))
    frame = pd.DataFrame(
        [{"FileName": "sample", "EF": 55.0, "ESV": 40.0, "EDV": 90.0, "cache_path": str(cache)}]
    )
    center, scale = _target_statistics(
        pd.concat([frame, frame.assign(EF=65.0, ESV=50.0, EDV=110.0)], ignore_index=True)
    )
    dataset = EchoNetDynamicCacheDataset(
        frame, target_center=center, target_scale=scale, augment=False
    )
    sample = dataset[0]
    assert tuple(sample["video"].shape) == (3, 16, 8, 8)
    assert tuple(sample["target"].shape) == (3,)
    assert sample["file_name"] == "sample"
