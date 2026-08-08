Exit code: 0
Wall time: 1 seconds
Output:
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hemfm.echonet_dynamic import (
    EchoNetDynamicCacheDataset,
    _archive_member,
    _sample_indices,
    _rasterize_trace,
    _target_statistics,
    _validated_manifest,
)
from hemfm.echonet_trace import (
    EchoNetTraceDataset,
    _binary_metrics,
    _unique_trace_positions,
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


def test_volume_trace_rasterization_produces_nonempty_lv_mask() -> None:
    rows = pd.DataFrame(
        {
            "X1": [2, 2, 3],
            "Y1": [2, 5, 8],
            "X2": [8, 8, 7],
            "Y2": [2, 5, 8],
        }
    )
    mask = _rasterize_trace(rows, width=12, height=12)
    assert mask.shape == (12, 12)
    assert mask.dtype == np.uint8
    assert int(mask.sum()) > 20


def test_trace_dataset_repeats_exact_frames_and_keeps_masks_aligned(tmp_path: Path) -> None:
    cache = tmp_path / "trace.npz"
    trace_frames = np.stack(
        [np.full((3, 8, 8), 32, dtype=np.uint8), np.full((3, 8, 8), 224, dtype=np.uint8)]
    )
    trace_masks = np.stack(
        [np.eye(8, dtype=np.uint8), np.fliplr(np.eye(8, dtype=np.uint8))]
    )
    np.savez_compressed(
        cache,
        trace_frames=trace_frames,
        trace_masks=trace_masks,
        trace_indices=np.asarray([3, 17], dtype=np.int32),
    )
    frame = pd.DataFrame(
        [{"FileName": "trace", "cache_path": str(cache), "trace_count": 2}]
    )
    sample = EchoNetTraceDataset(frame, frames=16, augment=False)[0]
    assert tuple(sample["video"].shape) == (3, 16, 8, 8)
    assert tuple(sample["mask"].shape) == (16, 8, 8)
    assert sample["trace_count"] == 2
    assert sample["trace_indices"][:2].tolist() == [3, 17]
    assert sample["trace_indices"][2:].eq(-1).all()
    assert np.array_equal(sample["mask"][0].numpy(), trace_masks[0])
    assert np.array_equal(sample["mask"][-1].numpy(), trace_masks[1])


def test_trace_metrics_and_unique_positions_are_deterministic() -> None:
    positions = _unique_trace_positions(2, 16)
    assert positions.tolist() == [0, 8]
    target = np.asarray([[[1, 0], [0, 1]]], dtype=np.uint8)
    metrics = _binary_metrics(target.copy(), target)
    assert metrics == {"dice": 1.0, "iou": 1.0}

