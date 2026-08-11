from __future__ import annotations

import numpy as np
import pytest

from hemfm.ted_temporal import (
    _centroid_motion_consistency,
    _cyclic_distance,
    _principal_axis_landmarks,
    _sequence_metrics,
    fuse_temporal_windows,
    sliding_temporal_windows,
    temporal_fusion_weights,
)


def test_cyclic_distance_wraps_at_end_of_cycle() -> None:
    assert _cyclic_distance(0, 9, 10) == 1
    assert _cyclic_distance(2, 7, 10) == 5


def test_motion_consistency_rewards_correct_translation() -> None:
    torch = pytest.importorskip("torch")
    mask = torch.zeros((1, 3, 16, 16), dtype=torch.long)
    for frame in range(3):
        mask[0, frame, 4:8, 3 + frame : 7 + frame] = 1
        mask[0, frame, 8:11, 4 + frame : 8 + frame] = 2
    probabilities = torch.nn.functional.one_hot(mask, num_classes=3).permute(0, 4, 1, 2, 3).float()
    aligned = _centroid_motion_consistency(probabilities, mask)
    stationary = probabilities.clone()
    stationary[:, :, 1:] = stationary[:, :, :1]
    misaligned = _centroid_motion_consistency(stationary, mask)
    assert float(aligned) < float(misaligned)


def test_sequence_metrics_are_exact_for_identical_tracks() -> None:
    truth = np.zeros((5, 24, 24), dtype=np.uint8)
    for frame, radius in enumerate((5, 4, 3, 4, 5)):
        yy, xx = np.ogrid[:24, :24]
        cavity = (yy - 12) ** 2 + (xx - (10 + frame)) ** 2 <= radius**2
        outer = (yy - 12) ** 2 + (xx - (10 + frame)) ** 2 <= (radius + 2) ** 2
        truth[frame, outer] = 2
        truth[frame, cavity] = 1
    metrics = _sequence_metrics(truth, truth, np.array([0.5, 0.5]))
    assert metrics["hausdorff95_mm"] == pytest.approx(0.0)
    assert metrics["ed_es_cyclic_frame_error"] == pytest.approx(0.0)
    assert metrics["temporal_area_acceleration_mae"] == pytest.approx(0.0)
    assert metrics["derived_landmark_tracking_error_mm"] == pytest.approx(0.0)


def test_principal_axis_landmarks_use_physical_spacing() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 4:6] = True
    landmarks = _principal_axis_landmarks(mask, np.array([2.0, 1.0]))
    assert landmarks is not None
    assert abs(landmarks[1, 0] - landmarks[0, 0]) == pytest.approx(10.0)


def test_sliding_windows_cover_every_native_frame() -> None:
    windows = sliding_temporal_windows(37, window_frames=16, overlap_frames=8)
    assert windows[0].tolist() == list(range(16))
    assert windows[-1].tolist() == list(range(21, 37))
    coverage = np.bincount(np.concatenate(windows), minlength=37)
    assert np.all(coverage > 0)
    assert np.any(coverage > 1)


def test_short_cine_is_resampled_without_uncovered_frames() -> None:
    windows = sliding_temporal_windows(5, window_frames=16, overlap_frames=8)
    assert len(windows) == 1
    assert set(windows[0]) == set(range(5))


def test_overlap_fusion_blends_windows_and_preserves_edges() -> None:
    indices = sliding_temporal_windows(24, window_frames=16, overlap_frames=8)
    first = np.zeros((2, 16, 3, 3), dtype=np.float32)
    second = np.full((2, 16, 3, 3), 2.0, dtype=np.float32)
    fused = fuse_temporal_windows([(indices[0], first), (indices[1], second)], frame_count=24)
    assert fused.shape == (2, 24, 3, 3)
    assert np.all(fused[:, :8] == 0)
    assert np.all(fused[:, 16:] == 2)
    assert np.all((fused[:, 8:16] > 0) & (fused[:, 8:16] < 2))
    assert np.all(temporal_fusion_weights(16) > 0)
