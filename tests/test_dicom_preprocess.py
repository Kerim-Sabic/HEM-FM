import numpy as np

from hemfm.dicom_preprocess import PixelRegion, _letterbox, spectral_pixel_to_velocity_mps, static_overlay_mask, uniform_frame_indices


def test_uniform_sampling_includes_endpoints():
    indices = uniform_frame_indices(59, 16)
    assert indices[0] == 0
    assert indices[-1] == 58
    assert np.all(np.diff(indices) >= 0)


def test_static_bright_caliper_is_detected_but_motion_is_not():
    frames = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    frames[:, 5:7, 5:20] = 255
    for index in range(8):
        frames[index, 20, 3 + index] = 255
    mask = static_overlay_mask(frames)
    assert mask[5, 10]
    assert not mask[20, 10]


def test_letterbox_preserves_aspect_ratio_and_shape():
    frames = np.zeros((2, 50, 100, 3), dtype=np.uint8)
    result, scale, pad_x, pad_y = _letterbox(frames, 224)
    assert result.shape == (2, 3, 224, 224)
    assert scale == 2.24
    assert pad_x == 0
    assert pad_y == 56


def test_spectral_velocity_uses_reference_pixel_and_cm_per_second_units():
    region = PixelRegion(0, 0, 100, 100, 0.1, 2.0, 4, 7, 0.0, 50.0, 0.0, 0.0)
    assert spectral_pixel_to_velocity_mps(region, 10, 100) == 1.0

