import numpy as np
import pytest

from hemfm.calibration import RegionCalibration, round_trip_error, select_region, spectral_velocity_mps


def test_spatial_round_trip_is_subpixel():
    region = RegionCalibration(17, 23, 616, 422, 0.031, 0.044, "cm", "cm", "2d")
    points = np.array([[17.0, 23.0], [211.5, 119.25], [616.0, 422.0]])
    assert round_trip_error(region, points) < 1e-9


def test_multi_region_selects_containing_region_before_largest():
    large = RegionCalibration(0, 0, 999, 999, 0.05, 0.05, "cm", "cm", "2d")
    inset = RegionCalibration(100, 100, 199, 199, 0.01, 0.01, "cm", "cm", "zoom")
    assert select_region([large, inset], point=(150, 150)) == inset
    assert select_region([inset], point=(150, 150)) == inset


def test_spectral_doppler_unit_conversion():
    assert spectral_velocity_mps(2.0, "cm/s", 100, 150) == pytest.approx(1.0)


def test_unsupported_units_fail_closed():
    region = RegionCalibration(0, 0, 10, 10, 1.0, 1.0, "pixels", "pixels")
    with pytest.raises(ValueError):
        region.image_to_physical(2, 2)


def test_zero_physical_delta_fails_closed():
    with pytest.raises(ValueError):
        RegionCalibration(0, 0, 10, 10, 0.0, 0.1, "cm", "cm")

