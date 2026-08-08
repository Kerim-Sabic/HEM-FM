from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


LENGTH_UNITS_TO_MM = {"mm": 1.0, "cm": 10.0}
TIME_UNITS_TO_SECONDS = {"s": 1.0, "ms": 1e-3}
VELOCITY_UNITS_TO_MPS = {"m/s": 1.0, "cm/s": 1e-2, "mm/s": 1e-3}


@dataclass(frozen=True)
class RegionCalibration:
    x0: int
    y0: int
    x1: int
    y1: int
    delta_x: float
    delta_y: float
    x_unit: str
    y_unit: str
    region_type: str = "unknown"

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Ultrasound region bounds are inverted")
        if not np.isfinite(self.delta_x) or not np.isfinite(self.delta_y) or self.delta_x == 0 or self.delta_y == 0:
            raise ValueError("Physical pixel deltas must be finite and non-zero")

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0 + 1) * max(0, self.y1 - self.y0 + 1)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def image_to_physical(self, x: float, y: float) -> tuple[float, float]:
        if self.x_unit not in LENGTH_UNITS_TO_MM or self.y_unit not in LENGTH_UNITS_TO_MM:
            raise ValueError("This region does not define two spatial length axes")
        return (
            (x - self.x0) * self.delta_x * LENGTH_UNITS_TO_MM[self.x_unit],
            (y - self.y0) * self.delta_y * LENGTH_UNITS_TO_MM[self.y_unit],
        )

    def physical_to_image(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return (
            x_mm / (self.delta_x * LENGTH_UNITS_TO_MM[self.x_unit]) + self.x0,
            y_mm / (self.delta_y * LENGTH_UNITS_TO_MM[self.y_unit]) + self.y0,
        )


def select_region(regions: Iterable[RegionCalibration], point: tuple[float, float] | None = None) -> RegionCalibration:
    candidates = list(regions)
    if point is not None:
        containing = [region for region in candidates if region.contains(*point)]
        if containing:
            spatial_containing = [
                region
                for region in containing
                if region.x_unit in LENGTH_UNITS_TO_MM and region.y_unit in LENGTH_UNITS_TO_MM
            ]
            if spatial_containing:
                return min(spatial_containing, key=lambda region: region.area)
    spatial = [r for r in candidates if r.x_unit in LENGTH_UNITS_TO_MM and r.y_unit in LENGTH_UNITS_TO_MM]
    if not spatial:
        raise ValueError("No spatially calibrated ultrasound region is available")
    return max(spatial, key=lambda region: region.area)


def round_trip_error(region: RegionCalibration, points: np.ndarray) -> float:
    reconstructed = []
    for x, y in np.asarray(points, dtype=float):
        reconstructed.append(region.physical_to_image(*region.image_to_physical(x, y)))
    return float(np.max(np.abs(np.asarray(reconstructed) - points)))


def spectral_velocity_mps(delta: float, unit: str, zero_pixel: float, pixel: float) -> float:
    if unit not in VELOCITY_UNITS_TO_MPS:
        raise ValueError(f"Unsupported Doppler velocity unit: {unit}")
    return (pixel - zero_pixel) * delta * VELOCITY_UNITS_TO_MPS[unit]


def compose_homogeneous(*transforms: np.ndarray) -> np.ndarray:
    result = np.eye(3, dtype=float)
    for transform in transforms:
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("Every spatial transform must be a 3x3 homogeneous matrix")
        result = matrix @ result
    return result

