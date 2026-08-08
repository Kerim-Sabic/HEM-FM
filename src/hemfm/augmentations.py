from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def augmentation_matrix(
    center: tuple[float, float],
    rotation_degrees: float,
    zoom: float,
    translation_pixels: tuple[float, float],
) -> np.ndarray:
    if zoom <= 0:
        raise ValueError("Zoom must be positive")
    cx, cy = center
    tx, ty = translation_pixels
    angle = math.radians(rotation_degrees)
    cosine = math.cos(angle) * zoom
    sine = math.sin(angle) * zoom
    to_origin = np.asarray([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    rotate_scale = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    from_origin = np.asarray([[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]])
    return from_origin @ rotate_scale @ to_origin


def apply_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    transformed = (np.asarray(matrix, dtype=float) @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def physical_distance(points: np.ndarray, image_to_physical: np.ndarray) -> float:
    physical = apply_points(image_to_physical, points)
    return float(np.linalg.norm(physical[1] - physical[0]))


def verify_physical_augmentation_parity(output_json: str | Path, trials: int = 2000, seed: int = 20260807) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    max_point_error_mm = 0.0
    max_length_error_mm = 0.0
    max_transform_round_trip_pixels = 0.0
    for _ in range(trials):
        delta_x_mm = rng.uniform(0.1, 1.2)
        delta_y_mm = rng.uniform(0.1, 1.2)
        origin = rng.uniform(-20, 20, size=2)
        image_to_physical = np.asarray(
            [[delta_x_mm, 0.0, origin[0]], [0.0, delta_y_mm, origin[1]], [0.0, 0.0, 1.0]]
        )
        points = rng.uniform([20, 20], [200, 200], size=(2, 2))
        augment = augmentation_matrix(
            center=(112, 112),
            rotation_degrees=rng.uniform(-15, 15),
            zoom=rng.uniform(0.85, 1.15),
            translation_pixels=tuple(rng.uniform(-12, 12, size=2)),
        )
        augmented_points = apply_points(augment, points)
        augmented_to_physical = image_to_physical @ np.linalg.inv(augment)
        original_physical = apply_points(image_to_physical, points)
        recovered_physical = apply_points(augmented_to_physical, augmented_points)
        max_point_error_mm = max(max_point_error_mm, float(np.max(np.abs(original_physical - recovered_physical))))
        original_length = physical_distance(points, image_to_physical)
        recovered_length = physical_distance(augmented_points, augmented_to_physical)
        max_length_error_mm = max(max_length_error_mm, abs(original_length - recovered_length))
        round_trip = apply_points(np.linalg.inv(augment), augmented_points)
        max_transform_round_trip_pixels = max(
            max_transform_round_trip_pixels,
            float(np.max(np.abs(points - round_trip))),
        )
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(
            max_point_error_mm < 1e-9
            and max_length_error_mm < 1e-9
            and max_transform_round_trip_pixels < 1e-9
        ),
        "trials": trials,
        "seed": seed,
        "anisotropic_spacing": True,
        "max_point_error_mm": max_point_error_mm,
        "max_length_error_mm": max_length_error_mm,
        "max_transform_round_trip_pixels": max_transform_round_trip_pixels,
        "contract": "labels and calibration chains are transformed together; physical measurements are invariant",
    }
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

