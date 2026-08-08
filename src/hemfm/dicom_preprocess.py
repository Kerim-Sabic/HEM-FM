from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pydicom
from scipy.ndimage import binary_dilation

from .calibration import RegionCalibration, compose_homogeneous, round_trip_error, select_region


@dataclass(frozen=True)
class PixelRegion:
    x0: int
    y0: int
    x1: int
    y1: int
    delta_x: float | None
    delta_y: float | None
    x_unit_code: int | None
    y_unit_code: int | None
    reference_pixel_x: float | None = None
    reference_pixel_y: float | None = None
    reference_value_x: float | None = None
    reference_value_y: float | None = None

    @property
    def area(self) -> int:
        return max(0, self.x1 - self.x0 + 1) * max(0, self.y1 - self.y0 + 1)


@dataclass
class PreprocessedCine:
    video: np.ndarray
    overlay_mask: np.ndarray
    image_to_model: np.ndarray
    model_to_image: np.ndarray
    source_shape: tuple[int, ...]
    crop: PixelRegion
    calibration_type: str
    spatial_round_trip_error_pixels: float | None
    overlay_fraction: float
    selected_frame_indices: list[int]

    def metadata(self) -> dict:
        return {
            "source_shape": list(self.source_shape),
            "crop": asdict(self.crop),
            "calibration_type": self.calibration_type,
            "spatial_round_trip_error_pixels": self.spatial_round_trip_error_pixels,
            "overlay_fraction": self.overlay_fraction,
            "selected_frame_indices": self.selected_frame_indices,
            "image_to_model": self.image_to_model.tolist(),
            "model_to_image": self.model_to_image.tolist(),
        }


def ultrasound_regions(dataset: pydicom.Dataset) -> list[PixelRegion]:
    regions = []
    for item in getattr(dataset, "SequenceOfUltrasoundRegions", []) or []:
        try:
            region = PixelRegion(
                x0=int(item.RegionLocationMinX0),
                y0=int(item.RegionLocationMinY0),
                x1=int(item.RegionLocationMaxX1),
                y1=int(item.RegionLocationMaxY1),
                delta_x=float(item.PhysicalDeltaX) if hasattr(item, "PhysicalDeltaX") else None,
                delta_y=float(item.PhysicalDeltaY) if hasattr(item, "PhysicalDeltaY") else None,
                x_unit_code=int(item.PhysicalUnitsXDirection) if hasattr(item, "PhysicalUnitsXDirection") else None,
                y_unit_code=int(item.PhysicalUnitsYDirection) if hasattr(item, "PhysicalUnitsYDirection") else None,
                reference_pixel_x=float(item.ReferencePixelX0) if hasattr(item, "ReferencePixelX0") else None,
                reference_pixel_y=float(item.ReferencePixelY0) if hasattr(item, "ReferencePixelY0") else None,
                reference_value_x=float(item.ReferencePixelPhysicalValueX) if hasattr(item, "ReferencePixelPhysicalValueX") else None,
                reference_value_y=float(item.ReferencePixelPhysicalValueY) if hasattr(item, "ReferencePixelPhysicalValueY") else None,
            )
            if region.area > 0:
                regions.append(region)
        except (AttributeError, TypeError, ValueError):
            continue
    return regions


def select_calibrated_crop(dataset: pydicom.Dataset, calibration_type: Literal["spatial", "spectral"]) -> tuple[PixelRegion, RegionCalibration | None]:
    regions = ultrasound_regions(dataset)
    if calibration_type == "spatial":
        spatial_pairs = []
        for pixel in regions:
            if pixel.x_unit_code == 0x0003 and pixel.y_unit_code == 0x0003 and pixel.delta_x and pixel.delta_y:
                try:
                    calibration = RegionCalibration(
                        pixel.x0,
                        pixel.y0,
                        pixel.x1,
                        pixel.y1,
                        pixel.delta_x,
                        pixel.delta_y,
                        "cm",
                        "cm",
                    )
                    spatial_pairs.append((pixel, calibration))
                except ValueError:
                    continue
        if not spatial_pairs:
            raise ValueError("No valid two-axis spatial calibration")
        selected_calibration = select_region([pair[1] for pair in spatial_pairs])
        selected_pixel = next(pixel for pixel, calibration in spatial_pairs if calibration == selected_calibration)
        return selected_pixel, selected_calibration
    if calibration_type == "spectral":
        spectral = [
            region
            for region in regions
            if (region.x_unit_code == 0x0007 and region.delta_x not in (None, 0))
            or (region.y_unit_code == 0x0007 and region.delta_y not in (None, 0))
        ]
        if not spectral:
            raise ValueError("No valid spectral velocity calibration")
        return max(spectral, key=lambda region: region.area), None
    raise ValueError(f"Unsupported calibration type: {calibration_type}")


def uniform_frame_indices(frame_count: int, requested: int) -> np.ndarray:
    if frame_count < 1 or requested < 1:
        raise ValueError("Frame counts must be positive")
    return np.rint(np.linspace(0, frame_count - 1, requested)).astype(int)


def spectral_pixel_to_velocity_mps(region: PixelRegion, x: float, y: float) -> float:
    if region.x_unit_code == 0x0007 and region.delta_x not in (None, 0) and region.reference_pixel_x is not None:
        reference = region.reference_value_x or 0.0
        return float((reference + (x - region.reference_pixel_x) * region.delta_x) * 0.01)
    if region.y_unit_code == 0x0007 and region.delta_y not in (None, 0) and region.reference_pixel_y is not None:
        reference = region.reference_value_y or 0.0
        return float((reference + (y - region.reference_pixel_y) * region.delta_y) * 0.01)
    raise ValueError("Spectral region lacks a calibrated velocity axis and reference pixel")


def static_overlay_mask(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected RGB frames shaped [T,H,W,3]")
    if frames.shape[0] < 3:
        return np.zeros(frames.shape[1:3], dtype=bool)
    maximum = frames.max(axis=-1)
    minimum = frames.min(axis=-1)
    brightness = maximum.mean(axis=0)
    temporal_std = maximum.std(axis=0)
    saturation = (maximum - minimum).mean(axis=0)
    white_static = (brightness >= 220.0) & (temporal_std <= 4.0)
    colour_static = (brightness >= 150.0) & (saturation >= 45.0) & (temporal_std <= 6.0)
    return binary_dilation(white_static | colour_static, iterations=1)


def _as_rgb(pixel_array: np.ndarray) -> np.ndarray:
    array = np.asarray(pixel_array)
    if array.ndim == 2:
        array = array[None, :, :, None]
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        array = array[None, :, :, :3]
    elif array.ndim == 3:
        array = array[:, :, :, None]
    elif array.ndim == 4:
        array = array[:, :, :, :3]
    else:
        raise ValueError(f"Unsupported decoded pixel shape: {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.dtype != np.uint8:
        low, high = np.percentile(array, [0.5, 99.5])
        array = np.clip((array.astype(np.float32) - low) / max(high - low, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return array


def _letterbox(frames: np.ndarray, output_size: int) -> tuple[np.ndarray, float, int, int]:
    import torch
    import torch.nn.functional as functional

    height, width = frames.shape[1:3]
    scale = min(output_size / width, output_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    resized = functional.interpolate(tensor, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    output = torch.zeros((frames.shape[0], 3, output_size, output_size), dtype=torch.float32)
    pad_x = (output_size - resized_width) // 2
    pad_y = (output_size - resized_height) // 2
    output[:, :, pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return output.clamp(0, 255).byte().numpy(), scale, pad_x, pad_y


def preprocess_dicom_cine(
    path: str | Path,
    calibration_type: Literal["spatial", "spectral"],
    frames: int = 16,
    output_size: int = 224,
    clean_overlays: bool = False,
) -> PreprocessedCine:
    # Read metadata separately and stream only the requested frames.  Some MIMIC
    # studies contain large colour cines; decoding the complete cine can require
    # several hundred MiB per DataLoader worker before cropping or sampling.
    # pydicom's path-based iterator keeps peak memory bounded to one source frame.
    source_path = Path(path)
    dataset = pydicom.dcmread(source_path, stop_before_pixels=True, force=False)
    crop, spatial = select_calibrated_crop(dataset, calibration_type)
    frame_count = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    height = int(dataset.Rows)
    width = int(dataset.Columns)
    indices = uniform_frame_indices(frame_count, frames)
    unique_indices = sorted(set(int(index) for index in indices))
    decoded_by_index: dict[int, np.ndarray] = {}
    for index, frame_array in zip(
        unique_indices,
        pydicom.pixels.iter_pixels(source_path, indices=unique_indices, raw=False),
        strict=True,
    ):
        decoded_by_index[index] = _as_rgb(frame_array)[0]
    selected = np.stack([decoded_by_index[int(index)] for index in indices])
    source_shape = (frame_count, height, width, 3)
    x0 = min(max(crop.x0, 0), width - 1)
    x1 = min(max(crop.x1, x0), width - 1)
    y0 = min(max(crop.y0, 0), height - 1)
    y1 = min(max(crop.y1, y0), height - 1)
    selected = selected[:, y0 : y1 + 1, x0 : x1 + 1]
    overlay = static_overlay_mask(selected)
    if clean_overlays:
        selected = selected.copy()
        temporal_median = np.median(selected, axis=0).astype(np.uint8)
        temporal_median[overlay] = 0
        selected[:, overlay] = temporal_median[overlay]
    video, scale, pad_x, pad_y = _letterbox(selected, output_size)
    image_to_crop = np.asarray([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
    crop_to_model = np.asarray([[scale, 0.0, pad_x], [0.0, scale, pad_y], [0.0, 0.0, 1.0]])
    image_to_model = compose_homogeneous(image_to_crop, crop_to_model)
    model_to_image = np.linalg.inv(image_to_model)
    spatial_error = None
    if spatial is not None:
        points = np.asarray([[spatial.x0, spatial.y0], [spatial.x1, spatial.y1]], dtype=float)
        spatial_error = round_trip_error(spatial, points)
    overlay_resized, _, _, _ = _letterbox(np.repeat(overlay[None, :, :, None], 3, axis=-1).astype(np.uint8) * 255, output_size)
    return PreprocessedCine(
        video=video.transpose(1, 0, 2, 3),
        overlay_mask=overlay_resized[0, 0].astype(bool),
        image_to_model=image_to_model,
        model_to_image=model_to_image,
        source_shape=source_shape,
        crop=PixelRegion(
            x0,
            y0,
            x1,
            y1,
            crop.delta_x,
            crop.delta_y,
            crop.x_unit_code,
            crop.y_unit_code,
            crop.reference_pixel_x,
            crop.reference_pixel_y,
            crop.reference_value_x,
            crop.reference_value_y,
        ),
        calibration_type=calibration_type,
        spatial_round_trip_error_pixels=spatial_error,
        overlay_fraction=float(overlay.mean()),
        selected_frame_indices=indices.tolist(),
    )

