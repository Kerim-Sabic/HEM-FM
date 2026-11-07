from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def create(path: Path, frames: int = 20, size: int = 112) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:size, :size]
    cine = []
    for index in range(frames):
        phase = 2 * np.pi * index / frames
        image = np.zeros((size, size), dtype=np.uint8)
        radius_x = 18 + 3 * np.sin(phase)
        radius_y = 28 + 4 * np.sin(phase)
        left = ((xx - 45) / radius_x) ** 2 + ((yy - 58) / radius_y) ** 2 <= 1
        right = ((xx - 72) / 15) ** 2 + ((yy - 60) / 25) ** 2 <= 1
        image[left] = 185
        image[right] = 125
        image[((xx - 50) ** 2 + (yy - 25) ** 2) <= 11**2] = 150
        image[((xx - 72) ** 2 + (yy - 28) ** 2) <= 10**2] = 105
        image = np.clip(image + (xx / size * 18).astype(np.uint8), 0, 255)
        cine.append(image)
    pixels = np.stack(cine)

    sop_uid = generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.PatientID = "SYNTHETIC-NONCLINICAL"
    dataset.Modality = "US"
    dataset.Rows = size
    dataset.Columns = size
    dataset.NumberOfFrames = str(frames)
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.FrameTime = 33.3
    region = Dataset()
    region.RegionLocationMinX0 = 0
    region.RegionLocationMinY0 = 0
    region.RegionLocationMaxX1 = size - 1
    region.RegionLocationMaxY1 = size - 1
    region.PhysicalUnitsXDirection = 0x0003
    region.PhysicalUnitsYDirection = 0x0003
    region.PhysicalDeltaX = 0.05
    region.PhysicalDeltaY = 0.05
    dataset.SequenceOfUltrasoundRegions = Sequence([region])
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a nonclinical synthetic echo cine")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    create(args.output, frames=args.frames)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
