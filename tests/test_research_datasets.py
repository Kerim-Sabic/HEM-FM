from pathlib import Path

from hemfm.research_datasets import _camus_patient_id, _unity_prefix, parse_metaimage_header
from hemfm.ted_temporal import TEDSequenceDataset


def test_ted_identity_maps_to_camus_padding() -> None:
    assert _camus_patient_id("patient001") == "patient0001"
    assert _camus_patient_id("patient098") == "patient0098"


def test_unity_prefix_removes_only_frame_suffix() -> None:
    name = "01-abc_3-0001.png"
    assert _unity_prefix(name) == "01-abc_3"


def test_parse_metaimage_header(tmp_path: Path) -> None:
    path = tmp_path / "sample.mhd"
    path.write_text(
        "DimSize = 10 20 30\n"
        "ElementSpacing = 0.3 0.4 1.5\n"
        "ElementType = MET_UCHAR\n"
        "ElementDataFile = sample.raw\n",
        encoding="ascii",
    )
    assert parse_metaimage_header(path) == {
        "dimensions": (10, 20, 30),
        "spacing": (0.3, 0.4, 1.5),
        "element_type": "MET_UCHAR",
        "data_file": "sample.raw",
    }


def test_ted_sequence_dataset_reads_metaimage_order(tmp_path: Path) -> None:
    import numpy as np

    patient = "patient001"
    root = tmp_path / "database" / patient
    root.mkdir(parents=True)
    stem = root / f"{patient}_4CH_sequence"
    header = (
        "DimSize = 4 3 2\n"
        "ElementSpacing = 0.5 0.25 1.0\n"
        "ElementType = MET_UCHAR\n"
    )
    Path(f"{stem}.mhd").write_text(header + f"ElementDataFile = {patient}.raw\n", encoding="ascii")
    Path(f"{stem}_gt.mhd").write_text(header + f"ElementDataFile = {patient}_gt.raw\n", encoding="ascii")
    np.arange(24, dtype=np.uint8).tofile(Path(f"{stem}.raw"))
    np.zeros(24, dtype=np.uint8).tofile(Path(f"{stem}_gt.raw"))
    sample = TEDSequenceDataset(tmp_path, [patient], frames=2, resolution=8)[0]
    assert sample["video"].shape == (3, 2, 8, 8)
    assert sample["mask"].shape == (2, 8, 8)

