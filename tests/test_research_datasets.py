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


def test_camus_a2c_manual_edes_uses_only_manual_phase_files(tmp_path: Path) -> None:
    import nibabel as nib
    import numpy as np

    from hemfm.camus_dense import CamusSequenceDataset

    (tmp_path / "database_split").mkdir()
    (tmp_path / "database_split" / "subgroup_training.txt").write_text(
        "patient0001\n", encoding="utf-8"
    )
    patient_root = tmp_path / "database_nifti" / "patient0001"
    patient_root.mkdir(parents=True)
    affine = np.eye(4)
    for phase, value in (("ED", 1), ("ES", 3)):
        image = np.full((6, 5), 20 * value, dtype=np.uint8)
        mask = np.zeros((6, 5), dtype=np.uint8)
        mask[1:5, 1:4] = value
        nib.save(
            nib.Nifti1Image(image, affine),
            patient_root / f"patient0001_2CH_{phase}.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(mask, affine),
            patient_root / f"patient0001_2CH_{phase}_gt.nii.gz",
        )
    sample = CamusSequenceDataset(
        tmp_path,
        "training",
        frames=2,
        resolution=8,
        views=("2CH",),
        annotation_mode="manual_edes",
    )[0]
    assert sample["video"].shape == (3, 2, 8, 8)
    assert sample["mask"].shape == (2, 8, 8)
    assert sample["view"] == "2CH"
    assert set(sample["mask"].unique().tolist()) == {0, 1, 3}


def test_camus_surface_hd95_is_zero_for_identical_masks() -> None:
    import numpy as np

    from hemfm.camus_dense import _surface_hd95_mm

    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 4:8] = True
    assert _surface_hd95_mm(mask, mask, np.array([0.5, 0.75])) == 0.0
