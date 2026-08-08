from hemfm.external_ood import _development_partition, _without_nii_suffix


def test_nii_suffix_is_removed_without_losing_identity():
    from pathlib import Path

    assert _without_nii_suffix(Path("patient-1-4_image.nii.gz")) == "patient-1-4_image"
    assert _without_nii_suffix(Path("patient-2-4_image.nii")) == "patient-2-4_image"


def test_external_partition_is_stable():
    assert _development_partition("CardiacNet:PAH:12", 20260807) == _development_partition(
        "CardiacNet:PAH:12", 20260807
    )

