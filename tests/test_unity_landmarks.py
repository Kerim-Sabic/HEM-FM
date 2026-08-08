import json
from pathlib import Path

import pytest

from hemfm.unity_landmarks import MAX_UNAVAILABLE_IMAGE_FRACTION, UnityLandmarkDataset, _prefix


def test_unity_prefix_removes_only_frame_number():
    name = "01-05f272936a4c31bc3930b27c741e3cc0_3-0001.png"
    assert _prefix(name) == "01-05f272936a4c31bc3930b27c741e3cc0_3"


def test_unity_prefix_keeps_clip_identity():
    assert _prefix("01-hash_2-0001.png") != _prefix("01-hash_3-0001.png")


def _write_split(root: Path, split: str, names: list[str], annotated: list[str]) -> None:
    labels = root / "extracted" / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    (labels / f"labels-{split}.txt").write_text("\n".join(names), encoding="utf-8")
    (labels / f"labels-{split}.json").write_text(
        json.dumps({name: {"labels": {}} for name in annotated}), encoding="utf-8"
    )


def _names(count: int) -> list[str]:
    return [f"01-hash{index:05d}_1-0001.png" for index in range(count)]


def test_unity_dataset_excludes_records_absent_from_the_png_cache(tmp_path):
    names = _names(1000)
    _write_split(tmp_path, "train", names, names)
    images = {name: tmp_path / name for name in names[1:]}

    dataset = UnityLandmarkDataset(tmp_path, "train", images=images, physical={}, views={})

    assert dataset.excluded_unavailable == (names[0],)
    assert len(dataset) == 999
    assert names[0] not in dataset.names


def test_unity_dataset_fails_closed_above_the_unavailable_cap(tmp_path):
    names = _names(1000)
    _write_split(tmp_path, "train", names, names)
    permitted = int(1000 * MAX_UNAVAILABLE_IMAGE_FRACTION)
    images = {name: tmp_path / name for name in names[permitted + 1 :]}

    with pytest.raises(FileNotFoundError, match="png-cache"):
        UnityLandmarkDataset(tmp_path, "train", images=images, physical={}, views={})


def test_unity_dataset_never_tolerates_a_missing_annotation(tmp_path):
    names = _names(1000)
    _write_split(tmp_path, "train", names, names[1:])
    images = {name: tmp_path / name for name in names}

    with pytest.raises(FileNotFoundError, match="without annotations"):
        UnityLandmarkDataset(tmp_path, "train", images=images, physical={}, views={})


def test_unity_dataset_keeps_every_record_when_the_cache_is_complete(tmp_path):
    names = _names(1000)
    _write_split(tmp_path, "train", names, names)
    images = {name: tmp_path / name for name in names}

    dataset = UnityLandmarkDataset(tmp_path, "train", images=images, physical={}, views={})

    assert dataset.excluded_unavailable == ()
    assert dataset.names == names

