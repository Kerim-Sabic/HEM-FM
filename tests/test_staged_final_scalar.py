from pathlib import Path

from hemfm.staged_final_scalar import (
    StagedScalarConfig,
    _cache_path,
    stage_for_epoch,
)


def test_staged_schedule_is_frozen_then_peft_then_selective() -> None:
    config = StagedScalarConfig(epochs=10, frozen_epochs=1, peft_epochs=4)
    assert stage_for_epoch(1, config) == "frozen"
    assert stage_for_epoch(2, config) == "peft"
    assert stage_for_epoch(5, config) == "peft"
    assert stage_for_epoch(6, config) == "selective_unfreeze"
    assert stage_for_epoch(10, config) == "selective_unfreeze"


def test_video_cache_key_is_path_normalized_and_calibration_specific() -> None:
    root = Path("cache")
    windows = _cache_path(root, r"a\b\cine.dcm", "spatial")
    portable = _cache_path(root, "a/b/cine.dcm", "spatial")
    spectral = _cache_path(root, "a/b/cine.dcm", "spectral")
    assert windows == portable
    assert windows != spectral
    assert windows.suffix == ".npz"

