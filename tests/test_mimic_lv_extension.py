from pathlib import Path

from hemfm.mimic_lv_extension import _load_metadata
from hemfm.mimic_lv_training import _select_targets, _target_statistics


def test_extension_metadata_is_patient_split(tmp_path: Path) -> None:
    path = tmp_path / "FileList.csv"
    path.write_text(
        "patient_id,study_id,parent_dicom_path,LVEDV_A4C,LVESV_A4C,LVEF_A4C,LVEDV_BP,LVESV_BP,LVEF_BP\n"
        "1,10,files/a.dcm,100,40,60,101,41,59.4\n"
        "1,11,files/b.dcm,90,30,66.7,91,31,65.9\n",
        encoding="utf-8",
    )
    frame = _load_metadata(path, 20260807)
    assert frame["split"].nunique() == 1
    assert frame["patient_id"].tolist() == ["1", "1"]


def test_biplane_targets_are_preferred_with_a4c_fallback() -> None:
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(
        {
            "LVEF_BP": [55.0, np.nan], "LVEF_A4C": [60.0, 45.0],
            "LVEDV_BP": [100.0, np.nan], "LVEDV_A4C": [110.0, 90.0],
            "LVESV_BP": [45.0, np.nan], "LVESV_A4C": [44.0, 50.0],
        }
    )
    selected = _select_targets(frame)
    assert selected["LVEF"].tolist() == [55.0, 45.0]
    assert selected["LVEF_source"].tolist() == ["biplane", "a4c"]
    center, scale = _target_statistics(selected)
    assert center.shape == scale.shape == (3,)

