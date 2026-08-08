import pandas as pd
import pytest

from hemfm.splits import assign_patient_splits, audit_no_patient_leakage


def test_every_study_for_patient_has_one_split():
    frame = pd.DataFrame({"subject_id": [1, 1, 2, 2, 3], "study_id": [10, 11, 20, 21, 30]})
    assigned = assign_patient_splits(frame, seed=20260807)
    report = audit_no_patient_leakage(assigned)
    assert report["passed"]


def test_leakage_is_fatal():
    frame = pd.DataFrame({"subject_id": [1, 1], "split": ["train", "internal_test"]})
    with pytest.raises(AssertionError):
        audit_no_patient_leakage(frame)

