import numpy as np

from hemfm.shortcut_controls import _border_mask, _select_rows


def test_border_mask_marks_only_outer_band():
    mask = _border_mask(100, 200, 0.10)
    assert mask[0, 100]
    assert mask[50, 0]
    assert not mask[50, 100]
    assert np.isclose(mask.mean(), 0.36)


def test_shortcut_sampler_never_invents_locked_test_rows():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "target": ["EF"] * 8,
            "split": ["train"] * 4 + ["validation"] * 4,
            "subject_id": [str(index) for index in range(8)],
            "value": list(range(8)),
        }
    )
    selected = _select_rows(frame, {"train": 2, "validation": 2}, 11)
    assert len(selected) == 4
    assert set(selected["split"]) == {"train", "validation"}

