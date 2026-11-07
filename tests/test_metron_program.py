from __future__ import annotations

from hemfm.metron_program import STAGES


def test_metron_stage_budget_matches_v8_programme() -> None:
    assert [stage.stage_id for stage in STAGES] == [
        "S0",
        "S0.5",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
    ]
    assert sum(stage.gpu_hours for stage in STAGES) == 4850


def test_no_training_stage_precedes_reference_standard() -> None:
    stage_ids = [stage.stage_id for stage in STAGES]
    assert stage_ids.index("S0.5") < stage_ids.index("S1")
    assert stage_ids.index("S0.5") < stage_ids.index("S3")
