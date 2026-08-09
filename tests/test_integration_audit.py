from hemfm.integration_audit import TARGET_NAME, _configured_views, _select_scalar_route


def test_all_six_specialists_have_config_targets():
    assert TARGET_NAME == {
        "EF": "EF",
        "LVEDV": "LVEDV",
        "LVESV": "LVESV",
        "LVOT_DIAMETER": "LVOT_DIAMETER",
        "RV_BASAL_DIAMETER": "RV_BASAL_DIAMETER",
        "AV_PEAK_VELOCITY": "AV_PEAK_VELOCITY",
    }


def test_spectral_anchor_views_are_not_hidden():
    assert _configured_views({"anchor_views": ["A5C", "Doppler_PLAX"]}) == [
        "A5C",
        "Doppler_PLAX",
    ]


def test_staged_three_seed_improvement_has_route_precedence():
    selected = _select_scalar_route(
        {"selected_architecture": "dinov3"},
        {"promote_over_frozen_single": True, "mae_change_vs_best_single": -0.1},
        {
            "passed": True,
            "promotion_eligible": True,
            "ensemble_validation_mae": 5.9,
            "mae_change_vs_current_g6": -0.8,
        },
        {
            "passed": True,
            "promotion_eligible": True,
            "ensemble_validation_mae": 5.9,
            "mae_change_vs_current_g6": -0.8,
        },
    )
    assert selected["scalar_path"] == "staged_echojepa_vitl"
    assert selected["development_mae_change"] == -0.8


def test_incomplete_staged_candidate_cannot_displace_baseline():
    selected = _select_scalar_route(
        {"selected_architecture": "dinov3"},
        {"promote_over_frozen_single": False},
        {"passed": False, "promotion_eligible": True, "mae_change_vs_current_g6": -0.8},
    )
    assert selected["scalar_path"] == "best_single_frozen_specialist"


def test_staged_candidate_without_matched_uncertainty_cannot_be_promoted():
    selected = _select_scalar_route(
        {"selected_architecture": "dinov3"},
        {"promote_over_frozen_single": False},
        {
            "passed": True,
            "promotion_eligible": True,
            "ensemble_validation_mae": 3.9,
            "mae_change_vs_current_g6": -0.8,
        },
    )
    assert selected["scalar_path"] == "best_single_frozen_specialist"


def test_failed_matched_calibration_blocks_staged_promotion():
    selected = _select_scalar_route(
        {"selected_architecture": "dinov3"},
        {"promote_over_frozen_single": False},
        {
            "passed": True,
            "promotion_eligible": True,
            "ensemble_validation_mae": 3.9,
            "mae_change_vs_current_g6": -0.8,
        },
        {"passed": True, "promotion_eligible": False},
    )
    assert selected["scalar_path"] == "best_single_frozen_specialist"
