from hemfm.integration_audit import TARGET_NAME, _configured_views


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

