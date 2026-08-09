import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from hemfm.calibration import RegionCalibration
from hemfm.caliper_pilot import (
    ARTIFACT_HASH_SCHEME,
    GEOMETRY_VERIFIER_VERSION,
    REVIEWER_ID_SCHEME,
    SAMPLE_HASH_SCHEME,
    CalibratedRegion,
    GlyphEndpoints,
    PilotThresholds,
    build_p3_manifest,
    evaluate_caliper_pilot,
    evaluate_p1,
    evaluate_p2,
    evaluate_p3,
    evaluate_p4_clean_recovery,
    evaluate_p5_blinded_audit,
    parse_ocr_measurement,
    parse_ocr_measurements,
    select_p5_audit_candidates,
    select_p2_candidates,
    verify_glyph_geometry,
    shortcut_probe_evidence_digest,
    write_caliper_pilot_artifacts,
)


def _hash(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def _inventory(rows: int = 240) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_index": np.arange(rows),
            "subject_id": [f"patient-{index}" for index in range(rows)],
            "study_id": [f"study-{index}" for index in range(rows)],
            "relative_path": [f"private/{index}.dcm" for index in range(rows)],
            "split": ["train" if index < rows - 20 else "internal_test" for index in range(rows)],
            "readable": True,
            "image_type": ["DERIVED\\SECONDARY" if index % 2 else "ORIGINAL\\PRIMARY" for index in range(rows)],
            "frames": [1 if index % 3 == 0 else 32 for index in range(rows)],
            "spatial_region_count": [1 if index % 4 else 0 for index in range(rows)],
            "spectral_region_count": [1 if index % 4 == 0 else 0 for index in range(rows)],
            "manufacturer": ["GE" if index % 2 else "Philips" for index in range(rows)],
            "software_version": ["v1" if index % 3 else "v2" for index in range(rows)],
            "measurement_screen_layout": ["layout-a" if index % 2 else "layout-b" for index in range(rows)],
        }
    )


def _shortcut_probe() -> dict[str, object]:
    evidence = {
        "complete": True,
        "predeclared": True,
        "probe_protocol_version": "clean-only-probe-v1",
        "metric_name": "normalized_absolute_error",
        "metric_direction": "lower_is_better",
        "paired_samples": 50,
        "annotated_metric": 0.51,
        "clean_metric": 0.50,
        "relative_performance_gap": 0.02,
        "clean_only_probe_passed": True,
        "probe_split": "validation",
        "locked_test_accessed": False,
        "annotated_frames_used_for_fitting": False,
    }
    evidence["evidence_sha256"] = shortcut_probe_evidence_digest(evidence)
    return evidence


def test_versioned_ocr_vocabulary_normalizes_explicit_units():
    length = parse_ocr_measurement("LVIDd: 4.8 cm")
    assert length.key == "LV_END_DIASTOLIC_DIAMETER"
    assert length.value == pytest.approx(48.0)
    assert length.unit == "mm"
    assert length.vocabulary_version == "hemfm-caliper-v1.0.0"

    velocity = parse_ocr_measurement("AV Vmax = 250 cm/sec")
    assert velocity.key == "AV_PEAK_VELOCITY"
    assert velocity.value == pytest.approx(2.5)
    assert velocity.unit == "m/s"


def test_ocr_parser_rejects_missing_unit_and_ambiguous_screen():
    with pytest.raises(ValueError, match="found 0"):
        parse_ocr_measurement("LVOT Diam 22")
    assert len(parse_ocr_measurements("LVIDd 4.8 cm; LVIDs 3.1 cm")) == 2
    with pytest.raises(ValueError, match="found 2"):
        parse_ocr_measurement("LVIDd 4.8 cm; LVIDs 3.1 cm")


def test_spatial_glyph_verification_uses_explicit_linked_region():
    measurement = parse_ocr_measurement("LVOT Diam 20 mm")
    glyph = GlyphEndpoints(
        sample_hash=_hash("sample"),
        measurement_key="LVOT_DIAMETER",
        annotated_frame_hash=_hash("annotated"),
        parent_cine_hash=_hash("cine"),
        region_id="zoom",
        start_x=10,
        start_y=20,
        end_x=30,
        end_y=20,
        glyph_family="GE-crosshair-v1",
        endpoint_confidence=0.99,
        line_association_score=0.98,
        glyph_bank_version="glyph-bank-1.0.0",
    )
    regions = [
        CalibratedRegion(
            "full",
            "spatial_bmode",
            RegionCalibration(0, 0, 99, 99, 0.2, 0.2, "cm", "cm", "2d"),
        ),
        CalibratedRegion(
            "zoom",
            "spatial_bmode",
            RegionCalibration(10, 10, 50, 50, 0.1, 0.1, "cm", "cm", "zoom"),
        ),
    ]
    result = verify_glyph_geometry(glyph, measurement, 2.0, "cm", regions)
    assert result["accepted"] is True
    assert result["physical_value"] == pytest.approx(20.0)
    assert result["region_id"] == "zoom"
    assert result["annotated_frame_trainable"] is False

    wrong_role = [CalibratedRegion("zoom", "spectral_doppler", regions[1].calibration)]
    rejected = verify_glyph_geometry(glyph, measurement, 2.0, "cm", wrong_role)
    assert rejected["accepted"] is False
    assert "spatial B-mode" in " ".join(rejected["rejection_reasons"])


def test_velocity_verification_never_reuses_bmode_affine():
    measurement = parse_ocr_measurement("AV Vmax 2.0 m/s")
    glyph = GlyphEndpoints(
        sample_hash=_hash("doppler"),
        measurement_key="AV_PEAK_VELOCITY",
        annotated_frame_hash=_hash("annotated"),
        parent_cine_hash=_hash("cine"),
        region_id="spectral-1",
        start_x=50,
        start_y=100,
        end_x=50,
        end_y=60,
        glyph_family="Philips-spectral-v2",
        endpoint_confidence=0.99,
        line_association_score=0.99,
        glyph_bank_version="glyph-bank-1.0.0",
    )
    spectral = CalibratedRegion(
        "spectral-1",
        "spectral_doppler",
        RegionCalibration(0, 0, 199, 199, 0.002, 5.0, "s", "cm/s", "spectral"),
    )
    passed = verify_glyph_geometry(glyph, measurement, 200, "cm/s", [spectral])
    assert passed["accepted"] is True
    assert passed["physical_value"] == pytest.approx(2.0)

    bmode = CalibratedRegion(
        "spectral-1",
        "spatial_bmode",
        RegionCalibration(0, 0, 199, 199, 0.05, 0.05, "cm", "cm", "2d"),
    )
    failed = verify_glyph_geometry(glyph, measurement, 2.0, "m/s", [bmode])
    assert failed["accepted"] is False
    assert "spectral-Doppler" in " ".join(failed["rejection_reasons"])


def test_p2_and_p3_stratification_are_exact_private_and_test_sealed():
    inventory = _inventory(2_040)
    p2 = select_p2_candidates(inventory, "private-test-secret", target=200)
    p3 = build_p3_manifest(inventory, "private-test-secret", target=2_000)
    assert len(p2) == 200
    assert len(p3) == 2_000
    assert set(p2["split"]) == {"train"}
    assert set(p3["split"]) == {"train"}
    assert not {"subject_id", "study_id", "relative_path"} & set(p2.columns)
    assert p2["sample_hash_scheme"].eq(SAMPLE_HASH_SCHEME).all()
    assert p2["sample_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert p2["vendor"].nunique() == 2
    assert p2["software"].nunique() == 2
    assert p2["frame_bucket"].nunique() == 2
    assert not p3["annotated_frame_trainable"].any()


def test_candidate_selection_fails_closed_when_count_or_split_contract_is_missing():
    with pytest.raises(ValueError, match="required"):
        select_p2_candidates(_inventory(100), "private-test-secret", target=200)
    missing_split = _inventory(240).drop(columns="split")
    with pytest.raises(ValueError, match="failed closed"):
        select_p2_candidates(missing_split, "private-test-secret", target=200)


def _p3_results(count: int = 200) -> pd.DataFrame:
    manifest = build_p3_manifest(_inventory(count + 20), "private-test-secret", target=count)
    keys = ["LVOT_DIAMETER" if index % 2 else "AV_PEAK_VELOCITY" for index in range(count)]
    values = [20.0 if key == "LVOT_DIAMETER" else 2.0 for key in keys]
    units = ["mm" if key == "LVOT_DIAMETER" else "m/s" for key in keys]
    manifest["measurement_key"] = keys
    manifest["physical_value"] = values
    manifest["canonical_unit"] = units
    manifest["geometry_json"] = [
        json.dumps(
            {
                "start_x": 10.0,
                "start_y": 20.0,
                "end_x": 30.0,
                "end_y": 20.0,
                "glyph_family": "synthetic",
                "region_id": "r",
                "calibration": {
                    "x0": 0,
                    "y0": 0,
                    "x1": 99,
                    "y1": 99,
                    "delta_x": 0.1 if key == "LVOT_DIAMETER" else 10.0,
                    "delta_y": 0.1 if key == "LVOT_DIAMETER" else 0.002,
                    "x_unit": "cm" if key == "LVOT_DIAMETER" else "cm/s",
                    "y_unit": "cm" if key == "LVOT_DIAMETER" else "s",
                    "region_type": "2d" if key == "LVOT_DIAMETER" else "spectral",
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for key in keys
    ]
    manifest["vocabulary_version"] = "hemfm-caliper-v1.0.0"
    manifest["glyph_bank_version"] = "synthetic-glyph-bank-1.0.0"
    manifest["geometry_verifier_version"] = GEOMETRY_VERIFIER_VERSION
    manifest["parent_cine_hash"] = [_hash(f"cine-{index}") for index in range(count)]
    manifest["annotated_frame_hash"] = [_hash(f"annotated-{index}") for index in range(count)]
    manifest["artifact_hash_scheme"] = ARTIFACT_HASH_SCHEME
    manifest["region_id"] = "r"
    manifest["region_purpose"] = [
        "spatial_bmode" if key == "LVOT_DIAMETER" else "spectral_doppler" for key in keys
    ]
    manifest["endpoint_confidence"] = 0.99
    manifest["line_association_score"] = 0.99
    manifest["ocr_value"] = values
    manifest["study_value"] = values
    manifest["ocr_absolute_error"] = 0.0
    manifest["study_absolute_error"] = 0.0
    manifest["absolute_tolerance"] = [0.8 if key == "LVOT_DIAMETER" else 0.05 for key in keys]
    manifest["relative_tolerance"] = 0.03
    manifest["rejection_reasons"] = [[] for _ in range(count)]
    manifest["ocr_parse_valid"] = True
    manifest["glyph_localized"] = True
    manifest["physical_verified"] = True
    manifest["accepted"] = True
    manifest["annotated_frame_trainable"] = False
    return manifest


def _clean_recovery(p3: pd.DataFrame, bad_index: int | None = None) -> pd.DataFrame:
    rows = []
    for index, row in p3.reset_index(drop=True).iterrows():
        rows.append(
            {
                "sample_hash": row.sample_hash,
                "sample_hash_scheme": SAMPLE_HASH_SCHEME,
                "annotated_frame_hash": _hash(f"annotated-{index}"),
                "recovered_frame_hash": _hash(f"annotated-{index}" if index == bad_index else f"clean-{index}"),
                "parent_cine_hash": row.parent_cine_hash,
                "artifact_hash_scheme": ARTIFACT_HASH_SCHEME,
                "tissue_match_score": 0.995,
                "annotation_pixels_remaining": 0,
                "overlay_detector_passed": True,
                "clean_only_probe_eligible": True,
                "split": "validation",
            }
        )
    return pd.DataFrame(rows)


def test_p4_emits_only_clean_frames_and_never_annotated_hashes():
    p3 = _p3_results(10)
    thresholds = PilotThresholds(p3_count=10, p4_clean_recovery_min=0.8)
    report, training = evaluate_p4_clean_recovery(
        p3,
        _clean_recovery(p3, bad_index=0),
        thresholds,
        _shortcut_probe(),
    )
    assert report["passed"] is True
    assert report["usable_clean_frames"] == 9
    assert "annotated_frame_hash" not in training
    assert "clean_frame_hash" in training
    assert not training["annotated_frame_trainable"].any()
    assert _hash("annotated-0") not in set(training["clean_frame_hash"])


def _audits(
    p3: pd.DataFrame,
    clean_training: pd.DataFrame,
    thresholds: PilotThresholds,
    incorrect_sample: int | None = None,
) -> pd.DataFrame:
    selected = select_p5_audit_candidates(p3, clean_training, thresholds=thresholds)
    incorrect_hash = p3.iloc[incorrect_sample].sample_hash if incorrect_sample is not None else None
    rows = []
    for _, row in selected.reset_index(drop=True).iterrows():
        for reviewer_name in ("reviewer-a", "reviewer-b"):
            reviewer = _hash(reviewer_name)
            correct = row.sample_hash != incorrect_hash
            rows.append(
                {
                    "sample_hash": row.sample_hash,
                    "sample_hash_scheme": SAMPLE_HASH_SCHEME,
                    "reviewer_id": reviewer,
                    "reviewer_id_scheme": REVIEWER_ID_SCHEME,
                    "blinded": True,
                    "independent": True,
                    "source_answer_visible": False,
                    "blinding_evidence_hash": _hash(f"blind-packet-{reviewer_name}"),
                    "reviewer_correct": correct,
                    "convention_match": correct,
                    "reviewed_value": row.physical_value + (0.01 if row.canonical_unit == "m/s" else 0.1),
                    "split": row.split,
                    "selection_rank": row.selection_rank,
                    "selection_stage": row.selection_stage,
                    "selection_scheme": row.selection_scheme,
                    "selection_seed": row.selection_seed,
                    "selection_manifest_sha256": row.selection_manifest_sha256,
                }
            )
    return pd.DataFrame(rows)


def test_p5_requires_two_blinded_reviewers_and_reports_precision_error_yield_by_strata():
    p3 = _p3_results(200)
    p4_report, training = evaluate_p4_clean_recovery(
        p3,
        _clean_recovery(p3),
        PilotThresholds(p3_count=200),
        _shortcut_probe(),
    )
    assert p4_report["passed"] is True
    thresholds = PilotThresholds(p3_count=200, p5_count=200)
    report = evaluate_p5_blinded_audit(
        p3,
        training,
        _audits(p3, training, thresholds),
        thresholds=thresholds,
    )
    assert report["passed"] is True
    assert report["precision"] == 1.0
    assert report["decision_agreement"] == 1.0
    assert set(report["by_group"]) == {"measurement_key", "vendor", "software"}
    assert report["by_group"]["vendor"]["GE"]["yield"] == 1.0
    assert report["by_group"]["vendor"]["GE"]["decision_agreement"] == 1.0
    assert report["by_group"]["software"]["v1"]["p95_normalized_absolute_error"] < 1.0
    assert report["error_by_measurement"]["LVOT_DIAMETER"]["passed"] is True

    unblinded = _audits(p3, training, thresholds)
    unblinded.loc[0, "blinded"] = False
    failed = evaluate_p5_blinded_audit(p3, training, unblinded, thresholds=thresholds)
    assert failed["passed"] is False
    assert failed["checks"]["all_reviews_blinded"] is False


def test_p5_precision_threshold_is_fail_closed():
    p3 = _p3_results(200)
    thresholds = PilotThresholds(p3_count=200, p5_count=200, p5_precision_min=1.0)
    _, training = evaluate_p4_clean_recovery(
        p3,
        _clean_recovery(p3),
        thresholds,
        _shortcut_probe(),
    )
    report = evaluate_p5_blinded_audit(
        p3,
        training,
        _audits(p3, training, thresholds, incorrect_sample=0),
        thresholds=thresholds,
    )
    assert report["passed"] is False
    assert report["checks"]["minimum_precision"] is False
    assert report["precision"] == pytest.approx(199 / 200)


def test_top_level_promotable_requires_every_predeclared_stage_and_writes_public_evidence(tmp_path):
    thresholds = PilotThresholds(
        p1_min_instances=100,
        p2_count=10,
        p3_count=20,
        p5_count=20,
        active_group_min_population=100,
    )
    p2 = select_p2_candidates(_inventory(40), "private-test-secret", target=10)
    p2["manual_review_complete"] = True
    p2["annotation_rendering_reviewed"] = True
    p2["annotation_present"] = True
    p2["measurement_key"] = ["LVOT_DIAMETER" if index % 2 else "AV_PEAK_VELOCITY" for index in range(10)]
    p2["glyph_family"] = ["synthetic-spatial" if index % 2 else "synthetic-spectral" for index in range(10)]
    p3 = _p3_results(20)
    recovery = _clean_recovery(p3)
    _, p4_training = evaluate_p4_clean_recovery(p3, recovery, thresholds, _shortcut_probe())
    p1 = {
        "complete": True,
        "scan_target_complete": True,
        "status": "complete",
        "records": 100,
        "record_list_records": 100,
        "enriched_fields": ["image_type", "frames", "region_type", "vendor", "software"],
        "metadata_only": True,
        "pixel_data_read": False,
        "record_list_sha256": _hash("record-list"),
        "instance_parquet_sha256": _hash("instance-parquet"),
        "region_parquet_sha256": _hash("region-parquet"),
        "locked_test_pixels_accessed": False,
    }
    report, training = evaluate_caliper_pilot(
        p1,
        p2,
        p3,
        recovery,
        _audits(p3, p4_training, thresholds),
        shortcut_probe_evidence=_shortcut_probe(),
        thresholds=thresholds,
    )
    assert report["passed"] is True
    assert report["promotable"] is True
    assert report["gold_mined_weight_selection_allowed"] is True
    assert report["annotated_frames_trainable"] is False
    assert len(training) == 20
    paths = write_caliper_pilot_artifacts(tmp_path, report)
    assert (tmp_path / "caliper_pilot_go_no_go.json").exists()
    assert (tmp_path / "caliper_measurement_vocabulary.json").exists()
    assert set(paths) == {"go_no_go", "vocabulary"}

    blocked, _ = evaluate_caliper_pilot(
        {**p1, "complete": False},
        p2,
        p3,
        recovery,
        _audits(p3, p4_training, thresholds),
        shortcut_probe_evidence=_shortcut_probe(),
        thresholds=thresholds,
    )
    assert blocked["passed"] is False
    assert blocked["promotable"] is False
    assert blocked["remaining_stages"] == ["P1"]


def test_nonfinite_glyph_confidence_is_rejected_before_geometry_verification():
    with pytest.raises(ValueError, match="endpoint_confidence must be finite"):
        GlyphEndpoints(
            sample_hash=_hash("sample"),
            measurement_key="LVOT_DIAMETER",
            annotated_frame_hash=_hash("annotated"),
            parent_cine_hash=_hash("cine"),
            region_id="r",
            start_x=10,
            start_y=20,
            end_x=30,
            end_y=20,
            glyph_family="glyph-v1",
            endpoint_confidence=float("nan"),
            line_association_score=0.99,
            glyph_bank_version="bank-v1",
        )


def test_missing_partition_and_contradictory_metadata_fail_closed():
    thresholds = PilotThresholds(p1_min_instances=1, p2_count=10, p3_count=20, p5_count=20)
    p1 = {
        "complete": True,
        "scan_target_complete": True,
        "status": "complete",
        "records": 1,
        "record_list_records": 1,
        "enriched_fields": ["image_type", "frames", "region_type", "vendor", "software"],
        "metadata_only": True,
        "pixel_data_read": True,
        "record_list_sha256": _hash("source"),
        "instance_parquet_sha256": _hash("instances"),
        "region_parquet_sha256": _hash("regions"),
        "locked_test_pixels_accessed": False,
    }
    assert evaluate_p1(p1, thresholds)["checks"]["metadata_only_scan"] is False

    p2 = select_p2_candidates(_inventory(40), "private-test-secret", target=10)
    p2["manual_review_complete"] = True
    p2["annotation_rendering_reviewed"] = True
    p2["annotation_present"] = True
    p2["measurement_key"] = ["LVOT_DIAMETER" if i % 2 else "AV_PEAK_VELOCITY" for i in range(10)]
    p2["glyph_family"] = ["spatial" if i % 2 else "spectral" for i in range(10)]
    p2 = p2.drop(columns=["split", "test_locked"])
    p2_report = evaluate_p2(p2, thresholds)
    assert p2_report["passed"] is False
    assert p2_report["checks"]["partition_evidence_valid"] is False

    p3 = _p3_results(20).drop(columns=["split", "test_locked"])
    p3_report = evaluate_p3(p3, thresholds)
    assert p3_report["passed"] is False
    assert p3_report["checks"]["partition_evidence_valid"] is False


def test_p3_recomputes_geometry_instead_of_trusting_acceptance_booleans():
    p3 = _p3_results(20)
    p3.loc[0, "ocr_value"] = 30.0
    p3.loc[0, "ocr_absolute_error"] = 0.0
    report = evaluate_p3(p3, PilotThresholds(p3_count=20))
    assert report["passed"] is False
    assert report["checks"]["all_acceptance_claims_geometry_verified"] is False
    assert report["unverifiable_acceptance_claims"] == 1


def test_raw_identifier_columns_and_non_digest_ids_block_p3():
    p3 = _p3_results(20)
    p3["Subject-ID"] = [f"patient-{index}" for index in range(len(p3))]
    p3.loc[0, "sample_hash"] = "patient-0"
    report = evaluate_p3(p3, PilotThresholds(p3_count=20))
    assert report["passed"] is False
    assert report["checks"]["identifier_columns_excluded"] is False
    assert report["checks"]["source_linked_versioned_evidence"] is False


def test_p4_cross_sample_annotated_frame_is_never_emitted_as_clean():
    p3 = _p3_results(10)
    recovery = _clean_recovery(p3)
    cross_sample_annotated = p3.loc[1, "annotated_frame_hash"]
    recovery.loc[0, "recovered_frame_hash"] = cross_sample_annotated
    report, training = evaluate_p4_clean_recovery(
        p3,
        recovery,
        PilotThresholds(p3_count=10, p4_clean_recovery_min=0.8),
        _shortcut_probe(),
    )
    assert report["passed"] is True
    assert report["usable_clean_frames"] == 9
    assert cross_sample_annotated not in set(training["clean_frame_hash"])


def test_p4_requires_digest_backed_shortcut_probe_before_emitting_training_rows():
    p3 = _p3_results(10)
    report, training = evaluate_p4_clean_recovery(
        p3,
        _clean_recovery(p3),
        PilotThresholds(p3_count=10),
    )
    assert report["passed"] is False
    assert report["checks"]["annotated_vs_clean_shortcut_probe_passed"] is False
    assert training.empty
    assert report["shortcut_probe"]["remaining_work"]


def test_p5_rejects_cherry_picked_audit_set_not_predeclared_selection():
    p3 = _p3_results(240)
    thresholds = PilotThresholds(p3_count=240, p5_count=200)
    _, training = evaluate_p4_clean_recovery(
        p3,
        _clean_recovery(p3),
        thresholds,
        _shortcut_probe(),
    )
    audits = _audits(p3, training, thresholds)
    expected_hashes = audits["sample_hash"].drop_duplicates().tolist()
    cherry_hashes = p3["sample_hash"].iloc[:200].tolist()
    replacement = dict(zip(expected_hashes, cherry_hashes, strict=True))
    audits["sample_hash"] = audits["sample_hash"].map(replacement)
    report = evaluate_p5_blinded_audit(p3, training, audits, thresholds=thresholds)
    assert report["passed"] is False
    assert report["checks"]["predeclared_stratified_sample_exact"] is False


def test_public_artifact_writer_rejects_path_leakage(tmp_path):
    with pytest.raises(ValueError, match="filesystem path"):
        write_caliper_pilot_artifacts(tmp_path, {"note": r"C:\private\patient-1.dcm"})
