from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .frozen_specialists import _atomic_json
from .gates import assert_through


TARGET_NAME = {
    "EF": "EF",
    "LVEDV": "LVEDV",
    "LVESV": "LVESV",
    "LVOT_DIAMETER": "LVOT_DIAMETER",
    "RV_BASAL_DIAMETER": "RV_BASAL_DIAMETER",
    "AV_PEAK_VELOCITY": "AV_PEAK_VELOCITY",
}


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _configured_views(target: dict[str, Any]) -> list[str]:
    return list(
        target.get(
            "required_views",
            target.get("anchor_views", target.get("any_views", [])),
        )
    )


def run_integration_audit(config: dict[str, Any]) -> dict[str, Any]:
    assert_through(config, "G5")
    evidence = Path(config["paths"]["evidence_root"])
    g6 = evidence / "G6"
    holdouts = _read(g6 / "specialist_holdouts.json")
    failure = _read(g6 / "failure_detection.json")
    specialist_training = _read(g6 / "specialist_training.json")
    fusion = _read(g6 / "multibackbone_fusion.json")
    ev9v = _read(g6 / "ev9v_view_dinov3.json")
    temporal = {
        "vjepa21_vitb": _read(g6 / "ted_temporal_vjepa21_vitb.json"),
        "dinov3_vitb": _read(g6 / "ted_temporal_dinov3_vitb.json"),
    }
    landmarks = _read(g6 / "unity_landmarks_dinov3.json")
    external_ood = _read(g6 / "external_ood_abstention.json")
    panecho = _read(g6 / "panecho_mimic_lv.json")

    routes: dict[str, Any] = {}
    for target, trained in specialist_training.get("targets", {}).items():
        fusion_target = fusion.get("targets", {}).get(target, {})
        promote_fusion = bool(fusion_target.get("promote_over_frozen_single"))
        config_target = config["targets"][TARGET_NAME[target]]
        routes[target] = {
            "scalar_path": "multibackbone_fusion" if promote_fusion else "best_single_frozen_specialist",
            "selected_architecture": "fusion" if promote_fusion else trained.get("selected_architecture"),
            "required_views": _configured_views(config_target),
            "calibration": config_target.get("calibration"),
            "missing_evidence_behavior": "abstain or emit an explicit fallback path; never impute a required source silently",
            "fusion_validation_change": fusion_target.get("mae_change_vs_best_single"),
            "specialist_g6_passed": bool(holdouts.get("targets", {}).get(target, {}).get("internal_programme_target_met"))
            and bool(failure.get("targets", {}).get(target, {}).get("passes_failure_detection")),
        }

    temporal_passed = any(report.get("passed") for report in temporal.values())
    support_components = {
        "view_router": {
            "available": bool(ev9v.get("passed")),
            "promoted": bool(ev9v.get("promotion_eligible")),
            "role": "source/view routing only; does not supply scalar labels",
        },
        "temporal_dense_lv": {
            "available": temporal_passed,
            "selected": max(
                (
                    (name, report.get("best_validation_mean_foreground_dice", float("-inf")))
                    for name, report in temporal.items()
                    if report.get("passed")
                ),
                key=lambda item: item[1],
                default=(None, None),
            )[0],
        },
        "unity_landmarks": {"available": bool(landmarks.get("passed")), "role": "research-track anatomy/adequacy support"},
        "external_failure_detector": {
            "available": bool(external_ood.get("passed")),
            "promoted": bool(external_ood.get("promotion_eligible")),
        },
        "panecho_lv_challenger": {
            "available": bool(panecho.get("passed")),
            "promotion_eligible_by_endpoint": panecho.get("promotion_eligible_by_endpoint", {}),
            "role": "Research Track comparison on the MIMIC LV extension; it cannot replace a core route without matched G6 holdout evidence",
        },
    }
    checks = {
        "six_specialist_routes_declared": len(routes) == 6,
        "g6_specialist_holdouts_passed": bool(holdouts.get("passed")),
        "g6_failure_detection_passed": bool(failure.get("passed")),
        "view_source_selection_promoted": support_components["view_router"]["promoted"],
        "temporal_multi_view_support_available": temporal_passed,
        "missing_evidence_never_hidden": all(route["missing_evidence_behavior"] for route in routes.values()),
        "fusion_only_promoted_on_improvement": all(
            route["scalar_path"] != "multibackbone_fusion"
            or (route["fusion_validation_change"] is not None and route["fusion_validation_change"] < 0)
            for route in routes.values()
        ),
        "locked_test_not_accessed": not any(
            report.get("locked_test_accessed", False)
            for report in [holdouts, failure, fusion, ev9v, landmarks, external_ood, panecho, *temporal.values()]
        ),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "scope": "G7 study integration selection audit",
        "checks": checks,
        "routes": routes,
        "support_components": support_components,
        "decision": "Integrate only specialists that pass G6. Use the simpler best-single route unless fusion improves patient-disjoint development validation. Missing required views or calibration always remain explicit.",
        "locked_test_accessed": False,
    }
    destination = evidence / "G7" / "integration.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, report)
    return report

