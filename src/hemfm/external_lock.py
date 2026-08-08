from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .frozen_specialists import _atomic_json
from .gates import assert_through


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_external_lock(config: dict[str, Any]) -> dict[str, Any]:
    assert_through(config, "G5")
    evidence = Path(config["paths"]["evidence_root"])
    integration_path = evidence / "G7" / "integration.json"
    try:
        integration = json.loads(integration_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        integration = {}
    analysis = {
        "schema_version": 1,
        "protocol": "HEM-FM-v4.0",
        "gate": "G8",
        "analysis_population": "locked external multi-centre echocardiography cohort, opened once after this lock",
        "selection_prohibition": "No architecture, threshold, calibration, subgroup definition, or abstention rule may be changed after the external test is opened.",
        "patient_level_statistics": True,
        "bootstrap": {"unit": "patient", "replicates": 2000, "confidence_interval": 0.95, "seed": int(config["splits"]["seed"])},
        "metrics": {
            "EF": ["MAE", "RMSE", "Bland-Altman", "concordance", "selective risk-coverage"],
            "LVEDV_LVESV": ["MAE", "indexed error", "relative error", "contour Dice", "HD95", "ED/ES frame error", "pathway disagreement"],
            "LVOT_RV": ["MAE", "median AE", "P90", "P95", "within 1 mm", "within 2 mm", "endpoint error", "angle error"],
            "AV_Vmax": ["MAE", "directional bias", "within 0.1 m/s", "within 0.2 m/s", "envelope Dice", "peak error", "beat/window validity"],
            "reliability": ["accepted-case coverage", "accepted-case error", "failure AUROC", "OOD AUROC", "abstention reasons", "interval coverage"],
        },
        "subgroups": ["site", "vendor", "scanner", "sex", "age", "body size", "rhythm", "pathology", "quality"],
        "missing_data": "Report all missing required evidence and every abstention in the coverage denominator; no complete-case-only headline.",
        "routes": integration.get("routes", {}),
        "integration_evidence_sha256": hashlib.sha256(integration_path.read_bytes()).hexdigest() if integration_path.is_file() else None,
        "locked_external_accessed": False,
    }
    analysis_hash = _canonical_hash(analysis)
    checks = {
        "g7_integration_passed": bool(integration.get("passed")),
        "six_routes_frozen": len(analysis["routes"]) == 6,
        "patient_bootstrap_predeclared": analysis["bootstrap"]["unit"] == "patient",
        "metrics_and_subgroups_predeclared": bool(analysis["metrics"]) and bool(analysis["subgroups"]),
        "missing_data_and_coverage_predeclared": "coverage denominator" in analysis["missing_data"],
        "external_test_not_accessed": not analysis["locked_external_accessed"],
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "frozen": all(checks.values()),
        "checks": checks,
        "analysis_sha256": analysis_hash,
        "analysis": analysis,
        "next_action": "If frozen, make no more model changes and open the locked external cohort once. If not frozen, resolve the failed upstream gate without touching the external test.",
        "locked_test_accessed": False,
    }
    destination = evidence / "G8" / "external_analysis_lock.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, report)
    return report

