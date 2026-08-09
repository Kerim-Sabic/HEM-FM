import json
from pathlib import Path

import pytest
import yaml

from hemfm.gates import assert_through, evidence_state


def _write_evidence(root: Path, gate: str, name: str, passed: bool = True) -> None:
    destination = root / gate / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"passed": passed}), encoding="utf-8")


def _config(root: Path) -> dict:
    return {
        "paths": {"evidence_root": str(root)},
        "gates": {
            "G0": {
                "execution_required_evidence": ["smoke.json"],
                "scientific_required_evidence": ["reproduction.json"],
            },
            "G1": {
                "execution_required_evidence": ["calibration_test.json"],
                "scientific_required_evidence": ["calibration_gate.json"],
            },
        },
    }


def test_execution_success_is_not_a_scientific_gate_pass(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_evidence(tmp_path, "G0", "smoke.json")

    state = evidence_state(config)
    g0 = state["gates"]["G0"]

    assert g0["execution_checks_passed"] is True
    assert g0["execution_sequentially_unlocked"] is True
    assert g0["scientific_gate_passed"] is False
    assert g0["passed"] is False
    assert g0["sequentially_unlocked"] is False
    assert_through(config, "G0")


def test_scientific_status_is_sequential_and_fail_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for gate, names in {
        "G0": ["smoke.json", "reproduction.json"],
        "G1": ["calibration_test.json", "calibration_gate.json"],
    }.items():
        for name in names:
            _write_evidence(tmp_path, gate, name)

    state = evidence_state(config)

    assert state["gates"]["G0"]["sequentially_unlocked"] is True
    assert state["gates"]["G1"]["passed"] is True
    assert state["gates"]["G1"]["sequentially_unlocked"] is True


def test_legacy_required_evidence_is_execution_only(tmp_path: Path) -> None:
    config = {
        "paths": {"evidence_root": str(tmp_path)},
        "gates": {"G0": {"required_evidence": ["legacy_smoke.json"]}},
    }
    _write_evidence(tmp_path, "G0", "legacy_smoke.json")

    state = evidence_state(config)

    assert state["gates"]["G0"]["execution_checks_passed"] is True
    assert state["gates"]["G0"]["passed"] is False


def test_missing_execution_prerequisite_still_blocks_training(tmp_path: Path) -> None:
    config = _config(tmp_path)

    with pytest.raises(RuntimeError, match="local execution prerequisites"):
        assert_through(config, "G0")


def test_config_uses_the_g7_and_g8_files_written_by_the_audits() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "protocol.example.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["gates"]["G7"]["execution_required_evidence"] == [
        "integration.json"
    ]
    assert config["gates"]["G8"]["execution_required_evidence"] == [
        "external_analysis_lock.json"
    ]
