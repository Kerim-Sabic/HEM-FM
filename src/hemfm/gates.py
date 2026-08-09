from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file


def gate_order(config: dict[str, Any]) -> list[str]:
    return list(config["gates"])


def _evidence_files(root: Path, gate: str, names: list[str]) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / gate / name
        valid_json = False
        declared_pass = False
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                valid_json = isinstance(payload, dict)
                declared_pass = valid_json and payload.get("passed") is True
            except (OSError, json.JSONDecodeError):
                pass
        files[name] = {
            "exists": path.exists(),
            "valid_json": valid_json,
            "declared_pass": declared_pass,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return files


def evidence_state(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["evidence_root"])
    state: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status_semantics": {
            "passed": "direct HEM-FM v4 scientific gate evidence passed",
            "execution_checks_passed": "local mechanical/proxy prerequisites passed; this is not a clinical or scientific gate pass",
            "sequentially_unlocked": "this and every preceding HEM-FM v4 scientific gate passed",
            "execution_sequentially_unlocked": "this and every preceding local execution prerequisite passed",
        },
        "gates": {},
    }
    for gate, definition in config["gates"].items():
        # ``required_evidence`` is accepted only as a legacy name for execution
        # prerequisites.  It can never produce a scientific pass by itself.
        execution_names = list(
            definition.get(
                "execution_required_evidence",
                definition.get("required_evidence", []),
            )
        )
        scientific_names = list(definition.get("scientific_required_evidence", []))
        execution_files = _evidence_files(root, gate, execution_names)
        scientific_files = _evidence_files(root, gate, scientific_names)
        execution_checks_passed = bool(execution_names) and all(
            item["declared_pass"] for item in execution_files.values()
        )
        scientific_evidence_passed = bool(scientific_names) and all(
            item["declared_pass"] for item in scientific_files.values()
        )
        scientific_gate_passed = bool(
            execution_checks_passed and scientific_evidence_passed
        )
        state["gates"][gate] = {
            # ``passed`` deliberately means the v4 gate, never merely that a
            # smoke test, pilot, or development run executed successfully.
            "passed": scientific_gate_passed,
            "scientific_gate_passed": scientific_gate_passed,
            "execution_checks_passed": execution_checks_passed,
            "execution_files": execution_files,
            "scientific_files": scientific_files,
        }
    previous_execution = True
    previous_scientific = True
    for gate in gate_order(config):
        current = state["gates"][gate]
        current["execution_sequentially_unlocked"] = bool(
            previous_execution and current["execution_checks_passed"]
        )
        current["sequentially_unlocked"] = bool(
            previous_scientific and current["scientific_gate_passed"]
        )
        previous_execution = current["execution_sequentially_unlocked"]
        previous_scientific = current["sequentially_unlocked"]
    return state


def assert_through(config: dict[str, Any], through: str) -> dict[str, Any]:
    """Assert local execution prerequisites through ``through``.

    This compatibility entry point intentionally does not assert that the
    HEM-FM v4 scientific gates passed.  Training modules use it to fail closed
    on required local engineering checks while ``evidence_state`` reports the
    separate scientific status.
    """
    state = evidence_state(config)
    if through not in state["gates"]:
        raise KeyError(f"Unknown gate {through}")
    required = gate_order(config)[: gate_order(config).index(through) + 1]
    failed = [
        gate
        for gate in required
        if not state["gates"][gate]["execution_sequentially_unlocked"]
    ]
    if failed:
        details = []
        for gate in failed:
            missing = [
                name
                for name, item in state["gates"][gate]["execution_files"].items()
                if not item["declared_pass"]
            ]
            details.append(f"{gate}: {', '.join(missing)}")
        raise RuntimeError(
            "Training is locked; required local execution prerequisites have not passed: "
            + "; ".join(details)
        )
    return state
