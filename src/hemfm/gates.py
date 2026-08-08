from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .hashing import sha256_file


def gate_order(config: dict[str, Any]) -> list[str]:
    return list(config["gates"])


def evidence_state(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["paths"]["evidence_root"])
    state: dict[str, Any] = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "gates": {}}
    for gate, definition in config["gates"].items():
        files = {}
        for name in definition["required_evidence"]:
            path = root / gate / name
            valid_json = False
            declared_pass = False
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    valid_json = isinstance(payload, dict)
                    declared_pass = payload.get("passed") is True
                except (OSError, json.JSONDecodeError):
                    pass
            files[name] = {
                "exists": path.exists(),
                "valid_json": valid_json,
                "declared_pass": declared_pass,
                "sha256": sha256_file(path) if path.exists() else None,
            }
        state["gates"][gate] = {"passed": all(item["declared_pass"] for item in files.values()), "files": files}
    previous = True
    for gate in gate_order(config):
        current = state["gates"][gate]
        current["sequentially_unlocked"] = bool(previous and current["passed"])
        previous = bool(previous and current["passed"])
    return state


def assert_through(config: dict[str, Any], through: str) -> dict[str, Any]:
    state = evidence_state(config)
    if through not in state["gates"]:
        raise KeyError(f"Unknown gate {through}")
    required = gate_order(config)[: gate_order(config).index(through) + 1]
    failed = [gate for gate in required if not state["gates"][gate]["sequentially_unlocked"]]
    if failed:
        details = []
        for gate in failed:
            missing = [name for name, item in state["gates"][gate]["files"].items() if not item["declared_pass"]]
            details.append(f"{gate}: {', '.join(missing)}")
        raise RuntimeError("Training is locked; required evidence has not passed: " + "; ".join(details))
    return state

