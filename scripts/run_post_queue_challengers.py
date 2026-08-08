from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run challengers after the main research queue")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    run_root = args.run_root.resolve()
    main_status = run_root / "week_training" / "research_queue_status.json"
    own_status = run_root / "week_training" / "post_queue_status.json"
    while True:
        main = _read(main_status)
        phase = main.get("phase", "missing")
        if phase == "complete":
            break
        if phase == "failed":
            _write(own_status, {"schema_version": 1, "updated_utc": datetime.now(timezone.utc).isoformat(), "phase": "blocked", "reason": "main research queue failed", "locked_test_accessed": False})
            return 2
        _write(own_status, {"schema_version": 1, "updated_utc": datetime.now(timezone.utc).isoformat(), "phase": "waiting_for_research_queue", "upstream_phase": phase, "locked_test_accessed": False})
        time.sleep(15)
    wrapper = repository / "scripts" / "run_logged_job.py"
    log = run_root / "logs" / "panecho_mimic_lv_full.log"
    status = run_root / "logs" / "panecho_mimic_lv_full.json"
    existing = _read(status)
    if existing.get("phase") == "complete":
        _write(
            own_status,
            {
                "schema_version": 1,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "complete",
                "reused_completed_panecho_audit": True,
                "locked_test_accessed": False,
            },
        )
        return 0
    command = [
        sys.executable,
        str(wrapper),
        str(log),
        str(status),
        "-m",
        "hemfm",
        "--config",
        "configs\\protocol.yaml",
        "panecho",
        "audit",
        "--device",
        "0",
    ]
    _write(own_status, {"schema_version": 1, "updated_utc": datetime.now(timezone.utc).isoformat(), "phase": "panecho_mimic_lv", "locked_test_accessed": False})
    result = subprocess.run(command, cwd=repository, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    final = "complete" if result.returncode == 0 else "failed"
    _write(own_status, {"schema_version": 1, "updated_utc": datetime.now(timezone.utc).isoformat(), "phase": final, "returncode": result.returncode, "locked_test_accessed": False})
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

