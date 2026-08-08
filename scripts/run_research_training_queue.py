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


def _queue_status(path: Path, phase: str, **extra: Any) -> None:
    _write(
        path,
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "locked_test_accessed": False,
            **extra,
        },
    )


def _wait_existing(statuses: list[Path], queue_status: Path) -> None:
    while True:
        payloads = [_read(path) for path in statuses]
        if all(payload.get("phase") == "complete" for payload in payloads):
            return
        failures = [payload for payload in payloads if payload.get("phase") == "failed"]
        if failures:
            raise RuntimeError(f"prerequisite training failed: {failures}")
        _queue_status(
            queue_status,
            "waiting_for_current_mimic_runs",
            prerequisites=[payload.get("phase", "missing") for payload in payloads],
        )
        time.sleep(10)


def _run_wave(
    repository: Path,
    logs: Path,
    queue_status: Path,
    wave: str,
    jobs: list[tuple[str, list[str]]],
) -> None:
    wrappers: list[tuple[str, Path, subprocess.Popen[Any]]] = []
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for name, command in jobs:
        log = logs / f"{name}.log"
        status = logs / f"{name}.json"
        if _read(status).get("phase") == "complete":
            continue
        # Clear a stale failure before starting the replacement worker. Without
        # this transition the monitor loop can observe the previous failure in
        # the short interval before run_logged_job writes its own START state.
        _write(
            status,
            {
                "schema_version": 1,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "queued_for_retry",
                "command": command,
                "locked_test_accessed": False,
            },
        )
        wrapper = subprocess.Popen(
            [
                sys.executable,
                str(repository / "scripts" / "run_logged_job.py"),
                str(log),
                str(status),
                *command,
            ],
            cwd=repository,
            creationflags=creation_flags,
        )
        wrappers.append((name, status, wrapper))
    while wrappers:
        phases = {name: _read(status).get("phase", "starting") for name, status, _ in wrappers}
        _queue_status(queue_status, wave, jobs=phases)
        for name, status, process in wrappers:
            payload = _read(status)
            if payload.get("phase") == "failed" or (process.poll() not in (None, 0)):
                raise RuntimeError(f"{name} failed: {payload}")
        if all(_read(status).get("phase") == "complete" for _, status, _ in wrappers):
            break
        time.sleep(10)
    _queue_status(queue_status, f"{wave}_complete")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ordered HEM-FM research challenger queue")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    run_root = args.run_root.resolve()
    logs = run_root / "logs"
    queue_status = run_root / "week_training" / "research_queue_status.json"
    config = ["-m", "hemfm", "--config", "configs\\protocol.yaml"]
    try:
        _wait_existing(
            [
                logs / "mimic_lv_dinov3_vitb_job_v3.json",
                logs / "mimic_lv_vjepa21_vitb_job_v3.json",
            ],
            queue_status,
        )
        _run_wave(
            repository,
            logs,
            queue_status,
            "mimic_lv_bias_corrected_ema",
            [
                ("mimic_lv_dinov3_vitb_ema99", [*config, "mimic-lv", "train", "--backbone", "dinov3_vitb", "--device", "1", "--epochs", "12", "--frozen-epochs", "1", "--ema-decay", "0.99", "--run-tag", "ema99"]),
                ("mimic_lv_vjepa21_vitb_ema99", [*config, "mimic-lv", "train", "--backbone", "vjepa21_vitb", "--device", "0", "--epochs", "12", "--frozen-epochs", "1", "--ema-decay", "0.99", "--run-tag", "ema99"]),
            ],
        )
        _run_wave(
            repository,
            logs,
            queue_status,
            "ted_temporal_smoke",
            [
                ("ted_temporal_vjepa21_smoke", [*config, "temporal", "smoke", "--backbone", "vjepa21_vitb", "--device", "0"]),
                ("ted_temporal_dinov3_smoke", [*config, "temporal", "smoke", "--backbone", "dinov3_vitb", "--device", "1"]),
            ],
        )
        _run_wave(
            repository,
            logs,
            queue_status,
            "ted_temporal_full",
            [
                ("ted_temporal_vjepa21_full", [*config, "temporal", "train", "--backbone", "vjepa21_vitb", "--device", "0"]),
                ("ted_temporal_dinov3_full", [*config, "temporal", "train", "--backbone", "dinov3_vitb", "--device", "1"]),
            ],
        )
        _run_wave(
            repository,
            logs,
            queue_status,
            "view_and_landmark_smoke",
            [
                ("ev9v_view_smoke", [*config, "ev9v-view", "smoke", "--device", "0"]),
                ("unity_landmarks_smoke", [*config, "unity-landmarks", "smoke", "--device", "1"]),
            ],
        )
        _run_wave(
            repository,
            logs,
            queue_status,
            "view_and_landmark_full",
            [
                ("ev9v_view_full", [*config, "ev9v-view", "train", "--device", "0"]),
                ("unity_landmarks_full", [*config, "unity-landmarks", "train", "--device", "1"]),
            ],
        )
        _run_wave(
            repository,
            logs,
            queue_status,
            "external_ood_smoke",
            [("external_ood_smoke", [*config, "external-ood", "smoke", "--device", "0"])],
        )
        _run_wave(
            repository,
            logs,
            queue_status,
            "external_ood_full",
            [("external_ood_full", [*config, "external-ood", "audit", "--device", "0"])],
        )
        _queue_status(queue_status, "complete")
        return 0
    except Exception as error:
        _queue_status(queue_status, "failed", error=f"{type(error).__name__}: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

