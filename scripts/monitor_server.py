Exit code: 0
Wall time: 0.9 seconds
Output:
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import psutil


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _gpus() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=4)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 7:
            continue
        rows.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "utilization": float(values[2]),
                "memory_used_mib": float(values[3]),
                "memory_total_mib": float(values[4]),
                "temperature_c": float(values[5]),
                "power_w": float(values[6]),
            }
        )
    return rows


def _training_processes() -> list[dict[str, Any]]:
    rows = []
    for process in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if "hemfm" not in command.lower():
                continue
            memory = process.info.get("memory_info")
            rows.append(
                {
                    "pid": process.info["pid"],
                    "name": process.info.get("name") or "process",
                    "cpu_percent": process.info.get("cpu_percent") or 0.0,
                    "memory_mib": round((memory.rss if memory else 0) / 1024**2, 1),
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    return rows


def _payload(run_directory: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(run_directory.anchor or run_directory))
    dataset_downloads = []
    for path in sorted(run_directory.glob("dataset_download_status*.json")):
        row = _read_json(path)
        if row:
            dataset_downloads.append(row)
    dense_jobs = []
    dense_status_paths = sorted(run_directory.glob("dense_lv*_status.json"))
    specific_vjepa_status = run_directory / "dense_lv_vjepa21_vitb_status.json"
    if specific_vjepa_status.exists():
        dense_status_paths = [
            path for path in dense_status_paths if path.name != "dense_lv_status.json"
        ]
    for path in dense_status_paths:
        row = _read_json(path)
        if row:
            dense_jobs.append(row)
    echonet_jobs = []
    for path in sorted(run_directory.glob("echonet_dynamic*_status*.json")):
        row = _read_json(path)
        if row:
            echonet_jobs.append(row)
    staged_root = run_directory.parent / "staged_final"
    staged_cache = _read_json(staged_root / "cache_status.json")
    staged_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(staged_root.glob("*/seed_*/status.json")):
        row = _read_json(path)
        if row:
            mode = "smoke" if path.parent.name.endswith("_smoke") else "full"
            row["mode"] = mode
            key = (str(row.get("target", "")), int(row.get("seed", -1)))
            previous = staged_by_run.get(key)
            if previous is None or (previous.get("mode") == "smoke" and mode == "full"):
                staged_by_run[key] = row
    staged_jobs = sorted(
        staged_by_run.values(),
        key=lambda row: (str(row.get("target", "")), int(row.get("seed", -1))),
    )
    current_status = _read_json(run_directory / "status.json")
    if staged_cache and int(staged_cache.get("complete", 0)) < int(
        staged_cache.get("total", 0)
    ):
        current_status = staged_cache
    active_staged_jobs = [
        row
        for row in staged_jobs
        if int(row.get("epoch", 0)) < int(row.get("epochs", 0))
    ]
    if active_staged_jobs:
        current_status = max(
            active_staged_jobs, key=lambda row: str(row.get("updated_utc", ""))
        )
    elif echonet_jobs:
        current_status = max(
            echonet_jobs, key=lambda row: str(row.get("updated_utc", ""))
        )
    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": current_status,
        "dataset_download": _read_json(run_directory / "dataset_download_status.json"),
        "dataset_downloads": dataset_downloads,
        "dense_jobs": dense_jobs,
        "echonet_jobs": echonet_jobs,
        "staged_cache": staged_cache,
        "staged_jobs": staged_jobs,
        "gpus": _gpus(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_used_gib": round(memory.used / 1024**3, 1),
            "ram_total_gib": round(memory.total / 1024**3, 1),
            "disk_free_gib": round(disk.free / 1024**3, 1),
        },
        "processes": _training_processes(),
    }


def serve(run_directory: Path, html_path: Path, host: str, port: int) -> None:
    cache: dict[str, Any] = {"updated_utc": datetime.now(timezone.utc).isoformat()}
    cache_lock = threading.Lock()

    def refresh_cache() -> None:
        while True:
            try:
                latest = _payload(run_directory)
                with cache_lock:
                    cache.clear()
                    cache.update(latest)
            except Exception as exc:
                with cache_lock:
                    cache["monitor_error"] = f"{type(exc).__name__}: {exc}"
            time.sleep(2)

    threading.Thread(target=refresh_cache, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/status":
                with cache_lock:
                    body = json.dumps(cache).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in {"/", "/index.html"}:
                body = html_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def log_message(self, format: str, *args: object) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local HEM-FM training dashboard")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--html",
        type=Path,
        default=Path(__file__).with_name("live_dashboard.html"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.run_directory.resolve(), args.html.resolve(), args.host, args.port)


if __name__ == "__main__":
    main()

