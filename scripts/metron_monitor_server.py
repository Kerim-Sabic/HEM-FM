from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
from typing import Any

import psutil


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def gpu_status() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "utilization": float(fields[2]),
                "memory_used_mib": float(fields[3]),
                "memory_total_mib": float(fields[4]),
                "temperature_c": float(fields[5]),
                "power_w": float(fields[6]),
            }
        )
    return rows


def metron_processes() -> list[dict[str, Any]]:
    rows = []
    for process in psutil.process_iter(
        ["pid", "name", "cpu_percent", "memory_info", "cmdline"]
    ):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if not any(word in command.lower() for word in ("metron", "hemfm")):
                continue
            memory = process.info.get("memory_info")
            rows.append(
                {
                    "pid": int(process.info["pid"]),
                    "name": str(process.info.get("name") or "process"),
                    "cpu_percent": float(process.info.get("cpu_percent") or 0),
                    "memory_mib": round((memory.rss if memory else 0) / 1024**2, 1),
                }
            )
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    return rows


def payload(runtime_root: Path) -> dict[str, Any]:
    status = read_json(runtime_root / "status" / "status.json")
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(runtime_root))
    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "programme": status,
        "gpus": gpu_status(),
        "processes": metron_processes(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_used_gib": round(memory.used / 1024**3, 1),
            "ram_total_gib": round(memory.total / 1024**3, 1),
            "disk_free_gib": round(disk.free / 1024**3, 1),
            "disk_total_gib": round(disk.total / 1024**3, 1),
        },
    }


def serve(runtime_root: Path, html_path: Path, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/status":
                body = json.dumps(payload(runtime_root)).encode("utf-8")
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
    parser = argparse.ArgumentParser(description="METRON v8 live developer dashboard")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--html", type=Path, default=Path(__file__).with_name("metron_dashboard.html")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args()
    serve(args.runtime_root.resolve(), args.html.resolve(), args.host, args.port)


if __name__ == "__main__":
    main()
