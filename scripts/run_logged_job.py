from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a long local job with persistent logs")
    parser.add_argument("log", type=Path)
    parser.add_argument("status", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a child command is required")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    started = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "running",
        "command": args.command,
    }
    args.status.write_text(json.dumps(started, indent=2), encoding="utf-8")
    with args.log.open("a", encoding="utf-8") as output:
        output.write(f"\n[{started['updated_utc']}] START {' '.join(args.command)}\n")
        output.flush()
        result = subprocess.run([sys.executable, *args.command], stdout=output, stderr=subprocess.STDOUT)
        finished = {
            **started,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "complete" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
        }
        args.status.write_text(json.dumps(finished, indent=2), encoding="utf-8")
        output.write(f"[{finished['updated_utc']}] EXIT {result.returncode}\n")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

