from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _inventory(root: Path) -> tuple[list[Path], int]:
    files = [item for item in root.rglob("*") if item.is_file()]
    return files, sum(item.stat().st_size for item in files)


def _tree_digest(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a dataset archive and extend its audit status")
    parser.add_argument("seven_zip", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("status", type=Path)
    args = parser.parse_args()

    current = json.loads(args.status.read_text(encoding="utf-8"))
    current.update(
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "extracting",
            "archive": str(args.archive.resolve()),
            "extraction_root": str(args.destination.resolve()),
        }
    )
    _write_json(args.status, current)
    args.destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = subprocess.run(
        [str(args.seven_zip), "x", str(args.archive), f"-o{args.destination}", "-y", "-bb1"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        current.update(
            {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": "extraction_failed",
                "error": result.stderr[-4000:] or result.stdout[-4000:],
            }
        )
        _write_json(args.status, current)
        return result.returncode

    files, extracted_bytes = _inventory(args.destination)
    current.update(
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "extracted_complete",
            "extracted_files": len(files),
            "extracted_bytes": extracted_bytes,
            "extraction_seconds": round(time.perf_counter() - started, 3),
            "extracted_tree_manifest_sha256": _tree_digest(args.destination, files),
        }
    )
    _write_json(args.status, current)
    print(json.dumps(current, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

