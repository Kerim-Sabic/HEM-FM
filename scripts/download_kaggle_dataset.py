from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import kagglehub


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _inventory(root: Path) -> tuple[int, int]:
    files = [item for item in root.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a public Kaggle dataset with an auditable local status file")
    parser.add_argument("handle")
    parser.add_argument("destination", type=Path)
    parser.add_argument("status", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    base: dict[str, object] = {
        "schema_version": 1,
        "updated_utc": _utc_now(),
        "name": args.name,
        "phase": "downloading",
        "source_url": f"https://www.kaggle.com/datasets/{args.handle}",
        "source_handle": args.handle,
        "intended_role": args.role,
        "track": "R",
        "mutable_storage": "local C drive",
        "promotion_policy": "development-only; promote only after patient-disjoint held-out improvement",
        "locked_test_accessed": False,
    }
    _write_json(args.status, base)
    try:
        downloaded = Path(kagglehub.dataset_download(args.handle, output_dir=str(args.destination)))
        file_count, downloaded_bytes = _inventory(downloaded)
        if file_count == 0:
            raise RuntimeError("download completed without any files")
        completed = {
            **base,
            "updated_utc": _utc_now(),
            "phase": "complete",
            "progress_fraction": 1.0,
            "complete_files": file_count,
            "downloaded_bytes": downloaded_bytes,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "local_root": str(downloaded.resolve()),
            "tree_manifest_sha256": _tree_digest(downloaded),
        }
        _write_json(args.status, completed)
        print(json.dumps(completed, indent=2))
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        lowered = message.lower()
        phase = "requires_auth_or_consent" if any(term in lowered for term in ("401", "403", "unauthor", "forbidden", "credential", "consent")) else "failed"
        failed = {
            **base,
            "updated_utc": _utc_now(),
            "phase": phase,
            "error": message,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        _write_json(args.status, failed)
        print(json.dumps(failed, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

