from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download selected public Hugging Face dataset files")
    parser.add_argument("repo")
    parser.add_argument("destination", type=Path)
    parser.add_argument("status", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--file", action="append", required=True)
    args = parser.parse_args()

    base: dict[str, object] = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "phase": "downloading",
        "source_url": f"https://huggingface.co/datasets/{args.repo}",
        "source_revision": args.revision,
        "selected_files": args.file,
        "intended_role": args.role,
        "licence": args.license,
        "track": "R",
        "mutable_storage": "local C drive",
        "promotion_policy": "development-only; promote only after patient-disjoint held-out improvement",
        "locked_test_accessed": False,
    }
    _write_json(args.status, base)
    started = time.perf_counter()
    manifests: list[dict[str, object]] = []
    try:
        for index, filename in enumerate(args.file, start=1):
            local = Path(
                hf_hub_download(
                    repo_id=args.repo,
                    filename=filename,
                    repo_type="dataset",
                    revision=args.revision,
                    local_dir=args.destination,
                )
            )
            manifests.append(
                {
                    "filename": filename,
                    "bytes": local.stat().st_size,
                    "sha256": _sha256(local),
                }
            )
            _write_json(
                args.status,
                {
                    **base,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "phase": "downloading" if index < len(args.file) else "verifying",
                    "complete_files": index,
                    "total_files": len(args.file),
                    "downloaded_bytes": sum(int(item["bytes"]) for item in manifests),
                },
            )
        completed = {
            **base,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "complete",
            "progress_fraction": 1.0,
            "complete_files": len(manifests),
            "total_files": len(manifests),
            "downloaded_bytes": sum(int(item["bytes"]) for item in manifests),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "local_root": str(args.destination.resolve()),
            "files": manifests,
        }
        _write_json(args.status, completed)
        print(json.dumps(completed, indent=2))
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        lowered = message.lower()
        phase = "requires_auth_or_consent" if any(term in lowered for term in ("401", "403", "unauthor", "forbidden", "gated", "token")) else "failed"
        _write_json(
            args.status,
            {
                **base,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": phase,
                "error": message,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

