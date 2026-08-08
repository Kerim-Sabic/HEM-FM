from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a public Hugging Face dataset snapshot")
    parser.add_argument("repo")
    parser.add_argument("destination", type=Path)
    parser.add_argument("status", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    base: dict[str, object] = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "phase": "downloading",
        "source_url": f"https://huggingface.co/datasets/{args.repo}",
        "source_revision": args.revision,
        "intended_role": args.role,
        "licence": args.license,
        "track": "R",
        "mutable_storage": "local C drive",
        "promotion_policy": "robustness evaluation first; never promote without held-out benefit",
        "locked_test_accessed": False,
    }
    _write_json(args.status, base)
    started = time.perf_counter()
    try:
        root = Path(
            snapshot_download(
                repo_id=args.repo,
                repo_type="dataset",
                revision=args.revision,
                local_dir=args.destination,
                # EchoXFlow consists of many Zarr chunks.  A single transfer
                # worker avoids duplicating large HTTP/Xet buffers while two
                # training jobs are already using host memory.
                max_workers=1,
            )
        )
        files = [item for item in root.rglob("*") if item.is_file()]
        digest = hashlib.sha256()
        for item in sorted(files):
            digest.update(item.relative_to(root).as_posix().encode("utf-8"))
            digest.update(str(item.stat().st_size).encode("ascii"))
        completed = {
            **base,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "complete",
            "progress_fraction": 1.0,
            "complete_files": len(files),
            "downloaded_bytes": sum(item.stat().st_size for item in files),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "local_root": str(root.resolve()),
            "tree_manifest_sha256": digest.hexdigest(),
        }
        _write_json(args.status, completed)
        print(json.dumps(completed, indent=2))
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        lowered = message.lower()
        if "429" in lowered or "too many requests" in lowered:
            phase = "rate_limited"
        elif any(term in lowered for term in ("401", "403", "unauthor", "forbidden", "gated")):
            phase = "requires_auth_or_consent"
        else:
            phase = "failed"
        _write_json(args.status, {**base, "updated_utc": datetime.now(timezone.utc).isoformat(), "phase": phase, "error": message})
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

