from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


def _atomic_status(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def download(url: str, destination: Path, status_path: Path, name: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    existing = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HEM-FM-research-downloader/1.0",
            "Referer": f"{urllib.parse.urlparse(url).scheme}://{urllib.parse.urlparse(url).netloc}/",
        },
    )
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        partial_response = response.status == 206
        if existing and not partial_response:
            existing = 0
        mode = "ab" if partial_response else "wb"
        response_bytes = int(response.headers.get("Content-Length") or 0)
        total = existing + response_bytes if response_bytes else None
        downloaded = existing
        with temporary.open(mode) as stream:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.perf_counter() - started, 1e-6)
                _atomic_status(
                    status_path,
                    {
                        "schema_version": 1,
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "name": name,
                        "phase": "downloading",
                        "downloaded_bytes": downloaded,
                        "total_bytes": total,
                        "progress_fraction": downloaded / total if total else None,
                        "mib_per_second": round((downloaded - existing) / 1024**2 / elapsed, 2),
                        "destination": str(destination),
                        "source_url": url,
                        "track": "R",
                    },
                )
    digest = hashlib.sha256()
    with temporary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    temporary.replace(destination)
    _atomic_status(
        status_path,
        {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "phase": "complete",
            "downloaded_bytes": destination.stat().st_size,
            "total_bytes": destination.stat().st_size,
            "progress_fraction": 1.0,
            "sha256": digest.hexdigest(),
            "destination": str(destination),
            "source_url": url,
            "track": "R",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Research Track dataset download")
    parser.add_argument("--url", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    try:
        download(args.url, args.destination, args.status, args.name)
    except Exception as exc:
        _atomic_status(
            args.status,
            {
                "schema_version": 1,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "name": args.name,
                "phase": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "destination": str(args.destination),
                "source_url": args.url,
                "track": "R",
            },
        )
        raise


if __name__ == "__main__":
    main()

