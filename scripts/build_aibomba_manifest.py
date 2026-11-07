from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {".venv", "__pycache__", "results"}
EXCLUDED_NAMES = {"MANIFEST.sha256"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(bundle: Path) -> list[str]:
    bundle = bundle.resolve(strict=True)
    if not bundle.is_dir():
        raise NotADirectoryError(bundle)
    files = [
        path
        for path in bundle.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED_NAMES
        and not (set(path.relative_to(bundle).parts) & EXCLUDED_PARTS)
    ]
    lines = [
        f"{_sha256(path)}  {path.relative_to(bundle).as_posix()}"
        for path in sorted(files, key=lambda value: value.relative_to(bundle).as_posix())
    ]
    if not lines:
        raise ValueError("AIBomba bundle contains no manifestable files")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the AIBomba package integrity manifest")
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve(strict=True)
    destination = bundle / "MANIFEST.sha256"
    temporary = destination.with_suffix(".sha256.tmp")
    try:
        temporary.write_text("\n".join(build_manifest(bundle)) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
