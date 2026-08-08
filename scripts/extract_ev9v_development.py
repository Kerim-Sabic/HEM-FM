from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _labels(path: Path, split: str) -> dict[str, tuple[str, str]]:
    output: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, label = line.strip().rsplit(maxsplit=1)
        if name in output:
            raise ValueError(f"duplicate EV9V video name in {path.name}: {name}")
        output[name] = (split, label)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract EV9V development videos without touching the test split")
    parser.add_argument("root", type=Path)
    parser.add_argument("status", type=Path)
    args = parser.parse_args()

    selected = _labels(args.root / "train_labeled.txt", "train")
    validation = _labels(args.root / "validation_labeled.txt", "validation")
    overlap = set(selected) & set(validation)
    if overlap:
        raise RuntimeError(f"EV9V train/validation overlap: {sorted(overlap)[:5]}")
    selected.update(validation)
    current = json.loads(args.status.read_text(encoding="utf-8"))
    current.update(
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "extracting_development_only",
            "development_videos_expected": len(selected),
            "locked_test_accessed": False,
        }
    )
    _write_json(args.status, current)

    output_root = args.root / "development" / "Videos"
    output_root.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with tarfile.open(args.root / "Videos.tar", mode="r:") as archive:
        for member in archive:
            pure = PurePosixPath(member.name)
            if not member.isfile() or len(pure.parts) != 2 or pure.parts[0] != "Videos":
                continue
            name = pure.stem
            if name not in selected:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"unable to read selected member: {member.name}")
            destination = output_root / pure.name
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            extracted[name] = destination

    missing = sorted(set(selected) - set(extracted))
    if missing:
        raise RuntimeError(f"missing {len(missing)} EV9V development videos; first: {missing[:5]}")
    manifest = args.root / "development_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["video_name", "split", "view", "local_path"])
        writer.writeheader()
        for name in sorted(selected):
            split, label = selected[name]
            writer.writerow(
                {
                    "video_name": name,
                    "split": split,
                    "view": label,
                    "local_path": str(extracted[name].resolve()),
                }
            )
    total_bytes = sum(path.stat().st_size for path in extracted.values())
    current.update(
        {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "phase": "development_extracted_complete",
            "development_videos_extracted": len(extracted),
            "development_extracted_bytes": total_bytes,
            "train_videos": sum(split == "train" for split, _ in selected.values()),
            "validation_videos": sum(split == "validation" for split, _ in selected.values()),
            "manifest": str(manifest.resolve()),
            "locked_test_accessed": False,
        }
    )
    _write_json(args.status, current)
    print(json.dumps(current, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

