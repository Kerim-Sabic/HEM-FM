from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY = "timm/vit_base_patch16_dinov3.lvd1689m"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[3] / "work" / "hemfm-v4-runtime" / "checkpoints" / "dinov3"
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPOSITORY,
        local_dir=root,
        allow_patterns=["*.safetensors", "*.json", "README.md", "LICENSE*"],
    )
    weight_files = sorted(root.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError("DINOv3 download completed without a safetensors weight file")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "track": "R",
        "repository": REPOSITORY,
        "architecture": "DINOv3 ViT-B/16",
        "parameter_count": 85_600_000,
        "license": "dinov3-license",
        "weights": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in weight_files
        ],
    }
    (root / "asset-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

