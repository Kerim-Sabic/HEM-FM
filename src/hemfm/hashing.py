from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def private_hash(value: object, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), str(value).encode("utf-8"), hashlib.sha256).hexdigest()

