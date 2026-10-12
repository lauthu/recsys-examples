"""JSON artifacts, checksums, and source provenance without GPU imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_info(path):
    if not path or not Path(path).exists():
        return None
    path = Path(path).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.DEVNULL)
    try:
        root = Path(git("rev-parse", "--show-toplevel").decode().strip())
        status = git("status", "--porcelain", "--", str(path)).decode()
        patch = git("diff", "HEAD", "--binary", "--", str(path))
        names = git("ls-files", "--full-name", "--others", "--exclude-standard", "-z", "--", str(path)).decode().split("\0")
        untracked = {n: file_hash(root / n) for n in names if n and (root / n).is_file()}
        return {"path": str(path), "commit": git("rev-parse", "HEAD").decode().strip(),
                "dirty": bool(status), "patch_sha256": hashlib.sha256(patch).hexdigest(),
                "untracked_sha256": digest(untracked)}
    except (OSError, subprocess.CalledProcessError):
        files = {str(p.relative_to(path)): file_hash(p) for p in sorted(path.rglob("*.py")) if "__pycache__" not in p.parts}
        return {"path": str(path), "tree_sha256": digest(files)}


def suite_hash(root):
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(Path(root).rglob("*.py"))
             if not set(p.relative_to(root).parts) & {"results", ".cache", "tests", "__pycache__"}}
    return digest(files)


def model_info(path):
    path = Path(path)
    patterns = ("*.safetensors", "*.bin", "*.json", "*.txt", "*.model")
    files = sorted({p for pattern in patterns for p in path.glob(pattern) if p.is_file()})
    if not (path / "config.json").is_file() or not any(p.suffix in {".bin", ".safetensors"} for p in files):
        raise ValueError(f"{path}: expected config.json and model weights")
    hashes = {p.name: file_hash(p) for p in files}
    return {"path": str(path.resolve()), "files": hashes, "sha256": digest(hashes), "config": read(path / "config.json")}
