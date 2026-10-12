"""JSON artifacts, checksums, and source provenance without GPU imports."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

MODEL_PATTERNS = ("*.safetensors", "*.bin", "*.json", "*.txt", "*.model")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
        return subprocess.check_output(
            ["git", "-C", str(path), *args], stderr=subprocess.DEVNULL
        )

    try:
        root = Path(git("rev-parse", "--show-toplevel").decode().strip())
        status = git("status", "--porcelain", "--", str(path)).decode()
        patch = git("diff", "HEAD", "--binary", "--", str(path))
        names = (
            git(
                "ls-files",
                "--full-name",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                str(path),
            )
            .decode()
            .split("\0")
        )
        untracked = {
            n: file_hash(root / n) for n in names if n and (root / n).is_file()
        }
        remote = "origin"
        try:
            branch = git("symbolic-ref", "--short", "HEAD").decode().strip()
            remote = git("config", "--get", f"branch.{branch}.remote").decode().strip()
        except subprocess.CalledProcessError:
            pass
        try:
            remote_url = public_git_url(
                git("remote", "get-url", remote).decode().strip()
            )
        except subprocess.CalledProcessError:
            remote_url = None
        return {
            "path": str(path),
            "remote_url": remote_url,
            "subdirectory": str(path.relative_to(root)),
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(status),
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "untracked_sha256": digest(untracked),
        }
    except (OSError, subprocess.CalledProcessError):
        files = {
            str(p.relative_to(path)): file_hash(p)
            for p in sorted(path.rglob("*.py"))
            if "__pycache__" not in p.parts
        }
        return {"path": str(path), "tree_sha256": digest(files)}


def public_git_url(value):
    """Record repository locations without Git credentials or URL query tokens."""
    if "://" not in value:
        match = re.fullmatch(r"(?:[^@]+@)?([^/:]+):(.+)", value)
        if not match:
            return None
        value = f"https://{match[1]}/{match[2]}"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https", "ssh", "git"}:
        return None
    hostname = parsed.hostname or ""
    if parsed.scheme in {"http", "https"} and parsed.port:
        hostname += f":{parsed.port}"
    return urlunsplit(("https", hostname, parsed.path, "", ""))


def suite_hash(root):
    root = Path(root)
    paths = list(root.glob("*.py"))
    for directory in ["harness", "entrypoints"]:
        paths.extend((root / directory).rglob("*.py"))
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(paths)})


def model_info(path):
    path = Path(path)
    files = sorted(
        {p for pattern in MODEL_PATTERNS for p in path.glob(pattern) if p.is_file()}
    )
    if not (path / "config.json").is_file() or not any(
        p.suffix in {".bin", ".safetensors"} for p in files
    ):
        raise ValueError(f"{path}: expected config.json and model weights")
    hashes = {p.name: file_hash(p) for p in files}
    return {
        "path": str(path.resolve()),
        "files": hashes,
        "sha256": digest(hashes),
        "config": read(path / "config.json"),
    }


def resolve_model(spec):
    """Fetch a pinned snapshot, or verify an explicit local copy against that pin."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_id, revision = spec["id"], spec["revision"]
    metadata = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    if metadata.sha != revision:
        raise ValueError(f"{repo_id}: Hub revision differs from requested commit")
    expected = {
        item.rfilename: item
        for item in metadata.siblings
        if "/" not in item.rfilename
        and any(fnmatch(item.rfilename, pattern) for pattern in MODEL_PATTERNS)
    }
    path = spec.get("path")
    if not path:
        # A local_dir snapshot contains regular files, so a Docker bind mount of
        # this directory also contains the weights (Hub cache snapshots use symlinks).
        destination = Path(HF_HUB_CACHE) / repo_id.replace("/", "--") / revision
        path = snapshot_download(
            repo_id,
            revision=revision,
            local_dir=destination,
            allow_patterns=list(expected),
        )
    info = model_info(path)
    if set(info["files"]) != set(expected):
        raise ValueError(f"{path}: model files differ from {repo_id}@{revision}")
    for name, item in expected.items():
        if item.lfs:
            matches = info["files"][name] == item.lfs.sha256
        else:
            content = (Path(path) / name).read_bytes()
            blob = f"blob {len(content)}\0".encode() + content
            matches = hashlib.sha1(blob).hexdigest() == item.blob_id
        if not matches:
            raise ValueError(f"{path}/{name}: does not match {repo_id}@{revision}")
    return {**info, "repo_id": repo_id, "revision": revision}
