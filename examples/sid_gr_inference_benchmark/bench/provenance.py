# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pins and identities on the host: source commit, model files, Docker image.

Pins live in the configuration. A run refuses to start unless the engine source
and the checkpoint match them (`--allow-dirty` skips the source checks and the
report says so). The result is a manifest that the report prints.
"""

import hashlib
import json
import re
import subprocess

from bench.engines import fail

SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
# Files that define a checkpoint's numerics; tokenizer files are not read.
MODEL_SUFFIXES = (".safetensors", ".safetensors.index.json")
IMAGE_LABELS = (
    "org.opencontainers.image.source",
    "org.opencontainers.image.revision",
    "org.opencontainers.image.version",
)


def check_pin(pin, name="source"):
    """Validate a {"url", "commit"} pin from a configuration."""
    if not isinstance(pin, dict) or set(pin) != {"url", "commit"}:
        fail(f'{name} needs exactly the keys ["commit", "url"]')
    if type(pin["url"]) is not str or not pin["url"].startswith("https://"):
        fail(f"{name}.url must be an https:// URL")
    if type(pin["commit"]) is not str or not SHA1.fullmatch(pin["commit"]):
        fail(f"{name}.commit must be a full 40-character lowercase commit ID")


def check_model_pin(model):
    keys = ("id", "revision", "sha256")
    if not isinstance(model, dict) or set(model) != set(keys):
        fail(f"model needs exactly the keys {sorted(keys)}")
    if type(model["id"]) is not str or not model["id"]:
        fail("model.id must be a Hub model ID such as Qwen/Qwen3-0.6B")
    if type(model["revision"]) is not str or not SHA1.fullmatch(model["revision"]):
        fail("model.revision must be a full 40-character Hub commit ID")
    hashes = model["sha256"]
    if not isinstance(hashes, dict) or "config.json" not in hashes:
        fail("model.sha256 must map file names, including config.json, to SHA-256")
    if any(not SHA256.fullmatch(str(value)) for value in hashes.values()):
        fail("model.sha256 values must be 64-character lowercase SHA-256 digests")


def git(path, *args):
    command = ["git", "-C", str(path), *args]
    done = subprocess.run(command, capture_output=True, text=True)
    return done.returncode, done.stdout.strip()


def verify_source(path, pin, paths, allow_dirty=False):
    """Check the checkout against its pin; return its manifest entry.

    The pinned commit must exist in the checkout. Every path the engine reads
    must have the same content at HEAD as at the pinned commit and must have no
    uncommitted changes, so any commit that leaves those paths untouched also
    passes. `allow_dirty` records the differences instead of failing.
    """
    commit = pin["commit"]
    code, head = git(path, "rev-parse", "HEAD")
    if code:
        fail(f"{path} is not a git checkout; the source pin needs one")
    if git(path, "cat-file", "-e", f"{commit}^{{commit}}")[0]:
        fail(f"{path} does not contain pinned commit {commit}; fetch {pin['url']}")
    problems = []
    for item in paths:
        pinned = git(path, "rev-parse", f"{commit}:{item}")[1]
        if not pinned:
            problems.append(f"{item} does not exist at pinned commit {commit[:12]}")
        elif git(path, "rev-parse", f"HEAD:{item}")[1] != pinned:
            problems.append(f"{item} differs from pinned commit {commit[:12]}")
    status = git(path, "status", "--porcelain", "-uno", "--", *paths)[1]
    if status:
        problems.append(f"{', '.join(paths)} has uncommitted changes")
    if problems and not allow_dirty:
        fail(f"{path}: {'; '.join(problems)} (--allow-dirty runs anyway, unpinned)")
    return {
        "url": pin["url"],
        "commit": commit,
        "head": head,
        "enforced": not allow_dirty,
        "differences": problems,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(directory, pin):
    """Hash the checkpoint files and compare them with the pin."""
    found = {
        p.name: sha256_file(p)
        for p in sorted(directory.iterdir())
        if p.name == "config.json" or p.name.endswith(MODEL_SUFFIXES)
    }
    if set(found) != set(pin["sha256"]):
        extra, missing = set(found) - set(pin["sha256"]), set(pin["sha256"]) - set(
            found
        )
        fail(
            f"{directory} files differ from the pin: extra {sorted(extra)}, missing {sorted(missing)}"
        )
    wrong = sorted(name for name in found if found[name] != pin["sha256"][name])
    if wrong:
        fail(f"{directory}: SHA-256 differs from model.sha256 for {wrong}")
    return {"id": pin["id"], "revision": pin["revision"], "sha256": found}


def image_info(image):
    """Identity of a local image: ID, tags, repository digests and build labels."""
    done = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if done.returncode:
        fail(f"no local Docker image {image}; the runner never pulls images")
    info = json.loads(done.stdout)[0]
    labels = info.get("Config", {}).get("Labels") or {}
    return {
        "id": info["Id"],
        "tags": info.get("RepoTags") or [],
        "repo_digests": info.get("RepoDigests") or [],
        "labels": {key: labels[key] for key in IMAGE_LABELS if key in labels},
    }
