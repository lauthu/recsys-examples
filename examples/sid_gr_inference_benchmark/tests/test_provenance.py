# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import subprocess

import pytest

from bench import provenance

COMMIT = "a" * 40
PIN = {"url": "https://github.com/example/engine", "commit": COMMIT}


def git(path, *args):
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(path),
        "PATH": "/usr/bin:/bin",
    }
    done = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, env=env
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A checkout with engine code under python/ and unrelated files beside it."""
    git(tmp_path, "init", "-q")
    (tmp_path / "python").mkdir()
    (tmp_path / "python/engine.py").write_text("v1\n")
    (tmp_path / "docs.md").write_text("v1\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "one")
    return tmp_path


def pin_at(repo, revision="HEAD"):
    return {**PIN, "commit": git(repo, "rev-parse", revision)}


def test_pin_shapes():
    provenance.check_pin(PIN)
    for bad in (
        {**PIN, "commit": "df191d6"},  # abbreviated
        {**PIN, "commit": "A" * 40},
        {**PIN, "url": "git@github.com:x/y.git"},
        {"url": PIN["url"]},
    ):
        with pytest.raises(ValueError):
            provenance.check_pin(bad)


def test_the_pinned_commit_passes(repo):
    entry = provenance.verify_source(repo, pin_at(repo), ["python"])
    assert entry["enforced"] and entry["differences"] == []
    assert entry["head"] == entry["commit"]


def test_commits_that_leave_the_engine_paths_alone_still_pass(repo):
    pin = pin_at(repo)
    (repo / "docs.md").write_text("v2\n")
    git(repo, "commit", "-qam", "docs only")
    assert provenance.verify_source(repo, pin, ["python"])["head"] != pin["commit"]


def test_changed_engine_code_is_refused(repo):
    pin = pin_at(repo)
    (repo / "python/engine.py").write_text("v2\n")
    git(repo, "commit", "-qam", "engine change")
    with pytest.raises(ValueError, match="differs from pinned commit"):
        provenance.verify_source(repo, pin, ["python"])
    entry = provenance.verify_source(repo, pin, ["python"], allow_dirty=True)
    assert not entry["enforced"] and "differs" in entry["differences"][0]


def test_uncommitted_changes_are_refused(repo):
    (repo / "python/engine.py").write_text("edited\n")
    with pytest.raises(ValueError, match="uncommitted changes"):
        provenance.verify_source(repo, pin_at(repo), ["python"])


def test_unrelated_uncommitted_changes_are_ignored(repo):
    (repo / "docs.md").write_text("edited\n")
    assert provenance.verify_source(repo, pin_at(repo), ["python"])["enforced"]


def test_a_missing_pinned_commit_says_what_to_fetch(repo):
    with pytest.raises(ValueError, match="does not contain pinned commit"):
        provenance.verify_source(repo, PIN, ["python"])


def test_a_directory_that_is_not_a_checkout_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a git checkout"):
        provenance.verify_source(tmp_path, PIN, ["python"])


def checkpoint(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    for name, data in files.items():
        (model / name).write_bytes(data)
    for other in ("tokenizer.json", "generation_config.json", "tokenizer_config.json"):
        (model / other).write_text("not part of the pin")
    pin = {
        "id": "Org/Model",
        "revision": COMMIT,
        "sha256": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    }
    return model, pin


def test_checkpoint_matching_its_pin_is_accepted(tmp_path):
    model, pin = checkpoint(tmp_path)
    provenance.check_model_pin(pin)
    assert provenance.verify_model(model, pin)["sha256"] == pin["sha256"]


def test_changed_weights_are_refused(tmp_path):
    model, pin = checkpoint(tmp_path)
    (model / "model.safetensors").write_bytes(b"other")
    with pytest.raises(
        ValueError, match=r"differs from model.sha256 for \['model.safetensors'\]"
    ):
        provenance.verify_model(model, pin)


def test_extra_or_missing_weight_files_are_refused(tmp_path):
    model, pin = checkpoint(tmp_path)
    (model / "model-2.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match="extra"):
        provenance.verify_model(model, pin)
    (model / "model-2.safetensors").unlink()
    (model / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        provenance.verify_model(model, pin)


@pytest.mark.parametrize(
    "change",
    [
        {"revision": "main"},
        {"sha256": {"model.safetensors": "0" * 64}},
        {"sha256": {"config.json": "abc"}},
    ],
)
def test_model_pins_must_be_complete(tmp_path, change):
    _, pin = checkpoint(tmp_path)
    with pytest.raises(ValueError):
        provenance.check_model_pin({**pin, **change})
