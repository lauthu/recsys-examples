"""Pinned model downloads and source identities remain reproducible."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.artifacts import public_git_url, resolve_model, source_info


class ProvenanceTests(unittest.TestCase):
    def test_local_checkpoint_must_match_pinned_hub_contents(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text('{"vocab_size": 32}')
            (path / "model.safetensors").write_bytes(b"fixture weights")
            siblings = []
            for file in sorted(path.iterdir()):
                content = file.read_bytes()
                siblings.append(
                    SimpleNamespace(
                        rfilename=file.name,
                        lfs=(
                            SimpleNamespace(sha256=hashlib.sha256(content).hexdigest())
                            if file.suffix == ".safetensors"
                            else None
                        ),
                        blob_id=hashlib.sha1(
                            f"blob {len(content)}\0".encode() + content
                        ).hexdigest(),
                    )
                )
            hub = SimpleNamespace(
                HfApi=Mock(), snapshot_download=Mock(return_value=directory)
            )
            hub.HfApi.return_value.model_info.return_value = SimpleNamespace(
                sha=revision, siblings=siblings
            )
            spec = {"id": "fixture/model", "revision": revision, "path": directory}
            with patch.dict(
                sys.modules,
                {
                    "huggingface_hub": hub,
                    "huggingface_hub.constants": SimpleNamespace(
                        HF_HUB_CACHE=directory
                    ),
                },
            ):
                model = resolve_model(spec)
                self.assertEqual(model["repo_id"], spec["id"])
                self.assertEqual(model["revision"], revision)
                hub.snapshot_download.assert_not_called()

                del spec["path"]
                resolve_model(spec)
                self.assertEqual(
                    hub.snapshot_download.call_args.kwargs["revision"], revision
                )
                self.assertEqual(
                    hub.snapshot_download.call_args.kwargs["local_dir"],
                    path / "fixture--model" / revision,
                )

                spec["path"] = directory
                for name in ["model.safetensors", "config.json"]:
                    with self.subTest(tampered_file=name):
                        original = (path / name).read_bytes()
                        (path / name).write_bytes(original + b" ")
                        with self.assertRaisesRegex(ValueError, "does not match"):
                            resolve_model(spec)
                        (path / name).write_bytes(original)

                (path / "model.safetensors").unlink()
                with self.assertRaisesRegex(ValueError, "model weights"):
                    resolve_model(spec)

    def test_source_uses_tracking_remote_without_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "engine"
            source.mkdir()
            (source / "engine.py").write_text("pass\n")

            def git(*args):
                return (
                    subprocess.check_output(
                        ["git", "-C", directory, *args], stderr=subprocess.DEVNULL
                    )
                    .decode()
                    .strip()
                )

            git("init")
            git("add", ".")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-m",
                "fixture",
            )
            git("remote", "add", "origin", "https://github.com/upstream/engine.git")
            git(
                "remote",
                "add",
                "personal",
                "https://user:secret@github.com/owner/engine.git?token=secret",
            )
            branch = git("symbolic-ref", "--short", "HEAD")
            git("config", f"branch.{branch}.remote", "personal")
            info = source_info(source)
            self.assertEqual(info["remote_url"], "https://github.com/owner/engine.git")
            self.assertEqual(info["commit"], git("rev-parse", "HEAD"))
            self.assertEqual(info["subdirectory"], "engine")
            self.assertFalse(info["dirty"])
            self.assertNotIn("secret", str(info))
        self.assertEqual(
            public_git_url("git@github.com:owner/engine.git"),
            "https://github.com/owner/engine.git",
        )


if __name__ == "__main__":
    unittest.main()
