"""Reports connect machine-local settings to reproducible public identities."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.reporting.reproduction import configuration_rows


class ReportProvenanceTests(unittest.TestCase):
    def test_source_model_image_and_option_scopes(self):
        revision = "a" * 40
        source = {
            "path": "/checkout/corelib/gr_decode_atten",
            "remote_url": "https://github.com/owner/repo.git",
            "commit": revision,
            "subdirectory": "corelib/gr_decode_atten",
            "dirty": False,
        }
        model = {
            "repo_id": "Qwen/Qwen3-1.7B",
            "revision": revision,
            "path": "/models/qwen",
        }
        solutions = {
            name: {
                "name": name,
                "runtime": {
                    "kind": "docker",
                    "image": f"sha256:{name}",
                    "requested_image": f"image:{name}",
                },
                "engine": {"prefix_cache": True},
                "env": {"GR_DECODE_ATTEN_ROOT": source["path"]}
                if name == "sid"
                else {},
            }
            for name in ["sid", "sglang"]
        }
        cfg = {
            "solutions": list(solutions),
            "solution_configs": solutions,
            "environment": {
                name: {"IMAGE": f"image:{name}", "MODEL": model["path"]}
                for name in solutions
            },
        }
        environment, artifacts, options = configuration_rows(
            cfg,
            [{"model": model}],
            [{"solution": solution} for solution in solutions.values()],
            [{"adapter": {"decode_kernel": source}}, {}],
        )
        images = [row for row in environment if row[0] == "`IMAGE`"]
        self.assertEqual(
            images,
            [
                ["`IMAGE`", "sglang", "`image:sglang`", "`sha256:sglang`"],
                ["`IMAGE`", "sid", "`image:sid`", "`sha256:sid`"],
            ],
        )
        kernel = next(row for row in environment if row[0] == "`GR_DECODE_ATTEN_ROOT`")
        self.assertEqual(kernel[1], "sid")
        self.assertTrue(
            f"https://github.com/owner/repo/tree/{revision}/corelib/gr_decode_atten"
            in kernel[3]
        )
        checkpoint = next(row for row in environment if row[0] == "`MODEL`")
        self.assertEqual(checkpoint[1], "sid, sglang")
        self.assertTrue(
            f"https://huggingface.co/Qwen/Qwen3-1.7B/tree/{revision}" in checkpoint[3]
        )
        self.assertEqual(len(artifacts), 4)
        self.assertEqual(
            options, [[name, "`prefix_cache`", "`true`"] for name in solutions]
        )


if __name__ == "__main__":
    unittest.main()
