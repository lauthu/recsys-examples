"""CPU regression tests for worker resources and process cleanup."""

import json
import os
import signal
import sys
import tempfile
import time
import unittest
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.runtime import command, cpu_info, execute, select_cpu_affinity


class RuntimeTests(unittest.TestCase):
    def test_selects_distinct_allowed_physical_cores(self):
        topology = "# CPU,Core,Socket\n16,0,1\n10,0,0\n2,0,0\n3,1,0\n17,1,1\n"
        with patch(
            "harness.runtime.subprocess.check_output", return_value=topology
        ), patch(
            "harness.runtime.os.sched_getaffinity", return_value={2, 3, 10, 16, 17}
        ):
            self.assertEqual(select_cpu_affinity(3), [2, 3, 16])
            with self.assertRaisesRegex(ValueError, "only 4 physical cores"):
                select_cpu_affinity(5)

    def test_recorded_affinity_must_still_be_available(self):
        with patch("harness.runtime.os.sched_getaffinity", return_value={2, 3}):
            with self.assertRaisesRegex(ValueError, "no longer available"):
                cpu_info([2, 4])

    def test_local_worker_uses_selected_affinity(self):
        cpus = [min(os.sched_getaffinity(0))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {
                "run_dir": directory,
                "execution": {"gpu": "GPU-test", "cpu_affinity": cpus},
                "solution": {
                    "name": "test",
                    "runtime": {"kind": "local", "python": sys.executable},
                },
            }
            cmd, env, container = command(
                root / "request.json", root / "result.json", request
            )
            cmd[cmd.index("-m") :] = [
                "-c",
                "import json, os; print(json.dumps(sorted(os.sched_getaffinity(0))))",
            ]
            log = root / "worker.log"
            result = execute(cmd, env, log, timeout=5, container=container)
            self.assertEqual(result, {"returncode": 0, "timed_out": False})
            self.assertEqual(json.loads(log.read_text()), cpus)

    def test_timeout_kills_child_after_parent_exits(self):
        script = """
import os
import signal
import sys
import time

if os.fork() == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with open(sys.argv[1], "w") as stream:
        stream.write(str(os.getpid()))
time.sleep(60)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root / "child.pid"
            try:
                result = execute(
                    [sys.executable, "-c", script, str(pidfile)],
                    os.environ.copy(),
                    root / "worker.log",
                    timeout=1,
                )
                self.assertTrue(result["timed_out"])
                self.assertEqual(result["returncode"], -signal.SIGTERM)
                child = int(pidfile.read_text())
                stat = Path(f"/proc/{child}/stat")
                deadline = time.monotonic() + 1
                while True:
                    try:
                        if stat.read_text().split()[2] == "Z":
                            break
                    except FileNotFoundError:
                        break
                    self.assertLess(
                        time.monotonic(), deadline, "Child survived timeout"
                    )
                    time.sleep(0.01)
            finally:
                if pidfile.exists():
                    with suppress(ProcessLookupError):
                        os.kill(int(pidfile.read_text()), signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
