"""CPU regression tests for worker process cleanup."""

import os
import signal
import sys
import tempfile
import time
import unittest
from contextlib import suppress
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.runtime import execute


class RuntimeTests(unittest.TestCase):
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
