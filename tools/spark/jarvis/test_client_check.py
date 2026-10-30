"""client_check.py removes the Chrome profile it made on every way out, with every Chrome process gone.

    python3 -I tools/spark/jarvis/test_client_check.py

A fake Chrome (no browser needed) starts a helper in its own session that keeps writing into the profile, as Chrome's
crash handler does, then either exits before opening its debugging port (the failure path) or waits until the check is
stopped with SIGTERM (as `timeout` stops it).
"""
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = r'''#!/usr/bin/env python3
import os, subprocess, sys, time
profile = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--user-data-dir="))
helper = subprocess.Popen([sys.executable, "-c",
    "import os, sys, time\n"
    "open(sys.argv[2], 'w').write(str(os.getpid()))\n"
    "while True:\n"
    "    open(os.path.join(sys.argv[1], 'Crashpad.log'), 'a').write('x')\n"
    "    time.sleep(0.05)\n", profile, os.environ["FAKE_PID_FILE"]])
time.sleep(0.5)
if os.environ["FAKE_MODE"] == "exit":
    sys.exit(3)
time.sleep(60)
'''


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as f:  # a zombie is gone for this purpose
        return f.read().split(")")[-1].split()[0] != "Z"


class ProfileCleanup(unittest.TestCase):
    def run_check(self, mode):
        with tempfile.TemporaryDirectory() as root:
            fake = os.path.join(root, "chrome")
            with open(fake, "w") as f:
                f.write(FAKE)
            os.chmod(fake, 0o755)
            tmp = os.path.join(root, "tmp")
            os.mkdir(tmp)
            pid_file = os.path.join(root, "helper.pid")
            env = dict(os.environ, TMPDIR=tmp, FAKE_MODE=mode, FAKE_PID_FILE=pid_file)
            check = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "client_check.py"), "--base", "http://127.0.0.1:9",
                                      "--out", os.path.join(root, "out"), "--chrome", fake, "--steps", "chat"],
                                     env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(100):  # the helper is up
                if os.path.exists(pid_file) and open(pid_file).read():
                    break
                time.sleep(0.05)
            helper = int(open(pid_file).read())
            if mode == "term":
                time.sleep(1)
                self.assertIsNone(check.poll(), "the check should still be waiting for the debugging port")
                check.send_signal(signal.SIGTERM)
            rc = check.wait(60)
            self.assertNotEqual(rc, 0)
            self.assertEqual([n for n in os.listdir(tmp) if n.startswith("jarvis-client-")], [], "the profile is left behind")
            self.assertFalse(alive(helper), "a Chrome process outlived the check")

    def test_chrome_fails_before_its_port(self):
        self.run_check("exit")

    def test_stopped_by_sigterm(self):
        self.run_check("term")


if __name__ == "__main__":
    unittest.main()
