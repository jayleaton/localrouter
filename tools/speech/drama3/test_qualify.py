"""qualify.py's own git reads and its gpu_lock check say what is wrong (no GPU, no assets needed).

    python3 -I tools/speech/drama3/test_qualify.py

A qualify run as root on assets another user fetched (DGX-02, 2026-10-27) failed with "is not a git checkout": git
refuses a checkout another user owns. Its read-only git calls now name each checkout a safe.directory and take no
optional locks, and a refusal names the owner; a config without gpu_lock says what to set.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qualify  # noqa: E402


class GitEnv(unittest.TestCase):
    def test_each_path_is_a_safe_directory_after_the_callers_own(self):
        env = qualify.git_env("/a", "/b", base={"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.x", "GIT_CONFIG_VALUE_0": "1"})
        self.assertEqual(env["GIT_CONFIG_COUNT"], "3")
        self.assertEqual((env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_VALUE_0"]), ("core.x", "1"))
        self.assertEqual((env["GIT_CONFIG_KEY_1"], env["GIT_CONFIG_VALUE_1"]), ("safe.directory", "/a"))
        self.assertEqual((env["GIT_CONFIG_KEY_2"], env["GIT_CONFIG_VALUE_2"]), ("safe.directory", "/b"))
        self.assertEqual(env["GIT_OPTIONAL_LOCKS"], "0")


class GitHead(unittest.TestCase):
    def test_a_checkout_gives_its_head(self):
        with tempfile.TemporaryDirectory() as d:
            git = ["git", "-C", d, "-c", "user.name=t", "-c", "user.email=t@t"]
            subprocess.run(["git", "init", "-q", d], check=True)
            subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "x"], check=True)
            want = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            self.assertEqual(qualify.git_head(d), (want, None))

    def test_not_a_checkout_says_why(self):
        with tempfile.TemporaryDirectory() as d:
            head, why = qualify.git_head(d)
            self.assertIsNone(head)
            self.assertTrue(why)

    def test_another_owner_is_named(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(qualify.ownership_note(d))
            note = qualify.ownership_note(d, uid=os.getuid() + 1)
            self.assertIn(f"owned by uid {os.getuid()}", note)
            self.assertIn("dubious ownership", note)
            self.assertIsNone(qualify.ownership_note(os.path.join(d, "missing"), uid=os.getuid() + 1))


class GpuLock(unittest.TestCase):
    def test_missing_unset_and_present(self):
        self.assertIn("has no gpu_lock", qualify.gpu_lock_problem(""))
        with tempfile.TemporaryDirectory() as d:
            lock = os.path.join(d, "gpu.lock")
            self.assertIn("is not an existing file", qualify.gpu_lock_problem(lock))
            open(lock, "w").close()
            self.assertIsNone(qualify.gpu_lock_problem(lock))


if __name__ == "__main__":
    unittest.main()
