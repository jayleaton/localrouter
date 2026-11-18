"""fetch.sh against a stand-in curl and a one-file pins.json: a file whose digest fails never reaches WEIGHTS.
    python3 -I tools/video/wan/test_fetch.py      (bash; no network, no GPU)
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CURL = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do case $1 in -o) out=$2; shift;; *) url=$1;; esac; shift; done
cp "$FAKE_HUB/${url##*/}" "$out"
"""


class FetchTest(unittest.TestCase):
    def setUp(self):
        scratch = os.path.join(REPO, ".zig-cache")  # not the system temp dir: it may be mounted noexec, hiding the stand-in curl
        os.makedirs(scratch, exist_ok=True)
        self.root = tempfile.mkdtemp(prefix="test-fetch-", dir=scratch)
        self.addCleanup(shutil.rmtree, self.root)
        wan, package = os.path.join(self.root, "tools/video/wan"), os.path.join(self.root, "tools/package")
        os.makedirs(wan)
        os.makedirs(package)
        shutil.copy(os.path.join(HERE, "fetch.sh"), wan)
        shutil.copy(os.path.join(REPO, "tools/package/verify_assets.py"), package)
        self.fetch, self.pins = os.path.join(wan, "fetch.sh"), os.path.join(wan, "pins.json")
        self.hub, bin_ = os.path.join(self.root, "hub"), os.path.join(self.root, "bin")
        os.makedirs(self.hub)
        os.makedirs(bin_)
        with open(os.path.join(bin_, "curl"), "w") as f:
            f.write(CURL)
        os.chmod(os.path.join(bin_, "curl"), 0o755)
        self.env = dict(os.environ, PATH=bin_ + os.pathsep + os.environ["PATH"], FAKE_HUB=self.hub)
        self.weights = os.path.join(self.root, "models", "wan")

    def run_fetch(self, pinned, served):
        """pins.json pins model_index.json to `pinned`; the hub serves `served`."""
        pin = {"path": "model_index.json", "source": "wan", "src": "model_index.json", "size_bytes": len(pinned),
               "sha256": hashlib.sha256(pinned).hexdigest()}
        with open(self.pins, "w") as f:
            json.dump({"sources": {"wan": {"repo": "Wan-AI/x", "revision": "0" * 40}}, "selected_files": [pin]}, f)
        with open(os.path.join(self.hub, "model_index.json"), "wb") as f:
            f.write(served)
        return subprocess.run(["bash", self.fetch, self.weights], env=self.env, capture_output=True).returncode

    def published(self):
        with open(os.path.join(self.weights, "model_index.json"), "rb") as f:
            return f.read()

    def test_a_failed_digest_never_reaches_weights(self):
        self.assertNotEqual(self.run_fetch(b'{"v":1}', b'{"v":X}'), 0)  # the pinned size, the wrong bytes
        self.assertFalse(os.path.exists(os.path.join(self.weights, "model_index.json")))
        self.assertEqual(self.run_fetch(b'{"v":1}', b'{"v":1}'), 1)  # the bad file stays to inspect, not refetched
        os.remove(os.path.join(self.weights + ".staging", "model_index.json"))
        self.assertEqual(self.run_fetch(b'{"v":1}', b'{"v":1}'), 0)
        self.assertEqual(self.published(), b'{"v":1}')
        self.assertFalse(os.path.exists(self.weights + ".staging"))
        self.assertNotEqual(self.run_fetch(b'{"v":22}', b'{"v":XX}'), 0)  # a new pin, served corrupt
        self.assertEqual(self.published(), b'{"v":1}')  # the previous snapshot is untouched
        os.remove(os.path.join(self.weights + ".staging", "model_index.json"))
        self.assertEqual(self.run_fetch(b'{"v":22}', b'{"v":22}'), 0)
        self.assertEqual(self.published(), b'{"v":22}')


if __name__ == "__main__":
    unittest.main(verbosity=1)
