"""The macOS KnowLine server as the openai_server engine runs it: lifecycle, refusals, cancellation, failures, memory.

The front end (knowline_server.py, knowline_layout.py, tokenizer, template) is the pinned original repository's;
`KNOWLINE_FRONT_END` names it (default: its snapshot in the HF cache). Without the weights the server runs on a fake
backend (fixed label logprobs), so the process, protocol and refusal tests run in seconds; with `KNOWLINE_WEIGHTS` set
(under gpulock: it loads the model) the memory test also runs on the real model.

    ~/jarvis/venvs/decide-mlx/bin/python -I -m unittest discover -s tools/decide/knowline/macos -p 'test_*.py'
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory import footprint, peak  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REV = "779944d2cc8ffa19cef386aee83c83bcb967ade2"
FRONT_END = os.environ.get("KNOWLINE_FRONT_END") or os.path.expanduser(
    f"~/.cache/huggingface/hub/models--PelaAI--KnowLine-4B-Gen5.5/snapshots/{REV}")
WEIGHTS = os.environ.get("KNOWLINE_WEIGHTS")
HAVE_FRONT_END = all(os.path.exists(os.path.join(FRONT_END, f)) for f in ("knowline_server.py", "tokenizer.json"))

# A fake backend in the server process: label i scores -i (the first option wins), and a state containing "slow"
# holds the request for 30 s (a decision in flight, for the cancellation tests).
FAKE = """
import sys, time
sys.path.insert(0, {here!r})
import serve_mlx
class Fake:
    resident = 1234
    def __init__(self, weights, tok):
        self.tok = tok
    def warm(self, prefix, media):
        pass
    def score(self, text, media, ids):
        if media:
            raise ValueError("text only: pictures are not served on macOS")
        if "slow" in text:
            time.sleep(30)
        return [-float(i) for i in range(len(ids))], len(text) // 4
serve_mlx.MLX = Fake
sys.argv = ["serve_mlx.py"] + sys.argv[1:]
serve_mlx.main()
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, front_end=FRONT_END, fake=True, extra=()):
        self.port = free_port()
        args = [front_end, "--port", str(self.port), "--served-model-name", "knowline-test", *extra]
        cmd = [sys.executable, "-I", "-c", FAKE.format(here=HERE)] + args if fake else \
            [sys.executable, "-I", os.path.join(HERE, "serve_mlx.py")] + args
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def wait_healthy(self, timeout=300):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.p.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1) as r:
                    return r.status == 200
            except OSError:
                time.sleep(0.2)
        return False

    def post(self, body, timeout=60):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/systemone", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def stop(self):
        """SIGTERM, as the openai_server engine stops it; the seconds it took to exit."""
        t = time.monotonic()
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.p.stdout.close()
        self.p.stderr.close()
        return time.monotonic() - t


ROUTE = {"model": "knowline-test", "state": {"user_said": "draw a fox"},
         "questions": {"route": {"type": "choice", "instructions": "Which action?",
                                 "criteria": {"image": "make a picture", "video": "make a clip", "reply": None}},
                       "two": {"type": "noul"}, "urgency": {"type": "score", "criteria": ["low", "mid", "high"]}}}


@unittest.skipUnless(HAVE_FRONT_END, f"no front end at {FRONT_END} (fetch.sh, or set KNOWLINE_FRONT_END)")
class ProcessTest(unittest.TestCase):
    def test_lifecycle(self):
        s = Server()
        try:
            self.assertTrue(s.wait_healthy(60), s.p.stderr.read() if s.p.poll() is not None else "no health")
            code, r = s.post(ROUTE)
            self.assertEqual(200, code, r)
            a = r["answers"]
            self.assertEqual("image", a["route"]["choice"])
            self.assertAlmostEqual(1.0, sum(a["route"]["probabilities"].values()), places=9)
            self.assertTrue(0 <= a["two"]["noul"] <= 1)
            self.assertEqual(["0", "1", "2"], list(a["urgency"]["probabilities"]))
            self.assertEqual(3, r["usage"]["output_tokens"])
        finally:
            self.assertLess(s.stop(), 2)
        self.assertEqual(0, s.p.returncode)

    def test_refusals_leave_it_serving(self):
        s = Server()
        try:
            self.assertTrue(s.wait_healthy(60))
            pic = {"state": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,x"}},
                                                          {"type": "text", "text": "what is this"}]}],
                   "questions": {"q": {"type": "noul"}}}
            code, r = s.post(pic)
            self.assertEqual(422, code)
            self.assertIn("text only", r["error"])
            code, r = s.post({"state": "s", "questions": {"q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}}})
            self.assertEqual(422, code)
            self.assertIn("2..255", r["error"])
            code, r = s.post({"state": "s", "questions": {f"q{i}": {"type": "noul"} for i in range(65)}})
            self.assertEqual(422, code)
            code, r = s.post({"state": "s", "questions": {"q": {"type": "box"}}})
            self.assertEqual(422, code)
            code, _ = s.post(ROUTE)  # still serving after every refusal
            self.assertEqual(200, code)
        finally:
            s.stop()

    def test_cancelled_request_then_the_next(self):
        """A client that gives up (the daemon cancelling the job) does not wedge the server; SIGTERM mid-decision
        still exits at once."""
        s = Server()
        try:
            self.assertTrue(s.wait_healthy(60))
            body = dict(ROUTE, state="slow", questions={"q": {"type": "noul"}})
            with self.assertRaises(OSError):
                s.post(body, timeout=0.5)
            code, _ = s.post(ROUTE, timeout=60)  # a ThreadingHTTPServer: answered beside the abandoned one
            self.assertEqual(200, code)
            stuck = Server()
            try:
                self.assertTrue(stuck.wait_healthy(60))
                import threading
                threading.Thread(target=lambda: _ignore(stuck.post, body, 60), daemon=True).start()
                time.sleep(0.5)
            finally:
                self.assertLess(stuck.stop(), 2)  # SIGTERM with a decision in flight
            self.assertEqual(0, stuck.p.returncode)
        finally:
            s.stop()


def _ignore(f, *a):
    try:
        f(*a)
    except OSError:
        pass


@unittest.skipUnless(HAVE_FRONT_END, f"no front end at {FRONT_END}")
class FailureTest(unittest.TestCase):
    def copy_front_end(self, d):
        for f in os.listdir(FRONT_END):
            if not f.endswith(".safetensors") and os.path.isfile(os.path.join(FRONT_END, f)):
                shutil.copy(os.path.join(FRONT_END, f), d)

    def test_changed_front_end_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self.copy_front_end(d)
            with open(os.path.join(d, "knowline_server.py"), "a") as f:
                f.write("\n# changed\n")
            s = Server(front_end=d)
            self.assertFalse(s.wait_healthy(30))
            self.assertNotEqual(0, s.p.returncode)
            self.assertIn("not the reviewed file", s.p.stderr.read())
            s.stop()

    def test_missing_front_end_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            s = Server(front_end=d)
            self.assertFalse(s.wait_healthy(30))
            self.assertIn("missing", s.p.stderr.read())
            s.stop()

    def test_wrong_checkpoint_fails_the_load(self):
        """The real loader: a checkpoint of another family exits non-zero before serving (a failed load)."""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump({"model_type": "llama"}, f)
            s = Server(fake=False, extra=("--weights", d))
            self.assertFalse(s.wait_healthy(60))
            self.assertNotEqual(0, s.p.returncode)
            self.assertIn("not KnowLine's qwen3_5", s.p.stderr.read())
            s.stop()


def largest_read_twice():
    """The longest prompt a request can make: a state just inside read-twice's 3,000-token cut, written twice (a longer
    state is read once, so shorter), as a /v1/systemone body, with its prompt's token count."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, FRONT_END)
    from transformers import AutoTokenizer
    import knowline_layout as kl
    tok = AutoTokenizer.from_pretrained(FRONT_END)
    lines = [f"[{i // 60:02d}:{i % 60:02d}] sensor {i}: temperature nominal, door closed" for i in range(400)]
    q = {"alarm": {"type": "noul", "instructions": "Did any sensor raise an alarm in this log?"}}
    lo, hi = 1, len(lines)
    while lo < hi:  # the most lines whose request is still read twice
        mid = (lo + hi + 1) // 2
        r = kl.render_request({"state": "\n".join(lines[:mid]), "questions": q}, tok, kl.Layout(read_twice=True))
        lo, hi = (mid, hi) if r["twice"] else (lo, mid - 1)
    body = {"state": "\n".join(lines[:lo]), "questions": q}
    r = kl.render_request(body, tok, kl.Layout(read_twice=True))
    return body, len(tok.encode(r["prompts"]["alarm"][0][0], add_special_tokens=False))


@unittest.skipUnless(WEIGHTS and HAVE_FRONT_END, "KNOWLINE_WEIGHTS unset (a real-model run, under gpulock)")
class MemoryTest(unittest.TestCase):
    def test_footprint_within_booking_and_released_between_requests(self):
        booking = int(os.environ.get("KNOWLINE_BOOKING", str(11 << 30)))
        body, tokens = largest_read_twice()
        s = Server(fake=False, extra=("--weights", WEIGHTS))
        try:
            self.assertTrue(s.wait_healthy(300))
            idle = footprint(s.p.pid)
            after = []
            for _ in range(3):
                code, r = s.post(body, timeout=120)
                self.assertEqual(200, code, r)
                self.assertEqual(tokens, r["usage"]["input_tokens"])
                after.append(footprint(s.p.pid))
            top = peak(s.p.pid)
            print(f"\nidle {idle / 2**30:.2f} GiB, peak {top / 2**30:.2f} GiB on a {tokens}-token prompt, after each "
                  f"request {[round(x / 2**30, 2) for x in after]} GiB", file=sys.stderr)
            self.assertLess(top, booking)
            self.assertLess(after[-1] - after[0], 256 << 20)  # nothing kept growing between requests
        finally:
            s.stop()


if __name__ == "__main__":
    unittest.main()
