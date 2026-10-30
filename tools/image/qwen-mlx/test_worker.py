"""The macOS Qwen-Image worker against a fake MLX-Serve: lifecycle, memory, refusals, cancellation, failures, outputs.

    ROOT/venv/bin/python -I tools/image/qwen-mlx/test_worker.py      (fetch.sh's venv: Pillow; no model, no GPU)

The fake (this file run as `fake`) answers MLX-Serve's routes the way v26.10.3 does: /v1/models, /props, and
/v1/images/generations streaming progress events then the PNG; it logs each request's fields, without the images'
bytes, to FAKE_LOG. A prompt picks a failure: "slow" (many steps, for a cancel), "fail" (a 500), "refuse" (a 400),
"oom" (MLX-Serve's working-set 400), "die" (the server exits).
"""
import base64
import io
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from PIL import Image  # noqa: E402

import qwen_mlx_worker as qw  # noqa: E402

SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
ACTIVE = 17 << 30  # the fake's MLX active bytes


# -- the fake server -------------------------------------------------------------------------------------------------

def fake_main(argv):
    if "--version" in argv:
        print(f"mlx-serve {os.environ.get('FAKE_VERSION', '26.10.3')}\nmlx 0.32.3")
        return
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    port = int(argv[argv.index("--port") + 1])
    flog = os.environ["FAKE_LOG"]

    def record(obj):
        with open(flog, "a") as f:
            f.write(json.dumps(obj) + "\n")

    record({"started": os.getpid(), "argv": argv})

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply(200, {"data": [{"id": "x", "loaded": True, "state": "ready"}]})
            elif self.path == "/props":
                self.reply(200, {"memory": {"active_bytes": ACTIVE, "peak_bytes": ACTIVE}})
            else:
                self.reply(404, {"error": {"message": "no"}})

        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            keys = list(b)
            refs = [b["image"]] + b.get("ref_images", []) if "image" in b else []
            formats = [Image.open(io.BytesIO(base64.b64decode(r))).format for r in refs]
            sizes = [Image.open(io.BytesIO(base64.b64decode(r))).size for r in refs]
            record({k: v for k, v in b.items() if k not in ("image", "ref_images")} | {"keys": keys, "ref_formats": formats, "ref_sizes": sizes})
            p = b["prompt"]
            if p == "die":
                os._exit(3)
            if p == "fail":
                return self.reply(500, {"error": {"message": "generation failed"}})
            if p == "refuse":
                return self.reply(400, {"error": {"message": "'mode' must be \"edit\" or \"variation\""}})
            if p == "oom":
                return self.reply(400, {"error": {"message": "this edit's working set (references + target) needs more GPU memory than is free"}})
            if b.get("stream") is not True:  # MLX-Serve answers one JSON body: no progress, no prompt cancel
                record({"unstreamed": p})
                return self.reply(500, {"error": {"message": "the fake streams only"}})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            total = 400 if p == "slow" else 8
            try:
                for i in range(1, total + 1):
                    self.wfile.write(f'data: {{"type":"progress","stage":"denoise","step":{i},"total":{total}}}\n\n'.encode())
                    self.wfile.flush()
                    if p == "slow":
                        time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                record({"cancelled": p})
                return
            w, h = (int(x) for x in b["size"].split("x"))
            mode = "RGBA" if b.get("transparent") else "RGB"
            colour = (b["seed"] % 256, 100, 200, 255)[:len(mode)]
            buf = io.BytesIO()
            Image.new(mode, (w, h), colour).save(buf, "PNG")
            ev = {"type": "complete", "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())

    class Server(ThreadingHTTPServer):
        def server_bind(self):  # HTTPServer's own looks the host's name up (getfqdn), which can hang without DNS
            import socketserver
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "127.0.0.1", port

    Server(("127.0.0.1", port), H).serve_forever()


# -- helpers ---------------------------------------------------------------------------------------------------------

def png(path, size, colour=(10, 20, 30, 255), fmt="PNG", mode="RGBA"):
    Image.new(mode, size, colour[:len(mode)]).save(path, fmt)


class Env:
    """A temp dir with fake weights, a server wrapper, a job dir and the fake's log."""

    def __init__(self, options=None, version="26.10.3", sigmas=SIGMAS):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.weights = os.path.join(t, "weights")
        os.makedirs(self.weights)
        with open(os.path.join(self.weights, "model_index.json"), "w") as f:
            json.dump({"sample_sigmas": sigmas}, f)
        self.server = os.path.join(t, "mlx-serve")
        with open(self.server, "w") as f:
            f.write(f"#!/bin/sh\nexec '{sys.executable}' -I '{os.path.abspath(__file__)}' fake \"$@\"\n")
        os.chmod(self.server, 0o755)
        self.job = os.path.join(t, "job")
        os.makedirs(self.job)
        self.log = os.path.join(t, "fake.log")
        open(self.log, "w").close()  # noqa: SIM115
        self.env = dict(os.environ, FAKE_LOG=self.log, FAKE_VERSION=version)
        self.tool = {"id": "qwen-mac", "kind": "image", "weights": self.weights,
                     "options": {"server": self.server, "verify": "none", "warmup": False, **(options or {})}}

    def records(self):
        with open(self.log) as f:
            return [json.loads(x) for x in f]

    def requests(self):
        return [r for r in self.records() if "prompt" in r]

    def close(self):
        self.tmp.cleanup()


class Proc:
    """The worker as the daemon runs it: LOCALROUTER_TOOL, one JSON line a message each way."""

    def __init__(self, env):
        self.p = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "qwen_mlx_worker.py")],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                  env=dict(env.env, LOCALROUTER_TOOL=json.dumps(env.tool)), start_new_session=True)

    def send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def line(self, timeout=30):
        out = []
        t = threading.Thread(target=lambda: out.append(self.p.stdout.readline()))
        t.start()
        t.join(timeout)
        if not out:
            raise TimeoutError("no line from the worker")
        return json.loads(out[0]) if out[0] else None

    def reply(self, timeout=30):
        """The next line that is not progress, and the progress seen before it."""
        progress = []
        while True:
            m = self.line(timeout)
            if m is None or "progress" not in m:
                return m, progress
            progress.append(m["progress"])

    def gen(self, env, image, id="img_1"):
        self.send({"op": "generate", "id": id, "dir": env.job, "request": {"image": image}})
        return self.reply()

    def close(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGKILL)
        self.p.wait()
        self.p.stdin.close()
        self.p.stdout.close()


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def text_chunks(data):
    out, i = {}, 8
    while i < len(data):
        n = struct.unpack(">I", data[i:i + 4])[0]
        kind = data[i + 4:i + 8]
        if kind == b"tEXt":
            k, v = data[i + 8:i + 8 + n].split(b"\0", 1)
            out[k.decode()] = v.decode()
        i += 12 + n
    return out


# -- tests -----------------------------------------------------------------------------------------------------------

class Sizes(unittest.TestCase):
    def test_an_edit_without_a_size_keeps_the_last_images_shape_within_the_limits(self):
        # the Spark tool's own cases (qwen_image_tool.zig "an edit without a size keeps the image's shape")
        self.assertEqual(qw.edit_size(512, 512, 1360, 1048576), (1024, 1024))
        self.assertEqual(qw.edit_size(1920, 1080, 1360, 1048576), (1376 - 32, 768))
        for wh in [(928, 1184), (3000, 1000), (100, 4000), (1, 1)]:
            w, h = qw.edit_size(*wh, 1360, 1048576)
            self.assertTrue(max(w, h) <= 1360 and w * h <= 1048576 and w % 32 == 0 and h % 32 == 0, (wh, w, h))

    def test_dimensions_round_half_to_even_as_the_engine(self):
        self.assertEqual(qw.dimensions(1024, 768, 1024 * 1024), (1184, 896))
        self.assertRaises(ValueError, qw.dimensions, 1, 5000, 1024 * 1024)


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.env = Env({"warmup": True})
        self.w = Proc(self.env)

    def tearDown(self):
        self.w.close()
        self.env.close()

    def test_load_generate_exit(self):
        self.w.send({"op": "load"})
        r, _ = self.w.reply()
        self.assertEqual((r["ok"], r["cancel"]), (True, "signal"))
        self.assertGreaterEqual(r["resident"], ACTIVE)  # memory: MLX's active bytes or the server's RSS, the larger
        started = [x for x in self.env.records() if "started" in x][0]
        argv = started["argv"]
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertIn("--skip-mem-preflight", argv)
        self.assertIn("--no-w8a8", argv)

        warm = self.env.requests()
        self.assertEqual([(q["prompt"], q["size"]) for q in warm], [("warm up", "256x256")])
        r, progress = self.w.gen(self.env, {"prompt": "a fox", "width": 1024, "height": 768, "n": 2, "seed": 7})
        self.assertEqual((r["ok"], r["files"], r["seed"]), (True, ["0.png", "1.png"], 7))
        self.assertEqual(progress[:2], [{"phase": "denoise", "step": 1, "of": 8}, {"phase": "denoise", "step": 2, "of": 8}])
        reqs = self.env.requests()[1:]
        self.assertEqual([q["seed"] for q in reqs], [7, 8])
        self.assertEqual({(q["size"], q["steps"], q["transparent"], q["stream"]) for q in reqs}, {("1024x768", 8, True, True)})
        self.assertEqual(reqs[0]["keys"][-1], "prompt")  # MLX-Serve takes each field's first match: the prompt goes last
        self.assertNotIn("mode", reqs[0])
        for i, name in enumerate(r["files"]):
            with open(os.path.join(self.env.job, name), "rb") as f:
                data = f.read()
            self.assertEqual(text_chunks(data), {"prompt": "a fox", "seed": str(7 + i)})
            im = Image.open(io.BytesIO(data))
            self.assertEqual((im.size, im.mode, im.getpixel((0, 0))[0]), ((1024, 768), "RGBA", 7 + i))

        self.w.send({"op": "exit"})
        self.assertEqual(self.w.p.wait(timeout=30), 0)
        self.assertFalse(alive(started["started"]), "mlx-serve outlived the worker")

    def test_steps_and_seed_edges(self):
        self.w.send({"op": "load"})
        self.w.reply()
        r, _ = self.w.gen(self.env, {"prompt": "x", "width": 512, "height": 512, "steps": 8, "seed": (1 << 64) - 1, "n": 2})
        self.assertTrue(r["ok"], r)
        self.assertEqual([q["seed"] for q in self.env.requests()[1:]], [(1 << 64) - 1, 0])  # wraps as the Spark engine's u64


class Refusals(unittest.TestCase):
    """Refused before anything reaches the server (invalid_request_error), as the Spark tool's check."""

    def setUp(self):
        self.env = Env({"max_side": 1360, "max_pixels": 1048576})
        self.w = qw.Worker(self.env.tool)
        self.w.port = 1  # nothing listens: a request that got through would fail differently
        png(os.path.join(self.env.job, "a.png"), (512, 512))
        png(os.path.join(self.env.job, "m.png"), (256, 256))
        png(os.path.join(self.env.job, "thin.png"), (2500, 500))
        png(os.path.join(self.env.job, "tiny.png"), (8, 8))
        with open(os.path.join(self.env.job, "bad.png"), "wb") as f:
            f.write(b"not an image")

    def tearDown(self):
        self.env.close()

    def refused(self, image, why):
        with self.assertRaises(qw.Refused) as c:
            self.w.generate({"dir": self.env.job, "request": {"image": {"prompt": "p", "width": 1024, "height": 1024, **image}}}, None)
        self.assertIn(why, str(c.exception))

    def test_requests_the_model_cannot_serve(self):
        self.refused({"references": ["a.png"] * 6}, "up to 5 input images")
        self.refused({"steps": 25}, "8-step schedule: omit steps or send 8")
        self.refused({"width": 1376, "height": 768}, "up to 1360 pixels a side")
        self.refused({"width": 1360, "height": 1024}, "at most 1048576 pixels")
        self.refused({"mask": "m.png"}, "a mask needs an input image")
        self.refused({"references": ["a.png"], "width": 1008, "height": 1008}, "multiples of 32")
        self.refused({"references": ["a.png"], "mask": "m.png"}, "the mask is 256x256, the first image 512x512")
        self.refused({"references": ["thin.png"]}, "too long and thin")
        self.refused({"references": ["tiny.png"]}, "sides of at least 32 pixels")
        self.refused({"references": ["bad.png"]}, "could not be read")
        self.refused({"references": ["../a.png"]}, "files in the job directory")
        self.refused({"width": 0, "height": 0}, "needs a size")
        self.refused({"prompt": ""}, "prompt is required")
        self.w.opts.edit_refs = 2
        self.refused({"references": ["a.png"] * 3}, "up to 2 input images")

    def test_not_an_image_request(self):
        with self.assertRaises(qw.Refused):
            self.w.generate({"dir": self.env.job, "request": {"speech": {"input": "hi"}}}, None)


class Edits(unittest.TestCase):
    def setUp(self):
        self.env = Env({"max_side": 1360, "max_pixels": 1048576})
        self.w = Proc(self.env)
        self.w.send({"op": "load"})
        self.w.reply()

    def tearDown(self):
        self.w.close()
        self.env.close()

    def test_inputs_in_order_and_the_size_from_the_last(self):
        png(os.path.join(self.env.job, "ref_0.png"), (512, 512))
        png(os.path.join(self.env.job, "ref_1.jpg"), (640, 480), fmt="JPEG", mode="RGB")
        png(os.path.join(self.env.job, "ref_2.webp"), (1920, 1080), fmt="WEBP")
        r, _ = self.w.gen(self.env, {"prompt": "put them together", "width": 0, "height": 0, "seed": 3,
                                     "references": ["ref_0.png", "ref_1.jpg", "ref_2.webp"]})
        self.assertTrue(r["ok"], r)
        q = self.env.requests()[-1]
        self.assertEqual(q["mode"], "edit")
        self.assertEqual(q["ref_sizes"], [[512, 512], [640, 480], [1920, 1080]])  # first is `image`, the rest `ref_images`
        self.assertEqual(q["ref_formats"], ["PNG", "JPEG", "PNG"])  # WebP sent as PNG, the others as they came
        self.assertEqual(q["size"], "1344x768")  # the last image's shape within max_side 1360 / max_pixels 1 MP

    def test_a_mask_keeps_the_first_image_where_it_is_opaque(self):
        png(os.path.join(self.env.job, "ref_0.png"), (512, 512), colour=(10, 20, 30, 255))
        m = Image.new("RGBA", (512, 512), (0, 0, 0, 255))
        m.paste((0, 0, 0, 0), (0, 0, 256, 512))  # the left half clear: the edit goes there
        m.save(os.path.join(self.env.job, "mask_0.png"))
        r, _ = self.w.gen(self.env, {"prompt": "make it night", "width": 512, "height": 512, "seed": 200,
                                     "references": ["ref_0.png"], "mask": "mask_0.png"})
        self.assertTrue(r["ok"], r)
        with open(os.path.join(self.env.job, "0.png"), "rb") as f:
            data = f.read()
        im = Image.open(io.BytesIO(data))
        self.assertEqual(im.getpixel((10, 10)), (200, 100, 200, 255))  # the edit (the fake's colour from the seed)
        self.assertEqual(im.getpixel((500, 10)), (10, 20, 30, 255))  # the original, exactly
        self.assertEqual(text_chunks(data), {"prompt": "make it night", "seed": "200"})


class Cancel(unittest.TestCase):
    def test_sigusr1_ends_a_generate_and_the_model_stays_loaded(self):
        env = Env()
        w = Proc(env)
        try:
            w.send({"op": "load"})
            w.reply()
            pid = [x for x in env.records() if "started" in x][0]["started"]
            w.send({"op": "generate", "id": "a", "dir": env.job, "request": {"image": {"prompt": "slow", "width": 512, "height": 512}}})
            self.assertIn("progress", w.line())
            w.p.send_signal(signal.SIGUSR1)
            r, _ = w.reply()
            self.assertEqual((r["ok"], r["type"]), (False, "cancelled"))
            r, _ = w.gen(env, {"prompt": "after", "width": 512, "height": 512})
            self.assertTrue(r["ok"], r)
            self.assertTrue(alive(pid))
            self.assertEqual(len([x for x in env.records() if "started" in x]), 1)  # no reload
            deadline = time.time() + 10
            while not any(x.get("cancelled") == "slow" for x in env.records()) and time.time() < deadline:
                time.sleep(0.1)
            self.assertTrue(any(x.get("cancelled") == "slow" for x in env.records()), "the server never saw the cancel")
        finally:
            w.close()
            env.close()


class Failures(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.w = Proc(self.env)
        self.w.send({"op": "load"})
        self.w.reply()

    def tearDown(self):
        self.w.close()
        self.env.close()

    def test_server_answers_map_to_the_daemons_types(self):
        for prompt, kind in [("fail", "server_error"), ("refuse", "invalid_request_error"), ("oom", "service_unavailable")]:
            r, _ = self.w.gen(self.env, {"prompt": prompt, "width": 512, "height": 512})
            self.assertEqual((r["ok"], r["type"]), (False, kind), prompt)
        r, _ = self.w.gen(self.env, {"prompt": "still here", "width": 512, "height": 512})
        self.assertTrue(r["ok"])

    def test_a_server_that_dies_takes_the_worker_with_it(self):
        self.w.send({"op": "generate", "id": "x", "dir": self.env.job, "request": {"image": {"prompt": "die", "width": 512, "height": 512}}})
        self.assertIsNone(self.w.line())  # no reply: the daemon fails the job with a 503 and starts the model again
        self.assertEqual(self.w.p.wait(timeout=30), 1)


class LoadFailures(unittest.TestCase):
    def load(self, env):
        w = Proc(env)
        try:
            w.send({"op": "load"})
            r, _ = w.reply()
            return r
        finally:
            w.close()
            env.close()

    def test_an_old_server_runs_the_base_schedule_on_turbo(self):
        r = self.load(Env(version="26.10.1"))
        self.assertFalse(r["ok"])
        self.assertIn("26.10.2 or newer", r["error"])

    def test_the_schedule_must_have_the_tools_steps(self):
        r = self.load(Env({"steps": 6}))
        self.assertIn("sample_sigmas has 8 steps, the tool's steps option 6", r["error"])

    def test_pinned_files_must_be_there(self):
        r = self.load(Env({"verify": "size"}))
        self.assertIn("missing or not the pinned size", r["error"])

    def test_each_pack_is_checked_against_its_own_pins(self):
        env = Env({"verify": "size", "pack": "q4"})
        os.makedirs(os.path.join(env.weights, "text_encoder"))
        with open(os.path.join(env.weights, "text_encoder", "model.safetensors"), "wb") as f:
            f.truncate(17)  # not the 4-bit text encoder's pinned size
        r = self.load(env)
        self.assertFalse(r["ok"])
        self.assertIn("not the pinned size", r["error"])
        self.assertEqual(qw.PACKS["q4"], os.path.join(HERE, "pins-macos-q4.json"))

    def test_the_daemons_own_options_are_not_reported_as_unused(self):
        import contextlib
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            qw.Options({"weights": "/w", "options": {"reference_working_bytes": 2147483648}})
            self.assertEqual(err.getvalue(), "")
            qw.Options({"weights": "/w", "options": {"typo_bytes": 1}})
        self.assertIn("options not used by the worker: ['typo_bytes']", err.getvalue())

    def test_an_unknown_pack_is_a_bad_config(self):
        r = self.load(Env({"pack": "q2"}))
        self.assertIn("bad tool config: pack must be one of ['q4', 'q8']", r["error"])

    def test_a_bad_config_answers_every_load(self):
        env = Env()
        env.tool["weights"] = ""
        r = self.load(env)
        self.assertIn("bad tool config", r["error"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fake":
        fake_main(sys.argv[2:])
    else:
        unittest.main()
