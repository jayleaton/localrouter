"""The Idol worker against a fake engine: lifecycle, refusals, cancellation, idle SIGUSR1, outputs and the manifest.

    VENV/bin/python -I tools/asset3d/idol/test_worker.py      (the worker's venv: numpy and Pillow; no weights, no GPU)
"""
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import worker as w  # noqa: E402

NAMES = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


class FakeEngine:
    """The IdolEngine interface; records what it was asked. `fail` picks a failure: empty (the SDK's empty-alpha
    ValueError), crash (a RuntimeError), oom."""

    def __init__(self):
        self.p, self.calls, self.mem_peak, self.fail, self.on_step_hook = None, [], 0, None, None
        self.device, self.optimize = "mps", False

    def load(self):
        self.p = object()

    def resident(self):
        return 5 << 30

    def release(self):
        pass

    def torch_version(self):
        return "2.13.0"

    def allocator_bytes(self):
        return 0

    def start_sampling(self):
        pass

    def stop_sampling(self):
        pass

    def prepare(self, image, mode):
        if self.fail == "empty":
            raise ValueError("no object found in the image (the alpha channel is empty)")
        self.calls.append(("prepare", mode))
        return Image.new("RGB", (64, 64), (10, 20, 30))

    def encode(self, prepared, seed):
        self.calls.append(("encode", seed))
        return "state"

    def denoise(self, state, steps, guidance, on_step):
        self.calls.append(("denoise", steps, guidance))
        if self.fail == "crash":
            raise RuntimeError("boom")
        if self.fail == "oom":
            raise RuntimeError("MPS backend out of memory")
        for k in range(1, steps + 1):
            on_step(k)
            if self.on_step_hook:
                self.on_step_hook(k)
            w.check_cancel()
        return "sample"

    def decode(self, sample, count):
        self.calls.append(("decode", count))
        return count

    def camera(self, sample):
        return {"yaw": 0.0, "pitch": 0.0, "radius": 2.0, "fov": 40.0, "inverse_distance": 0.5}

    def save_ply(self, count, path):
        data = np.zeros(count, dtype=[(n, "<f4") for n in NAMES])
        data["x"] = np.linspace(-0.4, 0.3, count)
        data["y"] = -0.2
        data["z"] = np.linspace(0.1, 0.2, count)
        head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % count
        head += "".join(f"property float {n}\n" for n in NAMES) + "end_header\n"
        with open(path, "wb") as f:
            f.write(head.encode("ascii"))
            data.tofile(f)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def tool(weights="/nonexistent", **options):
    return {"id": "idol", "weights": weights, "options": options}


class Base(unittest.TestCase):
    def setUp(self):
        self.job = tempfile.mkdtemp(prefix="idol-test-")
        self.addCleanup(shutil.rmtree, self.job, True)
        Image.new("RGB", (80, 60), (200, 100, 50)).save(os.path.join(self.job, "image_0.png"))
        self.engine = FakeEngine()
        self.worker = w.Worker(tool(), engine=self.engine)
        self.worker.check_weights = lambda: None
        self.assertTrue(self.worker.load()["ok"])
        self.progress = []
        signal.signal(signal.SIGUSR1, w.on_cancel)

    def gen(self, **fields):
        req = {"image": "image_0.png", "input_mode": "synthetic", "format": "ply", "seed": 42}
        req.update(fields)
        req = {k: v for k, v in req.items() if v is not None}
        cmd = {"op": "generate", "id": "a3d_test", "dir": self.job, "request": {"asset3d": req}}
        return w.run_generate(self.worker, cmd, out=self.progress.append)


class Lifecycle(Base):
    def test_load_reply(self):
        r = w.Worker(tool(), engine=FakeEngine())
        r.check_weights = lambda: None
        self.assertEqual(r.load(), {"ok": True, "resident": 5 << 30, "cancel": "signal"})

    def test_missing_weights_refuse_load(self):
        r = w.Worker(tool(weights=self.job), engine=FakeEngine()).load()
        self.assertFalse(r["ok"])
        self.assertEqual(r["type"], "server_error")
        self.assertIn("idol/denoiser.safetensors (missing)", r["error"])
        self.assertIn("fetch.sh", r["error"])

    def test_generate_and_defaults(self):
        r = self.gen()
        self.assertEqual(r["files"], ["asset.ply", "asset.json", "input.png"])
        self.assertEqual(r["seed"], 42)
        for f in r["files"]:
            self.assertTrue(os.path.isfile(os.path.join(self.job, f)))
        self.assertIn(("denoise", 50, 3.0), self.engine.calls)
        self.assertIn(("decode", 262144), self.engine.calls)
        phases = [p["progress"]["phase"] for p in self.progress]
        self.assertEqual(sorted(set(phases), key=phases.index), ["preprocess", "encode", "denoise", "decode", "write"])
        steps = [p["progress"]["step"] for p in self.progress if p["progress"]["phase"] == "denoise"]
        self.assertEqual(steps, list(range(0, 51)))

    def test_guidance_zero_and_real_mode(self):
        r = self.gen(guidance=0, steps=4, num_gaussians=32768, input_mode="real")
        self.assertTrue(r["ok"], r)
        self.assertIn(("denoise", 4, 0.0), self.engine.calls)
        self.assertIn(("prepare", "real"), self.engine.calls)
        m = read_json(os.path.join(self.job, "asset.json"))
        self.assertEqual(m["settings"]["crop_margin"], 1.7)

    def test_server_errors_keep_worker(self):
        self.engine.fail = "crash"
        self.assertEqual(self.gen(steps=2)["type"], "server_error")
        self.engine.fail = "oom"
        self.assertEqual(self.gen(steps=2)["type"], "service_unavailable")
        self.engine.fail = None
        self.assertTrue(self.gen(steps=2, num_gaussians=32768)["ok"])


class Refusals(Base):
    def refused(self, **fields):
        r = self.gen(**fields)
        self.assertEqual(r.get("type"), "invalid_request_error", r)
        return r["error"]

    def test_counts_and_settings(self):
        self.refused(steps=0)
        self.refused(steps=101)
        self.refused(steps=True)
        self.refused(steps=2.5)
        self.refused(num_gaussians=32736)
        self.refused(num_gaussians=1048608)
        self.refused(num_gaussians=32768 + 16)
        self.refused(guidance=-1)
        self.refused(guidance=11)
        self.refused(seed=-1)
        self.refused(input_mode="photo")
        self.refused(format="glb")

    def test_images(self):
        self.refused(image="../image_0.png")
        self.refused(image="missing.png")
        with open(os.path.join(self.job, "junk.png"), "wb") as f:
            f.write(b"not an image")
        self.assertIn("could not be read", self.refused(image="junk.png"))
        Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(os.path.join(self.job, "clear.png"))
        self.assertIn("transparent", self.refused(image="clear.png"))
        self.engine.fail = "empty"
        self.assertIn("no object", self.refused())

    def test_wrong_kind(self):
        r = w.run_generate(self.worker, {"op": "generate", "dir": self.job, "request": {"image": {"prompt": "x"}}},
                           out=self.progress.append)
        self.assertEqual(r["type"], "invalid_request_error")


class Cancel(Base):
    def test_cancel_mid_denoise_then_next_works(self):
        self.engine.on_step_hook = lambda k: k == 3 and os.kill(os.getpid(), signal.SIGUSR1)
        self.assertEqual(self.gen(steps=10), {"ok": False, "type": "cancelled", "error": "cancelled"})
        self.assertFalse(os.path.exists(os.path.join(self.job, "asset.json")))
        self.engine.on_step_hook = None
        self.assertTrue(self.gen(steps=3, num_gaussians=32768)["ok"])

    def test_idle_signal_ignored(self):
        os.kill(os.getpid(), signal.SIGUSR1)
        self.assertFalse(w.CANCEL.is_set())
        self.assertTrue(self.gen(steps=3, num_gaussians=32768)["ok"])


class Manifest(Base):
    def test_shape(self):
        self.assertTrue(self.gen(steps=2, num_gaussians=32768)["ok"])
        m = read_json(os.path.join(self.job, "asset.json"))
        self.assertEqual(set(m), {"object", "version", "representation", "format", "file", "input", "bytes",
                                  "gaussians", "frame", "camera", "model", "settings", "runtime", "timing_ms"})
        self.assertEqual((m["object"], m["version"], m["representation"], m["format"]),
                         ("asset3d", 1, "gaussian_splat", "ply"))
        self.assertEqual(m["bytes"], os.path.getsize(os.path.join(self.job, "asset.ply")))
        self.assertEqual(m["gaussians"]["count"], 32768)
        self.assertEqual(set(m["gaussians"]["fields"]), {"position", "color", "opacity", "scale", "rotation", "normal"})
        self.assertEqual(m["frame"]["up"], [0.0, -1.0, 0.0])
        self.assertEqual(m["frame"]["bounds"]["min"][0], np.float32(-0.4).item())
        self.assertAlmostEqual(m["frame"]["bounds"]["max"][2], 0.2, places=6)
        cam = m["camera"]
        self.assertEqual(cam["projection"], "perspective")
        # yaw 0, pitch 0, radius 2: model (0, 2, 0) -> PLY (0, 0, 2); model up +z -> PLY -y; then framed on the splats
        self.assertAlmostEqual(cam["position"][2], 2.0)
        self.assertEqual(cam["up"], [0.0, -1.0, 0.0])
        self.assertTrue(cam["framing"].startswith("fitted"))
        self.assertEqual(set(cam["model"]), {"yaw_rad", "pitch_rad", "radius", "fov_deg"})
        self.assertEqual(set(m["settings"]), {"input_mode", "seed", "steps", "guidance", "rescale_t",
                                              "num_gaussians", "crop_margin"})
        self.assertEqual(m["settings"]["rescale_t"], 3.0)
        self.assertEqual(m["model"]["code"]["commit"], "4702a7f00850e34b6e83a2170cc123c3e9570744")
        self.assertEqual([c["license"] for c in m["model"]["components"]], ["Apache-2.0", "DINOv3 License", "Apache-2.0"])
        self.assertEqual(set(m["timing_ms"]), {"preprocess", "encode", "denoise", "decode", "write", "total"})

    def test_camera_math(self):
        pos, up = w.camera_in_frame(math.pi / 2, 0.0, 3.0)  # model +x
        np.testing.assert_allclose(pos, [3.0, 0.0, 0.0], atol=1e-12)
        pos, _ = w.camera_in_frame(0.0, math.pi / 2, 1.0)  # straight above: model +z -> PLY -y
        np.testing.assert_allclose(pos, [0.0, -1.0, 0.0], atol=1e-12)
        self.assertEqual(w.camera_block({"yaw": 0, "pitch": 0, "radius": 1000.0, "fov": 0.1,
                                         "inverse_distance": 1e-4})["projection"], "orthographic")

    def test_framing(self):
        # looking down PLY -z from (0, 0, 10) with up -y: view x = -PLY x, view y = -PLY y
        block = {"projection": "orthographic", "fov_y_deg": 50.0, "position": [0.0, 0.0, 10.0],
                 "target": [0.0, 0.0, 0.0], "up": [0.0, -1.0, 0.0]}
        xyz = np.array([[1.0, 1.0, 0.0], [3.0, 2.0, 0.0], [2.0, 1.5, 0.0], [50.0, 50.0, 0.0]])
        opacity = np.array([0.0, 1.0, 2.0, -1.0])  # the last is under half opaque: ignored
        f = w.frame_camera(block, xyz, opacity, 1.5)
        self.assertAlmostEqual(f["half_height"], 1.0 * 1.5)  # the box is 2 by 1; half its longer side, times margin
        np.testing.assert_allclose(f["target"], [2.0, 1.5, 0.0], atol=1e-12)
        np.testing.assert_allclose(f["position"], [2.0, 1.5, 10.0], atol=1e-12)
        self.assertAlmostEqual(f["fov_y_deg"], math.degrees(2 * math.atan(1.5 / 10)))
        p = w.frame_camera(dict(block, projection="perspective"), xyz, opacity, 1.0)
        self.assertNotIn("half_height", p)
        self.assertAlmostEqual(p["fov_y_deg"], math.degrees(2 * math.atan(0.1)))  # tangents 0.1 to 0.3 and 0.1 to 0.2
        self.assertEqual(w.frame_camera(block, xyz, np.full(4, -1.0), 1.0)["position"], block["position"])

    def test_dino_key_map(self):
        self.assertEqual(w.timm_to_transformers("blocks.3.attn.qkv.weight"),
                         [f"layer.3.attention.{p}_proj.weight" for p in "qkv"])
        self.assertEqual(w.timm_to_transformers("blocks.0.gamma_2"), ["layer.0.layer_scale2.lambda1"])
        self.assertEqual(w.timm_to_transformers("blocks.31.mlp.fc1_x.bias"), ["layer.31.mlp.up_proj.bias"])
        self.assertEqual(w.timm_to_transformers("reg_token"), ["embeddings.register_tokens"])
        self.assertIsNone(w.timm_to_transformers("head.weight"))


class Protocol(unittest.TestCase):
    """worker.py as the daemon runs it: stdout carries only protocol lines."""

    def run_worker(self, lines, config):
        env = dict(os.environ, LOCALROUTER_TOOL=json.dumps(config))
        p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "worker.py")], input="".join(
            json.dumps(x) + "\n" for x in lines), capture_output=True, text=True, env=env, timeout=60)
        return p.returncode, [json.loads(x) for x in p.stdout.splitlines()]

    def test_missing_weights_then_exit(self):
        with tempfile.TemporaryDirectory() as empty:
            code, out = self.run_worker([{"op": "load"}, {"op": "exit"}], tool(weights=empty))
        self.assertEqual(code, 0)
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]["ok"])
        self.assertIn("missing", out[0]["error"])

    def test_bad_config(self):
        code, out = self.run_worker([{"op": "load"}, {"op": "exit"}], {"id": "idol"})
        self.assertEqual(code, 0)
        self.assertIn("bad tool config", out[0]["error"])


if __name__ == "__main__":
    unittest.main()
