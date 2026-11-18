"""The Wan worker against a fake engine: lifecycle, refusals, cancellation, the first frame's fit and the MP4.
    VENV/bin/python -I tools/video/wan/test_worker.py      (the worker's venv: numpy and Pillow; ffmpeg on PATH; no weights, no GPU)
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import worker as w  # noqa: E402


class FakeEngine:
    """The Engine interface; records what it was asked. `fail`: crash (a RuntimeError) or oom."""

    def __init__(self):
        self.loads, self.calls, self.fail, self.cancel_at = 0, [], None, None

    def load(self):
        self.loads += 1

    def resident(self):
        return 23 << 30

    def generate(self, v, image, made, steps, guidance, negative, on_step):
        self.calls.append({"v": v, "image": image, "made": made, "steps": steps, "guidance": guidance, "negative": negative})
        if self.fail == "crash":
            raise RuntimeError("boom")
        if self.fail == "oom":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        for k in range(steps):
            if self.cancel_at == k:
                w.CANCEL.set()
            if w.CANCEL.is_set():
                raise w.Cancelled()
            on_step(k + 1)
        frames = np.zeros((made, v["height"], v["width"], 3), dtype=np.uint8)
        frames[:, :, :, 0] = np.arange(made, dtype=np.uint8)[:, None, None]
        return frames, 1 << 30


def tool(**opts):
    return {"id": "wan", "kind": "video", "weights": WEIGHTS, "options": opts}


def req(**v):
    base = {"prompt": "a toy car", "width": 256, "height": 160, "seconds": 1, "fps": 24, "seed": 7, "audio": False}
    base.update(v)
    return {"op": "generate", "id": "vid_1", "dir": JOB, "request": {"video": base}}


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,nb_frames,r_frame_rate",
                          "-of", "json", path], capture_output=True, check=True).stdout
    return json.loads(out)["streams"]


class WorkerTest(unittest.TestCase):
    def setUp(self):
        for f in os.listdir(JOB):
            os.remove(os.path.join(JOB, f))
        w.CANCEL.clear()
        self.engine = FakeEngine()
        self.worker = w.Worker(tool(max_seconds=5, max_side=1280, max_pixels=901120), self.engine)
        self.progress = []

    def run_job(self, cmd):
        return w.run_generate_with(self.worker, cmd, self.progress.append)

    def test_load_once_and_resident(self):
        self.assertEqual(self.worker.load(), {"ok": True, "resident": 23 << 30, "cancel": "signal"})
        self.worker.load()
        self.assertEqual(self.engine.loads, 1)

    def test_text_to_video_writes_the_clip_without_audio(self):
        r = self.run_job(req(seconds=2))
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["files"], ["video.mp4"])
        self.assertEqual(r["seed"], 7)
        call = self.engine.calls[0]
        self.assertEqual(call["made"], 49)  # 4k+1 frames, one past the 48 kept
        self.assertEqual((call["steps"], call["guidance"], call["negative"]), (50, 5.0, w.NEGATIVE))
        self.assertIsNone(call["image"])
        streams = probe(os.path.join(JOB, "video.mp4"))
        self.assertEqual([s["codec_type"] for s in streams], ["video"])
        self.assertEqual((streams[0]["width"], streams[0]["height"], streams[0]["nb_frames"], streams[0]["r_frame_rate"]),
                         (256, 160, "48", "24/1"))
        self.assertEqual(self.progress[-2], {"phase": "denoise", "step": 50, "of": 50})

    def test_request_steps_guidance_and_negative_win(self):
        self.worker = w.Worker(tool(steps=30, guidance=4.0), self.engine)
        self.assertTrue(self.run_job(req(steps=8, guidance=3.5, negative_prompt="blurry"))["ok"])
        self.assertEqual((self.engine.calls[0]["steps"], self.engine.calls[0]["guidance"], self.engine.calls[0]["negative"]),
                         (8, 3.5, "blurry"))
        self.assertTrue(self.run_job(req())["ok"])
        self.assertEqual((self.engine.calls[1]["steps"], self.engine.calls[1]["guidance"]), (30, 4.0))

    def test_first_frame_is_cut_to_the_clip_shape_not_stretched(self):
        im = Image.new("RGB", (300, 100), (0, 0, 255))
        im.paste((255, 0, 0), (100, 0, 200, 100))  # a red middle third
        im.save(os.path.join(JOB, "first_frame_0.png"))
        self.assertTrue(self.run_job(req(width=160, height=160, first_frame="first_frame_0.png"))["ok"])
        got = self.engine.calls[0]["image"]
        self.assertEqual(got.size, (160, 160))
        self.assertEqual(got.getpixel((80, 80)), (255, 0, 0))
        self.assertEqual(got.getpixel((2, 80)), (255, 0, 0))  # the blue sides are cut away

    def test_refusals(self):
        cases = [
            (req(audio=True), "makes no audio"),
            (req(fps=16), "24 fps only"),
            (req(width=250), "multiples of 32"),
            (req(width=1344, height=768), "at most 1280 pixels a side"),
            (req(width=1280, height=736), "at most 901120 pixels a frame"),
            (req(seconds=6), "seconds must be 1 to 5"),
            (req(prompt=" "), "prompt is required"),
            (req(first_frame="missing.png"), "not an image"),
            ({"op": "generate", "dir": JOB, "request": {"image": {"prompt": "x"}}}, "makes videos"),
        ]
        for cmd, why in cases:
            r = self.run_job(cmd)
            self.assertEqual(r["type"], "invalid_request_error", (cmd, r))
            self.assertIn(why, r["error"])
        self.assertEqual(self.engine.calls, [])

    def test_cancel_keeps_the_worker_loaded(self):
        self.engine.cancel_at = 3
        self.assertEqual(self.run_job(req())["type"], "cancelled")
        self.assertFalse(os.path.exists(os.path.join(JOB, "video.mp4")))
        self.engine.cancel_at = None
        self.assertTrue(self.run_job(req())["ok"])
        self.assertEqual(self.engine.loads, 1)

    def test_failures_name_the_cause(self):
        self.engine.fail = "oom"
        self.assertEqual(self.run_job(req())["type"], "service_unavailable")
        self.engine.fail = "crash"
        r = self.run_job(req())
        self.assertEqual((r["type"], r["error"]), ("server_error", "RuntimeError: boom"))

    def test_bad_config(self):
        with self.assertRaises(ValueError):
            w.Worker({"options": {"steps": 8}}, FakeEngine())
        with self.assertRaises(ValueError):
            w.Worker(tool(colour="red"), FakeEngine())
        with self.assertRaises(ValueError):
            w.Worker(tool(transformer_storage="int4"), FakeEngine())

    def test_protocol_over_stdio(self):
        env = dict(os.environ, LOCALROUTER_TOOL=json.dumps({"weights": os.path.join(WEIGHTS, "absent")}))
        p = subprocess.run([sys.executable, "-I", os.path.join(HERE, "worker.py")], input='{"op":"load"}\nnot json\n{"op":"exit"}\n',
                           capture_output=True, text=True, env=env, timeout=60)
        lines = [json.loads(x) for x in p.stdout.splitlines()]
        self.assertEqual([x["ok"] for x in lines], [False, False])
        self.assertIn("run fetch.sh", lines[0]["error"])
        self.assertEqual(lines[1]["error"], "not a JSON line")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as root:
        WEIGHTS, JOB = os.path.join(root, "weights"), os.path.join(root, "job")
        os.makedirs(WEIGHTS)
        os.makedirs(JOB)
        open(os.path.join(WEIGHTS, "model_index.json"), "w").write("{}")
        unittest.main(verbosity=1)
