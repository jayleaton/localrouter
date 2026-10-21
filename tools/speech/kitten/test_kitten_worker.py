"""The Kitten worker's protocol, request checks and asset checks, without the model (stdlib only).

    python3 -I -m unittest discover -s tools/speech/kitten -p 'test_*.py'
"""

import array
import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
import wave

spec = importlib.util.spec_from_file_location("kitten_worker",
                                              os.path.join(os.path.dirname(__file__), "kitten_worker.py"))
kw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kw)

VOICES = ["Bella", "Bruno", "German"]
PRESETS = ["stable", "expressive"]


def speech(**r):
    return kw.Speech(r, 50, VOICES, "Bruno", PRESETS)


class FakeEngine:
    """Speaks like KittenEngine: two chunks of progress, then a WAV in the job directory."""

    def __init__(self):
        self.loads = 0

    def load(self):
        self.loads += 1
        return 1234

    def request(self, r):
        return speech(**r)

    def generate(self, s, out_dir, progress):
        progress(1, 2)
        progress(2, 2)
        kw.write_wav(os.path.join(out_dir, kw.OUTPUT), array.array("h", [0, 1000, -1000]))
        return {"ok": True, "files": [kw.OUTPUT], "seed": s.seed, "ms": 5}


def run(engine, *commands):
    out = io.StringIO()
    kw.serve(engine, io.StringIO("".join(c if isinstance(c, str) else json.dumps(c) + "\n" for c in commands)), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


class ProtocolTest(unittest.TestCase):
    def test_load_generate_exit(self):
        e = FakeEngine()
        with tempfile.TemporaryDirectory() as d:
            lines = run(e, {"op": "load"}, {"op": "load"},
                        {"op": "generate", "id": "spc_1", "dir": d,
                         "request": {"speech": {"input": "Hello.", "voice": "", "seed": 7}}},
                        {"op": "exit"}, {"op": "load"})
            with wave.open(os.path.join(d, "speech.wav")) as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()),
                                 (1, 2, 24000, 3))
        self.assertEqual(e.loads, 1)  # a second load is answered without reloading; nothing after exit runs
        self.assertEqual(lines[0], {"ok": True, "resident": 1234})
        self.assertEqual(lines[2], {"progress": {"phase": "speech", "step": 1, "of": 2}})
        self.assertEqual(lines[4], {"ok": True, "files": ["speech.wav"], "seed": 7, "ms": 5})
        self.assertEqual(len(lines), 5)

    def test_refusals_and_failures(self):
        with tempfile.TemporaryDirectory() as d:
            lines = run(FakeEngine(),
                        {"op": "generate", "dir": d, "request": {"speech": {"input": "x"}}},
                        {"op": "load"},
                        {"op": "generate", "dir": d, "request": {"image": {"prompt": "x"}}},
                        {"op": "generate", "dir": d, "request": {"speech": {"input": "x", "speed": 2}}},
                        {"op": "generate", "dir": "/nonexistent/job", "request": {"speech": {"input": "x"}}},
                        "not json\n", {"op": "dance"})
        self.assertEqual(lines[0]["type"], "server_error")
        self.assertIn("before load", lines[0]["error"])
        self.assertEqual([lines[i]["type"] for i in (2, 3)], ["invalid_request_error"] * 2)
        self.assertIn("speech only", lines[2]["error"])
        self.assertEqual(lines[4]["type"], "server_error")
        self.assertEqual(lines[5]["error"], "malformed command")
        self.assertIn("unknown op", lines[6]["error"])

    def test_failed_load_is_reported(self):
        class Broken(FakeEngine):
            def load(self):
                raise RuntimeError("missing pinned file")

        lines = run(Broken(), {"op": "load"})
        self.assertEqual(lines, [{"ok": False, "error": "RuntimeError: missing pinned file", "type": "server_error"}])


class RequestTest(unittest.TestCase):
    def refused(self, why, **r):
        with self.assertRaises(kw.Refused) as c:
            speech(**r)
        self.assertIn(why, str(c.exception))

    def test_defaults(self):
        s = speech(input="Hello.", voice="", preset="")
        self.assertEqual((s.voice, s.preset), ("Bruno", None))
        self.assertTrue(0 <= s.seed < 1 << 63)
        self.assertEqual(speech(input="Hi", voice="German", speed=1.0, seed=0, preset="expressive").seed, 0)
        self.assertEqual(speech(input="Hi", seed=(1 << 64) - 1).seed, (1 << 64) - 1)

    def test_refusals(self):
        self.refused("input is required", input="   ")
        self.refused("input is required", voice="Bruno")
        self.refused("too long", input="x" * 51)
        self.refused("unknown voice", input="Hi", voice="Nobody")
        self.refused("speed", input="Hi", speed=0.5)
        self.refused("instructions", input="Hi", instructions="cheerful")
        self.refused("streaming", input="Hi", stream_format="sse")
        self.refused("preset", input="Hi", preset="loud")
        self.refused("seed", input="Hi", seed=-1)
        self.refused("seed", input="Hi", seed=1 << 64)
        self.refused("seed", input="Hi", seed="7")


class OptionsTest(unittest.TestCase):
    def test_lossless_by_default(self):
        o = kw.Options({"weights": "/m", "options": {"voices": ["Bruno"], "something_shared": 1, "min_speed": 1,
                                                      "max_speed": 1, "max_input_chars": 4096}})
        self.assertEqual((o.root, o.lm_weights, o.decoder, o.max_chars), ("/m", "packed", "default", 4096))
        one = {"min_speed": 1, "max_speed": 1}
        with self.assertRaisesRegex(ValueError, "lossy"):
            kw.Options({"options": {"lm_weights": "emb4", **one}})
        with self.assertRaisesRegex(ValueError, "lossy"):
            kw.Options({"options": {"decoder": "student_w4", **one}})
        self.assertEqual(kw.Options({"options": {"decoder": "student_w4", "allow_lossy": True, **one}}).decoder,
                         "student_w4")
        with self.assertRaises(ValueError):
            kw.Options({"options": {"verify": "maybe", **one}})
        with self.assertRaisesRegex(ValueError, "deterministic"):
            kw.Options({"options": {"deterministic": True, **one}})  # changes HiFT's output on CUDA
        with self.assertRaisesRegex(ValueError, "speed"):
            kw.Options({"options": {}})  # the daemon's default range, 0.25 to 4, would advertise a control it lacks

    def test_lm_decode(self):
        one = {"min_speed": 1, "max_speed": 1}
        self.assertEqual(kw.Options({"options": one}).lm_decode, "eager")  # the reference's bits unless asked
        o = kw.Options({"options": {"lm_decode": "graphs", "device": "cuda", **one}})
        self.assertEqual((o.lm_decode, o.graph_tokens), ("graphs", 2048))
        with self.assertRaisesRegex(ValueError, "lm_decode"):
            kw.Options({"options": {"lm_decode": "compiled", **one}})
        with self.assertRaisesRegex(ValueError, "cuda"):
            kw.Options({"options": {"lm_decode": "graphs", "device": "cpu", **one}})
        with self.assertRaisesRegex(ValueError, "graph_tokens"):
            kw.Options({"options": {"lm_decode": "graphs", "graph_tokens": 64, **one}})

    def test_config_example(self):
        with open(os.path.join(os.path.dirname(__file__), "config.example.json")) as f:
            config = json.load(f)
        modes = {}
        for tool in config["tools"]:  # the same model twice: eager (the reference's bits) and graphs
            o = kw.Options(tool)
            self.assertEqual((tool["kind"], len(o.voices), o.presets), ("speech", 47, ["stable", "expressive"]))
            self.assertIn(tool["options"]["default_voice"], o.voices)
            self.assertTrue(kw.LANGUAGE_VOICES <= set(o.voices))
            self.assertTrue(tool["qualification"].endswith(f"qualification-{o.lm_decode}.json"))
            modes[tool["id"]] = o.lm_decode
        self.assertEqual(modes, {"kitten-tts-2": "eager", "kitten-tts-2-graphs": "graphs"})
        self.assertEqual(config["defaults"]["text_to_speech"], "kitten-tts-2-graphs")


class AssetTest(unittest.TestCase):
    def test_verify(self):
        with tempfile.TemporaryDirectory() as root:
            pins = kw.load_pins()
            kitten_dir, s3gen = kw.asset_paths(root, pins)
            body = b"weights"
            digest = hashlib.sha256(body).hexdigest()
            needed = kw.needed_files(pins, "packed", "default")
            self.assertIn(("kitten", "lm/model-ternary.safetensors"), needed)
            self.assertNotIn(("kitten", "lm/model.safetensors"), needed)
            files = {"kitten": {}, "s3gen": {}}
            for repo, rel in needed:
                path = os.path.join(kitten_dir, rel) if repo == "kitten" else s3gen
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(body)
                files[repo][rel] = [len(body), digest]
            fake = dict(pins, files=files)
            for mode in ("size", "sha256"):
                kw.verify_assets(root, fake, mode, "packed", "default")
            files["s3gen"][pins["s3gen"]["file"]] = [len(body), "0" * 64]
            kw.verify_assets(root, fake, "size", "packed", "default")
            with self.assertRaisesRegex(RuntimeError, "sha256"):
                kw.verify_assets(root, fake, "sha256", "packed", "default")
            os.remove(s3gen)
            with self.assertRaisesRegex(RuntimeError, "missing pinned file"):
                kw.verify_assets(root, fake, "size", "packed", "default")

    def test_pins_cover_every_loaded_file(self):
        pins = kw.load_pins()
        for lm in pins["kitten"]["lm"]:
            for decoder in pins["kitten"]["decoders"]:
                for repo, rel in kw.needed_files(pins, lm, decoder):
                    size, digest = pins["files"][repo][rel]
                    self.assertGreater(size, 0)
                    self.assertEqual(len(digest), 64)


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy not installed")
class PcmTest(unittest.TestCase):
    def test_pcm16(self):
        import numpy as np
        got = kw.pcm16(np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5 / 32767, 1.5 / 32767], np.float32))
        self.assertEqual(got.dtype.str, "<i2")
        self.assertEqual(got.tolist(), [0, 32767, -32767, 32767, -32767, 0, 2])


if __name__ == "__main__":
    unittest.main()
