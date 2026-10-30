#!/usr/bin/env python3
"""CPU tests for the macOS Drama 3 path: the voice conversion and drama3_worker_macos.py against a stand-in launcher
and server (no weights, no GPU). Run: python3 -I tools/speech/drama3/test_worker_macos.py"""

import hashlib
import json
import os
import socket
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import voices_macos  # noqa: E402

# A stand-in for pilot_macos.py and the Metal server: the launcher's arguments, session.json, DRAMA_PILOT_READY, then
# GET /v1/voices (the catalog in --voices) and POST /v1/audio/speech (chunked PCM, or WAV) until SIGTERM.
FAKE_PILOT = r'''
import http.server, json, os, signal, socketserver, struct, sys
a = sys.argv[1:]
serve = a[a.index("--") + 1:] if "--" in a else []
arg = lambda k: a[a.index(k) + 1]
voices = serve[serve.index("--voices") + 1]
names = sorted(n[:-5] for n in os.listdir(voices) if n.endswith(".json"))
with open(os.path.join(arg("--runtime-dir"), "launches"), "a") as f:
    f.write(json.dumps({"argv": a, "voices": names}) + "\n")
ev = arg("--evidence-dir"); os.makedirs(ev)
json.dump({"server_pgid": os.getpid()}, open(os.path.join(ev, "session.json"), "w"))
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *x): pass
    def do_GET(self):
        body = json.dumps({"voices": ["default"] + names}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(os.path.join(arg("--runtime-dir"), "requests"), "a") as f:
            f.write(json.dumps(req) + "\n")
        if "references" in req or req.get("voice", "default") not in names + ["default"]:
            body = b'{"error":{"message":"refused"}}'
            self.send_response(400); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        pcm = struct.pack("<4h", 1, -1, 2, -2)
        if req.get("stream"):
            self.send_response(200); self.send_header("Transfer-Encoding", "chunked"); self.send_header("X-Sample-Rate", "44100")
            self.send_header("X-Channels", "1"); self.end_headers()
            for part in (pcm[:4], pcm[4:]):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
            self.wfile.write(b"0\r\n\r\n")
        else:
            body = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16, b"data", len(pcm)) + pcm
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.send_header("X-Finish-Reason", "stop")
            self.end_headers(); self.wfile.write(body)
class S(http.server.HTTPServer):
    def server_bind(self):  # HTTPServer's own looks the host name up (getfqdn), which can take 30 s
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]
srv = S(("127.0.0.1", int(arg("--port"))), H)
signal.signal(signal.SIGTERM, lambda *x: sys.exit(0))
print("DRAMA_PILOT_READY", flush=True)
srv.serve_forever()
'''


def grid(frames, seed):
    return [[(seed * 7 + k * 13 + f) % (4096 if k == 0 else 1024) for f in range(frames)] for k in range(10)]


class Conversion(unittest.TestCase):
    def test_named_voices_match_the_pins_and_round_trip(self):
        pins = json.load(open(os.path.join(HERE, "pins-macos.json")))
        with tempfile.TemporaryDirectory() as out:
            paths = [os.path.join(HERE, "voices", n) for n in sorted(pins["voices"]["files"])]
            self.assertEqual(voices_macos.convert(out, paths), pins["voices"]["files"])
            for path in paths:
                src = json.load(open(path))
                cat = json.load(open(os.path.join(out, os.path.basename(path))))
                self.assertEqual(cat["source"]["kind"], "synthetic")
                self.assertEqual(cat["references"][0]["text"], src["text"])
                self.assertEqual(voices_macos.codebook_major(cat["references"][0]["codes"]), src["vq_codes"])

    def test_catalog_limits(self):
        ok = {"text": "Hello.", "vq_codes": grid(5, 1)}
        voices_macos.catalog([ok], "consented", "note")
        for bad in ({"text": "Hi<|im_end|>", "vq_codes": grid(5, 1)}, {"text": " ", "vq_codes": grid(5, 1)},
                    {"text": "Hi.", "vq_codes": grid(647, 1)}, {"text": "Hi.", "vq_codes": [[0] * 5] * 9},
                    {"text": "Hi.", "vq_codes": [[0] * 5] + [[1024] * 5] * 9}):
            with self.assertRaises(voices_macos.Invalid):
                voices_macos.catalog([bad], "consented", "note")
        with self.assertRaises(voices_macos.Invalid):
            voices_macos.catalog([ok], "model-invented", "note")
        with self.assertRaises(voices_macos.Invalid):
            voices_macos.catalog([ok] * 5, "consented", "note")

    def test_pack_joins_the_smallest_adjacent_pairs(self):
        refs = [{"text": f"t{i}", "vq_codes": grid(n, i)} for i, n in enumerate((300, 10, 20, 300, 40, 50))]
        packed = voices_macos.pack(refs)
        self.assertEqual([len(r["vq_codes"][0]) for r in packed], [300, 30, 300, 90])
        self.assertEqual(packed[1]["text"], "t1 t2")
        self.assertEqual(packed[1]["vq_codes"][3], refs[1]["vq_codes"][3] + refs[2]["vq_codes"][3])
        self.assertIsNone(voices_macos.pack([{"text": "x", "vq_codes": grid(400, i)} for i in range(5)]))


class Worker(unittest.TestCase):
    def setUp(self):
        import drama3_worker_macos as mac
        self.mac = mac
        self.dir = tempfile.TemporaryDirectory()
        root = self.root = self.dir.name
        os.makedirs(os.path.join(root, "launcher"))
        with open(os.path.join(root, "launcher", "pilot_macos.py"), "w") as f:
            f.write(FAKE_PILOT)
        native = os.path.join(root, "bin", "tensorfold-native")
        os.makedirs(os.path.dirname(native))
        with open(native, "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(native, 0o755)
        voices_macos.convert(os.path.join(root, "voices-macos"),
                             [os.path.join(HERE, "voices", n) for n in ("Mae.json", "Arlo.json")])
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.tool = {"weights": root, "options": {"variant": "nvfp4", "voices": ["Mae", "Arlo"], "default_voice": "Mae",
                                                  "verify": "none", "port": port, "native_bin": native,
                                                  "pilot": os.path.join(root, "launcher", "pilot_macos.py")}}
        self.w = mac.MacWorker(self.tool, require_qualification=False, environ={})

    def tearDown(self):
        self.w._stop()
        self.dir.cleanup()

    def lines(self, name):
        with open(os.path.join(self.root, "runtime", name)) as f:
            return [json.loads(ln) for ln in f]

    def job(self, name, data):
        d = tempfile.mkdtemp(dir=self.root)
        if name:
            with open(os.path.join(d, name), "w") as f:
                json.dump(data, f)
        return d

    def test_lifecycle_voices_and_a_clone(self):
        self.assertEqual(self.w.load(), {"ok": True, "resident": 0, "cancel": "signal"})
        launch = self.lines("launches")[0]
        self.assertIn("--external-admission", launch["argv"])
        self.assertEqual(launch["argv"][launch["argv"].index("--parent-pid") + 1], str(os.getpid()))
        self.assertIn("--cfg-batched", launch["argv"])
        self.assertEqual(launch["voices"], ["Arlo", "Mae"])

        d = self.job(None, None)
        reply = self.w.generate({"dir": d, "request": {"speech": {"input": "Hi.", "voice": "", "stream": True}}})
        self.assertTrue(reply["ok"], reply)
        with open(os.path.join(d, "speech.wav"), "rb") as f:
            wav = f.read()
        self.assertEqual(self.mac.base.canonical_wav(wav), 44100)
        self.assertEqual(len(wav), 44 + 8)
        self.assertEqual(self.lines("requests")[-1]["voice"], "Mae")

        refused = self.w.generate({"dir": self.job("r.json", [{"text": "x", "vq_codes": grid(3, 1)}]),
                                   "request": {"speech": {"input": "Hi.", "references": "r.json"}}})
        self.assertEqual(refused["type"], "invalid_request_error")

        voice = {"format": "localrouter-voice-codes/1",
                 "references": [{"text": f"Part {i}.", "vq_codes": grid(100, i)} for i in range(6)]}
        d = self.job("voice.json", voice)
        reply = self.w.generate({"dir": d, "request": {"speech": {"input": "Hi.", "voice": "Sam",
                                                                  "voice_references": "voice.json"}}})
        self.assertTrue(reply["ok"], reply)
        name = self.lines("requests")[-1]["voice"]
        self.assertTrue(name.startswith("clone-"))
        launches = self.lines("launches")
        self.assertEqual(len(launches), 2)
        self.assertEqual(launches[1]["voices"], sorted(["Arlo", "Mae", name]))
        with open(os.path.join(self.root, "runtime", "clones", name + ".json")) as f:
            cat = json.load(f)
        self.assertEqual(cat["source"]["kind"], "consented")
        self.assertEqual(len(cat["references"]), 4)

        # the same voice again: no restart; and a fresh worker's load has it in the catalog already
        self.assertTrue(self.w.generate({"dir": self.job("voice.json", voice), "request": {"speech": {
            "input": "Again.", "voice": "Sam", "voice_references": "voice.json"}}})["ok"])
        self.assertEqual(len(self.lines("launches")), 2)
        self.assertEqual(self.w.exit(), {"ok": True})
        self.assertFalse(os.path.exists(os.path.join(self.root, "runtime", self.mac.SERVER_FILE)))
        self.assertTrue(self.w.load()["ok"])
        self.assertIn(name, self.lines("launches")[-1]["voices"])

    def test_options_refuse_what_the_metal_engine_lacks(self):
        for extra in ({"long_form": True}, {"references": True}, {"variant": "full"}, {"max_frames": 9000}):
            with self.assertRaises(ValueError):
                self.mac.MacOptions({**self.tool, "options": {**self.tool["options"], **extra}})


if __name__ == "__main__":
    unittest.main()
