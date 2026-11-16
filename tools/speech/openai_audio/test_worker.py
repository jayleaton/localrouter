#!/usr/bin/env python3
"""Tests of worker.py against a fake OpenAI audio server (standard library only, no GPU):
  python3 -I tools/speech/openai_audio/test_worker.py
The worker starts the fake server itself through options.server (as it starts a real one), speaks the protocol on
stdin/stdout, and stops the server on exit."""
import json, os, subprocess, sys, tempfile, wave

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = r'''
import json, sys, http.server
port = int(sys.argv[1])
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        b = json.dumps({"data": [{"id": "fake"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/v1/audio/speech":
            req = json.loads(body)
            if req.get("seed", 0) > 0xFFFFFFFF or req.get("voice") == "bad":
                b = b'{"error":{"message":"bad"}}'
                self.send_response(400); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
            pcm = b"\x01\x00" * 2400 + b"\x02"   # 0.1 s at 24 kHz plus one stray byte, sent chunked
            self.send_response(200); self.send_header("Transfer-Encoding", "chunked"); self.send_header("x-sample-rate", "24000"); self.end_headers()
            for i in range(0, len(pcm), 777):
                c = pcm[i:i + 777]; self.wfile.write(b"%x\r\n" % len(c) + c + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
        else:
            ok = b'name="file"' in body and b'name="language"' in body
            b = json.dumps({"text": " hello world " if ok else "missing fields"}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

def main():
    tmp = tempfile.mkdtemp()
    fake = os.path.join(tmp, "fake.py"); open(fake, "w").write(FAKE)
    cfg = {"id": "fake-audio", "load_timeout_s": 20, "options": {"server": [sys.executable, "-I", fake, "{port}"], "stream": True}}
    w = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         env={**os.environ, "LOCALROUTER_TOOL": json.dumps(cfg)}, text=True)
    def call(obj):
        w.stdin.write(json.dumps(obj) + "\n"); w.stdin.flush()
        while True:
            r = json.loads(w.stdout.readline())
            if "progress" not in r: return r
    fails = 0
    def check(name, cond, got=""):
        nonlocal fails
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {got}")); fails += not cond
    r = call({"op": "load"}); check("load", r.get("ok") is True, r)
    job = os.path.join(tmp, "job1"); os.mkdir(job)
    r = call({"op": "generate", "id": "s1", "dir": job, "request": {"speech": {"input": "Hi.", "voice": "v", "seed": 2**40 + 5, "stream": True}}})
    check("speech ok (64-bit seed folded)", r.get("ok") is True and r.get("files") == ["speech.wav"], r)
    with wave.open(os.path.join(job, "speech.wav")) as f:
        check("speech.wav: 24 kHz mono 16-bit, whole samples", (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()) == (24000, 1, 2, 2400),
              (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()))
    r = call({"op": "generate", "id": "s2", "dir": job, "request": {"speech": {"input": "Hi.", "voice": "bad"}}})
    check("server 4xx is an invalid_request_error", r.get("type") == "invalid_request_error", r)
    r = call({"op": "generate", "id": "s3", "dir": job, "request": {"speech": {"input": "  "}}})
    check("empty input refused", r.get("type") == "invalid_request_error", r)
    open(os.path.join(job, "audio.wav"), "wb").write(b"RIFF....WAVE")
    r = call({"op": "generate", "id": "t1", "dir": job, "request": {"transcription": {"audio": "audio.wav", "language": "en", "duration_ms": 1500}}})
    t = json.load(open(os.path.join(job, "transcript.json"))) if r.get("ok") else {}
    check("transcription: transcript.json", r.get("ok") is True and t.get("text") == "hello world" and t.get("duration") == 1.5 and len(t.get("segments", [])) == 1, (r, t))
    r = call({"op": "generate", "id": "t2", "dir": job, "request": {"transcription": {"audio": "../x.wav"}}})
    check("audio outside the job dir refused", r.get("type") == "invalid_request_error", r)
    w.stdin.write(json.dumps({"op": "exit"}) + "\n"); w.stdin.flush()
    check("exit 0", w.wait(timeout=30) == 0)
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
