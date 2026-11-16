#!/usr/bin/env python3
"""Tests of worker.py against a fake OpenAI audio server (standard library only, no GPU):
  python3 -I tools/speech/openai_audio/test_worker.py
The worker starts the fake server itself through options.server (as it starts a real one), speaks the protocol on
stdin/stdout, and stops the server on exit. The fake records every transcription request's form fields, so the tests
see what the worker forwarded."""
import json, os, subprocess, sys, tempfile, wave

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = r'''
import json, re, sys, http.server
port, record = int(sys.argv[1]), sys.argv[2]
def fields(body, ctype):
    bound = ctype.split("boundary=")[1].encode()
    out = []
    for part in body.split(b"--" + bound)[1:-1]:
        head, _, data = part.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]+)"', head).group(1).decode()
        if b"filename=" not in head:
            out.append([name, data[:-2].decode()])
        else:
            out.append([name, "<file>"])
    return out
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def reply(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.reply(200, {"data": [{"id": "fake"}]})
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/v1/audio/speech":
            req = json.loads(body)
            if req.get("seed", 0) > 0xFFFFFFFF or req.get("voice") == "bad":
                return self.reply(400, {"error": {"message": "bad"}})
            pcm = b"\x01\x00" * 2400 + b"\x02"   # 0.1 s at 24 kHz plus one stray byte, sent chunked
            self.send_response(200); self.send_header("Transfer-Encoding", "chunked"); self.send_header("x-sample-rate", "24000"); self.end_headers()
            for i in range(0, len(pcm), 777):
                c = pcm[i:i + 777]; self.wfile.write(b"%x\r\n" % len(c) + c + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            return
        f = fields(body, self.headers["Content-Type"])
        with open(record, "a") as r:
            r.write(json.dumps(f) + "\n")
        d = {}
        for k, v in f:
            d.setdefault(k, []).append(v)
        if float(d.get("temperature", ["0"])[0]) > 1:
            return self.reply(400, {"error": {"message": "temperature above 1 is not supported"}})
        ok = "file" in d and "language" in d
        res = {"text": " hello world " if ok else "missing fields"}
        grans = d.get("timestamp_granularities[]", [])
        if d.get("response_format") == ["verbose_json"]:
            res.update({"language": "english", "duration": 1.5})
            if "segment" in grans and d.get("prompt") != ["nothing"]:
                res["segments"] = [{"id": 0, "start": 0.0, "end": 0.7, "text": "hello"}, {"id": 1, "start": 0.7, "end": 1.5, "text": "world"}]
            if "word" in grans and d.get("prompt") != ["nothing"]:
                res["words"] = [{"word": "hello", "start": 0.1, "end": 0.6}, {"word": "world", "start": 0.8, "end": 1.4}]
        self.reply(200, res)
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

fails = 0
def check(name, cond, got=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f": {got}")); fails += not cond


class W:
    """One worker (and its fake server) for a tool config's options."""
    def __init__(self, tmp, name, **opts):
        self.record = os.path.join(tmp, name + ".jsonl")
        fake = os.path.join(tmp, "fake.py")
        cfg = {"id": "fake-audio", "load_timeout_s": 20, "options": {"server": [sys.executable, "-I", fake, "{port}", self.record], **opts}}
        self.p = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "worker.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env={**os.environ, "LOCALROUTER_TOOL": json.dumps(cfg)}, text=True)
    def call(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n"); self.p.stdin.flush()
        while True:
            r = json.loads(self.p.stdout.readline())
            if "progress" not in r: return r
    def sent(self):
        """The form fields of the transcription requests the server got, oldest first ([[name, value], ...] each)."""
        return [json.loads(l) for l in open(self.record)] if os.path.exists(self.record) else []
    def exit(self):
        self.p.stdin.write(json.dumps({"op": "exit"}) + "\n"); self.p.stdin.flush()
        return self.p.wait(timeout=30)


def transcribe(w, job, n, **q):
    r = w.call({"op": "generate", "id": f"t{n}", "dir": job, "request": {"transcription": {"audio": "audio.wav", "language": "en", "duration_ms": 1500, **q}}})
    t = json.load(open(os.path.join(job, "transcript.json"))) if r.get("ok") else {}
    return r, t


def field(sent, name):
    return [v for k, v in sent if k == name]


def main():
    tmp = tempfile.mkdtemp()
    open(os.path.join(tmp, "fake.py"), "w").write(FAKE)
    job = os.path.join(tmp, "job1"); os.mkdir(job)
    open(os.path.join(job, "audio.wav"), "wb").write(b"RIFF....WAVE")

    # a server without timestamps (Gufo's Qwen3-ASR: no timestamp_granularities, temperature 0 only)
    w = W(tmp, "plain", stream=True)
    r = w.call({"op": "load"}); check("load", r.get("ok") is True, r)
    r = w.call({"op": "generate", "id": "s1", "dir": job, "request": {"speech": {"input": "Hi.", "voice": "v", "seed": 2**40 + 5, "stream": True}}})
    check("speech ok (64-bit seed folded)", r.get("ok") is True and r.get("files") == ["speech.wav"], r)
    with wave.open(os.path.join(job, "speech.wav")) as f:
        check("speech.wav: 24 kHz mono 16-bit, whole samples", (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()) == (24000, 1, 2, 2400),
              (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getnframes()))
    r = w.call({"op": "generate", "id": "s2", "dir": job, "request": {"speech": {"input": "Hi.", "voice": "bad"}}})
    check("server 4xx is an invalid_request_error", r.get("type") == "invalid_request_error", r)
    r = w.call({"op": "generate", "id": "s3", "dir": job, "request": {"speech": {"input": "  "}}})
    check("empty input refused", r.get("type") == "invalid_request_error", r)
    r, t = transcribe(w, job, 1)
    check("transcription: transcript.json", r.get("ok") is True and t.get("text") == "hello world" and t.get("duration") == 1.5 and len(t.get("segments", [])) == 1, (r, t))
    s = w.sent()[-1]
    check("default: temperature 0, response_format json, no granularities", (field(s, "temperature"), field(s, "response_format"), field(s, "timestamp_granularities[]")) == (["0"], ["json"], []), s)
    r, t = transcribe(w, job, 2, temperature=0.7, prompt="names: Ada", timestamps="segment")
    s = w.sent()[-1]
    check("temperature 0.7 is forwarded, not replaced by 0", field(s, "temperature") == ["0.7"], s)
    check("language and prompt are forwarded", (field(s, "language"), field(s, "prompt")) == (["en"], ["names: Ada"]), s)
    check("segment, not declared: the whole clip as one segment, no granularity sent",
          r.get("ok") is True and field(s, "timestamp_granularities[]") == [] and t.get("segments") == [{"id": 0, "start": 0.0, "end": 1.5, "text": "hello world"}], (r, t, s))
    n = len(w.sent())
    r, t = transcribe(w, job, 3, temperature=0.7, timestamps="word")
    check("word, not declared: refused before the server is asked", r.get("type") == "invalid_request_error" and "word" in r.get("error", "") and len(w.sent()) == n, (r, len(w.sent())))
    r, t = transcribe(w, job, 4, temperature=1.5)
    check("a temperature the server refuses is the caller's 400, with the server's reason",
          r.get("type") == "invalid_request_error" and "temperature above 1" in r.get("error", "") and field(w.sent()[-1], "temperature") == ["1.5"], r)
    r, t = transcribe(w, job, 5, timestamps="phoneme")
    check("an unknown timestamps value is refused", r.get("type") == "invalid_request_error", r)
    r, t = transcribe(w, job, 6, temperature="warm")
    check("a temperature that is not a number is refused", r.get("type") == "invalid_request_error", r)
    r = w.call({"op": "generate", "id": "t7", "dir": job, "request": {"transcription": {"audio": "../x.wav"}}})
    check("audio outside the job dir refused", r.get("type") == "invalid_request_error", r)
    check("exit 0", w.exit() == 0)

    # a server with segment and word timestamps (an OpenAI-style Whisper server)
    w = W(tmp, "stamped", timestamps=["segment", "word"])
    r, t = transcribe(w, job, 10, temperature=0.7, timestamps="word")
    s = w.sent()[-1]
    check("word, declared: verbose_json with segment and word granularities",
          (field(s, "response_format"), field(s, "timestamp_granularities[]"), field(s, "temperature")) == (["verbose_json"], ["segment", "word"], ["0.7"]), s)
    check("word, declared: the words and the server's segments in transcript.json",
          r.get("ok") is True and [x["word"] for x in t.get("words", [])] == ["hello", "world"] and len(t.get("segments", [])) == 2, (r, t))
    r, t = transcribe(w, job, 11, timestamps="segment")
    s = w.sent()[-1]
    check("segment, declared: verbose_json with the segment granularity only",
          (field(s, "response_format"), field(s, "timestamp_granularities[]")) == (["verbose_json"], ["segment"]) and len(t.get("segments", [])) == 2 and "words" not in t, (s, t))
    r, t = transcribe(w, job, 12, timestamps="none")
    s = w.sent()[-1]
    check("none: no granularities, the default response_format", (field(s, "response_format"), field(s, "timestamp_granularities[]")) == (["json"], []), s)
    r, t = transcribe(w, job, 13, timestamps="word", prompt="nothing")
    check("word, declared, none returned: a server error, not a transcript without words", r.get("ok") is False and r.get("type") == "server_error" and "words" in r.get("error", ""), r)
    r, t = transcribe(w, job, 14, timestamps="segment", prompt="nothing")
    check("segment, declared, none returned: a server error", r.get("ok") is False and r.get("type") == "server_error", r)
    check("exit 0 (timestamps worker)", w.exit() == 0)
    print(f"{'FAILED' if fails else 'ok'}: {fails} failure(s)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
