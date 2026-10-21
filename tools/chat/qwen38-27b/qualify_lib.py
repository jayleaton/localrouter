"""Helpers for qualify.py: config and pins, hashes, processes, and the HTTP / server-sent events client (127.0.0.1)."""

import fcntl
import hashlib
import http.client
import json
import os
import signal
import socket
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = os.path.join(HERE, "pins.json")
PROMPTS = os.path.join(HERE, "prompts.json")
MODEL = "qwen3.8-27b"
FIELDS = ("content", "reasoning_content", "tool_calls", "finish_reason", "completion_tokens")
GIB = 1 << 30
STOP_GRACE_S = 15  # + the 10 s wait_gone stays under qualify.sh's 30 s kill-after: a timed-out step still cleans up


class Fail(Exception):
    pass


def on_term(signum, frame):
    raise SystemExit(128 + signum)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git_sha1_file(path, size):
    h = hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def gpu_name():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = r.stdout.strip().splitlines()
    return names[0].strip() if r.returncode == 0 and names else None


def platform_name():
    return {"aarch64": "linux-aarch64", "x86_64": "linux-x86_64"}.get(os.uname().machine)


def load_tool(args):
    cfg = read_json(args.config)
    for t in cfg.get("tools") or []:
        if t.get("id") == args.tool:
            return cfg, t
    raise Fail(f"{args.config} has no tool {args.tool!r}")


def options(tool):
    return tool.get("options") or {}


def server_argv(tool, port):
    argv = options(tool).get("server")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise Fail("options.server must be a non-empty array of strings")
    subs = {"{port}": str(port), "{id}": tool["id"], "{weights}": tool.get("weights", "")}
    out = []
    for a in argv:
        for k, v in subs.items():
            a = a.replace(k, v)
        out.append(a)
    return out


def server_env(tool):
    env = dict(os.environ)
    for k, v in (options(tool).get("env") or {}).items():
        if not isinstance(v, str):
            raise Fail(f"options.env.{k} must be a string")
        env[k] = v
    return env


def hf_cache(tool):
    return (options(tool).get("env") or {}).get("HF_HUB_CACHE") or tool.get("weights", "")


def processes_of(binary):
    """Pids whose executable or argv[0] is `binary` (other users' processes by argv[0] only)."""
    real, pids = os.path.realpath(binary), []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                argv0 = f.read().split(b"\0")[0].decode(errors="replace")
        except OSError:
            continue
        try:
            exe = os.readlink(f"/proc/{d}/exe").removesuffix(" (deleted)")
        except OSError:
            exe = ""
        if exe == real or (argv0 and os.path.realpath(argv0) == real):
            pids.append(int(d))
    return pids


def wait_gone(binary, seconds):
    deadline = time.monotonic() + seconds
    while True:
        pids = processes_of(binary)
        if not pids or time.monotonic() > deadline:
            return pids
        time.sleep(0.5)


def take_lock(cfg):
    """The host's GPU lock, as the engine takes it: must exist, exclusive, non-blocking. None when unset."""
    path = cfg.get("gpu_lock") or ""
    if not path:
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise Fail(f"the GPU lock {path} is held by another job")
    return fd


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tail(path, n=4000):
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - n))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def stop_process(p, grace):
    """SIGTERM, then SIGKILL after `grace` seconds; the exit status and how long it took."""
    t = time.monotonic()
    killed = False
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(grace)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            killed = True
    return {"exit": p.returncode, "stop_s": round(time.monotonic() - t, 3), "killed": killed}


# ── HTTP to 127.0.0.1 ────────────────────────────────────────────────────────────────────────────────────────────────

def http_get(port, path, timeout=10):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def get_json(port, path):
    try:
        status, body = http_get(port, path)
        return json.loads(body) if status == 200 else {"status": status, "body": body[:1024].decode(errors="replace")}
    except (OSError, http.client.HTTPException, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"}


def usage_fields(u):
    u = u or {}
    return {"completion_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}


def parse_whole(status, data, rec):
    if status != 200:
        rec["error"] = f"HTTP {status}: {data[:1024].decode(errors='replace')}"
        return
    obj = json.loads(data)
    choices = obj.get("choices") or []
    if len(choices) != 1:
        rec["error"] = f"{len(choices)} choices"
        return
    msg = choices[0].get("message") or {}
    rec["content"] = msg.get("content") or ""
    rec["reasoning_content"] = msg.get("reasoning_content") or ""
    rec["tool_calls"] = [{"name": (tc.get("function") or {}).get("name"),
                          "arguments": (tc.get("function") or {}).get("arguments")}
                         for tc in msg.get("tool_calls") or []]
    rec["finish_reason"] = choices[0].get("finish_reason")
    rec.update(usage_fields(obj.get("usage")))
    tf = obj.get("tensorfold") or {}
    rec["token_sha"] = tf.get("token_sha")
    rec["tokens_per_second"] = tf.get("tokens_per_second")


class Stream:
    """Server-sent events, line by line: the deltas joined, the finish reason, usage and [DONE]."""

    def __init__(self, t0):
        self.t0, self.content, self.reasoning, self.calls = t0, [], [], {}
        self.finish = self.usage = self.ttft = None
        self.done, self.events, self.after_done, self.errors = False, 0, 0, []

    def feed(self, data):
        self.events += 1
        if self.done:
            self.after_done += 1
            return
        if data == b"[DONE]":
            self.done = True
            return
        obj = json.loads(data)
        if "error" in obj:
            self.errors.append(obj["error"])
            return
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            if (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")) and self.ttft is None:
                self.ttft = round(time.monotonic() - self.t0, 3)
            self.content.append(d.get("content") or "")
            self.reasoning.append(d.get("reasoning_content") or "")
            for tc in d.get("tool_calls") or []:
                slot = self.calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                if name and name != slot["name"]:  # a repeated full name is not appended twice
                    slot["name"] += name
                slot["arguments"] += fn.get("arguments") or ""
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]
        if obj.get("usage"):
            self.usage = obj["usage"]

    def record(self, rec):
        rec.update({"content": "".join(self.content), "reasoning_content": "".join(self.reasoning),
                    "tool_calls": [self.calls[i] for i in sorted(self.calls)], "finish_reason": self.finish,
                    "done": self.done, "events": self.events, "after_done": self.after_done, "ttft_s": self.ttft})
        rec.update(usage_fields(self.usage))
        if self.errors:
            rec["error"] = f"error events: {self.errors}"
        elif not self.done:
            rec["error"] = "the stream ended without data: [DONE]"
        elif self.after_done:
            rec["error"] = f"{self.after_done} events after [DONE]"


def read_stream(r, sock, rec, raw_path, t0):
    s, lines = Stream(t0), []
    with open(raw_path, "wb") as raw:
        while True:
            try:
                line = r.readline()
            except socket.timeout:
                if s.done:  # the server kept the connection open after [DONE]
                    rec["tail_timeout"] = True
                    break
                raise
            if not line:
                break
            raw.write(line)
            text = line.rstrip(b"\r\n")
            if text == b"":
                if lines:
                    s.feed(b"\n".join(lines))
                    lines = []
                    if s.done:
                        sock.settimeout(15)
            elif text.startswith(b"data:"):
                v = text[5:]
                lines.append(v[1:] if v.startswith(b" ") else v)
        if lines:
            s.feed(b"\n".join(lines))
    s.record(rec)


def chat(port, body, timeout, raw_path):
    """One POST /v1/chat/completions, whole or streamed; the reply's comparable fields."""
    t0 = time.monotonic()
    rec = {"status": None, "error": None}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.connect()
        sock = c.sock
        c.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        rec["status"] = r.status
        rec["content_type"] = r.getheader("Content-Type", "")
        if body.get("stream") and r.status == 200:
            read_stream(r, sock, rec, raw_path, t0)
        else:
            data = r.read()
            with open(raw_path, "wb") as f:
                f.write(data)
            parse_whole(r.status, data, rec)
    except (OSError, http.client.HTTPException, ValueError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        c.close()
    rec["wall_s"] = round(time.monotonic() - t0, 3)
    return rec


def run_prompts(port, tool_id, passes, timeout, out_dir, runs):
    os.makedirs(out_dir, exist_ok=True)
    prompts = read_json(PROMPTS)["prompts"]
    for label in passes:  # pass-major: each prompt's repeat comes after the others ran
        for p in prompts:
            body = dict(p["body"], model=tool_id)
            stream = label.startswith("stream")
            if stream:
                body["stream"] = True
            raw = os.path.join(out_dir, f"{p['id']}-{label}.{'sse' if stream else 'json'}")
            rec = chat(port, body, timeout, raw)
            runs.setdefault(p["id"], {})[label] = rec
            print(f"{p['id']} {label}: status {rec['status']} finish {rec.get('finish_reason')} tokens "
                  f"{rec.get('completion_tokens')} {rec['wall_s']} s{' ERROR ' + rec['error'] if rec['error'] else ''}",
                  flush=True)


def runs_ok(runs):
    return all(not rec.get("error") for by_label in runs.values() for rec in by_label.values())


def wait_health(p, port, path, seconds, log):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if p.poll() is not None:
            raise Fail(f"the server exited with status {p.returncode} before {path} answered 200; "
                       f"log tail:\n{tail(log)}")
        try:
            status, body = http_get(port, path, timeout=5)
            if status == 200:
                try:
                    return json.loads(body)
                except ValueError:
                    return {"body": body[:1024].decode(errors="replace")}
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.5)
    raise Fail(f"{path} did not answer 200 within {seconds} s; log tail:\n{tail(log)}")
