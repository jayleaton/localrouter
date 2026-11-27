"""The chat tools' qualification, shared by each model's qualify.py (tools/chat/*/): config, hashes, processes, the
HTTP / server-sent events client (127.0.0.1), and the roles every model runs the same way (direct, daemon, compare,
marker, report). A model's qualify.py adds its own preflight and pins and names its prompt set."""

import argparse
import fcntl
import hashlib
import http.client
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

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


def server_running(argv):
    """The pids of the executable the server `argv` starts that still run."""
    return processes_of(argv[0])


def wait_gone(argv, seconds):
    deadline = time.monotonic() + seconds
    while True:
        left = server_running(argv)
        if not left or time.monotonic() > deadline:
            return left
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


def run_prompts(port, tool_id, prompts_path, passes, timeout, out_dir, runs):
    os.makedirs(out_dir, exist_ok=True)
    prompts = read_json(prompts_path)["prompts"]
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


# ── the roles every model runs the same way ─────────────────────────────────────────────────────────────────────────

def stop_grace(tool):
    """How long a stop may take before SIGKILL: the tool's stop_timeout_s (default 8) and 5 s, at least STOP_GRACE_S."""
    return max(STOP_GRACE_S, float(options(tool).get("stop_timeout_s") or 8) + 5)


def direct(args, prompts):
    """Starts the tool's `options.server` argv itself (same env), waits for its health path, sends every prompt whole
    twice and streamed once, then SIGTERMs it and checks it is gone. OUT/direct.json."""
    cfg, tool = load_tool(args)
    opts = options(tool)
    port = int(opts.get("port") or 0) or free_port()
    argv = server_argv(tool, port)
    log = os.path.join(args.out, "direct-server.log")
    res = {"argv": argv, "env": opts.get("env") or {}, "port": port, "runs": {}, "error": None}
    lock = p = None
    try:
        lock = take_lock(cfg)
        busy = server_running(argv)
        if busy:
            raise Fail(f"{argv[0]} already runs ({busy})")
        res["mem_available_before"] = mem_available()
        t0 = time.monotonic()
        with open(log, "wb") as logf:
            p = subprocess.Popen(argv, env=server_env(tool), stdin=subprocess.DEVNULL, stdout=logf,
                                 stderr=subprocess.STDOUT)
        res["pid"] = p.pid
        res["health_ready"] = wait_health(p, port, opts.get("health") or "/health",
                                          int(tool.get("load_timeout_s") or 900), log)
        res["load_s"] = round(time.monotonic() - t0, 3)
        res["mem_available_loaded"] = mem_available()
        print(f"server ready in {res['load_s']} s on port {port}", flush=True)
        run_prompts(port, tool["id"], prompts, ("whole1", "whole2", "stream"),
                    int(tool.get("generate_timeout_s") or 600), os.path.join(args.out, "direct"), res["runs"])
        res["health_end"] = get_json(port, opts.get("health") or "/health")
    except SystemExit:
        res["error"] = "terminated (SIGTERM)"
        raise
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        if p is not None:
            res["stop"] = stop_process(p, stop_grace(tool))
            res["stop"]["within_stop_timeout_s"] = res["stop"]["stop_s"] <= float(opts.get("stop_timeout_s") or 8)
        res["left_running"] = wait_gone(argv, 10)
        res["server_gone"] = not res["left_running"]
        if lock is not None:
            os.close(lock)
        res["mem_available_after"] = mem_available()
        write_json(os.path.join(args.out, "direct.json"), res)
    print(json.dumps({"error": res["error"], "server_gone": res["server_gone"], "stop": res.get("stop")}))
    return 0 if res["error"] is None and res["server_gone"] and runs_ok(res["runs"]) else 1


def daemon(args, prompts):
    """The same prompts through `LOCALROUTER serve` (the tool without its gate, keep_loaded false), whole and streamed,
    and GET /v1/models; SIGTERMs the daemon, and the server must be gone. OUT/daemon.json."""
    cfg, tool = load_tool(args)
    data = os.path.join(args.out, "daemon-data")
    os.makedirs(data, exist_ok=True)
    t = dict(tool, keep_loaded=False)
    t.pop("qualification", None)
    dcfg = {"host": "127.0.0.1", "port": 0, "data_dir": data, "reserve_bytes": 8 * GIB,
            "defaults": {"chat_completion": t["id"]}, "tools": [t]}
    if cfg.get("gpu_lock"):
        dcfg["gpu_lock"] = cfg["gpu_lock"]
    cpath = os.path.join(args.out, "daemon-config.json")
    write_json(cpath, dcfg)
    argv = [args.bin, "serve", "--config", cpath, "--host", "127.0.0.1", "--port", "0", "--data", data]
    server = server_argv(tool, 0)
    res = {"argv": argv, "runs": {}, "error": None}
    p = None
    try:
        busy = server_running(server)
        if busy:
            raise Fail(f"{server[0]} already runs ({busy})")
        res["mem_available_before"] = mem_available()
        with open(os.path.join(args.out, "daemon.log"), "wb") as errf:
            p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errf)
        first = queue.Queue()

        def pump():  # the first stdout line to `first`, everything to daemon-stdout.log
            sent = False
            with open(os.path.join(args.out, "daemon-stdout.log"), "wb") as f:
                for line in iter(p.stdout.readline, b""):
                    f.write(line)
                    f.flush()
                    if not sent:
                        first.put(line)
                        sent = True
            if not sent:
                first.put(b"")

        threading.Thread(target=pump, daemon=True).start()
        try:
            line = first.get(timeout=60).decode(errors="replace").strip()
        except queue.Empty:
            raise Fail("the daemon printed nothing on stdout within 60 s")
        prefix = "localrouter listening on http://"
        if not line.startswith(prefix):
            raise Fail(f"unexpected first stdout line {line!r}; daemon.log tail:\n"
                       f"{tail(os.path.join(args.out, 'daemon.log'))}")
        port = int(line[len(prefix):].rsplit(":", 1)[1].rstrip("/"))
        res["port"] = port
        models = get_json(port, "/v1/models")
        entry = next((m for m in models.get("data") or [] if m.get("id") == t["id"]), None)
        res["models_entry"] = entry
        res["models_ok"] = bool(entry and entry.get("kind") == "chat"
                                and "chat_completion" in (entry.get("capabilities") or []))
        timeout = int(t.get("load_timeout_s") or 900) + int(t.get("generate_timeout_s") or 600) + 60
        run_prompts(port, t["id"], prompts, ("whole", "stream"), timeout, os.path.join(args.out, "daemon"),
                    res["runs"])
        res["tools_end"] = get_json(port, "/v1/tools")
        res["mem_available_loaded"] = mem_available()
    except SystemExit:
        res["error"] = "terminated (SIGTERM)"
        raise
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        if p is not None:
            res["stop"] = stop_process(p, stop_grace(tool))
        res["left_running"] = wait_gone(server, 10)
        res["server_gone"] = not res["left_running"]
        res["mem_available_after"] = mem_available()
        write_json(os.path.join(args.out, "daemon.json"), res)
    print(json.dumps({"error": res["error"], "models_ok": res.get("models_ok"), "server_gone": res["server_gone"],
                      "stop": res.get("stop")}))
    ok = res["error"] is None and res["server_gone"] and res.get("models_ok") and runs_ok(res["runs"])
    return 0 if ok else 1


def expectation(kind, want, rec):
    content, calls = rec.get("content") or "", rec.get("tool_calls") or []
    if kind == "content_or_length":
        return bool(content) or rec.get("finish_reason") == "length"
    if kind == "content":
        return bool(content)
    if kind == "content_contains":
        return any(w.lower() in content.lower() for w in want)
    if kind == "reasoning":
        return bool(rec.get("reasoning_content"))
    if kind == "no_reasoning":
        return rec.get("reasoning_content") == ""
    if kind == "tool_call":
        return any(c.get("name") == want for c in calls)
    if kind == "json_arguments":
        try:
            return bool(calls) and all(isinstance(json.loads(c.get("arguments") or ""), dict) for c in calls)
        except ValueError:
            return False
    if kind == "json_keys":
        try:
            obj = json.loads(content)
        except ValueError:
            return False
        return isinstance(obj, dict) and all(k in obj for k in want)
    if kind == "finish_reason":
        return rec.get("finish_reason") in want
    if kind == "max_completion_tokens":
        return isinstance(rec.get("completion_tokens"), int) and rec["completion_tokens"] <= want
    if kind == "stop_text_absent":
        return want not in content
    raise Fail(f"unknown expectation {kind}")


def compare(args, prompts, exceptions=None, relax=None):
    """Per prompt: direct run 1 = run 2, daemon = direct, stream = whole (direct and daemon) on every field of FIELDS,
    each stream ends with [DONE], and the prompt's structural expectations. `relax(field, label, a, b)` names the
    exception (a key of `exceptions`) that admits a difference, or None. OUT/compare.json; 1 on any failed check."""
    exceptions = exceptions or {}
    checks = []
    used = set()

    def add(name, ok, **info):
        checks.append({"check": name, "pass": bool(ok), **info})

    def load(name):
        try:
            return read_json(os.path.join(args.out, name))
        except (OSError, ValueError) as e:
            return {"error": f"{name}: {e}", "runs": {}}

    d, m = load("direct.json"), load("daemon.json")
    add("direct.ran", d.get("error") is None, error=d.get("error"))
    add("direct.server_gone", d.get("server_gone") is True, left_running=d.get("left_running"))
    add("daemon.ran", m.get("error") is None, error=m.get("error"))
    add("daemon.models_lists_chat_tool", m.get("models_ok") is True, entry=m.get("models_entry"))
    add("daemon.server_gone", m.get("server_gone") is True, left_running=m.get("left_running"))
    for p in read_json(prompts)["prompts"]:
        pid = p["id"]
        recs = {f"direct.{k}": v for k, v in (d.get("runs") or {}).get(pid, {}).items()}
        recs.update({f"daemon.{k}": v for k, v in (m.get("runs") or {}).get(pid, {}).items()})
        ok = {}
        for label in ("direct.whole1", "direct.whole2", "direct.stream", "daemon.whole", "daemon.stream"):
            rec = recs.get(label) or {}
            ct = rec.get("completion_tokens")
            ok[label] = rec.get("status") == 200 and not rec.get("error") and isinstance(ct, int) and ct > 0
            add(f"{pid}.{label}.ok", ok[label], status=rec.get("status"), error=rec.get("error"))
            if label.endswith("stream"):
                add(f"{pid}.{label}.event_stream", (rec.get("content_type") or "").startswith("text/event-stream"),
                    content_type=rec.get("content_type"))
                add(f"{pid}.{label}.done", rec.get("done") is True and not rec.get("after_done"),
                    done=rec.get("done"), after_done=rec.get("after_done"))
            else:
                add(f"{pid}.{label}.json", (rec.get("content_type") or "").startswith("application/json"),
                    content_type=rec.get("content_type"))
        for a, b in (("direct.whole2", "direct.whole1"), ("daemon.whole", "direct.whole1"),
                     ("direct.stream", "direct.whole1"), ("daemon.stream", "daemon.whole")):
            ra, rb = recs.get(a) or {}, recs.get(b) or {}
            for f in FIELDS:
                same = ok[a] and ok[b] and ra.get(f) == rb.get(f)
                name = None if same or not (ok[a] and ok[b]) or relax is None else relax(f, a, ra.get(f), rb.get(f))
                if name:
                    used.add(name)
                    add(f"{pid}.{a}=={b}.{f}", True, exception=name, **{a: ra.get(f), b: rb.get(f)})
                    continue
                info = {} if same else {a: ra.get(f), b: rb.get(f)}
                add(f"{pid}.{a}=={b}.{f}", same, **info)
        base = recs.get("direct.whole1") or {}
        for kind, want in p.get("expect", {}).items():
            add(f"{pid}.expect.{kind}", ok["direct.whole1"] and expectation(kind, want, base), want=want)
    failed = [c["check"] for c in checks if not c["pass"]]
    names = sorted(used)
    result = {"pass": not failed, "checks": len(checks), "failed": failed, "exceptions": names,
              "exception_terms": {n: exceptions[n] for n in names}, "list": checks}
    write_json(os.path.join(args.out, "compare.json"), result)
    print(json.dumps({"pass": result["pass"], "checks": len(checks), "failed": failed, "exceptions": names},
                     indent=1))
    return 0 if result["pass"] else 1


def marker(args, record):
    """Writes the marker atomically when OUT's pins and compare passed on a GB10: `record(compare, pins)` gives its
    fields after "qualified": true (it adds "report" and "qualified_at"). --revoke removes it if OUT wrote it."""
    path = args.marker
    try:
        existing = read_json(path)
    except (OSError, ValueError):
        existing = None
    if args.revoke:
        if isinstance(existing, dict) and existing.get("report") == args.out:
            os.remove(path)
            print(f"marker: removed {path}, written by this run")
        return 0
    try:
        comp, pins_r = read_json(os.path.join(args.out, "compare.json")), read_json(os.path.join(args.out, "pins.json"))
    except (OSError, ValueError) as e:
        raise Fail(f"no marker: {e}")
    if comp.get("pass") is not True or pins_r.get("pass") is not True:
        raise Fail(f"no marker: compare pass {comp.get('pass')}, pins pass {pins_r.get('pass')}")
    if os.uname().machine != "aarch64":
        raise Fail(f"no marker: this is {os.uname().machine}, not a GB10")
    gpu = gpu_name()
    if not gpu or "GB10" not in gpu:
        raise Fail(f"no marker: nvidia-smi names {gpu!r}, not a GB10")
    rec = {"qualified": True, **record(comp, pins_r, gpu), "report": args.out,
           "qualified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    text = json.dumps(rec, indent=1) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)
    with open(os.path.join(args.out, "marker.json"), "w") as f:
        f.write(text)
    print(f"marker: wrote {path}\n{text}", end="")
    return 0


def report(args):
    """OUT/report.json: step status, the memory peak from memory.log and phases.log, timings."""
    def load(name, default):
        try:
            return read_json(os.path.join(args.out, name))
        except (OSError, ValueError):
            return default

    samples, phases = [], {}
    for name, sink in (("memory.log", samples), ("phases.log", None)):
        try:
            with open(os.path.join(args.out, name)) as f:
                lines = f.read().splitlines()
        except OSError:
            lines = []
        for line in lines:
            parts = line.split()
            try:
                if sink is not None and len(parts) == 2:
                    sink.append((float(parts[0]), int(parts[1])))
                elif sink is None and len(parts) == 4:
                    phases.setdefault(parts[0], {})[parts[1]] = (float(parts[2]), int(parts[3]))
            except ValueError:
                pass
    mem = {}
    for name, ph in phases.items():
        if "start" not in ph or "end" not in ph:
            continue
        (t0, base), (t1, end) = ph["start"], ph["end"]
        inside = [v for t, v in samples if t0 <= t <= t1]
        low = min(inside + [end])
        mem[name] = {"baseline_mem_available": base, "min_mem_available": low, "end_mem_available": end,
                     "peak_drop_bytes": base - low, "peak_drop_gib": round((base - low) / GIB, 2),
                     "samples": len(inside), "seconds": round(t1 - t0, 1)}
    peak = max((v["peak_drop_bytes"] for v in mem.values()), default=None)
    d, m, comp = load("direct.json", {}), load("daemon.json", {}), load("compare.json", {})
    try:
        _, tool = load_tool(args)
    except (Fail, OSError, ValueError):
        tool = {}
    timings = {}
    for side, res in (("direct", d), ("daemon", m)):
        for pid, by_label in (res.get("runs") or {}).items():
            for label, rec in by_label.items():
                timings.setdefault(pid, {})[f"{side}.{label}"] = {
                    k: rec.get(k) for k in ("wall_s", "ttft_s", "completion_tokens", "prompt_tokens", "cached_tokens",
                                            "tokens_per_second")}
    rep = {"status": load("status.json", {}),
           "memory": {"phases": mem, "peak_drop_bytes": peak,
                      "peak_drop_gib": None if peak is None else round(peak / GIB, 2),
                      "configured_resident_bytes": tool.get("resident_bytes"),
                      "server_health_memory": (d.get("health_end") or {}).get("memory"),
                      "note": "peak_drop is the largest fall in MemAvailable (0.5 s samples) from the start of the "
                              "direct or daemon step: the measured candidate for resident_bytes (unified memory: "
                              "weights, drafter, KV, prompt cache, CUDA context and the daemon)"},
           "direct": {k: d.get(k) for k in ("load_s", "stop", "server_gone", "error", "health_ready")},
           "daemon": {k: m.get(k) for k in ("stop", "server_gone", "error", "models_entry")},
           "compare": {k: comp.get(k) for k in ("pass", "checks", "failed")},
           "timings": timings}
    write_json(os.path.join(args.out, "report.json"), rep)
    print(json.dumps(rep["memory"], indent=1))
    return 0


def main(model, roles):
    """The command line every qualify.py shares: ROLE --root --config --out [--tool] [--bin] [--marker] [--revoke]."""
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=sorted(roles))
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tool", default=model)
    ap.add_argument("--bin", help="the localrouter binary (preflight, daemon)")
    ap.add_argument("--marker", help="the qualification marker path (preflight, marker)")
    ap.add_argument("--revoke", action="store_true", help="marker: remove the marker if this OUT wrote it")
    args = ap.parse_args()
    args.root, args.out = os.path.abspath(args.root), os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    signal.signal(signal.SIGTERM, on_term)
    try:
        return roles[args.role](args) or 0
    except Fail as e:
        print(f"qualify.py {args.role}: {e}", file=sys.stderr)
        return 1
