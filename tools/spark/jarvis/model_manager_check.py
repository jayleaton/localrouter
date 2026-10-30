"""Live check of model management on the Jarvis box (stdlib only; python3 -I on DGX-01, against the running service).

  model_manager_check.py OUT SCENARIO [--ballast auto|none] [--url http://127.0.0.1:8190]

SCENARIO:
  chat    the web client's request (its system prompt, the skill from /v1/skills/model-manager, its tools, thinking off)
          asking for a 3 second video of a cat; Qwen's tool calls run as the client runs them (model tools through
          /mcp, the video through MCP generate_video). Every round is recorded.
  voice   the same request typed into a /v1/realtime session (a real WebSocket); every server event is recorded.
  self    asks Qwen to unload itself to free memory.
  mcp     an external MCP client following skills/jarvis-model-manager/SKILL.md step by step.

--ballast auto holds just enough memory (in its own bounded scope, never leaving MemAvailable under 6.5 GiB) that the
video does not fit by unloading the image model alone, so the voice models must go too; none: the box as it is.
Writes OUT/<scenario>.json: the warm set before and after, the calls, the events, the timings, pass/fail per check.
"""
import base64, http.client, json, os, socket, subprocess, sys, time, urllib.parse

OUT, SCENARIO = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
BALLAST_MODE = args[args.index("--ballast") + 1] if "--ballast" in args else "none"
URL = urllib.parse.urlsplit(args[args.index("--url") + 1] if "--url" in args else "http://127.0.0.1:8190")
AUTH = {"Authorization": "Bearer " + os.environ["LOCALROUTER_TOKEN"]} if os.environ.get("LOCALROUTER_TOKEN") else {}  # pairing.required
VIDEO, CHAT = "minimax-h3", "qwen3.8-27b"
ASK = "Make a 3 second video of a cat."
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
record = {"scenario": SCENARIO, "ballast": BALLAST_MODE, "checks": {}, "log": []}


def log(kind, **kw):
    kw.update(t=round(time.time() - T0, 2), kind=kind)
    record["log"].append(kw)
    print(json.dumps(kw)[:400], flush=True)


def request(method, path, body=None, timeout=900):
    c = http.client.HTTPConnection(URL.hostname, URL.port, timeout=timeout)
    c.request(method, path, json.dumps(body) if body is not None else None, {"Content-Type": "application/json", **AUTH})
    r = c.getresponse()
    data = r.read()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data.decode(errors="replace")


def mcp(name, arguments, timeout=1800):
    _, j = request("POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}, timeout)
    if "error" in j:
        return False, json.dumps({"error": j["error"].get("message")})
    res = j["result"]
    return not res.get("isError"), "\n".join(c["text"] for c in res.get("content", []) if c.get("type") == "text")


def models():
    _, j = request("GET", "/v1/tools")
    return {t["id"]: t["state"] for t in j["data"]}, j


def warm_set():
    m, _ = models()
    return sorted(k for k, v in m.items() if v in ("ready", "busy"))


def wait_warm(want, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        if set(want) <= set(warm_set()):
            return True
        time.sleep(2)
    return False


def mem_available_gib():
    with open("/proc/meminfo") as f:
        return next(int(l.split()[1]) for l in f if l.startswith("MemAvailable:")) / 1048576


BALLAST = r"""
import sys, threading, os, time
b = bytearray(int(sys.argv[1]) << 30)
for i in range(0, len(b), 4096):
    b[i] = 1
print("ready", flush=True)
def guard():
    floor = int(sys.argv[2]) << 20
    while True:
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable:"))
        if avail < floor:
            print("guard", flush=True)
            os._exit(0)
        time.sleep(0.2)
threading.Thread(target=guard, daemon=True).start()
sys.stdin.read()
"""


def ballast():
    """Enough held memory that the video does not fit by unloading the image model alone (the dry run says what it
    needs), bounded: MemAvailable never under 6.5 GiB; the ballast lets go at once below 6 GiB."""
    if BALLAST_MODE != "auto":
        return None
    ok, text = mcp("reserve_for", {"model": VIDEO, "seconds": 3, "dry_run": True})
    plan = json.loads(text)
    gib = 2 ** 30
    image = next((m for m in json.loads(mcp("list_models", {})[1])["models"] if m["kind"] == "image" and m["state"] == "ready"), None)
    image_bytes = image["resident_bytes"] if image else 0
    want = (plan["available_bytes"] + image_bytes - plan["needed_bytes"]) / gib + 1.0  # short by 1 GiB after the image
    size = max(1, min(int(want + 0.999), int(mem_available_gib() - 6.5)))
    p = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={size + 2}G", "timeout", "1800",
                          "python3", "-I", "-c", BALLAST, str(size), "6144"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if p.stdout.readline().strip() != "ready":
        p.kill()
        raise SystemExit("the ballast did not start")
    log("ballast", gib=size, mem_available_gib=round(mem_available_gib(), 1), plan_before=plan)
    return p


def release(p):
    if p:
        p.stdin.close()
        p.wait(timeout=60)
        log("ballast_released", mem_available_gib=round(mem_available_gib(), 1))


# ---------------------------------------------------------------- the web client's chat loop

DEFAULT_SYSTEM = "You are Jarvis, a helpful assistant running on the user's own machine. Be concise."
CHAT_TOOLS = [
    {"type": "function", "function": {"name": "generate_image", "description": "Make a picture on this machine from a detailed description. The user sees it inline.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "size": {"type": "string"}}, "required": ["prompt"]}}},
    {"type": "function", "function": {"name": "generate_video", "description": "Make a short video clip with sound on this machine. It takes minutes.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}, "seconds": {"type": "integer"}}, "required": ["prompt"]}}},
]
MODEL_TOOLS = [
    {"type": "function", "function": {"name": "list_models", "description": "This machine's models: state (ready, unloaded), needs_bytes, displaced (unloaded to make room, coming back), and memory.mem_available_bytes.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "reserve_for", "description": "Make room for a model before its job: dry_run true says what it would unload; false unloads that (never you), loads the model and holds it for its job. What it unloads comes back by itself after the job.",
     "parameters": {"type": "object", "properties": {"model": {"type": "string"}, "seconds": {"type": "integer"}, "dry_run": {"type": "boolean"}}, "required": ["model"]}}},
    {"type": "function", "function": {"name": "unload_model", "description": "Unload one model now (never you, the chat model); it comes back by itself after hold_s seconds.",
     "parameters": {"type": "object", "properties": {"model": {"type": "string"}, "hold_s": {"type": "integer"}}, "required": ["model"]}}},
    {"type": "function", "function": {"name": "load_model", "description": "Load a model now, for example one you unloaded.", "parameters": {"type": "object", "properties": {"model": {"type": "string"}}, "required": ["model"]}}},
]


def chat_loop(user_text, rounds=8):
    _, skill = request("GET", "/v1/skills/model-manager")
    record["skill_enabled"] = skill.get("enabled")
    msgs = [{"role": "system", "content": f"{DEFAULT_SYSTEM}\n\n{skill['text']}" if skill.get("enabled") else DEFAULT_SYSTEM},
            {"role": "user", "content": user_text}]
    calls_made = []
    for _ in range(rounds):
        body = {"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
                "tools": CHAT_TOOLS + (MODEL_TOOLS if skill.get("enabled") else []), "tool_choice": "auto", "max_tokens": 600}
        st, j = request("POST", "/v1/chat/completions", body)
        if st != 200:
            log("chat_error", status=st, body=str(j)[:400])
            break
        m = j["choices"][0]["message"]
        log("assistant", content=m.get("content"), tool_calls=[(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []])
        msgs.append({"role": "assistant", "content": m.get("content"), **({"tool_calls": m["tool_calls"]} if m.get("tool_calls") else {})})
        if not m.get("tool_calls"):
            break
        for c in m["tool_calls"]:
            name, a = c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")
            calls_made.append(name)
            if name == "generate_video":
                ok, text = mcp("generate_video", {"prompt": a.get("prompt", ""), "seconds": a.get("seconds", 3), "wait_s": 1200})
            elif name == "generate_image":
                ok, text = mcp("generate_image", {"prompt": a.get("prompt", ""), "inline_images": False, "wait_s": 600})
            else:
                ok, text = mcp(name, a)
            log("tool", name=name, arguments=a, ok=ok, result=text[:600], warm=warm_set())
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": text if ok else json.dumps({"error": text})})
    return calls_made, msgs


# ---------------------------------------------------------------- a voice session over a real WebSocket

class Ws:
    def __init__(self):
        self.s = socket.create_connection((URL.hostname, URL.port), timeout=900)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET /v1/realtime HTTP/1.1\r\nHost: {URL.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: realtime\r\n"
                        + "".join(f"{k}: {v}\r\n" for k, v in AUTH.items()) + "\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            head += self.s.recv(1)
        self.buf = b""

    def send(self, obj):
        data = json.dumps(obj).encode()
        mask = os.urandom(4)
        n = len(data)
        hdr = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + n.to_bytes(2, "big") if n < 65536 else bytes([0x80 | 127]) + n.to_bytes(8, "big"))
        self.s.sendall(hdr + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _take(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(65536)
            if not chunk:
                raise EOFError("the session closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def next(self):
        while True:
            h = self._take(2)
            op, n = h[0] & 0x0F, h[1] & 0x7F
            if n == 126:
                n = int.from_bytes(self._take(2), "big")
            elif n == 127:
                n = int.from_bytes(self._take(8), "big")
            data = self._take(n)
            if op == 8:
                raise EOFError("close frame")
            if op == 1:
                return json.loads(data)


def voice_session(user_text, timeout=1500):
    w = Ws()
    w.send({"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": user_text}]}})
    w.send({"type": "response.create"})
    end = time.time() + timeout
    counts, events, state = {}, [], {"paused": False, "resumed_at": None, "audio_after_resume": 0, "job_done": None}
    while time.time() < end:
        ev = w.next()
        t = ev.get("type", "")
        counts[t] = counts.get(t, 0) + 1
        if t == "response.audio.delta":
            if state["resumed_at"] is not None:
                state["audio_after_resume"] += 1
            continue
        if t in ("response.audio_transcript.delta", "response.text.delta", "conversation.item.input_audio_transcription.delta"):
            continue
        brief = {("job_kind" if k == "kind" else k): ev[k] for k in ("type", "model", "for_model", "status", "kind", "transcript", "text", "message") if k in ev}
        if t == "response.output_item.done" and ev.get("item", {}).get("type") == "function_call":
            brief["call"] = (ev["item"]["name"], ev["item"]["arguments"])
        if t == "conversation.item.created" and ev.get("item", {}).get("type") == "function_call_output":
            brief["output"] = ev["item"].get("output", "")[:300]
        if t == "error":
            brief["error"] = ev.get("error")
        log("event", **brief)
        events.append(brief)
        if t == "localrouter.voice.paused":
            state["paused"] = True
        if t == "localrouter.voice.resumed":
            state["resumed_at"] = time.time()
        if t == "localrouter.job.done":
            state["job_done"] = ev.get("status")
        if t == "response.done" and state["job_done"] and state["audio_after_resume"] > 0:
            break  # the follow-up after the video was spoken
        if t == "response.done" and state["job_done"] and not state["paused"]:
            break  # nothing paused: the follow-up is the end
    open_after = True
    try:
        w.send({"type": "session.update", "session": {}})
        w.next()
    except Exception:
        open_after = False
    w.s.close()
    return counts, events, state, open_after


# ---------------------------------------------------------------- scenarios

before = warm_set()
record["warm_before"] = before
b = None
try:
    if SCENARIO in ("chat", "voice", "mcp"):
        b = ballast()
    if SCENARIO == "chat":
        calls, msgs = chat_loop(ASK)
        record["calls"] = calls
        record["checks"]["checked_with_dry_run"] = any(e["kind"] == "tool" and e["name"] == "reserve_for" and e["arguments"].get("dry_run") for e in record["log"])
        record["checks"]["reserved_before_video"] = "reserve_for" in calls and "generate_video" in calls and calls.index("generate_video") > max(i for i, n in enumerate(calls) if n == "reserve_for")
        record["checks"]["made_the_video"] = any(e["kind"] == "tool" and e["name"] == "generate_video" and e["ok"] for e in record["log"])
        cut = next((n for n, e in enumerate(record["log"]) if e["kind"] == "tool" and e["name"] == "reserve_for" and not e["arguments"].get("dry_run")), len(record["log"]))
        record["checks"]["announced"] = any(e["kind"] == "assistant" and e.get("content") for e in record["log"][:cut])
        record["final_reply"] = next((e["content"] for e in reversed(record["log"]) if e["kind"] == "assistant"), "")
    elif SCENARIO == "voice":
        counts, events, state, open_after = voice_session(ASK)
        record.update(counts=counts, events=events, state={k: v for k, v in state.items() if k != "resumed_at"}, open_after=open_after)
        record["checks"]["voice_paused_notice"] = state["paused"]
        record["checks"]["video_done"] = state["job_done"] == "completed"
        record["checks"]["voice_resumed"] = state["resumed_at"] is not None or not state["paused"]
        record["checks"]["spoke_after_resume"] = state["audio_after_resume"] > 0 or not state["paused"]
        record["checks"]["call_survived"] = open_after
    elif SCENARIO == "self":
        calls, msgs = chat_loop("Unload yourself, the chat model, to free memory right now.", rounds=4)
        record["calls"] = calls
        record["checks"]["chat_still_loaded"] = models()[0].get(CHAT) in ("ready", "busy")
        record["final_reply"] = next((e["content"] for e in reversed(record["log"]) if e["kind"] == "assistant"), "")
        refused = [e for e in record["log"] if e["kind"] == "tool" and e["name"] == "unload_model" and e["arguments"].get("model") == CHAT]
        record["checks"]["tool_refused_or_not_called"] = all(not e["ok"] for e in refused)
    elif SCENARIO == "mcp":
        ok, plan = mcp("reserve_for", {"model": VIDEO, "seconds": 3, "dry_run": True})
        log("step", step="1 dry run", ok=ok, result=plan)
        ok, res = mcp("reserve_for", {"model": VIDEO, "seconds": 3})
        log("step", step="3 reserve", ok=ok, result=res, warm=warm_set())
        ok, vid = mcp("generate_video", {"prompt": "a cat stretching on a sunny windowsill", "seconds": 3, "wait_s": 1200})
        log("step", step="3 video", ok=ok, result=vid[:500], warm=warm_set())
        record["checks"]["reserved"] = ok
        record["checks"]["made_the_video"] = '"status":"completed"' in vid
    back = wait_warm(before, timeout=420)
    record["checks"]["warm_set_restored"] = back
finally:
    release(b)
record["warm_after"] = warm_set()
record["seconds"] = round(time.time() - T0, 1)
record["pass"] = all(record["checks"].values())
with open(os.path.join(OUT, f"{SCENARIO}.json"), "w") as f:
    json.dump(record, f, indent=1)
print(json.dumps({"scenario": SCENARIO, "pass": record["pass"], "checks": record["checks"], "before": before, "after": record["warm_after"], "seconds": record["seconds"]}))
sys.exit(0 if record["pass"] else 1)
