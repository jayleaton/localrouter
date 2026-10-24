"""Calls every LocalRouter MCP tool against a live daemon as an MCP client would, and keeps what comes back.

  python3 -I tools/jarvis/mcp_check.py OUT [URL]       URL default http://127.0.0.1:8190 (its /mcp)

Writes OUT/models.json, voices.json, chat.txt, image.png, edit.png (edit_image of image.png, "make it night", when a
model lists image_edit; else a SKIP line), video.mp4, speech-<voice>.wav (two voices), transcript.json (of the first
WAV, fetched by its URL) and calls.jsonl (every request and reply, base64 trimmed).
Exits 1 if any call fails. Stdlib only.
With pairing on (pairing.required), $LOCALROUTER_TOKEN is sent as a Bearer token (deploy/jarvis/pair.sh token NAME).
"""
import base64, http.client, json, os, sys, time, urllib.parse

OUT = sys.argv[1]
URL = urllib.parse.urlsplit(sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8190")
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT, "calls.jsonl"), "w")
failed = []
AUTH = {"Authorization": "Bearer " + os.environ["LOCALROUTER_TOKEN"]} if os.environ.get("LOCALROUTER_TOKEN") else {}
seq = 0


def rpc(method, params=None, timeout=1200):
    global seq
    seq += 1
    body = {"jsonrpc": "2.0", "id": seq, "method": method, "params": params or {}}
    c = http.client.HTTPConnection(URL.hostname, URL.port, timeout=timeout)
    t0 = time.time()
    c.request("POST", "/mcp", json.dumps(body), {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **AUTH})
    r = c.getresponse()
    reply = json.loads(r.read())
    trimmed = json.loads(json.dumps(reply))
    for item in ((trimmed.get("result") or {}).get("content") or []):
        if item.get("data"):
            item["data"] = f"<{len(item['data'])} base64 chars>"
    log.write(json.dumps({"request": body, "seconds": round(time.time() - t0, 2), "reply": trimmed}) + "\n")
    log.flush()
    return reply


def call(name, args, timeout=1200):
    reply = rpc("tools/call", {"name": name, "arguments": args}, timeout)
    res = reply.get("result") or {}
    if "error" in reply or res.get("isError"):
        failed.append(name)
        print(f"FAIL {name}: {json.dumps(reply)[:400]}", flush=True)
    else:
        print(f"ok   {name}", flush=True)
    return res.get("content") or []


def texts(content):
    return [c["text"] for c in content if c.get("type") == "text"]


def fetch(url, path):
    u = urllib.parse.urlsplit(url)
    c = http.client.HTTPConnection(URL.hostname, URL.port, timeout=600)  # the daemon's own file URLs
    c.request("GET", u.path)
    data = c.getresponse().read()
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


init = rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "mcp_check", "version": "1"}})
print("server", json.dumps(init["result"]["serverInfo"]))
tools = [t["name"] for t in rpc("tools/list")["result"]["tools"]]
print("tools", tools)
with open(os.path.join(OUT, "tools.json"), "w") as f:
    json.dump(tools, f)

models = texts(call("list_models", {}))
with open(os.path.join(OUT, "models.json"), "w") as f:
    f.write(models[0] if models else "")
voices_t = texts(call("list_voices", {}))
with open(os.path.join(OUT, "voices.json"), "w") as f:
    f.write(voices_t[0] if voices_t else "")
default = next((m for m in json.loads(voices_t[0])["speech_models"] if m["default"]), None) if voices_t else None
voices = (default or {}).get("voices") or []

chat = texts(call("chat", {"prompt": "In two sentences: what can a local AI box with chat, voice, images and video do for me?", "reasoning": False, "max_tokens": 200}))
with open(os.path.join(OUT, "chat.txt"), "w") as f:
    f.write(chat[0] if chat else "")

img = call("generate_image", {"prompt": "a red fox sitting in fresh snow, soft morning light", "size": "1024x1024", "seed": 7, "wait_s": 900})
for c in img:
    if c.get("type") == "image":
        with open(os.path.join(OUT, "image.png"), "wb") as f:
            f.write(base64.b64decode(c["data"]))
        break

# edit_image of that picture, sent as base64 (the daemon fetches a URL without a token, which pairing refuses), at the
# input's shape (no size)
editors = [m["id"] for m in (json.loads(models[0])["models"] if models else []) if "image_edit" in m.get("capabilities", [])]
png = open(os.path.join(OUT, "image.png"), "rb").read() if os.path.exists(os.path.join(OUT, "image.png")) else b""
if not editors:
    print("SKIP edit_image: no model lists image_edit", flush=True)
elif not png:
    failed.append("edit_image-input")
else:
    t0 = time.time()
    ed = call("edit_image", {"prompt": "make it night", "image": base64.b64encode(png).decode(), "wait_s": 900})
    for c in ed:
        if c.get("type") == "image":
            with open(os.path.join(OUT, "edit.png"), "wb") as f:
                f.write(base64.b64decode(c["data"]))
            break
    else:
        failed.append("edit_image-file")
    print(f"edit_image: {round(time.time() - t0, 1)} s", flush=True)

vid = texts(call("generate_video", {"prompt": "gentle waves at dusk, gulls calling", "seconds": 2, "wait_s": 30}))
job = json.loads(vid[-1]) if vid else {}
if job.get("status") not in ("completed", "failed"):
    vid = texts(call("get_job", {"id": job.get("id", ""), "wait_s": 1500}, timeout=1600))
    job = json.loads(vid[-1]) if vid else {}
if job.get("files"):
    fetch(job["files"][0], os.path.join(OUT, "video.mp4"))
else:
    failed.append("video-file")

wavs = []
for v in voices[:2]:
    sp = call("synthesize_speech", {"input": f"Hello, I am {v}. This is LocalRouter speaking over MCP.", "voice": v})
    for c in sp:
        if c.get("type") == "audio":
            p = os.path.join(OUT, f"speech-{v}.wav")
            with open(p, "wb") as f:
                f.write(base64.b64decode(c["data"]))
    links = [c["uri"] for c in sp if c.get("type") == "resource_link"]
    wavs += links
if len(voices) < 2:
    failed.append("two-voices")

if wavs:
    tr = texts(call("transcribe_audio", {"audio": wavs[0], "wait_s": 300}))
    with open(os.path.join(OUT, "transcript.json"), "w") as f:
        f.write(tr[1] if len(tr) > 1 else (tr[0] if tr else ""))

bad = rpc("tools/call", {"name": "generate_video", "arguments": {"prompt": "x", "size": "4000x4000", "wait_s": 5}})
print("error example:", ((bad.get("result") or {}).get("content") or [{}])[0].get("text"))

log.close()
print("failed:", failed or "none")
sys.exit(1 if failed else 0)
