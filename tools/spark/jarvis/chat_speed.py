"""Chat speed as the Jarvis client sees it (stdlib only; python3 -I on the Spark).

  chat_speed.py URL MODEL STATE [CASE ...]

Sends the web client's request shape (src/voice/voice.html chatBody: its default system prompt, the image and video
tools, thinking off, no temperature) and variants of it, each twice whole and once streamed. One JSON line per run:
the server's own figures from the whole reply (`tensorfold.time_to_first_token`, `tokens_per_second` and
`speculative`: rounds, drafted, accepted, tokens per round) and, for the stream, what the client saw (time to the
first content and decode tok/s = (completion_tokens - 1) / (last chunk - first chunk)). Every user turn starts with a
fresh nonce, so only the shared prefix (system prompt and tools) can come from the prompt cache.
"""
import http.client, json, os, sys, time, urllib.parse, uuid

URL, MODEL, STATE = sys.argv[1], sys.argv[2], sys.argv[3]
u = urllib.parse.urlsplit(URL)

SYSTEM = "You are Jarvis, a helpful assistant running on the user's own machine. Be concise."
TOOLS = [
    {"type": "function", "function": {"name": "generate_image", "description": "Make a picture on this machine from a detailed description. The user sees it inline.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "what to draw, in detail"},
      "size": {"type": "string", "description": "WIDTHxHEIGHT in multiples of 16, e.g. 1024x1024 (default) or 1344x768"}}, "required": ["prompt"]}}},
    {"type": "function", "function": {"name": "generate_video", "description": "Make a short video clip with sound on this machine. It takes minutes.",
     "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "what happens, in detail"},
      "seconds": {"type": "integer", "description": "length, 1 to 20 (default 5)"}}, "required": ["prompt"]}}},
]
PROSE = "How does a GPU speed up neural networks?"
CODE = "Write a Python function that merges two sorted lists."


def body(case):
    """The client's body for `case`; max_tokens 400 bounds every run (the client sends none)."""
    b = {"model": MODEL, "max_tokens": 400,
         "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": CODE if "code" in case else PROSE}],
         "chat_template_kwargs": {"enable_thinking": "think" in case}}
    if "think" not in case:
        b["reasoning_effort"] = "none"
    if "notools" not in case:
        b["tools"], b["tool_choice"] = TOOLS, "auto"
    if "greedy" in case:
        b["temperature"] = 0
    return b


CASES = ["client", "client-greedy", "client-notools", "code", "code-greedy", "client-think"]


def post(b):
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    c.request("POST", "/v1/chat/completions", json.dumps(b), {"Content-Type": "application/json", "Host": u.netloc})
    return c, c.getresponse()


def nonce(b):
    b = json.loads(json.dumps(b))
    tag = os.environ.get("CHAT_SPEED_NONCE") or uuid.uuid4().hex[:8]  # fixed: identical requests (A/B of two paths)
    b["messages"][-1]["content"] = f"[{tag}] " + b["messages"][-1]["content"]
    return b


def whole(case):
    b = nonce(body(case))
    t0 = time.perf_counter()
    c, r = post(b)
    d = json.loads(r.read())
    if r.status != 200:
        return {"error": r.status, "body": d}
    tf, sp, us = d.get("tensorfold") or {}, d.get("speculative") or {}, d.get("usage") or {}
    return {"wall_s": round(time.perf_counter() - t0, 3), "tokens": us.get("completion_tokens"),
            "prompt": us.get("prompt_tokens"), "cached": (us.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft_s": round(tf.get("time_to_first_token") or 0, 3), "tok_s": round(tf.get("tokens_per_second") or 0, 1),
            "sampling": tf.get("sampling"), "rounds": sp.get("rounds"), "drafted": sp.get("drafted"),
            "accepted": sp.get("accepted"), "per_round": round(sp.get("tokens_per_round") or 0, 2),
            "finish": d["choices"][0].get("finish_reason")}


def stream(case):
    b = nonce(body(case))
    b["stream"], b["stream_options"] = True, {"include_usage": True}
    t0 = time.perf_counter()
    c, r = post(b)
    if r.status != 200:
        return {"error": r.status, "body": r.read()[:300].decode(errors="replace")}
    first = last = None
    tokens = None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            tokens = ev["usage"].get("completion_tokens")
        for ch in ev.get("choices") or []:
            dl = ch.get("delta") or {}
            if dl.get("content") or dl.get("reasoning_content") or dl.get("tool_calls"):
                now = time.perf_counter()
                first = first or now
                last = now
    span = (last - first) if first and last else 0
    return {"wall_s": round(time.perf_counter() - t0, 3), "tokens": tokens,
            "client_ttft_s": round(first - t0, 3) if first else None,
            "client_tok_s": round((tokens - 1) / span, 1) if tokens and span > 0 else None}


for case in sys.argv[4:] or CASES:
    for i, fn in enumerate((whole, whole, stream)):
        rec = {"state": STATE, "url": URL, "case": case, "run": i, "mode": fn.__name__}
        rec.update(fn(case))
        print(json.dumps(rec), flush=True)
