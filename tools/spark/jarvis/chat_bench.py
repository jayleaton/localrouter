"""Chat-model timings against an OpenAI-compatible server (stdlib only; run with python3 -I on the Spark).

  chat_bench.py URL MODEL ready  TIMEOUT_S           seconds until GET /v1/models answers (cold load, from now)
  chat_bench.py URL MODEL smoke                       one streamed and one whole /v1/chat/completions reply
  chat_bench.py URL MODEL decode RUNS MAX_TOKENS      warm decode tok/s on a chat and a code prompt
  chat_bench.py URL MODEL ttft   RUNS PROMPT_TOKENS   time to first token for a fresh (uncached) prompt of about N tokens
  chat_bench.py URL MODEL fill   PROMPT_TOKENS        one long prompt (fills the context for a memory reading)
  chat_bench.py URL MODEL turn   PROMPT_TOKENS        TTFT of a long first turn, then of the next turn on that history

Each result is one JSON line on stdout. Decode tok/s is (completion_tokens - 1) / (last chunk - first chunk), so
prefill is excluded and drafted blocks count as the tokens they are. Prompts start with a fresh nonce, so a retained
prefix cache cannot answer them.
"""
import http.client, json, sys, time, urllib.parse, uuid

URL, MODEL, CMD = sys.argv[1], sys.argv[2], sys.argv[3]
u = urllib.parse.urlsplit(URL)

PROMPTS = {
    "chat": "Explain, in plain prose for a curious teenager, how a refrigerator moves heat out of its inside, "
            "and why the back of it gets warm. Use several paragraphs.",
    "code": "Write a complete, well commented Python module that implements an LRU cache class with get, put, "
            "resize and a thread-safe wrapper, followed by unit tests using unittest.",
}
FILLER = ("The quick survey of regional rail timetables noted that the morning service departs at seven fifteen, "
          "connects with the coastal line at the junction, and returns by the evening after a short layover. ")


def conn():
    return http.client.HTTPConnection(u.hostname, u.port, timeout=3600)


def post(body, stream):
    c = conn()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json", "Host": u.netloc})
    return c, c.getresponse()


def chat(messages, max_tokens, stream=True):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    c, r = post(body, stream)
    if r.status != 200:
        raise SystemExit(f"HTTP {r.status}: {r.read()[:500]!r}")
    if not stream:
        d = json.loads(r.read())
        return {"wall_s": time.perf_counter() - t0, "usage": d.get("usage"), "text": d["choices"][0]["message"].get("content"),
                "finish": d["choices"][0].get("finish_reason")}
    first = last = None
    usage, text, chunks, finish = None, [], 0, None
    for raw in r:
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        d = json.loads(data)
        if d.get("error"):
            raise SystemExit(f"stream error: {json.dumps(d['error'])[:500]}")
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
            if piece:
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
                text.append(piece)
            finish = ch.get("finish_reason") or finish
    c.close()
    n = (usage or {}).get("completion_tokens")
    res = {"ttft_s": first - t0 if first else None, "wall_s": time.perf_counter() - t0, "chunks": chunks, "usage": usage,
           "finish": finish, "text": "".join(text)}
    if n and first and last and last > first:
        res["decode_tok_s"] = (n - 1) / (last - first)
    return res


def long_prompt(tokens):
    # about 40 tokens a FILLER sentence with Qwen's tokenizer (1137 for 27 of them); usage reports the exact count
    return f"[{uuid.uuid4()}] " + FILLER * max(1, tokens // 40) +"\nIn one sentence: when does the morning service depart?"


def out(**kw):
    print(json.dumps(kw), flush=True)


if CMD == "ready":
    limit, t0 = float(sys.argv[4]), time.perf_counter()
    while time.perf_counter() - t0 < limit:
        try:
            c = conn(); c.request("GET", "/v1/models", headers={"Host": u.netloc}); r = c.getresponse(); r.read()
            if r.status == 200:
                out(cmd="ready", seconds=round(time.perf_counter() - t0, 2)); sys.exit(0)
        except OSError:
            pass
        time.sleep(0.2)
    out(cmd="ready", seconds=None, error="timeout"); sys.exit(1)
elif CMD == "smoke":
    msg = [{"role": "user", "content": "Reply with one short sentence: what is the capital of France?"}]
    s = chat(msg, 64, stream=True)
    w = chat(msg, 64, stream=False)
    out(cmd="smoke", streamed={k: s[k] for k in ("ttft_s", "chunks", "finish", "usage", "text")},
        whole={k: w[k] for k in ("wall_s", "finish", "usage", "text")})
elif CMD == "decode":
    runs, max_tokens = int(sys.argv[4]), int(sys.argv[5])
    chat([{"role": "user", "content": "Say hi."}], 16)  # warm-up
    for name, p in PROMPTS.items():
        for i in range(runs):
            r = chat([{"role": "user", "content": f"[{uuid.uuid4()}] {p}"}], max_tokens)
            out(cmd="decode", prompt=name, run=i, decode_tok_s=r.get("decode_tok_s"), ttft_s=r["ttft_s"], usage=r["usage"],
                finish=r["finish"])
elif CMD == "ttft":
    runs, tokens = int(sys.argv[4]), int(sys.argv[5])
    chat([{"role": "user", "content": "Say hi."}], 16)
    for i in range(runs):
        r = chat([{"role": "user", "content": long_prompt(tokens)}], 16)
        out(cmd="ttft", run=i, ttft_s=r["ttft_s"], prompt_tokens=(r["usage"] or {}).get("prompt_tokens"))
elif CMD == "turn":
    # a conversation's next turn: the long first turn, then the same history plus a short question (a prefix-cache hit)
    tokens = int(sys.argv[4])
    first = [{"role": "user", "content": long_prompt(tokens)}]
    a = chat(first, 32)
    b = chat(first + [{"role": "assistant", "content": a["text"]}, {"role": "user", "content": "And when does it return?"}], 32)
    out(cmd="turn", first_ttft_s=a["ttft_s"], next_ttft_s=b["ttft_s"], next_usage=b["usage"])
elif CMD == "fill":
    tokens = int(sys.argv[4])
    r = chat([{"role": "user", "content": long_prompt(tokens)}], 64)
    out(cmd="fill", ttft_s=r["ttft_s"], wall_s=r["wall_s"], usage=r["usage"], decode_tok_s=r.get("decode_tok_s"),
        text=r["text"][:200])
else:
    raise SystemExit(__doc__)
