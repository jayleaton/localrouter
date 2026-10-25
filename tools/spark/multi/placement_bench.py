"""Single box against placement (docs/dev/MULTI-SPARK.md), measured the same way on either (stdlib only; python3 -I).

  placement_bench.py contention URL --chat MODEL --tts MODEL [--rounds N] [--voice V] [--label L]
  placement_bench.py link FRONT NODE --chat MODEL [--rounds N] [--label L]

contention: per round, in this order: the chat request alone, the speech request alone, then both together (the
speech starts, the chat request follows 0.3 s later, as a voice reply's first sentence is spoken while the rest is
written). Chat: the server's own `tensorfold.tokens_per_second` and `time_to_first_token` from the whole reply (a
fixed long-answer prompt, max_tokens 600, thinking off, greedy). Speech: streamed PCM, first audio = the first byte
of samples after the request, RTF = wall time / audio length. In a single box both run on one GPU; in placement chat
runs on the other node.

link: per round, the same short streamed chat request to FRONT (forwarded to its node) and to NODE (that node's
daemon itself), alternated; the client's time to the first content and to the end. The difference is what the hop
from the front to the node costs a turn.

One JSON line per measurement on stdout; a summary (medians) as the last line.
"""
import argparse, http.client, json, statistics, sys, threading, time, urllib.parse

STORY = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle. Keep going for several paragraphs."
SPEECH = ("The lighthouse stood at the edge of the world, and every night its keeper climbed the long stair to light the lamp. "
          "The sea below was grey and restless, and the wind carried salt and the cries of gulls. "
          "One morning, walking the shore at low tide, he found a bottle wedged between two rocks, its glass clouded by years of water.")


def conn(url, timeout=600):
    u = urllib.parse.urlsplit(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def chat_whole(url, model, prompt, max_tokens):
    body = {"model": model, "stream": False, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}]}
    c = conn(url)
    t0 = time.monotonic()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    data = r.read()
    wall = time.monotonic() - t0
    if r.status != 200:
        return {"status": r.status, "error": data[:300].decode(errors="replace")}
    v = json.loads(data)
    tf = v.get("tensorfold") or {}
    return {"status": 200, "wall_s": round(wall, 3), "completion_tokens": (v.get("usage") or {}).get("completion_tokens"),
            "tok_s": tf.get("tokens_per_second"), "ttft_s": tf.get("time_to_first_token")}


def chat_stream(url, model, prompt, max_tokens):
    body = {"model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}]}
    c = conn(url)
    t0 = time.monotonic()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"status": r.status, "error": r.read()[:300].decode(errors="replace")}
    first = None
    while True:
        line = r.readline()
        if not line:
            break
        if first is None and line.startswith(b"data: {"):
            try:
                ch = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta") or {}
            except ValueError:
                ch = {}
            if ch.get("content"):
                first = time.monotonic() - t0
        if line.strip() == b"data: [DONE]":
            break
    return {"status": 200, "first_content_s": round(first, 4) if first is not None else None, "end_s": round(time.monotonic() - t0, 4)}


def speech(url, model, voice, text):
    body = {"model": model, "input": text, "response_format": "pcm", "stream": True}
    if voice:
        body["voice"] = voice
    c = conn(url)
    t0 = time.monotonic()
    c.request("POST", "/v1/audio/speech", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"status": r.status, "error": r.read()[:300].decode(errors="replace")}
    rate = int(r.getheader("x-sample-rate") or 0)
    first, n = None, 0
    while True:
        b = r.read1(65536)
        if not b:
            break
        if first is None:
            first = time.monotonic() - t0
        n += len(b)
    wall = time.monotonic() - t0
    audio_s = n / 2 / rate if rate else None
    return {"status": 200, "first_audio_s": round(first, 3) if first is not None else None, "wall_s": round(wall, 3),
            "audio_s": round(audio_s, 3) if audio_s else None, "rtf": round(wall / audio_s, 3) if audio_s else None}


def emit(rec):
    print(json.dumps(rec), flush=True)
    return rec


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def contention(a):
    out = {"alone_chat": [], "alone_speech": [], "together_chat": [], "together_speech": []}
    chat_whole(a.url, a.chat, "Say hello.", 8)  # warm both
    speech(a.url, a.tts, a.voice, "Hello.")
    for i in range(a.rounds):
        out["alone_chat"].append(emit({"label": a.label, "round": i, "case": "chat alone", **chat_whole(a.url, a.chat, STORY, 600)}))
        out["alone_speech"].append(emit({"label": a.label, "round": i, "case": "speech alone", **speech(a.url, a.tts, a.voice, SPEECH)}))
        res = {}
        t = threading.Thread(target=lambda: res.update(speech=speech(a.url, a.tts, a.voice, SPEECH)))
        t.start()
        time.sleep(0.3)
        res["chat"] = chat_whole(a.url, a.chat, STORY, 600)
        t.join()
        out["together_chat"].append(emit({"label": a.label, "round": i, "case": "chat while speaking", **res["chat"]}))
        out["together_speech"].append(emit({"label": a.label, "round": i, "case": "speech while chat decodes", **res["speech"]}))
    emit({"label": a.label, "summary": "contention", "rounds": a.rounds,
          "chat_tok_s_alone": med([r.get("tok_s") for r in out["alone_chat"]]),
          "chat_tok_s_while_speaking": med([r.get("tok_s") for r in out["together_chat"]]),
          "chat_ttft_s_alone": med([r.get("ttft_s") for r in out["alone_chat"]]),
          "chat_ttft_s_while_speaking": med([r.get("ttft_s") for r in out["together_chat"]]),
          "speech_rtf_alone": med([r.get("rtf") for r in out["alone_speech"]]),
          "speech_rtf_while_chat": med([r.get("rtf") for r in out["together_speech"]]),
          "speech_first_audio_s_alone": med([r.get("first_audio_s") for r in out["alone_speech"]]),
          "speech_first_audio_s_while_chat": med([r.get("first_audio_s") for r in out["together_speech"]])})


def link(a):
    prompt = "Reply with the single word: ready."
    front, node = [], []
    chat_stream(a.front, a.chat, prompt, 8)
    chat_stream(a.node, a.chat, prompt, 8)
    for i in range(a.rounds):
        for name, url, acc in (("front", a.front, front), ("node", a.node, node)) if i % 2 == 0 else (("node", a.node, node), ("front", a.front, front)):
            acc.append(emit({"label": a.label, "round": i, "via": name, **chat_stream(url, a.chat, prompt, 8)}))
    fm, nm = med([r.get("first_content_s") for r in front]), med([r.get("first_content_s") for r in node])
    fe, ne = med([r.get("end_s") for r in front]), med([r.get("end_s") for r in node])
    emit({"label": a.label, "summary": "link", "rounds": a.rounds, "first_content_s_front": fm, "first_content_s_node": nm,
          "first_content_overhead_ms": round((fm - nm) * 1000, 2) if fm is not None and nm is not None else None,
          "end_s_front": fe, "end_s_node": ne, "end_overhead_ms": round((fe - ne) * 1000, 2) if fe is not None and ne is not None else None})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contention")
    c.add_argument("url")
    c.add_argument("--chat", required=True)
    c.add_argument("--tts", required=True)
    c.add_argument("--voice")
    c.add_argument("--rounds", type=int, default=5)
    c.add_argument("--label", default="")
    l = sub.add_parser("link")
    l.add_argument("front")
    l.add_argument("node")
    l.add_argument("--chat", required=True)
    l.add_argument("--rounds", type=int, default=20)
    l.add_argument("--label", default="")
    a = ap.parse_args()
    {"contention": contention, "link": link}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
