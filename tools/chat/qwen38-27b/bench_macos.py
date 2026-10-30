#!/usr/bin/env python3
"""Qwen3.8-27B on macOS (TensorFold Metal): measures and qualifies the chat tool on this Mac. Standard library only.

    python3 -I bench_macos.py serve --config CONFIG --out OUT [--tool ID] [--no-drafter]
    python3 -I bench_macos.py port --port PORT --model ID --out OUT [--server-pid PID]
    python3 -I bench_macos.py daemon --bin LOCALROUTER --config CONFIG --out OUT [--tool ID]

serve  starts the tool's `options.server` argv itself (same env; --no-drafter drops `--drafter REPO`), times the load
       to its health path, runs the suite, then SIGTERMs it and checks it is gone. Wrap it in the host's GPU lock.
port   runs the suite against an OpenAI-compatible endpoint already on 127.0.0.1:PORT (the LocalRouter daemon).
       Memory is read from PID (default: the one running tensorfold-native, if any).
daemon starts `LOCALROUTER serve` with the tool alone (port 0, data under OUT), checks /v1/models, loads the tool with
       MCP `load_model` (timed), asks MCP `chat`, runs the suite through /v1/chat/completions, then SIGTERMs the
       daemon: the model server must be gone. Wrap it in the host's GPU lock.

The suite: prompts.json whole twice and streamed once with each prompt's expectations, whole = whole (determinism) and
stream = whole (TensorFold's reasoning trailing-newline exception admitted, as qualify.py), and whole = the Mac
reference baseline (baseline-macos.json, same TensorFold version); speed at a short code prompt and document prompts
sized ~2k / ~8k / ~28k tokens at 4 characters a token (the server's prompt_tokens is the real count, 10 to 25% more),
streamed, reasoning off, each cold (a unique prefix); cached follow-ups (a ~1.5k and an ~8k turn, then the same
conversation plus one question) against a cold prompt of the same length. Memory: the server's phys_footprint every
0.5 s, its lifetime maximum (proc_pid_rusage) and `footprint` snapshots after load and at the end. OUT/bench.json has
everything; OUT/summary.json the numbers.
"""

import argparse
import ctypes
import glob
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script's directory off sys.path
from qualify_lib import (FIELDS, PROMPTS, STOP_GRACE_S, Fail, chat, free_port, get_json, load_tool,  # noqa: E402
                         on_term, options, read_json, run_prompts, server_argv, server_env, stop_process, tail,
                         wait_health, write_json)
from qualify import expectation, stream_reasoning_tail  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
BASELINE = os.path.join(HERE, "baseline-macos.json")
TENSORFOLD = read_json(os.path.join(HERE, "pins-macos.json"))["tensorfold"]["version"]
GIB = 1 << 30
CHARS_PER_TOKEN = 4  # English prose with this tokenizer; the server's prompt_tokens gives the real count

# ── memory (macOS) ───────────────────────────────────────────────────────────────────────────────────────────────────

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
RUSAGE_INFO_V4 = 4
RI_PHYS_FOOTPRINT, RI_LIFETIME_MAX_PHYS_FOOTPRINT = 7, 28  # uint64 slots after ri_uuid[16] in rusage_info_v4


def footprint(pid):
    """(phys_footprint, lifetime max phys_footprint) in bytes; Metal buffers count, as in Activity Monitor."""
    raw = (ctypes.c_uint8 * (16 + 8 * 64))()
    if _libproc.proc_pid_rusage(int(pid), RUSAGE_INFO_V4, ctypes.byref(raw)) != 0:
        return None, None
    vals = (ctypes.c_uint64 * 64).from_buffer(raw, 16)
    return int(vals[RI_PHYS_FOOTPRINT]), int(vals[RI_LIFETIME_MAX_PHYS_FOOTPRINT])


def system_memory():
    """Total, and free + inactive + speculative + purgeable bytes (what the host could hand out without paging)."""
    total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    pages = {}
    for line in out.splitlines()[1:]:
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        if v.isdigit():
            pages[k.strip()] = int(v)
    free = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return {"total": total, "available": free * page}


def footprint_report(pid, path):
    r = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=120)
    with open(path, "w") as f:
        f.write(r.stdout + r.stderr)
    for line in r.stdout.splitlines():
        if "phys_footprint:" in line and "peak" not in line:
            return line.strip()
    return None


class Sampler(threading.Thread):
    def __init__(self, pid, path):
        super().__init__(daemon=True)
        self.pid, self.path, self.stop, self.peak = pid, path, threading.Event(), 0

    def run(self):
        with open(self.path, "w") as f:
            while not self.stop.is_set():
                cur, _ = footprint(self.pid)
                avail = system_memory()["available"]
                if cur is not None:
                    self.peak = max(self.peak, cur)
                f.write(f"{time.time():.3f} {cur} {avail}\n")
                f.flush()
                self.stop.wait(0.5)


def server_pids():
    r = subprocess.run(["pgrep", "-f", "tensorfold-native serve"], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]


# ── the suite ────────────────────────────────────────────────────────────────────────────────────────────────────────

def document(n_tokens, salt):
    """Real prose (this repository's docs) of about n_tokens, after a unique first line so no prompt cache applies."""
    files = sorted(glob.glob(os.path.join(REPO, "docs", "**", "*.md"), recursive=True))
    files.insert(0, os.path.join(REPO, "README.md"))
    parts, want = [f"Document {salt}.\n"], n_tokens * CHARS_PER_TOKEN
    size = len(parts[0])
    while size < want:
        for path in files:
            with open(path, errors="replace") as f:
                text = f.read()
            parts.append(text[: want - size])
            size += len(parts[-1])
            if size >= want:
                break
    return "".join(parts)


NO_THINK = {"temperature": 0, "seed": 7, "chat_template_kwargs": {"enable_thinking": False},
            "stream_options": {"include_usage": True}}
CODE = ("Write a complete, well commented Python module implementing a thread-safe LRU cache with a TTL per entry, "
        "a decorator that memoizes a function with it, and unittest tests covering eviction, expiry and concurrency.")
SUMMARY = "Summarize the document above as a detailed bulleted list, one bullet per section, in plain English."


def speed_case(port, model, label, messages, max_tokens, out_dir, timeout):
    body = dict(NO_THINK, model=model, messages=messages, max_tokens=max_tokens, stream=True)
    rec = chat(port, body, timeout, os.path.join(out_dir, f"{label}.sse"))
    ttft, n = rec.get("ttft_s"), rec.get("completion_tokens")
    if ttft and n and rec["wall_s"] > ttft:
        rec["decode_tok_s"] = round((n - 1) / (rec["wall_s"] - ttft), 1)
    if ttft and rec.get("prompt_tokens"):
        rec["prefill_tok_s"] = round((rec["prompt_tokens"] - (rec.get("cached_tokens") or 0)) / ttft, 1)
    keep = ("status", "error", "prompt_tokens", "cached_tokens", "completion_tokens", "finish_reason", "ttft_s",
            "wall_s", "decode_tok_s", "prefill_tok_s")
    short = {k: rec.get(k) for k in keep}
    print(f"{label}: {json.dumps(short)}", flush=True)
    return short


def speed(port, model, out_dir, timeout, salt):
    os.makedirs(out_dir, exist_ok=True)
    res = {"code_512": speed_case(port, model, "code_512", [{"role": "user", "content": f"[{salt}] {CODE}"}], 512,
                                  out_dir, timeout)}
    for k in (2, 8, 28):
        doc = document(k * 1000, f"{salt}-{k}k")
        res[f"doc_{k}k"] = speed_case(port, model, f"doc_{k}k", [{"role": "user", "content": f"{doc}\n\n{SUMMARY}"}],
                                      256, out_dir, timeout)
    return res


def follow_up(port, model, out_dir, timeout, salt):
    """A conversation's next turn: turn 1, then turn 1 + its reply + a question, against a cold prompt of turn 2's
    length; at ~1.5k tokens (under 1.1.4's 4,096-row resume chunk on CUDA) and ~8k."""
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for k in (1.5, 8):
        tag = f"{k:g}k"
        msgs = [{"role": "user", "content": f"{document(int(k * 1000), f'{salt}-conv{tag}')}\n\n{SUMMARY}"}]
        res[f"turn1_{tag}"] = speed_case(port, model, f"turn1_{tag}", msgs, 128, out_dir, timeout)
        with open(os.path.join(out_dir, f"turn1_{tag}.sse"), "rb") as f:
            reply = "".join((json.loads(line[6:])["choices"] or [{}])[0].get("delta", {}).get("content") or ""
                            for line in f if line.startswith(b"data: {"))
        msgs += [{"role": "assistant", "content": reply},
                 {"role": "user", "content": "Which section of the document is the longest? Answer in one sentence."}]
        res[f"turn2_cached_{tag}"] = speed_case(port, model, f"turn2_cached_{tag}", msgs, 64, out_dir, timeout)
        n = res[f"turn2_cached_{tag}"].get("prompt_tokens") or int(k * 1000)
        res[f"cold_{tag}"] = speed_case(port, model, f"cold_{tag}", [{"role": "user", "content": (
            f"{document(n - 30, f'{salt}-cold{tag}')}\n\n{SUMMARY}")}], 64, out_dir, timeout)
    return res


def features(port, model, out_dir, timeout):
    runs = {}
    run_prompts(port, model, ("whole1", "whole2", "stream"), timeout, out_dir, runs)
    baseline = read_json(BASELINE) if os.path.exists(BASELINE) else {}
    baseline = baseline.get("prompts") if baseline.get("tensorfold") == TENSORFOLD else None
    checks = []
    for p in read_json(PROMPTS)["prompts"]:
        r = runs[p["id"]]
        for label, rec in r.items():
            checks.append({"check": f"{p['id']} {label} ok", "pass": not rec.get("error"), "detail": rec.get("error")})
            for kind, want in (p.get("expect") or {}).items():
                checks.append({"check": f"{p['id']} {label} {kind}", "pass": expectation(kind, want, rec)})
        for f in FIELDS:
            checks.append({"check": f"{p['id']} whole1 = whole2 {f}", "pass": r["whole1"].get(f) == r["whole2"].get(f)})
            same = r["stream"].get(f) == r["whole1"].get(f)
            if not same and f == "reasoning_content":
                same = stream_reasoning_tail(r["stream"].get(f), r["whole1"].get(f), TENSORFOLD)
            checks.append({"check": f"{p['id']} stream = whole {f}", "pass": same})
        checks.append({"check": f"{p['id']} stream [DONE]", "pass": bool(r["stream"].get("done"))})
        want = (baseline or {}).get(p["id"])
        if want is not None:
            for f in FIELDS:
                checks.append({"check": f"{p['id']} whole1 = Mac baseline {f}", "pass": r["whole1"].get(f) == want[f]})
    failed = [c for c in checks if not c["pass"]]
    print(f"features: {len(checks) - len(failed)}/{len(checks)} checks pass", flush=True)
    for c in failed:
        print(f"  FAIL {c['check']} {c.get('detail') or ''}", flush=True)
    return {"runs": runs, "checks": checks, "failed": [c["check"] for c in failed]}


def suite(port, model, out, timeout, res):
    salt = f"{time.time_ns()}"
    res["features"] = features(port, model, os.path.join(out, "features"), timeout)
    res["speed"] = speed(port, model, os.path.join(out, "speed"), timeout, salt)
    res["follow_up"] = follow_up(port, model, os.path.join(out, "follow_up"), timeout, salt)


def summary(res):
    gib = lambda b: round(b / GIB, 2) if b else None  # noqa: E731
    s = {"mode": res["mode"], "drafter": res.get("drafter"), "load_s": res.get("load_s"),
         "footprint_loaded_gib": gib(res.get("footprint_loaded")), "footprint_peak_gib": gib(res.get("footprint_peak")),
         "features_failed": res.get("features", {}).get("failed"),
         "speed": {k: {x: v.get(x) for x in ("prompt_tokens", "ttft_s", "decode_tok_s", "prefill_tok_s")}
                   for k, v in (res.get("speed") or {}).items()},
         "follow_up": {k: {x: v.get(x) for x in ("prompt_tokens", "cached_tokens", "ttft_s")}
                       for k, v in (res.get("follow_up") or {}).items()},
         "error": res.get("error")}
    return s


def finish(out, res):
    write_json(os.path.join(out, "bench.json"), res)
    s = summary(res)
    write_json(os.path.join(out, "summary.json"), s)
    print(json.dumps(s, indent=1), flush=True)
    return 0 if not res.get("error") and not s["features_failed"] else 1


def serve(args):
    cfg, tool = load_tool(args)
    opts = options(tool)
    port = free_port()
    argv = server_argv(tool, port)
    if args.no_drafter:
        i = argv.index("--drafter")
        del argv[i:i + 2]
    log = os.path.join(args.out, "server.log")
    res = {"mode": "serve", "argv": argv, "drafter": not args.no_drafter, "port": port, "error": None,
           "system_before": system_memory()}
    p = sampler = None
    timeout = int(tool.get("generate_timeout_s") or 600)
    try:
        if server_pids():
            raise Fail(f"tensorfold-native serve already runs (pids {server_pids()})")
        t0 = time.monotonic()
        with open(log, "wb") as logf:
            p = subprocess.Popen(argv, env=server_env(tool), stdin=subprocess.DEVNULL, stdout=logf,
                                 stderr=subprocess.STDOUT)
        sampler = Sampler(p.pid, os.path.join(args.out, "memory.log"))
        sampler.start()
        res["health"] = wait_health(p, port, opts.get("health") or "/health", int(tool.get("load_timeout_s") or 900),
                                    log)
        res["load_s"] = round(time.monotonic() - t0, 3)
        res["footprint_loaded"], _ = footprint(p.pid)
        res["footprint_loaded_report"] = footprint_report(p.pid, os.path.join(args.out, "footprint-loaded.txt"))
        res["system_loaded"] = system_memory()
        print(f"server ready in {res['load_s']} s, footprint {res['footprint_loaded'] / GIB:.2f} GiB", flush=True)
        suite(port, tool["id"], args.out, timeout, res)
        res["footprint_end_report"] = footprint_report(p.pid, os.path.join(args.out, "footprint-end.txt"))
        res["footprint_end"], res["footprint_peak"] = footprint(p.pid)
        res["health_end"] = get_json(port, opts.get("health") or "/health")
    except SystemExit:
        res["error"] = "terminated (SIGTERM)"
        raise
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["log_tail"] = tail(log)
    finally:
        if sampler is not None:
            sampler.stop.set()
            sampler.join()
            res["footprint_peak_sampled"] = sampler.peak
            res["footprint_peak"] = max(res.get("footprint_peak") or 0, sampler.peak)
        if p is not None:
            t = time.monotonic()
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(STOP_GRACE_S)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                    res["killed"] = True
            res["stop"] = {"exit": p.returncode, "stop_s": round(time.monotonic() - t, 3)}
        time.sleep(2)
        res["left_running"] = server_pids()
        res["system_after"] = system_memory()
        rc = finish(args.out, res)
    return 1 if res["left_running"] else rc


def port_mode(args):
    res = {"mode": "port", "port": args.port, "model": args.model, "error": None}
    pid = args.server_pid or (server_pids() or [None])[0]
    sampler = None
    if pid:
        res["server_pid"] = pid
        res["footprint_loaded"], _ = footprint(pid)
        sampler = Sampler(pid, os.path.join(args.out, "memory.log"))
        sampler.start()
    try:
        res["models"] = get_json(args.port, "/v1/models")
        suite(args.port, args.model, args.out, args.timeout, res)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        if sampler is not None:
            sampler.stop.set()
            sampler.join()
            _, life = footprint(pid)
            res["footprint_peak"] = max(life or 0, sampler.peak)
    return finish(args.out, res)


def mcp_call(port, name, arguments, timeout):
    """One MCP tools/call over POST /mcp; the JSON-RPC result, its text and how long it took."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    t0 = time.monotonic()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request("POST", "/mcp", body=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
        r = c.getresponse()
        status, data = r.status, r.read()
    finally:
        c.close()
    rec = {"status": status, "wall_s": round(time.monotonic() - t0, 3)}
    try:
        text = data.decode()
        if text.lstrip().startswith("event:") or text.lstrip().startswith("data:"):
            text = next(ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:"))
        result = json.loads(text).get("result") or {}
    except (ValueError, StopIteration) as e:
        rec["error"] = f"{type(e).__name__}: {data[:512]!r}"
        return rec
    rec["is_error"] = bool(result.get("isError"))
    rec["text"] = "".join(c.get("text", "") for c in result.get("content") or [])[:2000]
    return rec


def daemon(args):
    """LOCALROUTER serve with the tool alone, the suite through it, MCP load_model and chat; then SIGTERM."""
    cfg, tool = load_tool(args)
    data = os.path.join(args.out, "daemon-data")
    os.makedirs(data)
    t = dict(tool)
    t.pop("qualification", None)
    dcfg = {"host": "127.0.0.1", "port": 0, "data_dir": data, "reserve_bytes": 8 * GIB,
            "defaults": {"chat_completion": t["id"]}, "tools": [t]}
    cpath = os.path.join(args.out, "daemon-config.json")
    write_json(cpath, dcfg)
    argv = [args.bin, "serve", "--config", cpath, "--host", "127.0.0.1", "--port", "0", "--data", data]
    res = {"mode": "daemon", "argv": argv, "drafter": "--drafter" in server_argv(tool, 0), "error": None,
           "system_before": system_memory()}
    p = sampler = pid = None
    timeout = int(t.get("load_timeout_s") or 900) + int(t.get("generate_timeout_s") or 600)
    try:
        if server_pids():
            raise Fail(f"tensorfold-native serve already runs (pids {server_pids()})")
        with open(os.path.join(args.out, "daemon.log"), "wb") as errf:
            p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errf)
        line = p.stdout.readline().decode(errors="replace").strip()
        prefix = "localrouter listening on http://"
        if not line.startswith(prefix):
            raise Fail(f"unexpected first stdout line {line!r}; daemon.log tail:\n"
                       f"{tail(os.path.join(args.out, 'daemon.log'))}")
        port = int(line[len(prefix):].rsplit(":", 1)[1].rstrip("/"))
        res["port"] = port
        models = get_json(port, "/v1/models")
        entry = next((m for m in models.get("data") or [] if m.get("id") == t["id"]), None)
        res["models_entry"] = entry
        if not (entry and entry.get("kind") == "chat" and "chat_completion" in (entry.get("capabilities") or [])):
            raise Fail(f"/v1/models does not list {t['id']} as chat: {models}")
        res["mcp_load_model"] = mcp_call(port, "load_model", {"model": t["id"]}, timeout)
        res["load_s"] = res["mcp_load_model"]["wall_s"]
        pid = (server_pids() or [None])[0]
        if pid is None:
            raise Fail(f"no tensorfold-native serve after load_model: {res['mcp_load_model']}")
        res["server_pid"] = pid
        res["footprint_loaded"], _ = footprint(pid)
        sampler = Sampler(pid, os.path.join(args.out, "memory.log"))
        sampler.start()
        print(f"loaded through the daemon in {res['load_s']} s", flush=True)
        res["mcp_chat"] = mcp_call(port, "chat", {"prompt": "Three facts about Canberra.", "reasoning": False,
                                                  "max_tokens": 200}, timeout)
        print(f"MCP chat: {json.dumps(res['mcp_chat'])[:400]}", flush=True)
        if res["mcp_chat"].get("error") or res["mcp_chat"].get("is_error") or "Canberra" not in res["mcp_chat"]["text"]:
            raise Fail(f"MCP chat failed: {res['mcp_chat']}")
        suite(port, t["id"], args.out, timeout, res)
        res["tools_end"] = get_json(port, "/v1/tools")
    except SystemExit:
        res["error"] = "terminated (SIGTERM)"
        raise
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        if sampler is not None:
            sampler.stop.set()
            sampler.join()
            _, life = footprint(pid)
            res["footprint_peak"] = max(life or 0, sampler.peak)
        if p is not None:
            res["stop"] = stop_process(p, STOP_GRACE_S)
        deadline = time.monotonic() + 10
        while server_pids() and time.monotonic() < deadline:
            time.sleep(0.5)
        res["left_running"] = server_pids()
        res["system_after"] = system_memory()
        rc = finish(args.out, res)
    return 1 if res["left_running"] else rc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--config", required=True)
    s.add_argument("--tool", default="qwen3.8-27b")
    s.add_argument("--out", required=True)
    s.add_argument("--no-drafter", action="store_true")
    q = sub.add_parser("port")
    q.add_argument("--port", type=int, required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--server-pid", type=int)
    q.add_argument("--timeout", type=int, default=600)
    d = sub.add_parser("daemon")
    d.add_argument("--bin", required=True)
    d.add_argument("--config", required=True)
    d.add_argument("--tool", default="qwen3.8-27b")
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if os.listdir(args.out):
        sys.exit(f"bench_macos.py: {args.out} is not empty: give each run a new OUT")
    signal.signal(signal.SIGTERM, on_term)
    return {"serve": serve, "port": port_mode, "daemon": daemon}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
