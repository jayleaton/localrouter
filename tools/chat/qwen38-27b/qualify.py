#!/usr/bin/env python3
"""Qwen3.8-27B (TensorFold 1.1.1) qualification roles; qualify.sh runs them in order. Standard library only.

    python3 -I qualify.py ROLE --root ROOT --config CONFIG --out OUT [--tool ID] [--bin LOCALROUTER] [--marker PATH]

preflight  the config's tool, the marker path and the host are fit to run (no server from the binary running).
pins       the archive, the extracted tree (against the archive's SHA256SUMS) and every pinned checkpoint and
           drafter file in the config's HF_HUB_CACHE, by sha256 or git blob sha1. OUT/pins.json.
direct     starts the tool's `options.server` argv itself (same env), waits for its health path, sends every prompt
           in prompts.json whole twice and streamed once, then SIGTERMs it. OUT/direct.json.
daemon     the same prompts through `LOCALROUTER serve` (the tool without its gate, keep_loaded false), whole and
           streamed, and GET /v1/models; SIGTERMs the daemon. OUT/daemon.json.
compare    determinism, daemon = direct, stream = whole, [DONE], and each prompt's structural expectations.
           OUT/compare.json; exit 1 on any failed check.
marker     writes PATH atomically when OUT's pins and compare passed on a GB10; --revoke removes it if OUT wrote it.
report     OUT/report.json: step status, the memory peak from memory.log and phases.log, timings.
"""

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import tarfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script's directory off sys.path
from qualify_lib import (FIELDS, GIB, MODEL, PINS, PROMPTS, STOP_GRACE_S, Fail, free_port, get_json,  # noqa: E402
                         git_sha1_file, gpu_name, hf_cache, load_tool, mem_available, on_term, options,
                         platform_name, processes_of, read_json, run_prompts, runs_ok, server_argv, server_env,
                         sha256_file, stop_process, tail, take_lock, wait_gone, wait_health, write_json)


def preflight(args):
    if sys.version_info < (3, 9):
        raise Fail(f"needs Python 3.9 or newer, this is {sys.version.split()[0]}")
    cfg, tool = load_tool(args)
    if tool.get("kind") != "chat" or tool.get("engine") != "openai_server":
        raise Fail(f"tool {args.tool} is kind {tool.get('kind')!r}, engine {tool.get('engine')!r}: "
                   "expected chat / openai_server")
    argv = server_argv(tool, 0)
    if not any("{port}" in a for a in options(tool)["server"]):
        raise Fail("options.server has no {port}")
    if not os.path.isabs(args.marker):
        raise Fail(f"the marker path {args.marker} is not absolute")
    q = tool.get("qualification") or ""
    if q and os.path.realpath(q) != os.path.realpath(args.marker):
        raise Fail(f"tool {args.tool}'s qualification is {q}, not {args.marker}: the daemon would not read this marker")
    if not q:
        print(f"preflight: WARNING: tool {args.tool} has no \"qualification\" path; the marker gates nothing")
    if not (args.bin and os.access(args.bin, os.X_OK)):
        raise Fail(f"LOCALROUTER_BIN {args.bin} is not executable")
    busy = processes_of(argv[0])
    if busy:
        raise Fail(f"{argv[0]} already runs (pids {busy}): stop the daemon or server that runs it first")
    print(f"preflight: tool {args.tool}, server {argv[0]}, marker {args.marker}, MemAvailable "
          f"{mem_available() / GIB:.1f} GiB")
    return 0


def pins_role(args):
    cfg, tool = load_tool(args)
    pins = read_json(PINS)
    tf, plat = pins["tensorfold"], platform_name()
    res = {"pass": False, "platform": plat, "problems": [], "files": []}
    problem = res["problems"].append
    tree = os.path.join(args.root, tf["dir"])
    a = tf["archives"].get(plat or "")
    if not a:
        problem(f"no archive pinned for {os.uname().machine}")
    else:
        path = os.path.join(args.root, "downloads", os.path.basename(a["url"]))
        size = os.path.getsize(path) if os.path.isfile(path) else None
        digest = sha256_file(path) if size is not None else None
        res["archive"] = {"path": path, "size": size, "sha256": digest,
                          "pass": size == a["size"] and digest == a["sha256"]}
        if not res["archive"]["pass"]:
            problem(f"{path}: size {size}, sha256 {digest}; pinned {a['size']}, {a['sha256']}")
        else:
            try:
                with open(tree + ".archive-sha256") as f:
                    stamp = f.read().strip()
            except OSError:
                stamp = None
            if stamp != a["sha256"]:
                problem(f"{tree} was not extracted from the pinned archive (stamp {stamp})")
            try:  # read one member; nothing is extracted
                with tarfile.open(path, "r:gz") as t:
                    sums = t.extractfile(t.getmember(a["top_dir"] + "/SHA256SUMS")).read()
            except (KeyError, OSError, tarfile.TarError) as e:
                problem(f"{path}: no {a['top_dir']}/SHA256SUMS ({e})")
                sums = b""
            listed = {}
            for line in sums.decode().splitlines():
                if line.strip():
                    digest_, name = line.split(None, 1)
                    listed[os.path.normpath(name.lstrip("*"))] = digest_
            try:
                with open(os.path.join(tree, "SHA256SUMS"), "rb") as f:
                    if f.read() != sums:
                        problem(f"{tree}/SHA256SUMS differs from the archive's")
            except OSError as e:
                problem(f"{tree}/SHA256SUMS: {e}")
            bad = [n for n, d in sorted(listed.items())
                   if not os.path.isfile(os.path.join(tree, n)) or sha256_file(os.path.join(tree, n)) != d]
            extra = sorted(os.path.relpath(os.path.join(dp, fn), tree) for dp, _, fns in os.walk(tree) for fn in fns
                           if os.path.relpath(os.path.join(dp, fn), tree) not in listed
                           and os.path.relpath(os.path.join(dp, fn), tree) != "SHA256SUMS")
            res["tree"] = {"dir": tree, "files": len(listed), "mismatched": bad, "extra": extra}
            if bad or extra:
                problem(f"{tree}: {len(bad)} files differ from the archive, extra files {extra}")
    # The config runs this binary on the pinned checkpoint and drafter.
    argv = server_argv(tool, 0)
    binary = os.path.join(tree, tf["binary"])
    if os.path.realpath(argv[0]) != os.path.realpath(binary):
        problem(f"options.server[0] is {argv[0]}, not the verified {binary}")
    ck, dr = pins["checkpoint"], pins["drafter"]
    if ck["repo"] not in argv[1:]:
        problem(f"options.server does not serve {ck['repo']}")
    at = argv.index("--drafter") if "--drafter" in argv else -1
    if at < 0 or at + 1 >= len(argv) or argv[at + 1] != dr["repo"]:
        problem(f"options.server does not pass --drafter {dr['repo']}")
    if not os.path.isfile(os.path.join(tree, tf["grammar_lib"])):
        problem(f"{tf['grammar_lib']} is missing from {tree}")
    hf = hf_cache(tool)
    res["hf_cache"] = hf
    for part in (ck, dr):
        cache = os.path.join(hf, "models--" + part["repo"].replace("/", "--"))
        try:
            with open(os.path.join(cache, "refs", "main")) as f:
                ref = f.read().strip()
        except OSError:
            ref = None
        if ref != part["revision"]:
            problem(f"{cache}/refs/main is {ref}, pinned {part['revision']}")
        snap = os.path.join(cache, "snapshots", part["revision"])
        for f in part["files"]:
            p = os.path.join(snap, f["path"])
            size = os.path.getsize(p) if os.path.isfile(p) else None
            kind = "sha256" if "sha256" in f else "git_sha1"
            got = None if size is None else (sha256_file(p) if kind == "sha256" else git_sha1_file(p, size))
            ok = size == f["size"] and got == f[kind]
            res["files"].append({"repo": part["repo"], "path": f["path"], "size": size, kind: got, "pass": ok})
            if not ok:
                problem(f"{p}: size {size}, {kind} {got}; pinned {f['size']}, {f[kind]}")
            print(f"{part['repo']}/{f['path']}: {'ok' if ok else 'MISMATCH'}", flush=True)
        pinned = {f["path"] for f in part["files"]}
        extra = sorted(os.path.relpath(os.path.join(dp, fn), snap) for dp, _, fns in os.walk(snap) for fn in fns
                       if os.path.relpath(os.path.join(dp, fn), snap) not in pinned)
        if extra:
            problem(f"{snap} holds files that are not pinned: {extra}")
    res["pins_sha256"] = sha256_file(PINS)
    res["pass"] = not res["problems"]
    write_json(os.path.join(args.out, "pins.json"), res)
    print(json.dumps({"pass": res["pass"], "problems": res["problems"]}, indent=1))
    return 0 if res["pass"] else 1
def direct(args):
    cfg, tool = load_tool(args)
    opts = options(tool)
    port = int(opts.get("port") or 0) or free_port()
    argv = server_argv(tool, port)
    log = os.path.join(args.out, "direct-server.log")
    res = {"argv": argv, "env": opts.get("env") or {}, "port": port, "runs": {}, "error": None}
    lock = p = None
    try:
        lock = take_lock(cfg)
        busy = processes_of(argv[0])
        if busy:
            raise Fail(f"{argv[0]} already runs (pids {busy})")
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
        run_prompts(port, tool["id"], ("whole1", "whole2", "stream"), int(tool.get("generate_timeout_s") or 600),
                    os.path.join(args.out, "direct"), res["runs"])
        res["health_end"] = get_json(port, opts.get("health") or "/health")
    except SystemExit:
        res["error"] = "terminated (SIGTERM)"
        raise
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        if p is not None:
            res["stop"] = stop_process(p, STOP_GRACE_S)
            res["stop"]["within_stop_timeout_s"] = res["stop"]["stop_s"] <= float(opts.get("stop_timeout_s") or 8)
        res["left_running"] = wait_gone(argv[0], 10)
        res["server_gone"] = not res["left_running"]
        if lock is not None:
            os.close(lock)
        res["mem_available_after"] = mem_available()
        write_json(os.path.join(args.out, "direct.json"), res)
    print(json.dumps({"error": res["error"], "server_gone": res["server_gone"], "stop": res.get("stop")}))
    return 0 if res["error"] is None and res["server_gone"] and runs_ok(res["runs"]) else 1


def daemon(args):
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
    server0 = server_argv(tool, 0)[0]
    res = {"argv": argv, "runs": {}, "error": None}
    p = None
    try:
        busy = processes_of(server0)
        if busy:
            raise Fail(f"{server0} already runs (pids {busy})")
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
        run_prompts(port, t["id"], ("whole", "stream"), timeout, os.path.join(args.out, "daemon"), res["runs"])
        res["tools_end"] = get_json(port, "/v1/tools")
        res["mem_available_loaded"] = mem_available()
    except SystemExit:
        res["error"] = "terminated (SIGTERM)"
        raise
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        if p is not None:
            res["stop"] = stop_process(p, STOP_GRACE_S)
        res["left_running"] = wait_gone(server0, 10)
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


# Named, narrow exceptions to exact equality, each admitted on recorded evidence; nothing else is relaxed. A check
# passed under one names it, and the marker lists every exception the run used.
STREAM_REASONING_TRAILING_WS = "tensorfold-stream-reasoning-trailing-newline"
EXCEPTIONS = {
    STREAM_REASONING_TRAILING_WS: {
        "tensorfold": ["1.1.1", "1.1.4"],
        "scope": "reasoning_content only, a stream against the whole reply of the same server only: the stream may "
                 "end with whitespace the whole reply lacks; everything else in both must be equal",
        "evidence": "DGX-01 qualify run1 (commit 6e83844, TensorFold 1.1.1) and the 1.1.4 run (commit 37755e0): P1 "
                    "stream reasoning_content 599 B vs whole 598 B, P2 352 B vs 351 B, on both versions; the stream's "
                    "last reasoning delta ends with \\n that the whole reply strips; the same bytes direct and through "
                    "LocalRouter",
        "upstream": "tools/chat/qwen38-27b/upstream-issue-stream-reasoning-newline.md",
    },
}


def stream_reasoning_tail(stream, whole, tensorfold):
    """True when `stream` is `whole` plus whitespace only, on a TensorFold version the exception names."""
    if tensorfold not in EXCEPTIONS[STREAM_REASONING_TRAILING_WS]["tensorfold"]:
        return False
    if not isinstance(stream, str) or not isinstance(whole, str) or len(stream) <= len(whole):
        return False
    return stream.startswith(whole) and stream[len(whole):].strip() == ""


def compare(args):
    checks = []
    used = set()
    tensorfold = read_json(PINS)["tensorfold"]["version"]

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
    for p in read_json(PROMPTS)["prompts"]:
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
                if (not same and ok[a] and ok[b] and f == "reasoning_content" and a.endswith(".stream")
                        and stream_reasoning_tail(ra.get(f), rb.get(f), tensorfold)):
                    used.add(STREAM_REASONING_TRAILING_WS)
                    add(f"{pid}.{a}=={b}.{f}", True, exception=STREAM_REASONING_TRAILING_WS,
                        stream_bytes=len(ra[f].encode()), whole_bytes=len(rb[f].encode()),
                        stream_tail=ra[f][len(rb[f]):])
                    continue
                info = {} if same else {a: ra.get(f), b: rb.get(f)}
                add(f"{pid}.{a}=={b}.{f}", same, **info)
        base = recs.get("direct.whole1") or {}
        for kind, want in p.get("expect", {}).items():
            add(f"{pid}.expect.{kind}", ok["direct.whole1"] and expectation(kind, want, base), want=want)
    failed = [c["check"] for c in checks if not c["pass"]]
    exceptions = sorted(used)
    result = {"pass": not failed, "checks": len(checks), "failed": failed, "exceptions": exceptions,
              "exception_terms": {n: EXCEPTIONS[n] for n in exceptions}, "list": checks}
    write_json(os.path.join(args.out, "compare.json"), result)
    print(json.dumps({"pass": result["pass"], "checks": len(checks), "failed": failed, "exceptions": exceptions},
                     indent=1))
    return 0 if result["pass"] else 1


def marker(args):
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
    record = {"qualified": True, "model": MODEL, "host": os.uname().nodename, "gpu": gpu,
              "pins_sha256": sha256_file(PINS), "tensorfold_sha256": pins_r["archive"]["sha256"],
              "checks": comp["checks"], "exceptions": comp.get("exceptions", []),
              "exception_evidence": {n: EXCEPTIONS[n]["evidence"] for n in comp.get("exceptions", [])},
              "report": args.out,
              "qualified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    text = json.dumps(record, indent=1) + "\n"
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["preflight", "pins", "direct", "daemon", "compare", "marker", "report"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tool", default=MODEL)
    ap.add_argument("--bin", help="the localrouter binary (preflight, daemon)")
    ap.add_argument("--marker", help="the qualification marker path (preflight, marker)")
    ap.add_argument("--revoke", action="store_true", help="marker: remove the marker if this OUT wrote it")
    args = ap.parse_args()
    args.root, args.out = os.path.abspath(args.root), os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    signal.signal(signal.SIGTERM, on_term)
    roles = {"preflight": preflight, "pins": pins_role, "direct": direct, "daemon": daemon, "compare": compare,
             "marker": marker, "report": report}
    try:
        return roles[args.role](args) or 0
    except Fail as e:
        print(f"qualify.py {args.role}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
