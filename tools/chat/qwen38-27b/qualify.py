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

import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # -I leaves the script's directory (and its parent) off sys.path
import qualify_lib as q  # noqa: E402
from qualify_lib import (GIB, Fail, git_sha1_file, hf_cache, load_tool, mem_available, options,  # noqa: E402
                         platform_name, read_json, server_argv, server_running, sha256_file, write_json)

PINS = os.path.join(HERE, "pins.json")
PROMPTS = os.path.join(HERE, "prompts.json")
MODEL = "qwen3.8-27b"


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
    busy = server_running(argv)
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


def relax(field, label, stream, whole):
    if field == "reasoning_content" and label.endswith(".stream") and \
            stream_reasoning_tail(stream, whole, read_json(PINS)["tensorfold"]["version"]):
        return STREAM_REASONING_TRAILING_WS
    return None


def record(comp, pins_r, gpu):
    return {"model": MODEL, "host": os.uname().nodename, "gpu": gpu, "pins_sha256": sha256_file(PINS),
            "tensorfold_sha256": pins_r["archive"]["sha256"], "checks": comp["checks"],
            "exceptions": comp.get("exceptions", []),
            "exception_evidence": {n: EXCEPTIONS[n]["evidence"] for n in comp.get("exceptions", [])}}


if __name__ == "__main__":
    sys.exit(q.main(MODEL, {"preflight": preflight, "pins": pins_role, "direct": lambda a: q.direct(a, PROMPTS),
                            "daemon": lambda a: q.daemon(a, PROMPTS),
                            "compare": lambda a: q.compare(a, PROMPTS, EXCEPTIONS, relax),
                            "marker": lambda a: q.marker(a, record), "report": q.report}))
