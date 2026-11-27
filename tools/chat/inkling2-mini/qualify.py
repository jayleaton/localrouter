#!/usr/bin/env python3
"""Inkling-2-Mini (the recipe's native CUDA engine, its spark profile) qualification roles, run in order by qualify.sh.
Standard library only.

    python3 -I qualify.py ROLE --root ROOT --config CONFIG --out OUT [--tool ID] [--bin LOCALROUTER] [--marker PATH]

preflight  the config's tool, the marker path and the host are fit to run (no server from the engine binary running).
pins       the recipe checkout at its pinned commit and clean, carrying the pinned kit release, the config's binary the one recipe prepare
           built in ROOT/engine, the config's server argv the recipe's (check_args.py against ROOT/serve-spark.json),
           and every checkpoint file the server loads (its model and its DSpark drafter, the same directory) by size
           and sha256. OUT/pins.json.
direct, daemon, compare, report: the shared roles (tools/chat/qualify_lib.py) on prompts.json.
marker     writes PATH atomically when OUT's pins and compare passed on a GB10; --revoke removes it if OUT wrote it.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # -I leaves the script's directory (and its parent) off sys.path
sys.path.insert(1, HERE)
import check_args  # noqa: E402
import qualify_lib as q  # noqa: E402
from qualify_lib import Fail, load_tool, mem_available, options, read_json, server_argv, sha256_file, write_json  # noqa: E402

PINS = os.path.join(HERE, "pins.json")
PROMPTS = os.path.join(HERE, "prompts.json")
HF_PINNED = os.path.join(HERE, "..", "..", "install", "hf_pinned.py")
MODEL = "Inkling-2-Mini"
ENGINE = "cuda"


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True).stdout.strip()


def binary_of(args):
    return os.path.join(args.root, "engine", read_json(PINS)["engines"][ENGINE]["binary"])


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
    if not os.access(argv[0], os.X_OK):
        raise Fail(f"options.server[0] {argv[0]} is not executable: run fetch.sh and setup.sh first")
    if not os.path.isabs(args.marker):
        raise Fail(f"the marker path {args.marker} is not absolute")
    marker = tool.get("qualification") or ""
    if marker and os.path.realpath(marker) != os.path.realpath(args.marker):
        raise Fail(f"tool {args.tool}'s qualification is {marker}, not {args.marker}: the daemon would not read this "
                   "marker")
    if not marker:
        print(f"preflight: WARNING: tool {args.tool} has no \"qualification\" path; the marker gates nothing")
    if not (args.bin and os.access(args.bin, os.X_OK)):
        raise Fail(f"LOCALROUTER_BIN {args.bin} is not executable")
    busy = q.server_running(argv)
    if busy:
        raise Fail(f"{argv[0]} already runs (pids {busy}): stop the daemon that runs it first")
    print(f"preflight: tool {args.tool}, engine {argv[0]}, marker {args.marker}, MemAvailable "
          f"{mem_available() / q.GIB:.1f} GiB")
    return 0


def pins_role(args):
    _, tool = load_tool(args)
    pins = read_json(PINS)
    res = {"pass": False, "problems": [], "parts": {}}
    problem = res["problems"].append
    d = os.path.join(args.root, "recipe")
    head = git(d, "rev-parse", "HEAD")
    dirty = git(d, "status", "--porcelain", "--untracked-files=no")
    res["recipe"] = {"dir": d, "head": head, "clean": not dirty}
    if head != pins["recipe"]["commit"] or dirty:
        problem(f"{d} is at {head or 'nothing'}{' with changes' if dirty else ''}, pinned {pins['recipe']['commit']}")
    try:
        with open(os.path.join(d, "kit", "SHA256SUMS")) as f:
            release = f.read().split()[0]
    except (OSError, IndexError):
        release = ""
    res["kit"] = {"commit": pins["kit"]["commit"], "sha256": release}
    if release != pins["kit"]["sha256"]:
        problem(f"{d}/kit holds the kit release {release or 'nothing'}, pinned {pins['kit']['sha256']}")
    argv = server_argv(tool, 0)
    binary = binary_of(args)
    res["engine"] = {"binary": argv[0], "built": binary}
    if os.path.realpath(argv[0]) != os.path.realpath(binary):
        problem(f"the config runs {argv[0]}, not the engine recipe prepare built ({binary})")
    elif os.path.isfile(binary):
        res["engine"]["sha256"] = sha256_file(binary)
        v = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60)
        res["engine"]["version"] = (v.stdout or v.stderr).strip()
    serve = os.path.join(args.root, f"serve-{pins['engines'][ENGINE]['profile']}.json")
    try:
        launches = read_json(serve)
        found, n = check_args.check(tool, launches[0])
        res["argv"] = {"serve_json": serve, "serve_json_sha256": sha256_file(serve), "flags": n, "problems": found}
        for p in found:
            problem(f"the server argv is not the recipe's: {p}")
    except (OSError, ValueError, IndexError, check_args.Fail) as e:
        problem(f"{serve}: {e}")
    weights = tool.get("weights") or ""
    at = argv.index("--drafter") if "--drafter" in argv else -1
    if at < 0 or os.path.realpath(argv[at + 1]) != os.path.realpath(weights):
        problem("the server does not pass --drafter {weights}: DSpark drafts from the checkpoint's mtp.safetensors")
    manifest = os.path.join(args.out, "checkpoint.manifest.json")
    write_json(manifest, pins["checkpoint"])
    r = subprocess.run([sys.executable, "-I", HF_PINNED, "verify", manifest, weights], capture_output=True, text=True)
    res["parts"]["checkpoint"] = {"dir": weights, "repo": pins["checkpoint"]["repo"],
                                  "revision": pins["checkpoint"]["revision"], "files": len(pins["checkpoint"]["files"]),
                                  "pass": r.returncode == 0, "problems": r.stderr.strip().splitlines()}
    print(f"{pins['checkpoint']['repo']}@{pins['checkpoint']['revision'][:12]} in {weights}: "
          f"{'ok' if r.returncode == 0 else 'MISMATCH'}", flush=True)
    if r.returncode:
        problem(f"{weights}: {r.stderr.strip()}")
    res["pins_sha256"] = sha256_file(PINS)
    res["pass"] = not res["problems"]
    write_json(os.path.join(args.out, "pins.json"), res)
    print(json.dumps({"pass": res["pass"], "problems": res["problems"]}, indent=1))
    return 0 if res["pass"] else 1


def record(comp, pins_r, gpu):
    return {"model": MODEL, "host": os.uname().nodename, "gpu": gpu, "pins_sha256": sha256_file(PINS),
            "recipe_commit": pins_r["recipe"]["head"], "kit_commit": pins_r["kit"]["commit"],
            "kit_sha256": pins_r["kit"]["sha256"],
            "engine_sha256": pins_r["engine"].get("sha256"), "engine_version": pins_r["engine"].get("version"),
            "serve_json_sha256": pins_r["argv"]["serve_json_sha256"], "checks": comp["checks"],
            "exceptions": comp.get("exceptions", [])}


if __name__ == "__main__":
    sys.exit(q.main(MODEL, {"preflight": preflight, "pins": pins_role, "direct": lambda a: q.direct(a, PROMPTS),
                            "daemon": lambda a: q.daemon(a, PROMPTS), "compare": lambda a: q.compare(a, PROMPTS),
                            "marker": lambda a: q.marker(a, record), "report": q.report}))
