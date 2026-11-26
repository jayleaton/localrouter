#!/usr/bin/env python3
"""Inkling-2-Mini (TensorFold sm_121 in Docker) qualification roles, run in order by qualify.sh. Standard library only.

    python3 -I qualify.py ROLE --root ROOT --config CONFIG --out OUT [--tool ID] [--bin LOCALROUTER] [--marker PATH]

preflight  the config's tool, the marker path and the host are fit to run (docker answers, no container of the tool).
pins       the recipe checkout at its pinned commit, the image the config runs (its tf.patches label), and every
           checkpoint and drafter file the config mounts, by size and sha256. OUT/pins.json.
direct, daemon, compare, report: the shared roles (tools/chat/qualify_lib.py) on prompts.json.
marker     writes PATH atomically when OUT's pins and compare passed on a GB10; --revoke removes it if OUT wrote it.
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # -I leaves the script's directory (and its parent) off sys.path
import qualify_lib as q  # noqa: E402
from qualify_lib import Fail, load_tool, mem_available, options, read_json, server_argv, sha256_file, write_json  # noqa: E402

PINS = os.path.join(HERE, "pins.json")
PROMPTS = os.path.join(HERE, "prompts.json")
HF_PINNED = os.path.join(HERE, "..", "..", "install", "hf_pinned.py")
MODEL = "Inkling-2-Mini"


def docker_parts(argv):
    """The tools/docker/serve.sh argv's docker options, image and command; Fail when it is not one."""
    if os.path.basename(argv[0]) != "serve.sh" or os.path.basename(os.path.dirname(argv[0])) != "docker":
        raise Fail(f"options.server[0] is {argv[0]}, not tools/docker/serve.sh")
    if "--" not in argv:
        raise Fail("options.server has no -- before the image")
    at = argv.index("--")
    return argv[2:at], argv[at + 1], argv[at + 2:]


def mounts(dopts):
    """{container path: host path} of the -v options."""
    out = {}
    for i, a in enumerate(dopts[:-1]):
        if a in ("-v", "--volume"):
            src, dst = dopts[i + 1].split(":")[:2]
            out[dst] = src
    return out


def image_labels(image):
    r = subprocess.run(["docker", "image", "inspect", "-f", "{{json .Config.Labels}}", image], capture_output=True,
                       text=True, timeout=60)
    if r.returncode:
        raise Fail(f"docker image inspect {image}: {r.stderr.strip()}")
    return json.loads(r.stdout) or {}


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
    docker_parts(argv)
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
        raise Fail(f"a container of {args.tool} already runs ({busy}): stop the daemon that runs it first")
    print(f"preflight: tool {args.tool}, image {docker_parts(argv)[1]}, marker {args.marker}, MemAvailable "
          f"{mem_available() / q.GIB:.1f} GiB")
    return 0


def pins_role(args):
    _, tool = load_tool(args)
    pins = read_json(PINS)
    res = {"pass": False, "problems": [], "parts": {}}
    problem = res["problems"].append
    recipe = os.path.join(args.root, "recipe")
    head = subprocess.run(["git", "-C", recipe, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", recipe, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    res["recipe"] = {"dir": recipe, "head": head, "clean": not dirty}
    if head != pins["recipe"]["commit"] or dirty:
        changes = " with changes" if dirty else ""
        problem(f"{recipe} is at {head or 'nothing'}{changes}, pinned {pins['recipe']['commit']}")
    dopts, image, command = docker_parts(server_argv(tool, 0))
    labels = image_labels(image)
    res["image"] = {"id": image, "labels": {k: labels.get(k) for k in ("tf.patches", "tf.kernels")}}
    for k in ("tf.patches", "tf.kernels"):
        if labels.get(k) != pins["image"][k]:
            problem(f"image {image} has {k} {labels.get(k)!r}, pinned {pins['image'][k]!r}")
    if command[:2] != ["tensorfold-native", "serve"] or command[2:3] != ["/models/ink"]:
        problem(f"the container command is {command[:3]}, not tensorfold-native serve /models/ink")
    at = command.index("--drafter") if "--drafter" in command else -1
    if at < 0 or command[at + 1:at + 2] != ["/models/draft"]:
        problem("the container command does not pass --drafter /models/draft")
    mounted = mounts(dopts)
    for part, dst in (("checkpoint", "/models/ink"), ("drafter", "/models/draft")):
        src = mounted.get(dst)
        if not src:
            problem(f"nothing is mounted at {dst}")
            continue
        manifest = os.path.join(args.out, f"{part}.manifest.json")
        write_json(manifest, pins[part])
        r = subprocess.run([sys.executable, "-I", HF_PINNED, "verify", manifest, src], capture_output=True, text=True)
        res["parts"][part] = {"dir": src, "repo": pins[part]["repo"], "revision": pins[part]["revision"],
                              "files": len(pins[part]["files"]), "pass": r.returncode == 0,
                              "problems": r.stderr.strip().splitlines()}
        print(f"{pins[part]['repo']}@{pins[part]['revision'][:12]} in {src}: "
              f"{'ok' if r.returncode == 0 else 'MISMATCH'}", flush=True)
        if r.returncode:
            problem(f"{src}: {r.stderr.strip()}")
    res["pins_sha256"] = sha256_file(PINS)
    res["pass"] = not res["problems"]
    write_json(os.path.join(args.out, "pins.json"), res)
    print(json.dumps({"pass": res["pass"], "problems": res["problems"]}, indent=1))
    return 0 if res["pass"] else 1


def record(comp, pins_r, gpu):
    return {"model": MODEL, "host": os.uname().nodename, "gpu": gpu, "pins_sha256": sha256_file(PINS),
            "image": pins_r["image"]["id"], "tf_patches": pins_r["image"]["labels"]["tf.patches"],
            "recipe_commit": pins_r["recipe"]["head"], "checks": comp["checks"],
            "exceptions": comp.get("exceptions", [])}


if __name__ == "__main__":
    sys.exit(q.main(MODEL, {"preflight": preflight, "pins": pins_role, "direct": lambda a: q.direct(a, PROMPTS),
                            "daemon": lambda a: q.daemon(a, PROMPTS), "compare": lambda a: q.compare(a, PROMPTS),
                            "marker": lambda a: q.marker(a, record), "report": q.report}))
