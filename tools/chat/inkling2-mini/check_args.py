#!/usr/bin/env python3
"""Holds a LocalRouter tool's server argv to the one the recipe renders (standard library only):
    python3 -I check_args.py CONFIG SERVE_JSON [--tool ID]
CONFIG: a daemon config ("-": stdin) with the tool ID (default Inkling-2-Mini). SERVE_JSON: what fetch.sh saved from
`recipe serve PROFILE ENGINE MODEL --dry-run` (ROOT/serve-mac.json, ROOT/serve-spark.json).

The binary, `serve` and the model (the config's {weights} filled in) must be the recipe's, and every flag after them the
recipe's, with the same values in the same order, except for these, which the daemon owns:
  --name {id}                  the recipe's served name; the daemon passes its tool id
  --port {port}                the recipe's port; the daemon picks a free one
  --host 127.0.0.1             added: the server listens on loopback only
  --prompt-cache-disk ABSPATH  the recipe's is relative to the directory `recipe serve` runs from
Paths compare by realpath. Every variable the recipe's launch sets must be in the tool's options.env with its value.
Exit 0 and one line when they agree; exit 1 and one line per difference."""

import json
import os
import sys

OWNED = ("--name", "--port", "--host", "--prompt-cache-disk")


class Fail(Exception):
    pass


def flags(args, where):
    """[(flag, value or None)] of args, each --flag followed by its value unless the next is a flag."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if not a.startswith("--"):
            raise Fail(f"{where}: {a!r} is not a flag")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            out.append((a, args[i + 1]))
            i += 2
        else:
            out.append((a, None))
            i += 1
    return out


def same(a, b):
    if a == b:
        return True
    return bool(a and b and os.path.isabs(a) and os.path.isabs(b) and os.path.realpath(a) == os.path.realpath(b))


def check(tool, launch):
    problems = []
    argv = (tool.get("options") or {}).get("server")
    if not isinstance(argv, list) or len(argv) < 3 or not all(isinstance(a, str) for a in argv):
        raise Fail(f"tool {tool.get('id')}: options.server must be an array of at least 3 strings")
    weights = tool.get("weights") or ""
    argv = [a.replace("{weights}", weights) for a in argv]
    want = launch.get("argv") or []
    if len(want) < 3:
        raise Fail("SERVE_JSON's argv has no binary, serve and model")
    for i, what in enumerate(("binary", "command", "model")):
        if not same(argv[i], want[i]):
            problems.append(f"the {what} is {argv[i]}, the recipe's {want[i]}")
    have, rec = flags(argv[3:], "the config"), flags(want[3:], "the recipe")
    mine = {f: v for f, v in have if f in OWNED}
    theirs = {f: v for f, v in rec if f in OWNED}
    for f in OWNED:
        if sum(1 for g, _ in have if g == f) > 1:
            problems.append(f"the config passes {f} more than once")
    if mine.get("--name") != "{id}":
        problems.append(f"--name is {mine.get('--name')!r}, not {{id}} (the recipe's {theirs.get('--name')!r})")
    if mine.get("--port") != "{port}":
        problems.append(f"--port is {mine.get('--port')!r}, not {{port}}")
    if "--host" in mine and mine["--host"] != "127.0.0.1":
        problems.append(f"--host is {mine['--host']!r}: only 127.0.0.1 may be added")
    if "--host" in theirs:
        problems.append(f"the recipe passes --host {theirs['--host']}; check_args.py has no rule for it")
    if "--prompt-cache-disk" in theirs:
        d = mine.get("--prompt-cache-disk")
        r = theirs["--prompt-cache-disk"]
        if not d or not os.path.isabs(d) or (os.path.isabs(r) and not same(d, r)):
            problems.append(f"--prompt-cache-disk is {d!r}, the recipe's {r!r}: it must be an absolute path"
                            + ("" if not os.path.isabs(r) else f" equal to {r}"))
    elif "--prompt-cache-disk" in mine:
        problems.append("the config passes --prompt-cache-disk, which the recipe does not")
    rest = [p for p in have if p[0] not in OWNED]
    rest_want = [p for p in rec if p[0] not in OWNED]
    if len(rest) != len(rest_want) or not all(a[0] == b[0] and (a[1] is None) == (b[1] is None) and
                                              (a[1] is None or same(a[1], b[1])) for a, b in zip(rest, rest_want)):
        fmt = lambda ps: " ".join(f if v is None else f"{f} {v}" for f, v in ps)  # noqa: E731
        problems.append(f"the flags differ:\n    config: {fmt(rest)}\n    recipe: {fmt(rest_want)}")
    env = (tool.get("options") or {}).get("env") or {}
    for k, v in (launch.get("env") or {}).items():
        if env.get(k) != v:
            problems.append(f"options.env.{k} is {env.get(k)!r}, the recipe's launch sets {v!r}")
    return problems, len(rest)


def main(argv):
    args, tool_id = list(argv), "Inkling-2-Mini"
    if "--tool" in args:
        i = args.index("--tool")
        if i + 1 >= len(args):
            print(__doc__, file=sys.stderr)
            return 2
        tool_id = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        cfg = json.load(sys.stdin) if args[0] == "-" else json.load(open(args[0]))
        with open(args[1]) as f:
            launches = json.load(f)
        if not isinstance(launches, list) or len(launches) != 1:
            raise Fail(f"{args[1]} has {len(launches) if isinstance(launches, list) else 'no'} launches, not one")
        tool = next((t for t in cfg.get("tools") or [] if t.get("id") == tool_id), None)
        if tool is None:
            raise Fail(f"{args[0]} has no tool {tool_id!r}")
        problems, n = check(tool, launches[0])
    except (OSError, ValueError, Fail) as e:
        print(f"check_args: {e}", file=sys.stderr)
        return 1
    if problems:
        for p in problems:
            print(f"check_args: {tool_id}: {p}", file=sys.stderr)
        print(f"check_args: {tool_id}'s server argv is not the recipe's ({args[1]})", file=sys.stderr)
        return 1
    print(f"check_args: {tool_id}'s server argv is the recipe's ({args[1]}: the binary, the model and {n} flags)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
