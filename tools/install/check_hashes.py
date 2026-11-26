#!/usr/bin/env python3
"""Refuses a lock file with an artifact it does not pin by hash (standard library only):
  python3 -I tools/install/check_hashes.py [FILE...]
With no FILE it checks every uv.lock and requirements*.txt under the checkout (git ls-files, else a walk). uv.lock:
every wheel and sdist of every package must carry hash = "sha256:<64 hex>" (uv sync --locked then refuses any download
whose bytes differ; an artifact without a hash would be installed unchecked). requirements*.txt: every requirement
must carry --hash=sha256:<64 hex> (pip/uv --require-hashes). Exit 1 and one line per artifact otherwise."""
import os
import re
import subprocess
import sys
import tomllib

SHA = re.compile(r"sha256:[0-9a-f]{64}")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def uv_lock(path):
    bad = []
    with open(path, "rb") as f:
        lock = tomllib.load(f)
    for p in lock.get("package", []):
        src = p.get("source", {})
        if not any(k in src for k in ("registry", "url")):
            continue  # virtual, editable, path, git: no downloaded artifact
        arts = ([p["sdist"]] if "sdist" in p else []) + list(p.get("wheels", []))
        if not arts:
            bad.append(f"{p['name']} {p.get('version', '')}: no artifact listed")
        for a in arts:
            if not SHA.fullmatch(a.get("hash", "")):
                bad.append(f"{p['name']} {p.get('version', '')}: {a.get('url') or a.get('path') or a.get('filename')} has no sha256")
    return bad


def requirements(path):
    bad, logical = [], ""
    for raw in open(path):
        line = raw.split(" #", 1)[0].rstrip("\n") if not raw.lstrip().startswith("#") else ""
        if line.rstrip().endswith("\\"):
            logical += line.rstrip()[:-1] + " "
            continue
        logical += line
        req, logical = logical.strip(), ""
        if not req or req.startswith("-"):
            continue  # an option line (--index-url, -r, ...)
        if not any(SHA.fullmatch(h) for h in re.findall(r"--hash[= ](\S+)", req)):
            bad.append(f"{req.split()[0]} has no --hash=sha256")
    return bad


def tracked():
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        out = [os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(ROOT) for f in fs]
    return [os.path.join(ROOT, f) for f in sorted(out)
            if os.path.basename(f) == "uv.lock" or re.fullmatch(r"requirements.*\.txt", os.path.basename(f))]


def main(argv):
    files = argv or tracked()
    failed = 0
    for f in files:
        bad = uv_lock(f) if os.path.basename(f).endswith(".lock") else requirements(f)
        for b in bad:
            print(f"{os.path.relpath(f, ROOT)}: {b}")
        failed += bool(bad)
    print(f"{len(files) - failed}/{len(files)} lock files pin every artifact by sha256")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
