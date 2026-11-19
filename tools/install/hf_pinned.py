#!/usr/bin/env python3
"""Hugging Face downloads pinned by revision and checked file by file (standard library only). A manifest is
{"repo", "revision", "files": {"path": [bytes, "sha256"], ...}}, the files at one commit of the repo.
  hf_pinned.py download MANIFEST DIR [--hf CMD]  `CMD download REPO FILES... --revision REV --local-dir DIR` (hf resumes;
                                                 CMD default "hf"), then verify
  hf_pinned.py verify MANIFEST DIR [--sizes]     every manifest file in DIR has its size and sha256 (--sizes: its size
                                                 only, the cheap check of a tree a full verify already passed)
  hf_pinned.py files MANIFEST                    the manifest's paths, one a line
  hf_pinned.py pin REPO REVISION [GLOB...]       writes a manifest for the files at REVISION matching GLOB (all when
                                                 none): sizes and sha256 from the Hub's LFS records, small files hashed
                                                 here. Needs the network; the others do not, except download
Exit 1 with one line per file that is missing or differs."""
import fnmatch
import hashlib
import json
import os
import shlex
import subprocess
import sys
import urllib.parse
import urllib.request

HUB = "https://huggingface.co"


def load(path):
    with open(path) as f:
        m = json.load(f)
    if len(m.get("revision", "")) != 40 or not m.get("files"):
        raise SystemExit(f"hf_pinned: {path} needs a 40-hex revision and files")
    return m


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def verify(m, root, sizes_only=False):
    bad = []
    for name, (size, sha) in sorted(m["files"].items()):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            bad.append(f"{name}: missing")
        elif os.path.getsize(p) != size:
            bad.append(f"{name}: {os.path.getsize(p)} bytes, the pin says {size}")
        elif not sizes_only and sha256_file(p) != sha:
            bad.append(f"{name}: sha256 differs from the pin {sha}")
    return bad


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "localrouter-hf-pinned/1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def pin(repo, rev, globs):
    tree = json.loads(get(f"{HUB}/api/models/{repo}/tree/{rev}?recursive=true"))
    files = {}
    for e in tree:
        if e.get("type") != "file" or (globs and not any(fnmatch.fnmatch(e["path"], g) for g in globs)):
            continue
        if e.get("lfs"):
            files[e["path"]] = [e["lfs"]["size"], e["lfs"]["oid"]]
        else:
            data = get(f"{HUB}/{repo}/resolve/{rev}/{urllib.parse.quote(e['path'])}")
            files[e["path"]] = [len(data), hashlib.sha256(data).hexdigest()]
    if not files:
        raise SystemExit(f"hf_pinned: nothing in {repo}@{rev} matches {globs}")
    return {"repo": repo, "revision": rev, "files": dict(sorted(files.items()))}


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "pin":
        print(json.dumps(pin(args[0], args[1], args[2:]), indent=1))
        return 0
    m = load(args[0])
    if cmd == "files":
        print("\n".join(sorted(m["files"])))
        return 0
    root = args[1]
    if cmd == "download":
        hf = shlex.split(args[args.index("--hf") + 1]) if "--hf" in args else ["hf"]
        r = subprocess.run([*hf, "download", m["repo"], *sorted(m["files"]), "--revision", m["revision"], "--local-dir", root],
                           stdout=subprocess.DEVNULL)
        if r.returncode:
            print(f"hf_pinned: hf download {m['repo']}@{m['revision'][:12]} failed ({r.returncode}); run again (it resumes)", file=sys.stderr)
            return 1
    elif cmd != "verify":
        print(__doc__)
        return 2
    bad = verify(m, root, sizes_only="--sizes" in args)
    for b in bad:
        print(f"hf_pinned: {m['repo']}@{m['revision'][:12]} in {root}: {b}", file=sys.stderr)
    if not bad and cmd == "download":
        print(f"{m['repo']}@{m['revision'][:12]}: {len(m['files'])} file(s) verified in {root}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
