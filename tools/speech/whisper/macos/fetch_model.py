"""python tools/speech/whisper/macos/fetch_model.py [--root DIR] [--verify DIR]: downloads the MLX conversion of
Whisper large-v3-turbo (mlx-community/whisper-large-v3-turbo, FP16) at the pinned revision into
ROOT/whisper-large-v3-turbo-mlx/<revision>/ and checks every file's sha256. Files already present with the right digest
are kept; anything else is fetched again. --verify DIR only checks that DIR holds exactly these files with these
digests (no network; exit 1 otherwise). Stdlib only.
Model license: MIT (OpenAI's weights; the conversion card declares none of its own)."""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO = "mlx-community/whisper-large-v3-turbo"
REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"
LICENSE = "mit"
DEFAULT_ROOT = os.path.expanduser("~/.cache/localrouter-speech/models")

# name: (bytes, sha256), from the files at REVISION (weights.safetensors: its LFS oid). mlx_whisper reads only these
# two; its tokenizer (tiktoken) ships inside the pinned package.
MANIFEST = {
    "config.json": (268, "b34fc29e4e11e0a25e812775dd67f4dd16fc2c8eb43d28ae25ff7d660ecb6379"),
    "weights.safetensors": (1613977612, "951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6"),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, dest):
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    part = dest + ".part"
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "localrouter-fetch-model/1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 22):
            h.update(chunk)
            f.write(chunk)
    return part, h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--verify", metavar="DIR", help="only check DIR against the manifest")
    args = ap.parse_args()
    if args.verify:
        files = {n: os.path.exists(os.path.join(args.verify, n)) and os.path.getsize(os.path.join(args.verify, n)) == size
                 and sha256_file(os.path.join(args.verify, n)) == digest for n, (size, digest) in MANIFEST.items()}
        extra = sorted(set(os.listdir(args.verify)) - set(MANIFEST))
        report = {"repo": REPO, "revision": REVISION, "dir": args.verify, "files": files, "extra": extra,
                  "ok": all(files.values()) and not extra}
        print(json.dumps(report, indent=1))
        sys.exit(0 if report["ok"] else 1)
    out = os.path.join(args.root, "whisper-large-v3-turbo-mlx", REVISION)
    os.makedirs(out, exist_ok=True)
    report = {"repo": REPO, "revision": REVISION, "license": LICENSE, "dir": out, "files": {}}
    ok = True
    for name, (size, digest) in MANIFEST.items():
        dest = os.path.join(out, name)
        if os.path.exists(dest) and os.path.getsize(dest) == size and sha256_file(dest) == digest:
            report["files"][name] = {"sha256": digest, "fetched": False, "ok": True}
            continue
        print(f"fetching {name} ({size} bytes)", file=sys.stderr, flush=True)
        part, got = fetch(name, dest)
        good = got == digest and os.path.getsize(part) == size
        if good:
            os.replace(part, dest)
        else:
            os.remove(part)
            ok = False
        report["files"][name] = {"sha256": got, "fetched": True, "ok": good}
    extra = sorted(set(os.listdir(out)) - set(MANIFEST))
    report["extra"] = extra
    report["ok"] = ok and not extra
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
