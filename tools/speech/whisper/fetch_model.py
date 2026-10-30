"""python tools/speech/whisper/fetch_model.py [--root DIR] [--verify DIR]: downloads openai/whisper-large-v3-turbo at the pinned
revision (only the files transformers reads) into ROOT/whisper-large-v3-turbo/<revision>/ and checks every file's
sha256. Files already present with the right digest are kept; anything else is fetched again. --verify DIR only checks
that DIR holds exactly these files with these digests (no network; exit 1 otherwise). Stdlib only.
Model license: MIT."""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO = "openai/whisper-large-v3-turbo"
REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
LICENSE = "mit"
DEFAULT_ROOT = os.path.expanduser("~/.cache/localrouter-speech/models")

# name: (bytes, sha256), from the files at REVISION (model.safetensors: its LFS oid).
MANIFEST = {
    "config.json": (1256, "c5b526b3e3cd64cd8940dabb45e8ba726629e22d8ed389c29b552f9140daf04a"),
    "generation_config.json": (3772, "cce11bfe3aaa6ae9e072ea2637caaec8795e68d9b67e655a5af16ee509681a4c"),
    "preprocessor_config.json": (340, "7ccc62c6f2765af1f3b46c00c9b5894426835a05021c8b9c01eecb6dfb542711"),
    "model.safetensors": (1617824864, "542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1"),
    "tokenizer.json": (2710337, "297b13372ac43916285644fb9687add3cc62ee2a1adb60da3dc25cc94c1871fd"),
    "tokenizer_config.json": (282843, "844b642c73a91359722f47b35705f7174686df33d252695d8572cf9ac03a6389"),
    "vocab.json": (1036558, "e2aa043ef015641d363d8288e7c241c85e36a5c761fb303598e0710233344387"),
    "merges.txt": (493869, "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6"),
    "normalizer.json": (52666, "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd"),
    "added_tokens.json": (34648, "3c51f66c4c21f9e126970078f11ae77a78c74aee8df606ee9daba86e467108e0"),
    "special_tokens_map.json": (2186, "baea4ea09372eb4fca86b4e4346139fd73cb807d5087e9de0948e971739c3e74"),
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
    out = os.path.join(args.root, "whisper-large-v3-turbo", REVISION)
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
