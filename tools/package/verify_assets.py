#!/usr/bin/env python3
"""Checks a model's weight directory against its pinned asset manifest, before anything loads it.

    python3 -I tools/package/verify_assets.py MANIFEST WEIGHTS_DIR [--quick] [--receipt FILE]

MANIFEST is a JSON object whose `selected_files` list names each file the model needs: `path` (relative to WEIGHTS_DIR),
`size_bytes`, and a digest, `lfs_sha256` / `sha256` (SHA-256 of the content) or, for a small file kept in git,
`git_blob_sha1` (git's blob hash). Every listed file must be present with that size and digest; `--quick` checks
presence and sizes only. A missing, short, or mismatched file fails the check: a partial or wrong download never reaches
the engine. Files in WEIGHTS_DIR that the manifest does not list are reported, not refused. Exit status 0 when all match.
"""

import argparse
import hashlib
import json
import os
import sys
import time

CHUNK = 16 << 20


def digest(path, kind, size):
    h = hashlib.sha256() if kind != "git_blob_sha1" else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb", buffering=0) as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("manifest")
    p.add_argument("weights")
    p.add_argument("--quick", action="store_true", help="presence and sizes only, no digests")
    p.add_argument("--receipt", help="write what was checked, and the result, as JSON")
    a = p.parse_args()
    m = json.load(open(a.manifest))
    files = m["selected_files"]
    root = os.path.realpath(a.weights)
    results, bad, t0 = [], 0, time.monotonic()
    total = sum(f["size_bytes"] for f in files)
    done = 0
    for f in files:
        path = os.path.join(root, f["path"])
        kind = next((k for k in ("lfs_sha256", "sha256", "git_blob_sha1") if f.get(k)), None)
        r = {"path": f["path"], "size_bytes": f["size_bytes"]}
        if not os.path.realpath(path).startswith(root + os.sep):
            # e.g. a Hugging Face cache snapshot's links into ../../blobs: not visible where only this directory is mounted
            r["error"] = "a link to outside the weights directory"
        elif not os.path.isfile(path):
            r["error"] = "missing"
        elif (size := os.path.getsize(path)) != f["size_bytes"]:
            r["error"] = f"size {size}, want {f['size_bytes']}"
        elif kind is None:
            r["error"] = "the manifest has no digest for it"
        elif not a.quick:
            got = digest(path, kind, f["size_bytes"])
            r[kind] = got
            if got != f[kind]:
                r["error"] = f"{kind} {got}, want {f[kind]}"
        done += f["size_bytes"]
        bad += "error" in r
        results.append(r)
        print(f"{'FAIL' if 'error' in r else 'ok  '} {f['path']}" + (f": {r['error']}" if "error" in r else ""), flush=True)
        if not a.quick and f["size_bytes"] > 1 << 30:
            print(f"     {done / total:.0%} of {total / 2**30:.1f} GiB, {time.monotonic() - t0:.0f} s", flush=True)
    listed = {f["path"] for f in files}
    extra = sorted(os.path.relpath(os.path.join(d, n), root) for d, _, names in os.walk(root) for n in names
                   if os.path.relpath(os.path.join(d, n), root) not in listed)
    for e in extra:
        print(f"note {e}: not in the manifest")
    summary = {"manifest": os.path.abspath(a.manifest), "repo_id": m.get("repo_id"), "revision": m.get("revision"),
               "weights": root, "mode": "quick" if a.quick else "digest", "files": len(files), "bytes": total,
               "failed": bad, "unlisted": extra, "seconds": round(time.monotonic() - t0, 1), "results": results,
               "result": "fail" if bad else "pass"}
    if a.receipt:
        with open(a.receipt, "w") as out:
            json.dump(summary, out, indent=1)
    print(f"{summary['result']}: {len(files) - bad}/{len(files)} files match ({total / 2**30:.1f} GiB, {summary['mode']})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
