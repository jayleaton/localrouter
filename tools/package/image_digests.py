#!/usr/bin/env python3
"""The digests that name an image archive (`docker save | zstd`), read from the archive itself:
    python3 -I tools/package/image_digests.py [--receipt RECEIPT.json] FILE.tar.zst   ->   one JSON object on stdout
{"manifest_digest", "config_digest", "platform", "repo_tags"}. A Docker daemon names a loaded image by one of the first
two: with the containerd image store its id is the manifest's digest, with the classic store the config's. Both blobs
are hashed as they stream past, so a digest is printed only if the archive's own bytes have it. Exits 1 if the archive
does not hold exactly one image manifest, or a blob does not match its name.
With --receipt (build-image.sh's), the file is checked against it, failing closed: its SHA-256 (and size, when the
receipt has it) must be the receipt's, the receipt's image_id must be one of the two digests, and each of
manifest_digest, config_digest and platform the receipt names must equal the archive's. Older receipts have only
image_id; the output's "receipt" says which fields were compared and which the receipt lacks. Exits 1 on a mismatch."""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile


class _Hashing(io.RawIOBase):
    """A stream that hashes what is read from it."""

    def __init__(self, f):
        self.f, self.h = f, hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, b):
        n = self.f.readinto(b)
        self.h.update(memoryview(b)[:n])
        return n

    def drain(self):
        while self.read(1 << 20):
            pass
        return self.h.hexdigest()


def read(path, layer=None):
    """The archive's digests, as main prints them. With `layer`, each layer of the image is also opened as a tar
    stream and passed with its digest, in the manifest's order of layers: layer(digest, tar). Every layer is hashed as
    it is read, and must match its name."""
    small = {}  # every blob small enough to be a manifest or a config, by its name's digest
    names = {}
    seen = {}  # what layer() returned for each large blob, by digest; ordered by the manifest at the end
    # on stdin: zstd skips a file argument that is a symbolic link
    with open(path, "rb") as f, subprocess.Popen(["zstd", "-dcq"], stdin=f, stdout=subprocess.PIPE) as z:
        with tarfile.open(fileobj=z.stdout, mode="r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                if m.name in ("index.json", "manifest.json"):
                    names[m.name] = json.load(tar.extractfile(m))
                elif not m.name.startswith("blobs/sha256/"):
                    continue
                elif m.size <= 1 << 20:
                    data = tar.extractfile(m).read()
                    want = m.name.rsplit("/", 1)[1]
                    if hashlib.sha256(data).hexdigest() != want:
                        sys.exit("image_digests: %s does not hash to its name" % m.name)
                    small["sha256:" + want] = data
                elif layer:
                    want = m.name.rsplit("/", 1)[1]
                    seen["sha256:" + want] = _layer(layer, "sha256:" + want, tar.extractfile(m))
    if z.returncode != 0:
        sys.exit("image_digests: zstd could not read %s" % path)
    entries = [e for e in names.get("index.json", {}).get("manifests", [])
               if e.get("mediaType") in ("application/vnd.oci.image.manifest.v1+json",
                                         "application/vnd.docker.distribution.manifest.v2+json")]
    if len(entries) != 1 or entries[0]["digest"] not in small:
        sys.exit("image_digests: %s does not hold exactly one image manifest" % path)
    manifest = json.loads(small[entries[0]["digest"]])
    config_digest = manifest["config"]["digest"]
    if config_digest not in small:
        sys.exit("image_digests: the manifest's config %s is not in the archive" % config_digest)
    config = json.loads(small[config_digest])
    tags = [t for x in names.get("manifest.json", []) for t in x.get("RepoTags") or []]
    out = {"manifest_digest": entries[0]["digest"], "config_digest": config_digest,
           "platform": "%s/%s" % (config["os"], config["architecture"]), "repo_tags": tags}
    if layer:
        layers, order = [], []
        for x in manifest["layers"]:
            d = x["digest"]
            if d in small:  # a layer small enough to have been kept as bytes
                seen[d] = _layer(layer, d, io.BytesIO(small[d]))
            if d not in seen:
                sys.exit("image_digests: the manifest's layer %s is not in the archive" % d)
            layers.append(seen[d])
            order.append(d)
        out["layers"], out["layer_digests"] = layers, order
    return out


def _layer(layer, digest, f):
    h = _Hashing(f)
    with tarfile.open(fileobj=io.BufferedReader(h), mode="r|*") as t:
        result = layer(digest, t)
    if "sha256:" + h.drain() != digest:
        sys.exit("image_digests: the layer %s does not hash to its name" % digest)
    return result


def check_receipt(path, receipt, out):
    """Compares the archive (its digests in `out`) with its build receipt; exits 1 on any difference."""
    r = json.load(open(receipt))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    bad, checked = [], ["file_sha256", "image_id"]
    if r.get("file_sha256") != h.hexdigest():
        bad.append("file_sha256: the file is %s, the receipt says %s" % (h.hexdigest(), r.get("file_sha256")))
    if r.get("image_id") not in (out["manifest_digest"], out["config_digest"]):
        bad.append("image_id %s is neither the manifest %s nor the config %s"
                   % (r.get("image_id"), out["manifest_digest"], out["config_digest"]))
    for k, actual in (("file_bytes", os.path.getsize(path)), ("manifest_digest", out["manifest_digest"]),
                      ("config_digest", out["config_digest"]), ("platform", out["platform"])):
        if k in r:
            checked.append(k)
            if r[k] != actual:
                bad.append("%s: the archive has %s, the receipt says %s" % (k, actual, r[k]))
    if bad:
        sys.exit("image_digests: %s does not match %s:\n  %s" % (path, receipt, "\n  ".join(bad)))
    return {"file": receipt, "checked": checked,
            "absent": [k for k in ("file_bytes", "manifest_digest", "config_digest", "platform") if k not in r]}


if __name__ == "__main__":
    args = sys.argv[1:]
    receipt = None
    if args[:1] == ["--receipt"] and len(args) == 3:
        receipt, args = args[1], args[2:]
    if len(args) != 1:
        sys.exit(__doc__)
    out = read(args[0])
    if receipt:
        out["receipt"] = check_receipt(args[0], receipt, out)
    print(json.dumps(out))
