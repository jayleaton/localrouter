#!/usr/bin/env python3
"""The digests that name an image archive (`docker save | zstd`), read from the archive itself:
    python3 -I tools/package/image_digests.py FILE.tar.zst   ->   one JSON object on stdout
{"manifest_digest", "config_digest", "platform", "repo_tags"}. A Docker daemon names a loaded image by one of the first
two: with the containerd image store its id is the manifest's digest, with the classic store the config's. Both blobs
are hashed as they stream past, so a digest is printed only if the archive's own bytes have it. Exits 1 if the archive
does not hold exactly one image manifest, or a blob does not match its name."""
import hashlib
import json
import subprocess
import sys
import tarfile


def main(path):
    small = {}  # every blob small enough to be a manifest or a config, by its name's digest
    names = {}
    # on stdin: zstd skips a file argument that is a symbolic link
    with open(path, "rb") as f, subprocess.Popen(["zstd", "-dcq"], stdin=f, stdout=subprocess.PIPE) as z:
        with tarfile.open(fileobj=z.stdout, mode="r|") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                if m.name in ("index.json", "manifest.json"):
                    names[m.name] = json.load(tar.extractfile(m))
                elif m.name.startswith("blobs/sha256/") and m.size <= 1 << 20:
                    data = tar.extractfile(m).read()
                    want = m.name.rsplit("/", 1)[1]
                    if hashlib.sha256(data).hexdigest() != want:
                        sys.exit("image_digests: %s does not hash to its name" % m.name)
                    small["sha256:" + want] = data
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
    print(json.dumps({"manifest_digest": entries[0]["digest"], "config_digest": config_digest,
                      "platform": "%s/%s" % (config["os"], config["architecture"]), "repo_tags": tags}))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
