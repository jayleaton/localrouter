#!/usr/bin/env python3
"""What an image build from one commit consumes, and the licence conditions on what it ships:
    python3 -I tools/package/build_fingerprint.py COMMIT [--base BASE] [--image FILE.tar.zst] [--manifest FILE]
                                                [--platform P] [--notices FILE] [--no-docker]
One JSON object on stdout, every value read from the commit, the image and the local Docker store, nothing assumed:
  source       the commit, its tree and base, and the git object of each path docker/Dockerfile COPYs into the build
  dockerfile   its SHA-256, ARG defaults, FROM lines and nvcc targets; each build-time download and apt install
  dependencies build.zig.zon's packages (their content hash is pinned: `zig build --fetch` refuses any other bytes)
  base_images  each FROM image as the local Docker store has it (id and repo digests), or null when it is absent
  licences     LICENSE's title, NOTICE's SHA-256, and whether NOTICE names each dependency
  image        with --image: its digests, and every Debian package in it with version, source and the licences its
               copyright file declares (layers read in order, each hashed against its name)
  model        with --manifest: the weights' repository, revision, file count and bytes (not in the image)
  conditions   what the fingerprint cannot pin, and what shipping the image obliges: downloads without a checksum,
               unpinned packages, GPL-licensed packages (the image must come with a source offer), NVIDIA EULA
               packages, a dependency NOTICE does not name
Run it again whenever the candidate head changes: nothing in it carries over from another commit."""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_digests  # noqa: E402


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def blob(commit, path):
    r = subprocess.run(["git", "cat-file", "blob", "%s:%s" % (commit, path)], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def sha(data):
    return hashlib.sha256(data).hexdigest() if data is not None else None


def dockerfile(text):
    """ARG defaults, FROM lines, COPYs from the context (not --from), downloads, apt installs and the lock files apt
    installs from (`xargs -a FILE apt-get install`), each with its stage; RUN continuations joined."""
    lines = re.sub(r"\\\n", " ", text).splitlines()
    args, froms, copies, fetches, apt, locks, sm, stage = {}, [], [], [], [], [], None, None
    for line in (x.strip() for x in lines):
        word, _, rest = line.partition(" ")
        word = word.upper()
        if word == "ARG" and "=" in rest:
            k, _, v = rest.partition("=")
            args.setdefault(k.strip(), v.strip())
        elif word == "FROM":
            froms.append(rest)
            m = re.search(r"\bAS\s+(\S+)", rest, re.I)
            stage = m.group(1) if m else "final"
        elif word == "COPY" and "--from" not in rest:
            words = [p for p in rest.split() if not p.startswith("--")]
            copies += [{"src": p, "dest": words[-1], "stage": stage} for p in words[:-1]]
        elif word == "RUN":
            for url in re.findall(r"https?://[^\s\"']+", rest):
                fetches.append({"url": url, "checksum_in_run": bool(re.search(r"sha256sum|--checksum|[0-9a-f]{64}", rest))})
            locks += [{"dest": f, "stage": stage} for f in re.findall(r"xargs\s+-a\s+(\S+)\s+apt-get install", rest)]
            for pkgs in re.findall(r"apt-get install ([^&;|]+)", rest):
                apt += [{"package": p, "stage": stage} for p in pkgs.split() if not p.startswith("-")]
            m = re.search(r"-Dsm=([0-9,]+)", rest)
            sm = m.group(1) if m else sm
    return {"args": args, "from": froms, "copies": copies, "fetches": fetches, "apt": apt, "locks": locks, "nvcc_sm": sm}


def expand(s, args):
    return re.sub(r"\$\{?(\w+)\}?", lambda m: args.get(m.group(1), m.group(0)), s)


def docker_cli():
    """The Docker CLI, unless --no-docker (or none is installed): then nothing here talks to a Docker daemon."""
    return None if NO_DOCKER else shutil.which("docker")


NO_DOCKER = False


def local_image(ref, platform):
    """The image as the local store has it for one platform: an index's id names every platform, so the platform's
    own manifest is what a build stage gets. None without a Docker CLI too."""
    if docker_cli() is None:
        return None
    r = subprocess.run(["docker", "image", "inspect", "--platform", platform, "--format", "{{json .}}", ref],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    i = json.loads(r.stdout)
    return {"id": i["Id"], "repo_digests": i.get("RepoDigests") or [], "platform": "%s/%s" % (i["Os"], i["Architecture"]),
            "platform_manifest": (i.get("Descriptor") or {}).get("digest")}


def zon_dependencies(text):
    return [{"name": n, "url": u, "hash": h} for n, u, h in
            re.findall(r"\.(\w+)\s*=\s*\.\{\s*\.url\s*=\s*\"([^\"]+)\",\s*\.hash\s*=\s*\"([^\"]+)\"", text)]


def debian_packages(path):
    """Every package in the image's dpkg status, with the licence names its copyright file declares (DEP-5 License
    fields; a copyright file in another format declares none we can read)."""
    def layer(digest, t):  # this layer's changes to the files we read: content, or None where it deletes one
        changes = {}
        for m in t:
            name = m.name[2:] if m.name.startswith("./") else m.name
            d, base = os.path.split(name)
            if base == ".wh..wh..opq":  # an opaque directory: what lower layers had in it is hidden
                changes[(d + "/", "opaque")] = None
            elif base.startswith(".wh."):
                changes[os.path.join(d, base[4:])] = None
            elif m.isfile() and (name == "var/lib/dpkg/status" or re.fullmatch(r"usr/share/doc/[^/]+/copyright", name)):
                changes[name] = t.extractfile(m).read().decode("utf-8", "replace")
        return changes

    digests = image_digests.read(path, layer)
    files = {}
    for changes in digests["layers"]:  # in the manifest's order: a later layer replaces an earlier one's files
        for k, v in changes.items():
            if isinstance(k, tuple):
                files = {n: x for n, x in files.items() if not n.startswith(k[0])}
            elif v is None:
                files = {n: x for n, x in files.items() if n != k and not n.startswith(k + "/")}
            else:
                files[k] = v
    status = files.get("var/lib/dpkg/status", "")
    copyright = {n: x for n, x in files.items() if n.endswith("/copyright")}
    packages = []
    for para in status.split("\n\n"):
        f = dict(re.findall(r"^([\w-]+): (.*)$", para, re.M))
        if "Package" not in f or "installed" not in f.get("Status", ""):
            continue
        text = copyright.get("usr/share/doc/%s/copyright" % f["Package"])
        licences = sorted({x.strip() for x in re.findall(r"^License: *([^\n]+)", text or "", re.M)})
        packages.append({"package": f["Package"], "version": f.get("Version"), "source": f.get("Source", f["Package"]).split()[0],
                         "copyright_file": text is not None, "licences": licences})
    return digests, packages


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("commit")
    p.add_argument("--base", help="default: the merge base of COMMIT and origin/main")
    p.add_argument("--image")
    p.add_argument("--manifest")
    p.add_argument("--platform", default="linux/arm64", help="the image's target platform (default linux/arm64)")
    p.add_argument("--notices", metavar="FILE", help="with --image: also write the image's third-party package list there")
    p.add_argument("--no-docker", action="store_true",
                   help="query no Docker daemon: base images are reported as not looked up (the image file is read directly)")
    a = p.parse_args()
    global NO_DOCKER
    NO_DOCKER = a.no_docker
    commit = git("rev-parse", "--verify", a.commit + "^{commit}").strip()
    base = git("rev-parse", "--verify", (a.base or git("merge-base", commit, "origin/main").strip()) + "^{commit}").strip()
    df_bytes = blob(commit, "docker/Dockerfile")
    if df_bytes is None:
        sys.exit("build_fingerprint: %s has no docker/Dockerfile" % commit)
    df = dockerfile(df_bytes.decode())
    zon = blob(commit, "build.zig.zon")
    deps = zon_dependencies(zon.decode()) if zon else []
    notice = blob(commit, "NOTICE")
    licence = blob(commit, "LICENSE")
    conditions = []

    host = "unknown"
    if docker_cli():
        host = subprocess.run(["docker", "version", "--format", "{{.Server.Os}}/{{.Server.Arch}}"], capture_output=True,
                              text=True).stdout.strip() or host
    # the build's own arguments: a COPY may name a file per target architecture
    vars_ = dict(df["args"], TARGETPLATFORM=a.platform, TARGETARCH=a.platform.split("/")[1], BUILDPLATFORM=host,
                 BUILDARCH=host.split("/")[-1])
    closure = {}
    for c in df["copies"]:
        c["src"] = expand(c["src"], vars_)
        r = subprocess.run(["git", "rev-parse", "%s:%s" % (commit, c["src"].rstrip("/"))], capture_output=True, text=True)
        closure[c["src"]] = r.stdout.strip() if r.returncode == 0 else None
        if r.returncode != 0:
            conditions.append("COPY %s: not in the commit" % c["src"])
    locks = []
    for lk in df["locks"]:
        src = [c["src"] for c in df["copies"] if c["stage"] == lk["stage"] and c["dest"] == lk["dest"]]
        data = blob(commit, src[0]) if len(src) == 1 else None
        if data is None:
            conditions.append("apt installs from %s in the %s stage, which is not one file of the commit" % (lk["dest"], lk["stage"]))
            continue
        entries = dict(x.split("=", 1) for x in data.decode().split() if "=" in x)
        locks.append({"file": src[0], "stage": lk["stage"], "sha256": sha(data), "packages": len(entries), "entries": entries})

    bases = {}
    for line in df["from"]:
        ref = expand(re.sub(r"--platform=\S+\s+", "", line).split()[0], df["args"])
        plat = re.search(r"--platform=(\S+)", line)
        plat = expand(plat.group(1), {"BUILDPLATFORM": host, "TARGETPLATFORM": a.platform}) if plat else a.platform
        bases[ref] = {"from": line, "platform": plat, "pinned": "@sha256:" in ref, "local": local_image(ref, plat)}
        if "@sha256:" not in ref:
            conditions.append("base image %s is not pinned by digest: a later build can get other bytes" % ref)
        if bases[ref]["local"] is None and not NO_DOCKER:
            conditions.append("base image %s for %s is not in the local Docker store: pull it before the build" % (ref, plat))
    for f in df["fetches"]:
        f["url"] = expand(f["url"], df["args"])
        if not f["checksum_in_run"]:
            conditions.append("downloaded at build time without a checksum: %s" % f["url"])
    for stage in sorted({x["stage"] for x in df["apt"]}):
        loose = [x["package"] for x in df["apt"] if x["stage"] == stage and "=" not in x["package"]]
        if loose and stage == "final":
            conditions.append("apt packages shipped in the image are not pinned to versions (%s): the image's own "
                              "inventory (--image) records what it got" % ", ".join(loose))
        elif loose:
            conditions.append("apt packages in the %s stage are not pinned to versions (%s): they do not ship, but "
                              "the build's tools can differ between builds" % (stage, ", ".join(loose)))
    named = (notice or b"").decode().lower()
    for d in deps:
        d["named_in_notice"] = d["name"].lower() in named
        if not d["named_in_notice"]:
            conditions.append("dependency %s is not named in NOTICE" % d["name"])

    out = {"source": {"commit": commit, "tree": git("rev-parse", commit + "^{tree}").strip(), "base": base,
                      "subject": git("log", "-1", "--format=%s", commit).strip(), "build_closure": closure,
                      "dockerignore_sha256": sha(blob(commit, ".dockerignore"))},
           "dockerfile": {"sha256": sha(df_bytes), "args": df["args"], "from": df["from"], "nvcc_sm": df["nvcc_sm"],
                          "fetches": df["fetches"], "apt": df["apt"],
                          "apt_locks": [{k: v for k, v in x.items() if k != "entries"} for x in locks]},
           "build_zig_sha256": sha(blob(commit, "build.zig")), "build_zig_zon_sha256": sha(zon),
           "dependencies": deps, "base_images": bases, "docker_store_queried": not NO_DOCKER,
           "licences": {"LICENSE": licence.decode().strip().splitlines()[0].strip() if licence else None,
                        "NOTICE_sha256": sha(notice)}}
    if not licence:
        conditions.append("the commit has no LICENSE")

    if a.image:
        digests, packages = debian_packages(a.image)
        gpl = sorted({p["package"] for p in packages if any(re.match(r"L?GPL|AGPL", x) for x in p["licences"])})
        eula = sorted(p["package"] for p in packages if re.match(r"(cuda|libcu|libnv|nvidia)", p["package"]))
        out["image"] = {"file": os.path.basename(a.image), "file_sha256": sha_file(a.image),
                        "manifest_digest": digests["manifest_digest"], "config_digest": digests["config_digest"],
                        "platform": digests["platform"], "layers": digests["layer_digests"], "packages": packages}
        installed = {p["package"]: p["version"] for p in packages}
        for lk in (x for x in locks if x["stage"] == "final"):
            off = sorted("%s=%s (image: %s)" % (k, v, installed.get(k, "absent")) for k, v in lk["entries"].items()
                         if installed.get(k) != v)
            out["image"]["lock_check"] = {"file": lk["file"], "packages": lk["packages"], "mismatches": off}
            if off:
                conditions.append("the image's packages differ from %s at %d entries, e.g. %s" % (lk["file"], len(off), ", ".join(off[:4])))
        if gpl:
            conditions.append("the image ships packages under (L)GPL licences (%d, e.g. %s): distribute it with their "
                              "corresponding source or a written offer" % (len(gpl), ", ".join(gpl[:6])))
        if eula:
            conditions.append("the image ships NVIDIA packages under the CUDA EULA (%s): redistributable only as the "
                              "base image's terms allow" % ", ".join(eula))
        unread = [p["package"] for p in packages if not p["licences"]]
        if unread:
            conditions.append("%d packages declare no machine-readable licence (their copyright file is missing or not "
                              "DEP-5; read it in the image), e.g. %s" % (len(unread), ", ".join(unread[:8])))
    if a.manifest:
        m = json.load(open(a.manifest))
        out["model"] = {"manifest_sha256": sha_file(a.manifest), "repo_id": m.get("repo_id"), "revision": m.get("revision"),
                        "files": len(m.get("selected_files") or []),
                        "bytes": sum(f.get("size_bytes") or 0 for f in m.get("selected_files") or []),
                        "notice_inputs": m.get("reference_notice_inputs")}
        conditions.append("the weights (%s) are not in the image: their own licence applies where they are installed"
                          % m.get("repo_id"))
    out["conditions"] = conditions
    if a.notices and a.image:
        with open(a.notices, "w") as f:
            f.write(notices(out))
    json.dump(out, sys.stdout, indent=1)
    print()


def notices(fp):
    """The image's Debian packages by source package, with versions and declared licences: the list a distributor
    of the image works from (it is not itself a source offer)."""
    i = fp["image"]
    by_source = {}
    for p in i["packages"]:
        by_source.setdefault(p["source"], []).append(p)
    lines = ["Third-party packages in %s (manifest %s, platform %s)," % (i["file"], i["manifest_digest"], i["platform"]),
             "built from %s. Read from the image's own dpkg database and /usr/share/doc/*/copyright files." % fp["source"]["commit"],
             "Licences are the DEP-5 License fields each package declares; '(see copyright file)' where it declares none.",
             "LocalRouter's own licence and its bundled code: LICENSE and NOTICE in the source tree.", ""]
    for src in sorted(by_source):
        lines.append("%s" % src)
        for p in sorted(by_source[src], key=lambda x: x["package"]):
            lines.append("  %s %s: %s" % (p["package"], p["version"], ", ".join(p["licences"]) or "(see copyright file)"))
    lines += ["", "Conditions:"] + ["- " + c for c in fp["conditions"]]
    return "\n".join(lines) + "\n"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit("build_fingerprint: %s: %s" % (" ".join(e.cmd), (e.stderr or "").strip()))
