"""build-kit.sh's Zig provenance, on the real builder: TOOLS_ONLY restages of a minimal kit under a PATH with no zig,
a zig that fails, prints nothing, prints a non-version, an older or a prerelease version, and a supported zig. Every
refusal must exit nonzero with the kit byte-for-byte unchanged and no operator spec written; every supported zig must
restage with build.json recording exactly that version. No GPU, nvcc or Zig build.

  python3 -I build-kit-fixture.py --pybundle DIR [--zig DIR] [--builder build-kit.sh] [--out DIR]

DIR holds the pinned original bundle's MANIFEST.sha256 and requirements.txt (operator_spec.py checks them); --zig is
the directory of the supported zig (default: the one on PATH); --builder defaults to this directory's build-kit.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
# case -> fake `zig version` (stdout, exit code), None for no zig at all; and whether the builder must accept it
CASES = [("zig-absent", None, False), ("zig-fails", ("", 3), False), ("zig-prints-nothing", ("", 0), False),
         ("zig-prints-non-version", ('0.17.0", "x": "y', 0), False), ("zig-older", ("0.16.0", 0), False),
         ("zig-prerelease-of-minimum", ("0.17.0-dev.42+abc", 0), False), ("zig-newer", ("0.17.1", 0), True)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(kit: Path) -> dict:
    """Every path under the kit with its mode, and its bytes' sha256 for files; plus the spec beside it."""
    out = {}
    for p in sorted(kit.rglob("*")):
        st = p.lstat()
        out[str(p.relative_to(kit))] = [oct(st.st_mode), sha256(p) if stat.S_ISREG(st.st_mode) else None]
    spec = kit.parent / (kit.name + ".operator-spec.json")
    out["<operator-spec>"] = sha256(spec) if spec.exists() else None
    return out


def make_kit(root: Path, pybundle: Path) -> Path:
    kit = root / "kit"
    for d in ("pybundle", "upstream", "inputs", "bin", "fatbin"):
        (kit / d).mkdir(parents=True)
    for n in ("MANIFEST.sha256", "requirements.txt"):
        shutil.copyfile(pybundle / n, kit / "pybundle" / n)
    (kit / "pybundle.json").write_text("{}\n")
    (kit / "upstream/README").write_text("fixture\n")
    (kit / "inputs/inputs.json").write_text("{}\n")
    (kit / "bin/k6v-qual").write_bytes(b"fixture binary\n")
    (kit / "fatbin/conv.fatbin").write_bytes(b"fixture conv\n")
    (kit / "fatbin/ops.fatbin").write_bytes(b"fixture ops\n")
    (kit / "build.json").write_text(json.dumps({"nvcc": "fixture nvcc"}) + "\n")
    lines = [f"{sha256(kit / n)}  {n}\n" for n in ("bin/k6v-qual", "fatbin/conv.fatbin", "fatbin/ops.fatbin")]
    (kit / "MANIFEST.sha256").write_text("".join(lines))
    return kit


def path_without_zig(root: Path) -> Path:
    """One directory of links to every command on PATH except zig."""
    farm = root / "path"
    farm.mkdir()
    for entry in os.environ["PATH"].split(os.pathsep):
        d = Path(entry)
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.name != "zig" and not (farm / p.name).exists() and os.access(p, os.X_OK) and not p.is_dir():
                (farm / p.name).symlink_to(p.resolve())
    return farm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pybundle", required=True, type=Path)
    ap.add_argument("--zig", type=Path, default=None)
    ap.add_argument("--builder", type=Path, default=HERE / "build-kit.sh")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    zig_dir = (a.zig or Path(shutil.which("zig") or "/nonexistent").parent).resolve()
    assert (zig_dir / "zig").exists(), f"no supported zig at {zig_dir}"
    real = subprocess.run([str(zig_dir / "zig"), "version"], capture_output=True, text=True, check=True).stdout.strip()
    work = Path(tempfile.mkdtemp(prefix="build-kit-fixture-"))
    results, failed = [], []
    try:
        farm = path_without_zig(work)
        images = work / "node-images.json"
        images.write_text(json.dumps({n: "sha256:" + hashlib.sha256(n.encode()).hexdigest() + " fixture"
                                      for n in ("dgx01", "dgx02")}))
        assets = work / "progress.json"
        assets.write_text(json.dumps([{"path": "vae/config.json"}, {"path": "vae/diffusion_pytorch_model.safetensors"}]))
        for name, fake, accept in CASES + [("zig-supported", "real", True)]:
            root = work / name
            root.mkdir()
            kit = make_kit(root, a.pybundle.resolve())
            path = [str(farm)]
            want = None
            if fake == "real":
                path.insert(0, str(zig_dir))
                want = real
            elif fake is not None:
                bindir = root / "zigbin"
                bindir.mkdir()
                out, code = fake
                (bindir / "zig").write_text(f"#!/bin/sh\nprintf '%s' '{out}'\nexit {code}\n")
                (bindir / "zig").chmod(0o755)
                path.insert(0, str(bindir))
                want = out
            before = snapshot(kit)
            env = {**os.environ, "PATH": os.pathsep.join(path), "TOOLS_ONLY": "1", "KIT": str(kit),
                   "NODE_IMAGES": str(images), "ASSETS_PROGRESS": str(assets), "PYTHONDONTWRITEBYTECODE": "1"}
            r = subprocess.run(["bash", str(a.builder.resolve())], env=env, capture_output=True, text=True)
            after = snapshot(kit)
            row = {"case": name, "accept": accept, "exit": r.returncode, "stderr_tail": r.stderr.strip().splitlines()[-1:]}
            if accept:
                build = json.loads((kit / "build.json").read_text())
                verified = subprocess.run(["sha256sum", "-c", "--strict", "--quiet", "MANIFEST.sha256"], cwd=kit).returncode == 0
                spec = kit.parent / "kit.operator-spec.json"
                row.update(recorded_zig=build.get("zig"), manifest_verified=verified, spec=spec.exists())
                ok = r.returncode == 0 and build.get("zig") == want and verified and spec.exists() \
                    and "window_budget" in json.loads(spec.read_text())
            else:
                row.update(kit_unchanged=before == after, spec=after["<operator-spec>"] is not None)
                if (kit / "build.json").exists():
                    row["recorded_zig"] = json.loads((kit / "build.json").read_text()).get("zig")
                ok = r.returncode != 0 and before == after
            row["ok"] = ok
            results.append(row)
            if not ok:
                failed.append(name)
    finally:
        shutil.rmtree(work)
    report = {"build_kit_fixture": "passed" if not failed else "FAILED", "builder": str(a.builder.resolve()),
              "builder_sha256": sha256(a.builder.resolve()), "supported_zig": real, "failed": failed, "cases": results}
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
