"""The D2 (Kandinsky-6 video VAE) jobs as an operator spec for the shared staging sealer
(tools/kandinsky6/staging/seal.py), which hard-links the artifacts into each node's kit, seals them in SHA256SUMS and
jobs.json, and runs them under run-node.py (immutable image ID, --memory/--memory-swap, read-only root and kit, models at
/models, K6_OUTPUT, window deadlines, memory / GPU / disk watch).

  python -I operator_spec.py --kit DIR --node-images FILE --assets progress.json > spec.json

Per job: the node's own image ID from FILE (`docker image inspect` on each node: the same repo digest has a different
local ID per node), the command inside the container (the kit is mounted at /ws/kit), budgets that the runner enforces.
The oracle jobs run the pinned native stack from this component's own checked bundle (pybundle.sh), copied unchanged to
each node and mounted read-only at /opt/k6-oracle-d2: its identity (manifest, requirements, versions) is pinned below,
must match the kit's pybundle/, and is what the runner admits (layout python/bin/python3.13 + site-packages).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GIB = 1024**3
WORKSPACE = "~/k6-d2-qual"
# the original D2 bundle, as the runner admits it (kind d2-pybundle): node path, identity, layout, container mount
ORACLE = {"kind": "d2-pybundle", "root": "~/k6-d2-oracle-ead020cbc102/pybundle",
          "manifest_sha256": "ead020cbc102a5cb10fdbef43f85dfa63aa14585d1835ce0b7f3bc686eac6d1d",
          "requirements_sha256": "6bb09d766dcb16babbf6bae448b0998cf0afaa053b6fa656d1520f0a37ccef0f",
          "interpreter_relative_path": "python/bin/python3.13", "site_packages_relative_path": "site-packages",
          "container_root": "/opt/k6-oracle-d2", "readonly": True,
          "versions": {"python": "3.13.16", "torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130",
                       "diffusers": "0.40.0", "numpy": "2.5.3", "accelerate": "1.15.0", "safetensors": "0.8.0"}}
ASSETS = ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
ARTIFACT_DIRS = ["bin", "fatbin", "tools", "inputs", "upstream"]
ARTIFACT_FILES = ["build.json", "MANIFEST.sha256"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def artifacts(kit: Path) -> list:
    files = [p for d in ARTIFACT_DIRS for p in sorted((kit / d).rglob("*"))
             if p.is_file() and ".git" not in p.relative_to(kit).parts and p.suffix != ".pyc"]
    files += [kit / f for f in ARTIFACT_FILES]
    return [{"source": str(p), "destination": str(p.relative_to(kit)), "sha256": sha256(p)} for p in files]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--node-images", required=True, type=Path)
    ap.add_argument("--assets", required=True, type=Path)
    a = ap.parse_args()
    kit = a.kit.resolve()
    for name, key in (("MANIFEST.sha256", "manifest_sha256"), ("requirements.txt", "requirements_sha256")):
        got = sha256(kit / "pybundle" / name)
        assert got == ORACLE[key], f"kit pybundle/{name} is {got}, not the pinned original bundle {ORACLE[key]}"
    inspect = json.loads(a.node_images.read_text())
    images = {node: line.split()[0] for node, line in inspect.items()}
    for node, image in images.items():
        assert image.startswith("sha256:") and len(image) == 71, f"{node}: not an image ID: {image}"
    progress = {f["path"]: f for f in json.loads(a.assets.read_text())}
    missing = [n for n in ASSETS if n not in progress]
    assert not missing, f"assets not acquired: {missing}"
    arts = artifacts(kit)
    common = {"workspace": WORKSPACE, "container_workspace": "/ws", "artifacts": arts, "asset_paths": ASSETS,
              "node_image_receipt": {"source": str(a.node_images), "sha256": sha256(a.node_images)}}

    def oracle(job_id, node, timeout, n_dec, other):
        env = {"VAE": "/models/vae", "ORACLE_ROOT": ORACLE["container_root"],
               "ORACLE_MANIFEST_SHA256": ORACLE["manifest_sha256"], "JOB_BUDGET_S": str(timeout - 120),
               "TIMEOUT_N_DEC": str(n_dec), **{f"TIMEOUT_{c}": str(other) for c in ("S_DEC", "F_DEC", "S_ENC", "N_ENC")}}
        return {"id": job_id, "node": node, "image_id": images[node], **common,
                "command": ["/bin/bash", "/ws/kit/tools/run-oracle.sh"], "environment": env,
                "shared_oracle": ORACLE,
                "timeout_seconds": timeout,
                # runner accounting on unified memory: max(cgroup + GPU, MemAvailable drop + GPU) + 4 GiB helpers,
                # so the GPU share is counted twice: ~2 x 30 GiB device + 10 GiB host + 4 GiB, not measured
                "max_resident_bytes": 80 * GIB, "max_disk_bytes": 12_000_000_000, "output_budget_bytes": 4 * GIB}

    engine = {"id": "d2-engine", "node": "dgx02", "image_id": images["dgx02"], **common,
              "command": ["/bin/bash", "/ws/kit/tools/run-engine.sh"],
              "environment": {"VAE": "/models/vae", "TIMEOUT": "1200"}, "timeout_seconds": 1500,
              # 15.6 GB scratch + 0.5 GB weights on the device (counted twice, as above), ~5 GiB host, 4 GiB helpers
              "max_resident_bytes": 48 * GIB, "max_disk_bytes": 8_000_000_000, "output_budget_bytes": 4 * GIB}
    oracles = [oracle("d2-oracle", "dgx01", 2700, 1200, 300),         # windows of 70+ minutes
               oracle("d2-oracle-short", "dgx01", 2100, 900, 240)]   # a 60-minute window
    spec = {"schema": 1, "component": "d2-video-vae", "jobs": [engine] + oracles}
    print(json.dumps(spec, indent=1))


if __name__ == "__main__":
    main()
