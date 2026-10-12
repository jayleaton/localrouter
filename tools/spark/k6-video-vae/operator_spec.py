"""The D2 (Kandinsky-6 video VAE) jobs as an operator spec for the shared staging sealer
(tools/kandinsky6/staging/seal.py), which hard-links the artifacts into each node's kit, seals them in SHA256SUMS and
jobs.json, and runs them under run-node.py (immutable image ID, --memory/--memory-swap as host accounting only, read-only
root and kit, models at /models, K6_OUTPUT, window deadlines, memory / GPU / disk watch, admission by process budget).

  python -I operator_spec.py --kit DIR --node-images FILE --assets progress.json > spec.json
  python -I operator_spec.py --self-test                   the window budget at its boundaries and the memory
                                                           declarations, no kit

Per job: the node's own image ID from FILE (`docker image inspect` on each node: the same repo digest has a different
local ID per node), the command inside the container (the kit is mounted at /ws/kit), budgets that the runner enforces.
The oracle jobs run the pinned native stack from this component's own checked bundle (pybundle.sh), copied unchanged to
each node and mounted read-only at /opt/k6-oracle-d2: its identity (manifest, requirements, versions) is pinned below,
must match the kit's pybundle/, and is what the runner admits (layout python/bin/python3.13 + site-packages).

Every job runs under the D2 memory policy gb10-unified-d2-v2 (D2-EVIDENCE.md): on each node its attribution probe
first, then that node's payload, admitted only with the probe's PASS from the same window (per-node gate). dgx01:
d2-probe-torch, d2-oracle-short (the oracle's sealed segments); dgx02: d2-probe-native, d2-engine. The full-length
d2-oracle is not emitted: the approved run is d2-oracle-short only. The window is itemized (window_plan()), at
most 180 minutes, and the full-producer audit runs offline at pre-stage against the exact staged tree
(stage_tree(), bound into the grant).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

GIB = 1024**3
MIB = 1024**2
POLICY = "gb10-unified-d2-v2"
SEGMENTS = ["env-probe", "s_dec", "f_dec", "s_enc", "n_enc", "n_dec"]
WORKSPACE = "~/k6-d2-qual"
# the original D2 bundle, as the runner admits it (kind d2-pybundle): node path, identity, layout, container mount
ORACLE = {"kind": "d2-pybundle", "root": "~/k6-d2-oracle-ead020cbc102/pybundle",
          "manifest_sha256": "ead020cbc102a5cb10fdbef43f85dfa63aa14585d1835ce0b7f3bc686eac6d1d",
          "requirements_sha256": "6bb09d766dcb16babbf6bae448b0998cf0afaa053b6fa656d1520f0a37ccef0f",
          "interpreter_relative_path": "python/bin/python3.13", "site_packages_relative_path": "site-packages",
          "container_root": "/opt/k6-oracle-d2", "readonly": True,
          "versions": {"python": "3.13.16", "torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130",
                       "diffusers": "0.40.0", "numpy": "2.5.3", "accelerate": "1.15.0", "safetensors": "0.8.0"}}
# the bundle's relative links, as the runner inventories them (count, sha256 of the sorted path/target list)
ORACLE_LINKS = (1048, "4c5cd1da47de32a185153d980a1e195add5302c4ac40777f63deb0d51eec24bf")
ASSETS = ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
ARTIFACT_DIRS = ["bin", "fatbin", "tools", "inputs", "upstream"]
ARTIFACT_FILES = ["build.json", "MANIFEST.sha256"]

# --- Window budget (D2-EVIDENCE.md, "Window budget"): the owner's itemized rulings as integer seconds from T0 ----------
# T0 is the grant's opened_at. No fixed node end: each node's sequence is solved forward from T0 through every itemized
# reservation, and the window is N whole minutes from the later node end plus the owner's 10-minute close, at most 180.
# The runner reads two sealed integers per job:
#   latest_start_seconds_from_open L: the job is admitted only while now < T0 + L, checked at plan time and again after
#     `docker create`, its ownership inspect and the setup-gate recheck (strictly less: a container created at T0 + L is
#     refused). The runner also caps every job at expires - 1200.
#   finish_margin_seconds F: the runner's stop is min(expires - 690, expires - F - 60), and a job starts only if
#     started + timeout_seconds <= stop (equality admitted): its payload ends by T0 + 60 N - 60 - F.
# Readiness, the event the setup budget serves, is job 1 admitted on its node (its L). Every step of a node's sequence
# is reserved below in whole seconds, reservations rounded up and deadlines down; a node that misses any deadline fails
# closed (setup abort before job 1; the runner refuses job 2), never extends.
WINDOW_MAX_S = 180 * 60
WINDOW_CLOSE_S = 600                # owner: export, cleanup and release by T0 + N - 10 min, on both nodes
RUNNER_NO_NEW_WORK_S = 1200         # run-node-d2-v5.py: launch cutoff expires - 1200
RUNNER_STOP_RESERVE_S = 690         # run-node-d2-v5.py: stop <= expires - 690
RUNNER_MARGIN_PAD_S = 60            # run-node-d2-v5.py: stop <= expires - finish_margin - 60
# Owner: dgx01 keeps at least its v6 deadlines (later where the reservations need it); at least 300 s of setup slack (the
# owner's floor for dgx02, applied to both nodes).
DEADLINE_FLOOR_S = {"dgx01": {"setup": 1267, "L1": 1416, "L2": 2245}, "dgx02": {"setup": 0, "L1": 0, "L2": 0}}
SETUP_SLACK_MIN_S = 300
# The runner's own hard bounds per job: before the payload `docker image inspect`, nvidia-smi, `docker ps`,
# `docker create` (30 s each) and the 5 s PID poll; after it `docker rm` (60 s) and two 10 s waits.
RUNNER_PRE_S = 4 * 30 + 5
RUNNER_POST_S = 60 + 2 * 10
STRICT_S = 1                        # the runner admits strictly before L
# Owner reservations for runner work that no measured row isolates, outside the bounds above: the runner's non-hash work
# before each payload (U1: marker verification, residency, evidence init, output layout, the setup-gate recheck), and
# the ownership `docker inspect` after create and the state inspects after the payload (U2), 30 s each.
RUNNER_PRELAUNCH_RESERVE_S = 60
DOCKER_INSPECT_S = 30
# U3: every cross-node marker read or owner operation, 60 s per SSH hop (a route through dgx01 to dgx02 is two), on top
# of the transfer terms. Hops per operation, per node and gap (run-node-d2-v5, input-adapter-v4, d2-control-files-v1,
# coordinate-d2-v3, d2-export-remote-v1 and the DEV flow's admitted operations). Each owner-marker read is counted as
# cross-node on both nodes (the runner's injected reader is the only admitted transport for dgx02).
CROSS_NODE_OP_S = 60
CROSS_NODE_OPS = {
    "dgx01": {
        "input": [("input create", 1), ("input guard", 1), ("kit copy devbox -> dgx01", 1), ("input verify", 1),
                  ("control files", 1), ("setup dispatch", 1)],
        "prelaunch": [("runner dispatch", 1), ("marker read: admission", 1), ("marker read: record", 1),
                      ("marker read: before create", 1), ("marker read: after create", 1)],
        "post_payload": [("marker read: before cleanup", 1)],
        "export": [("output copy dgx01 -> devbox", 1)],
        "cleanup": [("own container and output cleanup", 1)],
        "marker": [("marker check", 1), ("release probe marker", 1), ("release payload marker", 1)],
    },
    "dgx02": {
        "input": [("relay create on dgx01", 1), ("relay guard on dgx01", 1), ("input create on dgx02", 2),
                  ("input guard on dgx02", 2), ("kit copy devbox -> dgx01", 1), ("relay verify on dgx01", 1),
                  ("kit copy dgx01 -> dgx02", 2), ("input verify on dgx02", 2), ("control files", 2),
                  ("setup dispatch", 2)],
        "prelaunch": [("runner dispatch", 2), ("marker read: admission", 1), ("marker read: record", 1),
                      ("marker read: before create", 1), ("marker read: after create", 1)],
        "post_payload": [("marker read: before cleanup", 1)],
        "export": [("output copy dgx02 -> dgx01 relay", 2), ("relay inventory on dgx01", 1),
                   ("output copy dgx01 relay -> devbox", 1)],
        "cleanup": [("input relay cleanup on dgx01", 1), ("own container and output cleanup", 2),
                    ("output relay removal dgx02 -> dgx01", 1)],
        "marker": [("marker check", 1), ("release probe marker", 1), ("release payload marker", 1)],
    },
}
# Every duration derived from a rate not measured for that exact step is doubled: the per-hop LAN throughput and node
# hashing (the devbox's measured cold rate below). Measured DEV work is doubled too.
UNMEASURED_MARGIN = 2
# The per-hop rate: the slower of the two copies measured on the owner-approved DEV -> dgx01 pre-stage (receipt c671ed82:
# the 240-file kit at 19,787,536 B/s; the bundle ran at 24,782,206 B/s), under the owner's 25 MB/s cap. A sizing
# reference only: the 01 -> 02 hop and both export directions are unmeasured, so the 2x margin above still applies.
HOP_BYTES_PER_S = 19_787_536
HASH_BYTES_PER_S = 559_000_000      # devbox, cold page cache: the 22,810-file bundle hashed at 559.08 MB/s (receipt 457d14de)
# The setup admits an operator pack of at most 64 MiB, so that bound is reserved.
KIT_PACK_RESERVED_BYTES = 64 * MIB
# Setup verification of what each node holds in the window (the operator's setup: kit and, on dgx01, bundle inventories
# before and after promotion, both checkpoint files, the oracle admission, disk walks), the largest of three DEV
# measurements, doubled: a warm run of the setup (timing 89bc74c8), the cold devbox composition of the same operations
# (receipt 457d14de) and the actual DEV setup on the final pack (setup-term companion 2f910179: b9bb51a3 / d5600bb9).
SETUP_MEASURED_S = {"dgx01": max(28.215, 2 * 0.090 + 2 * (10.261 + 1.181) + 0.810 + 2 * 1.963 + 2 * 0.074, 38.482181),
                    "dgx02": max(1.151, 2 * 0.094 + 0.788 + 2 * 0.003, 1.443244)}
# The runner's own checks before each payload (sealed-file hashes, plan, kit and checkpoint hashes), cold, worst job of
# two trials (receipt 457d14de): kept as the floor under the measured DEV prelaunch work below.
RUNNER_CHECKS_MEASURED_S = {"dgx01": 11.148, "dgx02": 0.965}
# The sole revised DEV measurement (complete-flow 2d3ba780, checked handoff 76d59e0a), in seconds: disjoint depth-0 rows
# and their direct children, each placed in the gap where it ran (a nested row is never added again):
#   head: input + control; readiness: receipt gates + job 1's prelaunch hashes (plan with its nested oracle admission,
#   kit SHA256SUMS, VAE weights and config, oracle admission and PB MANIFEST on dgx01); between: job 1's receipt lock +
#   gates + job 2's prelaunch; tail: job 2's receipt lock + gates + output inventory + the export stage's own work;
#   export save/fsync and verify, cleanup and release: the export stage's children.
# The full-producer audit runs offline at pre-stage against the exact tree (stage_tree()), so its DEV rows (dgx01
# 8.377594 / 7.290590 s) are not in the window; only the runner's manifest and PB hashes remain.
DEV_MEASURED_S = {"dgx01": {"head": 0.407876, "readiness": 10.765709, "between": 9.805015, "tail": 1.907732,
                            "export_save_verify": 64.26713, "cleanup": 0.623559, "release": 0.630414},
                  "dgx02": {"head": 0.592173, "readiness": 0.929465, "between": 0.895982, "tail": 1.941433,
                            "export_save_verify": 32.070856, "cleanup": 1.457575, "release": 0.657656}}
# U4: the export of each node's full output profile, measured on DEV and reserved at 2x. The profile is synthetic and
# conservative (d2-u4-export-profile-v1, RECEIPT.sha256 below; per-node file lists at the owner's byte ceilings,
# OUTPUT_FILES blobs, the runner's files and incident samples); it claims no actual CUDA counts. The measurement runs the
# export's own sequence (source inventory, each hop's copy and readback, full verification, fsync of every saved file
# and directory, verification again) at the existing rsync flags and rate limit. Owner ruling (conservative): the export
# work is 2x that FULL wall time, transfer included, floored by the byte-rate hashing; the transport stays reserved
# separately at 2x the reference rate, once per hop (U3), and must cover 2x each measured hop's rsync: a hop whose
# doubled rsync exceeds its per-hop reservation refuses the plan, no fit and no trimming (the wall's own reservation
# is 2x itself by construction). Per node {"wall_seconds",
# "hop_transfer_seconds": [one per hop]} from the receipt; None until that receipt is bound here, and the budget refuses
# without it.
EXPORT_PROFILE_SHA256 = "115bf33e7d9a2daf775b2ce3afdd8eb55cf8c6522aa57a8b5aaba05aff60e038"
DEV_FLOW_SHA256 = "2d3ba780ff7e41ca1fc76f720a3811b97721a664477eef3017744118b8bea22b"
DEV_HANDOFF_SHA256 = "76d59e0abef34253a08f48735738b5a84129f51e606e0377cd0bf9fa4eaef61f"
# The measurement (u4-profile-measurement-v1, session exit 0, driver measure.py ea734725, profile b11f5dcc), pinned by
# its checked aggregate U4-AGGREGATE.json (whose U4-AGGREGATE.sha256 covers both nodes' RESULT.json, the launch log, the
# driver, profile and source pins). The RAW results' own note to subtract the transfer is superseded evidence: the
# owner's conservative ruling above applies, and the RAW files are unchanged.
EXPORT_PROFILE_MEASURED_S = {"dgx01": {"wall_seconds": 303.3327721859996, "hop_transfer_seconds": [198.38651319699966]},
                             "dgx02": {"wall_seconds": 195.7016741869993,
                                       "hop_transfer_seconds": [62.44003314699967, 54.635188617000495]}}
EXPORT_PROFILE_RECEIPT_SHA256 = "9000190b4e95c576c37f79edd026732a1098941df804698e409d7be92b3ed432"   # U4-AGGREGATE.json
EXPORT_PROFILE_RESULT_SHA256 = {"U4-AGGREGATE.sha256": "73d4198bb48d3b98f1b66e11acf5f1494a3ce59942a63ee09d031ac11f0f1edb",
                                "RECEIPT.sha256 (RAW)": "e160ce56d60c7ef679a4518769835117434f8d26e7815732e40679499dc53b9e",
                                "dgx01": "35f0c6185c46fb7e405ab3ff7d88f32ab5a69a9341a4a73d3b48feaca47246e9",
                                "dgx02": "4c7ce00d1c66ad1370a07afe0ba43d0d182405869a2fbe1e3bb852471836cfca",
                                "launch.raw.log": "3b6eb59ea3a2473feffd9670bed5239e77cbe7ca1134bdf7f21dad3d4c7df4aa"}
# Allocations, not measurements: the operator's handover from T0 to setup start, the node's cleanup after the saved
# copy is verified, and the marker removal. Each is enforced only by the deadlines that follow it.
HANDOVER_S = 300
CLEANUP_S = 120
MARKER_S = 60
# Per node: the bytes copied in the window before setup, the hops they take, and the hops the outputs take home.
# dgx01's bundle is pre-staged under its own receipt (c671ed82) and its checkpoint is owner-verified present; every kit
# this producer emits is a new pack, so dgx01's kit takes its one hop in the window. Both nodes' checkpoints are rehashed
# in setup, never copied. dgx02's kit takes both hops (devbox -> dgx01 -> dgx02) in the window, as do its outputs back.
INPUT = {"dgx01": (KIT_PACK_RESERVED_BYTES, 1), "dgx02": (KIT_PACK_RESERVED_BYTES, 2)}
EXPORT_HOPS = {"dgx01": 1, "dgx02": 2}
# dgx02's shorter payload, every case and record kept: the job and the engine's own timeout (run-engine.sh TIMEOUT;
# the 120 s between them covers preflight, the kit hash, the 10 s kill grace and the reaped record), and an output cap
# from the engine's records: 843,015,168 B of raw blobs at most (sha-named, so duplicates only lower it) plus 69 KB of
# ops.jsonl, from a --sim run's exact shapes (receipt d2-engine-output-sizing 658be32d). The engine's GPU time is
# unmeasured: a case past TIMEOUT fails the job, sticky for the node, never extended.
ENGINE_JOB_TIMEOUT_S = 1140
ENGINE_TIMEOUT_S = 1020
ENGINE_OUTPUT_BYTES = 1 * GIB
# What the window budget reads per job, in node order (probe, then payload); main() emits these values.
BUDGET_JOBS = {"dgx01": [("d2-probe-torch", 600, 256 * MIB), ("d2-oracle-short", 2100, 4 * GIB)],
               "dgx02": [("d2-probe-native", 600, 256 * MIB), ("d2-engine", ENGINE_JOB_TIMEOUT_S, ENGINE_OUTPUT_BYTES)]}
# What each payload writes under its output directory at most (files, directories), from the producers: a probe its log,
# steps, phases and four allocator receipts; d2-oracle-short 37 fixed files (steps, memavail, summary, 7 step logs,
# 4 allocator files per segment, the oracle's ops.jsonl, env.json and summary.json) + its sha-named blobs; d2-engine 13
# fixed files + its blobs. Blobs are bounded by names, not by the run: oracle.py's Records.put refuses a name recorded
# twice in a case (so a repeated decode.input fails the case instead of adding a blob) and k6v-qual's put/mark skip a
# seen name, so a case writes at most one blob per name it can record: the checkpoint's 124 module hook names
# (hook-names.txt of the U4 profile) + the oracle's 9 / the engine's 7 contract names. Per case (d2-oracle-short:
# 133 x 3 + 8 + 2 = 409; d2-engine: 131 x 3 + 6 + 2 = 401). With the output caps above this is the export profile U4
# measures (d2-u4-export-profile-v1); the runner's own files (receipts, logs, argv audit, incidents) come on top there.
OUTPUT_FILES = {"d2-probe-torch": (7, 1), "d2-oracle-short": (37 + 409, 3),
                "d2-probe-native": (7, 1), "d2-engine": (13 + 401, 3)}

# --- Memory (D2-EVIDENCE.md, "Accounting and admission"; owner's GB10 policy (a)-(d), rulings G1-G7) ----------------
# A process's budget is its allocator cap (Torch: the native per-process fraction; native: the k6v-qual ledger) plus the
# unmeasured non-allocator overhead (CUDA context, modules, library workspaces, driver): 4 GiB, doubled while unmeasured.
# The runner admits a launch only if MemAvailable >= the budgets of every job still resident on the node + one floor,
# and kills the container when MemAvailable crosses the floor. The container's --memory/--memory-swap limit is host
# (CPU) accounting only: GB10's cgroup does not contain GPU allocations.
HOST_FLOOR_BYTES = 12 * GIB
UNMEASURED_OVERHEAD_BYTES = 2 * 4 * GIB
CONTAINER_MEMORY_BYTES = 48 * GIB
# per job: allocator cap, the runner's resident limit
MEMORY = {"d2-probe-torch": (38 * GIB, 46 * GIB), "d2-oracle-short": (38 * GIB, 48 * GIB),
          "d2-probe-native": (24 * GIB, 32 * GIB), "d2-engine": (24 * GIB, 40 * GIB)}
POLICY_TERMS = {
    "a": "Torch: caching-allocator cap per process via set_per_process_memory_fraction, read back equal to the value "
         "set and <= allocator_cap_bytes, rechecked every sample. Native: every device buffer reserved in the k6v-qual "
         "ledger (cap allocator_cap_bytes) before it is allocated.",
    "b": "unmeasured_overhead_bytes (2 x 4 GiB) per process for context, modules, library workspaces and driver "
         "allocations; process_budget_bytes = allocator_cap_bytes + unmeasured_overhead_bytes <= max_resident_bytes. "
         "Replacing it by a measured overhead with >= 2x margin needs a later seal.",
    "c": "host MemAvailable floor host_memavailable_floor_bytes, sampled <= 0.5 s, container killed when crossed; "
         "each launch admitted only if MemAvailable >= the sum of every resident job's process_budget_bytes + one floor "
         "(unknown residency counts as resident).",
    "d": "sticky STOP_ALL on a global or foreign memcg OOM, an earlyoom or systemd-oomd kill, an NVRM Xid or an "
         "unverified cleanup (the D1 classifier). container_memory_bytes is host accounting, never a GPU bound.",
}


def memory_fields(job_id: str) -> dict:
    """The job's memory declaration; refuses a budget its resident limit cannot hold."""
    cap, resident = MEMORY[job_id]
    budget = cap + UNMEASURED_OVERHEAD_BYTES
    assert budget <= resident, f"{job_id}: process budget {budget} exceeds max_resident_bytes {resident}"
    assert HOST_FLOOR_BYTES >= 12 * GIB, f"host floor {HOST_FLOOR_BYTES} is below 12 GiB"
    return {"memory_policy": POLICY, "allocator_cap_bytes": cap, "unmeasured_overhead_bytes": UNMEASURED_OVERHEAD_BYTES,
            "process_budget_bytes": budget, "container_memory_bytes": CONTAINER_MEMORY_BYTES,
            "host_memavailable_floor_bytes": HOST_FLOOR_BYTES, "max_resident_bytes": resident}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def artifacts(kit: Path) -> list:
    """Every artifact the sealer links, with its sha256 (stage_tree() has refused any file a manifest does not cover)."""
    files = [p for d in ARTIFACT_DIRS for p in sorted((kit / d).rglob("*")) if p.is_file()]
    files += [kit / f for f in ARTIFACT_FILES]
    return [{"source": str(p), "destination": str(p.relative_to(kit)), "sha256": sha256(p)} for p in files]


def stage_tree(kit: Path, links: tuple = ORACLE_LINKS) -> dict:
    """The exact tree the offline pre-stage audit checks against what it stages (owner ruling U7): every entry under the
    kit, pybundle/ included, counted by type, and one hash, tree_sha256 = sha256 of the compact JSON list of entries
    sorted by their UTF-8 path, ["d", path] | ["f", path, bytes, sha256] | ["l", path, target]. Refuses a tree that a
    manifest does not fully cover: every regular file outside pybundle/ is listed in MANIFEST.sha256 (the kit's identity
    itself excepted), every regular file in pybundle/ in pybundle/MANIFEST.sha256, every link is one of the bundle's
    relative links (the runner's link inventory, `links` = (count, sha256)), no directory is empty, and every file either
    manifest lists is present."""
    kit = kit.resolve()

    def listed(manifest: Path, prefix: str) -> dict:
        names = {}
        for line in manifest.read_text().splitlines():
            digest, name = line.split("  ", 1)
            names[prefix + name.removeprefix("./")] = digest
        return names

    covered = listed(kit / "MANIFEST.sha256", "") | listed(kit / "pybundle" / "MANIFEST.sha256", "pybundle/")
    entries, bundle_links, refused = [], [], []
    counts = {"regular_files": 0, "directories": 0, "links": 0, "empty_directories": 0, "regular_bytes": 0}
    for path in sorted(kit.rglob("*"), key=lambda p: str(p.relative_to(kit)).encode()):
        name = str(path.relative_to(kit))
        if path.is_symlink():
            target = os.readlink(path)
            inside = name.startswith("pybundle/") and not os.path.isabs(target) \
                and path.resolve().is_relative_to(kit / "pybundle")
            if inside:
                bundle_links.append({"path": name.removeprefix("pybundle/"), "target": target})
            else:
                refused.append(f"link {name}")
            entries.append(["l", name, target])
            counts["links"] += 1
        elif path.is_dir():
            if not any(path.iterdir()):
                refused.append(f"empty directory {name}")
                counts["empty_directories"] += 1
            entries.append(["d", name])
            counts["directories"] += 1
        elif path.is_file():
            digest, size = sha256(path), path.stat().st_size
            if name != "MANIFEST.sha256" and covered.get(name) != digest:
                refused.append(f"unmanifested {name}")
            entries.append(["f", name, size, digest])
            counts["regular_files"] += 1
            counts["regular_bytes"] += size
        else:
            refused.append(f"special file {name}")
    present = {e[1] for e in entries if e[0] == "f"}
    refused += [f"missing {name}" for name in sorted(set(covered) - present)]
    bundle_links.sort(key=lambda e: e["path"])
    inventory = hashlib.sha256(json.dumps(bundle_links, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (len(bundle_links), inventory) != links:
        refused.append(f"bundle links {len(bundle_links)} / {inventory}, not {links[0]} / {links[1]}")
    assert not refused, f"the stage tree is not fully covered: {refused[:20]} ({len(refused)})"
    tree = json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode()
    return {"rule": "tree_sha256 = sha256(compact JSON of the entries sorted by UTF-8 path: [\"d\", path] | "
                    "[\"f\", path, bytes, sha256] | [\"l\", path, target]), every entry under the kit root, pybundle/ "
                    "included; files in MANIFEST.sha256 or pybundle/MANIFEST.sha256, links the bundle's, no empty directory",
            "counts": counts, "empty_directories": [], "tree_sha256": hashlib.sha256(tree).hexdigest(),
            "kit_manifest_sha256": sha256(kit / "MANIFEST.sha256"),
            "pybundle_manifest_sha256": sha256(kit / "pybundle" / "MANIFEST.sha256"), "pybundle_links_sha256": inventory}


def up(seconds: float) -> int:
    """A reservation: whole seconds, rounded up."""
    return math.ceil(seconds - 1e-9)


def node_sequence(node: str, jobs: list) -> dict:
    """One node's itemized sequence as integer seconds from T0, forward: the setup reservations and slack to the setup
    deadline, the prelaunch work to job 1's L (readiness), job 1, the gap to job 2's L, job 2, then the export to the
    verified saved copy, cleanup and marker removal to the node's end. `jobs` is the node's two jobs in order (probe,
    then payload), each with id, timeout_seconds and output_budget_bytes. The finish margins depend on the window and
    come from window_plan(). Raises AssertionError if a measurement is unbound or a reservation outgrows its allocation."""
    (j1, j2) = jobs
    assert EXPORT_PROFILE_RECEIPT_SHA256 and EXPORT_PROFILE_MEASURED_S[node] is not None, \
        f"{node}: U4, the export profile measurement, is not bound"
    m, ops, floor = DEV_MEASURED_S[node], CROSS_NODE_OPS[node], DEADLINE_FLOOR_S[node]

    def cross(group):
        return CROSS_NODE_OP_S * sum(hops for _, hops in ops[group])

    def prelaunch(measured):
        # one job's runner work before its L: the runner's checks (the larger of the cold devbox measurement and the
        # DEV prelaunch hashes, doubled), its unmeasured work, Docker's bounds, the post-create inspect, the marker reads
        return {"prelaunch_cross_node_operations": cross("prelaunch"),
                "runner_checks": max(up(UNMEASURED_MARGIN * RUNNER_CHECKS_MEASURED_S[node]), up(UNMEASURED_MARGIN * measured)),
                "runner_prelaunch_reserve": RUNNER_PRELAUNCH_RESERVE_S, "runner_pre_payload_bound": RUNNER_PRE_S,
                "post_create_inspect": DOCKER_INSPECT_S, "strict_admission": STRICT_S}

    post = {"runner_post_payload_bound": RUNNER_POST_S, "post_payload_inspect": DOCKER_INSPECT_S,
            "post_payload_cross_node_operations": cross("post_payload")}
    assert up(UNMEASURED_MARGIN * m["cleanup"]) <= CLEANUP_S and up(UNMEASURED_MARGIN * m["release"]) <= MARKER_S, \
        f"{node}: measured cleanup or release outgrows its allocation"
    in_bytes, in_hops = INPUT[node]
    hops = EXPORT_HOPS[node]
    out_bytes = j1["output_budget_bytes"] + j2["output_budget_bytes"]
    head = {
        "handover_allocation": HANDOVER_S,
        "input_copy": up(UNMEASURED_MARGIN * in_hops * in_bytes / HOP_BYTES_PER_S),
        "input_cross_node_operations": cross("input"),
        "input_control_measured": up(UNMEASURED_MARGIN * m["head"]),
        "setup_verification": up(UNMEASURED_MARGIN * SETUP_MEASURED_S[node]),
    }
    reserved = sum(head.values())
    setup_deadline = max(floor["setup"], reserved + SETUP_SLACK_MIN_S)
    before = prelaunch(m["readiness"])
    l1 = max(floor["L1"], setup_deadline + sum(before.values()))
    before["deadline_floor_slack"] = l1 - setup_deadline - sum(before.values())
    between = {**post, **prelaunch(m["between"])}
    l2 = max(floor["L2"], l1 + j1["timeout_seconds"] + sum(between.values()))
    between["deadline_floor_slack"] = l2 - l1 - j1["timeout_seconds"] - sum(between.values())
    u4 = EXPORT_PROFILE_MEASURED_S[node]
    assert len(u4["hop_transfer_seconds"]) == EXPORT_HOPS[node], f"{node}: U4 measured {len(u4['hop_transfer_seconds'])} hops"
    assert 0 <= sum(u4["hop_transfer_seconds"]) <= u4["wall_seconds"], f"{node}: U4 transfer outside its wall time"
    export = {**post,
              "export_outputs": up(UNMEASURED_MARGIN * hops * out_bytes / HOP_BYTES_PER_S),
              "export_cross_node_operations": cross("export"),
              # the larger of the byte-rate hash term and the measured save, fsync and verify (DEV flow, U4 profile)
              # the export's whole measured chain (U4's full wall, transfer included; the DEV flow's rows of the same
              # chain never exceed it), at 2x, floored by the byte-rate hashing of every copy
              "export_work": max(up(UNMEASURED_MARGIN * (hops + 1) * out_bytes / HASH_BYTES_PER_S),
                                 up(UNMEASURED_MARGIN * max(m["tail"] + m["export_save_verify"], u4["wall_seconds"])))}
    # U3 coverage: each hop's transport reservation must hold 2x that hop's measured rsync
    hop_reserved = export["export_outputs"] / hops
    assert all(UNMEASURED_MARGIN * t <= hop_reserved for t in u4["hop_transfer_seconds"]), \
        f"{node}: 2x a measured U4 hop {u4['hop_transfer_seconds']} exceeds its transport reservation {hop_reserved}"
    cleanup = {"cleanup_cross_node_operations": cross("cleanup"), "cleanup_allocation": CLEANUP_S}
    marker = {"marker_cross_node_operations": cross("marker"), "marker_removal_allocation": MARKER_S}
    payload_end = l2 + j2["timeout_seconds"]
    saved = payload_end + sum(export.values())
    cleaned = saved + sum(cleanup.values())
    end = cleaned + sum(marker.values())
    return {
        "node_end": end, "setup_deadline": setup_deadline, "setup_reserved": reserved, "setup_slack": setup_deadline - reserved,
        "readiness": {"job": j1["id"], "latest_start_seconds_from_open": l1, "before": {"setup_deadline": setup_deadline, **before}},
        "jobs": {
            j1["id"]: {"latest_start_seconds_from_open": l1, "payload_end": l1 + j1["timeout_seconds"], "timeout_seconds": j1["timeout_seconds"]},
            j2["id"]: {"latest_start_seconds_from_open": l2, "payload_end": payload_end, "timeout_seconds": j2["timeout_seconds"]},
        },
        "setup_reservations": head, "between_jobs": between, "after_payload": {**export, **cleanup, **marker},
        "finish_deadlines": {"saved_copy_verified_by": saved, "cleanup_verified_by": cleaned, "marker_removed_by": end},
        "cross_node_operations": {group: dict(items) for group, items in ops.items()},
        "cross_node_operation_seconds_per_hop": CROSS_NODE_OP_S,
        "export_bytes_reserved": out_bytes, "export_hops": hops,
        "export_payload_files": sum(OUTPUT_FILES[j["id"]][0] for j in jobs),
        "export_payload_directories": sum(OUTPUT_FILES[j["id"]][1] for j in jobs),
        "input_bytes_reserved": in_bytes, "input_hops": in_hops, "transfer_reference_bytes_per_second": HOP_BYTES_PER_S,
    }


def window_plan(jobs_by_node: dict) -> tuple:
    """Every node's sequence, then the window: N whole minutes from the later node end plus the 10-minute close, at most
    180, and each job's finish margin F against it. Returns (N, budgets). Raises AssertionError if anything does not fit."""
    budgets = {node: node_sequence(node, jobs) for node, jobs in jobs_by_node.items()}
    minutes = -(-(max(b["node_end"] for b in budgets.values()) + WINDOW_CLOSE_S) // 60)
    window = minutes * 60
    assert window <= WINDOW_MAX_S, f"the window needs {minutes} min, over {WINDOW_MAX_S // 60}"
    for node, b in budgets.items():
        assert b["setup_slack"] >= SETUP_SLACK_MIN_S, f"{node}: setup slack {b['setup_slack']} s"
        assert b["node_end"] + WINDOW_CLOSE_S <= window, f"{node}: ends after the window's close"
        for job in b["jobs"].values():
            job["finish_margin_seconds"] = window - RUNNER_MARGIN_PAD_S - job["payload_end"]
            # L and F bind: the runner's no-new-work cutoff and stop reserve fall later
            assert 0 <= job["latest_start_seconds_from_open"] <= window - RUNNER_NO_NEW_WORK_S, f"{node}: L past no-new-work"
            assert job["payload_end"] <= window - RUNNER_STOP_RESERVE_S and job["finish_margin_seconds"] >= 600, \
                f"{node}: payload end past the runner's stop reserve"
    return minutes, budgets


def self_test() -> None:
    """The window plan: every chain sums exactly to its end, the owner's floors and slack hold, the runner's admission
    and stop rules hold at L - 1 and refuse at L, each reservation moves the node's end by exactly the number of times it
    is itemized (nothing counted twice), the window refuses at 1 s past 180 min, and the plan refuses an unbound U4. The
    stage tree: a covered tree is accepted and hashed; an unmanifested file, a leftover .git, a .pyc, an empty directory,
    a foreign link and a missing kit or bundle file are refused. The memory declarations: budgets inside their resident limits, refused at 1 byte past
    and at the v1 values."""
    import sys
    import tempfile
    me = sys.modules[__name__]

    def jobs(node, extra=0):
        js = [{"id": i, "timeout_seconds": t, "output_budget_bytes": b} for i, t, b in BUDGET_JOBS[node]]
        js[1]["timeout_seconds"] += extra
        return js

    def plan(extra=0, **bump):
        keep = {k: getattr(me, k) for k in bump}
        try:
            for k, v in bump.items():
                setattr(me, k, getattr(me, k) + v if isinstance(v, int) else v)
            return window_plan({n: jobs(n, extra if n == "dgx02" else 0) for n in BUDGET_JOBS})
        except AssertionError:
            return None
        finally:
            for k, v in keep.items():
                setattr(me, k, v)

    checked = 0

    def check(ok, what):
        nonlocal checked
        assert ok, what
        checked += 1

    for seconds, want in ((0, 0), (1, 1), (1.0000000001, 1), (1.001, 2), (35.4, 36), (559.9, 560)):
        check(up(seconds) == want, f"up({seconds}) = {up(seconds)}, not {want}")
    bound = EXPORT_PROFILE_MEASURED_S, EXPORT_PROFILE_RECEIPT_SHA256
    me.EXPORT_PROFILE_MEASURED_S, me.EXPORT_PROFILE_RECEIPT_SHA256 = {"dgx01": None, "dgx02": None}, None
    check(plan() is None, "U4: the plan refuses an unbound export measurement")
    me.EXPORT_PROFILE_MEASURED_S, me.EXPORT_PROFILE_RECEIPT_SHA256 = bound
    if bound[1] is None:
        # before the receipt is bound, the algebra below runs on stand-in values; the sealed plan needs the receipt
        me.EXPORT_PROFILE_MEASURED_S = {"dgx01": {"wall_seconds": 160.0, "hop_transfer_seconds": [60.0]},
                                        "dgx02": {"wall_seconds": 70.0, "hop_transfer_seconds": [15.0, 15.0]}}
        me.EXPORT_PROFILE_RECEIPT_SHA256 = "0" * 64
    try:
        minutes, budgets = window_plan({n: jobs(n) for n in BUDGET_JOBS})
        window = minutes * 60
        check(window <= WINDOW_MAX_S and max(b["node_end"] for b in budgets.values()) + WINDOW_CLOSE_S > window - 60,
              "the window is the later node end plus the close, in whole minutes")
        report = {}
        for node, b in budgets.items():
            (i1, t1, _), (i2, t2, _) = BUDGET_JOBS[node]
            j1, j2 = b["jobs"][i1], b["jobs"][i2]
            l1, l2 = j1["latest_start_seconds_from_open"], j2["latest_start_seconds_from_open"]
            before = dict(b["readiness"]["before"])
            check(before.pop("setup_deadline") == b["setup_deadline"], f"{node}: readiness starts at the setup deadline")
            # exact chains, forward from T0
            check(sum(b["setup_reservations"].values()) + b["setup_slack"] == b["setup_deadline"], f"{node}: setup chain")
            check(b["setup_deadline"] + sum(before.values()) == l1 == b["readiness"]["latest_start_seconds_from_open"], f"{node}: readiness chain")
            check(l1 + t1 == j1["payload_end"] and l2 + t2 == j2["payload_end"], f"{node}: payload chains")
            check(j1["payload_end"] + sum(b["between_jobs"].values()) == l2, f"{node}: between-jobs chain")
            check(j2["payload_end"] + sum(b["after_payload"].values()) == b["node_end"], f"{node}: tail chain")
            f = b["finish_deadlines"]
            tail = b["after_payload"]
            check(f["marker_removed_by"] == b["node_end"]
                  and f["cleanup_verified_by"] == b["node_end"] - tail["marker_cross_node_operations"] - tail["marker_removal_allocation"]
                  and f["saved_copy_verified_by"] == f["cleanup_verified_by"] - tail["cleanup_cross_node_operations"] - tail["cleanup_allocation"],
                  f"{node}: finish deadlines")
            # the owner's floors and slack
            floor = DEADLINE_FLOOR_S[node]
            check(b["setup_deadline"] >= floor["setup"] and l1 >= floor["L1"] and l2 >= floor["L2"], f"{node}: deadline floors")
            check(b["setup_slack"] >= SETUP_SLACK_MIN_S, f"{node}: setup slack")
            check(min(before["deadline_floor_slack"], b["between_jobs"]["deadline_floor_slack"]) >= 0, f"{node}: floor slack")
            check(tail["cleanup_allocation"] == 120 and tail["marker_removal_allocation"] == 60
                  and b["setup_reservations"]["handover_allocation"] == 300, f"{node}: owner allocations unchanged")
            check(not any("audit" in k for gap in ("setup_reservations", "between_jobs", "after_payload") for k in b[gap])
                  and not any("audit" in k for k in before), f"{node}: no audit in the window")
            # the runner's rules with these fields, T0 = 0, expires = window
            for j, t in ((j1, t1), (j2, t2)):
                L, F = j["latest_start_seconds_from_open"], j["finish_margin_seconds"]
                cutoff = min(window - RUNNER_NO_NEW_WORK_S, L)
                stop = min(window - RUNNER_STOP_RESERVE_S, window - F - RUNNER_MARGIN_PAD_S)
                check(cutoff == L and stop == j["payload_end"], f"{node}: the sealed L/F are the binding cutoff and stop")
                check(L - 1 < cutoff and L - 1 + t <= stop, f"{node}: a start at L - 1 is admitted and ends by its stop")
                check(not L < cutoff, f"{node}: a start at L is refused")
                check(L >= 0 and F >= 600, f"{node}: the runner's field ranges")
            report[node] = {"setup_deadline": b["setup_deadline"], "slack": b["setup_slack"], "L": [l1, l2],
                            "F": [j1["finish_margin_seconds"], j2["finish_margin_seconds"]], "end": b["node_end"],
                            # prelaunch and post-payload operations happen once per job, the rest once per node
                            "cross_node_hops": sum(h * (2 if group in ("prelaunch", "post_payload") else 1)
                                                   for group, items in CROSS_NODE_OPS[node].items() for _, h in items)}
        # each reservation is itemized exactly as often as the sequence has it: dgx02 has no floors, so its end moves by
        # that count per second added
        hops = report["dgx02"]["cross_node_hops"]
        end02 = budgets["dgx02"]["node_end"]
        for name, times in (("HANDOVER_S", 1), ("CLEANUP_S", 1), ("MARKER_S", 1), ("SETUP_SLACK_MIN_S", 1),
                            ("RUNNER_PRE_S", 2), ("RUNNER_POST_S", 2), ("STRICT_S", 2), ("RUNNER_PRELAUNCH_RESERVE_S", 2),
                            ("DOCKER_INSPECT_S", 4), ("CROSS_NODE_OP_S", hops)):
            bumped = plan(**{name: 1})
            check(bumped is not None and bumped[1]["dgx02"]["node_end"] == end02 + times, f"dgx02: {name} itemized {times} times")
        # U4 (owner ruling): the full wall at 2x, transfer included; the transport is reserved separately per hop, and a
        # measured hop above that reservation refuses
        stand_in = me.EXPORT_PROFILE_MEASURED_S
        for node in ("dgx01", "dgx02"):
            def end_with(wall, transfers):
                me.EXPORT_PROFILE_MEASURED_S = {**stand_in, node: {"wall_seconds": wall, "hop_transfer_seconds": transfers}}
                try:
                    return node_sequence(node, jobs(node))
                except AssertionError:
                    return None
                finally:
                    me.EXPORT_PROFILE_MEASURED_S = stand_in
            w, x = stand_in[node]["wall_seconds"], stand_in[node]["hop_transfer_seconds"]
            base = end_with(w, x)
            check(base["after_payload"]["export_work"] == up(2 * w), f"{node}: U4 export work is 2x the full wall")
            check(end_with(w, [t + 1 for t in x])["node_end"] == base["node_end"], f"{node}: the transfer inside the wall is not subtracted")
            check(end_with(w + 1, x)["node_end"] == base["node_end"] + 2, f"{node}: one second of U4 wall adds 2")
            reserved = base["after_payload"]["export_outputs"] / len(x)
            check(end_with(reserved * len(x) + 100, [reserved / 2] * len(x)) is not None,
                  f"{node}: a hop whose doubled rsync equals its reservation is admitted")
            check(end_with(reserved * len(x) + 100, [reserved / 2 + 0.001] + [0.0] * (len(x) - 1)) is None,
                  f"{node}: a hop whose doubled rsync exceeds its transport reservation is refused")
            check(end_with(w, x[:-1] + [x[-1], 0.0]) is None, f"{node}: a U4 receipt with the wrong hop count is refused")
            check(end_with(1.0, [2.0] + [0.0] * (len(x) - 1)) is None, f"{node}: a U4 transfer longer than its wall is refused")
        check(budgets["dgx02"]["setup_deadline"] == sum(budgets["dgx02"]["setup_reservations"].values()) + SETUP_SLACK_MIN_S,
              "dgx02: setup deadline = reservations + 300 s")
        # the window refuses 1 s past 180 min: grow dgx02's payload until it is the later node and the window is full
        most = WINDOW_MAX_S - WINDOW_CLOSE_S - end02
        check(plan(most) is not None and plan(most)[0] == WINDOW_MAX_S // 60 and plan(most + 1) is None, "window at 180 min")
        # the cleanup and marker allocations hold their measured work, doubled
        check(plan(DEV_MEASURED_S={**DEV_MEASURED_S, "dgx02": {**DEV_MEASURED_S["dgx02"], "cleanup": 60.0}}) is not None
              and plan(DEV_MEASURED_S={**DEV_MEASURED_S, "dgx02": {**DEV_MEASURED_S["dgx02"], "cleanup": 60.001}}) is None,
              "cleanup allocation boundary")
    finally:
        me.EXPORT_PROFILE_MEASURED_S, me.EXPORT_PROFILE_RECEIPT_SHA256 = bound

    # the stage tree on a small kit: covered and hashed, then each uncovered entry refused
    with tempfile.TemporaryDirectory(prefix="d2-stage-tree-") as tmp:
        kit = Path(tmp) / "kit"
        (kit / "pybundle" / "lib").mkdir(parents=True)
        (kit / "tools").mkdir()
        (kit / "tools" / "a.py").write_text("a\n")
        (kit / "pybundle" / "lib" / "x").write_text("x\n")
        (kit / "pybundle" / "y").symlink_to("lib/x")
        (kit / "pybundle" / "MANIFEST.sha256").write_text(f"{sha256(kit / 'pybundle/lib/x')}  ./lib/x\n")

        def manifest():
            names = sorted(str(p.relative_to(kit)) for p in kit.rglob("*")
                           if p.is_file() and not p.is_symlink() and p.name != "MANIFEST.sha256" and "pybundle/lib" not in str(p))
            names.append("pybundle/MANIFEST.sha256")
            (kit / "MANIFEST.sha256").write_text("".join(f"{sha256(kit / n)}  {n}\n" for n in sorted(set(names))))

        manifest()
        pin = (1, hashlib.sha256(json.dumps([{"path": "y", "target": "lib/x"}], sort_keys=True, separators=(",", ":")).encode()).hexdigest())

        def tree(**kw):
            try:
                return stage_tree(kit, links=pin, **kw)
            except AssertionError:
                return None

        t = tree()
        check(t is not None and t["counts"] == {"regular_files": 4, "directories": 3, "links": 1, "empty_directories": 0,
                                                "regular_bytes": t["counts"]["regular_bytes"]}, "stage tree: covered kit accepted")
        check(tree()["tree_sha256"] == t["tree_sha256"], "stage tree: the hash is deterministic")
        for make, undo in (
                (lambda: (kit / "tools" / "b.py").write_text("b\n"), lambda: (kit / "tools" / "b.py").unlink()),
                (lambda: (kit / "tools" / "__pycache__").mkdir() or (kit / "tools" / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"\0"),
                 lambda: (kit / "tools" / "__pycache__" / "a.cpython-313.pyc").unlink() or (kit / "tools" / "__pycache__").rmdir()),
                (lambda: (kit / "upstream" / ".git").mkdir(parents=True) or (kit / "upstream" / ".git" / "HEAD").write_text("r\n"),
                 lambda: (kit / "upstream" / ".git" / "HEAD").unlink() or (kit / "upstream" / ".git").rmdir() or (kit / "upstream").rmdir()),
                (lambda: (kit / "inputs").mkdir(), lambda: (kit / "inputs").rmdir()),
                (lambda: (kit / "tools" / "c").symlink_to("a.py"), lambda: (kit / "tools" / "c").unlink()),
                (lambda: (kit / "tools" / "a.py").write_text("changed\n"), lambda: (kit / "tools" / "a.py").write_text("a\n"))):
            make()
            check(tree() is None, "stage tree: an uncovered entry is refused")
            undo()
            check(tree() is not None and tree()["tree_sha256"] == t["tree_sha256"], "stage tree: restored")
        # a file a manifest lists but the tree lacks (kit, then bundle), with nothing else changed
        for listing, line in ((kit / "MANIFEST.sha256", f"{'0' * 64}  tools/gone.py\n"),
                              (kit / "pybundle" / "MANIFEST.sha256", f"{'0' * 64}  ./lib/gone\n")):
            keep = listing.read_text()
            listing.write_text(keep + line)
            if listing.parent.name == "pybundle":
                manifest()   # the kit manifest pins the bundle manifest's new digest, so only the missing file is uncovered
            try:
                stage_tree(kit, links=pin)
                refusal = ""
            except AssertionError as e:
                refusal = str(e)
            check("missing " in refusal and "(1)" in refusal, f"stage tree: a listed file the tree lacks is refused ({listing.name})")
            listing.write_text(keep)
            manifest()
            check(tree() is not None and tree()["tree_sha256"] == t["tree_sha256"], "stage tree: restored")

    # memory (owner rulings G1-G3, G6): every job's process budget fits its resident limit, the floor is 12 GiB, and the
    # v1 declarations (floor 10 GiB, Torch probe resident 32 GiB under the payload's 38 GiB cap) are refused
    def declared(**bump):
        keep = {k: getattr(me, k) for k in bump}
        try:
            for k, v in bump.items():
                setattr(me, k, v)
            return [memory_fields(j) for j in MEMORY]
        except AssertionError:
            return None
        finally:
            for k, v in keep.items():
                setattr(me, k, v)

    check(set(MEMORY) == {i for node in BUDGET_JOBS.values() for i, _, _ in node}, "memory: one declaration per job")
    check(set(OUTPUT_FILES) == set(MEMORY), "export profile: one entry per job")
    now = declared()
    check(now is not None, "memory: the declarations are accepted")
    for f in now:
        check(f["memory_policy"] == "gb10-unified-d2-v2" and f["host_memavailable_floor_bytes"] == 12 * GIB, "memory: v2, floor")
        check(f["process_budget_bytes"] == f["allocator_cap_bytes"] + 8 * GIB <= f["max_resident_bytes"], "memory: budget")
        check(f["container_memory_bytes"] == 48 * GIB, "memory: container accounting")
    caps = {j: memory_fields(j)["allocator_cap_bytes"] for j in MEMORY}
    check(caps["d2-probe-torch"] == caps["d2-oracle-short"] and caps["d2-probe-native"] == caps["d2-engine"],
          "memory: each probe runs its payload's cap")
    check(declared(HOST_FLOOR_BYTES=10 * GIB) is None, "memory: the v1 floor is refused")
    check(declared(MEMORY={**MEMORY, "d2-probe-torch": (38 * GIB, 32 * GIB)}) is None, "memory: the v1 probe resident is refused")
    check(declared(UNMEASURED_OVERHEAD_BYTES=8 * GIB + 1) is None, "memory: a budget 1 byte past the probe resident is refused")
    check(set(POLICY_TERMS) == {"a", "b", "c", "d"}, "memory: terms (a)-(d)")
    print(json.dumps({"self_test": "passed", "checks": checked, "u4_bound": bound[1] is not None,
                      "window_minutes": minutes, "nodes": report}))


def main() -> None:
    import sys
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
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
    stage = stage_tree(kit)   # refuses an unmanifested file, an empty directory or a foreign link before anything else
    arts = artifacts(kit)
    common = {"workspace": WORKSPACE, "container_workspace": "/ws", "artifacts": arts, "asset_paths": ASSETS,
              "node_image_receipt": {"source": str(a.node_images), "sha256": sha256(a.node_images)}}

    def policy(job_id, gate=None, segments=None):
        # the runner's unified accounting under the D2-only policy: max(max(memory.current, memory.peak) + helpers
        # + 4 GiB, MemAvailable drop + 4 GiB) <= max_resident_bytes; budgets, floor and admission per memory_fields()
        p = memory_fields(job_id)
        if gate:
            p["requires_probe_pass"] = gate   # this node's probe, this window (hash-bound by the runner)
        if segments:
            p["segments"] = segments          # sealed order, one fresh process and nonce each
        return p

    def probe(job_id, node, kind, oracle):
        env = {"ORACLE_ROOT": ORACLE["container_root"], "ORACLE_MANIFEST_SHA256": ORACLE["manifest_sha256"]} if oracle else {}
        return {"id": job_id, "node": node, "image_id": images[node], **common,
                "command": ["/bin/bash", "/ws/kit/tools/run-probe.sh", kind], "environment": env,
                **({"shared_oracle": ORACLE} if oracle else {}),
                # 16 GiB probe allocation + context + interpreter, accounted; the Torch probe runs the payload's cap
                **policy(job_id), "timeout_seconds": 600,
                "max_disk_bytes": 1_000_000_000, "output_budget_bytes": 256 * MIB}

    oracle = {"id": "d2-oracle-short", "node": "dgx01", "image_id": images["dgx01"], **common,
              "command": ["/bin/bash", "/ws/kit/tools/run-oracle.sh"],
              "environment": {"VAE": "/models/vae", "ORACLE_ROOT": ORACLE["container_root"],
                              "ORACLE_MANIFEST_SHA256": ORACLE["manifest_sha256"], "JOB_BUDGET_S": "1980",
                              "TIMEOUT_ENV_PROBE": "240", "TIMEOUT_N_DEC": "900",
                              **{f"TIMEOUT_{c}": "240" for c in ("S_DEC", "F_DEC", "S_ENC", "N_ENC")}},
              "shared_oracle": ORACLE, **policy("d2-oracle-short", "d2-probe-torch", SEGMENTS),
              # resident 48 GiB: the n_dec Torch peak is unmeasured (estimate 20-30 GiB) + host + helpers + 4 GiB
              "timeout_seconds": 2100, "max_disk_bytes": 12_000_000_000, "output_budget_bytes": 4 * GIB}
    engine = {"id": "d2-engine", "node": "dgx02", "image_id": images["dgx02"], **common,
              "command": ["/bin/bash", "/ws/kit/tools/run-engine.sh"],
              "environment": {"VAE": "/models/vae", "TIMEOUT": str(ENGINE_TIMEOUT_S)},
              **policy("d2-engine", "d2-probe-native"), "timeout_seconds": ENGINE_JOB_TIMEOUT_S,
              # resident 40 GiB: ledger peak 15.55 GB (n_dec case buffer + weights image), ~3 GiB host, helpers, 4 GiB
              "max_disk_bytes": 8_000_000_000, "output_budget_bytes": ENGINE_OUTPUT_BYTES}
    jobs = [probe("d2-probe-torch", "dgx01", "torch", True), oracle,
            probe("d2-probe-native", "dgx02", "native", False), engine]
    by_node = {}
    for node in ("dgx01", "dgx02"):
        mine = [j for j in jobs if j["node"] == node]
        shape = [(j["id"], j["timeout_seconds"], j["output_budget_bytes"]) for j in mine]
        assert shape == BUDGET_JOBS[node], f"{node}: jobs {shape} differ from BUDGET_JOBS"
        by_node[node] = mine
    minutes, budget = window_plan(by_node)
    for node, mine in by_node.items():
        for job in mine:
            fields = budget[node]["jobs"][job["id"]]
            job["latest_start_seconds_from_open"] = fields["latest_start_seconds_from_open"]
            job["finish_margin_seconds"] = fields["finish_margin_seconds"]
    evidence = {"dev_measurement": {"complete_flow_sha256": DEV_FLOW_SHA256, "checked_handoff_sha256": DEV_HANDOFF_SHA256,
                                    "seconds": DEV_MEASURED_S},
                "export_profile": {"profile_sha256": EXPORT_PROFILE_SHA256, "receipt_sha256": EXPORT_PROFILE_RECEIPT_SHA256,
                                   "result_sha256": EXPORT_PROFILE_RESULT_SHA256,
                                   "seconds": EXPORT_PROFILE_MEASURED_S,
                                   "output_files": {j: {"files": f, "directories": d} for j, (f, d) in OUTPUT_FILES.items()}},
                "setup_measured_seconds": SETUP_MEASURED_S, "runner_checks_measured_seconds": RUNNER_CHECKS_MEASURED_S}
    spec = {"schema": 1, "component": "d2-video-vae", "memory_policy": POLICY, "memory_policy_terms": POLICY_TERMS,
            "gate": "per-node", "window_minutes": minutes, "jobs": jobs, "window_budget": budget,
            "window_evidence": evidence, "prestage_audit": stage}
    print(json.dumps(spec, indent=1))


if __name__ == "__main__":
    main()
