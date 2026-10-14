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
d2-oracle is not emitted: the approved window is N = 95 with d2-oracle-short only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
ASSETS = ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
ARTIFACT_DIRS = ["bin", "fatbin", "tools", "inputs", "upstream"]
ARTIFACT_FILES = ["build.json", "MANIFEST.sha256"]

# --- Window budget (D2-EVIDENCE.md, "Window budget"): the owner's SHORT95 rulings as integer seconds from T0 -----------
# T0 is the grant's opened_at; the window ends at T0 + 5700 (expires_at). The runner (run-node-d2-v3.py) reads two
# sealed integers per job:
#   latest_start_seconds_from_open L: the job is admitted only while now < T0 + L, checked at plan time and again right
#     after `docker create` (strictly less: a container created at T0 + L is refused). The runner also caps every job
#     at expires - 1200.
#   finish_margin_seconds F: the runner's stop is min(expires - 690, expires - F - 60), and a job starts only if
#     started + timeout_seconds <= stop (equality admitted): its payload ends by T0 + 5640 - F.
# Readiness, the event the setup budget serves, is job 1 admitted on its node (its L). Every step of a node's sequence
# is reserved below in whole seconds, reservations rounded up and deadlines down; a node that misses any deadline fails
# closed (setup abort before job 1; the runner refuses job 2), never extends.
WINDOW_S = 95 * 60
RUNNER_NO_NEW_WORK_S = 1200         # run-node-d2-v3.py: launch cutoff expires - 1200
RUNNER_STOP_RESERVE_S = 690         # run-node-d2-v3.py: stop <= expires - 690
RUNNER_MARGIN_PAD_S = 60            # run-node-d2-v3.py: stop <= expires - finish_margin - 60
# Owner: dgx01's export, cleanup and release by T0 + N - 10 min; nothing on dgx02 after T0 + 2940, its both-hop export,
# verified saved copy, cleanup and marker removal included. Setup ceilings: setup done by T0 + 2310 / T0 + 840.
NODE_END_S = {"dgx01": WINDOW_S - 600, "dgx02": 2940}
SETUP_CEILING_S = {"dgx01": 2310, "dgx02": 840}
# The runner's own hard bounds per job: before the payload `docker image inspect`, nvidia-smi, `docker ps`,
# `docker create` (30 s each) and the 5 s PID poll; after it `docker rm` (60 s) and two 10 s waits.
RUNNER_PRE_S = 4 * 30 + 5
RUNNER_POST_S = 60 + 2 * 10
# Every duration derived from a rate not measured for that exact step is doubled: the per-hop LAN throughput and node
# hashing (the devbox's measured cold rate below).
UNMEASURED_MARGIN = 2
# The per-hop rate: the slower of the two copies measured on the owner-approved DEV -> dgx01 pre-stage (receipt c671ed82:
# the 240-file kit at 19,787,536 B/s; the bundle ran at 24,782,206 B/s), under the owner's 25 MB/s cap. A sizing
# reference only: the 01 -> 02 hop and both export directions are unmeasured, so the 2x margin above still applies.
HOP_BYTES_PER_S = 19_787_536
HASH_BYTES_PER_S = 559_000_000      # devbox, cold page cache: the 22,810-file bundle hashed at 559.08 MB/s (receipt 457d14de)
# The setup admits an operator pack of at most 64 MiB, so that bound is reserved; the checked final packs are
# 39,235,709 B (dgx01, 4 s at this rate, doubled) and 39,233,516 B (dgx02, 8 s, both hops), and the reservation is
# recomputed whenever the bound changes.
KIT_PACK_RESERVED_BYTES = 64 * MIB
# Setup verification of what each node holds in the window (the operator's setup-d2-v2: kit and, on dgx01, bundle
# inventories before and after promotion, both checkpoint files, the oracle admission per job, disk walks), the larger of
# two DEV measurements, doubled for the node: a warm run of the setup itself (timing 89bc74c8) and the cold devbox
# composition of the same operations, worst trial per item (receipt 457d14de):
#   dgx01  2 x 0.090 kit + 2 x (10.261 + 1.181) bundle hash and walk + 0.810 checkpoints + 2 x 1.963 oracle admission
#          + 2 x 0.074 disk walk = 27.948 s;  dgx02  2 x 0.094 kit + 0.788 checkpoints + 2 x 0.003 disk walk = 0.982 s
SETUP_MEASURED_S = {"dgx01": max(28.215, 2 * 0.090 + 2 * (10.261 + 1.181) + 0.810 + 2 * 1.963 + 2 * 0.074),
                    "dgx02": max(1.151, 2 * 0.094 + 0.788 + 2 * 0.003)}
# The runner's own checks before each payload (sealed-file hashes, plan, kit and checkpoint hashes; on dgx01 the bundle
# walk and its full hash), cold, worst job of two trials (receipt 457d14de); run-node-d2-v3 performs the same I/O.
RUNNER_CHECKS_MEASURED_S = {"dgx01": 11.148, "dgx02": 0.965}
# Allocations, not measurements: the operator's handover from T0 to setup start, the node's cleanup after the saved
# copy is verified, and the marker removal. Each is enforced only by the deadlines that follow it.
HANDOVER_S = 300
STRICT_S = 1                        # the runner admits strictly before L
CLEANUP_S = 120
MARKER_S = 60
# Per node: the bytes copied in the window before setup, the hops they take, and the hops the outputs take home.
# dgx01's bundle is pre-staged under its own receipt (c671ed82, excluded by the owner's policy) and its checkpoint is
# owner-verified present; that receipt's kit is the earlier pack, and every kit this producer emits is a new pack, so
# dgx01's kit takes its one hop in the window. Both nodes' checkpoints are rehashed in setup, never copied.
# dgx02's kit takes both hops (devbox -> dgx01 -> dgx02) in the window, as do its outputs back.
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
    files = [p for d in ARTIFACT_DIRS for p in sorted((kit / d).rglob("*"))
             if p.is_file() and ".git" not in p.relative_to(kit).parts and p.suffix != ".pyc"]
    files += [kit / f for f in ARTIFACT_FILES]
    return [{"source": str(p), "destination": str(p.relative_to(kit)), "sha256": sha256(p)} for p in files]


def up(seconds: float) -> int:
    """A reservation: whole seconds, rounded up."""
    return math.ceil(seconds - 1e-9)


def window_budget(node: str, jobs: list) -> dict:
    """One node's sequence as integer seconds from T0, latest first: from its hard end back to the readiness and setup
    deadlines, then forward from T0 through the setup reservations. `jobs` is the node's two jobs in order (probe, then
    payload), each with timeout_seconds and output_budget_bytes. Raises AssertionError if the node does not fit."""
    (j1, j2) = jobs
    hops = EXPORT_HOPS[node]
    out_bytes = j1["output_budget_bytes"] + j2["output_budget_bytes"]
    checks = up(UNMEASURED_MARGIN * RUNNER_CHECKS_MEASURED_S[node])
    tail = {
        "runner_post_payload_bound": RUNNER_POST_S,
        "export_outputs": up(UNMEASURED_MARGIN * hops * out_bytes / HOP_BYTES_PER_S),
        "export_hash_verification": up(UNMEASURED_MARGIN * (hops + 1) * out_bytes / HASH_BYTES_PER_S),
        "cleanup_allocation": CLEANUP_S,
        "marker_removal_allocation": MARKER_S,
    }
    payload_end = NODE_END_S[node] - sum(tail.values())
    l2 = payload_end - j2["timeout_seconds"]
    # admission is strictly before L: a step chain that lands exactly on L is refused, so each chain reserves 1 s more
    between = {"runner_post_payload_bound": RUNNER_POST_S, "runner_checks": checks, "runner_pre_payload_bound": RUNNER_PRE_S,
               "strict_admission": STRICT_S}
    l1 = l2 - sum(between.values()) - j1["timeout_seconds"]
    setup_deadline = min(SETUP_CEILING_S[node], l1 - checks - RUNNER_PRE_S - STRICT_S)
    in_bytes, in_hops = INPUT[node]
    head = {
        "handover_allocation": HANDOVER_S,
        "input_copy": up(UNMEASURED_MARGIN * in_hops * in_bytes / HOP_BYTES_PER_S),
        "setup_verification": up(UNMEASURED_MARGIN * SETUP_MEASURED_S[node]),
    }
    budget = {
        "node_end": NODE_END_S[node], "setup_ceiling": SETUP_CEILING_S[node],
        "setup_deadline": setup_deadline, "setup_reserved": sum(head.values()),
        "setup_slack": setup_deadline - sum(head.values()),
        "readiness": {"job": j1["id"], "latest_start_seconds_from_open": l1,
                      "before": {"setup_deadline": setup_deadline, "runner_checks": checks, "runner_pre_payload_bound": RUNNER_PRE_S,
                                 "strict_admission": STRICT_S}},
        "jobs": {
            j1["id"]: {"latest_start_seconds_from_open": l1, "finish_margin_seconds": WINDOW_S - RUNNER_MARGIN_PAD_S - (l1 + j1["timeout_seconds"]),
                       "payload_end": l1 + j1["timeout_seconds"], "timeout_seconds": j1["timeout_seconds"]},
            j2["id"]: {"latest_start_seconds_from_open": l2, "finish_margin_seconds": WINDOW_S - RUNNER_MARGIN_PAD_S - payload_end,
                       "payload_end": payload_end, "timeout_seconds": j2["timeout_seconds"]},
        },
        "setup_reservations": head, "between_jobs": between, "after_payload": tail,
        "export_bytes_reserved": out_bytes, "export_hops": hops, "input_bytes_reserved": in_bytes, "input_hops": in_hops,
        "transfer_reference_bytes_per_second": HOP_BYTES_PER_S,
    }
    assert budget["setup_slack"] >= 0, f"{node}: setup reservations {budget['setup_reserved']} s exceed the setup deadline {setup_deadline} s"
    assert l2 <= WINDOW_S - RUNNER_NO_NEW_WORK_S, f"{node}: the runner's no-new-work cutoff would bind before job 2's own deadline"
    assert payload_end <= WINDOW_S - RUNNER_STOP_RESERVE_S, f"{node}: payload end after the runner's stop reserve"
    for job in budget["jobs"].values():
        assert 0 <= job["latest_start_seconds_from_open"] <= 6000 and 600 <= job["finish_margin_seconds"] <= 6000, \
            f"{node}: outside the runner's sealed-field ranges: {job}"
    return budget


def self_test() -> None:
    """The window budget at its boundaries: every chain sums exactly to its end, the runner's own admission and stop
    rules hold at L - 1 and refuse at L, and each reservation fits up to the node's slack and fails 1 s past it. The
    memory declarations: budgets inside their resident limits, refused at 1 byte past and at the v1 values."""
    import sys
    me = sys.modules[__name__]

    def jobs(node):
        return [{"id": i, "timeout_seconds": t, "output_budget_bytes": b} for i, t, b in BUDGET_JOBS[node]]

    def fits(**bump):
        keep = {k: getattr(me, k) for k in bump}
        try:
            for k, v in bump.items():
                setattr(me, k, getattr(me, k) + v)
            window_budget(node, jobs(node))
            return True
        except AssertionError:
            return False
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
    report = {}
    for node in ("dgx01", "dgx02"):
        b = window_budget(node, jobs(node))
        (i1, t1, _), (i2, t2, _) = BUDGET_JOBS[node]
        j1, j2 = b["jobs"][i1], b["jobs"][i2]
        l1, l2 = j1["latest_start_seconds_from_open"], j2["latest_start_seconds_from_open"]
        # exact chains, latest first
        check(j2["payload_end"] + sum(b["after_payload"].values()) == b["node_end"] == NODE_END_S[node], f"{node}: tail chain")
        check(l2 + t2 == j2["payload_end"] and l1 + t1 == j1["payload_end"], f"{node}: payload chains")
        check(j1["payload_end"] + sum(b["between_jobs"].values()) == l2, f"{node}: between-jobs chain")
        check(b["setup_deadline"] + RUNNER_PRE_S + STRICT_S + b["readiness"]["before"]["runner_checks"] == l1, f"{node}: readiness chain")
        check(b["setup_reserved"] + b["setup_slack"] == b["setup_deadline"] <= b["setup_ceiling"], f"{node}: setup chain")
        check(b["setup_slack"] >= 0, f"{node}: slack")
        # the runner's rules (run-node-d2-v3.py) with these fields, T0 = 0, expires = WINDOW_S
        for j, t in ((j1, t1), (j2, t2)):
            L, F = j["latest_start_seconds_from_open"], j["finish_margin_seconds"]
            cutoff = min(WINDOW_S - RUNNER_NO_NEW_WORK_S, L)
            stop = min(WINDOW_S - RUNNER_STOP_RESERVE_S, WINDOW_S - F - RUNNER_MARGIN_PAD_S)
            check(cutoff == L and stop == j["payload_end"], f"{node}: the sealed L/F are the binding cutoff and stop")
            check(L - 1 < cutoff and L - 1 + t <= stop, f"{node}: a start at L - 1 is admitted and ends by its stop")
            check(not L < cutoff, f"{node}: a start at L is refused")
            check(0 <= L <= 6000 and 600 <= F <= 6000, f"{node}: the runner's field ranges")
        check(j1["payload_end"] + RUNNER_POST_S + b["readiness"]["before"]["runner_checks"] + RUNNER_PRE_S < l2,
              f"{node}: job 2 reaches docker create strictly before its L")
        # each reservation fits up to the slack and fails 1 s past it; the runner's bounds and the strict second are
        # reserved twice (between the jobs and before readiness; post-payload after each job), so share the slack
        slack = b["setup_slack"]
        for name, times in (("HANDOVER_S", 1), ("CLEANUP_S", 1), ("MARKER_S", 1),
                            ("RUNNER_POST_S", 2), ("RUNNER_PRE_S", 2), ("STRICT_S", 2)):
            most = slack // times
            check(fits(**{name: most}) and not fits(**{name: most + 1}), f"{node}: {name} at the slack boundary")
        report[node] = {"setup_deadline": b["setup_deadline"], "slack": slack, "L": [l1, l2],
                        "F": [j1["finish_margin_seconds"], j2["finish_margin_seconds"]]}
    node = "dgx02"
    slack = report[node]["slack"]
    check(NODE_END_S[node] == 2940, "dgx02 ends at T0 + 2940")
    for extra, want in ((slack, True), (slack + 1, False)):
        longer = jobs(node)
        longer[1]["timeout_seconds"] += extra
        try:
            window_budget(node, longer)
            got = True
        except AssertionError:
            got = False
        check(got == want, f"dgx02: engine timeout + {extra} s")

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
    print(json.dumps({"self_test": "passed", "checks": checked, "nodes": report}))


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
    budget = {}
    for node in ("dgx01", "dgx02"):
        mine = [j for j in jobs if j["node"] == node]
        shape = [(j["id"], j["timeout_seconds"], j["output_budget_bytes"]) for j in mine]
        assert shape == BUDGET_JOBS[node], f"{node}: jobs {shape} differ from BUDGET_JOBS"
        budget[node] = window_budget(node, mine)
        for job in mine:
            fields = budget[node]["jobs"][job["id"]]
            job["latest_start_seconds_from_open"] = fields["latest_start_seconds_from_open"]
            job["finish_margin_seconds"] = fields["finish_margin_seconds"]
    spec = {"schema": 1, "component": "d2-video-vae", "memory_policy": POLICY, "memory_policy_terms": POLICY_TERMS,
            "gate": "per-node", "window_minutes": 95, "jobs": jobs, "window_budget": budget}
    print(json.dumps(spec, indent=1))


if __name__ == "__main__":
    main()
