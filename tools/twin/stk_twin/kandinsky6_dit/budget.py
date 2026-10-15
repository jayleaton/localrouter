"""The GB10 uma-node Torch budget of a Spark node job step (owner policy 2026-10-12, rule (a)).

On GB10, device memory comes from the same pool as host memory, and a cgroup's memory.max does not contain device
allocations. Every CUDA process a node job starts therefore caps torch's caching allocator at the step's declared
Torch budget with `set_per_process_memory_fraction`, before any allocation. It reads the fraction back and refuses
unless the readback equals what it set and the resulting cap is at most the budget.

node-job.sh sets the environment of each step:
  STK_K6_TORCH_BUDGET_BYTES  the declared per-process Torch budget
  STK_K6_CAP_DIR             where each capped process writes cap-<pid>.json (node-job checks every record after the step)
Outside a node job neither is set and apply() does nothing (dev runs). A step with only one of them set is refused.
host_only() is for a parent process that must never create a CUDA context (gate.py's parent): it is budgeted as a
host process, so it refuses once CUDA is initialized in it.
"""

from __future__ import annotations

import json
import math
import os

ENV_BUDGET = "STK_K6_TORCH_BUDGET_BYTES"
ENV_CAPS = "STK_K6_CAP_DIR"
SCHEMA = "k6-dit-torch-cap-v1"


class CapError(RuntimeError):
    pass


def fraction_for(budget: int, total: int) -> float:
    """The largest fraction whose cap, int(fraction * total) as the caching allocator computes it, is <= budget."""
    if not 0 < budget <= total:
        raise CapError(f"Torch budget {budget} B outside (0, device total {total} B]")
    f = budget / total
    while int(f * total) > budget:
        f = math.nextafter(f, 0.0)
    return f


def apply(where: str, torch=None) -> dict | None:
    """Cap this process's caching allocator at the step's Torch budget; None outside a node job."""
    raw, caps = os.environ.get(ENV_BUDGET), os.environ.get(ENV_CAPS)
    if raw is None and caps is None:
        return None
    if raw is None or caps is None:
        raise CapError(f"{ENV_BUDGET} and {ENV_CAPS} must both be set (got {raw!r}, {caps!r})")
    if not raw.isdigit():
        raise CapError(f"{ENV_BUDGET} must be a positive integer of bytes (got {raw!r})")
    if torch is None:
        import torch
    budget = int(raw)
    torch.cuda.init()
    dev = torch.cuda.current_device()
    reserved = torch.cuda.memory_reserved(dev)
    if reserved != 0:
        raise CapError(f"{where}: {reserved} B already reserved before the cap")
    _free, total = torch.cuda.mem_get_info(dev)  # cudaMemGetInfo's total: what the allocator multiplies the fraction by
    f = fraction_for(budget, total)
    torch.cuda.set_per_process_memory_fraction(f, dev)
    back = float(torch.cuda.get_per_process_memory_fraction(dev))
    rec = {"schema": SCHEMA, "pid": os.getpid(), "where": where, "device": dev, "budget_bytes": budget,
           "device_total": total, "fraction_set": f, "fraction_readback": back, "cap_bytes": int(back * total),
           "reserved_before_cap": reserved}
    if back != f or rec["cap_bytes"] > budget:
        raise CapError(f"{where}: cap readback {back!r} != set {f!r}, or cap {rec['cap_bytes']} B > budget {budget} B")
    path = os.path.join(caps, f"cap-{os.getpid()}.json")
    with open(path, "x") as fh:  # one record per process; a second cap in the same process is refused
        json.dump(rec, fh, indent=1)
    return rec


def host_only(where: str, torch=None) -> None:
    """Refuse if this process holds a CUDA context: its budget is a host allowance with no Torch term."""
    if torch is None:
        import torch
    if torch.cuda.is_initialized():
        raise CapError(f"{where}: CUDA is initialized in a process budgeted as host-only")
