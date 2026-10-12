#!/usr/bin/env python3
"""The D2 Torch allocator guard (policy gb10-unified-d2-v2; contract in D2-EVIDENCE.md): the entry point of every
Torch process of the D2 oracle jobs, run by the original bundle's interpreter under run-oracle.sh / run-probe.sh's
`env -i` whitelist.

  d2-allocator-guard.py --segment SEG [-- PAYLOAD.py ARGS]   one oracle segment: env-probe (built in: the bundle's
                                                             manifest and pins) or a case (PAYLOAD.py run in-process)
  d2-allocator-guard.py --probe                              the Torch attribution probe, phases P0..P5
  d2-allocator-guard.py --record-exit SEG|- CODE             the launcher's reaped record, after it waited for the
                                                             producer (no Torch; SEG "-" is the probe)
  d2-allocator-guard.py --self-test                          the above on fakes, no CUDA

A segment's process applies the native per-process fraction for a 38 GiB Torch cap with read-back, writes its ready
receipt, samples every second from its own thread (Torch reserved / allocated, mem_get_info), runs the payload, stops
the sampler within 2 s and writes its terminal receipt; a sample that fails ends the process (exit 67), so the payload
cannot finish as a qualified case without evidence. Files: allocator-<SEG>-{ready.json, metrics.jsonl, terminal.json,
producer-exit.json} under K6_OUTPUT, each bound to the segment's own nonce from K6_SEGMENT_NONCES.

The cap limits Torch's caching allocator only: the CUDA context, cuDNN / cuBLAS workspaces and the driver's allocations
are the job's declared unmeasured overhead (its process budget = cap + overhead, admitted by the runner against host
MemAvailable; the container's memory limit is host accounting, not a GPU bound), and the upstream tile planner's mem_get_info is not changed by it (a Torch peak above the cap
fails the case with an OOM, it never re-plans).
"""
import time

T_START = time.monotonic()  # before any import that could load a CUDA library

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402
import platform  # noqa: E402
import queue  # noqa: E402
import re  # noqa: E402
import runpy  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402

POLICY = "gb10-unified-d2-v2"
CAP = 38 * 1024**3
ALLOC_CONFIG = "backend:native,throw_on_cudamalloc_oom:True"
ROOT = Path("/opt/k6-oracle-d2")   # the original D2 bundle, mounted read-only
MANIFEST = "ead020cbc102a5cb10fdbef43f85dfa63aa14585d1835ce0b7f3bc686eac6d1d"
PINS = {"python": "3.13.16", "torch": "2.14.1+cu130", "torchvision": "0.29.1+cu130", "diffusers": "0.40.0",
        "numpy": "2.5.3", "accelerate": "1.15.0", "safetensors": "0.8.0"}
SEGMENTS = ["env-probe", "s_dec", "f_dec", "s_enc", "n_enc", "n_dec"]
SEGMENT_JOBS = ("d2-oracle", "d2-oracle-short")
PROBE_JOB = "d2-probe-torch"
PHASES = ["pre-context", "context", "alloc-4G", "free-4G", "alloc-16G", "free-16G"]
PROBE_BYTES = (4 * 1024**3, 16 * 1024**3)
HOLD_S = 10.0
METRICS_BUDGET = 16 * 1024**2
# the launcher's env -i whitelist: nothing else may reach the process
COMMON_ENV = {"PATH", "HOME", "LANG", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH",
              "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "K6_MEMORY_POLICY", "K6_MEMORY_NONCE", "K6_JOB_ID",
              "K6_OUTPUT"}
OPTIONAL_ENV = {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"}
SEGMENT_ENV = COMMON_ENV | {"K6_SEGMENT_NONCES"}
PROBE_ENV = COMMON_ENV | {"K6_ALLOCATOR_RECEIPT", "K6_ALLOCATOR_LOG"}
HEX32 = re.compile(r"[0-9a-f]{32}")


class GuardError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise GuardError(message)


def integer(value, name, positive=False):
    require(type(value) is int and value >= (1 if positive else 0), "unsupported " + name)
    return value


def bounded(call, seconds):
    """No retry or fabricated metric when a query fails or exceeds its deadline."""
    reply = queue.Queue(maxsize=1)
    start = time.monotonic()

    def work():
        try:
            reply.put((True, call()))
        except BaseException as error:
            reply.put((False, error))

    threading.Thread(target=work, daemon=True, name="d2-allocator-query").start()
    try:
        ok, value = reply.get(timeout=seconds)
    except queue.Empty:
        raise GuardError("allocator query monotonic timeout") from None
    require(time.monotonic() - start < seconds, "allocator query monotonic timeout")
    if not ok:
        raise GuardError("allocator query failed: " + repr(value)) from value
    return value


def write_receipt(path, value):
    require(not path.exists() and not path.is_symlink(), f"{path.name} already exists")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("x") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def identity(environ, segment, check_env=True):
    """The runner's variables, checked; `segment` None is the probe's single producer."""
    if check_env:
        allowed = PROBE_ENV if segment is None else SEGMENT_ENV
        keys = set(environ)
        require(allowed <= keys and keys <= allowed | OPTIONAL_ENV,
                f"environment is not the launcher's whitelist: missing {sorted(allowed - keys)}, "
                f"extra {sorted(keys - allowed - OPTIONAL_ENV)}")
        require(environ["PYTORCH_ALLOC_CONF"] == ALLOC_CONFIG and environ["PYTORCH_CUDA_ALLOC_CONF"] == ALLOC_CONFIG,
                "native allocator configuration differs")
    job, nonce = environ.get("K6_JOB_ID", ""), environ.get("K6_MEMORY_NONCE", "")
    require(environ.get("K6_MEMORY_POLICY") == POLICY and HEX32.fullmatch(nonce), "D2 memory policy / nonce required")
    output = Path(environ.get("K6_OUTPUT", ""))
    require(output.is_absolute() and output.is_dir() and not output.is_symlink() and output.resolve() == output,
            "runner output root required")
    base = {"schema": 1, "policy": POLICY, "job_id": job, "pid": os.getpid()}
    if segment is None:
        require(job == PROBE_JOB, "memory policy job differs")
        for key, name in (("K6_ALLOCATOR_RECEIPT", "allocator-ready.json"), ("K6_ALLOCATOR_LOG", "allocator-metrics.jsonl")):
            require(environ.get(key) == str(output / name), "allocator receipt path differs")
        return output, {**base, "nonce": nonce}, "allocator-"
    require(job in SEGMENT_JOBS and segment in SEGMENTS, "memory policy job / segment differs")
    pairs = [p.split(":", 1) for p in environ.get("K6_SEGMENT_NONCES", "").split(",")]
    require([p[0] for p in pairs] == SEGMENTS and all(len(p) == 2 and HEX32.fullmatch(p[1]) for p in pairs),
            "segment nonces are not the sealed segment list")
    nonces = [p[1] for p in pairs]
    require(len(set(nonces + [nonce])) == len(nonces) + 1, "segment nonces are not fresh")
    index = SEGMENTS.index(segment)
    return output, {**base, "nonce": nonces[index], "job_nonce": nonce, "segment": segment,
                    "segment_index": index}, f"allocator-{segment}-"


class Torch:
    """The guard's view of torch: the queries and the cap (fakes in --self-test)."""

    def __init__(self, torch):
        self.torch = torch

    def sample(self):
        t = self.torch
        free, total = t.cuda.mem_get_info(0)
        free, total = integer(free, "CUDA free bytes"), integer(total, "CUDA total bytes", positive=True)
        reserved = integer(t.cuda.memory_reserved(0), "Torch reserved bytes")
        allocated = integer(t.cuda.memory_allocated(0), "Torch allocated bytes")
        require(free <= total and allocated <= reserved <= CAP, "allocator sample exceeds policy or is inconsistent")
        return {"free_bytes": free, "total_bytes": total, "allocated_bytes": allocated, "reserved_bytes": reserved,
                "monotonic_seconds": time.monotonic()}

    def apply_policy(self):
        t = self.torch
        require(t.cuda.device_count() == 1, "exactly one CUDA device required")
        require(t.cuda.memory.get_allocator_backend() == "native", "native Torch allocator required")
        first = self.sample()
        # native setMemoryFraction rounds fraction * totalGlobalMem, which can differ from mem_get_info's total on GB10
        native_total = integer(t.cuda.get_device_properties(0).total_memory, "native allocator total", positive=True)
        require(native_total > CAP, "native allocator capacity insufficient")
        fraction = math.nextafter(CAP / native_total, 0.0)
        t.cuda.memory.set_per_process_memory_fraction(fraction, 0)
        readback = t.cuda.memory.get_per_process_memory_fraction(0)
        require(type(readback) in (int, float) and math.isfinite(readback) and 0 < readback <= 1
                and abs(readback - fraction) <= 1e-12, "allocator fraction readback differs")
        effective = math.floor(native_total * readback + 0.5)
        require(0 < effective <= CAP, "native allocator readback exceeds 38 GiB")
        last = self.sample()
        require(last["total_bytes"] == first["total_bytes"], "CUDA total changed during preflight")
        self.readback = readback
        return {"allocator_backend": "native", "allocator_cap_bytes": CAP, "total_bytes": native_total,
                "mem_get_info_total_bytes": last["total_bytes"], "fraction": fraction, "readback_fraction": readback,
                "effective_limit_bytes": effective}, [first, last]

    def checked_sample(self, policy):
        t = self.torch
        require(t.cuda.memory.get_allocator_backend() == "native", "allocator backend changed")
        readback = t.cuda.memory.get_per_process_memory_fraction(0)
        require(type(readback) in (int, float) and math.isfinite(readback)
                and abs(readback - policy["readback_fraction"]) <= 1e-12, "allocator fraction changed")
        row = self.sample()
        require(row["total_bytes"] == policy["mem_get_info_total_bytes"], "CUDA total changed")
        return row


class Producer:
    """ready, periodic samples from its own thread, terminal: one per process."""

    def __init__(self, output, base, prefix, torch, period):
        self.output, self.base, self.prefix, self.torch, self.period = output, base, prefix, torch, period
        self.log_path = output / (prefix + "metrics.jsonl")
        require(not (output / (prefix + "ready.json")).exists() and not self.log_path.exists(),
                "allocator evidence already exists")
        self.log = self.log_path.open("x", buffering=1)
        self.lock, self.last, self.stop = threading.Lock(), None, threading.Event()

    def write(self, row):
        with self.lock:
            require(self.log.tell() < METRICS_BUDGET, "allocator metric log budget exceeded")
            if "status" not in row:
                require(self.last is None or row["monotonic_seconds"] > self.last["monotonic_seconds"],
                        "allocator sample time did not advance")
                self.last = {**self.base, **row}
                line = self.last
            else:
                line = {**self.base, **row}
            self.log.write(json.dumps(line, allow_nan=False) + "\n")
            self.log.flush()

    def fail(self, error):
        try:
            self.write({"status": "failed", "monotonic_seconds": time.monotonic(), "error": repr(error)})
        finally:
            os._exit(67)

    def start(self, extra):
        try:
            self.policy, rows = bounded(self.torch.apply_policy, 5.0)
            for row in rows:
                self.write(row)
            write_receipt(self.output / (self.prefix + "ready.json"),
                          {**self.base, **self.policy, "status": "passed", "python_version": platform.python_version(),
                           "torch_version": self.torch.torch.__version__, "sample_period_s": self.period,
                           "process_start_monotonic_seconds": T_START, "monotonic_seconds": time.monotonic(), **extra,
                           "limitation": "Native Torch caching allocator only; the CUDA context, library workspaces "
                                         "and driver allocations are the declared unmeasured overhead."})
        except BaseException as error:
            self.write({"status": "failed", "monotonic_seconds": time.monotonic(), "error": repr(error)})
            self.log.close()
            raise

        def monitor():
            while not self.stop.wait(self.period):
                try:
                    self.write(bounded(lambda: self.torch.checked_sample(self.policy), 1.0))
                except BaseException as error:
                    self.fail(error)

        self.worker = threading.Thread(target=monitor, daemon=True, name="d2-allocator-monitor")
        self.worker.start()

    def finish(self, code, extra=None):
        self.stop.set()
        self.worker.join(timeout=2.0)
        if self.worker.is_alive():
            self.fail(GuardError("allocator sampler failed to stop"))
        try:
            self.write(bounded(lambda: self.torch.checked_sample(self.policy), 1.0))
        except BaseException as error:
            self.write({"status": "failed", "monotonic_seconds": time.monotonic(), "error": repr(error)})
            code = 67
        self.log.close()
        size = self.log_path.stat().st_size
        require(size <= METRICS_BUDGET, "terminal metrics budget exceeded")
        write_receipt(self.output / (self.prefix + "terminal.json"),
                      {**self.base, "status": "completed" if code == 0 else "failed", "payload_exit_code": code,
                       "sampler_stopped": True, "last_sample_monotonic_seconds": self.last["monotonic_seconds"],
                       "monotonic_seconds": time.monotonic(), "metrics_bytes": size,
                       "metrics_sha256": hashlib.sha256(self.log_path.read_bytes()).hexdigest(), **(extra or {})})
        return code


def verify_bundle(root):
    """Every file of the original bundle against its manifest, the manifest against the sealed hash."""
    manifest = root / "MANIFEST.sha256"
    require(hashlib.sha256(manifest.read_bytes()).hexdigest() == MANIFEST, "bundle manifest differs")
    files = 0
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) and not name.startswith("/") and ".." not in name.split("/"),
                "unsupported bundle checksum line")
        path = root / name
        require(path.is_file() and not path.is_symlink(), f"bundle member missing: {name}")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                h.update(block)
        require(h.hexdigest() == digest, f"bundle file differs: {name}")
        files += 1
    return files


def env_probe(torch_module, root):
    """The env-probe segment's payload: the bundle, the interpreter and the pinned versions, exactly."""
    files = verify_bundle(root)
    require(Path(sys.executable).resolve() == root / "python/bin/python3.13", "not the bundle's interpreter")
    import accelerate, diffusers, numpy, safetensors, torchvision  # noqa: E401
    got = {"python": platform.python_version(), "torch": torch_module.__version__,
           "torchvision": torchvision.__version__, "diffusers": diffusers.__version__, "numpy": numpy.__version__,
           "accelerate": accelerate.__version__, "safetensors": safetensors.__version__}
    require(got == PINS, f"oracle stack differs from the pins: { {k: (got[k], v) for k, v in PINS.items() if got[k] != v} }")
    require(Path(torch_module.__file__).resolve().is_relative_to(root / "site-packages"), "torch is not the bundle's")
    print(json.dumps({"manifest": "ok", "files": files, **got, "cuda": torch_module.version.cuda,
                      "gpu": torch_module.cuda.get_device_name(0),
                      "capability": torch_module.cuda.get_device_capability(0)}), flush=True)


def run_segment(environ, segment, payload, torch_module=None, root=ROOT, check_env=True):
    output, base, prefix = identity(environ, segment, check_env)
    require(platform.machine() == "aarch64" or torch_module is not None, "aarch64 required")
    if torch_module is None:
        require(platform.python_version() == PINS["python"], "oracle interpreter identity differs")
        import torch as torch_module
        require(torch_module.__version__ == PINS["torch"] and torch_module.version.cuda == "13.0",
                "oracle Torch / CUDA identity differs")
    producer = Producer(output, base, prefix, Torch(torch_module), 1.0)
    producer.start({"segment_payload": payload[0] if payload else "env-probe"})
    code, original = 0, sys.argv
    try:
        if segment == "env-probe":
            require(not payload, "env-probe takes no payload")
            env_probe(torch_module, root)
        else:
            require(payload and payload[0].endswith(".py"), "a case segment needs its payload script")
            sys.argv = list(payload)
            runpy.run_path(payload[0], run_name="__main__")
    except SystemExit as error:
        code = error.code if type(error.code) is int else (0 if error.code is None else 1)
    except BaseException as error:
        print(f"segment {segment} payload failed: {error!r}", file=sys.stderr, flush=True)
        code = 1
    finally:
        sys.argv = original
    return producer.finish(code)


def run_probe(environ, torch_import=None, hold=HOLD_S, check_env=True):
    """P0 before torch is imported; the allocator evidence starts with the context (P1)."""
    output, base, prefix = identity(environ, None, check_env)
    phases_path = output / "probe-phases.jsonl"
    phases = phases_path.open("x", buffering=1)
    index = 0

    def phase(x, start):
        nonlocal index
        phases.write(json.dumps({**base, "phase": f"P{index}", "name": PHASES[index], "index": index, "x_bytes": x,
                                 "start_monotonic_seconds": start, "end_monotonic_seconds": time.monotonic()},
                                allow_nan=False) + "\n")
        phases.flush()
        os.fsync(phases.fileno())
        index += 1

    t = time.monotonic()
    require("torch" not in sys.modules or torch_import is not None, "torch loaded before P0")
    time.sleep(hold)
    phase(0, t)
    if torch_import is None:
        import torch
        require(platform.python_version() == PINS["python"] and torch.__version__ == PINS["torch"],
                "oracle interpreter / Torch identity differs")
    else:
        torch = torch_import()
    producer = Producer(output, base, prefix, Torch(torch), 0.5)
    producer.start({"probe": "torch"})
    torch.cuda.synchronize()
    t = time.monotonic()
    time.sleep(hold)
    phase(0, t)
    for x in PROBE_BYTES:
        block = torch.empty(x, dtype=torch.uint8, device="cuda")
        block.fill_(1)
        torch.cuda.synchronize()
        t = time.monotonic()
        time.sleep(hold)
        phase(x, t)
        del block
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t = time.monotonic()
        time.sleep(hold)
        phase(x, t)
    phases.close()
    data = phases_path.read_bytes()
    return producer.finish(0, {"phases_bytes": len(data), "phases_sha256": hashlib.sha256(data).hexdigest()})


def record_exit(environ, segment, code):
    """Run by the launcher only after it reaped the producer. Imports no Torch and performs no CUDA operation."""
    require(type(code) is int and 0 <= code <= 255, "invalid reaped producer exit")
    output, base, prefix = identity(environ, None if segment == "-" else segment, check_env=False)
    ready = output / (prefix + "ready.json")
    seen = json.loads(ready.read_text()) if ready.is_file() and not ready.is_symlink() else {}
    require(not seen or (seen.get("nonce") == base["nonce"] and seen.get("job_id") == base["job_id"]),
            "reaped producer identity differs")
    terminal = output / (prefix + "terminal.json")
    digest = None
    if terminal.is_file() and not terminal.is_symlink():
        require(terminal.stat().st_size <= 1024**2, "terminal receipt oversized")
        digest = hashlib.sha256(terminal.read_bytes()).hexdigest()
    write_receipt(output / (prefix + "producer-exit.json"),
                  {**base, "pid": seen.get("pid"), "producer_exit_code": code, "monotonic_seconds": time.monotonic(),
                   "terminal_sha256": digest,
                   "evidence": "sealed launcher waited/reaped the producer before this CPU helper"})
    return 0


class FakeTorch:
    """Enough of torch for the guard's code paths: the cap's arithmetic, samples, the probe's allocations."""
    __version__ = PINS["torch"]
    uint8 = "uint8"

    def __init__(self):
        fake = self

        class Memory:
            fraction = 1.0
            def get_allocator_backend(self): return "native"
            def set_per_process_memory_fraction(self, f, device): self.fraction = f
            def get_per_process_memory_fraction(self, device): return self.fraction

        class Cuda:
            memory = Memory()
            def device_count(self): return 1
            def get_device_properties(self, d): return type("P", (), {"total_memory": 128 * 1024**3})()
            def mem_get_info(self, d): return 120 * 1024**3 - fake.reserved, 120 * 1024**3
            def memory_reserved(self, d): return fake.reserved
            def memory_allocated(self, d): return fake.allocated
            def synchronize(self): pass
            def empty_cache(self): fake.reserved = fake.allocated
            def get_device_name(self, d): return "fake"
            def get_device_capability(self, d): return (12, 1)

        self.cuda, self.reserved, self.allocated = Cuda(), 0, 0
        self.version = type("V", (), {"cuda": "13.0"})()

    def empty(self, n, dtype, device):
        fake = self
        fake.allocated += n
        fake.reserved = max(fake.reserved, fake.allocated + 2 * 1024**2)  # the caching allocator's rounding

        class Block:
            def fill_(self, v): return self
            def __del__(self): fake.allocated -= n
        return Block()


def fake_environ(output, job, segment_nonces=None):
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "site-packages"), "PYTORCH_ALLOC_CONF": ALLOC_CONFIG,
           "PYTORCH_CUDA_ALLOC_CONF": ALLOC_CONFIG, "K6_MEMORY_POLICY": POLICY,
           "K6_MEMORY_NONCE": hashlib.md5(job.encode()).hexdigest(), "K6_JOB_ID": job, "K6_OUTPUT": str(output)}
    if segment_nonces is None:
        env.update(K6_ALLOCATOR_RECEIPT=str(output / "allocator-ready.json"),
                   K6_ALLOCATOR_LOG=str(output / "allocator-metrics.jsonl"))
    else:
        env["K6_SEGMENT_NONCES"] = ",".join(f"{s}:{n}" for s, n in zip(SEGMENTS, segment_nonces))
    return env


def self_test(output=None):
    """The guard's own paths on fakes: two segments and their reaped records, the probe, and refusals."""
    import tempfile
    work = Path(output or tempfile.mkdtemp(prefix="d2-guard-self-test-")).resolve()
    seg_out, probe_out = work / "segments", work / "probe"
    seg_out.mkdir(parents=True), probe_out.mkdir()
    nonces = [hashlib.md5(s.encode()).hexdigest() for s in SEGMENTS]
    env = fake_environ(seg_out, "d2-oracle-short", nonces)
    payload = work / "payload.py"
    payload.write_text("import sys, time\ntime.sleep(float(sys.argv[1]))\nsys.exit(int(sys.argv[2]))\n")
    codes = {}
    for segment, exit_code in (("s_dec", 0), ("f_dec", 3)):
        codes[segment] = run_segment(env, segment, [str(payload), "1.2", str(exit_code)], FakeTorch())
        record_exit(env, segment, codes[segment])
    require(codes == {"s_dec": 0, "f_dec": 3}, f"segment exit codes {codes}")
    refused = 0
    for bad in ({**env, "LD_LIBRARY_PATH": "/x"}, {k: v for k, v in env.items() if k != "PYTHONPATH"},
                {**env, "K6_SEGMENT_NONCES": env["K6_SEGMENT_NONCES"].replace("s_dec:", "s_enc:", 1)},
                {**env, "K6_MEMORY_POLICY": "gb10-unified-v1"}, {**env, "K6_JOB_ID": "k6-d3-a01"}):
        try:
            identity(bad, "s_enc")
        except GuardError:
            refused += 1
    try:   # a segment's evidence is written once
        run_segment(env, "s_dec", [str(payload), "0", "0"], FakeTorch())
    except GuardError:
        refused += 1
    penv = fake_environ(probe_out, PROBE_JOB)
    fake = FakeTorch()
    require(run_probe(penv, lambda: fake, hold=0.6) == 0, "fake probe failed")
    record_exit(penv, "-", 0)
    require(refused == 6, f"refusals {refused}")
    print(json.dumps({"self_test": "passed", "refusals": refused, "CUDA_invocation": False, "output": str(work)}))
    return work


def main(argv):
    if argv[:1] == ["--self-test"] and len(argv) <= 2:
        self_test(argv[1] if len(argv) == 2 else None)
        return 0
    if len(argv) == 3 and argv[0] == "--record-exit":
        return record_exit(os.environ, argv[1], int(argv[2]))
    if argv == ["--probe"]:
        return run_probe(os.environ)
    require(len(argv) >= 2 and argv[0] == "--segment" and (len(argv) == 2 or argv[2] == "--"),
            "usage: d2-allocator-guard.py --segment SEG [-- PAYLOAD.py ARGS] | --probe | --record-exit SEG|- CODE")
    return run_segment(os.environ, argv[1], argv[3:])


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except GuardError as error:
        print("d2 allocator guard refused: " + str(error), file=sys.stderr, flush=True)
        sys.exit(67)
