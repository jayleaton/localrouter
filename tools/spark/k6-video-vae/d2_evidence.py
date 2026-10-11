"""The D2 evidence contract's reference checker and its fixtures (D2-EVIDENCE.md). It reads what the D2 producers write
(d2-allocator-guard.py, k6v-qual) and applies the rules the shared runner's D2-only readers implement: producer
evidence, the oracle's segment sequence, and the attribution probe's PASS / FAIL. It is the specification in code, run
on the host, never a substitute for the runner's own checks; it touches no GPU and no node.

  python -I d2_evidence.py producer DIR [--prefix allocator-SEG-] --kind torch|native --job ID --nonce HEX
  python -I d2_evidence.py segments DIR --job ID --job-nonce HEX --nonces SEG:HEX,...
  python -I d2_evidence.py probe DIR --kind torch|native        (probe-phases.jsonl, allocator-metrics.jsonl,
                                                                 runner-samples.jsonl, phase-reads.json)
  python -I d2_evidence.py fixtures OUT --qual K6V_QUAL [--vae FILE --inputs DIR]   builds and checks every fixture

Exit 0 when the evidence passes, 1 when it fails (the reasons are printed as JSON).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

POLICY = "gb10-unified-d2-v1"
GIB, MIB = 1024**3, 1024**2
TORCH_CAP, NATIVE_CAP = 38 * GIB, 24 * GIB
SEGMENTS = ["env-probe", "s_dec", "f_dec", "s_enc", "n_enc", "n_dec"]
PHASES = ["pre-context", "context", "alloc-4G", "free-4G", "alloc-16G", "free-16G"]
PHASE_X = [0, 0, 4 * GIB, 4 * GIB, 16 * GIB, 16 * GIB]
HOLD_S, SETTLE_S, MIN_SAMPLES, MAX_GAP_S, MAX_EDGE_S, MAX_READ_S = 10.0, 3.0, 12, 1.0, 0.75, 5.0
READY_S, SEGMENT_GAP_S, TERMINAL_S, SAMPLE_STALE_S = 30.0, 30.0, 2.0, 2.0
METRICS_BUDGET = 16 * MIB
RUNNER_FIELDS = ("monotonic_seconds", "memavailable_bytes", "cgroup_current_bytes", "cgroup_peak_bytes",
                 "helper_rss_bytes", "accounted_bytes")


class Fail(Exception):
    pass


def require(ok, message):
    if not ok:
        raise Fail(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"{path.name} missing")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path.name} is not an object")
    return value


def number(value) -> bool:
    return type(value) in (int, float) and value == value and abs(value) != float("inf")


def jsonl(path: Path, limit=METRICS_BUDGET) -> list:
    require(path.is_file() and not path.is_symlink(), f"{path.name} missing")
    data = path.read_bytes()
    require(len(data) <= limit, f"{path.name} oversized")
    require(data.endswith(b"\n"), f"{path.name} truncated (incomplete final line)")
    return [json.loads(line) for line in data.splitlines()]


# --- producer evidence -------------------------------------------------------------------------------------------


def producer(out: Path, prefix: str, kind: str, job: str, nonce: str, segment=None, fixture=False) -> dict:
    """One producer's ready / samples / terminal / reaped record, final (after the producer was reaped)."""
    cap = TORCH_CAP if kind == "torch" else NATIVE_CAP

    def ident(entry, what):
        require(entry.get("schema") == 1 and entry.get("policy") == POLICY and entry.get("job_id") == job
                and entry.get("nonce") == nonce and type(entry.get("pid")) is int and entry["pid"] > 0,
                f"{what}: identity differs (policy / job / nonce / pid)")
        if segment is not None:
            require(entry.get("segment") == segment and entry.get("segment_index") == SEGMENTS.index(segment),
                    f"{what}: segment differs")

    ready = read_json(out / f"{prefix}ready.json")
    ident(ready, "ready")
    require(ready.get("status") == "passed" and ready.get("allocator_cap_bytes") == cap, "ready: preflight failed or cap differs")
    start, at = ready.get("process_start_monotonic_seconds"), ready.get("monotonic_seconds")
    require(number(start) and number(at) and start <= at <= start + READY_S, "ready: not within 30 s of the process start")
    if kind == "torch":
        require(ready.get("allocator_backend") == "native" and ready.get("torch_version") == "2.14.1+cu130"
                and (fixture or ready.get("python_version") == "3.13.16"), "ready: Torch identity differs")
        total, eff, frac, back = (ready.get(k) for k in ("total_bytes", "effective_limit_bytes", "fraction", "readback_fraction"))
        require(type(total) is int and total > 0 and type(eff) is int and 0 < eff <= cap and number(frac) and number(back)
                and 0 < frac <= 1 and abs(frac - back) <= 1e-12 and abs(eff - total * back) <= 1,
                "ready: allocator fraction was not applied / read back")
    else:
        require(ready.get("allocator_backend") == "explicit" and (ready.get("runtime") == "cuda-driver"
                or (fixture and ready.get("runtime") == "sim")), "ready: native runtime differs")
        require(isinstance(ready.get("engine_sha256"), str) and len(ready["engine_sha256"]) == 64,
                "ready: engine identity missing")
    require(type(ready.get("mem_get_info_total_bytes")) is int and ready["mem_get_info_total_bytes"] > 0,
            "ready: memory total unsupported")
    log = out / f"{prefix}metrics.jsonl"
    rows = jsonl(log)
    samples = []
    for row in rows:
        ident(row, "sample")
        require(row["pid"] == ready["pid"], "sample: process differs from ready")
        require("status" not in row, f"sample: producer reported failure: {row.get('error')}")
        t = row.get("monotonic_seconds")
        require(number(t) and (not samples or t > samples[-1]["monotonic_seconds"]), "sample: time did not advance")
        for key in ("free_bytes", "total_bytes", "allocated_bytes", "reserved_bytes"):
            require(type(row.get(key)) is int and row[key] >= 0, f"sample: unsupported {key}")
        require(row["total_bytes"] == ready["mem_get_info_total_bytes"] and row["free_bytes"] <= row["total_bytes"]
                and row["allocated_bytes"] <= row["reserved_bytes"] <= cap, "sample: exceeds the cap or differs")
        require(kind == "torch" or row["allocated_bytes"] == row["reserved_bytes"], "sample: native ledger fields differ")
        require(not samples or t - samples[-1]["monotonic_seconds"] <= SAMPLE_STALE_S, "samples: stale (gap above 2 s)")
        samples.append(row)
    require(len(samples) >= 2, "samples: fewer than 2")
    terminal_path, exit_path = out / f"{prefix}terminal.json", out / f"{prefix}producer-exit.json"
    require(exit_path.is_file(), "reaped record missing")
    terminal, exited = read_json(terminal_path), read_json(exit_path)
    ident(terminal, "terminal")
    ident(exited, "reaped record")
    require(exited["pid"] == terminal["pid"] == ready["pid"], "terminal / reaped PID differs")
    require(exited.get("producer_exit_code") == 0 and terminal.get("payload_exit_code") == 0,
            f"producer exited {exited.get('producer_exit_code')} / payload {terminal.get('payload_exit_code')}")
    require(exited.get("terminal_sha256") == sha256(terminal_path), "terminal receipt hash differs from the reaped record")
    require(terminal.get("status") == "completed" and terminal.get("sampler_stopped") is True, "sampler did not stop / complete")
    last, end, reaped = samples[-1]["monotonic_seconds"], terminal.get("monotonic_seconds"), exited.get("monotonic_seconds")
    require(number(end) and number(reaped) and last <= end <= reaped and end - last <= TERMINAL_S and reaped - end <= TERMINAL_S,
            "terminal / reaped times missing, stale or out of order")
    require(terminal.get("last_sample_monotonic_seconds") == last, "terminal final sample differs")
    require(terminal.get("metrics_bytes") == log.stat().st_size and terminal.get("metrics_sha256") == sha256(log),
            "terminal raw samples changed")
    return {"pid": ready["pid"], "start": start, "ready": at, "first": samples[0]["monotonic_seconds"], "last": last,
            "terminal": end, "reaped": reaped, "samples": len(samples), "peak_reserved": max(r["reserved_bytes"] for r in samples)}


def segments(out: Path, job: str, job_nonce: str, nonces: dict, fixture=False) -> list:
    """The oracle's sealed sequence: every segment, in order, one fresh process each, none overlapping, gaps <= 30 s."""
    require(list(nonces) == SEGMENTS and len(set(nonces.values()) | {job_nonce}) == len(SEGMENTS) + 1,
            "segment nonces are not the sealed list or not fresh")
    seen, previous = [], None
    for seg in SEGMENTS:
        try:
            s = producer(out, f"allocator-{seg}-", "torch", job, nonces[seg], seg, fixture)
        except Fail as error:
            raise Fail(f"segment {seg}: {error}") from None
        for name in ("ready", "terminal", "producer-exit"):
            entry = read_json(out / f"allocator-{seg}-{name}.json")
            require(entry.get("job_nonce") == job_nonce, f"segment {seg}: job nonce differs")
        require(all(s["pid"] != p["pid"] for p in seen), f"segment {seg}: PID reused (not a fresh process)")
        if previous is not None:
            require(s["start"] >= previous["reaped"] and s["first"] > previous["reaped"],
                    f"segment {seg}: started before {SEGMENTS[len(seen) - 1]} was reaped (overlap or reorder)")
            require(s["ready"] - previous["reaped"] <= SEGMENT_GAP_S, f"segment {seg}: more than 30 s after the previous one")
        seen.append(s)
        previous = s
    return seen


# --- attribution probe -------------------------------------------------------------------------------------------


def window(rows, start, end, what):
    """The samples of one phase window [start + 3 s, end], with the count, gap and edge bounds; their median (low)."""
    pick = [r for r in rows if start + SETTLE_S <= r["monotonic_seconds"] <= end]
    require(len(pick) >= MIN_SAMPLES, f"{what}: {len(pick)} samples in the window, fewer than {MIN_SAMPLES}")
    ts = [r["monotonic_seconds"] for r in pick]
    require(all(b - a <= MAX_GAP_S for a, b in zip(ts, ts[1:])), f"{what}: sample gap above {MAX_GAP_S} s")
    require(ts[0] - (start + SETTLE_S) <= MAX_EDGE_S and end - ts[-1] <= MAX_EDGE_S, f"{what}: window edge not covered")
    return pick


def med(rows, key):
    return statistics.median_low([r[key] for r in rows])


def probe(phases: list, allocator: list, runner: list, reads: dict, kind: str) -> dict:
    """PASS / FAIL of one node's probe (D2-EVIDENCE.md, section 3). Integer arithmetic for every bound."""
    checks, errors, mode = [], [], None

    def check(name, x, value, ok, bound):
        checks.append({"name": name, "x_bytes": x, "value": value, "bound": bound, "ok": bool(ok)})

    try:
        require(len(phases) == len(PHASES), f"{len(phases)} phase records, not {len(PHASES)}")
        ident = {k: phases[0].get(k) for k in ("policy", "job_id", "nonce", "pid")}
        require(ident["policy"] == POLICY and ident["job_id"] == ("d2-probe-torch" if kind == "torch" else "d2-probe-native"),
                "phase identity differs")
        for i, p in enumerate(phases):
            require({k: p.get(k) for k in ident} == ident, f"P{i}: identity differs")
            require(p.get("phase") == f"P{i}" and p.get("name") == PHASES[i] and p.get("index") == i
                    and p.get("x_bytes") == PHASE_X[i], f"P{i}: phase record out of order or differs")
            s, e = p.get("start_monotonic_seconds"), p.get("end_monotonic_seconds")
            require(number(s) and number(e) and e - s >= HOLD_S, f"P{i}: held less than {HOLD_S} s")
            require(i == 0 or s >= phases[i - 1]["end_monotonic_seconds"], f"P{i}: overlaps P{i - 1}")
            r = reads.get(f"P{i}")
            require(number(r) and e <= r <= e + MAX_READ_S, f"P{i}: record read more than {MAX_READ_S} s after its end")
        for row in allocator:
            require(row.get("pid") == ident["pid"] and row.get("nonce") == ident["nonce"], "allocator sample identity differs")
        for row in runner:
            require(all(number(row.get(k)) for k in RUNNER_FIELDS), "runner sample fields missing")
        v = {}
        for i, p in enumerate(phases):
            s, e = p["start_monotonic_seconds"], p["end_monotonic_seconds"]
            w = window(runner, s, e, f"P{i} runner samples")
            v[i] = {"mem": med(w, "memavailable_bytes"), "cur": med(w, "cgroup_current_bytes"),
                    "acc": med(w, "accounted_bytes")}
            if i > 0:
                a = window(allocator, s, e, f"P{i} allocator samples")
                v[i]["G"] = med(a, "reserved_bytes" if kind == "torch" else "allocated_bytes")
        check("context share", 0, v[0]["mem"] - v[1]["mem"], v[0]["mem"] - v[1]["mem"] <= 4 * GIB, "<= 4 GiB")
        modes, dA = {}, {}
        for x, a, f in ((4 * GIB, 2, 3), (16 * GIB, 4, 5)):
            D = v[1]["mem"] - v[a]["mem"]
            check("raw host drop", x, D, 100 * D >= 95 * x and 100 * D <= 110 * x + 100 * 512 * MIB, "0.95X..1.10X+512MiB")
            C = v[a]["cur"] - v[1]["cur"]
            modes[x] = "charged" if 100 * C >= 95 * x else "uncharged" if 100 * C <= 5 * x else None
            check("attribution", x, C, modes[x] is not None, "C >= 0.95X (charged) or C <= 0.05X (uncharged)")
            G = v[a]["G"] - v[1]["G"]
            check("guard quantity", x, G, x <= G <= x + 64 * MIB if kind == "torch" else G == x,
                  "X..X+64MiB" if kind == "torch" else "== X")
            dA[x] = v[a]["acc"] - v[1]["acc"]
            check("accounted, no under-count", x, dA[x], 100 * dA[x] >= 95 * x, ">= 0.95X")
            check("no double count", x, dA[x], 100 * dA[x] <= 110 * x + 100 * 512 * MIB, "<= 1.10X+512MiB")
            check("recovery MemAvailable", x, v[f]["mem"] - v[1]["mem"], v[f]["mem"] >= v[1]["mem"] - 512 * MIB, ">= P1 - 512MiB")
            check("recovery memory.current", x, v[f]["cur"] - v[1]["cur"], v[f]["cur"] <= v[1]["cur"] + 256 * MIB, "<= P1 + 256MiB")
        mode = modes[4 * GIB] if modes[4 * GIB] == modes[16 * GIB] else None
        check("same attribution mode", 0, [modes[4 * GIB], modes[16 * GIB]], mode is not None, "4 GiB and 16 GiB alike")
        lin = dA[16 * GIB] - dA[4 * GIB]
        check("linearity", 0, lin, 100 * abs(lin - 12 * GIB) <= 5 * 12 * GIB, "12 GiB +- 5%")
    except Fail as error:
        errors.append(str(error))
    ok = not errors and checks and all(c["ok"] for c in checks)
    return {"verdict": "PASS" if ok else "FAIL", "mode": mode if ok else None, "errors": errors,
            "failed": [f"{c['name']}@{c['x_bytes']}" for c in checks if not c["ok"]], "checks": checks}


def probe_dir(d: Path, kind: str) -> dict:
    try:
        phases = jsonl(d / "probe-phases.jsonl", 1 << 20)
        allocator = [r for r in jsonl(d / "allocator-metrics.jsonl") if "status" not in r]
        runner = jsonl(d / "runner-samples.jsonl", 64 * MIB)
        reads = read_json(d / "phase-reads.json")
    except Fail as error:
        return {"verdict": "FAIL", "mode": None, "errors": [str(error)], "failed": [], "checks": []}
    return probe(phases, allocator, runner, reads, kind)


# --- fixtures ----------------------------------------------------------------------------------------------------


def load_guard():
    spec = importlib.util.spec_from_file_location("d2_guard", Path(__file__).with_name("d2-allocator-guard.py"))
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard


def run_sequence(guard, out: Path, job: str, nonces: list, order: list, exits: dict, payload_s=1.2):
    """The launcher's loop on fakes: each segment a fresh (forked) process running the guard's own code, reaped, then
    its reaped record written by the guard's --record-exit path."""
    out.mkdir(parents=True)
    env = guard.fake_environ(out, job, nonces)
    payload = out.parent / "payload.py"
    if not payload.exists():
        payload.write_text("import sys, time\ntime.sleep(float(sys.argv[1]))\nsys.exit(int(sys.argv[2]))\n")
    for seg in order:
        pid = os.fork()
        if pid == 0:
            code = 70
            try:
                guard.T_START = time.monotonic()
                guard.env_probe = lambda torch, root: print(json.dumps({"manifest": "fake"}))
                args = [] if seg == "env-probe" else [str(payload), str(payload_s), str(exits.get(seg, 0))]
                if seg == "env-probe":
                    time.sleep(payload_s)
                code = guard.run_segment(env, seg, args, guard.FakeTorch())
            finally:
                os._exit(code)
        _, status = os.waitpid(pid, 0)
        guard.record_exit(env, seg, os.waitstatus_to_exitcode(status))
    return env


def rewrite(out: Path, prefix: str, edit_ready=None, edit_rows=None, edit_terminal=None, drop_exit=False):
    """Mutates one producer's files, then re-binds the chain (metrics -> terminal -> reaped record) so that only the
    intended defect remains."""
    if edit_ready:
        p = out / f"{prefix}ready.json"
        p.write_text(json.dumps(edit_ready(json.loads(p.read_text()))) + "\n")
    log = out / f"{prefix}metrics.jsonl"
    if edit_rows:
        rows = edit_rows([json.loads(x) for x in log.read_text().splitlines()])
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    t = out / f"{prefix}terminal.json"
    term = json.loads(t.read_text())
    term.update(metrics_bytes=log.stat().st_size, metrics_sha256=sha256(log))
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    term["last_sample_monotonic_seconds"] = rows[-1]["monotonic_seconds"]
    if edit_terminal:
        term = edit_terminal(term)
    t.write_text(json.dumps(term) + "\n")
    e = out / f"{prefix}producer-exit.json"
    if drop_exit:
        e.unlink()
    else:
        ex = json.loads(e.read_text())
        ex["terminal_sha256"] = sha256(t)
        if edit_terminal and "pid" in term:
            ex["pid"] = term["pid"]
        e.write_text(json.dumps(ex) + "\n")


def shift(out: Path, prefix: str, dt: float):
    """Moves one producer's times by dt (a late process), keeping its chain bound."""
    def rows(rs):
        return [{**r, "monotonic_seconds": r["monotonic_seconds"] + dt} for r in rs]

    def ready(r):
        return {**r, "monotonic_seconds": r["monotonic_seconds"] + dt,
                "process_start_monotonic_seconds": r["process_start_monotonic_seconds"] + dt}

    def term(t):
        return {**t, "monotonic_seconds": t["monotonic_seconds"] + dt}
    rewrite(out, prefix, ready, rows, term)
    e = out / f"{prefix}producer-exit.json"
    ex = json.loads(e.read_text())
    ex["monotonic_seconds"] += dt
    e.write_text(json.dumps(ex) + "\n")


def synthetic_probe(d: Path, kind: str, mode="charged", phase_edit=None, runner_edit=None, alloc_edit=None,
                    reads_edit=None, records_edit=None):
    """A probe's evidence with exact values: phases of 10 s from t=1000, samples every 0.5 s at .25 / .75 (never on a
    boundary), constant within a phase, so each window's median is the phase's value."""
    d.mkdir(parents=True)
    job = "d2-probe-torch" if kind == "torch" else "d2-probe-native"
    ident = {"schema": 1, "policy": POLICY, "nonce": "ab" * 16, "job_id": job, "pid": 4242}
    t0 = 1000.0
    records = [{**ident, "phase": f"P{i}", "name": PHASES[i], "index": i, "x_bytes": PHASE_X[i],
                "start_monotonic_seconds": t0 + 10 * i, "end_monotonic_seconds": t0 + 10 * (i + 1)} for i in range(6)]
    base = 100 * GIB
    if mode == "charged":   # CUDA memory charged to the container's cgroup
        mem = [base, base - GIB, base - GIB - 4 * GIB - 100 * MIB, base - GIB - 50 * MIB,
               base - GIB - 16 * GIB - 200 * MIB, base - GIB - 60 * MIB]
        cur = [2 * GIB, 3 * GIB, 7 * GIB + 50 * MIB, 3 * GIB + 100 * MIB, 19 * GIB + 80 * MIB, 3 * GIB + 120 * MIB]
    else:                   # not charged: visible only as the host drop
        mem = [base, base - 2 * GIB, base - 2 * GIB - 4 * GIB - 100 * MIB, base - 2 * GIB - 50 * MIB,
               base - 2 * GIB - 16 * GIB - 200 * MIB, base - 2 * GIB - 60 * MIB]
        cur = [GIB, GIB + 10 * MIB, GIB + 20 * MIB, GIB + 15 * MIB, GIB + 30 * MIB, GIB + 20 * MIB]
    helper = 300 * MIB
    G = [None, 0, 4 * GIB + (2 * MIB if kind == "torch" else 0), 0, 16 * GIB + (2 * MIB if kind == "torch" else 0), 0]
    table = {"mem": mem, "cur": cur, "G": G, "acc": [None] * 6}
    if phase_edit:
        phase_edit(table)
    runner, alloc, peak = [], [], 0
    t = t0 + 0.25 - 1.0
    while t < t0 + 60:
        i = max(0, min(5, int((t - t0) // 10)))
        peak = max(peak, table["cur"][i])
        acc = max(peak + helper + 4 * GIB, (base - table["mem"][i]) + 4 * GIB)
        runner.append({"monotonic_seconds": t, "memavailable_bytes": table["mem"][i],
                       "cgroup_current_bytes": table["cur"][i], "cgroup_peak_bytes": peak, "helper_rss_bytes": helper,
                       "accounted_bytes": table["acc"][i] if table["acc"][i] is not None else acc})
        if i >= 1 and t >= t0 + 10:
            g = table["G"][i]
            alloc.append({**ident, "free_bytes": 64 * GIB, "total_bytes": 128 * GIB,
                          "allocated_bytes": g if kind == "native" else max(0, g - 2 * MIB), "reserved_bytes": g,
                          "monotonic_seconds": t})
        t += 0.5
    reads = {f"P{i}": r["end_monotonic_seconds"] + 0.5 for i, r in enumerate(records)}
    for edit, value in ((runner_edit, runner), (alloc_edit, alloc), (reads_edit, reads), (records_edit, records)):
        if edit:
            edit(value)
    (d / "probe-phases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (d / "allocator-metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in alloc))
    (d / "runner-samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runner))
    (d / "phase-reads.json").write_text(json.dumps(reads) + "\n")


def drop_window(rows, phase, keep):
    """Keeps only the samples of `phase`'s window selected by keep(k, t) (k: index within the window)."""
    s = 1000.0 + 10 * phase + SETTLE_S
    e = 1000.0 + 10 * (phase + 1)
    inside = [r for r in rows if s <= r["monotonic_seconds"] <= e]
    gone = {id(r) for k, r in enumerate(inside) if not keep(k, r["monotonic_seconds"])}
    rows[:] = [r for r in rows if id(r) not in gone]


def ceil_div(a, b):
    return -(-a // b)


def probe_cases():
    """(name, kind, kwargs, expected verdict, the check expected to fail): every bound at its edge, then just past it."""
    X4, X16 = 4 * GIB, 16 * GIB
    lo4 = ceil_div(95 * X4, 100)                 # 0.95 X, rounded up: the smallest passing integer
    hi4 = (110 * X4 + 100 * 512 * MIB) // 100   # 1.10 X + 512 MiB, rounded down: the largest passing integer

    def mem2(D):        # P2 MemAvailable such that D == value
        return lambda t: t["mem"].__setitem__(2, t["mem"][1] - D)

    def cur2(C):
        return lambda t: t["cur"].__setitem__(2, t["cur"][1] + C)

    def acc2(dA):       # the runner's accounted value in P2 relative to P1's
        def edit(t):
            p1 = max(t["cur"][0], t["cur"][1]) + 300 * MIB + 4 * GIB
            p1 = max(p1, (100 * GIB - t["mem"][1]) + 4 * GIB)
            t["acc"][1], t["acc"][2] = p1, p1 + dA
        return edit

    def g2(g):
        return lambda t: t["G"].__setitem__(2, g)

    def acc24(d4, d16):
        def edit(t):
            acc2(d4)(t)
            t["acc"][4] = t["acc"][1] + d16
        return edit
    lin_hi = (100 * 12 * GIB + 5 * 12 * GIB) // 100
    yield "pass-charged", "torch", {}, "PASS", None
    yield "pass-uncharged", "torch", {"mode": "uncharged"}, "PASS", None
    yield "pass-native", "native", {}, "PASS", None
    yield "host-drop-at-0.95X", "torch", {"phase_edit": mem2(lo4)}, "PASS", None
    yield "host-drop-below-0.95X", "torch", {"phase_edit": mem2(lo4 - 1)}, "FAIL", "raw host drop@4294967296"
    yield "host-drop-at-1.10X+512MiB", "torch", {"phase_edit": mem2(hi4)}, "PASS", None
    yield "host-drop-above-1.10X+512MiB", "torch", {"phase_edit": mem2(hi4 + 1)}, "FAIL", "raw host drop@4294967296"
    yield "charged-at-0.95X", "torch", {"phase_edit": cur2(lo4)}, "PASS", None
    yield "attribution-between-modes", "torch", {"phase_edit": cur2(lo4 - 1)}, "FAIL", "attribution@4294967296"
    yield "uncharged-at-0.05X", "torch", {"mode": "uncharged", "phase_edit": cur2(5 * X4 // 100)}, "PASS", None
    yield "uncharged-above-0.05X", "torch", {"mode": "uncharged", "phase_edit": cur2(5 * X4 // 100 + 1)}, "FAIL", "attribution@4294967296"
    yield "mixed-modes", "torch", {"phase_edit": lambda t: t["cur"].__setitem__(4, t["cur"][1])}, "FAIL", "same attribution mode@0"
    yield "torch-G-at-X+64MiB", "torch", {"phase_edit": g2(X4 + 64 * MIB)}, "PASS", None
    yield "torch-G-above-X+64MiB", "torch", {"phase_edit": g2(X4 + 64 * MIB + 1)}, "FAIL", "guard quantity@4294967296"
    yield "torch-G-below-X", "torch", {"phase_edit": g2(X4 - 1)}, "FAIL", "guard quantity@4294967296"
    yield "native-G-not-X", "native", {"phase_edit": g2(X4 + 1)}, "FAIL", "guard quantity@4294967296"
    yield "accounted-at-0.95X", "torch", {"phase_edit": acc2(lo4)}, "PASS", None
    yield "accounted-under-count", "torch", {"phase_edit": acc2(lo4 - 1)}, "FAIL", "accounted, no under-count@4294967296"
    yield "accounted-at-1.10X+512MiB", "torch", {"phase_edit": acc24(hi4, hi4 + 12 * GIB)}, "PASS", None
    yield "accounted-double-count", "torch", {"phase_edit": acc2(hi4 + 1)}, "FAIL", "no double count@4294967296"
    yield "linearity-at-5%", "torch", {"phase_edit": acc24(X4, X4 + lin_hi)}, "PASS", None
    yield "linearity-past-5%", "torch", {"phase_edit": acc24(X4, X4 + lin_hi + 1)}, "FAIL", "linearity@0"
    yield "recovery-memavail-at-512MiB", "torch", {"phase_edit": lambda t: t["mem"].__setitem__(3, t["mem"][1] - 512 * MIB)}, "PASS", None
    yield "recovery-memavail-short", "torch", {"phase_edit": lambda t: t["mem"].__setitem__(3, t["mem"][1] - 512 * MIB - 1)}, "FAIL", "recovery MemAvailable@4294967296"
    yield "recovery-cgroup-at-256MiB", "torch", {"phase_edit": lambda t: t["cur"].__setitem__(5, t["cur"][1] + 256 * MIB)}, "PASS", None
    yield "recovery-cgroup-short", "torch", {"phase_edit": lambda t: t["cur"].__setitem__(5, t["cur"][1] + 256 * MIB + 1)}, "FAIL", "recovery memory.current@17179869184"
    yield "context-share-at-4GiB", "torch", {"phase_edit": lambda t: [t["mem"].__setitem__(i, t["mem"][i] - 3 * GIB) for i in range(1, 6)]}, "PASS", None
    yield "context-share-above-4GiB", "torch", {"phase_edit": lambda t: [t["mem"].__setitem__(i, t["mem"][i] - 3 * GIB - 1) for i in range(1, 6)]}, "FAIL", "context share@0"
    yield "samples-12", "torch", {"runner_edit": lambda r: drop_window(r, 2, lambda k, t: k not in (5, 9))}, "PASS", None
    yield "samples-11", "torch", {"runner_edit": lambda r: drop_window(r, 2, lambda k, t: k not in (3, 6, 9))}, "FAIL", "P2 runner samples: 11 samples"
    yield "gap-1.0s", "torch", {"runner_edit": lambda r: drop_window(r, 2, lambda k, t: k != 6)}, "PASS", None
    yield "gap-1.5s", "torch", {"runner_edit": lambda r: drop_window(r, 2, lambda k, t: k not in (6, 7))}, "FAIL", "P2 runner samples: sample gap"
    yield "edge-0.75s", "torch", {"runner_edit": lambda r: drop_window(r, 4, lambda k, t: k != 0)}, "PASS", None
    yield "edge-1.25s", "torch", {"runner_edit": lambda r: drop_window(r, 4, lambda k, t: k > 1)}, "FAIL", "P4 runner samples: window edge"
    yield "edge-late-end", "torch", {"runner_edit": lambda r: drop_window(r, 3, lambda k, t: t < 1039.0)}, "FAIL", "P3 runner samples: window edge"
    yield "read-at-5s", "torch", {"reads_edit": lambda r: r.__setitem__("P4", 1050.0 + 5.0)}, "PASS", None
    yield "read-late", "torch", {"reads_edit": lambda r: r.__setitem__("P4", 1050.0 + 5.25)}, "FAIL", "P4: record read"
    yield "phase-missing", "torch", {"records_edit": lambda r: r.pop(5)}, "FAIL", "5 phase records"
    yield "phase-reordered", "torch", {"records_edit": lambda r: r.__setitem__(slice(2, 4), [r[3], r[2]])}, "FAIL", "P2: phase record out of order"
    yield "phase-nonce-differs", "torch", {"records_edit": lambda r: r[3].__setitem__("nonce", "cd" * 16)}, "FAIL", "P3: identity differs"
    yield "allocator-samples-missing", "torch", {"alloc_edit": lambda a: a.__setitem__(slice(None), [x for x in a if not 1043 <= x["monotonic_seconds"] <= 1050])}, "FAIL", "P4 allocator samples"
    yield "runner-sample-field-missing", "torch", {"runner_edit": lambda r: r[40].pop("accounted_bytes")}, "FAIL", "runner sample fields missing"


def expect(results, name, got_ok, want_ok, reason, detail):
    results.append({"fixture": name, "expected": "PASS" if want_ok else "FAIL", "got": "PASS" if got_ok else "FAIL",
                    "reason": detail, "expected_reason": reason,
                    "ok": got_ok == want_ok and (want_ok or reason is None or reason in detail)})


def fixtures(out: Path, qual: Path, vae: Path | None, inputs: Path | None) -> int:
    out = out.resolve()
    require(not out.exists(), f"{out} exists")
    out.mkdir(parents=True)
    results = []
    guard = load_guard()
    job, nonces = "d2-oracle-short", [hashlib.md5(f"seg-{s}".encode()).hexdigest() for s in SEGMENTS]
    nmap = dict(zip(SEGMENTS, nonces))

    def seg_check(d, env, label, want_ok, reason):
        try:
            segments(d, job, env["K6_MEMORY_NONCE"], nmap, fixture=True)
            expect(results, label, True, want_ok, reason, "")
        except Fail as error:
            expect(results, label, False, want_ok, reason, str(error))

    # oracle segments: real guard code in forked processes, fake Torch
    seq = out / "segments"
    good = seq / "positive"
    env = run_sequence(guard, good, job, nonces, SEGMENTS, {})
    seg_check(good, env, "segments/positive", True, None)
    run_sequence(guard, seq / "reordered", job, nonces, ["env-probe", "s_dec", "s_enc", "f_dec", "n_enc", "n_dec"], {})
    seg_check(seq / "reordered", env, "segments/reordered", False, "segment s_enc: started before f_dec was reaped")
    run_sequence(guard, seq / "payload-exit-nonzero", job, nonces, SEGMENTS, {"n_enc": 3})
    seg_check(seq / "payload-exit-nonzero", env, "segments/payload-exit-nonzero", False, "segment n_enc: producer exited 3")
    run_sequence(guard, seq / "missing-segment", job, nonces, SEGMENTS[:-1], {})
    seg_check(seq / "missing-segment", env, "segments/missing-segment", False, "segment n_dec: allocator-n_dec-ready.json missing")
    run_sequence(guard, seq / "stale-samples-base", job, nonces, SEGMENTS, {}, payload_s=4.2)

    def mutate(name, fn, reason, base=good):
        d = seq / name
        shutil.copytree(base, d)
        fn(d)
        seg_check(d, env, f"segments/{name}", False, reason)

    p = "allocator-s_enc-"
    mutate("stale-nonce", lambda d: rewrite(d, p, lambda r: {**r, "nonce": nonces[2]}, lambda rs: [{**r, "nonce": nonces[2]} for r in rs],
                                            lambda t: {**t, "nonce": nonces[2]}), "segment s_enc: ready: identity differs")
    mutate("reused-pid", lambda d: rewrite(d, p, lambda r: {**r, "pid": read_json(d / "allocator-f_dec-ready.json")["pid"]},
                                           lambda rs: [{**r, "pid": read_json(d / "allocator-f_dec-ready.json")["pid"]} for r in rs],
                                           lambda t: {**t, "pid": read_json(d / "allocator-f_dec-ready.json")["pid"]}),
           "segment s_enc: PID reused")
    mutate("overlap", lambda d: shift(d, p, -(read_json(d / f"{p}ready.json")["process_start_monotonic_seconds"]
                                              - read_json(d / "allocator-f_dec-producer-exit.json")["monotonic_seconds"]) - 0.5),
           "segment s_enc: started before f_dec was reaped")
    mutate("gap-over-30s", lambda d: [shift(d, f"allocator-{s}-", 31.0) for s in SEGMENTS[3:]], "segment s_enc: more than 30 s")
    mutate("sampler-not-stopped", lambda d: rewrite(d, p, edit_terminal=lambda t: {**t, "sampler_stopped": False}),
           "segment s_enc: sampler did not stop")
    mutate("cap-exceeded", lambda d: rewrite(d, p, edit_rows=lambda rs: rs[:1] + [{**rs[1], "reserved_bytes": TORCH_CAP + 1}] + rs[2:]),
           "segment s_enc: sample: exceeds the cap")
    mutate("stale-samples", lambda d: rewrite(d, p, edit_rows=lambda rs: rs[:2] + rs[5:]), "segment s_enc: samples: stale",
           base=seq / "stale-samples-base")
    mutate("truncated-log", lambda d: (d / f"{p}metrics.jsonl").write_bytes((d / f"{p}metrics.jsonl").read_bytes()[:-1]),
           "segment s_enc: allocator-s_enc-metrics.jsonl truncated")
    mutate("log-line-dropped", lambda d: (d / f"{p}metrics.jsonl").write_text(
        "".join((d / f"{p}metrics.jsonl").read_text().splitlines(keepends=True)[:-1])), "segment s_enc: terminal")
    mutate("missing-terminal", lambda d: (d / f"{p}terminal.json").unlink(), "segment s_enc: allocator-s_enc-terminal.json missing")
    mutate("missing-reaped-record", lambda d: (d / f"{p}producer-exit.json").unlink(), "segment s_enc: reaped record missing")
    mutate("terminal-changed-after-reap", lambda d: (d / f"{p}terminal.json").write_text(
        (d / f"{p}terminal.json").read_text().replace('"completed"', '"completed" ')), "segment s_enc: terminal receipt hash differs")
    mutate("ready-late", lambda d: rewrite(d, p, lambda r: {**r, "process_start_monotonic_seconds": r["monotonic_seconds"] - 30.5}),
           "segment s_enc: ready: not within 30 s")
    shutil.rmtree(seq / "stale-samples-base")   # the long-payload base; its mutation is the fixture
    (seq / "payload.py").unlink()

    # native producer: the real k6v-qual (sim: no GPU), the probe and, with --vae/--inputs, the engine path
    nat = out / "native"
    nonce = "0123456789abcdef0123456789abcdef"

    def native(name, args, jid):
        d = nat / name
        d.mkdir(parents=True)
        e = {"PATH": "/usr/bin:/bin", "K6_MEMORY_POLICY": POLICY, "K6_MEMORY_NONCE": nonce, "K6_JOB_ID": jid,
             "K6_OUTPUT": str(d), "K6_ALLOCATOR_RECEIPT": str(d / "allocator-ready.json"),
             "K6_ALLOCATOR_LOG": str(d / "allocator-metrics.jsonl")}
        rc = subprocess.run([str(qual), *args], env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        subprocess.run([str(qual), "--record-exit", str(rc)], env=e, check=True)
        return d

    def nat_check(d, jid, label, want_ok, reason, fixture=True):
        try:
            producer(d, "allocator-", "native", jid, nonce, fixture=fixture)
            expect(results, label, True, want_ok, reason, "")
        except Fail as error:
            expect(results, label, False, want_ok, reason, str(error))

    d = native("probe-sim", ["--probe", "--sim", "--hold-ms", "1200"], "d2-probe-native")
    nat_check(d, "d2-probe-native", "native/probe-sim", True, None)
    nat_check(d, "d2-probe-native", "native/probe-sim-as-production", False, "ready: native runtime differs", fixture=False)
    phases = jsonl(d / "probe-phases.jsonl")
    term = read_json(d / "allocator-terminal.json")
    expect(results, "native/probe-sim-phases", [p["name"] for p in phases] == PHASES and [p["x_bytes"] for p in phases] == PHASE_X
           and term.get("phases_sha256") == sha256(d / "probe-phases.jsonl"), True, None, "phase records sealed in the terminal")
    for name, rows, reason in (("ledger-over-cap", lambda rs: rs[:1] + [{**rs[1], "allocated_bytes": NATIVE_CAP + 1, "reserved_bytes": NATIVE_CAP + 1}] + rs[2:], "sample: exceeds the cap"),
                               ("ledger-fields-differ", lambda rs: rs[:1] + [{**rs[1], "allocated_bytes": rs[1]["reserved_bytes"] - 1 if rs[1]["reserved_bytes"] else 0, "reserved_bytes": max(1, rs[1]["reserved_bytes"])}] + rs[2:], "sample: native ledger fields differ")):
        m = nat / name
        shutil.copytree(d, m)
        rewrite(m, "allocator-", edit_rows=rows)
        nat_check(m, "d2-probe-native", f"native/{name}", False, reason)
    if vae and inputs:
        d = native("engine-sim", ["--sim", "--vae", str(vae), "--inputs", str(inputs), "--out", str(nat / "engine-sim-records"), "--cases", "s_dec"], "d2-engine")
        nat_check(d, "d2-engine", "native/engine-sim", True, None)
        shutil.rmtree(nat / "engine-sim-records")

    # Torch probe: the guard's own probe path (forked, fake Torch, short holds) for the file schema
    tp = out / "torch-probe-run"
    tp.mkdir()
    pid = os.fork()
    if pid == 0:
        code = 70
        try:
            guard.T_START = time.monotonic()
            fake = guard.FakeTorch()
            code = guard.run_probe(guard.fake_environ(tp, "d2-probe-torch"), lambda: fake, hold=1.2)
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    penv = guard.fake_environ(tp, "d2-probe-torch")
    guard.record_exit(penv, "-", os.waitstatus_to_exitcode(status))
    try:
        producer(tp, "allocator-", "torch", "d2-probe-torch", penv["K6_MEMORY_NONCE"], fixture=True)
        ph = jsonl(tp / "probe-phases.jsonl")
        require([p["name"] for p in ph] == PHASES and [p["x_bytes"] for p in ph] == PHASE_X
                and read_json(tp / "allocator-terminal.json").get("phases_sha256") == sha256(tp / "probe-phases.jsonl"),
                "torch probe phases differ")
        expect(results, "torch-probe-run", True, True, None, "")
    except Fail as error:
        expect(results, "torch-probe-run", False, True, None, str(error))

    # probe verdicts: synthetic, exact boundaries
    for name, kind, kw, want, reason in probe_cases():
        d = out / "probe" / name
        synthetic_probe(d, kind, **kw)
        v = probe_dir(d, kind)
        (d / "VERDICT.json").write_text(json.dumps(v, indent=1) + "\n")
        detail = "; ".join(v["errors"] + v["failed"])
        expect(results, f"probe/{name}", v["verdict"] == "PASS", want == "PASS", reason, detail)
    (out / "EXPECTED.json").write_text(json.dumps(results, indent=1) + "\n")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in files))
    bad = [r for r in results if not r["ok"]]
    print(json.dumps({"fixtures": len(results), "as_expected": len(results) - len(bad), "unexpected": bad,
                      "sha256sums": sha256(out / "SHA256SUMS")}, indent=1))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("producer")
    p.add_argument("dir", type=Path)
    p.add_argument("--prefix", default="allocator-")
    p.add_argument("--kind", choices=("torch", "native"), required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--nonce", required=True)
    s = sub.add_parser("segments")
    s.add_argument("dir", type=Path)
    s.add_argument("--job", required=True)
    s.add_argument("--job-nonce", required=True)
    s.add_argument("--nonces", required=True)
    q = sub.add_parser("probe")
    q.add_argument("dir", type=Path)
    q.add_argument("--kind", choices=("torch", "native"), required=True)
    f = sub.add_parser("fixtures")
    f.add_argument("out", type=Path)
    f.add_argument("--qual", type=Path, required=True)
    f.add_argument("--vae", type=Path)
    f.add_argument("--inputs", type=Path)
    a = ap.parse_args()
    try:
        if a.cmd == "producer":
            print(json.dumps(producer(a.dir, a.prefix, a.kind, a.job, a.nonce)))
        elif a.cmd == "segments":
            nonces = dict(x.split(":", 1) for x in a.nonces.split(","))
            print(json.dumps(segments(a.dir, a.job, a.job_nonce, nonces)))
        elif a.cmd == "probe":
            v = probe_dir(a.dir, a.kind)
            print(json.dumps(v, indent=1))
            return 0 if v["verdict"] == "PASS" else 1
        else:
            return fixtures(a.out, a.qual, a.vae, a.inputs)
    except Fail as error:
        print(json.dumps({"verdict": "FAIL", "error": str(error)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
