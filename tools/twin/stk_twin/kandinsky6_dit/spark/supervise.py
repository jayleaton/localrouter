#!/usr/bin/env python3
"""Supervise one node-job step under the GB10 uma-node host guard (owner policy 2026-10-12, rule (c)).

usage: supervise.py --floor-bytes B --record FILE --session-file FILE [--timeout S] [--poll S] [--max-gap S]
                    [--meminfo PATH] [--vmstat PATH] -- ARGV...

Starts ARGV in a new session, with this process as its child subreaper, so every descendant stays in its tree, even
one that double-forks or calls setsid. Then it samples MemAvailable every --poll seconds (0.1). It kills the whole
process tree at once with SIGKILL, with no TERM grace, when:
  - MemAvailable < --floor-bytes;
  - two samples are more than --max-gap seconds apart (0.5): the floor would not have been watched;
  - /proc/meminfo or /proc/vmstat is unreadable;
  - the kernel oom_kill counter rises (after.sh and classify.py attribute the kill from the kernel journal);
  - --timeout seconds have passed;
  - this supervisor gets SIGTERM, SIGINT or SIGHUP.
After the step's own exit it kills and reaps whatever is left in the tree, then shows none remains.

--session-file gets "PID SID" of the step as soon as it starts. node-job's fallback, if this supervisor itself stops
answering, is `kill -KILL -- -SID`. --record gets the supervision record:
  argv, pid, sid, polls, max_gap_s, min_avail, floor_bytes, kill (reason, detect-to-gone seconds) or null, rc,
  leftovers killed after a normal exit, tree_verified.
Exit: the step's exit code; 137 when this supervisor killed it (the record says why); 3 on a supervisor fault (record
written when possible). Stdlib only, Linux only.
"""
import argparse
import ctypes
import json
import os
import signal
import sys
import time

PR_SET_CHILD_SUBREAPER = 36
SCHEMA = "k6-dit-step-supervision-v1"


def mem_available(path):
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise ValueError("no MemAvailable line")


def oom_kills(path):
    with open(path, "rb") as f:
        for line in f:
            if line.startswith(b"oom_kill "):
                return int(line.split()[1])
    raise ValueError("no oom_kill line")


def children_map():
    """ppid -> [pid] for every live process (a process that exits mid-scan is skipped)."""
    kids = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat", "rb") as f:
                st = f.read()
        except OSError:
            continue
        rest = st[st.rindex(b")") + 2:].split()
        kids.setdefault(int(rest[1]), []).append(int(d))
    return kids


def tree(root):
    """Every live descendant of root (root excluded)."""
    kids, out, todo = children_map(), [], [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


EXITED = {}  # pid -> exit code of every child this supervisor reaped (the step and adopted orphans)


def reap():
    """Collect every exited child: the step, and the orphans this subreaper inherits."""
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
        EXITED[pid] = os.waitstatus_to_exitcode(status)


def kill_tree(sid, deadline_s=10.0):
    """SIGKILL the step's session and every descendant of this supervisor until none remains; True if shown gone."""
    me, end = os.getpid(), time.monotonic() + deadline_s
    while True:
        try:
            os.killpg(sid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        left = tree(me)
        for p in left:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        reap()
        if not tree(me):
            return True
        if time.monotonic() > end:
            return False
        time.sleep(0.01)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-bytes", type=int, required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--session-file", required=True)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--poll", type=float, default=0.1)
    ap.add_argument("--max-gap", type=float, default=0.5)
    ap.add_argument("--meminfo", default="/proc/meminfo")
    ap.add_argument("--vmstat", default="/proc/vmstat")
    ap.add_argument("argv", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.argv[1:] if a.argv[:1] == ["--"] else a.argv
    if not cmd or a.floor_bytes <= 0 or not 0 < a.poll < a.max_gap:
        ap.error("need -- ARGV, a positive --floor-bytes and 0 < --poll < --max-gap")
    rec = {"schema": SCHEMA, "argv": cmd, "floor_bytes": a.floor_bytes, "poll_s": a.poll, "max_gap_allowed_s": a.max_gap,
           "timeout_s": a.timeout, "pid": None, "sid": None, "polls": 0, "max_gap_s": 0.0, "min_avail": None,
           "kill": None, "rc": None, "leftovers_killed": [], "tree_verified": False}

    def write():
        tmp = f"{a.record}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, a.record)

    stop = {"sig": None}

    def on_signal(n, _frame):
        stop["sig"] = signal.Signals(n).name

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER)")
        avail, ooms = mem_available(a.meminfo), oom_kills(a.vmstat)
        if avail < a.floor_bytes:
            rec["kill"] = {"reason": f"not started: MemAvailable {avail} B < floor {a.floor_bytes} B", "gone_s": 0.0}
            rec["min_avail"], rec["tree_verified"] = avail, True
            write()
            return 137
    except (OSError, ValueError) as e:
        rec["fault"] = f"setup: {e!r}"
        write()
        return 3
    import subprocess
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, start_new_session=True)
    pid = rec["pid"] = rec["sid"] = p.pid
    with open(a.session_file + ".tmp", "w") as f:
        f.write(f"{pid} {pid}\n")
    os.replace(a.session_file + ".tmp", a.session_file)
    t0 = last = time.monotonic()
    reason = None
    while reason is None:
        now = time.monotonic()
        gap = now - last
        rec["max_gap_s"], last, rec["polls"] = max(rec["max_gap_s"], round(gap, 4)), now, rec["polls"] + 1
        try:
            avail, o = mem_available(a.meminfo), oom_kills(a.vmstat)
        except (OSError, ValueError) as e:
            reason = f"meminfo / vmstat unreadable: {e!r}"
            break
        rec["min_avail"] = avail if rec["min_avail"] is None else min(rec["min_avail"], avail)
        if avail < a.floor_bytes:
            reason = f"floor: MemAvailable {avail} B < {a.floor_bytes} B"
        elif gap > a.max_gap:
            reason = f"supervision gap {gap:.3f} s > {a.max_gap} s"
        elif o != ooms:
            reason = f"kernel oom_kill +{o - ooms} during the step"
        elif a.timeout is not None and now - t0 > a.timeout:
            reason = f"timeout {a.timeout} s"
        elif stop["sig"]:
            reason = f"supervisor got {stop['sig']}"
        else:
            reap()
            if pid in EXITED:
                break
            time.sleep(a.poll)
    if reason is not None:
        t = time.monotonic()
        rec["tree_verified"] = kill_tree(pid)
        rec["kill"] = {"reason": reason, "gone_s": round(time.monotonic() - t, 4)}
        rec["rc"] = p.returncode = EXITED.get(pid)
        write()
        return 137
    rec["rc"] = p.returncode = EXITED[pid]
    rec["leftovers_killed"] = tree(os.getpid())
    rec["tree_verified"] = kill_tree(pid)
    write()
    return rec["rc"] if rec["tree_verified"] else 3


if __name__ == "__main__":
    sys.exit(main())
