#!/usr/bin/env python3
"""Classify one node's Kandinsky-6 DiT jobs for the production operator: PASS, FAIL_LOCAL or STOP_ALL (sticky).

Ports the STOP_ALL rules of the D1 qualification classifier uma-probe-classify.py
(sha256 db3608630f1214378f30692cf4a46f88787c1b55cb3a7a9882a5fdf023261cf1, RULES_SOURCE_SHA256 below): the
regexes HEADER, KILLED, USERSPACE, XID and the functions field(), parse_kernel(), read_counter() are copied
verbatim (parse_kernel's local class is named JOB_SCOPE_OOM instead of PROBE_SCOPE_OOM). test_classify.py checks it.

usage: classify.py --node N --t0 T0 --vmstat-before FILE --vmstat-after FILE --kernel-log FILE --kernel-rc RC
         --system-log FILE --system-rc RC --cleanup-verified 0|1 --unit NAME [--unit NAME ...]
         --job-result FILE [--job-result FILE ...] --out FILE

Inputs are captured by after.sh outside every job scope:
  vmstat-before  "EPOCH OOM_KILL" written before the node's first job started (/proc/vmstat oom_kill)
  vmstat-after   same, read after this run's units were killed and stopped
  kernel-log     journalctl -k -o short-unix --since @T0           (rc in --kernel-rc)
  system-log     journalctl --system -o short-unix --since @T0 -u earlyoom -u systemd-oomd   (rc in --system-rc)
  unit           one of this run's systemd scope units (e.g. k6-dit-smoke-1900000000.scope); repeat for each
  job-result     a k6-dit-job-result-v1 JSON written by node-job.sh: {"schema", "job", "result", ...}; repeat

Rules (fail closed; the first matching class wins in the order STOP_ALL, FAIL_LOCAL, PASS):
  STOP_ALL  JOURNAL_UNREADABLE       a journal query failed: global safety cannot be shown, so it is not assumed
            OOM_COUNTER_UNREADABLE   either vmstat capture is missing or malformed
            GLOBAL_OOM               a kernel OOM kill whose header is not CONSTRAINT_MEMCG
            FOREIGN_MEMCG_OOM        a memcg OOM kill whose oom_memcg is not one of this run's --unit scopes
            UNATTRIBUTED_OOM_KILL    a "Killed process" line with no preceding oom-kill header
            OOM_EVIDENCE_INCOMPLETE  kernel "Killed process" lines != vmstat oom_kill delta (lost or extra lines)
            USERSPACE_OOM_KILL       earlyoom or systemd-oomd killed or signalled a process
            NVRM_XID                 a GPU driver Xid since T0
            CLEANUP_UNVERIFIED       the after-script could not verify the run's units and processes are gone
  FAIL_LOCAL  no STOP_ALL reason, but a job result is missing, unreadable, of another schema, or not "ok"
            (cgroup-local OOMs of the run's own scopes, refusals, timeouts and step failures stay on this node)
  PASS      every job result "ok", every OOM kill (if any) inside this run's scopes and counted, journals read,
            cleanup verified
The operator adds NODE_UNRESPONSIVE (no after.json from a node by its deadline, or the node unreachable) itself.
Exit codes: 0 PASS, 1 FAIL_LOCAL, 4 STOP_ALL, 2 usage.
"""
import argparse, hashlib, json, os, re, sys

SCHEMA = "k6-dit-job-classification-v1"
RESULT_SCHEMA = "k6-dit-job-result-v1"
RULES_SOURCE_SHA256 = "db3608630f1214378f30692cf4a46f88787c1b55cb3a7a9882a5fdf023261cf1"
STOP_CONTRACT = {
    "PASS": "node done; release its production marker only after DEV verifies the saved full results",
    "FAIL_LOCAL": "node done (no global effect shown); release its production marker only after DEV verifies the "
                  "saved full results",
    "STOP_ALL": "sticky: stop all further work on BOTH nodes now (run the after-script on the other node), "
                "start nothing else, release no production marker, report to the owner for release + restore",
}
# HEADER, KILLED, USERSPACE, XID, field, parse_kernel and read_counter: verbatim from the rules source.
HEADER = re.compile(r"oom-kill:constraint=(?P<constraint>[A-Z_]+),(?P<rest>.*)")
KILLED = re.compile(r"(?P<kind>Memory cgroup out of memory|Out of memory)[^:]*: Killed process (?P<pid>\d+) "
                    r"\((?P<comm>[^)]*)\)")
USERSPACE = re.compile(r"(earlyoom|systemd-oomd)(\[\d+\])?: .*(sending SIG|[Kk]illed|[Kk]illing)")
XID = re.compile(r"NVRM: Xid")


def field(rest, name):
    m = re.search(rf"(?:^|,){name}=([^,]*)", rest)
    return m.group(1) if m else None


# The only change from the source: the local class PROBE_SCOPE_OOM is named JOB_SCOPE_OOM.
def parse_kernel(text, units):
    """Every Killed-process line, attributed to its preceding oom-kill header and classified."""
    kills, header = [], None
    for line in text.splitlines():
        h = HEADER.search(line)
        if h:
            header = {"constraint": h.group("constraint"), "oom_memcg": field(h.group("rest"), "oom_memcg"),
                      "task_memcg": field(h.group("rest"), "task_memcg"), "line": line.strip()[:300]}
            continue
        k = KILLED.search(line)
        if not k:
            continue
        kill = {"pid": int(k.group("pid")), "comm": k.group("comm"), "kind": k.group("kind"),
                "line": line.strip()[:300], "header": header}
        if header is None:
            kill["class"] = "UNATTRIBUTED_OOM_KILL"
        elif header["constraint"] != "CONSTRAINT_MEMCG" or k.group("kind") != "Memory cgroup out of memory":
            kill["class"] = "GLOBAL_OOM"
        elif (header["oom_memcg"] or "").rsplit("/", 1)[-1] not in units:
            kill["class"] = "FOREIGN_MEMCG_OOM"
        else:
            kill["class"] = "JOB_SCOPE_OOM"
        kills.append(kill)
    return kills


def read_counter(path):
    try:
        epoch, count = open(path).read().split()
        return float(epoch), int(count)
    except (OSError, ValueError):
        return None


def read_result(path):
    """One job result file: its sha256, job and result, or the reason it cannot be used."""
    res = {"path": path, "present": os.path.exists(path)}
    if not res["present"]:
        return res
    try:
        with open(path, "rb") as f:
            raw = f.read()
        res["sha256"] = hashlib.sha256(raw).hexdigest()
        r = json.loads(raw)
        if r["schema"] != RESULT_SCHEMA:
            raise ValueError(f"schema {r['schema']!r} is not {RESULT_SCHEMA!r}")
        if not isinstance(r["job"], str) or not isinstance(r["result"], str):
            raise TypeError("job and result must be strings")
        res.update(job=r["job"], result=r["result"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        res["error"] = repr(e)
    return res


def classify(a):
    units = set(a.unit)
    stop, local = [], []
    results = [read_result(p) for p in a.job_result]
    if a.kernel_rc != 0 or a.system_rc != 0:
        stop.append(f"JOURNAL_UNREADABLE: kernel rc {a.kernel_rc}, system rc {a.system_rc}")
    try:
        with open(a.kernel_log, errors="replace") as f:
            ktext = f.read()
        with open(a.system_log, errors="replace") as f:
            stext = f.read()
    except OSError as e:
        ktext = stext = ""
        stop.append(f"JOURNAL_UNREADABLE: {e!r}")
    kills = parse_kernel(ktext, units)
    for k in kills:
        if k["class"] != "JOB_SCOPE_OOM":
            stop.append(f"{k['class']}: {k['line']}")
    before, after = read_counter(a.vmstat_before), read_counter(a.vmstat_after)
    oom = {"vmstat_before": before, "vmstat_after": after, "journal_killed_lines": len(kills),
           "job_scope_kills": [k["line"] for k in kills if k["class"] == "JOB_SCOPE_OOM"]}
    if before is None or after is None:
        stop.append("OOM_COUNTER_UNREADABLE: vmstat capture missing or malformed")
    else:
        oom["delta"] = after[1] - before[1]
        if oom["delta"] != len(kills):
            stop.append(f"OOM_EVIDENCE_INCOMPLETE: vmstat oom_kill +{oom['delta']} but {len(kills)} kernel "
                        "'Killed process' line(s) since T0")
    users = [l.strip()[:300] for l in stext.splitlines() if USERSPACE.search(l)]
    stop += [f"USERSPACE_OOM_KILL: {l}" for l in users]
    xids = [l.strip()[:300] for l in ktext.splitlines() if XID.search(l)]
    stop += [f"NVRM_XID: {l}" for l in xids]
    if not a.cleanup_verified:
        stop.append("CLEANUP_UNVERIFIED: job units or processes not shown gone")
    for r in results:
        if not r["present"]:
            local.append(f"NO_RESULT: {r['path']}")
        elif r.get("error"):
            local.append(f"RESULT_UNREADABLE: {r['path']}: {r['error']}")
        elif r["result"] != "ok":
            local.append(f"RESULT_{r['job']}: {r['result']}")
    cls = "STOP_ALL" if stop else ("FAIL_LOCAL" if local else "PASS")
    return {"schema": SCHEMA, "node": a.node, "t0": a.t0, "class": cls, "sticky": cls == "STOP_ALL",
            "stop_reasons": stop, "local_reasons": local, "contract": STOP_CONTRACT[cls], "results": results,
            "oom": oom, "userspace_oom_lines": users, "nvrm_xid_lines": xids,
            "journal_rc": {"kernel": a.kernel_rc, "system": a.system_rc}, "cleanup_verified": bool(a.cleanup_verified),
            "units": sorted(units), "rules_source_sha256": RULES_SOURCE_SHA256}


def main(argv=None):
    ap = argparse.ArgumentParser()
    for n in ("--node", "--vmstat-before", "--vmstat-after", "--kernel-log", "--system-log", "--out"):
        ap.add_argument(n, required=True)
    ap.add_argument("--t0", required=True, type=int)
    ap.add_argument("--kernel-rc", required=True, type=int)
    ap.add_argument("--system-rc", required=True, type=int)
    ap.add_argument("--cleanup-verified", required=True, type=int, choices=[0, 1])
    ap.add_argument("--unit", required=True, action="append")
    ap.add_argument("--job-result", required=True, action="append")
    a = ap.parse_args(argv)
    res = classify(a)
    tmp = f"{a.out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, a.out)
    print(json.dumps({"class": res["class"], "stop_reasons": res["stop_reasons"][:5],
                      "local_reasons": res["local_reasons"]}))
    return {"PASS": 0, "FAIL_LOCAL": 1, "STOP_ALL": 4}[res["class"]]


if __name__ == "__main__":
    sys.exit(main())
