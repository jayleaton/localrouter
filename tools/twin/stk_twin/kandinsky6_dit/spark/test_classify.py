#!/usr/bin/env python3
"""CPU-only, offline checks of classify.py and after.sh: `python3 -I test_classify.py -v` from this directory.

Parity with the D1 rules source (skipped only when that file is absent on this host), every STOP_ALL reason, the
local classes, exit codes and the atomic output; after.sh against PATH stubs of systemctl, journalctl and pgrep.
"""
import contextlib, hashlib, importlib.util, inspect, io, json, os, shutil, subprocess, sys, tempfile, unittest, warnings
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # nothing written next to this file or the rules source
sys.path.insert(0, HERE)
import classify as CL  # noqa: E402

# the D1 rules source, read from its sealed v4 kit (K6_D1_RULES_KIT overrides the path); parity is skipped without it
RULES_KIT = os.environ.get("K6_D1_RULES_KIT", "/home/sireggy/Documents/Projects/spark-work/k6-d1-uma-probe-kit/v4/"
                           "k6-d1-uma-probe-39dfcb126221.tar.gz")
RULES_MEMBER = "k6-d1-uma-probe/uma-probe-classify.py"
_TMP = tempfile.TemporaryDirectory(prefix="k6-rules-")
SOURCE = os.path.join(_TMP.name, "uma-probe-classify.py")
if os.path.exists(RULES_KIT):
    import tarfile
    with tarfile.open(RULES_KIT) as t, open(SOURCE, "wb") as f:
        f.write(t.extractfile(RULES_MEMBER).read())
AFTER = os.path.join(HERE, "after.sh")
HOST_PY = "/usr/bin/python3"
T0 = 1900000000
APP = "/user.slice/user-1000.slice/user@1000.service/app.slice"
UNITS = [f"k6-dit-smoke-{T0}.scope", f"k6-dit-twin-{T0}.scope"]


def read(path, mode="r"):
    with open(path, mode) as f:
        return f.read()


def hdr(memcg, constraint="CONSTRAINT_MEMCG"):
    if constraint == "CONSTRAINT_MEMCG":
        return (f"{T0 + 5}.000001 dgx kernel: oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=user.slice,"
                f"mems_allowed=0,oom_memcg={memcg},task_memcg={memcg},task=python3,pid=4242,uid=1000")
    return (f"{T0 + 5}.000001 dgx kernel: oom-kill:constraint={constraint},nodemask=(null),cpuset=/,mems_allowed=0,"
            f"global_oom,task_memcg={memcg},task=python3,pid=4242,uid=1000")


def killed(pid=4242, memcg=True):
    kind = "Memory cgroup out of memory" if memcg else "Out of memory"
    return (f"{T0 + 5}.000002 dgx kernel: {kind}: Killed process {pid} (python3) total-vm:297176kB, "
            "anon-rss:261480kB, file-rss:11132kB, shmem-rss:0kB, UID:1000 pgtables:604kB oom_score_adj:800")


def scope(u):
    return f"{APP}/{u}"


KERNEL_FIXTURES = [
    "",
    "\n".join([hdr(scope(UNITS[0])), killed()]),
    "\n".join([hdr(scope(UNITS[1]), "CONSTRAINT_NONE"), killed(memcg=False)]),
    "\n".join([hdr("/system.slice/docker-1fc1.scope"), killed()]),
    killed(),
    "\n".join([hdr(scope(UNITS[0])), killed(1), f"{T0 + 5}.1 dgx kernel: Tasks in {scope(UNITS[0])} are going to be "
               "killed due to memory.oom.group set", killed(2), killed(3)]),
    "\n".join([hdr(scope(UNITS[0])), killed(1), hdr(scope(UNITS[0]), "CONSTRAINT_NONE"), killed(2, memcg=False)]),
    "\n".join([hdr(scope(f"k6-dit-smoke-{T0 - 1}.scope")), killed()]),
    "\n".join([hdr(scope(UNITS[1])), "unrelated line", killed(7)]),
    f"{T0 + 9}.0 dgx kernel: NVRM: Xid (PCI:000f:01:00): 31, pid=4242, name=python3, MMU Fault",
    "oom-kill:constraint=CONSTRAINT_MEMCG,task_memcg=/x\nMemory cgroup out of memory: Killed process 9 (a)",
]


def load_source():
    spec = importlib.util.spec_from_file_location("uma_probe_classify_source", SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(os.path.exists(SOURCE), f"rules kit {RULES_KIT} absent on this host: parity not checked")
class Parity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = load_source()

    def test_source_sha256(self):
        self.assertEqual(hashlib.sha256(read(SOURCE, "rb")).hexdigest(), CL.RULES_SOURCE_SHA256)

    def test_patterns(self):
        for n in ("HEADER", "KILLED", "USERSPACE", "XID"):
            self.assertEqual(getattr(CL, n).pattern, getattr(self.src, n).pattern, n)
            self.assertEqual(getattr(CL, n).flags, getattr(self.src, n).flags, n)

    def test_verbatim_functions(self):
        for n in ("field", "read_counter"):
            self.assertEqual(inspect.getsource(getattr(CL, n)), inspect.getsource(getattr(self.src, n)), n)
        self.assertEqual(inspect.getsource(CL.parse_kernel),
                         inspect.getsource(self.src.parse_kernel).replace("PROBE_SCOPE_OOM", "JOB_SCOPE_OOM"))

    def test_parse_kernel_same_results(self):
        units = set(UNITS)
        for i, text in enumerate(KERNEL_FIXTURES):
            mine = CL.parse_kernel(text, units)
            theirs = self.src.parse_kernel(text, units)
            for k in theirs:
                if k["class"] == "PROBE_SCOPE_OOM":
                    k["class"] = "JOB_SCOPE_OOM"
            self.assertEqual(mine, theirs, f"fixture {i}")
        self.assertIn("JOB_SCOPE_OOM", [k["class"] for t in KERNEL_FIXTURES for k in CL.parse_kernel(t, units)])


class Classify(unittest.TestCase):
    def setUp(self):
        # read_counter is the source's verbatim `open(path).read()`; its unclosed-file warning is not a finding here
        warnings.filterwarnings("ignore", category=ResourceWarning, module="classify")
        self.d = tempfile.mkdtemp(prefix="k6-classify-")
        self.addCleanup(shutil.rmtree, self.d)

    def write(self, name, text):
        p = os.path.join(self.d, name)
        with open(p, "w") as f:
            f.write(text)
        return p

    def result(self, job="smoke", result="ok", schema="k6-dit-job-result-v1", raw=None):
        p = os.path.join(self.d, f"result-{job}.json")
        with open(p, "w") as f:
            f.write(raw if raw is not None else json.dumps({"schema": schema, "job": job, "result": result}))
        return p

    def run_cl(self, kernel="", system="", before=(T0 + 1, 30), after=(T0 + 60, 30), krc=0, src=0, clean=1,
               results=None, units=UNITS):
        args = ["--node", "spark-1", "--t0", str(T0), "--kernel-rc", str(krc), "--system-rc", str(src),
                "--cleanup-verified", str(clean), "--kernel-log", self.write("k.log", kernel),
                "--system-log", self.write("s.log", system), "--out", os.path.join(self.d, "after.json")]
        for flag, v, n in (("--vmstat-before", before, "vb"), ("--vmstat-after", after, "va")):
            args += [flag, self.write(n, f"{v[0]} {v[1]}\n") if v is not None else os.path.join(self.d, n + ".none")]
        for u in units:
            args += ["--unit", u]
        for r in (results if results is not None else [self.result()]):
            args += ["--job-result", r]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = CL.main(args)
        r = json.loads(read(os.path.join(self.d, "after.json")))
        self.assertEqual(json.loads(out.getvalue())["class"], r["class"])
        self.assertEqual(rc, {"PASS": 0, "FAIL_LOCAL": 1, "STOP_ALL": 4}[r["class"]])
        self.assertEqual(r["sticky"], r["class"] == "STOP_ALL")
        self.assertEqual(r["contract"], CL.STOP_CONTRACT[r["class"]])
        return rc, r

    def expect(self, cls, needle="", **kw):
        rc, r = self.run_cl(**kw)
        self.assertEqual(r["class"], cls, r["stop_reasons"] + r["local_reasons"])
        if needle:
            self.assertTrue(any(x.startswith(needle) for x in r["stop_reasons"] + r["local_reasons"]),
                            (needle, r["stop_reasons"], r["local_reasons"]))
        return r

    def test_pass_clean(self):
        r = self.expect("PASS")
        self.assertEqual((r["schema"], r["node"], r["t0"]), ("k6-dit-job-classification-v1", "spark-1", T0))
        self.assertEqual(r["stop_reasons"] + r["local_reasons"], [])
        self.assertEqual(r["units"], sorted(UNITS))
        self.assertEqual(r["rules_source_sha256"], CL.RULES_SOURCE_SHA256)
        self.assertEqual(r["oom"]["delta"], 0)
        self.assertEqual(r["journal_rc"], {"kernel": 0, "system": 0})
        self.assertTrue(r["cleanup_verified"])
        res = r["results"][0]
        raw = read(res["path"], "rb")
        self.assertEqual((res["sha256"], res["job"], res["result"]), (hashlib.sha256(raw).hexdigest(), "smoke", "ok"))
        self.assertEqual(set(r), {"schema", "node", "t0", "class", "sticky", "stop_reasons", "local_reasons",
                                  "contract", "results", "oom", "userspace_oom_lines", "nvrm_xid_lines",
                                  "journal_rc", "cleanup_verified", "units", "rules_source_sha256"})

    def test_job_scope_oom_stays_local(self):
        r = self.expect("PASS", kernel="\n".join([hdr(scope(UNITS[1])), killed()]), after=(T0 + 60, 31))
        self.assertEqual(len(r["oom"]["job_scope_kills"]), 1)
        r = self.expect("FAIL_LOCAL", "RESULT_twin: stopped at gate", kernel="\n".join([hdr(scope(UNITS[1])), killed()]),
                        after=(T0 + 60, 31), results=[self.result("twin", "stopped at gate")])
        self.assertEqual(r["stop_reasons"], [])

    def test_journal_unreadable(self):
        self.expect("STOP_ALL", "JOURNAL_UNREADABLE: kernel rc 98, system rc 0", krc=98)
        self.expect("STOP_ALL", "JOURNAL_UNREADABLE: kernel rc 0, system rc 1", src=1)

    def test_journal_file_missing(self):
        args = ["--node", "n", "--t0", str(T0), "--kernel-rc", "0", "--system-rc", "0", "--cleanup-verified", "1",
                "--kernel-log", os.path.join(self.d, "missing"), "--system-log", os.path.join(self.d, "missing"),
                "--vmstat-before", self.write("vb", "1 0"), "--vmstat-after", self.write("va", "2 0"),
                "--unit", UNITS[0], "--job-result", self.result(), "--out", os.path.join(self.d, "o.json")]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(CL.main(args), 4)
        r = json.loads(read(os.path.join(self.d, "o.json")))
        self.assertTrue(r["stop_reasons"][0].startswith("JOURNAL_UNREADABLE: FileNotFoundError"))

    def test_oom_counter_unreadable(self):
        self.expect("STOP_ALL", "OOM_COUNTER_UNREADABLE", before=None)
        self.expect("STOP_ALL", "OOM_COUNTER_UNREADABLE", after=("x", "y"))

    def test_global_oom(self):
        self.expect("STOP_ALL", "GLOBAL_OOM", kernel="\n".join([hdr(scope(UNITS[0]), "CONSTRAINT_NONE"),
                                                                 killed(memcg=False)]), after=(T0 + 60, 31))
        self.expect("STOP_ALL", "GLOBAL_OOM", kernel="\n".join([hdr(scope(UNITS[0])), killed(1),
                                                                 hdr(scope(UNITS[0]), "CONSTRAINT_NONE"),
                                                                 killed(2, memcg=False)]), after=(T0 + 60, 32))

    def test_foreign_memcg_oom(self):
        self.expect("STOP_ALL", "FOREIGN_MEMCG_OOM", kernel="\n".join([hdr("/system.slice/docker-1fc1.scope"),
                                                                        killed()]), after=(T0 + 60, 31))
        self.expect("STOP_ALL", "FOREIGN_MEMCG_OOM", kernel="\n".join([hdr(scope(f"k6-dit-smoke-{T0 - 1}.scope")),
                                                                        killed()]), after=(T0 + 60, 31))

    def test_unattributed(self):
        self.expect("STOP_ALL", "UNATTRIBUTED_OOM_KILL", kernel=killed(), after=(T0 + 60, 31))

    def test_evidence_incomplete(self):
        k = "\n".join([hdr(scope(UNITS[0])), killed()])
        self.expect("STOP_ALL", "OOM_EVIDENCE_INCOMPLETE", kernel=k, after=(T0 + 60, 32))
        self.expect("STOP_ALL", "OOM_EVIDENCE_INCOMPLETE", kernel=k)

    def test_userspace_oom_kill(self):
        self.expect("STOP_ALL", "USERSPACE_OOM_KILL",
                    system=f"{T0 + 9}.0 dgx earlyoom[812]: sending SIGTERM to process 4242 uid 1000 \"python3\"")
        self.expect("STOP_ALL", "USERSPACE_OOM_KILL",
                    system=f"{T0 + 9}.0 dgx systemd-oomd[700]: Killed {APP}/x.scope due to memory pressure")

    def test_nvrm_xid(self):
        self.expect("STOP_ALL", "NVRM_XID",
                    kernel=f"{T0 + 9}.0 dgx kernel: NVRM: Xid (PCI:000f:01:00): 31, pid=4242, name=python3, MMU Fault")

    def test_cleanup_unverified(self):
        self.expect("STOP_ALL", "CLEANUP_UNVERIFIED", clean=0)

    def test_stop_order(self):
        rc, r = self.run_cl(krc=1, before=None, clean=0, kernel="\n".join([killed(), "NVRM: Xid 79"]),
                            system="earlyoom: killing process 1", results=[os.path.join(self.d, "none.json")])
        self.assertEqual([x.split(":")[0] for x in r["stop_reasons"]],
                         ["JOURNAL_UNREADABLE", "UNATTRIBUTED_OOM_KILL", "OOM_COUNTER_UNREADABLE",
                          "USERSPACE_OOM_KILL", "NVRM_XID", "CLEANUP_UNVERIFIED"])
        self.assertEqual(r["local_reasons"], [f"NO_RESULT: {os.path.join(self.d, 'none.json')}"])

    def test_fail_local_results(self):
        missing = os.path.join(self.d, "missing.json")
        self.expect("FAIL_LOCAL", f"NO_RESULT: {missing}", results=[self.result(), missing])
        bad = self.result("bad", raw='{"schema": "k6-dit-job-result-v1", "jo')
        r = self.expect("FAIL_LOCAL", f"RESULT_UNREADABLE: {bad}: JSONDecodeError", results=[bad])
        self.assertIn("sha256", r["results"][0])
        other = self.result("other", schema="k6-dit-job-result-v0")
        self.expect("FAIL_LOCAL", f"RESULT_UNREADABLE: {other}: ValueError", results=[other])
        lst = self.result("lst", raw="[1]")
        self.expect("FAIL_LOCAL", f"RESULT_UNREADABLE: {lst}: TypeError", results=[lst])
        nojob = self.result("nojob", raw='{"schema": "k6-dit-job-result-v1", "result": "ok"}')
        self.expect("FAIL_LOCAL", f"RESULT_UNREADABLE: {nojob}: KeyError", results=[nojob])
        r = self.expect("FAIL_LOCAL", "RESULT_blocks: completed with failures",
                        results=[self.result(), self.result("blocks", "completed with failures")])
        self.assertEqual(len(r["local_reasons"]), 1)
        self.assertEqual([x["result"] for x in r["results"]], ["ok", "completed with failures"])

    def test_usage(self):
        base = ["--node", "n", "--t0", "1", "--vmstat-before", "a", "--vmstat-after", "b", "--kernel-log", "c",
                "--kernel-rc", "0", "--system-log", "d", "--system-rc", "0", "--cleanup-verified", "1",
                "--out", os.path.join(self.d, "o.json")]
        for argv in (base + ["--job-result", "r"], base + ["--unit", "u"],
                     [x if x != "1" else "2" for x in base] + ["--unit", "u", "--job-result", "r"]):
            with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()):
                CL.main(argv)
            self.assertEqual(e.exception.code, 2)
        self.assertFalse(os.path.exists(os.path.join(self.d, "o.json")))

    def test_atomic_output(self):
        out = os.path.join(self.d, "after.json")
        self.write("after.json", "stale")
        self.run_cl()
        self.assertEqual(json.loads(read(out))["class"], "PASS")
        self.assertEqual([f for f in os.listdir(self.d) if ".tmp." in f], [])
        replaced = []
        real = os.replace
        with unittest.mock.patch.object(CL.os, "replace", lambda a, b: (replaced.append((a, b)), real(a, b))):
            self.run_cl()
        self.assertEqual(replaced, [(f"{out}.tmp.{os.getpid()}", out)])


STUBS = {
    "systemctl": r"""#!/bin/bash
echo "systemctl $*" >> "$STUB_LOG"
u=${!#}
case " $* " in
  *" show "*) echo "LoadState=loaded"; echo "ActiveState=inactive"; echo "Result=success" ;;
  *" is-active "*) for a in $STUB_ACTIVE_UNITS; do [ "$a" = "$u" ] && { echo active; exit 0; }; done
                   echo inactive; exit 3 ;;
esac
exit 0
""",
    "journalctl": r"""#!/bin/bash
echo "journalctl $*" >> "$STUB_LOG"
case " $* " in
  *" -n 1 "*) [ "${STUB_JOURNAL_EMPTY:-0}" = 1 ] || echo "1900000000.0 host kernel: boot"; exit 0 ;;
  *" -k "*) printf '%s' "${STUB_KLOG:-}"; exit "${STUB_KRC:-0}" ;;
  *) printf '%s' "${STUB_SLOG:-}"; exit 0 ;;
esac
""",
    "pgrep": r"""#!/bin/bash
echo "pgrep $*" >> "$STUB_LOG"
[ -n "${STUB_PGREP:-}" ] || exit 1
printf '%s\n' "$STUB_PGREP"
""",
}


def oom_kill_now():
    for line in read("/proc/vmstat").splitlines():
        k, v = line.split()
        if k == "oom_kill":
            return int(v)
    raise RuntimeError("no oom_kill in /proc/vmstat")


@unittest.skipUnless(os.path.exists("/proc/vmstat") and os.access(HOST_PY, os.X_OK), "needs Linux and /usr/bin/python3")
class After(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="k6-after-")
        self.addCleanup(shutil.rmtree, self.d)
        self.bin = os.path.join(self.d, "bin")
        os.mkdir(self.bin)
        for n, body in STUBS.items():
            p = os.path.join(self.bin, n)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, 0o755)
        self.out = os.path.join(self.d, "out")
        self.work = os.path.join(self.d, "work (x)")
        self.log = os.path.join(self.d, "stub.log")
        self.vb = os.path.join(self.d, "vmstat.before")
        self.res = os.path.join(self.d, "result-smoke.json")
        with open(self.res, "w") as f:
            json.dump({"schema": "k6-dit-job-result-v1", "job": "smoke", "result": "ok"}, f)

    def run_after(self, args=None, **env):
        with open(self.vb, "w") as f:
            f.write(f"{T0} {oom_kill_now()}\n")
        e = {k: v for k, v in os.environ.items() if not k.startswith("STUB_") and k != "K6_FALLBACK_PY"}
        e.update(PATH=self.bin + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"), STUB_LOG=self.log, **env)
        if args is None:
            args = [self.out, str(T0), HOST_PY, "spark-1", self.vb, self.work, *UNITS, "--", self.res]
        p = subprocess.run(["bash", AFTER, *args], capture_output=True, text=True, timeout=120, env=e)
        r = json.loads(read(os.path.join(self.out, "after.json"))) if os.path.exists(self.out) else None
        return p, r

    def test_clean_pass(self):
        p, r = self.run_after()
        self.assertEqual(p.returncode, 0, (p.stderr, r))
        self.assertEqual(r["class"], "PASS")
        self.assertEqual(r["units"], sorted(UNITS))
        self.assertEqual(r["results"][0]["job"], "smoke")
        txt = read(os.path.join(self.out, "after.txt"))
        self.assertIn("cleanup_verified=1", txt)
        self.assertIn(f"classifier_python={HOST_PY} exit=0", txt)
        log = read(self.log)
        user = [] if os.getuid() == 0 else ["--user"]
        for u in UNITS:
            for verb in (["kill", "-s", "SIGKILL"], ["stop"], ["reset-failed"], ["is-active"]):
                self.assertIn(" ".join(["systemctl", *user, *verb, u]), log)
        wre = "".join("\\" + c if c in "[]\\.*^$+?(){}|" else c for c in self.work)
        self.assertIn(f"pgrep -af -- {wre}/kit-|node-job\\.sh|spark/supervise\\.py", log)
        self.assertEqual(sorted(os.listdir(self.out)), ["after.json", "after.txt", "kernel.err", "kernel.log",
                                                        "system.err", "system.log", "vmstat.after"])

    def test_active_unit_stops_all(self):
        p, r = self.run_after(STUB_ACTIVE_UNITS=UNITS[1])
        self.assertEqual(p.returncode, 4)
        self.assertEqual(r["class"], "STOP_ALL")
        self.assertTrue(r["sticky"])
        self.assertIn("CLEANUP_UNVERIFIED: job units or processes not shown gone", r["stop_reasons"])

    def test_remaining_process_stops_all(self):
        p, r = self.run_after(STUB_PGREP=f"4242 bash {self.work}/kit-abc/node-job.sh smoke")
        self.assertEqual(p.returncode, 4)
        self.assertTrue(any(x.startswith("CLEANUP_UNVERIFIED") for x in r["stop_reasons"]))
        p, r = self.run_after(STUB_PGREP=f"4243 bash {self.work}/kit-abc/tools/after.sh {self.out}")
        self.assertEqual(p.returncode, 0, r)

    def test_empty_journal_is_unreadable(self):
        p, r = self.run_after(STUB_JOURNAL_EMPTY="1")
        self.assertEqual(p.returncode, 4)
        self.assertIn("JOURNAL_UNREADABLE: kernel rc 98, system rc 98", r["stop_reasons"])
        self.assertIn("journal_rc kernel=98 system=98", read(os.path.join(self.out, "after.txt")))

    def test_failed_result_is_local(self):
        with open(self.res, "w") as f:
            json.dump({"schema": "k6-dit-job-result-v1", "job": "smoke", "result": "stopped at kernels"}, f)
        p, r = self.run_after()
        self.assertEqual(p.returncode, 1)
        self.assertEqual(r["local_reasons"], ["RESULT_smoke: stopped at kernels"])

    def test_fallback_interpreter(self):
        p, r = self.run_after(args=[self.out, str(T0), "/bin/false", "spark-1", self.vb, self.work, UNITS[0], "--",
                                    self.res], K6_FALLBACK_PY=HOST_PY)
        self.assertEqual((p.returncode, r["class"]), (0, "PASS"))
        p, r = self.run_after(args=[self.out, str(T0), "/bin/false", "spark-1", self.vb, self.work, UNITS[0], "--",
                                    self.res], K6_FALLBACK_PY="/bin/false")
        self.assertEqual(p.returncode, 4)
        self.assertEqual((r["class"], r["stop_reasons"]), ("STOP_ALL", ["CLASSIFIER_ERROR: exit 1"]))

    def test_usage(self):
        ok = [self.out, str(T0), HOST_PY, "spark-1", self.vb, self.work]
        for args in ([], ok + [UNITS[0], self.res], ok + ["--", self.res], ok + [UNITS[0], "--"],
                     ok + [UNITS[0], UNITS[1], "--"],
                     [self.out, "12x", *ok[2:], UNITS[0], "--", self.res],
                     [*ok[:3], "Spark_1", *ok[4:], UNITS[0], "--", self.res],
                     [*ok[:5], "work", UNITS[0], "--", self.res],
                     ok + ["k6-uma-probe-s-1.scope", "--", self.res],
                     ok + ["k6-dit-smoke.service", "--", self.res]):
            p, r = self.run_after(args=args)
            self.assertEqual(p.returncode, 2, args)
            self.assertIsNone(r, args)


if __name__ == "__main__":
    unittest.main()
