#!/usr/bin/env python3
"""CPU-only, offline checks of the GB10 uma-node policy in the DiT node job: `python3 -I test_policy.py -v` from here.

Owner policy 2026-10-12, all of (a)-(d): (a) budget.py's Torch cap, set and read back; (b) the declared overhead and
the empty raw-CUDA-allocation ledger; (c) the 12 GiB floor, start admission and supervise.py's 0.1 s sampling with an
immediate whole-tree SIGKILL; (d) is classify.py / after.sh (test_classify.py). K6_NODE_JOB points the NodeJob checks
at another node-job.sh (the pre-policy script fails them: they are the before/after control).
"""
import json, os, re, signal, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG.parent.parent))
from stk_twin.kandinsky6_dit import budget  # noqa: E402

NODE_JOB = Path(os.environ.get("K6_NODE_JOB", HERE / "node-job.sh"))
SUP = HERE / "supervise.py"
GIB = 1 << 30
TOTAL = 130659676160  # GB10 dgx01 cudaMemGetInfo total (D1 run 1791795729)


class FakeCuda:
    def __init__(self, total=TOTAL, reserved=0, skew=False, initialized=False):
        self.total, self.reserved, self.skew, self.initialized, self.fraction = total, reserved, skew, initialized, None

    def init(self):
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def current_device(self):
        return 0

    def memory_reserved(self, dev):
        return self.reserved

    def mem_get_info(self, dev):
        return self.total // 2, self.total

    def set_per_process_memory_fraction(self, f, dev):
        self.fraction = f

    def get_per_process_memory_fraction(self, dev):
        return self.fraction * (1.0000001 if self.skew else 1.0)


def fake_torch(**kw):
    return SimpleNamespace(cuda=FakeCuda(**kw))


class Env:
    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.values}
        for k, v in self.values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TorchCap(unittest.TestCase):
    """(a): a per-process caching-allocator cap at the declared budget, readback == set, cap <= budget."""

    def test_fraction_is_largest_within_budget(self):
        for total in (TOTAL, 128502063104, 3 * GIB + 1):
            for b in (1 * GIB, 16 * GIB, 28 * GIB, 3 * GIB):
                if b > total:
                    continue
                f = budget.fraction_for(b, total)
                self.assertLessEqual(int(f * total), b)
                self.assertGreater(int(__import__("math").nextafter(f, 1.0) * total) + 2, b - 2)
        with self.assertRaises(budget.CapError):
            budget.fraction_for(0, TOTAL)
        with self.assertRaises(budget.CapError):
            budget.fraction_for(TOTAL + 1, TOTAL)

    def test_off_node_job_no_cap(self):
        with Env(STK_K6_TORCH_BUDGET_BYTES=None, STK_K6_CAP_DIR=None):
            self.assertIsNone(budget.apply("x", fake_torch()))

    def test_half_configured_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for env in ({"STK_K6_TORCH_BUDGET_BYTES": str(GIB), "STK_K6_CAP_DIR": None},
                        {"STK_K6_TORCH_BUDGET_BYTES": None, "STK_K6_CAP_DIR": d},
                        {"STK_K6_TORCH_BUDGET_BYTES": "16G", "STK_K6_CAP_DIR": d}):
                with Env(**env), self.assertRaises(budget.CapError):
                    budget.apply("x", fake_torch())

    def test_cap_written_and_bounded(self):
        with tempfile.TemporaryDirectory() as d, Env(STK_K6_TORCH_BUDGET_BYTES=str(16 * GIB), STK_K6_CAP_DIR=d):
            t = fake_torch()
            rec = budget.apply("gate x", t)
            self.assertEqual(rec["fraction_readback"], rec["fraction_set"])
            self.assertLessEqual(rec["cap_bytes"], 16 * GIB)
            self.assertEqual(t.cuda.fraction, rec["fraction_set"])
            on_disk = json.loads(Path(d, f"cap-{os.getpid()}.json").read_text())
            self.assertEqual(on_disk, rec)
            with self.assertRaises(FileExistsError):  # one cap record per process
                budget.apply("again", fake_torch())

    def test_refused_after_allocation_or_bad_readback(self):
        with tempfile.TemporaryDirectory() as d, Env(STK_K6_TORCH_BUDGET_BYTES=str(GIB), STK_K6_CAP_DIR=d):
            with self.assertRaises(budget.CapError):
                budget.apply("x", fake_torch(reserved=2 << 20))
            with self.assertRaises(budget.CapError):
                budget.apply("x", fake_torch(skew=True))
            self.assertEqual(os.listdir(d), [])

    def test_host_only(self):
        budget.host_only("parent", fake_torch())
        with self.assertRaises(budget.CapError):
            budget.host_only("parent", fake_torch(initialized=True))


def descendants_alive(marker):
    out = subprocess.run(["pgrep", "-f", f"^{marker}-"], capture_output=True, text=True).stdout.split()
    return [p for p in out if int(p) != os.getpid()]


class Supervisor(unittest.TestCase):
    """(c): MemAvailable sampled every 0.1 s, floor 12 GiB, immediate SIGKILL of the whole tree."""

    def setUp(self):
        self.d = tempfile.TemporaryDirectory(prefix="k6-sup-")
        self.dir = Path(self.d.name)
        self.mi, self.vm = self.dir / "meminfo", self.dir / "vmstat"
        self.set_avail(64 * GIB)
        self.vm.write_text("oom_kill 3\n")
        self.marker = f"k6suptest{os.getpid()}x{id(self)}"

    def tearDown(self):
        subprocess.run(["pkill", "-KILL", "-f", f"^{self.marker}-"])
        self.d.cleanup()

    def set_avail(self, b):
        tmp = self.dir / "mi.tmp"
        tmp.write_text(f"MemTotal: 130000000 kB\nMemAvailable: {b // 1024} kB\n")
        os.replace(tmp, self.mi)

    def start(self, script, *extra):
        argv = [sys.executable, "-I", str(SUP), "--floor-bytes", str(12 * GIB), "--record", str(self.dir / "rec.json"),
                "--session-file", str(self.dir / "sess"), "--meminfo", str(self.mi), "--vmstat", str(self.vm), *extra,
                "--", "bash", "-c", script.replace("MARK", self.marker)]
        return subprocess.Popen(argv)

    def record(self):
        return json.loads((self.dir / "rec.json").read_text())

    # a step whose tree escapes its session: a setsid'd, double-forked sleeper and a plain background one
    ESCAPING = 'setsid bash -c "exec -a MARK-a sleep 300 & disown" & (exec -a MARK-b sleep 300) & exec -a MARK-c sleep 300'

    def wait_started(self, n=3):
        end = time.monotonic() + 5
        while len(descendants_alive(self.marker)) < n and time.monotonic() < end:
            time.sleep(0.05)
        self.assertEqual(len(descendants_alive(self.marker)), n)

    def test_floor_kills_whole_tree_at_once(self):
        p = self.start(self.ESCAPING)
        self.wait_started()
        t = time.monotonic()
        self.set_avail(12 * GIB - 4096)
        self.assertEqual(p.wait(10), 137)
        self.assertLess(time.monotonic() - t, 0.5)
        r = self.record()
        self.assertTrue(r["kill"]["reason"].startswith("floor"), r)
        self.assertTrue(r["tree_verified"])
        self.assertEqual(descendants_alive(self.marker), [])
        self.assertLessEqual(r["max_gap_s"], 0.5)

    def test_normal_exit_kills_leftovers_keeps_rc(self):
        p = self.start('(exec -a MARK-left sleep 300) & exit 5')
        self.assertEqual(p.wait(10), 5)
        r = self.record()
        self.assertIsNone(r["kill"])
        self.assertEqual(r["rc"], 5)
        self.assertEqual(len(r["leftovers_killed"]), 1)
        self.assertTrue(r["tree_verified"])
        self.assertEqual(descendants_alive(self.marker), [])

    def test_oom_kill_counter_kills(self):
        p = self.start(self.ESCAPING)
        self.wait_started()
        self.vm.write_text("oom_kill 4\n")
        self.assertEqual(p.wait(10), 137)
        self.assertIn("oom_kill +1", self.record()["kill"]["reason"])
        self.assertEqual(descendants_alive(self.marker), [])

    def test_unreadable_meminfo_kills(self):
        p = self.start(self.ESCAPING)
        self.wait_started()
        self.mi.unlink()
        self.assertEqual(p.wait(10), 137)
        self.assertIn("unreadable", self.record()["kill"]["reason"])
        self.assertEqual(descendants_alive(self.marker), [])

    def test_sampling_gap_kills(self):
        p = self.start(self.ESCAPING)
        self.wait_started()
        os.kill(p.pid, signal.SIGSTOP)
        time.sleep(0.8)
        os.kill(p.pid, signal.SIGCONT)
        self.assertEqual(p.wait(10), 137)
        self.assertIn("supervision gap", self.record()["kill"]["reason"])
        self.assertEqual(descendants_alive(self.marker), [])

    def test_timeout_and_signal_kill(self):
        p = self.start(self.ESCAPING, "--timeout", "0.5")
        self.assertEqual(p.wait(10), 137)
        self.assertIn("timeout", self.record()["kill"]["reason"])
        self.assertEqual(descendants_alive(self.marker), [])
        p = self.start(self.ESCAPING)
        self.wait_started()
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(10), 137)
        self.assertIn("SIGTERM", self.record()["kill"]["reason"])
        self.assertEqual(descendants_alive(self.marker), [])

    def test_not_started_below_floor(self):
        self.set_avail(8 * GIB)
        p = self.start("exec -a MARK-never sleep 300")
        self.assertEqual(p.wait(10), 137)
        self.assertIsNone(self.record()["pid"])
        self.assertEqual(descendants_alive(self.marker), [])


def node_job(*args, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("FLOOR_GIB", "MODES", "STEP_MAX_MIN"))}
    e.update(env)
    return subprocess.run(["bash", str(NODE_JOB), *args], capture_output=True, text=True, env=e, timeout=60)


def embedded(name):
    """A python program embedded in node-job.sh as NAME='...'."""
    m = re.search(rf"^{name}='\n(.*?)\n'$", NODE_JOB.read_text(), re.S | re.M)
    if not m:
        raise AssertionError(f"{name} not found in {NODE_JOB}")
    return m.group(1)


class NodeJob(unittest.TestCase):
    """The step table, admission and wiring of node-job.sh against (a)-(c); the pre-policy script fails these."""

    def budgets(self, job, **env):
        r = node_job(job, "--budgets", **env)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        consts = dict(zip(lines[-1].split()[0::2], lines[-1].split()[1::2]))
        rows = [dict(zip(lines[0].split("\t"), l.split("\t"))) for l in lines[1:-1]]
        return rows, consts

    def test_budgets_follow_policy(self):
        for job in ("smoke", "twin", "ops", "blocks"):
            rows, c = self.budgets(job)
            self.assertGreaterEqual(int(c["floor_gib"]), 12)
            # (b) unmeasured for this job: at least the 4 GiB minimum with 2x margin, for every process counted
            for k in ("cuda_overhead_gib", "host_proc_gib", "job_shell_gib"):
                self.assertGreaterEqual(int(c[k]), 8, k)
            self.assertLessEqual(float(c["poll_s"]), 0.5)
            self.assertLessEqual(float(c["max_gap_s"]), 0.5)
            self.assertEqual(rows[0]["step"], "env")
            for r in rows:
                t, cu, h = int(r["torch_gib"]), int(r["cuda_procs"]), int(r["host_procs"])
                self.assertGreaterEqual(t, 1)
                want = cu * (t + int(c["cuda_overhead_gib"])) + h * int(c["host_proc_gib"]) + int(c["job_shell_gib"])
                self.assertEqual(int(r["budget_gib"]), want, r)
                self.assertEqual(int(r["admission_gib"]), want + int(c["floor_gib"]), r)
                if r["step"].startswith("gate_"):
                    self.assertEqual((cu, h), (1, 1), r)  # a host-only parent and one CUDA child at a time

    def test_floor_never_below_12(self):
        self.assertNotEqual(node_job("twin", "--budgets", FLOOR_GIB="11").returncode, 0)
        rows, c = self.budgets("twin", FLOOR_GIB="16")
        self.assertEqual(c["floor_gib"], "16")
        self.assertTrue(all(int(r["admission_gib"]) == int(r["budget_gib"]) + 16 for r in rows))

    def test_no_gpu_bound_from_cgroup_or_nvidia_smi(self):
        src = NODE_JOB.read_text()
        self.assertNotIn("memory.used", src)
        self.assertNotIn("TOTAL_GIB", src)
        self.assertNotRegex(src, r"(?m)^gpu_kib\(\)")

    def test_steps_run_under_supervisor_with_cap(self):
        src = NODE_JOB.read_text()
        self.assertRegex(src, r'STK_K6_TORCH_BUDGET_BYTES=\$\(\( torch \* 1073741824 \)\) STK_K6_CAP_DIR=')
        self.assertRegex(src, r'"\$PY" -I -s "\$SUP" --floor-bytes \$\(\( FLOOR_GIB \* 1073741824 \)\)')
        self.assertRegex(src, r'avail < \(budget \+ FLOOR_GIB\) \* GIB_KIB')
        self.assertRegex(src, r'--max-gib "\$torch"')
        for name in ("IMPORT_PY", "KERNELS_PY", "ENV_PY"):
            self.assertIn("budget.apply(", embedded(name), name)
        self.assertIn("not torch.cuda.is_initialized()", embedded("VENV_PY"))

    def test_policy_record_check(self):
        prog = embedded("POLICY_PY")
        with tempfile.TemporaryDirectory() as d:
            S = Path(d)
            (S / "caps").mkdir()

            def run(torch=16, cuda=1, rc=0):
                out = subprocess.run([sys.executable, "-I", "-c", prog, str(S), str(torch), str(cuda), str(rc), "0.5"],
                                     capture_output=True, text=True, check=True).stdout
                return out.rstrip("\n").split("\t")

            sv = {"schema": "k6-dit-step-supervision-v1", "tree_verified": True, "max_gap_s": 0.11,
                  "min_avail": 100 * GIB, "kill": None}
            (S / "supervision.json").write_text(json.dumps(sv))
            self.assertEqual(run()[0], "BAD")  # no cap record for the CUDA process
            cap = {"schema": "k6-dit-torch-cap-v1", "budget_bytes": 16 * GIB, "fraction_set": 0.1,
                   "fraction_readback": 0.1, "cap_bytes": 16 * GIB}
            (S / "caps" / "cap-1.json").write_text(json.dumps(cap))
            self.assertEqual(run(), ["ok", "100.0", "0.11", "1", "-"])
            self.assertEqual(run(torch=28)[0], "BAD")  # budget other than declared
            (S / "caps" / "cap-2.json").write_text(json.dumps({**cap, "fraction_readback": 0.2}))
            self.assertEqual(run()[0], "BAD")
            (S / "caps" / "cap-2.json").unlink()
            (S / "supervision.json").write_text(json.dumps({**sv, "max_gap_s": 0.7}))
            self.assertEqual(run()[0], "BAD")
            (S / "supervision.json").write_text(json.dumps({**sv, "tree_verified": False}))
            self.assertEqual(run()[0], "BAD")
            (S / "supervision.json").write_text(json.dumps({**sv, "kill": {"reason": "floor: x"}}))
            self.assertEqual(run(rc=137), ["ok", "100.0", "0.11", "1", "floor: x"])


    def test_job_result_is_what_classify_reads(self):
        prog = embedded("JOBRES_PY")
        sys.path.insert(0, str(HERE))
        import classify
        with tempfile.TemporaryDirectory() as d:
            R = Path(d)
            (R / "steps.tsv").write_text("step\tstatus\trc\nenv\tok\t0\ngate_blocks_fp32\tkilled by supervisor: floor\t137\n")
            for result, local in (("ok", None), ("stopped at gate_blocks_fp32", "RESULT_blocks")):
                subprocess.run([sys.executable, "-I", "-c", prog, str(R), "blocks", "a" * 64, "blocks-1900000000",
                                result, "1900000000", "1900000600", "12"], check=True)
                res = json.loads((R / "job-result.json").read_text())
                self.assertEqual(res["schema"], "k6-dit-job-result-v1")
                self.assertEqual(res["steps"][1]["status"], "killed by supervisor: floor")
                reasons = classify.classify(SimpleNamespace(
                    node="dgx01", t0=1900000000, vmstat_before=str(R / "vb"), vmstat_after=str(R / "vb"),
                    kernel_log=str(R / "steps.tsv"), kernel_rc=0, system_log=str(R / "steps.tsv"), system_rc=0,
                    cleanup_verified=1, unit=["k6-dit-blocks-1900000000.scope"],
                    job_result=[str(R / "job-result.json")]))
                if local is None:
                    self.assertEqual(reasons["local_reasons"], [])
                else:
                    self.assertTrue(reasons["local_reasons"][0].startswith(local), reasons["local_reasons"])


# Every native allocation entry point the owner ruling names (cuMemAlloc*, cuMemHostAlloc*, cudaMalloc*, cuMemCreate),
# plus the other pinned-host / pool / registration entry points.
RAW_ALLOC = re.compile(r"\b(cuMemAlloc\w*|cuMemHostAlloc\w*|cudaMalloc\w*|cuMemCreate\w*|cudaHostAlloc\w*|"
                       r"cuMemHostRegister\w*|cudaHostRegister\w*|cuMemPoolCreate\w*|cudaMemPoolCreate\w*)\b")
# Native allocations outside torch in the job's processes, with their per-allocation ledger entries (owner policy (b)):
# none. torch's own CUDA allocations go through its caching allocator (capped by (a)); the CUDA context and module
# images are the (b) overhead allowance.
RAW_ALLOCATION_LEDGER = {}
# The driver symbols kernels.py binds (the only native code of the twin's processes): module load and launch only.
DRIVER_SYMBOLS = {"cuCtxGetCurrent", "cuCtxSetCurrent", "cuDeviceGet", "cuDevicePrimaryCtxRetain", "cuFuncSetAttribute",
                  "cuGetErrorString", "cuInit", "cuLaunchKernel", "cuModuleGetFunction", "cuModuleLoadData",
                  "cuModuleUnload"}
# What PR #5's Zig engine uses of TensorFold's cuda module: no DeviceBuffer / HostBuffer / Carveout (the only TensorFold
# paths that allocate); the pipeline owns every buffer and the engine borrows them.
ZIG_CUDA_USES = {"Driver", "Error", "Function", "Module", "Stream", "launch"}
# Every mention in TensorFold v1.0.7 (K6_TF_DIR = its root): the calls are reached only through DeviceBuffer, HostBuffer
# and Carveout, which PR #5's engine never uses.
TF_CALL_SITES = {"zig/src/cuda/memory.zig:61": "call, DeviceBuffer.alloc (cuMemAlloc_v2)",
                 "zig/src/cuda/memory.zig:175": "call, HostBuffer.alloc / allocMapped (cuMemHostAlloc)",
                 "zig/src/cuda/carveout.zig:128": "call, Carveout (cuMemHostRegister_v2)",
                 "zig/src/cuda/carveout.zig:130": "call, Carveout (cuMemHostRegister_v2)",
                 "zig/src/cuda/carveout.zig:16": "comment (alignment constant)",
                 "zig/src/cuda/abi.zig:157": "declaration, driver table", "zig/src/cuda/abi.zig:159": "declaration, driver table",
                 "zig/src/cuda/abi.zig:162": "declaration, driver table"}


def call_sites(root, files):
    found = {}
    for p in files:
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if RAW_ALLOC.search(line) and not line.lstrip().startswith(("//", "#")):
                found[f"{p.relative_to(root)}:{n}"] = line.strip()
    return found


KIT_HARNESS = r"""
die() { echo "die: $*"; exit 9; }
%s
mode=$1 KD=$2
case $mode in
  modes-only)   # the field failure (dgx02, T0 1791816926): root, a-w modes, no private namespace
    chmod -R a-w "$KD"
    [[ ! -w $KD/MANIFEST.txt ]] && echo "OLD-CHECK: pass" || echo "OLD-CHECK: writable"
    why=$(kit_writable "$KD") && echo "PROBE: $why"
    why=$(freeze_kit "$KD") && echo FROZEN || echo "REFUSED: $why" ;;
  job | spoofed | mount-fails | user)
    [[ $mode == mount-fails ]] && mount() { return 32; }
    [[ $mode == spoofed ]] && export K6_PRIVATE_MOUNTS=1    # claims a namespace it never entered
    [[ $mode == user ]] || private_mounts "$@"
    why=$(freeze_kit "$KD") && echo FROZEN || { echo "REFUSED: $why"; exit 3; }
    for t in 'echo x >> "$0/MANIFEST.txt"' 'touch "$0/new"' 'chmod u+w "$0/MANIFEST.txt"' 'rm -f "$0/a.py"' \
             'mv "$0/a.py" "$0/b.py"' 'ln -s x "$0/l"'; do
      if bash -c "$t" "$KD" 2> /dev/null; then echo "TAMPER-WROTE: $t"; else echo "TAMPER-REFUSED: $t"; fi
    done ;;
esac
"""


class KitImmutability(unittest.TestCase):
    """The extracted kit is immutable for every process of the job, root included (dgx02 runs jobs as root, where
    a-w file modes bind nothing). Root cases run in their own pid namespace, so pid 1 stands for the node's init, and
    their own mount namespace, so nothing touches the host's mounts: as real root when the suite runs as root, else as
    uid 0 of a user namespace (hosts that restrict unprivileged user namespaces need the real-root form)."""

    def section(self):
        m = re.search(r"^# ---- kit immutability \(begin\)\n(.*?)^# ---- kit immutability \(end\)$",
                      NODE_JOB.read_text(), re.S | re.M)
        if not m:
            raise AssertionError(f"kit immutability section not found in {NODE_JOB}")
        return m.group(1)

    def run_case(self, mode, root=True):
        with tempfile.TemporaryDirectory() as d:
            kd = Path(d) / "kit-0123456789ab"
            kd.mkdir()
            (kd / "MANIFEST.txt").write_text("# kit\n")
            (kd / "a.py").write_text("print(1)\n")
            if mode == "user":
                subprocess.run(["chmod", "-R", "a-w", str(kd)], check=True)
            before = {f.name: f.read_bytes() for f in kd.iterdir()}
            harness = Path(d) / "harness.sh"
            harness.write_text(KIT_HARNESS % self.section())
            argv = ["bash", str(harness), mode, str(kd)]
            if root:    # pid 1 is the outer shell (bash -c with a second command never execs its first)
                fake = [] if os.geteuid() == 0 else ["--user", "--map-root-user"]
                argv = ["unshare", *fake, "--pid", "--fork", "--mount-proc", "bash", "-c",
                        'bash "$@"; exit $?', "-", str(harness), mode, str(kd)]
            r = subprocess.run(argv, capture_output=True, text=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
            after = {f.name: f.read_bytes() for f in kd.iterdir()}
            mounted = str(kd) in Path("/proc/self/mountinfo").read_text()
            subprocess.run(["chmod", "-R", "u+w", d], check=True)
            return r, before, after, mounted

    def assert_tamper_refused(self, r, before, after, owner_chmod=False):
        """Every write, create, remove, rename and link is refused; chmod too, except that a non-root owner may still
        change its own files' modes (that needs a deliberate second write, and verify_kit runs before and after every
        step)."""
        wrote = [line for line in r.stdout.splitlines() if line.startswith("TAMPER-WROTE")]
        chmod = ['TAMPER-WROTE: chmod u+w "$0/MANIFEST.txt"'] if owner_chmod else []
        self.assertEqual(wrote, chmod, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("TAMPER-REFUSED"), 6 - len(wrote))
        self.assertEqual(after, before)                     # byte for byte, no file added or removed

    def test_modes_alone_bind_nothing_for_root(self):
        r, before, after, _ = self.run_case("modes-only")
        self.assertIn("OLD-CHECK: writable", r.stdout, r.stderr)    # the old check refused every root job ...
        self.assertIn("PROBE: a file can be created in", r.stdout)  # ... because root really can write the kit
        self.assertIn("REFUSED: root job outside its own mount namespace", r.stdout)

    def test_root_job_kit_is_read_only_bind_mount_in_own_namespace(self):
        r, before, after, mounted = self.run_case("job")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("FROZEN", r.stdout)
        self.assert_tamper_refused(r, before, after)
        self.assertFalse(mounted)                           # the read-only mount ended with the job's namespace

    def test_root_refuses_without_a_real_private_namespace(self):
        for mode, why in (("spoofed", "REFUSED: root job outside its own mount namespace"),
                          ("mount-fails", "REFUSED: read-only bind mount of")):
            with self.subTest(mode=mode):
                r, before, after, _ = self.run_case(mode)
                self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
                self.assertIn(why, r.stdout)
                self.assertNotIn("FROZEN", r.stdout)

    def test_non_root_job_kit_is_read_only_by_modes(self):
        r, before, after, _ = self.run_case("user", root=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assert_tamper_refused(r, before, after, owner_chmod=True)

    def test_job_freezes_kit_and_reverifies_it_after_each_step(self):
        text = NODE_JOB.read_text()
        self.assertIn('(( PREFLIGHT_ONLY == 0 )) && [[ $OPT != --budgets ]] && private_mounts "$@"', text)
        self.assertIn('if ! why=$(freeze_kit "$KD"); then\n    RESULT="kit $KD is not read-only for this job', text)
        self.assertNotIn("[[ ! -w $KD/MANIFEST.txt ]]", text)
        run_step = text[text.index("run_step() {"):text.index("\n}\n", text.index("run_step() {"))]
        before, after = run_step.index("verify_kit"), run_step.rindex("verify_kit")
        self.assertLess(before, run_step.index('wait "$CUR_PID"'))
        self.assertGreater(after, run_step.index('wait "$CUR_PID"'))
        self.assertIn("[[ $verdict == ok ]] && (( kit_ok )) || return 1", run_step)


class KitVerifiedAfterEachStep(unittest.TestCase):
    """The post-step MANIFEST check (node-job.sh verify_kit) catches whatever a step could still do to the kit: the
    owner's chmod then a write (non-root), a new file, a removed file, an edited MANIFEST; the untouched kit passes."""

    def verify_kit(self):
        m = re.search(r"^verify_kit\(\) \{\n.*?^\}$", NODE_JOB.read_text(), re.S | re.M)
        self.assertIsNotNone(m, f"verify_kit not found in {NODE_JOB}")
        return m.group(0)

    def run_case(self, tamper):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            (src / "pkg").mkdir(parents=True)
            (src / "pkg" / "a.py").write_text("print(1)\n")
            (src / "b.txt").write_text("b\n")
            sums = subprocess.run("find . -type f | sort | xargs sha256sum", shell=True, cwd=src, check=True,
                                  capture_output=True, text=True).stdout
            (src / "MANIFEST.txt").write_text("# kit\n" + sums)
            kit = Path(d) / "kit.tar.gz"
            subprocess.run(["tar", "-czf", str(kit), "-C", str(src), "."], check=True)
            kd = Path(d) / "kit-0123456789ab"
            kd.mkdir()
            subprocess.run(["tar", "-xzf", str(kit), "-C", str(kd)], check=True)
            subprocess.run(["chmod", "-R", "a-w", str(kd)], check=True)
            script = f"KIT={kit}\n{self.verify_kit()}\n{tamper}\nverify_kit {kd}\n"
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                               env={"PATH": "/usr/bin:/bin"}, cwd=d)
            subprocess.run(["chmod", "-R", "u+w", d], check=True)
            return r

    def test_untouched_kit_verifies(self):
        r = self.run_case(":")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_any_change_after_a_step_fails(self):
        kd = "kit-0123456789ab"
        cases = {
            "owner chmod then edit": f"chmod u+w {kd}/pkg/a.py {kd}/pkg && echo x >> {kd}/pkg/a.py",
            "new file": f"chmod u+w {kd}/pkg && : > {kd}/pkg/new.py",
            "removed file": f"chmod u+w {kd} && rm -f {kd}/b.txt",
            "edited MANIFEST": f"chmod u+w {kd} {kd}/MANIFEST.txt && echo '# x' >> {kd}/MANIFEST.txt",
        }
        for name, tamper in cases.items():
            with self.subTest(name):
                r = self.run_case(tamper)
                self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("kit:", r.stdout)


class RawAllocationLedger(unittest.TestCase):
    """(b): every native allocation call site of every binary of the job, against the (empty) ledger."""

    def binaries(self):
        k6 = REPO / "kernels" / "cuda" / "kandinsky6"
        return {
            "twin python processes": [p for p in PKG.rglob("*.py") if not p.name.startswith("test_")],
            "fatbins (device code)": [p for p in k6.rglob("*") if p.suffix in (".cu", ".cuh", ".zig")],
            "PR5 Zig engine": list((REPO / "src" / "engines" / "kandinsky6").rglob("*.zig")),
        }

    def test_every_binary_matches_ledger(self):
        ledger = {}
        for name, files in self.binaries().items():
            self.assertTrue(files, name)
            ledger[name] = {"files": len(files), "call_sites": call_sites(REPO, files)}
            self.assertEqual(ledger[name]["call_sites"], RAW_ALLOCATION_LEDGER, name)
        self.assertIn("kernels.py", {p.name for p in self.binaries()["twin python processes"]})
        out = os.environ.get("K6_LEDGER_OUT")
        if out:
            with open(out, "w") as f:
                json.dump({"pattern": RAW_ALLOC.pattern, "ledger": RAW_ALLOCATION_LEDGER, "binaries": ledger,
                           "driver_symbols": sorted(DRIVER_SYMBOLS), "zig_cuda_uses": sorted(ZIG_CUDA_USES),
                           "tensorfold_call_sites_unused": TF_CALL_SITES}, f, indent=1, sort_keys=True)

    def test_pattern_catches_every_family(self):
        for name in ("cuMemAlloc_v2", "cuMemAllocManaged", "cuMemAllocHost", "cuMemAllocAsync", "cuMemHostAlloc",
                     "cudaMalloc", "cudaMallocHost", "cudaMallocManaged", "cudaMallocAsync", "cuMemCreate",
                     "cudaHostAlloc", "cuMemHostRegister_v2", "cudaHostRegister", "cuMemPoolCreate"):
            self.assertRegex(f"x = lib.{name}(p, n)", RAW_ALLOC, name)

    def test_twin_binds_only_load_and_launch(self):
        src = (PKG / "kernels.py").read_text()
        self.assertEqual(set(re.findall(r'"(cu[A-Z]\w+)"', src)), DRIVER_SYMBOLS)

    def test_zig_engine_borrows_buffers(self):
        uses = set()
        for p in self.binaries()["PR5 Zig engine"]:
            uses |= set(re.findall(r"\bcuda\.([A-Za-z_]+)", p.read_text()))
        self.assertEqual(uses, ZIG_CUDA_USES)

    @unittest.skipUnless(os.environ.get("K6_TF_DIR"), "K6_TF_DIR (TensorFold v1.0.7 root) not set")
    def test_tensorfold_call_sites_are_the_known_unused_ones(self):
        root = Path(os.environ["K6_TF_DIR"])
        self.assertEqual(set(call_sites(root, list((root / "zig").rglob("*.zig")))), set(TF_CALL_SITES))


if __name__ == "__main__":
    unittest.main()
