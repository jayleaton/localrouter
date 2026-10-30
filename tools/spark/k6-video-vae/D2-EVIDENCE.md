# D2 memory evidence on GB10: policy `gb10-unified-d2-v2`

What the D2 (Kandinsky-6 video VAE) producers write, and what the shared staging runner's D2-only readers must check.
D2 owns the producers (`d2-allocator-guard.py`, `k6v-qual`, `run-probe.sh`, `run-oracle.sh`, `run-engine.sh`). The
runner owns admission, the segment sequencer, the evidence readers, the probe classifier, the per-node gate, sticky
FAIL, cleanup and receipt sealing. `d2_evidence.py` is the reference checker: the rules below, written as code and run
on fixtures. It does not replace the runner's own checks.

The policy is distinct. It is never satisfied by D3's `gb10-unified-v1` evidence or by D0's terms, and the reverse
holds too. Any job, node, oracle or variable not listed here is refused.

## 1. Accounting and admission

`accounted = max(max(memory.current, memory.peak) + helper RSS + 4 GiB, (baseline MemAvailable - MemAvailable) + 4 GiB)`
is sampled every 0.5 s and must stay `<= max_resident_bytes`. Also: host MemAvailable floor 12 GiB, and the node guard.
`--memory 48g --memory-swap 48g` (no swap) is host (CPU) accounting only: on GB10 the cgroup does not contain GPU
allocations, so it is never a GPU bound. The probe shows whether D2's measured allocations are charged to
the cgroup. That is evidence for those allocations only, not a promise about later ones. So the host-drop term,
the floor and the guard stay in force for the whole job.

The owner's GB10 policy, as D2 meets it:

- (a) Torch: the native per-process fraction caps the caching allocator at `allocator_cap_bytes`, read back equal to
  the value set and rechecked every sample. Native: every device buffer is reserved in k6v-qual's ledger (cap
  `allocator_cap_bytes`) before it is allocated.
- (b) The CUDA context, modules, library workspaces and driver allocations are not measured. Each process declares them
  as `unmeasured_overhead_bytes` = 8 GiB (4 GiB, doubled while unmeasured).
  `process_budget_bytes = allocator_cap_bytes + unmeasured_overhead_bytes <= max_resident_bytes`, and
  `operator_spec.py` refuses a job whose budget does not fit.
- (c) The runner admits a launch only if host MemAvailable >= the `process_budget_bytes` of every job still resident
  on the node (unknown residency counts as resident) + one floor, rechecked before each launch. The container is killed
  when MemAvailable crosses the floor.
- (d) A global or foreign memcg OOM, an earlyoom or systemd-oomd kill, an NVRM Xid or an unverified cleanup is a sticky
  STOP_ALL (the runner's D1 classifier). Evidence under `gb10-unified-d2-v1` is refused.

| Job id | Node | Oracle | Producer | Needs | Timeout | Cap / budget | max_resident | Disk / output |
|---|---|---|---|---|---|---|---|---|
| `d2-probe-torch` | dgx01 | d2-pybundle `ead020cb…` (NumPy 2.5.3) | Torch probe | nothing | 600 s | 38 / 46 GiB | 46 GiB | 1 GB / 256 MiB |
| `d2-oracle-short` | dgx01 | the same | 6 Torch segments | this window's PASS of `d2-probe-torch` | 2100 s | 38 / 46 GiB | 48 GiB | 12 GB / 4 GiB |
| `d2-probe-native` | dgx02 | none | native probe | nothing | 600 s | 24 / 32 GiB | 32 GiB | 1 GB / 256 MiB |
| `d2-engine` | dgx02 | none | native ledger | this window's PASS of `d2-probe-native` | 1140 s (engine `TIMEOUT` 1020 s) | 24 / 32 GiB | 40 GiB | 8 GB / 1 GiB |

`d2-oracle`, the full-length oracle, is in the approved admission table. It is not scheduled: the approved run is
`d2-oracle-short` only. `operator_spec.py` writes the four jobs above. Each carries `memory_policy`,
`allocator_cap_bytes`, `unmeasured_overhead_bytes`, `process_budget_bytes`, `container_memory_bytes` (host accounting),
`host_memavailable_floor_bytes`, `max_resident_bytes`, and where it applies
`requires_probe_pass` (the gate) and `segments` (the sealed order). Each also carries the window fields `latest_start_seconds_from_open` and
`finish_margin_seconds` (section 7).

## 2. Runner variables

Every D2 process runs under `env -i` with exactly these variables. The guard refuses a missing or extra variable.

| Variable | Probes and engine | Oracle segments |
|---|---|---|
| `K6_MEMORY_POLICY` | `gb10-unified-d2-v2` | same |
| `K6_MEMORY_NONCE` | fresh 32 lowercase hex: the producer's nonce | the job's nonce |
| `K6_JOB_ID`, `K6_OUTPUT` | job id; absolute output directory | same |
| `K6_ALLOCATOR_RECEIPT`, `K6_ALLOCATOR_LOG` | `$K6_OUTPUT/allocator-ready.json`, `$K6_OUTPUT/allocator-metrics.jsonl` | not used |
| `K6_SEGMENT_NONCES` | not used | `env-probe:<hex>,s_dec:<hex>,f_dec:<hex>,s_enc:<hex>,n_enc:<hex>,n_dec:<hex>`: exactly that order, all distinct from each other and from the job nonce |

Torch processes also get `PATH`, `HOME`, `LANG`, `PYTHONNOUSERSITE`, `PYTHONDONTWRITEBYTECODE`, `PYTHONPATH` (the
bundle's `site-packages`) and `PYTORCH_ALLOC_CONF = PYTORCH_CUDA_ALLOC_CONF =
backend:native,throw_on_cudamalloc_oom:True`. Native processes also get `PATH`, `HOME` and `LANG`. Both may get
`CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES`. The oracle and torch-probe jobs also need the spec's
`ORACLE_ROOT=/opt/k6-oracle-d2` and `ORACLE_MANIFEST_SHA256`. These are checked by the launcher and are not passed on.

## 3. Producer files (one producer = one process)

Prefix `allocator-` for the probes and the engine, `allocator-<segment>-` for an oracle segment. Every line and
receipt starts with the identity `{"schema": 1, "policy": "gb10-unified-d2-v2", "nonce", "job_id", "pid"}`. Oracle
segments add `"job_nonce"`, `"segment"` and `"segment_index"` (0..5). Times are `CLOCK_MONOTONIC` seconds, the
runner's own clock: containers share the host's, with no time namespace.

| File | Written by | Fields after the identity |
|---|---|---|
| `ready.json` | producer, atomically (`.tmp` + rename, never over an existing one) | `status: "passed"`, `allocator_cap_bytes`, `mem_get_info_total_bytes`, `process_start_monotonic_seconds`, `monotonic_seconds`. Torch: `allocator_backend: "native"`, `total_bytes`, `fraction`, `readback_fraction`, `effective_limit_bytes`, `python_version`, `torch_version`, `sample_period_s`. Native: `runtime: "cuda-driver"`, `engine_sha256` (`/proc/self/exe`), `allocator_backend: "explicit"`, `sample_period_ms` |
| `metrics.jsonl` | producer's own thread, appended, flushed per line | `free_bytes`, `total_bytes`, `allocated_bytes`, `reserved_bytes`, `monotonic_seconds`. Native: `allocated == reserved ==` the ledger. A failed sample is a line with `status: "failed"` and `error`, then exit 67 |
| `terminal.json` | producer, after its sampler stopped (2 s bound) and a final sample | `status: "completed"` (exit 0) or `"failed"`, `payload_exit_code`, `sampler_stopped: true`, `last_sample_monotonic_seconds`, `monotonic_seconds`, `metrics_bytes`, `metrics_sha256`. Probes add `phases_bytes` and `phases_sha256`; native adds `ledger_peak_bytes` |
| `producer-exit.json` | the launcher, only after it reaped the process (`--record-exit`, no CUDA) | `producer_exit_code`, `monotonic_seconds`, `terminal_sha256` (null without a terminal), `pid` (from ready; null without one) |

Caps: Torch 38 GiB (`40802189312`) via the native per-process fraction with read-back. Native 24 GiB
(`25769803776`): `Buf.alloc` refuses an allocation that would exceed it, before allocating. Sample period: 1 s for
payloads, 0.5 s for probes.

### Producer acceptance (final, after the reap)

All of these must hold:

- The identity matches the runner's, and the PID is the same in every file.
- Ready:
  - `status: "passed"` and the cap is exact;
  - `ready - process_start <= 30 s`;
  - Torch: version `2.14.1+cu130`, Python `3.13.16`, `|fraction - readback| <= 1e-12`, `|effective - total * readback| <= 1`;
  - native: runtime `cuda-driver` (`sim` is refused) and a 64-hex `engine_sha256`.
- Samples:
  - at least 2, with strictly increasing time and consecutive gaps of `<= 2 s`;
  - non-negative integers, `total == mem_get_info_total_bytes`, `free <= total`, `allocated <= reserved <= cap`;
  - native also requires `allocated == reserved`;
  - no `status` line.
- Terminal and reaped record:
  - `last sample <= terminal <= reaped`, `terminal - last <= 2 s` and `reaped - terminal <= 2 s`;
  - `last_sample_monotonic_seconds` equals the last sample;
  - `metrics_bytes` and `metrics_sha256` match the file, and `terminal_sha256` matches the terminal receipt;
  - both exit codes are 0.

### Oracle segment sequence (`d2-oracle-short`)

The segments are `["env-probe", "s_dec", "f_dec", "s_enc", "n_enc", "n_dec"]`, in that order, each a fresh process
under `d2-allocator-guard.py`:

- `env-probe` verifies, after its own ready receipt:
  - every bundle file against `MANIFEST.sha256`, and that manifest against `ead020cb…`;
  - the interpreter and the exact pins.
- A case runs `oracle.py --cases <case>` in the guarded process.

Each segment must pass producer acceptance with its own nonce. In addition:

- the PIDs are all distinct;
- segment *i+1*'s `process_start` and its first sample come after segment *i*'s reaped record, so the segments neither
  overlap nor run out of order;
- segment *i+1*'s ready time is no more than 30 s after segment *i*'s reaped record.

A missing segment fails the job. So does a gap, a reorder, an overlap, a reused PID or nonce, or a missing terminal or
reaped record. The launcher ends the sequence at the first segment that fails, times out or does not fit
`JOB_BUDGET_S`. Short-job timeouts: env-probe 240 s, cases 240 s, n_dec 900 s, budget 1980 s.

Limits:

- The fraction caps Torch's caching allocator only. The CUDA context, library workspaces and driver allocations are
  the declared unmeasured overhead (section 1, (b)).
- The tile planner reads `mem_get_info` free, which the cap does not change. A Torch peak above 38 GiB fails the case
  with an OOM; the planner never re-plans.

## 4. Attribution probe (the gate)

`run-probe.sh torch|native` writes the producer files above plus `probe-phases.jsonl`: one sealed line per phase as it
ends (`fsync`). Each line has the identity, then `phase` (`P0`..`P5`), `name`, `index`, `x_bytes`,
`start_monotonic_seconds` and `end_monotonic_seconds`. The allocator evidence starts with the context (P1).

| Phase | Action | Hold | `x_bytes` |
|---|---|---|---|
| P0 `pre-context` | process running, no CUDA library loaded | 10 s | 0 |
| P1 `context` | context created (Torch: import + fraction; native: `cuInit`, primary context), nothing allocated | 10 s | 0 |
| P2 `alloc-4G` | allocate and touch 4 GiB (Torch `empty(uint8).fill_(1)`; native `cuMemAlloc` through the ledger + `cuMemsetD8`), synchronize | 10 s | 4 GiB |
| P3 `free-4G` | free (`del` + `empty_cache`; `cuMemFree`) | 10 s | 4 GiB |
| P4 `alloc-16G` | as P2 with 16 GiB | 10 s | 16 GiB |
| P5 `free-16G` | free | 10 s | 16 GiB |

### Runner samples (input to the classifier)

One row every 0.5 s: `{"monotonic_seconds", "memavailable_bytes", "cgroup_current_bytes", "cgroup_peak_bytes",
"helper_rss_bytes", "accounted_bytes"}`. The runner also records the time it read each phase record:
`phase-reads.json` = `{"P0": t, ...}`.

### Correlation, per phase (any violation is FAIL, never a skipped check)

- Every phase record is present, with the same identity, in order, with the right `name` and `x_bytes`.
- Each phase is held at least 10 s, and no phase overlaps the previous one.
- Each record is read at `end <= t_read <= end + 5 s`.
- Window = samples with `start + 3 s <= t <= end`. The window needs:
  - at least 12 samples;
  - consecutive gaps of `<= 1.0 s`;
  - its first sample `<= 0.75 s` after `start + 3 s`, and its last `<= 0.75 s` before `end`.
- The runner's window gives the medians (`median_low`, so the values stay integers) of MemAvailable, `memory.current`
  and `accounted`.
- From P1 on, the producer's allocator samples in the same window, under the same bounds, give `G`: Torch
  `reserved_bytes`, native `allocated_bytes`.

### PASS / FAIL, for X = 4 GiB (P2, recovery P3) and X = 16 GiB (P4, recovery P5)

Definitions:

- `D = P1.memavail - Palloc.memavail`
- `C = Palloc.current - P1.current`
- `G = Palloc.G - P1.G`
- `dA = Palloc.accounted - P1.accounted`

Every bound is evaluated in integers: `100 * value` against `95 * X`, `110 * X + 100 * 512 MiB`, and so on.

| Check | PASS when |
|---|---|
| raw host drop | `0.95 X <= D <= 1.10 X + 512 MiB` |
| attribution | `C >= 0.95 X` (charged) or `C <= 0.05 X` (uncharged); the same mode for both X; recorded |
| guard quantity | Torch `X <= G <= X + 64 MiB`; native `G == X` |
| no under-count | `dA >= 0.95 X` |
| no double count | `dA <= 1.10 X + 512 MiB` |
| linearity | `abs((dA16 - dA4) - 12 GiB) <= 5 % of 12 GiB` |
| recovery (`memory.current`, never `memory.peak`) | `Pfree.memavail >= P1.memavail - 512 MiB` and `Pfree.current <= P1.current + 256 MiB` |
| context share | `P0.memavail - P1.memavail <= 4 GiB` (recorded: it sizes what is outside the ledger) |

The verdict is `{"verdict": "PASS"|"FAIL", "mode", "errors", "failed", "checks": [{"name", "x_bytes", "value",
"bound", "ok"}]}`.

## 5. Gate and FAIL handling (approved; runner)

**Per-node gate.**

- A node's payload is admitted only with that node's own probe PASS from the same window, bound by hash to the
  probe's sealed receipt.
- The other node may still finish its own work, unless the D1 classifier's sticky STOP_ALL fires (section 1, (d)).
- Any FAIL means no D2 qualification result for that window.

**What counts as FAIL.** Missing, stale or otherwise failed evidence (probe or payload) is a sticky FAIL for that node
and that window.

**What a FAIL does.**

- It blocks every further payload on that node, including a lighter fallback.
- The job stops. Its receipt (phase records, raw samples, quantities, verdict) is sealed read-only and reported.
- Nothing is re-run under another policy.

**Markers.**

- The operator removes their own job marker only after the saved cleanup has been verified.
- After a sticky FAIL, the owner releases.
- The DEV deadman expiry and the production restore owner stay as they are.

## 6. Fixtures and the reference checker

```
python3 -I d2_evidence.py fixtures OUT --qual k6v-qual [--vae vae/diffusion_pytorch_model.safetensors --inputs DIR]
```

This builds and checks every fixture and writes `OUT/EXPECTED.json` (per fixture: the expected and actual verdicts and
the reason) and `OUT/SHA256SUMS`. No GPU is used.

**Real producer code, on fakes:**

- Oracle segments: the guard's own code, each segment a separately forked process, with a fake Torch. The launcher's
  `--record-exit` path runs after each reap.
- Native: the real `k6v-qual --probe --sim` and `--sim` engine run, plus `--record-exit`.
- The Torch probe: the guard's own probe path.

Fixture mode relaxes only the host interpreter version and the native `sim` runtime. The production check refuses
`sim`: this is fixture `native/probe-sim-as-production`.

**Segment failures, each the only defect**, re-bound through metrics → terminal → reaped record:

- reorder, overlap, gap over 30 s;
- reused PID, stale nonce;
- missing segment, missing terminal, missing reaped record;
- nonzero payload exit, sampler not stopped, cap exceeded;
- stale samples, truncated log, dropped log line;
- terminal changed after the reap, late ready.

**Native failures:** ledger over the cap, and ledger fields that differ.

**Probe verdicts:** synthetic, with exact values. Every threshold is tested at its bound (PASS) and one byte or one
sample interval past it (FAIL). The same is done for:

- the 12-sample minimum, the 1.0 s gap, the 0.75 s edges and the 5 s read;
- a missing, reordered or misidentified phase record;
- missing allocator samples, and missing runner sample fields.

Every case also passes in charged, uncharged and native modes.

```
python3 -I d2_evidence.py producer DIR --kind torch|native --job ID --nonce HEX [--prefix allocator-SEG-]
python3 -I d2_evidence.py segments DIR --job d2-oracle-short --job-nonce HEX --nonces env-probe:HEX,...
python3 -I d2_evidence.py probe DIR --kind torch|native
```

## 7. Window budget (owner's itemized rulings)

`operator_spec.py` derives two sealed integers per job from one itemized budget per node, and emits that budget as
`window_budget`. Nothing in it is entered by hand. `python3 -I operator_spec.py --self-test` checks it at its
boundaries. The owner's SHORT95 bounds are superseded: there is no fixed node end and no setup ceiling. Each node's
sequence is solved forward from T0 through every reservation. The window is N whole minutes, at most 180, from the
later node end plus the owner's 10-minute close. Historical seals keep their own SHORT95 budgets unchanged.

**Reference.** All values are whole seconds from T0, the grant's `opened_at`. The window ends at T0 + 60 N, the
grant's `expires_at`. The runner (`run-node-d2-v5.py`) applies the two fields as follows.

| Field | Runner rule | Semantics |
|---|---|---|
| `latest_start_seconds_from_open` (L) | launch cutoff `min(expires - 1200, opened + L)`. The job is admitted only if `now < cutoff`, checked at plan time and again after `docker create`, its ownership inspect and the setup-gate recheck. | Strictly less: a container created at T0 + L is refused. L <= 60 N - 1200, so L always binds. |
| `finish_margin_seconds` (F) | stop `min(expires - 690, expires - F - 60)`. A job starts only if `started + timeout_seconds <= stop`. | Equality is admitted: the payload ends by T0 + 60 N - 60 - F, which is the budget's payload end. |

**Rounding and double counting.** Every reservation is rounded up to whole seconds, and every deadline is the sum of
the reservations before it. A measured row is counted once, in the gap where it ran; a nested row is never added
again. The DEV full-producer audit rows (dgx01 8.377594 / 7.290590 s) are not in the window. That audit runs offline at
pre-stage, against the exact staged tree (`prestage_audit`, below).

**Reservations.**

- **Measured terms, at 2x.** From the sole revised DEV flow (`2d3ba780`, handoff `76d59e0a`): input and control,
  readiness and between-job prelaunch hashes, the tail, and the export's save, fsync and verify.
  - Setup verification is 2x the largest of: the warm setup run (`89bc74c8`), the cold devbox composition (`457d14de`)
    and the DEV setup on the final pack (`2f910179`).
  - Runner checks are 2x the larger of the cold devbox checks (`457d14de`) and the DEV prelaunch hashes.
- **Transfers, at 2x the reference rate.** The input copy is 2 x hops x the 64 MiB pack bound / 19,787,536 B/s.
  The output export is 2 x hops x both jobs' output caps / that rate.
- **Export work.** The export's whole measured chain: source inventory, each hop's copy and readback, full
  verification, fsync of every saved file and directory, then verification again. It is one term, the larger of:
  - the byte-rate hashing floor, 2 x (hops + 1) x the caps / the devbox's 559 MB/s;
  - 2x the full U4 wall time, transfer included (owner ruling: conservative, no subtraction). The DEV flow's rows of
    the same chain (tail plus export save/verify) are a further floor; they never exceed U4.
- **U1.** 60 s per job, for the runner's prelaunch work that no row isolates: marker verification, residency, evidence
  init, output layout and the setup-gate recheck.
- **U2.** 30 s each for the post-create ownership `docker inspect` (before L) and the post-payload state inspects. These
  come on top of the runner's own 125 s pre-payload and 80 s post-payload bounds, which are counted once.
- **U3.** 60 s per SSH hop of every cross-node marker read or owner operation (a route through dgx01 to dgx02 counts
  two hops), listed per gap in `cross_node_operations`. The five owner-marker reads per job are counted on both nodes.
  - dgx01: 23 hops.
  - dgx02: 41 hops.
- **U4.** The export of each node's full synthetic output profile (`d2-u4-export-profile-v1`, `115bf33e`), measured
  on DEV with the export's own sequence at the existing rsync flags and rate limit, and reserved at 2x its full wall
  time. The transport stays a separate U3 term, at 2x the reference rate once per hop, and must cover 2x every measured
  hop's rsync; a hop whose doubled rsync exceeds its per-hop reservation refuses the plan (no fit, no trimming). `operator_spec.py`
  refuses to emit a budget until that measurement's receipt is bound.
- **Allocations.** Handover 300 s, cleanup 120 s and marker removal 60 s. The measured DEV cleanup and release, at 2x,
  must fit inside the last two.
- **Deadlines and slack.**
  - dgx01 keeps at least its earlier deadlines: setup 1267, L1 1416, L2 2245.
  - Both nodes keep at least 300 s of setup slack.
  - Any surplus above a floor is shown as `deadline_floor_slack`.

**Setup.** The setup deadline is `max(floor, reserved + 300)`.

| Node | Handover | Input copy | Input cross-node | Input/control | Setup verification | Reserved | Setup deadline | Slack |
|---|---|---|---|---|---|---|---|---|
| dgx01 | 300 | 7 | 360 | 1 | 77 | 745 | 1267 | 522 |
| dgx02 | 300 | 14 | 960 | 2 | 3 | 1279 | 1579 | 300 |

**Readiness and the gap between jobs.** L1 = setup deadline + the prelaunch items. L2 = L1 + job 1's timeout + the
post-payload items + the prelaunch items.

| Node | Prelaunch cross-node | Runner checks | U1 | Pre-payload bound | Post-create inspect | Strict | Post-payload bound + inspect + cross-node | L1 | L2 |
|---|---|---|---|---|---|---|---|---|---|
| dgx01 | 300 | 23 | 60 | 125 | 30 | 1 | 80 + 30 + 60 | 1806 | 3115 |
| dgx02 | 360 | 2 | 60 | 125 | 30 | 1 | 80 + 30 + 60 | 2157 | 3505 |

**After job 2.** The items are: the post-payload items, the export transport, its cross-node operations and the export
work (to `saved_copy_verified_by`); then cleanup (`cleanup_verified_by`); then marker removal (the node end).

U4 (aggregate `9000190b`, session exit 0, profile `b11f5dcc`): dgx01 303.333 s wall; its one hop's rsync, 198.387 s, doubled is
396.773 s, within its 462 s transport reservation. dgx02 195.702 s wall; its hops' rsync, 62.440 and 54.635 s, doubled
are 124.880 and 109.270 s, each within 136 s. The export work is 2x the full wall: 607 and 392 s.

| Node | Post-payload | Export transport | Export cross-node | Export work | Cleanup | Marker | Node end |
|---|---|---|---|---|---|---|---|
| dgx01 | 80 + 30 + 60 | 462 | 60 | 607 | 60 + 120 | 180 + 60 | 6934 |
| dgx02 | 80 + 30 + 60 | 272 | 240 | 392 | 240 + 120 | 180 + 60 | 6319 |

**Window.** N = 126 minutes: the later node end, 6934, plus the 600 s close, rounded up to whole minutes.

| Node | Job | timeout | output cap | L | F | payload end |
|---|---|---|---|---|---|---|
| dgx01 | `d2-probe-torch` | 600 | 256 MiB | 1806 | 5094 | 2406 |
| dgx01 | `d2-oracle-short` | 2100 | 4 GiB | 3115 | 2285 | 5215 |
| dgx02 | `d2-probe-native` | 600 | 256 MiB | 2157 | 4743 | 2757 |
| dgx02 | `d2-engine` | 1140 | 1 GiB | 3505 | 2855 | 4645 |

**Inputs.**

- Both nodes' checkpoints are owner-verified present at `~/k6-staging-1018d582/models/vae/`. Setup rehashes both files
  on each node in the window (inside setup verification); nothing is copied.
- dgx01's bundle is pre-staged under receipt `c671ed82`.
- Every kit this producer emits is a new pack, so dgx01's kit takes one hop in the window and dgx02's takes two. The
  64 MiB pack bound is reserved for each hop.

**Pre-stage audit (`prestage_audit`).** The full-producer audit moves offline, to the owner-approved DEV pre-stage,
against the exact bytes to stage. The spec's `prestage_audit` covers the whole kit root, `pybundle/` included.

- Counts: `regular_files`, `directories`, `links`, `empty_directories` (always 0) and `regular_bytes`.
- `tree_sha256`: the sha256 of the compact JSON of every entry, sorted by UTF-8 path. Each entry is `["d", path]`,
  `["f", path, bytes, sha256]` or `["l", path, target]`.
- `operator_spec.py` refuses a file outside `MANIFEST.sha256` and `pybundle/MANIFEST.sha256`, a file either manifest
  lists that the tree lacks, an empty directory, and a link set other than the bundle's 1048 links (`4c5cd1da`).
- `build-kit.sh` removes `upstream/.git` and every `__pycache__` outside `pybundle/` before it writes the manifest.
- The pre-stage tool recomputes the receipt on the staged bytes, and the grant binds it. In the window, only the
  existing manifest and PB hashes are checked.

**dgx02's shorter payload.** Every case and record is kept. The engine's own `TIMEOUT` is 1020 s inside a 1140 s job.
The output cap is 1 GiB. The engine's records hold at most 843,015,168 B of raw blobs plus about 69 KB of `ops.jsonl`,
taken from a `--sim` run's exact shapes (receipt `d2-engine-output-sizing` `658be32d`). The engine's GPU time is
unmeasured: a case past `TIMEOUT` fails the job, and the FAIL is sticky.

**Outcomes.**

- A node not ready by its setup deadline aborts setup before any payload. It saves its cleanup evidence and releases
  per the owner's setup-abort protocol. That is a setup abort, not a payload FAIL.
- Once a payload has started, a refusal for L or for the stop is a sticky FAIL (section 5).
