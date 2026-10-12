# Kandinsky-6 DiT: the Spark GPU job kit

Scripts that qualify the Kandinsky-6 Pro-distill DiT kernels and the block runtime against the twin and upstream
kandinsky-6 `eac06e39` on DGX Spark nodes (GB10, sm_121):

- `kit.sh` runs on the dev host. It builds one immutable, reproducible tarball from committed `HEAD` (no network, no
  docker, no build step).
- `node-job.sh` runs one job on one node, inside a time window, from that tarball. Its header documents the hard
  contract (kit sha, deadline, memory policy, disk, GPU lock, offline, oracle settings), and the script checks each item.
- `supervise.py` runs each step under the GB10 host guard; `after.sh` and `classify.py` classify a node's jobs after
  they end (PASS, FAIL_LOCAL or sticky STOP_ALL). `test_policy.py` and `test_classify.py` check all of it offline.

No step builds an image, downloads anything or reads weights: every input is random at the checkpoint's shapes
(`test_ops.random_weights`, and gate.py's own inputs). The oracle environment comes prebuilt from the dev host, as
either the pinned image or the offline venv fallback (see "What the stager places on each node").

## Jobs

All times and sizes below are **estimates**. None of them is measured yet, and each is deliberately conservative.
`max min` is a hard timeout. A step starts only if its `max min` ends before `DEADLINE - 10 min`. The output budget
and the step's memory admission (see Memory) are checked before each step starts. Every job begins with an `env` step
(3 min max) that records the environment.

| job | steps (one supervised process tree each) | purpose | est. minutes | max min | Torch GiB per CUDA process (step budget GiB) | output |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke` | `env`, `import`, `kernels`, `test_ops_small` | checks that the twin and pinned upstream import from the kit only, the three fatbins load and match the MANIFEST digests, and `test_ops --small --fp32-linear tf32_rna` runs | 6 | 3 + 3 + 3 + 15 | 1, 1, 1, 4 (8, 8, 8, 11) | < 10 MB |
| `twin` | `env`, `test_ops_{tf32_rna,tf32_truncate,fp32}` | `test_ops` at the checkpoint's widths (model_dim 4096, ff 16384, 2 fused + 1 text block, 231 visual tokens): the bit-exact assertions plus the faithfulness receipt for each Fp32Linear form | 1 + 3 x 8 = 25 | 3 + 3 x 20 | 16 (23) | < 30 MB |
| `ops` | `env`, `gate_ops_tf32_rna`, `gate_fp32_{tf32_truncate,fp32}` | every `gate --phase ops` site once (37 sites plus the RoPE / time-freq tables), then only the 12 `fp32/*` sites for the other two Fp32Linear forms (no other ops site depends on the form) | 1 + 25 + 2 x 5 = 36 | 3 + 40 + 2 x 10 | 16 (27) | < 7 GB budget (expected far less) |
| `blocks` | `env`, `gate_blocks_{tf32_rna,tf32_truncate,fp32}` | `gate --phase blocks` (time chain, both text blocks, embeddings, one fused block at 50,220 tokens) for each form | 1 + 3 x 10 = 31 | 3 + 3 x 20 | 28 (39) | < 30 GB budget (expected far less) |

`bash node-job.sh <job> --budgets` prints the exact table: each step's Torch budget, process counts, budget and
admission level.

Basis for the estimates:

- The `twin` Torch budget comes from test_ops' shapes. A fused block holds about 29 B / 60, or 0.5 B, parameters
  (about 1 GiB in bf16). The reduced forward plus embeddings is a few GiB (an estimated 12 GiB peak), and the twin and
  native modules share one weight dict; 16 GiB leaves room for the caching allocator.
- The `ops` and `blocks` Torch budgets cover gate.py's own per-site estimates (`gate --list`, run in the oracle image):
  the largest ops site is `ffn/video_up` at 9.3 GiB, the largest block unit `block/fused` at 20.3 GiB. gate.py runs one
  child per site and refuses a site whose estimate exceeds `--max-gib`, which node-job.sh sets to the step's Torch
  budget. Each child records its real `max_memory_allocated`. Minutes assume 10-20 s per site plus native 50,220^2
  attention; replace them with measured ones after the first window.
- Each step's supervision record has its lowest MemAvailable and its largest sampling gap (`steps.tsv`).
- `STEP_MAX_MIN=<n>` overrides the max minutes of the `twin`, `ops` and `blocks` steps.

## Two nodes, one window (N = 120 min)

Owner deadlines, with T0 the window start: no new work at T0+100 (N-20), everything cleaned by T0+110 (N-10), expiry
T0+120 (N). Each job gets `DEADLINE=T0+110`, the time its node must be cleaned by. node-job.sh then starts no step
after T0+90 and only a step whose max minutes end by T0+100; its guard stops a step still running at T0+100, and the
job itself at T0+110. That leaves T0+100 to T0+110 for after, export and cleanup.
Node A is dgx01 (`WORK=/home/dgx01/k6-d1-qual`), node B is dgx02 (`WORK=/root/k6-d1-qual`, reached through dgx01);
each node writes only its own `WORK`. Every job is a separate host run in its own memory scope (discrete, resumable): `MODES` splits a job
by Fp32Linear form, so a job that no longer fits is resumed in a later window instead of truncated.

Per job, estimate / worst case (worst = every step at its timeout), in minutes:

| # | node A (dgx01) | est | max | node B (dgx02) | est | max |
| --- | --- | --- | --- | --- | --- | --- |
| H | handover at T0: read-only node checks, production marker, kit staged, `vmstat` before | 2 | 3 | the same | 2 | 3 |
| 0 | setup: kit sha256, take `node-job.sh` out of it, preflight (first one checks every prefix file) | 6 | 10 | the same | 6 | 10 |
| 1 | `smoke` | 6 | 24 | `smoke` | 6 | 24 |
| 2 | `twin` (all 3 forms) | 25 | 63 | `ops` (every site under tf32_rna, then `fp32/*` under the other two) | 36 | 63 |
| 3 | `blocks` MODES="tf32_rna tf32_truncate" | 21 | 43 | `blocks` MODES=fp32 | 11 | 23 |
| 4 | after: `after.sh` classification, then export of the results tarballs + sha256 | 2 | 5 | the same | 2 | 5 |
| 5 | cleanup: remove the extracted kit (results stay in WORK) | 2 | 5 | the same | 2 | 5 |
| | **end (est)** | **T0+64** | | | **T0+65** | |

Jobs H and 0-2 (the required set) fit even at their worst case: they end by T0+100 on both nodes. Job 3 fits the
estimate with at least 39 min to spare; if earlier steps run long, `node-job.sh` records its steps `not-run: time` and
that job is resumed in a later window with the same kit and MODES. After, export and cleanup run between T0+100 and
T0+110 at the latest.

## Memory (owner GB10 uma-node policy, 2026-10-12)

On GB10, device memory comes from the same pool as host memory, and a cgroup's `memory.max` does not contain device
allocations: D1 run 1791795729 held 8 GiB of device memory inside a 6 GiB scope. So no cgroup and no nvidia-smi figure
bounds anything here (`nvidia-smi` reports GB10 memory as `[N/A]`). The jobs apply all four rules of the policy:

- **(a) Torch cap.** Every CUDA process of a step calls `budget.apply` before its first allocation:
  `set_per_process_memory_fraction` caps torch's caching allocator at the step's declared Torch budget, read back and
  refused unless the readback equals what was set and the cap is at most the budget. Each process writes
  `<step>/caps/cap-<pid>.json`, and node-job.sh rejects the step unless every record checks out and there is at least
  one per CUDA process. gate.py's parent is budgeted as host-only and refuses if CUDA ever starts in it. Preflight
  opens no CUDA context.
- **(b) Non-allocator overhead.** The CUDA context, module images, cuBLAS / cuDNN state and host Python have not been
  measured for this kit yet. Until they are, the allowance is 6 GiB per CUDA process (2 x an estimated 3 GiB) and
  4 GiB per host-only process (2 x 1.5 GiB, raised to the 4 GiB minimum), plus 1 GiB for the job script.
  - Raw CUDA allocation ledger: empty. The fatbins are loaded into torch's primary context with `cuModuleLoadData` and
    launched with `cuLaunchKernel` on torch tensors; no code path calls `cuMemAlloc` / `cudaMalloc` or a relative.
    `test_policy.py` scans the sources and fails on any such call without a ledger entry.
  - step budget = CUDA processes x (Torch + 6) + host processes x 4 + 1.
- **(c) Host guard.** A step starts only if MemAvailable >= its budget + the 12 GiB floor (`FLOOR_GIB`, never lower).
  Steps run one at a time, so this is the sum of the budgets alive at once. `supervise.py` runs each step in its own
  session as its child subreaper. It samples MemAvailable every 0.1 s and SIGKILLs the whole process tree at once
  (descendants that double-fork or `setsid` included) when MemAvailable is below the floor, two samples are more than
  0.5 s apart, `/proc` is unreadable, the kernel `oom_kill` counter rises, the step times out, or the job stops it.
- **(d) Sticky STOP_ALL.** After a node's jobs end, `after.sh` kills and stops their scope units, shows no job process
  remains, captures `/proc/vmstat` `oom_kill` and both journals since T0, and classifies with `classify.py`. Its STOP_ALL
  rules (global OOM, foreign memcg OOM, unattributed or uncounted OOM kills, earlyoom / systemd-oomd kills, NVRM Xid,
  unreadable journals, unverified cleanup) are copied verbatim from the D1 classifier (sha256 `db360863…`, checked by
  `test_classify.py`). A STOP_ALL stops all work on both nodes and is reported to the owner.

Each job still runs on the host in its own cgroup v2 scope with `memory.max` <= 96 GiB and `MemorySwapMax=0`
(`systemd-run --scope`, which execs the job inside the scope, so no launcher client stays outside). That is host
accounting and a refusal check, not a GPU bound. Preflight refuses a shared or unlimited cgroup, and refuses to run
inside a container, whose launcher would sit outside its pid namespace. The guard logs the scope's `memory.current` and
the RSS of the job's launcher / login ancestors outside the scope every second. Helper discovery fails closed: every
live ancestor's `/proc` stat and cgroup must read and parse (no partial list), and a running step is stopped if either
figure turns unreadable.

## What the stager places on each node

All of it is files-only before T0; loading, extraction and setup happen after T0, inside the window.

1. **The kit tarball** (`k6-dit-kit-<commit12>.tar.gz`) and its sha256 (`KIT_SHA`, printed by `kit.sh` and stored
   in its `.sha256` file). Nothing else from the repository is needed.
2. **The oracle environment**, one of:
   - **the image** (dev-host only: lists the pinned closure and runs offline import / `gate --list` checks; jobs are
     refused inside a container, see Memory): the `docker save` tar of `k6-dit-oracle:<lock12>`, built on the dev host from
     `oracle.Dockerfile` (pinned `python:3.13-slim-bookworm` digest, offline `--require-hashes` install of the
     `lock.txt` wheel set: `torch==2.14.1+cu130` cp313 aarch64 and its declared dependencies, nothing else). The tar
     sha256 and the loaded image ID are both checked. Inside it, `VENV=/usr/local`.
   - **the shared oracle prefix** (staging-owned, the route the jobs use): sealed `k6-oracle-prefix.tar.gz` sha256 `74118af7…`, PBS
     3.13.16 + the same torch pins (+ numpy 2.5.4, safetensors 0.8.0; freeze sha256 `d793ad20…`), `MANIFEST.sha256`
     sha256 `a1af1b3b…`; one shared read-only extraction per node (staging-owned, after T0), and preflight verifies
     every file.
   - **the offline venv fallback** (if the prefix is unavailable): the same wheel set with `lock.txt` and `SHA256SUMS`, plus python-build-standalone
     `cpython-3.13.16+20261009-aarch64-unknown-linux-gnu-install_only.tar.gz`; `setup-venv.sh` builds `$WORK/venv`
     with no network and checks the pins.

   `flash_attn`, `flash_attn_interface` and `sageattention` are absent on purpose: the native reference uses SDPA.
3. **The NVIDIA driver**: `libcuda.so.1` with CUDA 13.0 support, and `nvidia-smi`, on a GB10 (capability 12.1), and
   systemd (`systemd-run --scope`, or `--user --scope` without root) for the job's memory scope.
4. **`WORK`**: a directory this job alone writes, on a filesystem with at least the job's budget plus 5 GB free.
   `WORK` (the extracted kit, results, caches and TMPDIR) stays under 200 GB.
5. **Host tools**: bash, util-linux `flock`, coreutils (`timeout`, `sha256sum`, `du`, `df`), GNU tar, gzip, awk and cmp, and
   systemd's `systemctl` / `journalctl` for `after.sh`. The guard and `supervise.py` read `/proc` directly (no procps
   for the jobs; `after.sh` uses `pgrep`). Preflight checks for each tool the jobs use.

The third-party Python imports of the DiT path, listed in full. Each entry is `file:line` in the kit; upstream paths
are under `kandinsky6-upstream/`.

| import | where | required |
| --- | --- | --- |
| `torch`, `torch.nn`, `torch.nn.functional` | `kandinsky6_dit/kernels.py:22`, `block.py:19`, `native.py:29-30`, `test_ops.py:31-33`; `kandinsky/core/components/dit.py:13-15`, `rope.py:3-4`, `kandinsky/core/tensors.py:5-6`, `kandinsky/core/algo/prepare_ropes.py:9-10`, `kandinsky/runtime/kernels/sdpa.py:5-6` | yes |
| `torch`, `torch.nn`, `torch.nn.functional` (gate) | `kandinsky6_dit/gate.py:100-102`; the rest of gate.py's imports are the standard library (`gate.py:79-98`) | yes |
| `libcuda.so.1` (ctypes, the driver API) | `kandinsky6_dit/kernels.py:161` | yes (driver) |
| `flash_attn` | `kandinsky/runtime/kernels/flash_attention_2.py:6`, via `attention_engine.py:7` via `core/components/attention/dispatch.py:6` via `dit.py:23` | no: `try`/`except Exception` |
| `flash_attn_interface` | `kandinsky/runtime/kernels/flash_attention_3.py:6`, via `attention_engine.py:8` | no: `try`/`except Exception` |
| `sageattention` | `kandinsky/runtime/kernels/sage.py:6`, via `attention_engine.py:9` | no: `try`/`except Exception` |

Everything else in the path is the standard library (`kernels.py:15-20`, `native.py:19-27`, `test_ops.py:26-29`,
`attention_engine.py:5`) or the pinned checkout's own modules. These are `piflow_dit.py:5`, `dit.py:17-24`,
`rope.py:6`, `attention/dispatch.py:6` and `attention_engine.py:7-10`; the package `__init__.py` files of
`kandinsky.core`, `kandinsky.core.components`, `kandinsky.core.components.attention` and `kandinsky.core.algo` are
empty.

`native.upstream()` stands empty namespace modules in for `kandinsky`, `kandinsky.runtime` and
`kandinsky.runtime.kernels`. As a result, `kandinsky/__init__.py` (the whole pipeline: diffusers, transformers,
pydantic) and `kandinsky/runtime/kernels/__init__.py:9` (`..base`, pydantic) never execute. The smoke `import` step
imports gate.py too, and fails if anything else is needed.

## Commands

On the dev host, from a clean, committed tree with gate.py committed:

```sh
FATBINS=/path/to/fatbins K6_SRC=/tmp/kandinsky6-upstream \
    bash tools/twin/stk_twin/kandinsky6_dit/spark/kit.sh /path/to/kits
# prints KIT_SHA=<sha256> and CREATED_UTC=...; the same inputs + CREATED_UTC reproduce the same sha256
```

Each job runs on the node's host. Node A runs `JOB=smoke`, `twin`, then `blocks` with `MODES="tf32_rna tf32_truncate"`;
node B runs `smoke`, `ops`, then `blocks` with `MODES=fp32`. Each job's preflight (`--preflight`) goes first.
With the shared oracle prefix: staging keeps one sealed copy per node in `k6-staging-1018d582/oracle`
(`k6-oracle-prefix.tar.gz`, 3,344,315,223 bytes, sha256 `74118af7c8381f74ed810c6ddb3b009edc9f280d13844f2ace70f6f65ba0c474`)
and, after T0, provides one shared read-only extraction per node (`PREFIX`, staging's path); no job copies the prefix
into its own `WORK`. The job runs on the host inside its own scope, and preflight checks every prefix file against the
MANIFEST:

```sh
systemd-run --scope --quiet --unit k6-dit-$JOB-$T0 -p MemoryMax=96G -p MemorySwapMax=0 \
    env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE RUN_ID=$JOB-$T0 \
    DEADLINE=$DEADLINE KIT_SHA=$KIT_SHA MODES="$MODES" WORK=$WORK KIT=$WORK/k6-dit-kit-<commit12>.tar.gz \
    VENV=$PREFIX ORACLE_MANIFEST=$PREFIX/MANIFEST.sha256 \
    ORACLE_MANIFEST_SHA256=a1af1b3b5c25bb343ebe8ecc71aaa406199a9c2c74c5957c1425251fdc124b90 \
    bash $WORK/node-job.sh $JOB       # node-job.sh taken from the verified kit
```

The full MANIFEST check adds an estimated 2-4 min to the first preflight (inside the 10 min setup maximum);
`ORACLE_VERIFY=0` skips it on later preflights of the same window.
Use `systemd-run --user` where the node account has no root. The offline venv fallback is `setup-venv.sh` followed by
the same command with `VENV=$WORK/venv` and no `ORACLE_*` variables.

After the node's last job, `after.sh` (from the verified kit, outside every job scope) classifies the node:

```sh
bash $KD/repo/tools/twin/stk_twin/kandinsky6_dit/spark/after.sh $WORK/results/after-$T0 $T0 $PREFIX/bin/python3.13 \
    $NODE $WORK/vmstat-$T0.before $WORK k6-dit-smoke-$T0.scope ... -- $WORK/results/smoke-$T0/job-result.json ...
```

`$WORK/vmstat-$T0.before` ("EPOCH OOM_KILL" from `/proc/vmstat`) is written before the node's first job.

Each job exits 0 only if every step ran and returned 0. `twin` returns 1 if a bit-exact assertion fails; it still
writes its receipt, and the next mode still runs. A step the supervisor or guard killed, or whose policy records do
not check out, stops the job. Optional environment variables: `FLOOR_GIB` (12, never lower), `SEED` (0),
`STEP_MAX_MIN`, `MODES` and `RUN_ID`.

## What to send back

From each node, send these:

- every `$WORK/results/<RUN_ID>.tar.gz` and its `.sha256`, as printed in each job's `RESULTS_SHA=` line;
- the console output of the preflights and of the jobs.

Each results directory holds:

- `env/env.json`: hostname, `nvidia-smi -q` summary, driver, kernel, kit sha and MANIFEST header, python, torch, CUDA
  and cuDNN versions, the relevant environment, the env step's Torch cap;
- `nvidia-smi-q.txt` and `preflight.log`;
- `job.log`: every decision, plus du before and after each step;
- `steps.tsv` and `SUMMARY.txt`: per step, status, rc, minutes, max min, Torch and step budget, lowest MemAvailable,
  largest sampling gap, Torch cap records and du;
- `job-result.json` (`k6-dit-job-result-v1`): the job's result and steps, read by `classify.py`;
- `memavail.log`: one sample a second, as epoch, MemAvailable, step, cgroup `memory.current` and helper RSS;
- `guard.log`: the guard's stops;
- `<step>/log`, `<step>/exit_code`, `<step>/command.txt`, `<step>/supervision.json` and `<step>/caps/`;
- the receipts: `test_ops` `receipt.json`, `kernels.json`, and gate.py's `--out` directory.

From `after.sh`: `$WORK/results/after-$T0/after.json` and `after.txt`.

Do not send the extracted kit (`$WORK/kit-<sha12>`); it is read-only and identical to the tarball.
