# Kandinsky-6 DiT: the Spark GPU job kit

Two scripts qualify the Kandinsky-6 Pro-distill DiT kernels and the block runtime against the twin and upstream
kandinsky-6 `eac06e39` on DGX Spark nodes (GB10, sm_121):

- `kit.sh` runs on the dev host. It builds one immutable, reproducible tarball from committed `HEAD` (no network, no
  docker, no build step).
- `node-job.sh` runs one job on one node, inside a time window, from that tarball. Its header documents the hard
  contract (kit sha, deadline, memory guard, disk, GPU lock, offline, oracle settings), and the script checks each item.

No step builds an image, downloads anything or reads weights: every input is random at the checkpoint's shapes
(`test_ops.random_weights`, and gate.py's own inputs). The oracle environment comes prebuilt from the dev host, as
either the pinned image or the offline venv fallback (see "What the stager places on each node").

## Jobs

All times and sizes below are **estimates**. None of them is measured yet, and each is deliberately conservative.
`max min` is a hard timeout. A step starts only if its `max min` ends before `DEADLINE - 10 min`. The output budget
and the declared peak are checked before each step starts.

| job | steps (one python process each) | purpose | est. minutes | max min | est. peak GiB (declared) | output |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke` | `import`, `kernels`, `test_ops_small` | checks that the twin and pinned upstream import from the kit only, the three fatbins load and match the MANIFEST digests, and `test_ops --small --fp32-linear tf32_rna` runs | 5 | 3 + 3 + 15 | 4 (step peak 8) | < 10 MB |
| `twin` | `test_ops_{tf32_rna,tf32_truncate,fp32}` | `test_ops` at the checkpoint's widths (model_dim 4096, ff 16384, 2 fused + 1 text block, 231 visual tokens): the bit-exact assertions plus the faithfulness receipt for each Fp32Linear form | 3 x 8 = 24 | 3 x 20 | 12 (24) | < 30 MB |
| `ops` | `gate_ops_tf32_rna`, `gate_fp32_{tf32_truncate,fp32}` | every `gate --phase ops` site once (37 sites plus the RoPE / time-freq tables), then only the 12 `fp32/*` sites for the other two Fp32Linear forms (no other ops site depends on the form) | 25 + 2 x 5 = 35 | 40 + 2 x 10 | 9.3 per site (24, 16) | < 7 GB budget (expected far less) |
| `blocks` | `gate_blocks_{tf32_rna,tf32_truncate,fp32}` | `gate --phase blocks` (time chain, both text blocks, embeddings, one fused block at 50,220 tokens) for each form | 3 x 10 = 30 | 3 x 20 | 20.3 (32) | < 30 GB budget (expected far less) |

Basis for the estimates:

- The `twin` figures come from test_ops' shapes. A fused block holds about 29 B / 60, or 0.5 B, parameters (about
  1 GiB in bf16). The reduced forward plus embeddings is a few GiB, and the twin and native modules share one weight
  dict.
- The `ops` and `blocks` peaks are gate.py's own per-site estimates (`gate --list`, run in the oracle image): the
  largest ops site is `ffn/video_up` at 9.3 GiB, the largest block unit `block/fused` at 20.3 GiB. gate.py runs one
  child per site, checks each estimate against `--max-gib` (GATE_MAX_GIB, default 90) and records the real peak.
  Minutes assume 10-20 s per site plus native 50,220^2 attention; replace them with measured ones after the first
  window.
- The guard records each step's real peak in `SUMMARY.txt`. That peak is the larger of the step session's RSS plus
  GPU memory and the drop in MemAvailable since the step started. GB10 memory is unified, so a device allocation
  shows up as a drop in MemAvailable, not in RSS.
- `STEP_MAX_MIN=<n>` overrides the max minutes of the `twin`, `ops` and `blocks` steps.

## Two nodes, one window (N = 120 min)

Owner deadlines, with T0 the window start: no new work at T0+100 (N-20), everything cleaned by T0+110 (N-10), expiry
T0+120 (N). `node-job.sh` enforces them with `DEADLINE=T0+120`: a step starts only if its max minutes end by T0+110.
Node A is dgx01 (`WORK=/home/dgx01/k6-d1-qual`), node B is dgx02 (`WORK=/root/k6-d1-qual`, reached through dgx01);
each node writes only its own `WORK`. Every job is a separate container run (discrete, resumable): `MODES` splits a job
by Fp32Linear form, so a job that no longer fits is resumed in a later window instead of truncated.

Per job, estimate / worst case (worst = every step at its timeout), in minutes:

| # | node A (dgx01) | est | max | node B (dgx02) | est | max |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | setup: sha256 of image tar + kit, `docker load`, check image ID, preflight | 6 | 10 | the same | 6 | 10 |
| 1 | `smoke` | 5 | 21 | `smoke` | 5 | 21 |
| 2 | `twin` (all 3 forms) | 24 | 60 | `ops` (every site under tf32_rna, then `fp32/*` under the other two) | 35 | 60 |
| 3 | `blocks` MODES="tf32_rna tf32_truncate" | 20 | 40 | `blocks` MODES=fp32 | 10 | 20 |
| 4 | export: results tarballs + sha256 | 2 | 5 | the same | 2 | 5 |
| 5 | cleanup: `docker rmi`, remove the extracted kit (results stay in WORK) | 2 | 5 | the same | 2 | 5 |
| | **end (est)** | **T0+59** | | | **T0+60** | |

Jobs 0-2 (the required set) fit even at their worst case: they end by T0+91 on both nodes, inside T0+100. Job 3 fits
the estimate with 40 min to spare; if earlier steps run long, `node-job.sh` records its steps `not-run: time` and that
job is resumed in a later window with the same kit and MODES. Export and cleanup always run before T0+110.

Memory is bounded and measured for the whole job, at most 100 GiB in total. Each job runs in its own cgroup v2 with
`memory.max` <= 96 GiB (the container with `--memory 96g`, or a host `systemd-run --scope -p MemoryMax=96G`);
preflight refuses a shared or unlimited cgroup. All of the job's processes (guard, tar, python, page cache) are
charged to it. Driver allocations are not cgroup-charged, so the guard adds nvidia-smi's device-wide `memory.used`
every second, plus a fixed 2 GiB reserve for the processes outside the cgroup (the docker / systemd-run client and
the login shell). The GPU accounting fails closed: `[N/A]`, "Not Supported", an empty or partly numeric answer, a
timeout or an error fails preflight, refuses a step and kills a running one; no estimate replaces it. The guard kills
a step at cgroup + GPU + reserve > 100 GiB, and a step starts only if that total plus its declared peak fits.
MemAvailable >= 8 GiB stays as supplemental headroom. If GB10's driver does not report `memory.used`, these jobs do
not run until a hard-bounded alternative for device allocations is agreed. The largest declared step peaks are 32 GiB
(`blocks`), 24 GiB (`twin`, `ops`) and 8 GiB (`smoke`); gate.py's largest site estimates are 20.3 and 9.3 GiB.

## What the stager places on each node

All of it is files-only before T0; loading, extraction and setup happen after T0, inside the window.

1. **The kit tarball** (`k6-dit-kit-<commit12>.tar.gz`) and its sha256 (`KIT_SHA`, printed by `kit.sh` and stored
   in its `.sha256` file). Nothing else from the repository is needed.
2. **The oracle environment**, one of:
   - **the image** (preferred): the `docker save` tar of `k6-dit-oracle:<lock12>`, built on the dev host from
     `oracle.Dockerfile` (pinned `python:3.13-slim-bookworm` digest, offline `--require-hashes` install of the
     `lock.txt` wheel set: `torch==2.14.1+cu130` cp313 aarch64 and its declared dependencies, nothing else). The tar
     sha256 and the loaded image ID are both checked. Inside it, `VENV=/usr/local`.
   - **the shared oracle prefix** (staging-owned): sealed `k6-oracle-prefix.tar.gz` sha256 `74118af7…`, PBS
     3.13.16 + the same torch pins (+ numpy 2.5.4, safetensors 0.8.0; freeze sha256 `d793ad20…`), `MANIFEST.sha256`
     sha256 `a1af1b3b…`; one shared read-only extraction per node (staging-owned, after T0), and preflight verifies
     every file.
   - **the offline venv fallback**: the same wheel set with `lock.txt` and `SHA256SUMS`, plus python-build-standalone
     `cpython-3.13.16+20261009-aarch64-unknown-linux-gnu-install_only.tar.gz`; `setup-venv.sh` builds `$WORK/venv`
     with no network and checks the pins.

   `flash_attn`, `flash_attn_interface` and `sageattention` are absent on purpose: the native reference uses SDPA.
3. **The NVIDIA driver**: `libcuda.so.1` with CUDA 13.0 support, and `nvidia-smi`, on a GB10 (capability 12.1); with
   the image, the NVIDIA container runtime (`--gpus all`).
4. **`WORK`**: a directory this job alone writes, on a filesystem with at least the job's budget plus 5 GB free.
   `WORK` (the extracted kit, results, caches and TMPDIR) stays under 200 GB.
5. **Tools** (present in the image): bash, util-linux `flock` and `setsid`, coreutils (`timeout`, `sha256sum`, `du`,
   `df`), GNU tar, gzip, awk and cmp. The memory guard reads `/proc` directly (no procps). Preflight checks for each.

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

Each job is one container run from the loaded oracle image (`IMAGE` is the checked image ID). The container
verifies the kit sha256 before taking `node-job.sh` out of it, and writes only the bind-mounted `WORK`:

```sh
# node A: WORK=/home/dgx01/k6-d1-qual   node B: WORK=/root/k6-d1-qual   (the kit tarball inside WORK)
docker run --rm --name k6-d1-$JOB --user "$(id -u):$(id -g)" --gpus all --network none \
    --memory 96g --memory-swap 96g \
    -e DEADLINE -e KIT_SHA -e MODES -e WORK=/work -e KIT=/work/k6-dit-kit-<commit12>.tar.gz -e VENV=/usr/local \
    -v "$WORK":/work "$IMAGE" bash -c 'echo "$KIT_SHA  $KIT" | sha256sum -c --quiet - &&
    tar -xzOf "$KIT" ./repo/tools/twin/stk_twin/kandinsky6_dit/spark/node-job.sh > /tmp/node-job.sh &&
    exec bash /tmp/node-job.sh "$@"' _ $JOB        # append --preflight for the preflight
```

Node A runs `JOB=smoke`, `twin`, then `blocks` with `-e MODES="tf32_rna tf32_truncate"`; node B runs `smoke`, `ops`,
then `blocks` with `-e MODES=fp32`. Each job's preflight goes first.
With the shared oracle prefix: staging keeps one sealed copy per node in `k6-staging-1018d582/oracle`
(`k6-oracle-prefix.tar.gz`, 3,344,315,223 bytes, sha256 `74118af7c8381f74ed810c6ddb3b009edc9f280d13844f2ace70f6f65ba0c474`)
and, after T0, provides one shared read-only extraction per node (`PREFIX`, staging's path); no job copies the prefix
into its own `WORK`. The job runs on the host inside its own memory-capped scope, and preflight checks every prefix
file against the MANIFEST:

```sh
systemd-run --scope --quiet -p MemoryMax=96G -p MemorySwapMax=0 env -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
    DEADLINE=$DEADLINE KIT_SHA=$KIT_SHA MODES="$MODES" WORK=$WORK KIT=$WORK/k6-dit-kit-<commit12>.tar.gz \
    VENV=$PREFIX ORACLE_MANIFEST=$PREFIX/MANIFEST.sha256 \
    ORACLE_MANIFEST_SHA256=a1af1b3b5c25bb343ebe8ecc71aaa406199a9c2c74c5957c1425251fdc124b90 \
    bash $WORK/node-job.sh $JOB       # node-job.sh taken from the verified kit
```

The full MANIFEST check adds an estimated 2-4 min to the first preflight (inside the 10 min setup maximum);
`ORACLE_VERIFY=0` skips it on later preflights of the same window.
Use `systemd-run --user` where the node account has no root. The offline venv fallback is `setup-venv.sh` followed by
the same command with `VENV=$WORK/venv` and no `ORACLE_*` variables.

Each job exits 0 only if every step ran and returned 0. `twin` returns 1 if a bit-exact assertion fails; it still
writes its receipt, and the next mode still runs. Optional environment variables: `MIN_GIB` (8), `SEED` (0),
`GATE_MAX_GIB` (90) and `STEP_MAX_MIN`.

## What to send back

From each node, send these:

- every `$WORK/results/<job>-<UTC>.tar.gz` and its `.sha256`, as printed in each job's `RESULTS_SHA=` line;
- the console output of the preflights and of the jobs.

Each results directory holds:

- `env.json`: hostname, `nvidia-smi -q` summary, driver, kernel, kit sha and MANIFEST header, python, torch, CUDA and
  cuDNN versions, the relevant environment;
- `nvidia-smi-q.txt` and `preflight.log`;
- `job.log`: every decision, plus du before and after each step;
- `steps.tsv` and `SUMMARY.txt`: per step, status, rc, minutes, peak GiB, max min and du;
- `memavail.log`: one sample a second, as epoch, MemAvailable, step, RSS, GPU, drop and use;
- `guard.log`: the guard's kills;
- `<step>/log`, `<step>/exit_code` and `<step>/command.txt`;
- the receipts: `test_ops` `receipt.json`, `kernels.json`, and gate.py's `--out` directory.

Do not send the extracted kit (`$WORK/kit-<sha12>`); it is read-only and identical to the tarball.
