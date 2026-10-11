# Kandinsky-6 audio kernels: GPU qualification kit

Standalone harness that compares every kernel in `kernels/cuda/kandinsky6/audio/audio.cu` with the native
Kandinsky-6 twin, bit for bit. Two independent jobs, one GPU node each, run in parallel:

| Job | Node | Covers |
| --- | --- | --- |
| A `decoder` | node A (either DGX node) | activation-normalization sweep, MPConv1D stored->prepared weights (CPU and device), prepared conv1d with gains, scaled SiLU, mp_sum, latent unscale, mel affine, clamp, nearest-exact upsample, QKV layout, SDPA backend preflight, full decoder chain at T=218 |
| B `vocoder` | node B (the other node) | SnakeBeta, alias-free up/down FIR, dilated conv1d, conv_transpose1d, avg3 / add / clamp, one AMPBlock1, full mel->PCM BigVGAN v2 for mel [1,128,436], checkpoint key-set assertion, weight-norm fold diagnostic |

**Route: reference-only, requires Captain approval.** The native reference imports the pinned Kandinsky-6 checkout
(restricted-native `edm2_utils` objects: `normalize`, `MPConv1D`, `mp_silu`, `mp_sum`, through `vae_modules` / `vae` /
`autoencoder`) as a black box and only calls it. Nothing from `edm2_utils.py` is copied, quoted, translated or vendored
here, and the harness never reads it as text: it hashes the file and refuses the checkout unless
`sha256(kandinsky/core/components/mmaudio/ext/autoencoder/edm2_utils.py)` is
`9a61ca2344f3b6ddd1a745e89669725eb182575453a395c224e7df8152e956a7` and (when git reports one) the commit is
`eac06e39b904940baff2c408e925cdb0c4529269`. The candidate normalization reads the native `normalize` signature's
`eps` default (introspection only, recorded as `normalize_eps_signature_default`; `--norm-eps` overrides).

Results are **qualification evidence only after review.** A comparison passes only with **0 unequal float32 bit
patterns** (int32 view); max abs diff and NaN-mask mismatch are diagnostics. There are no tolerances. The sweep cases
(normalization orders, acc / div variants, SDPA candidates) are expected to contain failing variants: they are data.

## Prerequisites (per node)

| Item | Requirement |
| --- | --- |
| Interpreter | Staging's sealed portable prefix, mounted read-only (e.g. `/opt/k6-oracle`): `PREFIX/bin/python3.13` (CPython 3.13.16) and `PREFIX/lib/python3.13/site-packages` with torch 2.14.1+cu130 (the native pin), numpy, safetensors and, optionally, cuda-bindings (ctypes over libcuda.so.1 otherwise). Set `ORACLE_MANIFEST` (sha256sum-format list of the prefix files) and `ORACLE_MANIFEST_SHA256`; both are required and `run_job.sh` checks them and every listed file. Run with `-I -B`. The container (node-specific NGC image for glibc and the driver only, network none) runs this prefix; the image's own torch is never the reference. |
| CUDA | NVIDIA driver with libcuda.so.1 for SM121 (GB10). Kernels load through cuda-python (`cuda.bindings.driver`, falling back to `cuda.cuda`), or ctypes over libcuda.so.1, into torch's current context. |
| Upstream | `UPSTREAM` = Kandinsky-6 checkout at `eac06e39` (d2-kit/upstream), with the sha256 check above. |
| Checkpoints | `MODELS/audio_vae/diffusion_pytorch_model.safetensors` 1711158920 B, sha256 `dfccdf9d0d8d3de270a23b097400f21adb91f4ec683bdc26369cb65b08a5af87` (decoder keys under `vae.`); `MODELS/vocoder/diffusion_pytorch_model.safetensors` 488692656 B, sha256 `abf0536824bdce8dcf8ee10b815770111f4b5d05d0122e1bab4b5719324edca7`. Both are verified before use; a mismatch is an infrastructure failure. `--weights synthetic` replaces them with seeded synthetic weights (smoke test only, not evidence). |
| Fatbin | `nvcc -fatbin -O3 --fmad=false -gencode=arch=compute_121a,code=sm_121a -gencode=arch=compute_120a,code=sm_120a -o audio.fatbin kernels/cuda/kandinsky6/audio/audio.cu` (no `--use_fast_math`, `-ftz`, `-prec-div=false`, `-prec-sqrt=false`). Record the nvcc version with the results. |
| Limits | `ulimit -v unlimited` (CUDA reserves a large VA space); at least 20 GiB free under OUT. No network, docker or image builds. |

Paths on the nodes are relative to the workspace `~/k6-staging-1018d582` (dgx01: `/home/dgx01`, dgx02: `/root`).
Pass them explicitly; `run_job.sh` has no defaults.

## Run

```sh
# node A (ORACLE_MANIFEST and ORACLE_MANIFEST_SHA256 exported)
kit/run_job.sh A "$WS" /opt/k6-oracle upstream MODELS kit/audio.fatbin out/k6a-A
# node B (in parallel)
kit/run_job.sh B "$WS" /opt/k6-oracle upstream MODELS kit/audio.fatbin out/k6a-B
```

`run_job.sh A|B WORKSPACE PREFIX UPSTREAM MODELS FATBIN OUT [jobs.py args]`: relative paths resolve against
WORKSPACE; OUT must be new or empty. It verifies the kit's `SHA256SUMS` (when present) and the prefix manifest, then runs
`PREFIX/bin/python3.13 -I -B jobs.py JOB --fatbin FATBIN --native-src UPSTREAM --models MODELS --out OUT
--site-packages PREFIX/lib/python3.13/site-packages --time-limit-min 25` under `timeout 28m` and logs to `OUT/job-JOB.log`.
Exit 0 means the job ran (mismatches included); 2 / 3 infrastructure failure; 124 timeout; 65 precondition.

**Probe stage (job A, `--probe`).** Appended to job A's own argument list (`run_job.sh A ... OUT --probe`), it runs
after job A's cases in the same process and changes none of them. For the native ops whose order is not yet reproduced
it records which kernels the native libraries chose and dumps what an offline reading needs: decoder
`mid.block_1.conv1` (gain 1) and `conv_out` (gain 1 and the loaded `learnable_gain + 1`) with `conv_in` as a passing
control; scaled SiLU with torch's `exp(-x)`, `sigmoid` and `silu` of the same input and both k6a variants; the native
strided SDPA [1,1,218,2048] plus the math backend's steps composed from public ops (scaled q / k^T, scores, softmax and
`_safe_softmax`, output), each compared with the native output; and, from `MODELS/vocoder` (key set asserted as in
job B), vocoder `conv_pre` and the dilation-1 convs `resblocks.9/10/11.convs1.0` with `resblocks.9.convs1.1` (dilation
3) as a passing control, with stored weight and bias. Each op's input uses the same seed tag as its job A / job B case,
so its native sha256 must equal that case's (`native_sha_equals_case_*` in job A; vocoder against the job B record).
Each op runs once under `torch.profiler` (CPU + CUDA): kernel names, counts and device time in the JSON, the Chrome trace
(grid, block, registers, shared memory) in `OUT/probe/traces/`. cuDNN (`CUDNN_LOGLEVEL_DBG=3`), cuBLAS
(`CUBLAS_LOGINFO_DBG=1`) and cuBLASLt (`CUBLASLT_LOG_LEVEL=5`) API logs for the whole process go to
`OUT/probe/logs/{cudnn,cublas,cublaslt}.log`; jobs.py sets them before importing torch, and refuses `--probe` when torch
was already imported (a same-process wrapper) unless the wrapper exported exactly those six values beforehand.
`probe/summary` fails the job (exit 2) above 1 GiB of probe output. The probe is diagnostic only, never qualification.

Local CPU smoke (tiny shapes, native paths only, no kernel launches):
`python -I -B jobs.py A --dry-run-cpu --weights synthetic --native-src UPSTREAM --out /tmp/k6a-dry/A`.

## Budgets (each job)

| | Limit | Estimate |
| --- | --- | --- |
| Wall clock | < 30 min (28 min hard timeout, 25 min self-limit checked between cases; later cases are recorded as skipped) | A 5-10 min (checkpoint hashing, CPU load + prepare, 29 re-preparations, 572 normalization launches, 4 decoder passes); B 5-10 min (4 BigVGAN passes of ~1 TFLOP through the direct kernels). Plus ~1 min prefix verification before the timeout starts. |
| GPU memory | < 40 GiB (self-checked) | A < 4 GiB; B < 3 GiB |
| Host RSS | < 100 GiB (self-checked) | A < 10 GiB (1.7 GB checkpoint + encoder/decoder on CPU); B < 4 GiB |
| Disk | < 20 GB (`ulimit -f` caps each file) | A ~0.8 GB (CPU-prepared decoder weights); B ~10 MB; probe < 1 GiB (checked), ~0.1 GB expected |

## Outputs (in OUT)

- `results-A.json` / `results-B.json`: environment (torch, CUDA, driver, device / capability, cuDNN, TF32 flags off,
  deterministic algorithms on, CUBLAS_WORKSPACE_CONFIG, harness and fatbin sha256), native provenance, every case with
  status, timing, per-comparison `unequal` / `max_abs_diff` / `nan_mask_mismatch` / `pass_`, the variants chosen for
  the chains, and a summary. Rewritten after each case, so a killed job leaves its partial results.
- `results-X-sha256.tsv`: sha256 of every input and output tensor, per case.
- `tensors/`: raw little-endian fp32 (`.f32le`) with sha256 in the JSON `dumps`. Job A: the **CPU-prepared weights of
  all 29 decoder convs** (`prepared_cpu/vae.decoder.<path>.weight`, the gate for the Zig host prepare on aarch64), the
  seeded latent rows [1,218,40], native z and mel, kernel mel per acc. Job B: the seeded mel, native PCM, kernel PCM per acc.
- `job-X.log`, `SHA256SUMS` over everything in OUT.

What each job records: A: per normalization variant (orders 0 / 1, order 2 with R = 2..1024, order 3 = PyTorch's
reduction order `k6a_l2norm_torch` split from the recorded multiprocessor count and max threads per multiprocessor;
s rounding; fused / split denominator) the unequal count per case and the variants zero in all cases; the 29 checkpoint convs re-prepared
on CPU and on the device against the loaded CPU-prepared weights, plus synthetic shapes [2048,40,3], [2048,2048,3],
[1024,2048,1], [6144,2048,1], [128,512,3] with the device each ran on; conv1d over prepared weights for gain 1, 1.3 and
the loaded `learnable_gain + 1`, acc 0 / 1; div 0 / 1 for SiLU, mp_sum, latent unscale; SDPA for q/k/v [1,1,218,2048]
and [1,1,436,2048] in the native strided layout and contiguous: `_fused_sdp_choice`, each backend forced (ok / error,
hash, equal to default), and `k6a_sdpa_math` against each; the decoder chain free-running and teacher-forced per node.
The chain's normalization and attention are candidate kernels (best op-level variant of the same run). B: the
checkpoint key set (667 plain tensors, no `weight_g` / `weight_v` / parametrizations; the native loader runs no fold),
per-op results on the checkpoint's own modules (stage 3, 96 channels), one AMPBlock1 step by step, and the vocoder
chain per stage, free-running and teacher-forced, with the final PCM.

## Kit

`make_kit.sh FATBIN DESTDIR` copies these six files and the fatbin (as `audio.fatbin`) into a read-only directory with
`SHA256SUMS`, then writes a reproducible `k6a-kit-<fatbin sha12>-<harness sha12>.tar.gz` and its `.sha256`. It refuses
to overwrite an existing kit.
