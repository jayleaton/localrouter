# Kandinsky-6 Pro-distill PiFlow sampler and DiT weight loader

The multimodal PiFlow sampler and the DiT checkpoint loader of Kandinsky-6.0 Pro-distill 5 s (Hugging Face
`kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`, revision `1018d5828716e8bea3de99609f45c924de9c5564`), ported from
upstream kandinsky-6 at `eac06e39b904940baff2c408e925cdb0c4529269`: `kandinsky/core/algo/{piflow_sampler,piflow_math,denoise_loop,prepare_latents}.py`,
`kandinsky/runtime/{weights,sampler}.py`, `kandinsky/pipeline/pipeline.py`, `kandinsky/configs/checkpoints/pro-distill.yaml`.
Nothing here has been compiled, run or qualified yet (see Gates).

| File | Contents |
| --- | --- |
| `policy.cu` | `k6_piflow_substep_{bf16,f32}_{bf16,f32}` (one Euler substep of the DX policy rollout), `k6_piflow_video_input_{bf16,f32}` (the DiT's visual input, `pretrain` scheme) |
| `src/engines/kandinsky6/sampler.zig` | `Config`, `Schedule` / `Step`, `Rollout` / `Substep` (host f32 scalars), `Kernels` (load, `substep`, `rollout`, `advance`, `videoInput`), `Memory`, `expectTable` |
| `src/engines/kandinsky6/dit_weights.zig` | `Header` (strict safetensors), `Verify` (strict manifest check), `Plan`, `Device` (BF16 on the device, `get(name) ?u64`), `Resident`, `Checkpoint` |
| `tools/twin/stk_twin/kandinsky6_sampler/` | `capture.py` (upstream, recorded), `twin.py` (sampler.zig's arithmetic and the fatbin), `gate.py` (bit gates S0, S1) |

## Upstream contract

Settings (`pro-distill.yaml` `piflow`, `generation`): `num_steps` 10, `shift` 5.0, `eps` 1e-6, `final_step_size_scale`
0.5, `num_policy_substeps` 128, `dx_num_grid_points` 10, `guidance_weight` 1.0 (no CFG). The pipeline reshapes the
latents to batched `[B, T, H, W, 16]` and `[B, A, 40]` (`pipeline.py:488-532`), so `piflow_denoise_loop` takes its
`video.ndim == 5` branch: with B = 1 every schedule tensor (`tau`, `sigma`, the policy's `raw_t_src` / `raw_t_dst` /
`segment_size`, `num_substeps`, the rollout's `raw_t` / `sigma_t`) has one f32 element, and every substep is active.
PiFlow refuses `sample_audio = False` (`pipeline.py:356-358`): video and audio advance from the same DiT call every
step, video first, and `Kernels.advance` takes both.

Per step (`piflow_sampler.py:90-174`): `segment = base * (final ? fss : 1)` with `base = 1 / (num_steps - (1 - max(fss, eps)))`
in Python floats; `tau_dst = clamp(tau_src - segment, eps)`; `sigma = shift_timesteps(tau_src)`; the DiT at `time =
sigma * 1000` (both modalities); the policy (`DXPolicy`, `piflow_math.py:29-36`) `raw_src = _unwarp_t(sigma)`, `raw_dst =
clamp(raw_src - segment, 0)`, `seg = clamp(raw_src - raw_dst, eps)`, `x0[g] = x_src - sigma * pred[g]`; the rollout
(`piflow_math.py:155-189`) from `tau_src` to `tau_dst` in `n = clamp(round((tau_src - tau_dst) * 128), 1)` substeps of
`(tau_src - tau_dst) / n`, each `x += pi(x, sigma) * (shift_timesteps(clamp(raw - h, 0)) - sigma)` with `pi =
(x - interp(x0, (_unwarp_t(sigma) - raw_dst) / seg)) / clamp(sigma, eps)`. Note the policy's `raw_src` is the unwarped
sigma, not `tau_src`, and its `raw_dst` clamps at 0 where `tau_dst` clamps at `eps`: both are reproduced as upstream
computes them.

## Precision receipt

Every value is f32 unless stated. "f32(x)" is a Python float converted by torch for an f32 tensor op (round to nearest
even). Each op rounds once (Zig never contracts; the kernels use `__f*_rn` intrinsics). No division has a CPU-scalar
divisor, so none takes torch's reciprocal-multiply path; every division is IEEE.

| Value | Upstream | Native | Status |
| --- | --- | --- | --- |
| `base`, `segment` | Python float (f64) arithmetic | f64, then `f32(segment)` once | S0 pending |
| `tau_dst` | `sub(tau, f32(seg))`, `clamp_min(f32(eps))` | same, NaN-propagating clamp | S0 pending |
| `sigma` | `mul(t, f32(5))`, `mul(t, f32(4))`, `add(·, 1)`, `div` | same order | S0 pending |
| `time` | `mul(sigma, f32(1000))` | same | S0 pending |
| `raw_src`, `raw_dst`, `seg` | `mul(sigma, f32(-4))`, `add(·, f32(5))`, `div`; `sub`, `clamp_min(0)`; `sub`, `clamp_min(f32(eps))` | same | S0 pending |
| `n` substeps | `mul(delta, f32(128))`, `round` (half to even), `long`, `clamp(1)` | `rint` half to even | S0 pending |
| `h` | `div(delta, f32(n))` (long tensor promoted) | same | S0 pending |
| `t`, `g0`, `w0`, `w1` | `div`; `clamp(0, 1)`, `mul(f32(9))`, `floor`, `long`, `clamp(0, 8)`; `sub(f32(g0 + 1), tc)`, `sub(tc, f32(g0))` | same, host | S0 (t) / S1 (weights) pending |
| `x0[g]` | `sub(x_src, mul(sigma, pred[g]))`: bf16 operands widened exactly (type promotion) | `__fsub_rn(xs, __fmul_rn(sigma, p))`, recomputed each substep (same ops, same bits) | S1 pending |
| interp | `add(mul(w0, x0[g0]), mul(w1, x0[g0 + 1]))` | `__fadd_rn(__fmul_rn, __fmul_rn)` | S1 pending |
| `pi` | `div(sub(x, x0), clamp_min(sigma, f32(eps)))` | `__fdiv_rn(__fsub_rn)` | S1 pending |
| Euler | `add(x, mul(pi, sub(sigma_next, sigma)))`, `where(active)` all true | `__fadd_rn(x, __fmul_rn(v, dsigma))` | S1 pending |
| state dtype | bf16 noise at step 0 (`prepare_*_latents` default, the dtype is read from the capture), f32 from the first substep on (`where` promotes) | substep 0 reads the bf16 or f32 input, writes f32 | S1 pending |
| visual input | `cat([video, cond, mask])` in the video's dtype; I2V `first_frames.to(video.dtype)` | copy; `__float2bfloat16_rn` for an f32 reference into a bf16 state | S1 pending |
| weights | CPU `.to(bfloat16)` of F32 tensors (RNE); others as stored | `(u + 0x7FFF + lsb) >> 16` for non-NaN; NaN refused (torch's scalar and vectorized NaN bits differ); F16 and every non-BF16 / F32 dtype refused for the BF16 binding | source logic only; host tests written, not run; checkpoint unread |

Not supported (refused, not approximated): `visual_cond_scheme` `i2v` and `tail_cond_first_frame` (and their final
overwrite), `visual_token_type_ids`, B > 1, `sample_audio = False`, `n_grid < 2`, state or grid dtypes other than bf16
and f32, NaN in an F32 weight. NaN in latents is outside the contract (`Schedule` / `Rollout` refuse non-finite scalars;
elementwise NaN payloads are not gated).

## ABI

The pipeline owns the driver, context, stream, every buffer and the fatbin image; the sampler borrows them, never
allocates, synchronizes or reads back. Per job:

```zig
const sampler = @import("kandinsky6_sampler"); // root: src/engines/kandinsky6/sampler.zig (wiring: Central)
var k = try sampler.Kernels.load(&driver, policy_fatbin);
defer k.unload();
const c = sampler.Config.distill;
var sch = try sampler.Schedule.init(c);
while (try sch.next()) |st| {
    try k.videoInput(s, state_v, T * H * W, 16, first, H * W, x_video); // DiT x_video [T*H*W, 33], state dtype
    // upload st.time to time_v / time_a ([1, 1] f32), then the DiT call (PR #5, after merge):
    //   try dit.forward(.{ .x_video = x_video, .x_audio = state_a, ... }, .{ .video = grids_v, .audio = grids_a });
    try k.advance(s, c, st, .{ .src = state_v, .pred = grids_v, .out = next_v }, .{ .src = state_a, .pred = grids_a, .out = next_a }, cancel);
    // swap: state_* = next_* (f32); next_* = the other f32 buffer
}
```

- `Modality.pred`: the DiT ABI's `Outputs`, grid-major `[n_grid, n]` bf16 (`[10, T, H, W, 16]`, `[10, A, 40]`).
- `Modality.out` must not alias `src` (every substep reads the step's input); it is the next step's `src`.
- `cancel` is read between launches; a cancelled step leaves `out` partial and the job is discarded.
- Weights: `Checkpoint.open`, `Verify` over `layout.forEachTensor(Config.distill, &verify)` then `finish`, then
  `Device.load(gpa, io, driver, stream, &ck, cancel, quiesce)`, which returns `Load`. `Device.get` is the lookup for
  `bindWeights(config, &device, ...)`. `Resident` over the manifest gives `needs` the device bytes without I/O.
  - `cancel` is read between 32 MiB chunks. `Quiesce` (`budget_ms`, `poll_ms`, its own `cancel`) bounds every wait:
    reuse of a staging chunk (its event), the final check and the cleanup of a failed load (`cuStreamQuery` polling).
  - Custody is decided by raw driver calls through TensorFold's `Driver.api`: `cuMemcpyHtoDAsync_v2`,
    `cuEventRecord`, `cuEventQuery`, `cuStreamQuery`, `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemFreeHost` and
    `cuEventDestroy_v2`. Every result that decides custody is the CUresult as returned (`RawFailure{rc, call}`), not
    TensorFold's generic error. No TensorFold change is needed. The chunks and events are allocated through
    `HostBuffer.alloc` / `Event.init` (a failure there is already an error) and freed only by the checked raw calls.
    TensorFold's `free` / `deinit`, which drop their results, are never used on them.
  - `ok` (`Loaded{device, staging_release}`) is returned only after `cuStreamQuery` returned CUDA_SUCCESS following
    the last copy. The staging is then freed. A raw failure there is reported in `staging_release`, and the failed
    chunks and events stay owned by the device.
  - `failed` (`Failure`) carries:
    - `primary`, the error that stopped the load: read, NaN, cancel, `CopyEnqueueFailed`, `EventRecordFailed`,
      `EventQueryFailed`, `CopyTimeout`, `NotQuiescent`, an allocation;
    - `primary_raw`, the CUresult when a raw call stopped it;
    - `cleanup`, the `ReleaseResult`;
    - `custody`, null only when `cleanup` is `released`.
  - `ReleaseResult` has four outcomes:
    - `released{idle | quiet}`: every allocation was freed, each by exactly one successful call.
    - `busy{QuiesceResult}`: no evidence, nothing freed. The `QuiesceResult` is `timeout`, `cancelled`, or
      `failed{RawFailure}` for a query CUresult other than not-ready.
    - `partial{first RawFailure}`: evidence held, but a free or destroy failed. This is sticky (below).
    - `no_evidence`.
  - A `Custody` holds the device allocation, the plan, the live pinned chunks and events, the stream, the evidence
    once taken, and the first raw free failure. `outstanding()` lists what is still owned, `failed` counting the items
    whose free failed.
  - `release(io, quiesce)` first takes the evidence: `idle` if nothing was ever enqueued, else the stream's quiet
    answer. A `busy` answer frees nothing and may be repeated, since it only polls.
  - Once the evidence is held, every allocation gets exactly one free or destroy call, ever:
    - A success clears it, and the plan goes with the device allocation it describes.
    - A failure marks it failed and makes the custody sticky. Every later `release` answers the same first failure
      with no call, and `outstanding()` stays the same.
    - A failed free is never reissued. What the driver did with it is unknown, and this API has no proof that a
      retry would be safe and offers none.
  - The owner's obligation: keep the custody until it answers `released`. If it answered `partial`, keep it until the
    CUDA context is destroyed (the worker process exits), then call `forgetAfterContextDestroyed`. That makes no CUDA
    call, and the caller asserts the context is gone. Dropping a custody leaks `outstanding()`.
  - A chunk becomes `unrecorded` before its enqueue call, so a failed enqueue, or an enqueue followed by a failed
    `cuEventRecord`, needs the stream's quiet answer. A recorded event never stands in for it. A failure in
    initialization (a chunk allocated, its event not) leaves the allocated part owned and released on `idle`
    evidence.
  - `Device` has a state of `live`, `release_failed`, `released` or `forgotten`.
    - `get` (the `bindWeights` lookup) answers null unless the device is `live`, so a device whose release has started
      can't be looked up or bound, and its freed plan is never read.
    - `release(io, users, quiesce)` needs every stream in `users` (each stream the owner ran work on that read the
      weights, polled after its last such enqueue) to return CUDA_SUCCESS. With no streams it returns `no_evidence`
      and stays `live`; on `busy` it stays `live` with nothing freed (both repeatable). After the evidence every
      owned allocation gets its one call.
    - `released` is final: a repeated release answers `released` and calls nothing.
    - `partial` leaves the device `release_failed` for good: a repeated release answers the same first failure with no
      call, and lookups stay refused. A staging free that failed after the load (`Loaded.staging_release`) makes the
      eventual release `partial` the same way.
    - `forgetAfterContextDestroyed` (after the context's destruction) moves it to `forgotten`: no lookups, and
      `release` answers `no_evidence` with no call.
  - A free CUDA reports as failed may or may not have released the memory, so it is never retried. It stays owned
    until the context is destroyed. Whether a retry could ever be safe is not established, and none is offered.

## Memory (computed from shapes, not measured)

At 864x480, 5 s (T 31, H 60, W 108; A 218), B = 1:

| Owner | Buffer | Bytes |
| --- | --- | --- |
| sampler | video states, 2 x f32 [3,214,080] | 25,712,640 |
| sampler | audio states, 2 x f32 [8,720] | 69,760 |
| sampler | DiT visual input, f32 [200,880, 33] (bf16 at step 0 fits in it) | 26,516,160 |
| pipeline | step-0 noise, bf16 video + audio (captured dtype) | 6,445,600 |
| pipeline | DiT grid outputs, bf16 [10, 3,214,080] + [10, 8,720] | 64,456,000 |
| DiT | scratch `Plan.bytes()` (PR #5) | 4.0 GB per its README |
| loader | weights, one device allocation, each tensor 256-aligned | `Plan.resident` = sum of `alignForward(2 * elements, 256)`: 60,278,847,520 + at most 4471 x 254 padding for the 30,139,423,760-element manifest |
| loader | pinned staging during load (freed once the stream is quiet; held in `Custody` after a failed load until then) | 4 x (32 MiB + 8 KiB) = 134,250,496 |
| loader | parsed header (host) | `Header.hostBytes()`, read at run time |

On GB10, pinned host and device memory are the same physical memory; the load's peak is resident + staging + header.
The capture itself (upstream at 60.3 GB of bf16 weights plus its own activations) is a separate admission and is not
planned here.

## Gates (bit equality, no tolerance, nothing waived)

| Gate | What | Needs | Status |
| --- | --- | --- | --- |
| checks | `zig build`, `zig build test`, `zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120`, `zig fmt --check build.zig build src tests` and these paths, `git diff --check`, zero new findings vs the base | a granted Spark; the build wiring below | not run |
| S0 | schedule: `capture.py schedule` (upstream's tensors, stub DiT, no checkpoint) -> `src/engines/kandinsky6/sampler_table.json`; `gate.py schedule` (twin) and a Zig test `expectTable(@embedFile("sampler_table.json"))` | a GPU grant (torch + upstream source) | not run; the table does not exist yet, so the Zig test is not added |
| S1 | sampler alone: captured DiT grids through the native policy; per step and chained, the DiT visual input, time, video and audio state after every rollout, final latents | `capture.py run` (the 80.6 GB checkpoint, owner admission and memory plan), the fatbin | not run |
| S2 | full native step: native DiT + native policy from captured step inputs, per step | PR #5 merged, M1b | blocked |
| S3 | final video `[1, 31, 60, 108, 16]` and audio `[1, 218, 40]` latents from the native pipeline | PR #5 merged, M1b | blocked |
| release | decoded frames and audio samples bit-exact (D2 / D3 decoders) | S3 | blocked |
| L-host | dit_weights.zig host tests: header refusals, RNE cast, verify, plan, resident, raw quiesce (quiet / timeout / raw CUresult / cancel), evidence after an event record failure following a successful enqueue, a scripted-`Api` custody (busy stream frees nothing; repeated busy polls free nothing; a raw `cuEventDestroy_v2` failure is sticky (repeated release adds zero free or destroy calls and zero queries, same first CUresult, same `outstanding()`); raw `cuStreamQuery` failure reported; init failure released on idle evidence with no query and its `cuMemFreeHost` failure sticky; raw `cuMemFree_v2` failure sticky with lookups refused; `forgotten` answers `no_evidence` with no call; a clean release is final) | the check run | written, not run |
| L-gpu | load fault fixtures (below) | a granted GPU + a driver shim | specified, not written as code |
| G-host | `gate.py selftest`: exact-shape rejection (same bytes and numel, other shape; extra unit axis), dtype, single bit | torch on the CPU | written, not run |

Future load fault fixtures (L-gpu). Each uses an `LD_PRELOAD` shim over `libcuda.so.1` that forwards every call and
fails the selected one, plus a small synthetic checkpoint (BF16 and F32 tensors spanning more than `ring` chunks):

1. `cuEventRecord` fails on chunk k after its `cuMemcpyHtoDAsync` succeeded (the copy stays enqueued). Expect
   `failed`, `primary` = the record error, and custody retained while the shim holds `cuStreamQuery` at
   `CUDA_ERROR_NOT_READY`. `cleanup` is `timeout` for a short budget; `bytes()` equals the plan plus the pinned chunks;
   once the shim lets the query succeed, `release` returns `quiet`, frees, and later `release` calls stay `quiet`.
2. `cuStreamQuery` returns a sticky error during cleanup. Expect `cleanup.failed` carrying that raw error and custody
   retained (nothing freed).
3. `cuEventQuery` stays not-ready on chunk reuse. Expect `primary` = `CopyTimeout`, then stream-gated cleanup as in 1.
4. Load `cancel` set mid-file. Expect `primary` = `Cancelled`; with a `Quiesce` that has its own unset cancel, cleanup
   `quiet` and `custody` null; with the cleanup cancel set as well, `cleanup` = `cancelled` and custody retained.
5. `cuMemcpyHtoDAsync` fails. Expect custody retained until the stream is quiet, since the chunk was marked before the
   call.
6. The final `cuStreamQuery` stays not-ready. Expect `primary` = `NotQuiescent` with custody; no `ok`.
7. `Device.release` while a kernel reading the weights still runs on a listed stream. Expect `timeout`, device intact;
   then `quiet` after it finishes.
8. The shim fails `cuMemFree_v2`, `cuMemFreeHost` or `cuEventDestroy_v2` once, after quiet, then answers success.
   Expect `partial` with that CUresult, `outstanding()` listing exactly that item as failed, and `get` null. Every
   other allocation gets exactly one successful call. Repeated releases add no free, destroy or query call and answer
   the same CUresult; after the context is destroyed, `forgetAfterContextDestroyed` adds no call.
9. A `Device.get` / `bindWeights` after `release` began (whether `release_failed` or `released`). Expect null /
   `MissingWeight`, with no read of the freed plan.

Each fixture also checks that the shim saw no `cuMemFreeHost` or `cuMemFree` of a held allocation before the quiet
answer.

Until S2 and S3 pass, any tolerance figure is information only. Labels for artifacts: "reference" (upstream output),
"native diagnostic" (unqualified native output), "native qualified" (only after the bit gates pass). No native latent,
frame or sample exists today.

## Integration (Central's wiring proposal; not applied)

Central's proposal (`central-m2-wiring.patch` `c7425ccd...`, HANDOFF `d90957d0...`) against its integration base
`a71e0a6`:
- `policy.cu` is its own source set (prefix `k6_sampler`), compiled per `-Dsm` with `-O3 --fmad=false`. The kernels
  already round every op explicitly, so the flag changes no bits here. It is embedded as `k6_sampler_fatbin_policy`
  and exposed as `kandinsky6_kernels.sampler_policy`, which is empty in host-only builds, so `Kernels.load` refuses it.
- Modules `kandinsky6_sampler` (root `src/engines/kandinsky6/sampler.zig`) and `kandinsky6_dit_weights` (root
  `src/engines/kandinsky6/dit_weights.zig`, renamed from `weights.zig` so it never displaces Central's tracked snapshot
  loader `src/engines/kandinsky6/weights.zig`). Each imports only `cuda` and has its own test, run by `zig build test`
  and `test-unit`. The `kandinsky6` engine imports neither.
- Landing: these sources and the patch land atomically in one commit, and only after the admitted Spark checks on the
  integrated tree show zero new findings against the refreshed integration base. The M2 sources stay on base
  `5d5cc87` until that is authorized.
- Later, not part of the wiring:
  - after S0, `src/engines/kandinsky6/sampler_table.json` and the `expectTable` test;
  - after PR #5 and M1b, the pipeline. Which sampler it runs, this one or Central's `piflow.zig` /
    `k6_piflow_rollout`, is a runtime decision then, made explicitly (never a silent selection). Text encoders, RoPE
    and time tables and the noise come from their own components or captures, never re-derived here. The engine
    entry's `needs` comes from `Resident` + `Memory` + `Plan.bytes()`.
