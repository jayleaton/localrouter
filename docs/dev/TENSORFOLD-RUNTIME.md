# TensorFold runtime pin

LocalRouter builds TensorFold's CUDA runtime from source, host side only. This page records the pinned revision,
what LocalRouter consumes from it and the contract engines may rely on.

## Pin

| | |
|---|---|
| Release | TensorFold v1.1.4 (annotated tag object `585df8ba`, peeled commit below) |
| Commit | `eab8582819688325e371907e4706e1c602b8a273` |
| Zig package hash | `tensorfold-1.1.4-CphTLduviwLsoiZR_ltHyW5xZtYz0ZnoMMJIxn1J6X13` (computed by `zig fetch` from the commit) |
| Previous | v1.0.7 `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b`, `tensorfold-1.0.7-CphTLULywQAUe2LxgT29-AserW8gMmK9C-ltOxIaw1pG`; before it `88c424e5` (0.6.5) |

The chat server is a separate pin: the 1.1.4 release archive, `tools/chat/qwen38-27b/pins.json`. Drama 3's engine
is a third (`tools/speech/drama3/pins*.json`): the deployed NVFP4 engine is plotarmordev's `drama-nvfp4` branch on
TensorFold `c7137bb3`, the only engine that loads those weights, and it does not follow this pin.

## What is consumed

`build.zig` creates three modules from the dependency and nothing else:

- `cuda`: `zig/src/cuda/root.zig` with `kernel_options.with_kernels = false` (no TensorFold fatbins are embedded;
  every engine ships its own), `stagger` from `zig/src/core/stagger.zig`, and since 1.1.x `sys`
  (`zig/src/platform/sys.zig`) and `gpu_memory` (`zig/src/gpu_memory.zig`), wired as upstream's
  `zig/build/cuda_runtime.zig` does.
- `tokenizer`: `zig/src/core/tokenizer/tokenizer.zig` (Hugging Face `tokenizer.json`).

TensorFold's checkpoint/safetensors, quantization, lane and model code is not built. LocalRouter's safetensors and
pack loaders are its own. The NVFP4 kernels in `kernels/cuda/nvfp4*` are local copies and do not follow the pin.

## Source delta from 1.0.7 that reaches the build

Byte-identical: `core/stagger.zig`, `cuda/triton.zig`, `cuda/segments.zig`. The new
files in `zig/src/cuda` (EXL3, TP, RoCE, training, arena, pack, ...) are not referenced by LocalRouter, so not compiled
in, except through the calls below.

| File | Change | Effect on LocalRouter |
|---|---|---|
| `cuda/driver.zig` | `std.DynLib` → `sys.DynLib`; `Error` gains `CudaContextLost`, `GraphNotPermitted`, copy-source errors; failed calls publish sticky faults (`context_health.zig`) and log `knownName`; optional `cuFuncGetName` | Same driver calls. A sticky execution fault (700, 719, ...) is now `error.CudaContextLost` instead of `CudaFailed`: engines treat both as a failed job. |
| `cuda/abi.zig` | six more table entries: `cuDevicePrimaryCtxSetFlags_v2`, `cuCtxGetDevice`, `cuMemcpy2DAsync_v2`, `cuStreamCreateWithPriority`, `cuCtxGetStreamPriorityRange`, `cuEventRecordWithFlags` | `Driver.open` now requires them; all are in CUDA 12's libcuda. None allocates: `cuMemAlloc_v2`, `cuMemHostAlloc` and `cuMemHostRegister_v2` are still the table's only allocating entries, and memcap patches all three. |
| `cuda/context.zig` | `init` registers the context with `context_health` and `copy_source`; `deinit` drops copy-source spares | Bookkeeping only (a fixed table of 64 contexts a process; a worker holds one). |
| `cuda/stream.zig`, `cuda/graph.zig` | synchronize/query/destroy go through `copy_source` (which frees a copy's private source once its stream is proven done); `beginCapture` opens a capture scope, `endCapture` closes it **only on success**; `instantiate` maps 800 to `GraphNotPermitted` | Our engines close the scope themselves when `endCapture` fails (`qwen_image/dit.zig`, `minimax_h3/dit.zig`): left open, the next capture on that stream and every `DeviceBuffer.upload`/`fromHost` in the worker is refused (`UnownedCaptureSource`). |
| `cuda/memory.zig` | `DeviceBuffer.upload`/`fromHost` copy through `copy_source.synchronousCopy` (a private host copy, 64 MiB chunks, then a sync); async uploads through `enqueue` (a private pinned copy); optional allocation limit and `setHook` allocator; `Usage` gains `host_peak`, `allocs` | Engines copy with their own `cuMemcpyHtoD*` calls and are unaffected; H3's two `fromHost` inputs (an image-to-video frame and the audio tags) take one more host copy of the same bytes. No limit and no hook is installed, so allocation is the same `cuMemAlloc_v2` (the memcap coverage test fails if LocalRouter ever installs one). |
| `cuda/arena.zig`, `cuda/pack.zig` | a VMM sub-allocator (`cuMemCreate`, found with `lookup`) that serves `DeviceBuffer.alloc` once installed as the hook | Never installed by LocalRouter (checked by the coverage test), so never reached. |
| `cuda/launch.zig`, `cuda/module.zig` | a failed launch names its kernel and grid; `Module.holds`; `rowThreads` | Log text only on failure. |
| `core/tokenizer/*` | segment memo cache, `encodeMasked`/`encodePlain` (request spans that may not form added tokens), EOS resolver, possessive quantifiers in pre-tokenizer patterns, NFKD table | `encode`, which our text encoders call, keeps its algorithm (added tokens split first, then each segment); the parity tests in `src/engines/*/tokenizer*` pass unchanged. |

## Contract for engines

An engine's pipeline opens one `cuda.Driver`, one `cuda.Context` (device 0) and one blocking `cuda.Stream`
(`Stream.init(&d, false)`, ordered with synchronous copies) in `load`, as `src/engines/minimax_h3/pipeline.zig`
does. Its components (text encoder, DiT, decoders) borrow pointers to these and never create their own. Device memory is `cuda.DeviceBuffer` (`alloc`/`free`, 256-byte aligned, zero bytes allocate nothing) or
an engine's slab carved from it; tensors are passed as device addresses (`u64`) with dtype, shape and strides
described by the engine. A view into another buffer is a plain address and must never be freed; when wrapping one in
a `DeviceBuffer`, set `borrowed = true`. Pinned host staging is `cuda.HostBuffer`. Kernels launch on the given
stream; graph capture uses `cuda.graph.beginCapture(stream, .thread_local)` and only stream-ordered calls inside it.

`cuda.usage(false)` reports bytes held through `DeviceBuffer`/`HostBuffer` in this process; it does not see raw
`cuMemAlloc` calls, driver/context reservations or cuBLASLt workspaces, so it is a lower bound, not the resident
figure a worker reports on load.

Unload releases everything: every engine-owned buffer, module, graph exec and stream, then the context, then the
driver. The worker process exits after unload, which is the final release boundary.

## Verification

Host: `zig build`, `zig build test`, `zig fmt --check` on every touched file (Zig 0.17.0), compared with the
previous pin. GPU: shared engine regressions (Qwen-Image and H3 bits, load/unload) must be rerun on the target GPU
before the pin ships in a release; matching signatures do not prove unchanged output.

1.1.4 on GB10 (DGX-02, 2026-10-25): the deployed image and video tools (Qwen-Image 2.1 Turbo FP8, MiniMax H3 NVFP4)
served by `localrouter` built at main `bd233f9` (1.0.7) and at the 1.1.4 pin, one daemon at a time in the order old,
new, new, old, with the same requests and seeds: text to image 1024x1024 (cold, then warm), an edit of one fixed input,
text to video and image to video at 768x448, 2 s. Every output is the same in all four runs: PNG pixels, MP4 video
frames and audio samples, compared decoded (sha256). Times are within run-to-run noise (cold image 11.5–12.8 s,
warm 6.6–6.7 s, edit 11.2 s, text to video 30.6–32.7 s, image to video 32.6–33.1 s, on both), and so is each job's
peak footprint (41.1 GiB image, 94.3 GiB with the video model loaded beside it).

