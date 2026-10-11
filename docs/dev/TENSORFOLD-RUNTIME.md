# TensorFold runtime pin

LocalRouter builds TensorFold's CUDA runtime from source, host side only. This page records the pinned revision,
what LocalRouter consumes from it and the contract engines may rely on.

## Pin

| | |
|---|---|
| Release | TensorFold v1.0.7 (annotated tag object `dee65c75`, peeled commit below) |
| Commit | `e7c523ebfa16b8094cf9e35e3773a7d25befcc5b` |
| Zig package hash | `tensorfold-1.0.7-CphTLULywQAUe2LxgT29-AserW8gMmK9C-ltOxIaw1pG` (computed by `zig fetch` from the commit) |
| Previous | `88c424e5350a702372771915507bcbed9d513546`, `tensorfold-0.6.5-CphTLZQzRwCMYupcJ_rKtDbwn1X18_Mk3s6-IR_adVMM` |

## What is consumed

`build.zig` creates three modules from the dependency and nothing else:

- `cuda`: `zig/src/cuda/root.zig` with `kernel_options.with_kernels = false` (no TensorFold fatbins are embedded;
  every engine ships its own) and `stagger` from `zig/src/core/stagger.zig`.
- `tokenizer`: `zig/src/core/tokenizer/tokenizer.zig` (Hugging Face `tokenizer.json`).

TensorFold's checkpoint/safetensors, quantization, lane and model code is not built. LocalRouter's safetensors and
pack loaders are its own. The NVFP4 kernels in `kernels/cuda/nvfp4*` are local copies and do not follow the pin.

## Source delta from 0.6.5 that reaches the build

Byte-identical: `cuda/{context,graph,launch,module,nccl,segments,stream,triton}.zig`, `core/stagger.zig`.

| File | Change | Effect on LocalRouter |
|---|---|---|
| `cuda/memory.zig` | process-wide device/host byte counters and device peak (`cuda.usage(reset_peak)`); `DeviceBuffer.borrowed` (default `false`) | Allocation and free are unchanged calls; counters are atomics beside them. Free of a non-borrowed buffer is unchanged. |
| `cuda/memory.zig` | `DeviceBuffer.upload` now calls `cuStreamSynchronize(null)` after `cuMemcpyHtoD` | Used only by development replay/bench commands (`src/cli/*`); engines copy with their own `cuMemcpyHtoD_v2` / `cuMemcpyHtoDAsync_v2` and are unaffected. No `upload` is called inside a graph capture. |
| `cuda/abi.zig`, `cuda/driver.zig` | `cuMemHostRegister_v2`, `cuMemHostUnregister` added to the resolved table; `Driver.capability(ordinal)` | `Driver.open` now also requires these two symbols, present in every libcuda since CUDA 4. |
| `cuda/cublaslt.zig` | `@intFromEnum` → `@backingInt` for the transpose attributes | Same values. LocalRouter does not use cuBLASLt. |
| `cuda/kernels.zig`, `cuda/aot.zig` | more fatbin names; captured-kernel folder pick | Empty slices with `with_kernels = false`; unused. |
| `cuda/root.zig` | exports `usage`, `Usage`, `carveout`/`Carveout`, `qmmf`, `fp8`, `nvfp4`, `grouped`, `qlinear`, `experts`, `modelopt` | All self-contained in `zig/src/cuda`; not referenced, so not compiled in. `Carveout` stays unused. |
| `core/tokenizer/{tokenizer,steps}.zig` | `Tokenizer.tokenBytes`, `Decoder.tokenLocal`, `Context.raw_bytes` (default `false`) | Decode only; the encode path, which engines use, is unchanged. Default decode output is unchanged. |

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

Host: `zig build`, `zig build test`, `zig fmt --check build.zig build src tests` (Zig 0.17.0), compared with the
previous pin. GPU: shared engine regressions (Qwen-Image and H3 bits, load/unload) must be rerun on the target GPU
before the pin ships in a release; matching signatures do not prove unchanged output.
