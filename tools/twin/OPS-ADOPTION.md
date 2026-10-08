# Adopting stk_twin.ops (after `python -m stk_twin.test_ops` passes on the GPU)

`from . import ops` in dit.py and qwen_image.py. Each row replaces one torch expression; the recorder blocks stay as they are.

| where | today | becomes |
|---|---|---|
| dit.py `_dense` (l.176) | `y = F.linear(x, w)` | `y = ops.dense_bf16(x, w)` |
| dit.py `_pointwise` (l.182) | `{"silu": F.silu, "tanh": torch.tanh, "gelu_tanh": ...}[fn](x)` | `y = ops.POINTWISE[fn](x)` |
| dit.py `_residual` (l.237) | `x.addcmul_(a, g)` | `ops.gated_residual_(x, a, g)` (g may be the expanded zero-mod view; it is made contiguous) |
| dit.py `_time` (l.244-248) | `tr = ...; emb = torch.cat([cos, sin])...` | `emb = ops.time_sinusoid(t)` ([B+1, 256] bf16) |
| dit.py `_text` (l.263-264) | `c = context.float(); y = (c * rsqrt(...) * w).to(bf16)` | `y = ops.rms_norm_f32(context, self.txt_norm, self.cfg.eps)` |
| dit.py `_block` prefix V (l.282) | `vv = qkv[:, :, 2].contiguous()` | `vv = torch.empty_like(q); ops.copy_rows(vv, qkv[:, :, 2])` |
| dit.py `_block` step V (l.290) | `vv[:, P:].copy_(qkv[:, :, 2])` | `ops.copy_rows(vv[:, P:], qkv[:, :, 2])` |
| dit.py `__call__` (l.336-337) | `kbuf[:, :P].copy_(pk); vbuf[:, :P].copy_(pv)` | `ops.copy_rows(kbuf[:, :P], pk); ops.copy_rows(vbuf[:, :P], pv)` |
| qwen_image.py `sample` (l.101) | `x = (x.float() + (sig[i+1] - sig[i]) * v.float()).to(bf16)` | `x = ops.euler(x, v, sig[i+1] - sig[i])` (out of place, so the recorded `x` input stays intact) |
| tfimage `Fp8Linear.__call__` (dynamic act scale only) | `a = (x.abs().amax().float() / 448).clamp_min(1e-12)`; `xq = (x.float() / a).clamp(...).to(e4m3)`; pad to 16 rows | `stat = ops.absmax_stat(x); xq, a = ops.quant_e4m3(x, stat, 16)`; then `torch._scaled_mm(xq, w8.t(), scale_a=a, scale_b=ws, out_dtype=bf16)` and `y[:m]` |

Notes
- The FP8 wrapper needs a small subclass of `tfimage.linear.Fp8Linear` (or a patch in `L.make`) overriding `__call__`. With a calibrated static `act` the scale is a constant, not derived from an absmax, so that path stays as it is (the quant kernel always derives `a` from `stat[0]`).
- Not bit-equal to torch by construction: `dense_bf16` (own fp32 accumulation order vs cuBLAS) and, in the last bit of the sum order, `rms_norm_f32`. Once adopted the twin's reference for these ops is ours, and the Zig side runs the identical kernels.
- `ops.py` JIT-builds on first use (cache under `$TORCH_EXTENSIONS_DIR`, keyed by a hash of ops.cu); kernel source is found via `$STK_KROOT/cuda/qwen_image/ops.cu`.
- Still torch/Triton and not covered here: adaln, rms_rope, swiglu, attention, the NVFP4 path, `torch.cat`/`chunk`/`transpose` reshapes, rope tables.
