"""FP8 on TensorFold's own prompt GEMM (mode A8): e4m3 weights in its fragment order, rows quantized under a static,
calibrated input scale. The same kernel family as the NVFP4 path (`gemm_ws`), so the Zig engine copies one set of
kernels, and no cuBLASLt call remains in LocalRouter."""

from __future__ import annotations

import torch

TILE = 12  # the prompt tile tfimage uses for NVFP4 too (the bulk-copy GEMM on sm_90+)


class TfFp8Linear:
    kind = "fp8"

    def __init__(self, w8: torch.Tensor, scale: float, act: float):
        from tensorfold.cuda.nvfp4.linear import Fp8Linear

        self.lin = Fp8Linear.from_checkpoint(w8.view(torch.uint8).view(torch.float8_e4m3fn) if w8.dtype != torch.float8_e4m3fn else w8,
                                             float(scale), act=float(act))
        self.n, self.k, self.act, self.tile = self.lin.n, self.lin.k, float(act), TILE

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        from tensorfold.cuda.nvfp4 import checkpoint

        return checkpoint.prompt(checkpoint.A8, x, self.lin, tile=self.tile)

    def nbytes(self) -> int:
        return self.lin.w8.numel()
