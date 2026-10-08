"""MiniMax H3's video VAE decoder (ComfyUI 0.37.0 `comfy/ldm/minimax/vae.py`, `MiniMaxH3VideoVAE.decode`, the fp16
checkpoint `minimax_h3_video_vae_fp16.safetensors`) op for op on LocalRouter's kernels (kernels/cuda/minimax/gemm_f16.cu
and vae_video.cu), so the Zig engine, which launches the same kernels, is bit-exact with the twin. Text to video, batch 1.

What it does, as ComfyUI does it (tools/twin/VVAE-PORT.md has the arithmetic, the tiling tables and the naming):
  decode(z [1, 24, T, H, W])       z = z * std + mean in fp16 (two roundings), then `decode_temporal`: clips of 7 latent
                                   frames every 5 (the last pads by repeating the final latent frame), each clip spatially
                                   tiled in 256 px tiles (overlap >= 64 px) that the ViT3D decoder decodes ONE at a time
                                   (ComfyUI batches up to 4; the per-tile math is the same), blended in fp16 into a clip
                                   canvas, the clip's two 20-frame halves blended in time (5 frames), finalised
                                   (fp32 * pixel_std + pixel_mean, clamp 0..1) and converted to uint8 ((x * 255) truncated).
  one tile                         post_quant_conv (1x1x1, a GEMM) -> x_embedder -> + 4 register tokens + 1 zero token ->
                                   36 blocks (RMSNorm, qkv GEMM, per-head RMSNorm + split-half RoPE on 48 of 64 dims,
                                   attention as S = q k^T / softmax / P v, to_out GEMM + residual addcmul, RMSNorm, w1,
                                   SwiGLU, w2 + addcmul) -> LayerNorm -> proj_out -> unshuffle 16 x 16 x 4 patches.
Every GPU op runs inside `REC.op` under a stable name (the scheme is in VVAE-PORT.md, "Recorder names").

Speed knobs (none changes a bit of the output; test_vae_video.py proves each against the plain path):
  STK_VVAE_REF=1     every GEMM on the reference kernel (stk_gemm_f16_ref, the first version) and the whole attention in one
                     piece: the old path, for the comparison and the proofs
  STK_VVAE_HEADS=n   heads an attention group runs (default 2; 32 or 0: all heads at once). A tile that is not recorded
                     runs q k^T, softmax and P v group by group, so the scores of a group (6.5 MB a head) stay in the L2
                     instead of making four trips (write, read, write, read) of 207 MB through the LPDDR5X
  STK_VVAE_PROF=1    CUDA events around every op: the GPU milliseconds of a decode by class go to stderr
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path

import torch

from ..rec import REC

KDIR = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[4] / "kernels")) / "cuda" / "minimax"

# ---------------------------------------------------------------------------------------------------- constants
LAYERS, HEADS, HD, DIM, FFN = 36, 32, 64, 2048, 8192
ZC, OUT_C = 24, 3
RATIO, RATIO_T = 16, 4                      # spatial / temporal patch of the decoder
EPS = 1e-5
CLIP_LENGTH, TOKEN_DROP = 17, 3
FRAME_PRE_PADDING = (-CLIP_LENGTH) % RATIO_T                      # 3
TOKENS_CHUNK = math.ceil(CLIP_LENGTH / RATIO_T)                    # 5
TOKEN_OVERLAP = (-TOKEN_DROP) % TOKENS_CHUNK                       # 2
FRAME_OVERLAP = max(TOKEN_OVERLAP * RATIO_T - FRAME_PRE_PADDING, 0)  # 5
TILE_SIZE, TILE_OVERLAP_MIN = 256, 64
NSUF = 5                                    # 4 register tokens + 1 zero token
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------------------------------- the kernels
_DECLS = """
void k_gemm(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias, at::Tensor c, c10::optional<at::Tensor> res,
            c10::optional<at::Tensor> rscale, int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc,
            int64_t a_off, int64_t b_off, int64_t c_off, int64_t sA, int64_t sB, int64_t sC, int64_t batch, int64_t ldr,
            int64_t sR);
void k_gemm_v(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias, at::Tensor c, c10::optional<at::Tensor> res,
              c10::optional<at::Tensor> rscale, int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc,
              int64_t a_off, int64_t b_off, int64_t c_off, int64_t sA, int64_t sB, int64_t sC, int64_t batch, int64_t ldr,
              int64_t sR, int64_t variant);
at::Tensor k_denorm(at::Tensor z, at::Tensor stdv, at::Tensor mean);
at::Tensor k_gather_rows(at::Tensor z, int64_t t0, int64_t Tn, int64_t y0, int64_t h, int64_t x0, int64_t w);
at::Tensor k_rope_table(at::Tensor inv_freq, int64_t T, int64_t H, int64_t W, int64_t nsuf);
void k_suffix_(at::Tensor h, at::Tensor reg, int64_t nP);
at::Tensor k_rms_norm(at::Tensor x, at::Tensor w, double eps);
at::Tensor k_layer_norm(at::Tensor x, at::Tensor w, at::Tensor b, double eps);
void k_rms_rope_(at::Tensor qkv, at::Tensor table, int64_t S, int64_t H, double eps, int64_t variant);
at::Tensor k_vt(at::Tensor qkv, int64_t S, int64_t SP, int64_t H);
void k_softmax_(at::Tensor s, int64_t S, int64_t SP, double scale);
void k_nan_to_num_(at::Tensor x);
at::Tensor k_swiglu(at::Tensor gu);
at::Tensor k_unshuffle(at::Tensor rows, int64_t T, int64_t H, int64_t W);
void k_place_tile(at::Tensor b, c10::optional<at::Tensor> ytail, int64_t ey, c10::optional<at::Tensor> ltail, int64_t ex,
                  at::Tensor canvas, int64_t oy, int64_t ox, int64_t oh, int64_t ow);
void k_finalize(c10::optional<at::Tensor> a, int64_t a0, at::Tensor b, int64_t b0, int64_t ext, int64_t copy,
                std::vector<double> sm, c10::optional<at::Tensor> out_u8, c10::optional<at::Tensor> out_f32, int64_t pos);
"""

_LAUNCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include "gemm_f16.cu"
#include "vae_video.cu"
#define HP(t) reinterpret_cast<__half*>((t).data_ptr())
#define CHP(t) reinterpret_cast<const __half*>((t).data_ptr())
#define STREAM at::cuda::getCurrentCUDAStream()
static inline unsigned cdiv(long long a, long long b) { return (unsigned)((a + b - 1) / b); }
static void chk(const at::Tensor& t, at::ScalarType s, const char* n) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == s && t.is_contiguous(), n, ": contiguous CUDA tensor of the right dtype");
}
static void chkh(const at::Tensor& t, const char* n) { chk(t, at::kHalf, n); }
static void chk_launch(const char* what) {
    cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e));
}

// The GEMM's kernels (gemm_f16.cu): 0 = stk_gemm_f16_ref (the first version: grid (n, m) of 128 x 128), 1 = stk_gemm_f16
// (256 x 128, the linears), 2 = _s (128 x 128, K <= 64), 3 = _n (64 x 64, N <= 64); grid (m, n, batch) for 1 .. 3. All four
// are bit-equal (test_vae_video.py); the default is by shape, the reference when STK_VVAE_REF is set. The Zig engine's
// Ops.gemm picks the same way.
static int gemm_pick(int64_t M, int64_t N, int64_t K) {
    static const char* e = std::getenv("STK_VVAE_REF");
    if (e && e[0] && !(e[0] == '0' && !e[1])) return 0;
    return K <= 64 ? 2 : (N <= 64 ? 3 : 1);
}
// One of the template tiles: dynamic shared memory opt-in (above 48 KiB), grid (m, n, batch) of Cf::THREADS threads.
#define GEMM_TILE(KERN, CF)                                                                                                     \
    do {                                                                                                                        \
        cudaFuncSetAttribute(KERN, cudaFuncAttributeMaxDynamicSharedMemorySize, stk_gemm16::CF::SMEM);                          \
        KERN<<<dim3(cdiv(M, stk_gemm16::CF::BM), cdiv(N, stk_gemm16::CF::BN), (unsigned)batch), stk_gemm16::CF::THREADS,       \
               stk_gemm16::CF::SMEM, STREAM>>>(ap, bq, bp, cp, (int)M, (int)N, (int)K, lda, ldb, ldc, sA, sB, sC, rp, sp, ldr, sR); \
    } while (0)
void k_gemm_v(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias, at::Tensor c, c10::optional<at::Tensor> res,
              c10::optional<at::Tensor> rscale, int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc,
              int64_t a_off, int64_t b_off, int64_t c_off, int64_t sA, int64_t sB, int64_t sC, int64_t batch, int64_t ldr,
              int64_t sR, int64_t variant) {
    chkh(a, "a"); chkh(b, "b"); chkh(c, "c");
    TORCH_CHECK(K % 8 == 0 && lda % 8 == 0 && ldb % 8 == 0 && a_off % 8 == 0 && b_off % 8 == 0 && ldc % 2 == 0 &&
                sA % 8 == 0 && sB % 8 == 0, "gemm_f16 alignment: K, lda, ldb, offsets, strides % 8, ldc even");
    const __half* bp = nullptr; const __half* rp = nullptr; const __half* sp = nullptr;
    if (bias.has_value()) { chkh(*bias, "bias"); bp = CHP(*bias); }
    if (res.has_value()) { chkh(*res, "res"); chkh(*rscale, "rscale"); rp = CHP(*res); sp = CHP(*rscale); }
    c10::cuda::CUDAGuard g(a.device());
    const __half* ap = CHP(a) + a_off; const __half* bq = CHP(b) + b_off; __half* cp = HP(c) + c_off;
    if (variant < 0) variant = gemm_pick(M, N, K);
    if (variant == 0)
        stk_gemm_f16_ref<<<dim3(cdiv(N, 128), cdiv(M, 128), (unsigned)batch), 256, 0, STREAM>>>(
            ap, bq, bp, cp, (int)M, (int)N, (int)K, lda, ldb, ldc, sA, sB, sC, rp, sp, ldr, sR);
    else if (variant == 1) GEMM_TILE(stk_gemm_f16, CfgWide);
    else if (variant == 2) GEMM_TILE(stk_gemm_f16_s, CfgStd);
    else if (variant == 3) GEMM_TILE(stk_gemm_f16_n, CfgNarrow);
    else TORCH_CHECK(false, "gemm variant: 0 .. 3");
    chk_launch("stk_gemm_f16");
}
void k_gemm(at::Tensor a, at::Tensor b, c10::optional<at::Tensor> bias, at::Tensor c, c10::optional<at::Tensor> res,
            c10::optional<at::Tensor> rscale, int64_t M, int64_t N, int64_t K, int64_t lda, int64_t ldb, int64_t ldc,
            int64_t a_off, int64_t b_off, int64_t c_off, int64_t sA, int64_t sB, int64_t sC, int64_t batch, int64_t ldr,
            int64_t sR) {
    k_gemm_v(a, b, bias, c, res, rscale, M, N, K, lda, ldb, ldc, a_off, b_off, c_off, sA, sB, sC, batch, ldr, sR, -1);
}
at::Tensor k_denorm(at::Tensor z, at::Tensor stdv, at::Tensor mean) {
    chkh(z, "z"); chkh(stdv, "std"); chkh(mean, "mean");
    auto y = at::empty_like(z);
    long long C = z.size(0), n = z.numel();
    vv_denorm<<<cdiv(n, 256), 256, 0, STREAM>>>(CHP(z), CHP(stdv), CHP(mean), HP(y), n / C, n);
    return y;
}
at::Tensor k_gather_rows(at::Tensor z, int64_t t0, int64_t Tn, int64_t y0, int64_t h, int64_t x0, int64_t w) {
    chkh(z, "z");
    TORCH_CHECK(z.dim() == 4, "z: [C, T, H, W]");
    auto rows = at::empty({Tn * h * w, z.size(0)}, z.options());
    vv_gather_rows<<<cdiv(rows.numel(), 256), 256, 0, STREAM>>>(CHP(z), HP(rows), (int)z.size(0), (int)z.size(1), (int)z.size(2),
                                                                (int)z.size(3), (int)t0, (int)Tn, (int)y0, (int)h, (int)x0, (int)w);
    return rows;
}
at::Tensor k_rope_table(at::Tensor inv_freq, int64_t T, int64_t H, int64_t W, int64_t nsuf) {
    chkh(inv_freq, "inv_freq");
    TORCH_CHECK(inv_freq.numel() == 8, "inv_freq: 8 values");
    long long S = T * H * W + nsuf;
    auto t = at::empty({S, 24, 4}, inv_freq.options());
    vv_rope_table<<<cdiv(S * 24, 256), 256, 0, STREAM>>>(CHP(inv_freq), HP(t), (int)T, (int)H, (int)W, (int)nsuf);
    return t;
}
void k_suffix_(at::Tensor h, at::Tensor reg, int64_t nP) {
    chkh(h, "h"); chkh(reg, "reg");
    long long D = h.size(1);
    vv_suffix<<<cdiv(5 * D, 256), 256, 0, STREAM>>>(HP(h), CHP(reg), nP, D);
}
at::Tensor k_rms_norm(at::Tensor x, at::Tensor w, double eps) {
    chkh(x, "x"); chkh(w, "w");
    long long D = w.numel();
    auto y = at::empty_like(x);
    vv_rms_norm<<<(unsigned)(x.numel() / D), 256, 0, STREAM>>>(CHP(x), CHP(w), HP(y), D, (float)eps);
    return y;
}
at::Tensor k_layer_norm(at::Tensor x, at::Tensor w, at::Tensor b, double eps) {
    chkh(x, "x"); chkh(w, "w"); chkh(b, "b");
    long long D = w.numel();
    auto y = at::empty_like(x);
    vv_layer_norm<<<(unsigned)(x.numel() / D), 256, 0, STREAM>>>(CHP(x), CHP(w), CHP(b), HP(y), D, (float)eps);
    return y;
}
void k_rms_rope_(at::Tensor qkv, at::Tensor table, int64_t S, int64_t H, double eps, int64_t variant) {
    chkh(qkv, "qkv"); chkh(table, "table");
    TORCH_CHECK(qkv.dim() == 2 && qkv.size(0) >= S && qkv.size(1) == H * 192, "qkv: [S, H * 192]");
    TORCH_CHECK(table.numel() >= S * 24 * 4, "table: [S, 24, 4]");
    unsigned grid = cdiv(S * H, 4);
    long long rs = qkv.size(1);
    if (variant == 0) vv_rms_rope<<<grid, 128, 0, STREAM>>>(HP(qkv), CHP(table), S, (int)H, rs, 192, (float)eps);
    else if (variant == 1) vv_rms_rope_v1<<<grid, 128, 0, STREAM>>>(HP(qkv), CHP(table), S, (int)H, rs, 192, (float)eps);
    else vv_rms_rope_v2<<<grid, 128, 0, STREAM>>>(HP(qkv), CHP(table), S, (int)H, rs, 192, (float)eps);
}
at::Tensor k_vt(at::Tensor qkv, int64_t S, int64_t SP, int64_t H) {
    chkh(qkv, "qkv");
    auto vt = at::empty({H, 64, SP}, qkv.options());
    vv_vt<<<cdiv(vt.numel(), 256), 256, 0, STREAM>>>(CHP(qkv), HP(vt), S, SP, (int)H, qkv.size(1), 192);
    return vt;
}
void k_softmax_(at::Tensor s, int64_t S, int64_t SP, double scale) {
    chkh(s, "s");
    TORCH_CHECK(s.numel() % SP == 0 && SP >= S, "softmax shape");
    vv_softmax<<<(unsigned)(s.numel() / SP), 256, 0, STREAM>>>(HP(s), S, SP, (float)scale);
}
void k_nan_to_num_(at::Tensor x) {
    chkh(x, "x");
    vv_nan_to_num<<<cdiv(x.numel(), 256), 256, 0, STREAM>>>(HP(x), x.numel());
}
at::Tensor k_swiglu(at::Tensor gu) {
    chkh(gu, "gu");
    long long F = gu.size(-1) / 2, M = gu.numel() / (2 * F);
    auto out = at::empty({M, F}, gu.options());
    vv_swiglu<<<dim3(cdiv(F, 256), (unsigned)M), 256, 0, STREAM>>>(CHP(gu), HP(out), F);
    return out;
}
at::Tensor k_unshuffle(at::Tensor rows, int64_t T, int64_t H, int64_t W) {
    chkh(rows, "rows");
    TORCH_CHECK(rows.dim() == 2 && rows.size(1) == 3072 && rows.size(0) >= T * H * W, "rows: [>= T*H*W, 3072]");
    auto out = at::empty({3, 4 * T, 16 * H, 16 * W}, rows.options());
    vv_unshuffle<<<cdiv(out.numel(), 256), 256, 0, STREAM>>>(CHP(rows), HP(out), (int)T, (int)H, (int)W);
    return out;
}
void k_place_tile(at::Tensor b, c10::optional<at::Tensor> ytail, int64_t ey, c10::optional<at::Tensor> ltail, int64_t ex,
                  at::Tensor canvas, int64_t oy, int64_t ox, int64_t oh, int64_t ow) {
    chkh(b, "b"); chkh(canvas, "canvas");
    TORCH_CHECK(b.dim() == 4 && canvas.dim() == 4 && b.size(1) == canvas.size(1), "tile / canvas: [3, F, h, w]");
    const __half* yp = nullptr; const __half* lp = nullptr;
    int tha = 0, twa = 0, thl = 0, twl = 0;
    if (ytail.has_value()) { chkh(*ytail, "ytail"); yp = CHP(*ytail); tha = (int)ytail->size(2); twa = (int)ytail->size(3); }
    if (ltail.has_value()) { chkh(*ltail, "ltail"); lp = CHP(*ltail); thl = (int)ltail->size(2); twl = (int)ltail->size(3); }
    c10::cuda::CUDAGuard g(b.device());
    long long n = 3LL * b.size(1) * oh * ow;
    vv_place_tile<<<cdiv(n, 256), 256, 0, STREAM>>>(CHP(b), (int)b.size(1), (int)b.size(2), (int)b.size(3), yp, tha, twa, (int)ey, lp,
                                                    thl, twl, (int)ex, HP(canvas), (int)canvas.size(2), (int)canvas.size(3),
                                                    (int)oy, (int)ox, (int)oh, (int)ow);
}
void k_finalize(c10::optional<at::Tensor> a, int64_t a0, at::Tensor b, int64_t b0, int64_t ext, int64_t copy,
                std::vector<double> sm, c10::optional<at::Tensor> out_u8, c10::optional<at::Tensor> out_f32, int64_t pos) {
    chkh(b, "b");
    TORCH_CHECK(sm.size() == 6, "sm: std[3], mean[3]");
    const __half* ap = nullptr; int Fa = 0;
    if (a.has_value()) { chkh(*a, "a"); ap = CHP(*a); Fa = (int)a->size(1); }
    uint8_t* up = nullptr; float* fp = nullptr;
    if (out_u8.has_value()) { chk(*out_u8, at::kByte, "out_u8"); up = out_u8->data_ptr<uint8_t>(); }
    if (out_f32.has_value()) { chk(*out_f32, at::kFloat, "out_f32"); fp = out_f32->data_ptr<float>(); }
    c10::cuda::CUDAGuard g(b.device());
    int H = (int)b.size(2), W = (int)b.size(3);
    vv_finalize<<<cdiv((long long)copy * H * W * 3, 256), 256, 0, STREAM>>>(ap, Fa, (int)a0, CHP(b), (int)b.size(1), (int)b0, (int)ext,
                                                                         (int)copy, H, W, (float)sm[0], (float)sm[1], (float)sm[2],
                                                                         (float)sm[3], (float)sm[4], (float)sm[5], up, fp, (int)pos);
}
"""

NAMES = ["k_gemm", "k_gemm_v", "k_denorm", "k_gather_rows", "k_rope_table", "k_suffix_", "k_rms_norm", "k_layer_norm", "k_rms_rope_",
         "k_vt", "k_softmax_", "k_nan_to_num_", "k_swiglu", "k_unshuffle", "k_place_tile", "k_finalize"]


@lru_cache(maxsize=1)
def mod():
    """The extension: gemm_f16.cu and vae_video.cu in one translation unit (no fast-math: the engine's fatbin does not
    use it either; expf, rsqrtf, cosf / sinf are the CUDA math library's)."""
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}" + ("a" if major >= 9 else "")
    flags = ["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    src = (KDIR / "gemm_f16.cu").read_bytes() + (KDIR / "vae_video.cu").read_bytes()
    digest = hashlib.sha256(src + _LAUNCH.encode() + _DECLS.encode()).hexdigest()
    return load_inline(f"stk_h3_vvae_{digest[:12]}", cpp_sources=_DECLS, cuda_sources=f"// {digest}\n" + _LAUNCH,
                       functions=NAMES, extra_cuda_cflags=flags, extra_include_paths=[str(KDIR)], with_cuda=True)


# ---------------------------------------------------------------------------------------------------- the plan (host)
def split_tiles(input_len: int, tile_size: int = TILE_SIZE, overlap_min: int = TILE_OVERLAP_MIN, ratio: int = RATIO):
    """`MiniMaxH3VideoVAE.split_tiles`: (starts, lengths, overlaps) in pixels."""
    if tile_size >= input_len:
        return [0], [input_len], []
    n = math.ceil(input_len / tile_size)
    while True:
        overlaps = [overlap_min] * (n - 1)
        remaining = tile_size * n - sum(overlaps) - input_len
        if remaining < 0:
            n += 1
        else:
            break
    for i in range(remaining // ratio):
        overlaps[i % (n - 1)] += ratio
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return starts, [tile_size] * n, overlaps


def temporal_chunks(z_len: int) -> tuple[int, int]:
    """`_decode_temporal_chunks`: (pad_tokens, num_chunks)."""
    pseudo = z_len + TOKEN_DROP
    pad_tokens = (-pseudo) % TOKENS_CHUNK
    pseudo += pad_tokens
    num_chunks = pseudo // TOKENS_CHUNK - int(TOKEN_DROP > 0)
    if num_chunks < 1:
        pad_tokens += TOKENS_CHUNK
        num_chunks += 1
    return pad_tokens, num_chunks


def _pad_frames(z_len: int, pad_tokens: int) -> int:
    if pad_tokens <= 0:
        return 0
    intra_tail = CLIP_LENGTH % RATIO_T
    if intra_tail == 0:
        return pad_tokens * RATIO_T
    before = z_len - pad_tokens
    return sum(intra_tail if (before + k) % TOKENS_CHUNK == 0 else RATIO_T for k in range(pad_tokens))


def _frame_plan(z_len: int, num_chunks: int, pad_tokens: int) -> int:
    chunk_dec = TOKENS_CHUNK * RATIO_T
    split_count = int(TOKEN_DROP > 0) + 1
    total = overlap_frames = 0
    for i in range(num_chunks):
        t0 = i * TOKENS_CHUNK
        t1 = t0 + TOKENS_CHUNK + TOKEN_OVERLAP
        clip_frames = max(0, min(t1, z_len) - min(t0, z_len)) * RATIO_T
        for j in range(split_count):
            f0 = j * chunk_dec
            f1 = min(f0 + chunk_dec, clip_frames)
            frames = max(0, f1 - f0 - FRAME_PRE_PADDING)
            if j == 0:
                total += frames
            else:
                overlap_frames = frames
    return total + overlap_frames - _pad_frames(z_len, pad_tokens)


def output_frames(z_len: int) -> int:
    """`decode_output_shape`'s frame count for z_len latent frames."""
    if z_len == 1:
        return 1
    pad_tokens, num_chunks = temporal_chunks(z_len)
    return _frame_plan(z_len + pad_tokens, num_chunks, pad_tokens)


def f16_const(vals) -> list[float]:
    """ImageNet constants as the decoder holds them: fp32 tensor -> module.to(fp16) -> .to(fp32)."""
    return torch.tensor(vals, dtype=torch.float32).half().float().tolist()


def rope_inv_freq() -> torch.Tensor:
    """RotaryEmbeddingND(48, 100, 3).inv_freq: 1 / 100 ** arange(0, 1, 6 / 48) in fp32 (CPU, as at construction), then the
    module's .to(fp16)."""
    return (1 / 100.0 ** torch.arange(0, 1, 2 * 3 / 48, dtype=torch.float32)).half()


@contextmanager
def _recording(on: bool):
    saved = REC.on
    REC.on = saved and on
    try:
        yield
    finally:
        REC.on = saved


# ---------------------------------------------------------------------------------------------------- the decoder
class VideoVAE:
    """The decode path of MiniMaxH3VideoVAE (fp16). `from_checkpoint(path)` loads the decoder, post_quant_conv and the
    latent statistics from the fp16 file; `decode(latents)` returns uint8 frames [F, 16 H, 16 W, 3] (CUDA)."""

    def __init__(self, w: dict | None, mean: torch.Tensor, std: torch.Tensor, device="cuda"):
        self.w = w
        self.dev = torch.device(device)
        self.mean = mean.to(self.dev, torch.float16).contiguous()
        self.std = std.to(self.dev, torch.float16).contiguous()
        self.inv_freq = rope_inv_freq().to(self.dev).contiguous()
        self.pix_std = f16_const(IMAGENET_STD)
        self.pix_mean = f16_const(IMAGENET_MEAN)
        self._scores: torch.Tensor | None = None
        ref = os.environ.get("STK_VVAE_REF", "")
        self.ref_path = bool(ref) and ref != "0"
        g = int(os.environ.get("STK_VVAE_HEADS", "2"))
        self.groups = HEADS if self.ref_path or g <= 0 or g >= HEADS else g  # heads an attention group runs
        self.prof = os.environ.get("STK_VVAE_PROF", "") not in ("", "0")
        self._ev: list = []                                                  # (class, start, end) of the decode so far

    # ------------------------------------------------------------------------------------------------ phase timing
    def _ph(self, cls: str):
        """STK_VVAE_PROF: CUDA events around an op, summed by class (gemm, attn_gemm, softmax, norm, rope, swiglu, other,
        place) when the decode ends. A no-op otherwise."""
        if not self.prof:
            return nullcontext()
        return self._timed(cls)

    @contextmanager
    def _timed(self, cls: str):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        try:
            yield
        finally:
            b.record()
            self._ev.append((cls, a, b))

    def _report(self) -> None:
        torch.cuda.synchronize()
        tot: dict[str, float] = {}
        for cls, a, b in self._ev:
            tot[cls] = tot.get(cls, 0.0) + a.elapsed_time(b)
        self._ev = []
        tot["sum"] = sum(tot.values())
        print("vvae_phase_ms " + " ".join(f"{k}={v:.1f}" for k, v in sorted(tot.items())), file=sys.stderr, flush=True)

    # ------------------------------------------------------------------------------------------------ loading
    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike, device="cuda") -> "VideoVAE":
        from safetensors import safe_open

        w: dict = {}
        with safe_open(str(path), framework="pt") as f:
            keys = set(f.keys())
            if any(k.endswith(".comfy_quant") for k in keys):
                raise ValueError("the int8 ConvRot checkpoint is not supported: use minimax_h3_video_vae_fp16.safetensors")

            def get(k, shape):
                t = f.get_tensor(k)
                if tuple(t.shape) != tuple(shape):
                    raise ValueError(f"{k}: shape {tuple(t.shape)}, expected {tuple(shape)}")
                return t.to(device, torch.float16).contiguous()

            d = "decoder."
            w["pqc.w"] = get("post_quant_conv.weight", (ZC, ZC, 1, 1, 1)).view(ZC, ZC).contiguous()
            w["pqc.b"] = get("post_quant_conv.bias", (ZC,))
            w["embed.w"], w["embed.b"] = get(d + "x_embedder.weight", (DIM, ZC)), get(d + "x_embedder.bias", (DIM,))
            w["reg"] = get(d + "register_tokens", (1, 4, DIM)).view(4, DIM).contiguous()
            w["norm_out.w"], w["norm_out.b"] = get(d + "norm_out.weight", (DIM,)), get(d + "norm_out.bias", (DIM,))
            w["proj.w"] = get(d + "proj_out.weight", (OUT_C * RATIO_T * RATIO * RATIO, DIM))
            w["proj.b"] = get(d + "proj_out.bias", (OUT_C * RATIO_T * RATIO * RATIO,))
            layers = []
            for i in range(LAYERS):
                p = f"{d}transformer_blocks.{i}."
                layers.append({
                    "norm1": get(p + "norm1.weight", (DIM,)),
                    "qkv.w": get(p + "attn.to_qkv.weight", (3 * DIM, DIM)), "qkv.b": get(p + "attn.to_qkv.bias", (3 * DIM,)),
                    "out.w": get(p + "attn.to_out.weight", (DIM, DIM)), "out.b": get(p + "attn.to_out.bias", (DIM,)),
                    "scale1": get(p + "scale1", (DIM,)),
                    "norm2": get(p + "norm2.weight", (DIM,)),
                    "w1.w": get(p + "ff.w1.weight", (2 * FFN, DIM)), "w1.b": get(p + "ff.w1.bias", (2 * FFN,)),
                    "w2.w": get(p + "ff.w2.weight", (DIM, FFN)), "w2.b": get(p + "ff.w2.bias", (DIM,)),
                    "scale2": get(p + "scale2", (DIM,)),
                })
            w["layers"] = layers
            mean, std = get("latents_mean", (ZC,)), get("latents_std", (ZC,))
        return cls(w, mean, std, device)

    @classmethod
    def bare(cls, mean: torch.Tensor, std: torch.Tensor, device="cuda") -> "VideoVAE":
        """No decoder weights: only the tiling / blending / finalising around a `tile_fn` (the structure checks)."""
        return cls(None, mean, std, device)

    # ------------------------------------------------------------------------------------------------ one tile
    def _lin(self, name, x, wt, bias, *, out=None, res=None, rscale=None):
        """x [M, K] . wt [N, K]^T + bias, rounded to half; with `res` / `rscale`: res = half(res + out * rscale)
        (ComfyUI's torch.addcmul(residual, out, scale)), written to `out` (which may be `res`)."""
        M, K = x.shape
        N = wt.shape[0]
        if out is None:
            out = torch.empty(M, N, dtype=torch.float16, device=x.device)
        ins = {"x": x} if res is None else {"x": x, "res": res}
        attrs = {"M": M, "N": N, "K": K, "epilogue": "bias" if res is None else "bias+addcmul"}
        with self._ph("gemm"), REC.op(name, "gemm_f16", attrs, **ins) as o:
            mod().k_gemm(x, wt, bias, out, res, rscale, M, N, K, K, K, out.stride(0), 0, 0, 0, 0, 0, 0, 1,
                         0 if res is None else res.stride(0), 0)
            o.out(y=out[:M])
        return out

    def _attention(self, p: str, qkv: torch.Tensor, S: int) -> torch.Tensor:
        """softmax(q k^T / 8) v per head, non-causal, from the rotated qkv buffer [S, 6144] -> rows [S, 2048]:
        S = q k^T (our fp16 GEMM, fp16 out), a fp32 row softmax (fp16 out), O = P v (our GEMM), all 32 heads batched.
        A tile that is not recorded runs the three steps `self.groups` heads at a time (`_attention_grouped`): a head's
        bits depend on that head alone, so the output is the same, and the group's scores never leave the L2."""
        m = mod()
        SP = (S + 7) // 8 * 8
        if self._scores is None or self._scores.numel() != HEADS * S * SP:
            # zeros once: the pad columns [S, SP) are written 0 by every softmax and never by the GEMM
            self._scores = torch.zeros(HEADS, S, SP, dtype=torch.float16, device=qkv.device)
        sc = self._scores
        with self._ph("other"), REC.op(f"{p}.vt", "vt", {"S": S, "SP": SP}, qkv=qkv) as o:
            vt = m.k_vt(qkv, S, SP, HEADS)
            o.out(y=vt)
        if self.groups < HEADS and not REC.on:
            att = self._attention_grouped(qkv, vt, sc, S, SP)
        else:
            with self._ph("attn_gemm"), REC.op(f"{p}.sc", "gemm_f16", {"M": S, "N": S, "K": HD, "batch": HEADS, "epilogue": "none"}, qkv=qkv) as o:
                m.k_gemm(qkv, qkv, None, sc, None, None, S, S, HD, qkv.stride(0), qkv.stride(0), SP, 0, HD, 0, 3 * HD, 3 * HD,
                         S * SP, HEADS, 0, 0)
                o.out(y=sc)
            with self._ph("softmax"), REC.op(f"{p}.sm", "softmax_rows", {"S": S, "SP": SP, "scale": 0.125}, s=sc) as o:
                m.k_softmax_(sc, S, SP, 0.125)
                o.out(y=sc)
            att = torch.empty(S, DIM, dtype=torch.float16, device=qkv.device)
            with self._ph("attn_gemm"), REC.op(f"{p}.pv", "gemm_f16", {"M": S, "N": HD, "K": SP, "batch": HEADS, "epilogue": "none"}, p=sc, vt=vt) as o:
                m.k_gemm(sc, vt, None, att, None, None, S, HD, SP, SP, SP, DIM, 0, 0, 0, S * SP, HD * SP, HD, HEADS, 0, 0)
                o.out(y=att)
        with self._ph("other"), REC.op(f"{p}.nan", "nan_to_num", {}, x=att) as o:
            m.k_nan_to_num_(att)
            o.out(y=att)
        return att

    def _attention_grouped(self, qkv: torch.Tensor, vt: torch.Tensor, sc: torch.Tensor, S: int, SP: int,
                           groups: int | None = None) -> torch.Tensor:
        """The attention's three steps for `groups` heads at a time on the first `groups` heads' worth of the score buffer
        (6.5 MB a head at 1797 tokens: two heads fit the GB10's L2, where all 32 are 207 MB through memory four times).
        Head h0 + i of a group is the same GEMM, softmax and GEMM on the same halves as in the all-heads launches: q, k as
        windows into qkv (a_off, b_off by h0 * 192), v^T by h0 * 64 * SP, the output columns by h0 * 64."""
        m = mod()
        G = groups or self.groups
        att = torch.empty(S, DIM, dtype=torch.float16, device=qkv.device)
        for h0 in range(0, HEADS, G):
            g = min(G, HEADS - h0)
            with self._ph("attn_gemm"):
                m.k_gemm(qkv, qkv, None, sc, None, None, S, S, HD, qkv.stride(0), qkv.stride(0), SP, h0 * 3 * HD, HD + h0 * 3 * HD, 0,
                         3 * HD, 3 * HD, S * SP, g, 0, 0)
            with self._ph("softmax"):
                m.k_softmax_(sc[:g], S, SP, 0.125)
            with self._ph("attn_gemm"):
                m.k_gemm(sc, vt, None, att, None, None, S, HD, SP, SP, SP, DIM, 0, h0 * HD * SP, h0 * HD, S * SP, HD * SP, HD, g, 0, 0)
        return att

    def decode_tile(self, z: torch.Tensor, t0: int, Tn: int, y0: int, h: int, w: int, x0: int, name: str) -> torch.Tensor:
        """ViT3DDecoder on post_quant_conv of one tile: z [24, Tz, Hz, Wz] (the denormalised latents), the crop
        [t0, t0 + Tn) x [y0, y0 + h) x [x0, x0 + w) (frames past Tz repeat the last) -> [3, 4 Tn, 16 h, 16 w] fp16.
        Note the argument order: (t0, Tn, y0, h, w, x0): the crop's spatial rows start at y0, columns at x0."""
        m, W = mod(), self.w
        nP = Tn * h * w
        S = nP + NSUF
        with REC.op(f"{name}.gather", "gather_rows", {"t0": t0, "Tn": Tn, "y0": y0, "h": h, "x0": x0, "w": w}, z=z) as o:
            rows = m.k_gather_rows(z, t0, Tn, y0, h, x0, w)
            o.out(y=rows)
        x = self._lin(f"{name}.pqc", rows, W["pqc.w"], W["pqc.b"])
        hs = torch.zeros(S, DIM, dtype=torch.float16, device=z.device)
        self._lin(f"{name}.embed", x, W["embed.w"], W["embed.b"], out=hs)
        with REC.op(f"{name}.suffix", "suffix", {"nP": nP}, h=hs[:nP], reg=W["reg"]) as o:
            m.k_suffix_(hs, W["reg"], nP)
            o.out(y=hs[nP:])
        with REC.op(f"{name}.rope", "rope_table", {"T": Tn, "H": h, "W": w}, inv_freq=self.inv_freq) as o:
            table = m.k_rope_table(self.inv_freq, Tn, h, w, NSUF)
            o.out(y=table)
        for i, b in enumerate(W["layers"]):
            p = f"{name}.L{i}"
            with self._ph("norm"), REC.op(f"{p}.norm1", "rms_norm_f16", {"eps": EPS}, x=hs) as o:
                n = m.k_rms_norm(hs, b["norm1"], EPS)
                o.out(y=n)
            qkv = self._lin(f"{p}.qkv", n, b["qkv.w"], b["qkv.b"])
            with self._ph("rope"), REC.op(f"{p}.rr", "rms_rope_f16", {"eps": EPS, "rot_dim": 48}, qkv=qkv, table=table) as o:
                m.k_rms_rope_(qkv, table, S, HEADS, EPS, 0)
                o.out(qkv=qkv)
            att = self._attention(p, qkv, S)
            self._lin(f"{p}.out", att, b["out.w"], b["out.b"], out=hs, res=hs, rscale=b["scale1"])
            with self._ph("norm"), REC.op(f"{p}.norm2", "rms_norm_f16", {"eps": EPS}, x=hs) as o:
                n = m.k_rms_norm(hs, b["norm2"], EPS)
                o.out(y=n)
            gu = self._lin(f"{p}.w1", n, b["w1.w"], b["w1.b"])
            with self._ph("swiglu"), REC.op(f"{p}.swi", "swiglu_f16", {}, x=gu) as o:
                a = m.k_swiglu(gu)
                o.out(y=a)
            self._lin(f"{p}.w2", a, b["w2.w"], b["w2.b"], out=hs, res=hs, rscale=b["scale2"])
        # the head runs on the patch rows only: ComfyUI normalises and projects all S rows and drops the 5 suffix rows;
        # a row's bits depend on that row alone, so the kept rows are the same
        xs = hs[:nP]
        with self._ph("norm"), REC.op(f"{name}.norm_out", "layer_norm_f16", {"eps": EPS}, x=xs) as o:
            n = m.k_layer_norm(xs, W["norm_out.w"], W["norm_out.b"], EPS)
            o.out(y=n)
        rows = self._lin(f"{name}.proj", n, W["proj.w"], W["proj.b"])
        with REC.op(f"{name}.unshuf", "unshuffle", {"T": Tn, "H": h, "W": w}, x=rows) as o:
            tile = m.k_unshuffle(rows, Tn, h, w)
            o.out(y=tile)
        return tile

    # ------------------------------------------------------------------------------------------------ a clip
    def _decode_clip(self, z, ci: int, t0: int, Tn: int, tile_fn, rec) -> torch.Tensor:
        """`tiled_decode` of the clip z[:, t0 : t0 + Tn] -> the clip canvas [3, 4 Tn, 16 Hz, 16 Wz] fp16 (every tile
        decoded alone, blended with the raw tiles above and to the left, cropped by its trailing overlaps)."""
        m = mod()
        H, W = z.shape[2] * RATIO, z.shape[3] * RATIO
        y_idx, y_len, y_ov = split_tiles(H)
        x_idx, x_len, x_ov = split_tiles(W)
        canvas = torch.zeros(OUT_C, Tn * RATIO_T, H, W, dtype=torch.float16, device=z.device)
        prev: list = [None] * len(x_idx)
        out_y = 0
        for i, (ip, il) in enumerate(zip(y_idx, y_len)):
            cur: list = []
            out_x = 0
            tile_h = 0
            for j, (jp, jl) in enumerate(zip(x_idx, x_len)):
                name = f"vvae.c{ci}.t{i}_{j}"
                on = rec(ci, i, j)
                with _recording(on):
                    tile = tile_fn(z, t0, Tn, ip // RATIO, il // RATIO, jl // RATIO, jp // RATIO, name)
                    ytail = prev[j] if i > 0 else None
                    ltail = cur[j - 1] if j > 0 else None
                    ey = y_ov[i - 1] if i > 0 else 0
                    ex = x_ov[j - 1] if j > 0 else 0
                    oh = tile.shape[2] - (y_ov[i] if i < len(y_idx) - 1 else 0)
                    ow = tile.shape[3] - (x_ov[j] if j < len(x_idx) - 1 else 0)
                    ins = {"b": tile}
                    if ytail is not None:
                        ins["ytail"] = ytail
                    if ltail is not None:
                        ins["ltail"] = ltail
                    attrs = {"ey": ey, "ex": ex, "oy": out_y, "ox": out_x, "oh": oh, "ow": ow}
                    with self._ph("place"), REC.op(f"{name}.place", "place_tile", attrs, **ins) as o:
                        m.k_place_tile(tile, ytail, ey, ltail, ex, canvas, out_y, out_x, oh, ow)
                        o.out(y=canvas[:, :, out_y:out_y + oh, out_x:out_x + ow])
                cur.append(tile)
                out_x += ow
                tile_h = oh
            prev = cur
            out_y += tile_h
        return canvas

    # ------------------------------------------------------------------------------------------------ the video
    def decode(self, latents: torch.Tensor, *, tile_fn=None, rec=None, return_float: bool = False):
        """latents [1, 24, T, H, W] (or [24, T, H, W]), fp32 / bf16 / fp16 as the sampler hands them over (ComfyUI moves them
        to the GPU as fp16: `.to(device, dtype=fp16)`) -> uint8 frames [F, 16 H, 16 W, 3] on the GPU, F = 17 k + 5 for
        T = 5 k + 2 (T = 1: one frame). `rec(chunk, row, col) -> bool` picks the tiles whose ops are recorded (default all;
        clip-level ops use row = col = -1). `tile_fn(z, t0, Tn, y0, h, w, x0, name)` replaces the ViT3D tile decoder (the
        structure checks). `return_float`: also the fp32 pixels [F, H, W, 3] in [0, 1] (what ComfyUI's VAE.decode returns,
        without the batch dim)."""
        m = mod()
        rec = rec or (lambda c, r, k: True)
        tile_fn = tile_fn or self.decode_tile
        z = latents[0] if latents.dim() == 5 else latents
        assert z.dim() == 4 and z.shape[0] == ZC, "latents: [1, 24, T, H, W]"
        z = z.to(self.dev, torch.float16).contiguous()
        Tz, hz, wz = z.shape[1], z.shape[2], z.shape[3]
        H, W = hz * RATIO, wz * RATIO
        with _recording(rec(-1, -1, -1)):
            with REC.op("vvae.denorm", "denorm", {}, z=z, std=self.std, mean=self.mean) as o:
                z = m.k_denorm(z, self.std, self.mean)
                o.out(y=z)
        F = output_frames(Tz)
        out = torch.zeros(F, H, W, 3, dtype=torch.uint8, device=self.dev)
        outf = torch.zeros(F, H, W, 3, dtype=torch.float32, device=self.dev) if return_float else None
        sm = self.pix_std + self.pix_mean
        pos = 0

        def write_part(name, b, b0, nb, a=None, a0=0, na=0):
            """write_part(): the carried overlap `a` (na frames from a0) blended into the first frames of the part (nb frames
            from b0 of canvas b), finalised, converted, written at `pos`."""
            nonlocal pos
            if nb <= 0:
                return
            copy = min(nb, max(0, F - pos))
            ext = min(na, nb, FRAME_OVERLAP) if a is not None else 0
            if copy > 0:
                ins = {"b": b[:, b0:b0 + nb]}
                if ext > 0:
                    ins["a"] = a[:, a0:a0 + ext]
                attrs = {"b0": b0, "nb": nb, "ext": ext, "copy": copy, "pos": pos}
                with REC.op(name, "finalize", attrs, **ins) as o:
                    m.k_finalize(a if ext > 0 else None, a0, b, b0, ext, copy, sm, out, outf, pos)
                    o.out(y=out[pos:pos + copy])
            pos += copy

        if Tz == 1:
            canvas = self._decode_clip(z, 0, 0, 1, tile_fn, rec)
            with _recording(rec(0, -1, -1)):
                write_part("vvae.c0.part0", canvas, canvas.shape[1] - 1, 1)
        else:
            pad_tokens, num_chunks = temporal_chunks(Tz)
            Tpad = Tz + pad_tokens
            chunk_dec = TOKENS_CHUNK * RATIO_T
            split_count = int(TOKEN_DROP > 0) + 1
            overlap = None  # (canvas, first frame, frames): the carried dec_overlap
            for ci in range(num_chunks):
                t0 = ci * TOKENS_CHUNK
                t1 = t0 + TOKENS_CHUNK + TOKEN_OVERLAP
                Tn = max(0, min(t1, Tpad) - t0)
                canvas = self._decode_clip(z, ci, t0, Tn, tile_fn, rec)
                with _recording(rec(ci, -1, -1)):
                    for j in range(split_count):
                        f0 = j * chunk_dec
                        f1 = min(f0 + chunk_dec, canvas.shape[1])
                        b0 = f0 + FRAME_PRE_PADDING
                        nb = max(0, f1 - b0)
                        if j == 0:
                            if overlap is not None:
                                write_part(f"vvae.c{ci}.part0", canvas, b0, nb, overlap[0], overlap[1], overlap[2])
                                overlap = None
                            else:
                                write_part(f"vvae.c{ci}.part0", canvas, b0, nb)
                        else:
                            overlap = (canvas, b0, nb)
                    if ci == num_chunks - 1 and overlap is not None:
                        write_part(f"vvae.c{ci}.part1", overlap[0], overlap[1], overlap[2])
                        overlap = None
        assert pos == F, f"wrote {pos} frames of {F}"
        if self.prof:
            self._report()
        return (out, outf) if return_float else out
