"""MiniMax H3's text encoder: ComfyUI 0.37.0's Qwen3-VL 32B text path (the NVFP4 AWQ checkpoint, 50 layers, fp32
activations, raw residual stream after layer 49, no final norm) op for op on LocalRouter's kernels
(kernels/cuda/minimax/te32.cu), launched as the Zig engine launches them, so the twin's bits are the engine's.

The weights are never expanded: each linear reads the checkpoint's own NVFP4 tensors (codes, swizzled e4m3 block
scales, fp32 tensor scale) and dequantizes them in the GEMM with comfy-kitchen's arithmetic. See tools/twin/TE32-PORT.md.

RoPE tables: ComfyUI computes inv_freq = 1 / (5e6 ** (arange(0, 128, 2).float() / 128)) with CUDA pow, and cos / sin with
CUDA cosf / sinf. Here inv_freq is exp(e * log(5e6)) in f64 by `smath` (fdlibm, operation for operation the same in
the Zig port) rounded to fp32, then 1 / that in fp32 (the correctly rounded powf); if kernels/minimax/te32_inv_freq.json
exists it overrides (freeze the GPU's 64 values there if the pod test reports differences). cos / sin of the fp32
angle f32(inv * pos): `smath.sincos` in f64 rounded to fp32 (as te.py does).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .. import smath
from ..rec import REC

KROOT = Path(os.environ.get("STK_KROOT", Path(__file__).resolve().parents[4] / "kernels"))
KDIR = KROOT / "cuda" / "minimax"

LAYERS, D, HEADS, KV_HEADS, HD, FFN = 50, 5120, 64, 8, 128, 25600
EPS = 1e-6
THETA = 5.0e6
VOCAB = 151936
EXTRA_TOKENS = {"<d>": 151669, "</d>": 151670, "<|cutoff|>": 151671, "<|lyrics_start|>": 151672,
                "<|lyrics_end|>": 151673, "<|caption_start|>": 151674, "<|caption_end|>": 151675}
PAD_ID = 151643

_DECLS = """
at::Tensor k_embed_i8(at::Tensor table, c10::optional<at::Tensor> scale, at::Tensor ids, int64_t bf16_round);
at::Tensor k_embed_bf16(at::Tensor table, at::Tensor ids);
at::Tensor k_add(at::Tensor a, at::Tensor b);
at::Tensor k_silu_mul(at::Tensor g, at::Tensor u);
at::Tensor k_rms_norm(at::Tensor x, at::Tensor w, double eps);
void k_rope_(at::Tensor x, at::Tensor cs, at::Tensor sn, int64_t mode);
at::Tensor k_attention(at::Tensor q, at::Tensor k, at::Tensor v, double scale);
at::Tensor k_linear_nvfp4(at::Tensor x, c10::optional<at::Tensor> pqs, at::Tensor codes, at::Tensor bscale,
                          double tscale, c10::optional<at::Tensor> bias);
"""

_LAUNCH = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "te32.cu"
#define F32(t) (t).data_ptr<float>()
#define STREAM at::cuda::getCurrentCUDAStream()
static unsigned cdiv(long long a, long long b) { return (unsigned)((a + b - 1) / b); }
static void chk(const at::Tensor& t, at::ScalarType s, const char* n) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == s && t.is_contiguous(), n, ": contiguous CUDA tensor of the right dtype");
}
at::Tensor k_embed_i8(at::Tensor table, c10::optional<at::Tensor> scale, at::Tensor ids, int64_t bf16_round) {
    chk(table, at::kChar, "table"); chk(ids, at::kInt, "ids");
    long long D = table.size(1), L = ids.numel();
    const float* sp = nullptr;
    if (scale.has_value()) { chk(*scale, at::kFloat, "scale"); TORCH_CHECK(scale->numel() == table.size(0), "scale rows"); sp = F32(*scale); }
    auto out = at::empty({L, D}, table.options().dtype(at::kFloat));
    te32_embed_i8<<<dim3(cdiv(D, 256), (unsigned)L), 256, 0, STREAM>>>((const int8_t*)table.data_ptr(), sp,
                                                                       ids.data_ptr<int>(), F32(out), D, (int)bf16_round);
    return out;
}
at::Tensor k_embed_bf16(at::Tensor table, at::Tensor ids) {
    chk(table, at::kBFloat16, "table"); chk(ids, at::kInt, "ids");
    long long D = table.size(1), L = ids.numel();
    auto out = at::empty({L, D}, table.options().dtype(at::kFloat));
    te32_embed_bf16<<<dim3(cdiv(D, 256), (unsigned)L), 256, 0, STREAM>>>((const te32_bf16*)table.data_ptr(),
                                                                         ids.data_ptr<int>(), F32(out), D);
    return out;
}
at::Tensor k_add(at::Tensor a, at::Tensor b) {
    chk(a, at::kFloat, "a"); chk(b, at::kFloat, "b");
    TORCH_CHECK(a.numel() == b.numel(), "add shapes");
    auto out = at::empty_like(a);
    long long n = a.numel();
    te32_add<<<cdiv(n, 256), 256, 0, STREAM>>>(F32(a), F32(b), F32(out), n);
    return out;
}
at::Tensor k_silu_mul(at::Tensor g, at::Tensor u) {
    chk(g, at::kFloat, "g"); chk(u, at::kFloat, "u");
    TORCH_CHECK(g.numel() == u.numel(), "silu_mul shapes");
    auto out = at::empty_like(g);
    long long n = g.numel();
    te32_silu_mul<<<cdiv(n, 256), 256, 0, STREAM>>>(F32(g), F32(u), F32(out), n);
    return out;
}
at::Tensor k_rms_norm(at::Tensor x, at::Tensor w, double eps) {
    chk(x, at::kFloat, "x"); chk(w, at::kFloat, "w");
    long long D = w.numel();
    TORCH_CHECK(x.numel() % D == 0, "rms_norm rows");
    auto y = at::empty_like(x);
    te32_rms_norm<<<(unsigned)(x.numel() / D), 256, 0, STREAM>>>(F32(x), F32(w), F32(y), D, (float)eps);
    return y;
}
void k_rope_(at::Tensor x, at::Tensor cs, at::Tensor sn, int64_t mode) {
    chk(x, at::kFloat, "x"); chk(cs, at::kFloat, "cos"); chk(sn, at::kFloat, "sin");
    TORCH_CHECK(x.dim() == 3 && x.size(2) == 128 && cs.size(1) == 64 && cs.size(0) >= x.size(0), "rope shapes");
    te32_rope_split_half<<<dim3((unsigned)x.size(0), (unsigned)x.size(1)), 64, 0, STREAM>>>(F32(x), F32(cs), F32(sn),
                                                                                           x.size(1), (int)mode);
}
at::Tensor k_attention(at::Tensor q, at::Tensor k, at::Tensor v, double scale) {
    chk(q, at::kFloat, "q"); chk(k, at::kFloat, "k"); chk(v, at::kFloat, "v");
    int L = (int)q.size(0), H = (int)q.size(1), KVH = (int)k.size(1);
    TORCH_CHECK(q.size(2) == 128 && k.size(2) == 128 && v.size(1) == KVH && k.size(0) == L && v.size(0) == L && H % KVH == 0,
                "attention shapes");
    auto out = at::empty({L, (long long)H * 128}, q.options());
    size_t smem = ((size_t)L + 256) * sizeof(float);
    if (smem > 48 * 1024)
        TORCH_CHECK(cudaFuncSetAttribute(te32_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) == cudaSuccess,
                    "attention: too many tokens for shared memory");
    te32_attention<<<dim3(L, H), 128, smem, STREAM>>>(F32(q), F32(k), F32(v), F32(out), L, H, KVH, (float)scale);
    return out;
}
at::Tensor k_linear_nvfp4(at::Tensor x, c10::optional<at::Tensor> pqs, at::Tensor codes, at::Tensor bscale,
                          double tscale, c10::optional<at::Tensor> bias) {
    chk(x, at::kFloat, "x"); chk(codes, at::kByte, "codes"); chk(bscale, at::kByte, "bscale");
    TORCH_CHECK(x.dim() == 2 && codes.dim() == 2, "linear shapes");
    long long M = x.size(0), K = x.size(1), N = codes.size(0);
    TORCH_CHECK(codes.size(1) * 2 == K && K % 16 == 0, "linear: codes [N, K / 2] with K % 16 == 0");
    long long cbc = (K / 16 + 3) / 4, rows = ((N + 127) / 128) * 128;
    TORCH_CHECK(bscale.numel() == rows * cbc * 4, "linear: block scales in the swizzled layout");
    const float* pp = nullptr; const float* bp = nullptr;
    if (pqs.has_value()) { chk(*pqs, at::kFloat, "pqs"); TORCH_CHECK(pqs->numel() == K, "pqs size"); pp = F32(*pqs); }
    if (bias.has_value()) { chk(*bias, at::kFloat, "bias"); TORCH_CHECK(bias->numel() == N, "bias size"); bp = F32(*bias); }
    auto y = at::empty({M, N}, x.options());
    te32_linear_nvfp4<<<dim3(cdiv(M, 128), cdiv(N, 64)), 256, 0, STREAM>>>(F32(x), pp, codes.data_ptr<uint8_t>(),
                                                                          bscale.data_ptr<uint8_t>(), (float)tscale, bp,
                                                                          F32(y), M, N, K);
    return y;
}
"""

NAMES = ["k_embed_i8", "k_embed_bf16", "k_add", "k_silu_mul", "k_rms_norm", "k_rope_", "k_attention", "k_linear_nvfp4"]


@lru_cache(maxsize=1)
def mod():
    from torch.utils.cpp_extension import load_inline

    major, minor = torch.cuda.get_device_capability()
    flags = ["-O3", f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"]  # no --use_fast_math
    digest = hashlib.sha256((KDIR / "te32.cu").read_bytes() + _LAUNCH.encode()).hexdigest()
    return load_inline(f"stk_te32_ops_{digest[:12]}", cpp_sources=_DECLS, cuda_sources=f"// {digest}\n" + _LAUNCH,
                       functions=NAMES, extra_cuda_cflags=flags, extra_include_paths=[str(KDIR)], with_cuda=True)


# ------------------------------------------------------------------------------------------------ RoPE tables
@lru_cache(maxsize=1)
def inv_freq() -> np.ndarray:
    """fp32 [64]: ComfyUI's `1.0 / (theta ** (arange(0, 128, 2).float() / 128))` on the GPU, portably (see the module
    docstring); kernels/minimax/te32_inv_freq.json ({"inv_freq": [64 floats]}) overrides when present."""
    frozen = KROOT / "minimax" / "te32_inv_freq.json"
    if frozen.exists():
        return np.array(json.loads(frozen.read_text())["inv_freq"], dtype=np.float32)
    ln = smath.log(THETA)
    out = np.empty(64, dtype=np.float32)
    for j in range(64):
        e = (2 * j) / 128.0  # arange(0, 128, 2).float() / 128: exact
        p = np.float32(smath.exp(e * ln))  # powf(5e6f, e), correctly rounded
        out[j] = np.float32(1.0) / p  # torch: reciprocal(p) * 1.0
    return out


def rope_tables(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos / sin [n, 64] fp32 for positions 0..n-1: freqs = f32(inv * pos) (the K = 1 matmul is one rounding),
    cos / sin in f64 by smath rounded to fp32."""
    inv = inv_freq()
    ang = np.arange(n, dtype=np.float32)[:, None] * inv[None, :]
    cos = np.empty((n, 64), dtype=np.float32)
    sin = np.empty((n, 64), dtype=np.float32)
    for (p, j), a in np.ndenumerate(ang):
        s, c = smath.sincos(float(a))
        sin[p, j], cos[p, j] = s, c
    return torch.from_numpy(cos), torch.from_numpy(sin)


# ------------------------------------------------------------------------------------------------ tokenizer
# The Qwen2.5 tokenizer files te32.tokenize reads, vendored in h3/tokenizer/ (the same bytes as ComfyUI 0.37.0's
# comfy/text_encoders/qwen25_tokenizer, so the twin needs no ComfyUI checkout). Their sha256s are checked on first use.
# TODO(upstream pin): the exact Hugging Face source (repo + revision) is not recorded yet; expected Qwen/Qwen2.5-7B (or
# -Instruct, same vocabulary and merges). Download vocab.json / merges.txt / tokenizer_config.json at a fixed revision,
# compare their sha256s with the table below, and fill in HF_TOKENIZER_REPO / HF_TOKENIZER_REV. If tokenizer_config.json
# differs (ComfyUI's copy may be trimmed), only vocab.json and merges.txt need to match the hub's.
TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizer"
HF_TOKENIZER_REPO = "Qwen/Qwen2.5-7B"      # TODO: confirm
HF_TOKENIZER_REV = None                    # TODO: pin the commit sha once the hub files are compared
TOKENIZER_SHA256 = {
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer_config.json": "49292bd4a58a43382bf01311bc3ed7a151a7f16fe07d155e900ea1af281555db",
}


def verify_tokenizer_files(d: Path) -> None:
    """Raises unless every file of TOKENIZER_SHA256 is in `d` with its pinned sha256 (the ids must not drift)."""
    for name, want in TOKENIZER_SHA256.items():
        f = d / name
        if not f.is_file():
            raise FileNotFoundError(f"tokenizer file {f} is missing")
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != want:
            raise ValueError(f"tokenizer file {f}: sha256 {got}, expected {want}")


@lru_cache(maxsize=1)
def tokenizer_dir() -> Path:
    """The directory of the Qwen2.5 tokenizer files (vocab.json, merges.txt, tokenizer_config.json): the vendored copy,
    or $STK_QWEN_TOKENIZER (a directory with the same files), verified against TOKENIZER_SHA256 on first use."""
    d = Path(os.environ.get("STK_QWEN_TOKENIZER") or TOKENIZER_DIR)
    verify_tokenizer_files(d)
    return d


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import Qwen2Tokenizer

    tok = Qwen2Tokenizer.from_pretrained(str(tokenizer_dir()))
    tok.add_special_tokens({"additional_special_tokens": list(EXTRA_TOKENS)})
    for s, i in EXTRA_TOKENS.items():
        assert tok.convert_tokens_to_ids(s) == i, f"special token {s} is not id {i}"
    return tok


def tokenize(prompt: str) -> list[int]:
    """ComfyUI's token ids for MiniMax H3 text to video (comfy/text_encoders/minimax.py `MiniMaxH3Tokenizer`, no images,
    no references): `SDTokenizer.tokenize_with_weights(disable_weights=True)` with `has_start_token = has_end_token =
    False`, no padding, no truncation, `min_length` 1 (an empty result is [151643]). No chat template, no BOS / EOS.
    `escape_important` / `unescape_important` turn the literal `\\(` and `\\)` into `(` and `)`; weights are not parsed;
    the text is split at `(?<=\\s)embedding:` and every piece is tokenized on its own. No embeddings directory is
    assumed (ComfyUI resolves `embedding:name` there when it exists; the workflow's prompts contain none)."""
    tok = _tokenizer()
    text = prompt.replace("\\)", "\0\1").replace("\\(", "\0\2")
    text = text.replace("\0\1", ")").replace("\0\2", "(")
    split = re.split(r"(?<=\s)embedding:", text)
    pieces = [split[0]] + ["embedding:" + s for s in split[1:]]
    ids: list[int] = []
    for word in (p for p in pieces if p != ""):
        ids += [int(t) for t in tok(word)["input_ids"]]
    return ids if ids else [PAD_ID]


# ------------------------------------------------------------------------------------------------ the model
@dataclass
class Lin:
    """One NVFP4 linear as the checkpoint stores it. `pqs` is the AWQ pre_quant_scale as fp32 (its bf16 values), or None."""
    key: str
    codes: torch.Tensor    # uint8 [N, K / 2]
    bscale: torch.Tensor   # uint8 (the float8_e4m3fn bytes), the swizzled layout
    tscale: float          # weight_scale_2
    pqs: torch.Tensor | None
    N: int
    K: int


def load_lin(f, key: str, dev) -> Lin:
    conf = json.loads(f.get_tensor(key + ".comfy_quant").numpy().tobytes())
    if conf.get("format") != "nvfp4":
        raise ValueError(f"{key}: comfy_quant {conf}, expected nvfp4")
    codes = f.get_tensor(key + ".weight")
    assert codes.dtype == torch.uint8
    N, K = codes.shape[0], codes.shape[1] * 2
    bs = f.get_tensor(key + ".weight_scale").contiguous().view(torch.uint8)
    cbc, rows = (K // 16 + 3) // 4, (N + 127) // 128 * 128
    assert bs.numel() == rows * cbc * 4, f"{key}: block scales {tuple(bs.shape)} for N={N}, K={K}"
    ts = f.get_tensor(key + ".weight_scale_2")
    assert ts.dtype == torch.float32 and ts.numel() == 1
    names = set(f.keys())
    pqs = None
    if key + ".pre_quant_scale" in names:
        pqs = f.get_tensor(key + ".pre_quant_scale").float().contiguous().to(dev)
        assert pqs.numel() == K, f"{key}: pre_quant_scale {tuple(pqs.shape)} for K={K}"
    assert key + ".input_scale" not in names  # unused by ComfyUI's full-precision matmul; none in this checkpoint
    return Lin(key, codes.contiguous().to(dev), bs.reshape(-1).to(dev), float(ts.reshape(())), pqs, N, K)


class TextEncoder32:
    """The checkpoint at `path` (qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) on the GPU: `forward(ids) -> [L, 5120]` fp32."""

    def __init__(self, path: str | Path, device="cuda", layers: int = LAYERS, rope_mode: int = 0):
        from safetensors import safe_open

        self.dev = torch.device(device)
        self.rope_mode = rope_mode
        self.nlayers = layers
        self.scale = float(np.float32(1.0 / math.sqrt(HD)))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            names = set(f.keys())
            self.embed_key = "model.embed_tokens"
            emb = f.get_tensor(self.embed_key + ".weight")
            if emb.dtype == torch.int8:
                conf = json.loads(f.get_tensor(self.embed_key + ".comfy_quant").numpy().tobytes())
                assert conf.get("format") == "int8_tensorwise" and not conf.get("convrot"), conf
                self.embed_scale = f.get_tensor(self.embed_key + ".weight_scale").float().contiguous().to(self.dev)
            else:
                assert emb.dtype == torch.bfloat16, emb.dtype
                self.embed_scale = None
            self.embed = emb.contiguous().to(self.dev)
            self.layers = []
            for i in range(layers):
                p = f"model.layers.{i}."
                nw = lambda k: f.get_tensor(p + k).float().contiguous().to(self.dev)  # bf16 -> fp32, exact
                lin = lambda k: load_lin(f, p + k, self.dev)
                self.layers.append({
                    "input_layernorm": nw("input_layernorm.weight"),
                    "post_attention_layernorm": nw("post_attention_layernorm.weight"),
                    "q_norm": nw("self_attn.q_norm.weight"), "k_norm": nw("self_attn.k_norm.weight"),
                    "q_proj": lin("self_attn.q_proj"), "k_proj": lin("self_attn.k_proj"),
                    "v_proj": lin("self_attn.v_proj"), "o_proj": lin("self_attn.o_proj"),
                    "gate_proj": lin("mlp.gate_proj"), "up_proj": lin("mlp.up_proj"), "down_proj": lin("mlp.down_proj"),
                })
            assert all(f"model.layers.{i}.self_attn.q_proj.weight" in names for i in range(layers))

    # ---- recorded ops
    def _lin(self, name: str, x: torch.Tensor, w: Lin) -> torch.Tensor:
        attrs = {"weight": w.key, "N": w.N, "K": w.K, "tscale": w.tscale, "pre_quant_scale": w.pqs is not None}
        with REC.op(name, "te32_linear_nvfp4", attrs, x=x) as o:
            y = mod().k_linear_nvfp4(x, w.pqs, w.codes, w.bscale, w.tscale, None)
            o.out(y=y)
        return y

    def _norm(self, name: str, x: torch.Tensor, w: torch.Tensor, key: str) -> torch.Tensor:
        with REC.op(name, "te32_rms_norm", {"eps": EPS, "weight": key, "D": w.numel()}, x=x) as o:
            y = mod().k_rms_norm(x, w, EPS)
            o.out(y=y)
        return y

    def _add(self, name: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        with REC.op(name, "te32_add", {}, a=a, b=b) as o:
            y = mod().k_add(a, b)
            o.out(y=y)
        return y

    @torch.inference_mode()
    def forward(self, ids) -> torch.Tensor:
        """ids [L] -> [L, 5120] fp32: the residual stream after layer index 49 (ComfyUI's CLIP.encode_from_tokens, [0])."""
        m = mod()
        ids = torch.as_tensor(ids).reshape(-1).to(self.dev, torch.int32)
        L = ids.numel()
        cos, sin = (t.to(self.dev) for t in rope_tables(L))
        with REC.op("te.embed", "te32_embed", {"weight": self.embed_key, "bf16_round": 1}, ids=ids) as o:
            h = (m.k_embed_i8(self.embed, self.embed_scale, ids, 1) if self.embed_scale is not None
                 else m.k_embed_bf16(self.embed, ids))
            o.out(y=h)
        for i, w in enumerate(self.layers):
            p, a = f"te.{i}", f"model.layers.{i}."
            x = self._norm(f"{p}.input_layernorm", h, w["input_layernorm"], a + "input_layernorm.weight")
            q = self._lin(f"{p}.q_proj", x, w["q_proj"])
            k = self._lin(f"{p}.k_proj", x, w["k_proj"])
            v = self._lin(f"{p}.v_proj", x, w["v_proj"]).view(L, KV_HEADS, HD)
            q = self._norm(f"{p}.q_norm", q.view(L, HEADS, HD), w["q_norm"], a + "self_attn.q_norm.weight")
            k = self._norm(f"{p}.k_norm", k.view(L, KV_HEADS, HD), w["k_norm"], a + "self_attn.k_norm.weight")
            with REC.op(f"{p}.rope", "te32_rope", {"mode": self.rope_mode}, q=q, k=k, cos=cos, sin=sin) as o:
                m.k_rope_(q, cos, sin, self.rope_mode)
                m.k_rope_(k, cos, sin, self.rope_mode)
                o.out(q=q, k=k)
            with REC.op(f"{p}.attention", "te32_attention", {"scale": self.scale, "causal": True}, q=q, k=k, v=v) as o:
                at = m.k_attention(q, k, v, self.scale)
                o.out(y=at)
            h = self._add(f"{p}.attn_residual", h, self._lin(f"{p}.o_proj", at, w["o_proj"]))
            x = self._norm(f"{p}.post_attention_layernorm", h, w["post_attention_layernorm"],
                           a + "post_attention_layernorm.weight")
            g = self._lin(f"{p}.gate_proj", x, w["gate_proj"])
            u = self._lin(f"{p}.up_proj", x, w["up_proj"])
            with REC.op(f"{p}.silu_mul", "te32_silu_mul", {}, g=g, u=u) as o:
                gu = m.k_silu_mul(g, u)
                o.out(y=gu)
            h = self._add(f"{p}.mlp_residual", h, self._lin(f"{p}.down_proj", gu, w["down_proj"]))
        return h

    def encode(self, prompt: str) -> torch.Tensor:
        ids = tokenize(prompt)
        REC.note("te32_ids", ids=ids)
        return self.forward(torch.tensor(ids))
