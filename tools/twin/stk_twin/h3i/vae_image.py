"""MiniMax H3-Image's VAE decoder (ComfyUI e9afbccc `comfy/ldm/minimax/vae.py`, `MiniMaxH3ImageVAE.decode`, the fp16
checkpoint `minimax_h3_image_vae_fp16.safetensors`; docs/dev/H3-IMAGE.md section 1.5 and "Decided") op for op, batch 1,
one frame, on the video decoder's kernels where they exist (`stk_twin.h3.vae_video.mod()`: gemm_f16.cu, vae_video.cu).

  decode(z [1, 24, 1, h, w])  z = z * std + mean in fp16 (the H3 tables of vae.py, two roundings), then ComfyUI's
                              tiled_decode: split_tiles (256 px, overlap >= 64, ratio 16) on both sides, the tiles in rows
                              top to bottom, left to right in a row, each decoded ALONE (ComfyUI batches up to 4 by free
                              memory: the declared substitution), blended first along y against the previous row's
                              already blended bottom strip, then along x against the left neighbour's already blended right
                              tail (blend in the tile's dtype), cropped by its trailing overlaps into the canvas, then
                              finalised: clamp(fp32 * pixel_std + pixel_mean, 0, 1) -> fp32 [H, W, 3] and uint8.
  one tile                    post_quant_conv (1x1, a GEMM) -> x_embedder 24 -> 1024 -> + 4 register tokens + 1 zero token
                              -> 24 blocks (RMSNorm, to_qkv, per-head RMSNorm (unit weight) + split-half RoPE on 32 of 64
                              dims over the 2-D grid (h, w), softmax attention, nan_to_num, to_out + addcmul scale1,
                              RMSNorm, w1, SwiGLU, w2 + addcmul scale2) -> LayerNorm -> proj_out 1024 -> 768 -> unshuffle
                              to [3, 16 h, 16 w].

Kernels: the GEMMs (bias, addcmul epilogue), denorm, gather, suffix, RMSNorm, LayerNorm, SwiGLU, the attention's three
steps (scores GEMM, row softmax, P v GEMM) and nan_to_num, the blend (`vv_place_tile`, onto a scratch tile so the
blended strips stay available) and finalize are the video decoder's, unchanged. No kernel exists yet (PR 1.6 adds them;
until then these run as torch ops with the kernels' arithmetic, marked `pending kernel`): the RoPE over 32 of 64 dims
(vv_rms_rope rotates 48) with its 2-axis table (computed on the host), and the 768-column unshuffle (vv_unshuffle is
3072). `ops` picks the provider: `GpuVaeOps` (default) or `cpu_ops.CpuVaeOps` (the CPU structure gate).
"""

from __future__ import annotations

import numpy as np
import torch

from ..rec import REC
from ..smath import sincos
from ..h3.vae_video import split_tiles

LAYERS, HEADS, HD, DIM, FFN = 24, 16, 64, 1024, 4096
ZC, OUT_C, RATIO = 24, 3, 16
ROT = 32                                    # rope_dim_ratio 0.5 of the 64-dim head
PAIRS = ROT // 2                            # 2 axes (h, w) x 8 frequencies
EPS = 1e-5
NSUF = 5                                    # 4 register tokens + 1 zero token
OUT_ROWS = OUT_C * RATIO * RATIO            # 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TWO_PI_F32 = np.float32(2.0 * np.pi)

# comfy/ldm/minimax/vae.py LATENTS_MEAN / LATENTS_STD (the image VAE file carries no statistics; sd.py sets these)
LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
)


def inv_freq(act: torch.dtype) -> torch.Tensor:
    """RotaryEmbeddingND(32, 100, 2).inv_freq: 1 / 100 ** arange(0, 1, 4 / 32) in fp32 (8 values), then the module's
    dtype (fp16 for the shipping decoder)."""
    return (1 / 100.0 ** torch.arange(0, 1, 2 * 2 / ROT, dtype=torch.float32)).to(act)


def _round(act: torch.dtype):
    if act == torch.float32:
        return lambda a: np.asarray(a, np.float32)
    npdt = {torch.float16: np.float16, torch.bfloat16: None}[act]
    if npdt is None:
        raise ValueError("the decoder runs in fp16 or fp32")
    return lambda a: np.asarray(a, np.float32).astype(npdt).astype(np.float32)


def rope_table(h: int, w: int, inv: torch.Tensor, act: torch.dtype) -> torch.Tensor:
    """The 2-D RoPE table [h * w + 5, 16, 4] (c, -s, s, c) in `act` (pending kernel: computed on the host, as the DiT's):
    create_token_ids over (h, w) in `act` (q = (i + 0.5) * (1 / n), id = 2 q - 1, each rounded), the 5 suffix tokens at
    id 0; angle = ((2 pi as fp32) * float(id)) * float(inv_freq[k]), pair = axis * 8 + k; cos / sin by LocalRouter's
    portable sincos (f64 of the fp32 angle, rounded to fp32), then rounded to `act`."""
    r = _round(act)

    def coord(n):
        i = np.arange(n, dtype=np.float32)
        q = r((i + np.float32(0.5)) * (np.float32(1.0) / np.float32(n)))
        return r(r(np.float32(2.0) * q) - np.float32(1.0))

    yy, xx = np.meshgrid(coord(h), coord(w), indexing="ij")
    ids = np.zeros((h * w + NSUF, 2), dtype=np.float32)
    ids[: h * w, 0], ids[: h * w, 1] = yy.reshape(-1), xx.reshape(-1)
    fr = inv.float().cpu().numpy().astype(np.float32)
    ang = ((TWO_PI_F32 * ids[:, :, None]).astype(np.float32) * fr[None, None, :]).astype(np.float32).reshape(-1, PAIRS)
    c, s = np.empty_like(ang), np.empty_like(ang)
    for idx, a in np.ndenumerate(ang):
        s[idx], c[idx] = sincos(float(a))
    table = np.stack([c, -s, s, c], -1)
    return torch.from_numpy(r(table)).to(act)


def f16_const(vals) -> list[float]:
    """ImageNet constants as the decoder holds them: fp32 tensor -> module.to(fp16) -> .to(fp32)."""
    return torch.tensor(vals, dtype=torch.float32).half().float().tolist()


class GpuVaeOps:
    """The image decoder's ops on LocalRouter's kernels (the video decoder's extension, `h3.vae_video.mod()`), fp16.
    `rms_rope_` (32 of 64 dims) and `unshuffle` (768 columns) have no kernel yet: torch on the device, the kernels'
    arithmetic (`cpu_ops.CpuVaeOps`'s, device-agnostic)."""

    act = torch.float16

    def __init__(self, device="cuda"):
        from .cpu_ops import CpuVaeOps

        self.dev = torch.device(device)
        self._torch = CpuVaeOps(torch.float16)
        self._scores = None

    @staticmethod
    def m():
        from ..h3.vae_video import mod

        return mod()

    def denorm(self, z, std, mean):
        return self.m().k_denorm(z.contiguous(), std, mean)

    def gather_rows(self, z, y0, h, x0, w):
        return self.m().k_gather_rows(z, 0, 1, y0, h, x0, w)

    def linear(self, x, wt, bias, *, out=None, res=None, rscale=None):
        M, K = x.shape
        N = wt.shape[0]
        if out is None:
            out = torch.empty(M, N, dtype=torch.float16, device=x.device)
        self.m().k_gemm(x, wt, bias, out, res, rscale, M, N, K, K, K, out.stride(0), 0, 0, 0, 0, 0, 0, 1,
                        0 if res is None else res.stride(0), 0)
        return out

    def suffix_(self, hs, reg, nP):
        self.m().k_suffix_(hs, reg, nP)

    def rms_norm(self, x, w, eps):
        return self.m().k_rms_norm(x, w, eps)

    def layer_norm(self, x, w, b, eps):
        return self.m().k_layer_norm(x, w, b, eps)

    def rms_rope_(self, qkv, table, S, H, eps, rot):
        self._torch.rms_rope_(qkv, table, S, H, eps, rot)       # pending kernel (vv_rms_rope is rot 48)

    def attention(self, qkv, S, H):
        """The video decoder's attention, H heads: S = q k^T (fp16 GEMM, fp16 out), the fp32 row softmax (fp16 out),
        O = P v (GEMM), then nan_to_num; qkv rows [S, H * 192]."""
        m = self.m()
        SP = (S + 7) // 8 * 8
        if self._scores is None or self._scores.shape != (H, S, SP):
            self._scores = torch.zeros(H, S, SP, dtype=torch.float16, device=qkv.device)
        sc = self._scores
        vt = m.k_vt(qkv, S, SP, H)
        m.k_gemm(qkv, qkv, None, sc, None, None, S, S, HD, qkv.stride(0), qkv.stride(0), SP, 0, HD, 0, 3 * HD, 3 * HD,
                 S * SP, H, 0, 0)
        m.k_softmax_(sc, S, SP, 0.125)
        att = torch.empty(S, H * HD, dtype=torch.float16, device=qkv.device)
        m.k_gemm(sc, vt, None, att, None, None, S, HD, SP, SP, SP, H * HD, 0, 0, 0, S * SP, HD * SP, HD, H, 0, 0)
        m.k_nan_to_num_(att)
        return att

    def swiglu(self, gu):
        return self.m().k_swiglu(gu)

    def unshuffle(self, rows, h, w):
        return self._torch.unshuffle(rows, h, w)                # pending kernel (vv_unshuffle is 3072 columns)

    def place_tile(self, b, ytail, ey, ltail, ex, canvas, oy, ox, oh, ow):
        self.m().k_place_tile(b, ytail, ey, ltail, ex, canvas, oy, ox, oh, ow)

    def finalize(self, b, sm):
        H, W = b.shape[2], b.shape[3]
        u8 = torch.zeros(1, H, W, 3, dtype=torch.uint8, device=b.device)
        f = torch.zeros(1, H, W, 3, dtype=torch.float32, device=b.device)
        self.m().k_finalize(None, 0, b, 0, 0, 1, list(sm), u8, f, 0)
        return u8[0], f[0]


class ImageVAE:
    """The decode path of MiniMaxH3ImageVAE. `from_checkpoint(path)` loads post_quant_conv and the decoder from the fp16
    file; `decode(latents)` returns (uint8 [16 h, 16 w, 3], fp32 [16 h, 16 w, 3] in [0, 1])."""

    def __init__(self, w: dict, ops=None, device="cuda"):
        self.ops = ops or GpuVaeOps(device)
        act = self.ops.act
        self.dev = self.ops.dev
        self.w = w
        self.mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32).to(self.dev, act)
        self.std = torch.tensor(LATENTS_STD, dtype=torch.float32).to(self.dev, act)
        self.inv_freq = inv_freq(act)
        self.pix = (f16_const(IMAGENET_STD) + f16_const(IMAGENET_MEAN)) if act == torch.float16 else \
            (list(IMAGENET_STD) + list(IMAGENET_MEAN))
        self._tables: dict = {}

    @classmethod
    def from_checkpoint(cls, path, device="cuda", ops=None) -> "ImageVAE":
        from safetensors import safe_open

        ops = ops or GpuVaeOps(device)
        with safe_open(str(path), framework="pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys() if k.startswith(("decoder.", "post_quant_conv."))}
        return cls(cls.weights(sd, ops.dev, ops.act), ops=ops)

    @staticmethod
    def weights(sd: dict, device, act) -> dict:
        """The decoder's tensors from a state dict with the published names (`post_quant_conv.*`, `decoder.*`), shapes
        checked, on `device` in `act`."""

        def get(k, shape):
            t = sd[k]
            if tuple(t.shape) != tuple(shape):
                raise ValueError(f"{k}: shape {tuple(t.shape)}, expected {tuple(shape)}")
            return t.to(device, act).contiguous()

        d = "decoder."
        n = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith(d + "transformer_blocks."))
        w = {"pqc.w": get("post_quant_conv.weight", (ZC, ZC, 1, 1)).view(ZC, ZC).contiguous(),
             "pqc.b": get("post_quant_conv.bias", (ZC,)),
             "embed.w": get(d + "x_embedder.weight", (DIM, ZC)), "embed.b": get(d + "x_embedder.bias", (DIM,)),
             "reg": get(d + "register_tokens", (1, 4, DIM)).view(4, DIM).contiguous(),
             "norm_out.w": get(d + "norm_out.weight", (DIM,)), "norm_out.b": get(d + "norm_out.bias", (DIM,)),
             "proj.w": get(d + "proj_out.weight", (OUT_ROWS, DIM)), "proj.b": get(d + "proj_out.bias", (OUT_ROWS,)),
             "layers": []}
        for i in range(n):
            p = f"{d}transformer_blocks.{i}."
            w["layers"].append({
                "norm1": get(p + "norm1.weight", (DIM,)),
                "qkv.w": get(p + "attn.to_qkv.weight", (3 * DIM, DIM)), "qkv.b": get(p + "attn.to_qkv.bias", (3 * DIM,)),
                "out.w": get(p + "attn.to_out.weight", (DIM, DIM)), "out.b": get(p + "attn.to_out.bias", (DIM,)),
                "scale1": get(p + "scale1", (DIM,)),
                "norm2": get(p + "norm2.weight", (DIM,)),
                "w1.w": get(p + "ff.w1.weight", (2 * FFN, DIM)), "w1.b": get(p + "ff.w1.bias", (2 * FFN,)),
                "w2.w": get(p + "ff.w2.weight", (DIM, FFN)), "w2.b": get(p + "ff.w2.bias", (DIM,)),
                "scale2": get(p + "scale2", (DIM,)),
            })
        return w

    def table(self, h: int, w: int) -> torch.Tensor:
        if (h, w) not in self._tables:
            self._tables[(h, w)] = rope_table(h, w, self.inv_freq, self.ops.act).to(self.dev).contiguous()
        return self._tables[(h, w)]

    # ------------------------------------------------------------------------------------------------ one tile
    def _lin(self, name, x, wt, bias, **kw):
        attrs = {"M": x.shape[0], "N": wt.shape[0], "K": x.shape[1], "epilogue": "bias+addcmul" if kw.get("res") is not None else "bias"}
        with REC.op(name, "gemm_f16", attrs, x=x) as o:
            y = self.ops.linear(x, wt, bias, **kw)
            o.out(y=y[: x.shape[0]])
        return y

    def decode_tile(self, z: torch.Tensor, y0: int, h: int, x0: int, w: int, name: str) -> torch.Tensor:
        """The decoder on post_quant_conv of the latent window [y0, y0 + h) x [x0, x0 + w) of z [24, 1, Hz, Wz] (the
        denormalised latents) -> [3, 1, 16 h, 16 w]."""
        o_, W = self.ops, self.w
        nP = h * w
        S = nP + NSUF
        with REC.op(f"{name}.gather", "gather_rows", {"y0": y0, "h": h, "x0": x0, "w": w}, z=z) as o:
            rows = o_.gather_rows(z, y0, h, x0, w)
            o.out(y=rows)
        x = self._lin(f"{name}.pqc", rows, W["pqc.w"], W["pqc.b"])
        hs = torch.zeros(S, DIM, dtype=o_.act, device=z.device)
        self._lin(f"{name}.embed", x, W["embed.w"], W["embed.b"], out=hs)
        with REC.op(f"{name}.suffix", "suffix", {"nP": nP}, reg=W["reg"]) as o:
            o_.suffix_(hs, W["reg"], nP)
            o.out(y=hs[nP:])
        table = self.table(h, w)
        for i, b in enumerate(W["layers"]):
            p = f"{name}.L{i}"
            with REC.op(f"{p}.norm1", "rms_norm_f16", {"eps": EPS}, x=hs) as o:
                n = o_.rms_norm(hs, b["norm1"], EPS)
                o.out(y=n)
            qkv = self._lin(f"{p}.qkv", n, b["qkv.w"], b["qkv.b"])
            with REC.op(f"{p}.rr", "rms_rope_f16", {"eps": EPS, "rot_dim": ROT, "pending_kernel": True}, qkv=qkv, table=table) as o:
                o_.rms_rope_(qkv, table, S, HEADS, EPS, ROT)
                o.out(qkv=qkv)
            with REC.op(f"{p}.attn", "attention_f16", {"S": S, "heads": HEADS}, qkv=qkv) as o:
                att = o_.attention(qkv, S, HEADS)
                o.out(y=att)
            self._lin(f"{p}.out", att, b["out.w"], b["out.b"], out=hs, res=hs, rscale=b["scale1"])
            with REC.op(f"{p}.norm2", "rms_norm_f16", {"eps": EPS}, x=hs) as o:
                n = o_.rms_norm(hs, b["norm2"], EPS)
                o.out(y=n)
            gu = self._lin(f"{p}.w1", n, b["w1.w"], b["w1.b"])
            with REC.op(f"{p}.swi", "swiglu_f16", {}, x=gu) as o:
                a = o_.swiglu(gu)
                o.out(y=a)
            self._lin(f"{p}.w2", a, b["w2.w"], b["w2.b"], out=hs, res=hs, rscale=b["scale2"])
        # the head on the patch rows only (ComfyUI projects all S rows and drops the suffix; a row depends on itself alone)
        xs = hs[:nP]
        with REC.op(f"{name}.norm_out", "layer_norm_f16", {"eps": EPS}, x=xs) as o:
            n = o_.layer_norm(xs, W["norm_out.w"], W["norm_out.b"], EPS)
            o.out(y=n)
        rows = self._lin(f"{name}.proj", n, W["proj.w"], W["proj.b"])
        with REC.op(f"{name}.unshuf", "unshuffle", {"H": h, "W": w, "pending_kernel": True}, x=rows) as o:
            tile = o_.unshuffle(rows, h, w)
            o.out(y=tile)
        return tile

    # ------------------------------------------------------------------------------------------------ the image
    @torch.inference_mode()
    def decode(self, latents: torch.Tensor, *, tile_fn=None) -> tuple[torch.Tensor, torch.Tensor]:
        """latents [1, 24, 1, h, w] (or [24, 1, h, w]), normalized, any float dtype (ComfyUI hands the VAE fp16) ->
        (uint8 [16 h, 16 w, 3], fp32 [16 h, 16 w, 3]). `tile_fn(z, y0, h, x0, w, name)` replaces the tile decoder (the
        tiling checks)."""
        o_ = self.ops
        tile_fn = tile_fn or self.decode_tile
        z = latents[0] if latents.dim() == 5 else latents
        if z.dim() != 4 or z.shape[0] != ZC or z.shape[1] != 1:
            raise ValueError("latents: [1, 24, 1, h, w] (one frame)")
        z = z.to(self.dev, o_.act).contiguous()
        with REC.op("vimg.denorm", "denorm", {}, z=z) as o:
            z = o_.denorm(z, self.std, self.mean)
            o.out(y=z)
        H, W = z.shape[2] * RATIO, z.shape[3] * RATIO
        y_idx, y_len, y_ov = split_tiles(H)
        x_idx, x_len, x_ov = split_tiles(W)
        canvas = torch.zeros(OUT_C, 1, H, W, dtype=o_.act, device=z.device)
        strip, out_y, ny, nx = None, 0, len(y_idx), len(x_idx)
        for i, (ip, il) in enumerate(zip(y_idx, y_len)):
            new_strip, left, out_x, tile_h = None, None, 0, 0
            for j, (jp, jl) in enumerate(zip(x_idx, x_len)):
                name = f"vimg.t{i}_{j}"
                tile = tile_fn(z, ip // RATIO, il // RATIO, jp // RATIO, jl // RATIO, name)
                th, tw = tile.shape[2], tile.shape[3]
                ey = y_ov[i - 1] if i > 0 else 0
                ex = x_ov[j - 1] if j > 0 else 0
                ytail = strip[:, :, :, jp:jp + tw].contiguous() if i > 0 else None
                ins = {"b": tile, **({"ytail": ytail} if ytail is not None else {}), **({"ltail": left} if left is not None else {})}
                blended = torch.empty_like(tile)
                with REC.op(f"{name}.blend", "place_tile", {"ey": ey, "ex": ex, "oy": 0, "ox": 0, "oh": th, "ow": tw}, **ins) as o:
                    o_.place_tile(tile, ytail, ey, left, ex, blended, 0, 0, th, tw)
                    o.out(y=blended)
                left = blended[:, :, :, tw - x_ov[j]:].contiguous() if j < nx - 1 else None
                ow = tw - (x_ov[j] if j < nx - 1 else 0)
                oh = th - (y_ov[i] if i < ny - 1 else 0)
                if i < ny - 1:
                    if new_strip is None:
                        new_strip = torch.empty(OUT_C, 1, y_ov[i], W, dtype=o_.act, device=z.device)
                    new_strip[:, :, :, out_x:out_x + ow] = blended[:, :, th - y_ov[i]:, :ow]
                with REC.op(f"{name}.place", "copy", {"oy": out_y, "ox": out_x, "oh": oh, "ow": ow}, x=blended) as o:
                    canvas[:, :, out_y:out_y + oh, out_x:out_x + ow] = blended[:, :, :oh, :ow]
                    o.out(y=canvas[:, :, out_y:out_y + oh, out_x:out_x + ow])
                out_x += ow
                tile_h = oh
            strip = new_strip
            out_y += tile_h
        with REC.op("vimg.finalize", "finalize", {}, b=canvas) as o:
            u8, f = o_.finalize(canvas, self.pix)
            o.out(y=u8)
        return u8, f


def tile_plan(side_px: int) -> dict:
    """The decoder's tiles along one side, as ComfyUI's split_tiles cuts them (pixels)."""
    starts, lens, ovs = split_tiles(side_px)
    return {"starts": starts, "lengths": lens, "overlaps": ovs}

