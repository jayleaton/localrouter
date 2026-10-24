"""H3 one-frame video VAE encoder on repository kernels. Canvas RGB fp32 [H,W,3] in [0,1] -> normalized
fp32 [1,24,1,H/16,W/16]. No temporal clip path. See docs/dev/H3-I2V.md for the reference and substitutions.
"""
from pathlib import Path
import torch
from ..rec import REC
from . import vae_enc_ops as E
from . import vae_video as V

MULT = (1, 2, 2, 4, 4, 8)

class VideoEncoder:
    def __init__(self, weights, mean, std, device="cuda"):
        self.dev = torch.device(device)
        self.w = weights
        self.mean = mean.to(self.dev, torch.float16).contiguous()
        self.std = std.to(self.dev, torch.float16).contiguous()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device="cuda"):
        from safetensors import safe_open
        weights = {}
        with safe_open(str(path), framework="pt") as f:
            if any(k.endswith(".comfy_quant") for k in f.keys()):
                raise ValueError("use the fp16 video VAE checkpoint")
            def get(k, shape):
                t = f.get_tensor(k)
                if tuple(t.shape) != tuple(shape) or t.dtype != torch.float16:
                    raise ValueError(f"{k}: expected fp16 {shape}, got {t.dtype} {tuple(t.shape)}")
                return t.to(device).contiguous()
            def conv(p, ci, co, k=3):
                # Causal T=1: keep the final temporal tap, then flatten in C,ky,kx order. Pad K for the GEMM.
                t = get(p + ".weight", (co, ci, k, k, k))[:, :, -1].reshape(co, ci*k*k)
                K = (ci*k*k+7)//8*8
                w = torch.zeros(co, K, device=device, dtype=torch.float16)
                w[:, :ci*k*k] = t
                weights[p] = (w.contiguous(), get(p + ".bias", (co,)), k)
            def norm(p, c):
                weights[p] = (get(p + ".weight", (c,)), get(p + ".bias", (c,)))
            conv("encoder.conv_in", 3, 128)
            ci = 128
            for l, mult in enumerate(MULT):
                co = 128*mult
                for i in range(2):
                    p = f"encoder.down.{l}.block.{i}"
                    norm(p + ".norm1", ci); norm(p + ".norm2", co)
                    conv(p + ".conv1", ci, co); conv(p + ".conv2", co, co)
                    if ci != co:
                        conv(p + ".nin_shortcut", ci, co, 1)
                    ci = co
                if l < 4:
                    conv(f"encoder.down.{l}.downsample.conv", co, co)
            norm("encoder.norm_out", ci)
            conv("encoder.conv_out", ci, 48)
            conv("quant_conv", 48, 48, 1)
            mean, std = get("latents_mean", (24,)), get("latents_std", (24,))
        return cls(weights, mean, std, device)

    def _conv(self, prefix, p, x, h, w, stride=1, pad=1):
        wt, bias, k = self.w[p]
        oh, ow = h//stride, w//stride
        with REC.op(prefix + ".im2col", "encoder_im2col", {"H": h, "W": w, "k": k, "stride": stride,
                                                           "pad": pad}, x=x) as o:
            col = E.mod().k_im2col(x, h, w, k, stride, pad if k==3 else 0)
            o.out(y=col)
        M, K = col.shape
        N = wt.shape[0]
        out = torch.empty(M, N, device=self.dev, dtype=torch.float16)
        with REC.op(prefix + ".gemm", "gemm_f16", {"weight": p, "M": M, "N": N, "K": K,
                                                  "epilogue": "bias"}, x=col) as o:
            V.mod().k_gemm(col, wt, bias, out, None, None, M, N, K, K, K, N, 0, 0, 0, 0, 0, 0, 1, 0, 0)
            o.out(y=out)
        return out, oh, ow

    def _norm(self, prefix, p, x):
        wt, bias = self.w[p]
        with REC.op(prefix + ".gn", "encoder_group_norm", {"weight": p, "groups": 32, "eps": 1e-6}, x=x) as o:
            y = E.mod().k_group_norm(x, wt, bias)
            o.out(y=y)
        with REC.op(prefix + ".silu", "silu_f16", {}, x=y) as o:
            E.mod().k_silu_(y)
            o.out(y=y)
        return y

    def encode_tile(self, rgb, y0, x0, h, w, name):
        with REC.op(name + ".pixels", "encoder_pixels", {"y0": y0, "x0": x0, "h": h, "w": w}, rgb=rgb) as o:
            x = E.mod().k_pixels(rgb, y0, x0, h, w)
            o.out(y=x)
        x, h, w = self._conv(name + ".conv_in", "encoder.conv_in", x, h, w)
        for l in range(6):
            for i in range(2):
                p, n = f"encoder.down.{l}.block.{i}", f"{name}.L{l}.B{i}"
                residual = x
                y = self._norm(n + ".norm1", p + ".norm1", x)
                y, _, _ = self._conv(n + ".conv1", p + ".conv1", y, h, w)
                if p + ".nin_shortcut" in self.w:
                    residual, _, _ = self._conv(n + ".shortcut", p + ".nin_shortcut", x, h, w)
                y = self._norm(n + ".norm2", p + ".norm2", y)
                x, _, _ = self._conv(n + ".conv2", p + ".conv2", y, h, w)
                with REC.op(n + ".add", "add_f16", {}, x=x, residual=residual) as o:
                    E.mod().k_add_(x, residual)
                    o.out(y=x)
            if l < 4:
                x, h, w = self._conv(f"{name}.L{l}.down", f"encoder.down.{l}.downsample.conv", x, h, w, 2, 0)
        x = self._norm(name + ".norm_out", "encoder.norm_out", x)
        x, _, _ = self._conv(name + ".conv_out", "encoder.conv_out", x, h, w)
        x, _, _ = self._conv(name + ".quant", "quant_conv", x, h, w)
        with REC.op(name + ".moments", "encoder_moments", {"h": h, "w": w}, x=x) as o:
            tile = E.mod().k_moments(x, h, w)
            o.out(y=tile)
        return tile

    def encode(self, rgb, *, tile_fn=None, rec=None):
        """`rec(row,col)` selects tiles; (-1,-1) selects the final latent. Spatial blend reuses vv_place_tile;
        its [3,F,h,w] layout with F=16 is exactly the encoder's 48-channel CHW moments. Raw neighbors are retained.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 32 or any(n%16 for n in rgb.shape[:2]):
            raise ValueError("canvas must be RGB [H,W,3], both sides >=32 and multiples of 16")
        rgb = rgb.to(self.dev, torch.float32).contiguous()
        tile_fn, rec = tile_fn or self.encode_tile, rec or (lambda r, c: True)
        H, W = rgb.shape[:2]
        yi, yl, yo = V.split_tiles(H)
        xi, xl, xo = V.split_tiles(W)
        canvas = torch.empty(3, 16, H//16, W//16, device=self.dev, dtype=torch.float16)
        prev, oy = [None]*len(xi), 0
        for r, (y0, h) in enumerate(zip(yi, yl)):
            cur, ox = [], 0
            for c, (x0, w) in enumerate(zip(xi, xl)):
                name = f"venc.t{r}_{c}"
                with V._recording(rec(r, c)):
                    tile = tile_fn(rgb, y0, x0, h, w, name)
                    above, left = prev[c] if r else None, cur[-1] if c else None
                    ey, ex = yo[r-1]//16 if r else 0, xo[c-1]//16 if c else 0
                    oh = h//16 - (yo[r]//16 if r<len(yi)-1 else 0)
                    ow = w//16 - (xo[c]//16 if c<len(xi)-1 else 0)
                    ins = {"b": tile}
                    if above is not None: ins["ytail"] = above
                    if left is not None: ins["ltail"] = left
                    with REC.op(name + ".place", "place_tile", {"ey": ey, "ex": ex, "oy": oy, "ox": ox,
                                                                "oh": oh, "ow": ow}, **ins) as o:
                        V.mod().k_place_tile(tile, above, ey, left, ex, canvas, oy, ox, oh, ow)
                        o.out(y=canvas[:, :, oy:oy+oh, ox:ox+ow])
                cur.append(tile)
                ox += ow
            prev, oy = cur, oy+oh
        with V._recording(rec(-1, -1)):
            with REC.op("venc.latent", "encoder_latent", {}, moments=canvas, mean=self.mean, std=self.std) as o:
                z = E.mod().k_latent(canvas, self.mean, self.std)
                o.out(y=z)
        return z
