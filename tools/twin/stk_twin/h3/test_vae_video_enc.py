"""GPU gate: python -m stk_twin.h3.test_vae_video_enc. Requires COMFY and VVAE (or VMODELS).
Checks kernels, tiled structure against ComfyUI, then per-channel cosine/relative L2 on the real encoder.
Thresholds VENC_MIN_COS=0.999 / VENC_MAX_REL=0.02 are provisional until measured. Missing references fail the gate.
"""
import json
import os
from pathlib import Path
import sys
import types
import torch
import torch.nn.functional as F
from . import vae_enc_ops as E
from . import vae_video as V
from .vae_video_enc import VideoEncoder
from .test_vae_video import import_comfy, make_shell, ulps

RESULTS = {}

def record(name, ok, **data):
    RESULTS[name] = {"ok": bool(ok), **data}
    print(f"{'PASS' if ok else 'FAIL'} {name} {data}", flush=True)

def kernels():
    torch.manual_seed(7)
    m = E.mod()
    rgb = torch.rand(48, 64, 3, device="cuda")
    got = m.k_pixels(rgb, 16, 16, 32, 32)
    ref = (rgb[16:48, 16:48]*2-1).half()
    mean = torch.tensor(V.IMAGENET_MEAN, device="cuda").half()
    std = torch.tensor(V.IMAGENET_STD, device="cuda").half()
    ref = (((ref+1)*0.5-mean)/std).reshape(-1, 3)
    record("pixels", torch.equal(got, ref))
    for C, k, stride, pad in ((3, 3, 1, 1), (128, 3, 2, 0), (256, 1, 1, 0)):
        x = torch.randn(32*48, C, device="cuda").half()
        col = m.k_im2col(x, 32, 48, k, stride, pad)
        chw = x.T.reshape(1, C, 32, 48)
        pads = (pad, pad, pad, pad) if pad else ((0, 1, 0, 1) if k==3 else (0, 0, 0, 0))
        ref = F.unfold(F.pad(chw.float(), pads, mode="reflect"), k, stride=stride)[0].T.half()
        expected = torch.zeros_like(col)
        expected[:, :ref.shape[1]] = ref
        record(f"im2col_{C}_{k}_{stride}", torch.equal(col, expected))
    for P, C in ((64, 128), (256, 512), (1024, 256)):
        x = torch.randn(P, C, device="cuda").half()
        w, b = torch.randn(C, device="cuda").half(), torch.randn(C, device="cuda").half()
        got = m.k_group_norm(x, w, b)
        ref = F.group_norm(x.T.reshape(1, C, 1, P).contiguous(), 32, w, b, 1e-6).reshape(C, P).T.contiguous()
        diff = (got.float()-ref.float()).abs()
        record(f"gn_{P}_{C}", float(diff.max()) < 0.008, max_abs=float(diff.max()), max_ulp=int(ulps(got, ref).max()))
        expected = F.silu(got)
        m.k_silu_(got)
        record(f"silu_{P}_{C}", int(ulps(got, expected).max()) <= 1)
        residual = torch.randn_like(got)
        expected = got+residual
        m.k_add_(got, residual)
        record(f"residual_{P}_{C}", torch.equal(got, expected))
    rows = torch.randn(12*16, 48, device="cuda").half()
    moments = m.k_moments(rows, 12, 16)
    record("moments", torch.equal(moments, rows.T.reshape(3, 16, 12, 16)))
    mean = torch.randn(24, device="cuda").half()
    std = (torch.rand(24, device="cuda")+0.5).half()
    z = m.k_latent(moments, mean, std)
    expected = (moments.reshape(48, 12, 16)[:24].float()-mean.float()[:, None, None])/std.float()[:, None, None]
    record("latent", torch.equal(z[0, :, 0], expected))

def structure(cvae):
    mean = torch.randn(24, device="cuda").half()
    std = (torch.rand(24, device="cuda")+0.5).half()
    shell = make_shell(cvae, mean, std)
    encoder = VideoEncoder(None, mean, std)
    def fake(x):
        # A local ramp makes neighboring raw tiles disagree, exercising both blend axes and trailing crops.
        base = x[:, :1, :, ::16, ::16].float().repeat(1, 48, 1, 1, 1)
        h, w = base.shape[-2:]
        ramp = (torch.arange(h, device=x.device)[:, None] + 2*torch.arange(w, device=x.device)[None, :])/17
        return (base+ramp).half()
    shell._encode_moments = fake
    def tile(rgb, y0, x0, h, w, name):
        x = rgb[y0:y0+h, x0:x0+w].permute(2, 0, 1)[None, :, None].contiguous()
        return fake(x)[0, :, 0].reshape(3, 16, h//16, w//16).contiguous()
    for H, W in ((32, 48), (256, 256), (448, 768), (768, 1344)):
        rgb = torch.rand(H, W, 3, device="cuda")
        refm = shell.tiled_encode(rgb.permute(2, 0, 1)[None, :, None].contiguous())
        ref = (refm[:, :24].float()-mean.float()[None, :, None, None, None])/std.float()[None, :, None, None, None]
        z = encoder.encode(rgb, tile_fn=tile)
        record(f"tiles_{H}_{W}", torch.equal(z, ref), shape=list(z.shape))

def end_to_end(ck):
    import comfy.sd
    import comfy.utils
    H, W = int(os.environ.get("VENC_H", 448)), int(os.environ.get("VENC_W", 768))
    rgb = torch.rand(H, W, 3, generator=torch.Generator().manual_seed(17))
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(ck)))
    with torch.inference_mode():
        ref = vae.encode(rgb[None]).to("cuda")
    del vae
    torch.cuda.empty_cache()
    ours = VideoEncoder.from_checkpoint(ck)
    z = ours.encode(rgb)
    cosine = F.cosine_similarity(z.flatten(2), ref.flatten(2), dim=2)[0]
    rel = (z-ref).flatten(2).norm(dim=2)[0]/ref.flatten(2).norm(dim=2)[0].clamp_min(1e-12)
    mincos, maxrel = float(os.environ.get("VENC_MIN_COS", 0.999)), float(os.environ.get("VENC_MAX_REL", 0.02))
    record("comfy_encoder", bool(torch.isfinite(z).all()) and float(cosine.min()) >= mincos and float(rel.max()) <= maxrel,
           cosine=cosine.tolist(), relative_l2=rel.tolist(), min_cos_allowed=mincos, max_rel_allowed=maxrel)
    record("determinism", torch.equal(z, ours.encode(rgb)))

def main():
    cvae = import_comfy()
    ck = Path(os.environ.get("VVAE") or str(Path(os.environ.get("VMODELS", ""))/"vae/minimax_h3_video_vae_fp16.safetensors"))
    if cvae is None or not ck.is_file():
        print(json.dumps({"pass": False, "error": "COMFY and the fp16 video VAE checkpoint are required"}))
        return 1
    with torch.inference_mode():
        kernels()
        structure(cvae)
        end_to_end(ck)
    ok = all(r["ok"] for r in RESULTS.values())
    print(json.dumps({"pass": ok, "checks": RESULTS}))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
