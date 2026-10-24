"""Qwen3-VL 32B vision tower on repository fp32 GEMM and vision kernels. One canvas image, no temporal video path.
Patch projection, interpolated absolute positions, 27 full-attention blocks and main/three deepstack mergers.
"""
from pathlib import Path
import math
import numpy as np
import torch
from .. import smath
from ..rec import REC
from . import ops as H
from . import vision_ops as V
from .te32 import inv_freq, tokenize, PAD_ID, load_lin, mod as te_mod

D, HEADS, HD, LAYERS, FFN, MERGE, OUT = 1152, 16, 72, 27, 4304, 4608, 5120
TAPS = (8, 16, 24)


def _trig(angles):
    cs, sn = np.empty_like(angles), np.empty_like(angles)
    for index, a in np.ndenumerate(angles):
        s, c = smath.sincos(float(a))
        cs[index], sn[index] = c, s
    return torch.from_numpy(cs), torch.from_numpy(sn)


def coordinates(gh, gw):
    return [(h*2+dy, w*2+dx) for h in range(gh//2) for w in range(gw//2) for dy in range(2) for dx in range(2)]


def linspace47(n):
    """torch.linspace(0, 47, n) in float32, as torch computes it: step = 47 / (n - 1), the first half start + i * step,
    the second half end - (n - 1 - i) * step (numpy's float64 linspace differs by an ulp at some sizes)."""
    if n == 1:
        return np.zeros(1, np.float32)
    step = np.float32(47) / np.float32(n - 1)
    i = np.arange(n)
    lo = np.float32(0) + i.astype(np.float32) * step
    hi = np.float32(47) - (n - 1 - i).astype(np.float32) * step
    return np.where(i < n // 2, lo, hi).astype(np.float32)


def vision_tables(gh, gw):
    coords = coordinates(gh, gw)
    # Vision rotary dim=36: 18 H frequencies followed by 18 W frequencies, then split-half rotate [72].
    inv = np.array([np.float32(1.)/np.float32(smath.exp((2*j/36)*smath.log(10000.))) for j in range(18)], np.float32)
    angles = np.array([[np.float32(h)*f for f in inv]+[np.float32(w)*f for f in inv] for h,w in coords], np.float32)
    cs, sn = _trig(angles)
    ids, weights = np.empty((4, len(coords)), np.int32), np.empty((4, len(coords)), np.float32)
    # Qwen35's bilinear interpolation of the 48x48 learned table, align-corners linspace.
    hs, ws = linspace47(gh), linspace47(gw)
    one = np.float32(1.)
    for i, (h,w) in enumerate(coords):
        hi, wi = int(hs[h]), int(ws[w]); hc, wc = min(hi+1,47), min(wi+1,47)
        dh, dw = np.float32(hs[h]-hi), np.float32(ws[w]-wi)
        ids[:,i] = hi*48+wi, hi*48+wc, hc*48+wi, hc*48+wc
        weights[:,i] = (one-dh)*(one-dw), (one-dh)*dw, dh*(one-dw), dh*dw
    return cs, sn, torch.from_numpy(ids), torch.from_numpy(weights)


def image_sequence(prompt, gh, gw):
    prefix = tokenize("<Picture 1>: ")+[151652]
    n, start = (gh//2)*(gw//2), len(prefix)
    suffix = [151653]+(tokenize(prompt) if prompt else [])
    ids = prefix+[PAD_ID]*n+suffix
    tags = [1]*(start-1)+[0]*(n+2)+[1]*(len(suffix)-1)
    positions = np.tile(np.arange(len(ids), dtype=np.float32), (3, 1))
    positions[0,start:start+n] = start
    # Match qwen2vl_mrope_position_ids literally (its H repeats are based on H rather than W).
    mh, mw = gh//2, gw//2
    positions[1,start:start+n] = np.repeat(np.arange(start,start+mh), math.ceil(n/mh))[:n]
    positions[2,start:start+n] = np.tile(np.arange(start,start+mw), math.ceil(n/mw))[:n]
    positions[:,start+n:] = np.arange(start+max(gh,gw)//2, start+max(gh,gw)//2+len(suffix), dtype=np.float32)
    return ids, tags, positions, start


def mrope_tables(positions):
    L = positions.shape[1]
    ang = np.empty((L,64), np.float32)
    inv = inv_freq()
    for j in range(64):
        axis = 1 if j%3==1 and j<60 else 2 if j%3==2 and j<60 else 0
        ang[:,j] = positions[axis]*inv[j]
    return _trig(ang)


class VisionEncoder:
    def __init__(self, path: str | Path, device="cuda"):
        from safetensors import safe_open
        self.dev, self.w = torch.device(device), {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            prefix = "visual."
            if not any(k.startswith(prefix) for k in keys):
                raise ValueError("checkpoint has no visual.* vision tensors")
            # Preserve NVFP4 linears if present; plain vision weights are stored in fp32 for the shared GEMM.
            for key in sorted(k for k in keys if k.startswith(prefix) and k.endswith(".weight")):
                p = key[:-7]
                if p+".comfy_quant" in keys:
                    self.w[p] = load_lin(f, p, self.dev)
                else:
                    weight = f.get_tensor(key)
                    if not weight.is_floating_point(): raise ValueError(f"{key}: unsupported vision weight {weight.dtype}")
                    self.w[key] = weight.float().contiguous().to(self.dev)
                if p+".bias" in keys: self.w[p+".bias"] = f.get_tensor(p+".bias").float().contiguous().to(self.dev)
        required = ["visual.patch_embed.proj.weight", "visual.pos_embed.weight"]
        if any(k not in self.w for k in required): raise ValueError("vision patch/position tensors must be unquantized")
        if self.w[required[0]].shape != (D,3,2,16,16) or self.w[required[1]].shape != (2304,D):
            raise ValueError("expected Qwen3-VL 32B vision patch/position shapes")

    def _lin(self, name, x, key):
        from .te32 import Lin
        w, bias = self.w.get(key), self.w.get(key+".bias")
        attrs = {"weight": key, "M": x.shape[0], "K": x.shape[1]}
        if isinstance(w, Lin):
            attrs.update(N=w.N, tscale=w.tscale, pre_quant_scale=w.pqs is not None)
            with REC.op(name,"te32_linear_nvfp4",attrs,x=x) as o:
                y = te_mod().k_linear_nvfp4(x,w.pqs,w.codes,w.bscale,w.tscale,bias); o.out(y=y)
        else:
            wt = self.w[key+".weight"].reshape(-1,x.shape[1]).contiguous()
            attrs.update(N=wt.shape[0], bias=bias is not None)
            with REC.op(name,"gemm_f32",attrs,x=x) as o:
                y = H.mod().k_gemm_f32(x,wt,bias); o.out(y=y)
        return y

    def _norm(self,name,x,key):
        with REC.op(name,"vision_norm",{"weight":key,"D":x.shape[-1],"eps":1e-6},x=x) as o:
            y=V.mod().norm(x,self.w[key+".weight"],self.w[key+".bias"]);o.out(y=y)
        return y

    def _gelu(self,name,x,approx):
        with REC.op(name,"vision_gelu",{"approximate":approx},x=x) as o:
            y=V.mod().gelu(x,int(approx));o.out(y=y)
        return y

    def _add(self,name,a,b):
        with REC.op(name,"te32_add",{},a=a,b=b) as o:
            y=te_mod().k_add(a,b);o.out(y=y)
        return y

    def _merge(self,name,x,key,deep=False):
        x=x.view(-1,MERGE) if deep else x
        x=self._norm(name+".norm",x,key+".norm").view(-1,MERGE)
        x=self._gelu(name+".gelu",self._lin(name+".fc1",x,key+".linear_fc1"),False)
        return self._lin(name+".fc2",x,key+".linear_fc2")

    @torch.inference_mode()
    def forward(self,rgb):
        # CPU preprocessing is shape/position setup; all learned arithmetic is dispatched to shared kernels.
        rgb=np.asarray(rgb,dtype=np.float32)
        h,w,c=rgb.shape
        if c!=3 or h%32 or w%32 or h*w<3136 or h*w>12845056:
            raise ValueError("vision canvas must have multiples of 32 and 3136..12845056 pixels")
        gh,gw=h//16,w//16
        normalized=(rgb-np.float32(.5))/np.float32(.5)
        repeat=np.stack([normalized.transpose(2,0,1)]*2)
        patches=repeat.reshape(1,2,3,gh//2,2,16,gw//2,2,16).transpose(0,3,6,4,7,2,1,5,8).reshape(gh*gw,1536).copy()
        pixels=torch.from_numpy(patches).to(self.dev)
        with REC.op("vision.patches","vision_patches",{"H":h,"W":w}) as o:o.out(y=pixels)
        cs,sn,ids,weights=(t.to(self.dev) for t in vision_tables(gh,gw))
        x=self._lin("vision.patch",pixels,"visual.patch_embed.proj")
        with REC.op("vision.position","vision_position",{"weight":"visual.pos_embed","L":gh*gw,"D":D},ids=ids,weights=weights) as o:
            pos=V.mod().position(self.w["visual.pos_embed.weight"],ids,weights);o.out(y=pos)
        x=self._add("vision.position_add",x,pos)
        deep=[]
        for i in range(LAYERS):
            name,key=f"vision.{i}",f"visual.blocks.{i}"
            z=self._norm(name+".norm1",x,key+".norm1")
            qkv=self._lin(name+".qkv",z,key+".attn.qkv").view(gh*gw,3,HEADS,HD)
            q,k,v=(qkv[:,j].contiguous() for j in range(3))
            with REC.op(name+".rope","vision_rope",{"H":HEADS,"D":HD},q=q,k=k,cos=cs,sin=sn) as o:
                V.mod().rope_(q,cs,sn);V.mod().rope_(k,cs,sn);o.out(q=q,k=k)
            scale=float(np.float32(1/math.sqrt(HD)))
            with REC.op(name+".attention","vision_attention",{"scale":scale},q=q,k=k,v=v) as o:
                z=V.mod().attention(q,k,v,scale).view(-1,D);o.out(y=z)
            x=self._add(name+".attention_add",x,self._lin(name+".proj",z,key+".attn.proj"))
            z=self._norm(name+".norm2",x,key+".norm2")
            z=self._gelu(name+".gelu",self._lin(name+".fc1",z,key+".mlp.linear_fc1"),True)
            x=self._add(name+".mlp_add",x,self._lin(name+".fc2",z,key+".mlp.linear_fc2"))
            if i in TAPS:
                j=TAPS.index(i);deep.append(self._merge(f"vision.deepstack.{j}",x,f"visual.deepstack_merger_list.{j}",True))
        return self._merge("vision.merger",x,"visual.merger"),deep
