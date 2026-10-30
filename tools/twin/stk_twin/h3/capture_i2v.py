"""Recorded image-conditioned H3 step: encoder, vision tower, raw text residual, refiner and DiT.
GPU entry point; --image requires PNG. Generates replay data for localrouter check h3-i2v.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import torch
from ..qwen_image import noise
from ..rec import REC
from .build import TE
from .dit import H3Dit
from .generate import VAE_VIDEO, align_frames
from .image import load_canvas
from .sampler import sigmas
from .shapes import PROMPT, latent_shapes
from .te32 import TextEncoder32
from .te_vision import VisionEncoder
from .vae_video_enc import VideoEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",default=os.environ.get("VMODELS",""))
    ap.add_argument("--pack",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--image",required=True)
    ap.add_argument("--size",default="768x448")
    ap.add_argument("--prompt",default=PROMPT)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--frames",type=int,default=56)
    ap.add_argument("--steps",type=int,default=8)
    ap.add_argument("--step",type=int,default=1)
    a=ap.parse_args()
    W,H=(int(v) for v in a.size.split("x"))
    if W%32 or H%32 or not 0<=a.step<a.steps: ap.error("size must be multiples of 32; step must be in schedule")
    rgb=load_canvas(a.image,W,H)
    frames=align_frames(a.frames)
    if (Path(a.out)/"ops.jsonl").exists(): ap.error("capture already exists; choose a fresh output directory")
    Path(a.out).mkdir(parents=True,exist_ok=True)
    if Path(a.image).resolve() != (Path(a.out)/"input.png").resolve(): shutil.copyfile(a.image,Path(a.out)/"input.png")
    REC.start(Path(a.out))
    REC.note("request",prompt=a.prompt,seed=a.seed,width=W,height=H,frames=frames,steps=a.steps,step=a.step,
             image="input.png",image_sha256=hashlib.sha256(Path(a.image).read_bytes()).hexdigest())
    with torch.inference_mode():
        with REC.op("image.canvas","image_canvas",{"width":W,"height":H}) as o:o.out(rgb=torch.from_numpy(rgb))
        enc=VideoEncoder.from_checkpoint(Path(a.models)/VAE_VIDEO)
        z=enc.encode(torch.from_numpy(rgb).cuda())
        del enc
        torch.cuda.empty_cache()
        with REC.op("first_frame","first_frame",{}) as o:o.out(rgb=torch.from_numpy(rgb),latent=z)
        vision=VisionEncoder(Path(a.models)/TE)
        merged,deep=vision.forward(rgb)
        del vision
        torch.cuda.empty_cache()
        from .te_vision import image_sequence
        ids,tags,positions,start=image_sequence(a.prompt,H//16,W//16)
        REC.note("te32_ids",ids=ids,tags=tags,image_start=start)
        te=TextEncoder32(Path(a.models)/TE)
        hidden=te.forward(ids,merged,start,deep,positions)
        context=hidden.to(torch.bfloat16).contiguous()
        del te,merged,deep
        torch.cuda.empty_cache()
        tags=torch.tensor(tags,dtype=torch.int64,device="cuda")
        with REC.op("text_encoder","text_encoder",{}) as o:o.out(hidden=hidden,context=context)
        with REC.op("text_tags","text_tags",{}) as o:o.out(tags=tags)
        dit=H3Dit.from_pack(a.pack,attn="int8",device="cuda")
        dit.prepare(context.unsqueeze(0),first_frame=z,seed=a.seed,text_tags=tags)
        vs,as_=latent_shapes(W,H,frames)
        nv,na=int(np.prod(vs)),int(np.prod(as_))
        samples=noise(a.seed,nv+na)
        xv=torch.from_numpy(samples[:nv]).view(vs).cuda().to(torch.bfloat16)
        xa=torch.from_numpy(samples[nv:]).view(as_).cuda().to(torch.bfloat16)
        schedule=sigmas(a.steps)
        REC.note("sigmas",values=schedule)
        with REC.op("step.inputs","inputs",{}) as o:o.out(video=xv,audio=xa)
        vv,va=dit(xv,xa,schedule[a.step])
        with REC.op("step.outputs","outputs",{}) as o:o.out(video=vv,audio=va)
    REC.stop()
    print(json.dumps({"h3_i2v_capture":str(a.out),"ops":REC.n,"tokens":len(ids),"sigma":schedule[a.step]}))

if __name__=="__main__":main()
