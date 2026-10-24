"""GPU gate vs ComfyUI 0.37.0: python -m stk_twin.h3.test_te_vision --ckpt PATH --image PNG.
Checks vision kernels, pixels/patches/positions/MRoPE/tags, main and three deepstack tensors, image-conditioned LM.
The absolute positions must be bit-equal; the other thresholds are provisional until a real checkpoint run. Missing references are failures, never skipped successes.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import vision_ops as V
from .image import load_canvas
from .te32 import TextEncoder32
from .te_vision import VisionEncoder,image_sequence,mrope_tables,vision_tables

RESULTS={}

def check(name,ok,**info):
    RESULTS[name]={"ok":bool(ok),**info}
    print(f"{'PASS' if ok else 'FAIL'} {name} {info}",flush=True)

def close(name,got,ref,cosmin=.99999,relmax=.005):
    if got.shape!=ref.shape:
        check(name,False,ours=list(got.shape),reference=list(ref.shape));return
    got,ref=got.double().flatten(),ref.double().flatten()
    finite=bool(torch.isfinite(got).all())
    cos=float(F.cosine_similarity(got,ref,dim=0));rel=float((got-ref).norm()/ref.norm())
    check(name,finite and cos>=cosmin and rel<=relmax,cosine=cos,relative_l2=rel,finite=finite)

def kernels():
    torch.manual_seed(7)
    m=V.mod()
    for d in (1152,4608):
        x=torch.randn(32,d,device="cuda");w=torch.randn(d,device="cuda");b=torch.randn_like(w)
        close(f"layernorm.{d}",m.norm(x,w,b),F.layer_norm(x,(d,),w,b,1e-6),.999999,1e-5)
        for approx in (False,True):
            got=m.gelu(x,int(approx));ref=F.gelu(x,approximate="tanh" if approx else "none")
            check(f"gelu.{d}.{approx}",float((got-ref).abs().max())<2e-6,max_abs=float((got-ref).abs().max()))
    q,k,v=(torch.randn(32,16,72,device="cuda") for _ in range(3))
    ref=F.scaled_dot_product_attention(q.transpose(0,1).double(),k.transpose(0,1).double(),v.transpose(0,1).double()).transpose(0,1).float()
    close("attention",m.attention(q,k,v,1/np.sqrt(72)),ref,.999999,1e-5)
    cs,sn,_,_=vision_tables(8,12);cs,sn=cs.cuda(),sn.cuda()
    x=torch.randn(96,16,72,device="cuda");ref=torch.cat((x[...,:36]*cs[:,None]-x[...,36:]*sn[:,None],x[...,36:]*cs[:,None]+x[...,:36]*sn[:,None]),-1)
    m.rope_(x,cs,sn);check("vision.rope",torch.equal(x,ref))

def references(a,rgb):
    sys.path.insert(0,a.comfy)
    import comfy.sd
    import comfy.model_management
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images,qwen2vl_mrope_position_ids
    from comfy.text_encoders.minimax import token_tags_from_embeds_info
    clip=comfy.sd.load_clip([a.ckpt],embedding_directory=None,clip_type=comfy.sd.CLIPType.MINIMAX)
    tokens=clip.tokenize(a.prompt,images=[torch.from_numpy(rgb)[None]])
    captured={}
    # Hook the reference visual tower itself; load_clip owns checkpoint quantization/dtype and model patching.
    visual=clip.cond_stage_model.qwen3vl_32b.transformer.visual
    def hook(module,inputs,output):
        captured["absolute_position"]=module.fast_pos_embed_interpolate(inputs[1]).detach().float().cpu()
        captured["patches"]=inputs[0].detach().float().cpu()
        captured["merged"]=output[0].detach().float().cpu()
        captured["deep"]=[x.detach().float().cpu() for x in output[1]]
    handle=visual.register_forward_hook(hook)
    try: cond=clip.encode_from_tokens(tokens)[0].float().cpu()
    finally: handle.remove()
    h,w=rgb.shape[:2];gh,gw=h//16,w//16
    ids,tags,positions,start=image_sequence(a.prompt,gh,gw)
    entries=tokens["qwen3vl_32b"][0]
    expected=[]
    for token,*_ in entries:
        expected.extend([151643]*((gh//2)*(gw//2)) if isinstance(token,dict) else [int(token)])
    check("tokens",ids==expected,ours=len(ids),reference=len(expected))
    info=[{"type":"image","index":start,"size":(gh//2)*(gw//2),"extra":torch.tensor([[1,gh,gw]])}]
    refpos=qwen2vl_mrope_position_ids(info,len(ids),"cpu")
    check("positions",np.array_equal(positions,refpos.numpy()))
    check("tags",tags==token_tags_from_embeds_info(len(ids),info).tolist())
    cs,sn=mrope_tables(positions)
    inv=1/(5.e6**(torch.arange(0,128,2,device="cuda").float()/128))
    freq=torch.from_numpy(positions).cuda()[:,:,None]*inv[None,None,:]
    expected=freq[0].clone()
    expected[:,1:60:3]=freq[1,:,1:60:3]
    expected[:,2:60:3]=freq[2,:,2:60:3]
    check("mrope.tables",float((cs.cuda()-expected.cos()).abs().max())<2e-5 and float((sn.cuda()-expected.sin()).abs().max())<2e-5)
    patches,grid=process_qwen2vl_images(torch.from_numpy(rgb)[None],patch_size=16,image_mean=[.5]*3,image_std=[.5]*3)
    check("patches.reference", torch.equal(captured["patches"],patches.cpu()))
    normalized=(rgb-np.float32(.5))/np.float32(.5)
    repeated=np.stack([normalized.transpose(2,0,1)]*2)
    ours=repeated.reshape(1,2,3,gh//2,2,16,gw//2,2,16).transpose(0,3,6,4,7,2,1,5,8).reshape(gh*gw,1536).copy()
    check("patches.twin", np.array_equal(ours,patches.numpy()))
    captured.update(hidden=cond)
    del clip
    comfy.model_management.unload_all_models();torch.cuda.empty_cache()
    return captured

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True);ap.add_argument("--image",required=True)
    ap.add_argument("--comfy",default=os.environ.get("COMFY",""));ap.add_argument("--size",default="128x96")
    ap.add_argument("--prompt",default="A quiet stream flows through a forest.")
    a=ap.parse_args()
    if not a.comfy:ap.error("--comfy or COMFY is required")
    w,h=(int(v) for v in a.size.split("x"))
    rgb=load_canvas(a.image,w,h)
    kernels();ref=references(a,rgb)
    vision=VisionEncoder(a.ckpt)
    merged,deep=vision.forward(rgb)
    _,_,pids,pweights=vision_tables(h//16,w//16)
    position=V.mod().position(vision.w["visual.pos_embed.weight"],pids.cuda(),pweights.cuda()).cpu()
    # h3v_position rounds as ComfyUI's bf16 interpolation does: equal bits (test_vision_position checks the arithmetic on the CPU)
    want=ref["absolute_position"]
    check("vision.absolute_position",position.shape==want.shape and torch.equal(position,want),
          differ=int((position!=want).sum()) if position.shape==want.shape else None)
    close("vision.merged",merged.cpu(),ref["merged"])
    for i,(got,want) in enumerate(zip(deep,ref["deep"])):close(f"vision.deepstack.{i}",got.cpu(),want)
    del vision;torch.cuda.empty_cache()
    ids,tags,positions,start=image_sequence(a.prompt,h//16,w//16)
    te=TextEncoder32(a.ckpt)
    hidden=te.forward(ids,merged,start,deep,positions).cpu()
    close("image.hidden",hidden,ref["hidden"],.99999,.005)
    print(json.dumps({"pass":all(v["ok"] for v in RESULTS.values()),"checks":RESULTS}))
    return 0 if all(v["ok"] for v in RESULTS.values()) else 1

if __name__=="__main__":raise SystemExit(main())
