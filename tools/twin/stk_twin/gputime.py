"""GPU time of one twin DiT step (torch profiler, CUDA kernels only): the twin's side of the M3 speed gate, measured
on GPU time because the twin's Python launch loop can be CPU-bound on a pod. usage: python -m stk_twin.gputime PACK WxH"""

import json
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from .dit import QwenImageDiT

pack, size = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "1024x1024"
w, h = (int(v) // 16 for v in size.split("x"))
dit = QwenImageDiT.from_pack(pack)
ctx = torch.full((1, 64, 4096), 0.0117, device="cuda", dtype=torch.bfloat16)
x = torch.full((1, 64, h, w), 0.0117, device="cuda", dtype=torch.bfloat16)
t = torch.tensor([0.5], device="cuda")
for _ in range(3):
    dit(x, t, ctx)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(5):
        dit(x, t, ctx)
    torch.cuda.synchronize()
us = sum(e.self_device_time_total for e in p.key_averages()) / 5
print(json.dumps({"twin_gputime": {"precision": dit.precision, "size": size, "step_ms": us / 1000}}))
