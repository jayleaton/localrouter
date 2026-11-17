#!/usr/bin/env python3
"""nodes/rocm_conv3d_patchify.py against PyTorch's own conv3d, on the CPU (no GPU, no ComfyUI):
  python -I deploy/strix-halo/comfyui/test_conv3d_patchify.py
Needs PyTorch (CI: a CPU build from requirements-test-cpu.txt, every file pinned by sha256)."""
import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

os.environ.pop("LOCALROUTER_CONV3D_PATCHIFY", None)
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rocm_conv3d_patchify", os.path.join(here, "nodes", "rocm_conv3d_patchify.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)  # on a ROCm build it tries comfy.ops, absent here, and only logs that

passed = failed = 0


def check(name, ok):
    global passed, failed
    if ok:
        passed += 1
    else:
        failed += 1
        print("FAIL:", name, file=sys.stderr)


def close(a, b, rtol, atol):
    return a.shape == b.shape and a.dtype == b.dtype and torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)


def case(name, x_shape, w_shape, stride, padding=0, dilation=1, groups=1, bias=True, expect=True):
    g = torch.Generator().manual_seed(len(name))
    x = torch.randn(x_shape, generator=g)
    w = torch.randn(w_shape, generator=g) / (w_shape[1] * w_shape[2] * w_shape[3] * w_shape[4]) ** 0.5
    b = torch.randn(w_shape[0], generator=g) if bias else None
    ref = F.conv3d(x, w, b, stride, padding, dilation, groups)
    out = m.patchify(x, w, b, stride, padding, dilation, groups)
    if not expect:
        check(f"{name}: not taken (PyTorch runs it)", out is None)
        return
    check(f"{name}: taken", out is not None)
    if out is None:
        return
    check(f"{name}: fp32 equals conv3d", close(out, ref, 1e-4, 1e-5))
    check(f"{name}: contiguous, as conv3d", out.is_contiguous())
    # fp16: against PyTorch's half conv3d where this build has one, else against fp32 on the same half inputs
    xh, wh, bh = x.half(), w.half(), b.half() if b is not None else None
    oh = m.patchify(xh, wh, bh, stride, padding, dilation, groups)
    try:
        rh = F.conv3d(xh, wh, bh, stride, padding, dilation, groups)
    except RuntimeError:
        rh = F.conv3d(xh.float(), wh.float(), bh.float() if bh is not None else None, stride, padding, dilation, groups).half()
    check(f"{name}: fp16 equals conv3d", oh is not None and close(oh, rh, 2e-2, 2e-2))


# the crash site: Qwen35VisionPatchEmbed (3 channels, temporal patch 2, patch 16, hidden 1152), patches as the batch
case("patch embed", (256, 3, 2, 16, 16), (1152, 3, 2, 16, 16), (2, 16, 16))
case("patch embed, no bias", (64, 3, 2, 16, 16), (1152, 3, 2, 16, 16), (2, 16, 16), bias=False)
case("1x1x1", (2, 8, 3, 5, 7), (4, 8, 1, 1, 1), 1)
case("multi-block", (2, 4, 4, 6, 9), (5, 4, 2, 3, 3), (2, 3, 3))
case("multi-block, rows left over", (2, 4, 5, 7, 10), (5, 4, 2, 3, 3), (2, 3, 3))
case("padding: passthrough", (1, 3, 4, 8, 8), (6, 3, 2, 2, 2), (2, 2, 2), padding=1, expect=False)
case("stride below kernel: passthrough", (1, 3, 4, 8, 8), (6, 3, 2, 2, 2), 1, expect=False)
case("dilation: passthrough", (1, 3, 4, 8, 8), (6, 3, 1, 2, 2), (1, 2, 2), dilation=2, expect=False)
case("groups: passthrough", (1, 4, 4, 8, 8), (6, 2, 2, 2, 2), (2, 2, 2), groups=2, expect=False)
check("smaller than a block: PyTorch's, which refuses it", m.patchify(torch.zeros(1, 3, 1, 8, 8), torch.zeros(6, 3, 2, 2, 2), None, (2, 2, 2), 0, 1, 1) is None)
check("'same' padding: passthrough", m.patchify(torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 1, 1, 1), None, 1, "same", 1, 1) is None)
check("'valid' padding: taken", m.patchify(torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 1, 1, 1), None, 1, "valid", 1, 1) is not None)


class Conv3d(torch.nn.Conv3d):
    """comfy.ops.disable_weight_init.Conv3d's shape: _conv_forward(input, weight, bias, autopad=None, *args, **kwargs)."""
    seen = []

    def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
        Conv3d.seen.append(autopad)
        if autopad == "causal_zero":
            weight = weight[:, :, -input.shape[2]:, :, :]
        return super()._conv_forward(input, weight, bias, *args, **kwargs)


check("not installed on a build without ROCm, unless asked", torch.version.hip or not m.install(Conv3d))
os.environ["LOCALROUTER_CONV3D_PATCHIFY"] = "0"
check("LOCALROUTER_CONV3D_PATCHIFY=0: not installed", not m.install(Conv3d, every=True))
del os.environ["LOCALROUTER_CONV3D_PATCHIFY"]
plain = Conv3d._conv_forward
check("installed when asked", m.install(Conv3d, every=True))
check("installed once", m.install(Conv3d, every=True) and Conv3d._conv_forward.__wrapped__ is plain)
torch.manual_seed(0)
with torch.no_grad():
    emb = Conv3d(3, 1152, (2, 16, 16), stride=(2, 16, 16))
    x = torch.randn(32, 3, 2, 16, 16)
    Conv3d.seen.clear()
    out = emb(x)  # nn.Conv3d.forward calls the replaced _conv_forward
    check("a patch embed through the class: the product, not the original", Conv3d.seen == [])
    check("a patch embed through the class equals conv3d", close(out, F.conv3d(x, emb.weight, emb.bias, emb.stride), 1e-4, 1e-5))
    pad = Conv3d(3, 6, 3, padding=1)
    y = torch.randn(1, 3, 4, 6, 6)
    check("a padded conv through the class: the original", torch.allclose(pad(y), F.conv3d(y, pad.weight, pad.bias, 1, 1)) and Conv3d.seen == [None])
    causal = Conv3d(3, 6, (3, 1, 1), stride=(3, 1, 1))
    z = torch.randn(1, 3, 2, 4, 4)
    out = causal._conv_forward(z, causal.weight, causal.bias, autopad="causal_zero")
    check("autopad goes to the original, as before", Conv3d.seen[-1] == "causal_zero" and torch.allclose(out, F.conv3d(z, causal.weight[:, :, -2:], causal.bias, (3, 1, 1))))

print(f"conv3d patchify: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
