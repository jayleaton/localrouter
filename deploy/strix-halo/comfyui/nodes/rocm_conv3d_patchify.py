# ROCm Conv3d workaround for ComfyUI (installed into custom_nodes by deploy/strix-halo/comfyui/install.sh).
# On ROCm (gfx1151, PyTorch 2.14.1+rocm7.2), PyTorch's 3-D convolution kills the ComfyUI process (SIGSEGV, no Python
# exception) on MiniMax H3's image to video path: the Qwen3.5 vision tower that encodes the first frame embeds its patches with a Conv3d whose kernel equals
# its stride (Qwen35VisionPatchEmbed). With no padding, unit dilation and one group, such a convolution reads disjoint
# blocks of its input, one block per output position, so it is exactly the product of each flattened block with the
# flattened weight: this computes those convolutions as that product (F.linear) and leaves every other one to PyTorch.
# The results equal PyTorch's up to the order of the sums (test_conv3d_patchify.py).
# At import it replaces comfy.ops' Conv3d._conv_forward (the classes every ComfyUI model's Conv3d derives from), only on a
# ROCm build of PyTorch and only for tensors on the GPU; it modifies no files. LOCALROUTER_CONV3D_PATCHIFY=0 leaves
# ComfyUI as it is; =1 takes the product on any build and device (the tests). docs/dev/UPSTREAM.md, entry 14.
import logging
import os

MARK = "_localrouter_patchify"
log = logging.getLogger("localrouter")


def _ints(v, n):
    if isinstance(v, int):
        return (v,) * n
    return tuple(v)


def patchify(input, weight, bias, stride, padding, dilation, groups, padding_mode="zeros"):
    """conv3d(input, weight, bias, stride, padding, dilation, groups) as a product when it tiles its input: kernel
    equal to stride, no padding, dilation 1, one group, a batched 5-D input at least one block deep in every dimension.
    None for any other convolution (the caller runs PyTorch's)."""
    import torch.nn.functional as F
    if input.dim() != 5 or weight.dim() != 5 or groups != 1 or padding_mode != "zeros":
        return None
    k = tuple(weight.shape[2:])
    if isinstance(padding, str):
        if padding != "valid":
            return None
    elif any(p != 0 for p in _ints(padding, 3)):
        return None
    if _ints(stride, 3) != k or any(d != 1 for d in _ints(dilation, 3)):
        return None
    n, c = input.shape[0], input.shape[1]
    if c != weight.shape[1] or any(i < kk for i, kk in zip(input.shape[2:], k)):
        return None
    grid = [i // kk for i, kk in zip(input.shape[2:], k)]
    (gd, gh, gw), (kd, kh, kw) = grid, k
    # what a stride-k convolution reads: the trailing rows that do not fill a block are never read
    x = input[:, :, :gd * kd, :gh * kh, :gw * kw]
    x = x.reshape(n, c, gd, kd, gh, kh, gw, kw).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(n * gd * gh * gw, c * kd * kh * kw)
    out = F.linear(x, weight.reshape(weight.shape[0], -1), bias)
    return out.view(n, gd, gh, gw, weight.shape[0]).permute(0, 4, 1, 2, 3).contiguous()


def install(cls=None, every=None):
    """Replaces cls._conv_forward (comfy.ops.disable_weight_init.Conv3d by default). True when it is replaced (now or
    before). Without `every`, only on a ROCm build, for GPU tensors; LOCALROUTER_CONV3D_PATCHIFY decides first."""
    import torch
    mode = os.environ.get("LOCALROUTER_CONV3D_PATCHIFY", "")
    if mode == "0":
        return False
    if every is None:
        every = mode == "1"
    if not every and not torch.version.hip:
        return False
    if cls is None:
        import comfy.ops
        cls = comfy.ops.disable_weight_init.Conv3d
    orig = cls._conv_forward
    if getattr(orig, MARK, False):
        return True

    def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
        if autopad is None and not args and not kwargs and (every or input.is_cuda):
            out = patchify(input, weight, bias, self.stride, self.padding, self.dilation, self.groups, self.padding_mode)
            if out is not None:
                return out
        return orig(self, input, weight, bias, autopad, *args, **kwargs)

    setattr(_conv_forward, MARK, True)
    _conv_forward.__wrapped__ = orig
    cls._conv_forward = _conv_forward
    return True


try:
    if install():
        log.info("localrouter: Conv3d with kernel == stride runs as a matrix product on ROCm (its 3-D convolution crashes)")
except Exception as e:  # never stop ComfyUI from starting: without it only the crash it avoids comes back
    log.warning("localrouter: Conv3d workaround not installed: %s", e)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
