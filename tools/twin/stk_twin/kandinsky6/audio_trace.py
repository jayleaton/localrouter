"""Component diagnostics of the audio decoder and the vocoder, recorded under the names the native runtime's hooks use
(src/engines/kandinsky6/audio/launch.zig `decoder_blocks`, `vocodeTraced`; src/cli/k6.zig `audio_stops`).

Nothing here computes anything of its own. `decode` runs upstream Decoder1D.forward, ResnetBlock1D.forward,
AttnBlock1D.forward and Upsample1D.forward statement by statement, calling the same upstream functions and modules in
the same order (vae_modules.normalize / nonlinearity / mp_sum, F.interpolate, F.scaled_dot_product_attention, the
MPConv1D modules), and VAE.decode's unnormalize. It only hands each intermediate to `record` as it is produced. The
vocoder is not re-run: `vocoder_hooks` puts forward hooks and pre-hooks on the upstream BigVGAN modules. Upstream
eac06e39 is checked by capture.py before anything is imported.

`record(name, tensor)` receives the bare name, e.g. `up0_first_shortcut` or `stage3_amp1`. capture.py stores it as
k6.aux.audio.decoder.<name> or k6.aux.audio.vocoder.<name>, role `raw`, uint8, the tensor's contiguous bytes. That is
exactly what native records.
"""

from __future__ import annotations

import sys

# --audio-trace value -> the decoder op after which the diagnostic stops; k6.zig `audio_stops` is the same table
STOPS = {
    "decoder": "first_conv",
    "residual": "first_silu",
    "residual-conv": "first_res_conv1",
    "residual-block": "first_residual",
    "attention": "attention_residual",
    "mid2": "mid2_residual",
    "up2-first": "up2_first_residual",
    "up2-second": "up2_second_residual",
    "up2-third": "up2_third_residual",
    "up1-first": "up1_first_residual",
    "up1-second": "up1_second_residual",
    "up1-third": "up1_third_residual",
    "upsample": "up1_upsample_conv",
    "up0-first": "up0_first_residual",
    "up0-second": "up0_second_residual",
    "up0-third": "up0_third_residual",
    "mel": "mel",
}
ORDINALS = ("first", "second", "third")
_BLOCK = ("norm", "silu", "res_conv1", "res_silu", "res_conv2")


def _names(block: str, shortcut: bool = False, clamp: bool = True) -> tuple:
    return tuple(f"{block}_{s}" for s in _BLOCK + (("shortcut",) if shortcut else ()) + ("residual",)
                 + (("clamp",) if clamp else ()))


# Every decoder hook of the native graph in order (launch.zig's test of the same list). The stage checks that what it
# recorded is this list, up to and including its stop.
DECODER_NAMES = (("latent_unscale", "first_conv") + _names("first", clamp=False)
                 + ("attention_qkv", "attention_norm", "attention_out", "attention_merged", "attention_proj",
                    "attention_residual")
                 + _names("mid2") + _names("up2_first") + _names("up2_second") + _names("up2_third")
                 + _names("up1_first", shortcut=True) + _names("up1_second") + _names("up1_third")
                 + ("up1_upsample", "up1_upsample_conv")
                 + _names("up0_first", shortcut=True) + _names("up0_second") + _names("up0_third")
                 + ("final_silu", "conv_out", "mel"))
assert len(DECODER_NAMES) == 91 and len(set(DECODER_NAMES)) == 91
assert all(stop in DECODER_NAMES for stop in STOPS.values())


class TraceStop(Exception):
    """Raised right after the stop op is recorded; the caller notes `diagnostic_stop` and ends the stage there."""

    def __init__(self, after: str):
        super().__init__(after)
        self.after = after


def _recorder(record, stop: str | None):
    def rec(name: str, t) -> None:
        record(name, t)
        if name == stop:
            raise TraceStop(name)
    return rec


def _block(vm, blk, x, rec, pre: str):
    """ResnetBlock1D.forward (vae_modules.py) with each intermediate recorded under `<pre>_*`."""
    if blk.use_norm:
        x = vm.normalize(x, dim=1)
    rec(pre + "_norm", x)
    h = x
    h = vm.nonlinearity(h)
    rec(pre + "_silu", h)
    h = blk.conv1(h)
    rec(pre + "_res_conv1", h)
    h = vm.nonlinearity(h)
    rec(pre + "_res_silu", h)
    h = blk.conv2(h)
    rec(pre + "_res_conv2", h)
    if blk.in_dim != blk.out_dim:
        if blk.use_conv_shortcut:
            x = blk.conv_shortcut(x)
        else:
            x = blk.nin_shortcut(x)
        rec(pre + "_shortcut", x)
    out = vm.mp_sum(x, h, t=0.3)
    rec(pre + "_residual", out)
    return out


def _attention(vm, attn, x, rec):
    """AttnBlock1D.forward (vae_modules.py) recorded as attention_*."""
    h = x
    y = attn.qkv(h)
    rec("attention_qkv", y)
    y = y.reshape(y.shape[0], attn.num_heads, -1, 3, y.shape[-1])
    y = vm.normalize(y, dim=2)
    rec("attention_norm", y)
    q, k, v = y.unbind(3)
    q = q.permute(0, 1, 3, 2)
    k = k.permute(0, 1, 3, 2)
    v = v.permute(0, 1, 3, 2)
    h = vm.F.scaled_dot_product_attention(q, k, v)
    rec("attention_out", h)
    h = h.permute(0, 1, 3, 2).reshape(h.shape[0], -1, h.shape[2])
    rec("attention_merged", h)
    h = attn.proj_out(h)
    rec("attention_proj", h)
    out = vm.mp_sum(x, h, t=0.3)
    rec("attention_residual", out)
    return out


def decode(vae, z, record, stop: str | None = None):
    """VAE.decode(z) (unnormalize=True) through Decoder1D.forward, recording every native decoder hook in order:
    first_conv, the residual blocks (mid.block_1 `first`, mid.block_2 `mid2`, up[L].block[i] `upL_<ordinal>`, each
    clamp `<block>_clamp`), the attention, up[L].upsample `upL_upsample` / `upL_upsample_conv`, final_silu, conv_out,
    mel. Raises TraceStop after `stop`. Returns the mel."""
    dec = vae.decoder
    vm = sys.modules[type(dec.mid.block_1).__module__]
    rec = _recorder(record, stop)
    h = dec.conv_in(z)
    rec("first_conv", h)
    h = _block(vm, dec.mid.block_1, h, rec, "first")
    h = _attention(vm, dec.mid.attn_1, h, rec)
    h = _block(vm, dec.mid.block_2, h, rec, "mid2")
    h = h.clamp(-dec.clip_act, dec.clip_act)
    rec("mid2_clamp", h)
    for i_level in reversed(range(dec.num_layers)):
        for i_block in range(dec.num_res_blocks + 1):
            pre = f"up{i_level}_{ORDINALS[i_block]}"
            h = _block(vm, dec.up[i_level].block[i_block], h, rec, pre)
            if len(dec.up[i_level].attn) > 0:
                raise RuntimeError(f"up[{i_level}] has attention; the native graph has none there")
            h = h.clamp(-dec.clip_act, dec.clip_act)
            rec(pre + "_clamp", h)
        if i_level in dec.down_layers:
            u = dec.up[i_level].upsample
            h = vm.F.interpolate(h, scale_factor=2.0, mode="nearest-exact")
            rec(f"up{i_level}_upsample", h)
            if u.with_conv:
                h = u.conv(h)
            rec(f"up{i_level}_upsample_conv", h)
    h = vm.nonlinearity(h)
    rec("final_silu", h)
    h = dec.conv_out(h, gain=(dec.learnable_gain + 1))
    rec("conv_out", h)
    mel = vae.unnormalize(h)
    rec("mel", mel)
    return mel


def vocoder_hooks(model, record) -> list:
    """Forward hooks / pre-hooks on upstream BigVGAN recording conv_pre, upsample<i>, stage<i>_amp<j>, stage<i>_avg (the
    merged AMP output, seen as the next stage's or activation_post's input), post_act, conv_post and wave. Returns
    the handles; the caller removes them."""
    hs = []

    def out_hook(name):
        def f(mod, args, out):
            record(name, out)
        return f

    def in_hook(name):
        def f(mod, args):
            record(name, args[0])
        return f

    hs.append(model.conv_pre.register_forward_hook(out_hook("conv_pre")))
    for i in range(model.num_upsamples):
        if len(model.ups[i]) != 1:
            raise RuntimeError(f"ups[{i}] has {len(model.ups[i])} layers; the native graph has one")
        hs.append(model.ups[i][0].register_forward_hook(out_hook(f"upsample{i}")))
        for j in range(model.num_kernels):
            hs.append(model.resblocks[i * model.num_kernels + j].register_forward_hook(out_hook(f"stage{i}_amp{j}")))
        nxt = model.ups[i + 1][0] if i + 1 < model.num_upsamples else model.activation_post
        hs.append(nxt.register_forward_pre_hook(in_hook(f"stage{i}_avg")))
    hs.append(model.activation_post.register_forward_hook(out_hook("post_act")))
    hs.append(model.conv_post.register_forward_hook(out_hook("conv_post")))
    hs.append(model.register_forward_hook(out_hook("wave")))
    return hs
