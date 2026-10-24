"""Offline vision gate for a later GPU window: python -m stk_twin.test_vision [--gpu].

CPU checks host plans against published reference methods/utilities. GPU checks every
new arithmetic kernel and a tiny random tower with actual 72-wide heads against HF SDPA
and eager backends. No checkpoint download. --model loads a local text_encoder directory.
Whole-tower gates are numeric: GEMM, reductions, attention and interpolation boundaries
can differ from installed reference releases; repeatability is checked bitwise.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from .vision import VisionEncoder, merge_coordinates, position_plan, rope_tables
from . import vision_ops


def metric(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return {"max_abs": float((a-b).abs().max()), "rel_l2": float((a-b).norm()/a.norm().clamp_min(1e-12)),
            "cosine": float(F.cosine_similarity(a[None], b[None]))}


def host_gate():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = Qwen3VLVisionConfig(hidden_size=144, num_heads=2, intermediate_size=160, depth=3,
                             out_hidden_size=128, num_position_embeddings=36, deepstack_visual_indexes=[0, 2])
    model = Qwen3VLVisionModel(cfg).to(dtype=torch.bfloat16).eval()
    grid = torch.tensor([[1, 4, 6], [2, 2, 4]])
    coords = merge_coordinates(grid)
    ids, weights = position_plan(grid, 6)
    if hasattr(model, "fast_pos_embed_interpolate"):
        ref = model.fast_pos_embed_interpolate(grid)
        products = model.pos_embed.weight[ids.long()] * weights[:, :, None]
        got = products[0]+products[1]+products[2]+products[3]
        torch.testing.assert_close(got, ref, rtol=0, atol=1e-6)
    else:
        from transformers.vision_utils import (get_vision_position_ids,
                                               get_vision_interpolation_indices_and_weights)
        torch.testing.assert_close(coords, get_vision_position_ids(grid, 2), rtol=0, atol=0)
        rid, rw = get_vision_interpolation_indices_and_weights(grid, 6, mode="bilinear",
                                                               align_corners=True, spatial_merge_size=2)
        torch.testing.assert_close(ids.long().T, rid, rtol=0, atol=0)
        torch.testing.assert_close(weights.T, rw.to(torch.bfloat16), rtol=0, atol=0)
    cos, sin = rope_tables(grid, model.rotary_pos_emb.inv_freq)
    inv = model.rotary_pos_emb.inv_freq.cpu()
    phase = (coords.float()[:, :, None]*inv).reshape(-1, 36)
    torch.testing.assert_close(cos, phase.cos(), atol=1e-6, rtol=0)
    torch.testing.assert_close(sin, phase.sin(), atol=1e-6, rtol=0)
    assert coords.shape == (40, 2) and ids.shape == weights.shape == (4, 40)
    assert merge_coordinates((1, 2, 2)).tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    for invalid in ((1, 3, 2), (0, 2, 2), (1.5, 2, 2)):
        try:
            merge_coordinates(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid grid accepted: {invalid}")
    return model, grid


def kernel_gate():
    torch.manual_seed(19)
    results = {}
    for dim in (144, 1152, 4608):
        x = torch.randn(7, dim, device="cuda", dtype=torch.bfloat16)
        w, b = torch.randn(dim, device="cuda", dtype=torch.bfloat16), torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        got = vision_ops.layer_norm(x, w, b)
        ref = F.layer_norm(x, (dim,), w, b, 1e-6)
        results[f"ln_{dim}"] = metric(ref, got)
        torch.testing.assert_close(got, ref, rtol=0.02, atol=0.04)
        assert torch.equal(got, vision_ops.layer_norm(x, w, b))
    grid = [(1, 4, 6), (1, 2, 4)]
    ids, weights = (t.cuda() for t in position_plan(grid, 6))
    table = torch.randn(36, 144, device="cuda", dtype=torch.bfloat16)
    products = table[ids.long()]*weights[:, :, None]
    ref = products[0]+products[1]+products[2]+products[3]
    got = vision_ops.position(table, ids, weights)
    assert torch.equal(got, ref)
    cos, sin = (t.cuda() for t in rope_tables(grid))
    x = torch.randn(32, 3*2*72, device="cuda", dtype=torch.bfloat16)
    q, k, v = vision_ops.qkv_rope_pad(x, cos, sin, 2)
    refq, refk, refv = x.reshape(32, 3, 2, 72).unbind(1)
    for got, ref in ((q, refq), (k, refk)):
        f = ref.float()
        rotated = torch.cat([-f[..., 36:], f[..., :36]], -1)
        expected = (f*cos.repeat(1, 2)[:, None]+rotated*sin.repeat(1, 2)[:, None]).to(torch.bfloat16)
        assert torch.equal(got[..., :72], expected)
        assert torch.count_nonzero(got[..., 72:]) == 0
    assert torch.equal(v[..., :72], refv) and torch.count_nonzero(v[..., 72:]) == 0
    assert torch.equal(vision_ops.unpad(v), refv.reshape(32, 144))
    x = torch.linspace(-8, 8, 4096, device="cuda", dtype=torch.float32)
    assert torch.equal(vision_ops.from_f32(x), x.to(torch.bfloat16))
    x = x.to(torch.bfloat16)
    got, ref = vision_ops.gelu(x), F.gelu(x, approximate="none")
    results["gelu_exact"] = metric(ref, got)
    torch.testing.assert_close(got, ref, atol=0.002, rtol=0.02)
    return results


def unpack(out):
    if hasattr(out, "pooler_output"):
        return out.pooler_output, out.deepstack_features
    return out


@torch.inference_mode()
def gpu_gate(model, grid):
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    patches = torch.randn(sum(int(t*h*w) for t, h, w in grid), model.patch_embed.proj.weight[0].numel())
    twin = VisionEncoder(model)
    got, deep = twin.forward(patches, grid)
    again, again_deep = twin.forward(patches, grid)
    deterministic = torch.equal(got, again) and all(torch.equal(a,b) for a,b in zip(deep, again_deep))
    # Two published backends show the reference's own numerical floor for the same weights.
    refs = []
    for backend in ("sdpa", "eager"):
        model.config._attn_implementation = backend
        refs.append(unpack(model(patches.cuda(), grid.cuda())))
    results = []
    for j, (ref, refdeep) in enumerate(refs):
        m = [metric(ref, got)] + [metric(a,b) for a,b in zip(refdeep, deep)]
        if len(refdeep) != len(deep):
            raise AssertionError("deepstack count mismatch")
        results.append({"backend": ("sdpa", "eager")[j], "outputs": m})
        if any(v["cosine"] < 0.99 or v["rel_l2"] > 0.08 for v in m):
            raise AssertionError(f"vision numeric gate failed: {results}")
    floor = [metric(refs[0][0], refs[1][0])] + [metric(a,b) for a,b in zip(refs[0][1], refs[1][1])]
    if not deterministic:
        raise AssertionError("vision kernels are not deterministic")
    return {"deterministic": deterministic, "reference": results, "reference_floor": floor}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model", help="local Qwen3-VL text_encoder directory; implies --gpu")
    args = parser.parse_args()
    torch.manual_seed(7)
    model, grid = host_gate()
    result = {"host": "passed"}
    if args.gpu or args.model:
        result["kernels"] = kernel_gate()
        if args.model:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
                                                                     torch_dtype=torch.bfloat16).model.visual
            grid = torch.tensor([[1, 8, 8], [1, 4, 8]])
        result["tower"] = gpu_gate(model, grid)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
