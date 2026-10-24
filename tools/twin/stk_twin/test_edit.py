"""Deferred edit gate: host contracts against local ComfyUI source, small random DiT against TensorFold on GPU.
Run in the granted window with --comfy pointing at the read-only 0.37.0 checkout; never downloads weights.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .edit import Span, positions, prefix_layout
from .dit import Config, QwenImageDiT, image_ids


def reference_method(path, name):
    """Execute the named published function in isolation; no ComfyUI startup or device discovery on import."""
    tree = ast.parse(Path(path).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[fn], type_ignores=[])
    scope = {"torch": torch, "math": __import__("math")}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


def host_gate(comfy):
    fn = reference_method(Path(comfy) / "comfy/ldm/qwen_image21/model.py", "build_sequence")
    pos_ref = reference_method(Path(comfy) / "comfy/text_encoders/qwen_vl.py", "qwen2vl_mrope_position_ids")
    # Identity projections expose the actual published token and rotary-id construction without model weights.
    fake = SimpleNamespace(txt_in=lambda x: x, img_in=lambda x: x, pe_embedder=lambda ids: ids.unsqueeze(1))
    cases = 0
    for count in range(1, 6):
        hw = [(2 + i % 2, 4 + i % 3) for i in range(count)]
        slots = tuple([0, 2, 2, 5, 7][:count])
        text = torch.arange(7 * 64, dtype=torch.float32).view(1, 7, 64)
        refs = [torch.full((1, 64, h, w), float(i + 1000)) for i, (h, w) in enumerate(hw)]
        target = torch.zeros(1, 64, 3, 5)
        sequence, pe, ref_segments = fn(fake, target, text, refs, list(slots))
        pieces, segments, next_pos = prefix_layout(7, hw, slots)
        parts, ids, pos = [], [], 0
        for start, end, i in pieces:
            n = end - start
            if n:
                parts.append(text[:, start:end])
                ids.append(torch.arange(pos, pos + n).float()[:, None].expand(n, 3))
                pos += n
            if i is not None:
                h, w = hw[i]
                parts.append(refs[i].flatten(2).transpose(1, 2))
                ids.append(image_ids(h, w, pos, (3, 5)))
                pos += max(h, w)
        want = torch.cat(parts, dim=1)
        assert torch.equal(sequence[:, :want.shape[1]], want)
        assert torch.equal(pe[0, :want.shape[1], 0], torch.cat(ids))
        assert pos == next_pos
        assert [(a, e, mask is not None) for a, e, mask in ref_segments[:-1]] == segments
        # Published masks independently prove that every reference sees its complete block, but no later rows.
        for (a, e, causal), (_, _, mask) in zip(segments, ref_segments):
            allow = torch.ones(e - a, e, dtype=torch.bool)
            if causal:
                assert torch.equal(mask, allow.tril(a))
            else:
                assert mask is None
        spans = [Span(4, 6, (1, 4, 6)), Span(13, 8, (1, 4, 8))]
        info = [{"type": "image", "index": s.start, "size": s.count,
                 "extra": {"grid": torch.tensor([s.grid])}} for s in spans]
        assert torch.equal(positions(25, spans), pos_ref(info, 25, "cpu"))
        cases += 1
    for bad in ((3, 2), (-1,), (8,)):
        try:
            prefix_layout(7, [(2, 2)] * len(bad), bad)
        except ValueError:
            continue
        raise AssertionError("invalid slots accepted")
    return {"host_reference_cases": cases}


def weights(cfg):
    g = torch.Generator().manual_seed(29)
    rand = lambda n, k: (torch.randn(n, k, generator=g) / k**0.5).to(torch.bfloat16)
    d, m = cfg.dim, cfg.mlp
    w = {"img_in.weight": rand(d, 64), "txt_in.text_norm.weight": torch.zeros(d),
         "txt_in.in_layer.weight": rand(d, d), "txt_in.out_layer.weight": rand(d, d),
         "time_text_embed.timestep_embedder.linear_1.weight": rand(d, 256),
         "time_text_embed.timestep_embedder.linear_2.weight": rand(d, d), "modulation.1.weight": rand(4*d, d),
         "norm_out.linear.weight": rand(d, d), "proj_out.weight": rand(64, d)}
    for i in range(cfg.layers):
        p = f"transformer_blocks.{i}."
        for name in ("to_q", "to_k", "to_v"):
            w[p + f"attn.{name}.weight"] = rand(d, d)
        w[p + "attn.to_out.0.weight"] = rand(d, d)
        w[p + "attn.norm_q.weight"] = torch.ones(128)
        w[p + "attn.norm_k.weight"] = torch.ones(128)
        w[p + "img_mlp.gate_layer.weight"] = rand(m, d)
        w[p + "img_mlp.proj.weight"] = rand(m, d)
        w[p + "img_mlp.out.weight"] = rand(d, m)
    return w


def gpu_gate():
    from tfimage.qwen_image21 import QwenImage21, Config as RefConfig

    cfg = Config(layers=2, dim=128, heads=1, mlp=256)
    w = weights(cfg)
    ours = QwenImageDiT.from_bf16(w, "bf16", cfg=cfg)
    ref = QwenImage21({k: v.cuda() for k, v in w.items()}, "bf16", cfg=RefConfig(layers=2, dim=128, heads=1, mlp=256))
    torch.manual_seed(37)
    context = torch.randn(1, 7, 128, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(1, 64, 3, 5, device="cuda", dtype=torch.bfloat16)
    reports = []
    with torch.inference_mode():
        for count in range(1, 6):
            refs = [torch.randn(1, 64, 2 + i % 2, 4 + i % 3, device="cuda", dtype=torch.bfloat16) for i in range(count)]
            slots = [0, 2, 2, 5, 7][:count]
            for sigma in (1.0, 0.6, 0.2):
                t = torch.tensor([sigma], device="cuda")
                a = ours(target, t, context, refs, slots)
                b = ref.forward(target, t, context, refs, slots)
                cos = float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))
                rel = float((a.float() - b.float()).norm() / b.float().norm())
                assert cos >= 0.999 and rel <= 0.05, (count, sigma, cos, rel)
                assert torch.equal(a, ours(target, t, context, refs, slots)), "cached repeat changed bits"
                ours.drop_prefixes()
                assert torch.equal(a, ours(target, t, context, refs, slots)), "fresh prefix changed bits"
                reports.append({"refs": count, "sigma": sigma, "cos": cos, "rel_l2": rel})
            saved = ours.prefixes[-1]
            refs[0].flatten()[0] += 1
            ours(target, t, context, refs, slots)
            assert ours.prefixes[-1] is not saved, "changed reference reused stale prefix"
    return {"dit": reports}


@torch.inference_mode()
def te_gate():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from .te import TextEncoder

    cfg = Qwen3VLTextConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                           num_attention_heads=1, num_key_value_heads=1, head_dim=128,
                           rope_parameters={"rope_type": "default", "rope_theta": 5000000.0,
                                            "mrope_section": [24, 20, 20], "mrope_interleaved": True})
    cfg._attn_implementation = "sdpa"
    lm = Qwen3VLTextModel(cfg).cuda().to(torch.bfloat16).eval()
    te = TextEncoder(SimpleNamespace(model=SimpleNamespace(language_model=lm)))
    ids = torch.arange(25, dtype=torch.int64)
    spans = [Span(4, 6, (1, 4, 6)), Span(13, 8, (1, 4, 8))]
    pos = positions(25, spans)
    embeds = [torch.randn(s.count, 128, device="cuda", dtype=torch.bfloat16) for s in spans]
    stacks = [[torch.randn_like(e) for _ in range(3)] for e in embeds]
    h = lm.embed_tokens(ids.cuda())[None]
    visual_mask = torch.zeros(1, len(ids), dtype=torch.bool, device="cuda")
    for span, emb in zip(spans, embeds):
        h[:, span.start:span.start + span.count] = emb
        visual_mask[:, span.start:span.start + span.count] = True
    deep = [torch.cat([bundle[i] for bundle in stacks]) for i in range(3)]
    raw = []
    # Hook the actual pre-final-norm stream, avoiding hidden_states[-1] semantics that changed across releases.
    hook = lm.norm.register_forward_pre_hook(lambda mod, args: raw.append(args[0].clone()))
    try:
        lm(inputs_embeds=h, position_ids=pos[:, None].cuda().long(), use_cache=False,
           visual_pos_masks=visual_mask, deepstack_visual_embeds=deep)
    finally:
        hook.remove()
    got = te.forward_mm(ids, pos, spans, embeds, stacks)
    ref = raw[0][0]
    cos = float(F.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0))
    rel = float((got.float() - ref.float()).norm() / ref.float().norm())
    assert cos >= 0.999 and rel <= 0.05, (cos, rel)
    assert torch.equal(got, te.forward_mm(ids, pos, spans, embeds, stacks))
    return {"cos": cos, "rel_l2": rel}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    report = host_gate(args.comfy)
    if args.gpu:
        report.update(gpu_gate())
        report["multimodal_te"] = te_gate()
    print(json.dumps({"edit_gate": report, "pass": True}))


if __name__ == "__main__":
    main()
