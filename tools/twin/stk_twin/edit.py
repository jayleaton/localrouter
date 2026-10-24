"""Qwen-Image 2.1 edit host preparation. Single still images, 1..5 references; no GPU arithmetic here.
Pillow owns decode / premultiplied Lanczos. The future engine decoder shares src/media/png_decode.zig with i2v.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SYSTEM = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
TEMPLATE = SYSTEM + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
VISION = "<|vision_start|><|image_pad|><|vision_end|>"


@dataclass
class Reference:
    rgba: Image.Image
    rgb: Image.Image
    patches: torch.Tensor       # [grid_h * grid_w, 1536], fp32, merge-block order
    grid: tuple[int, int, int]  # (1, pixel_h / 16, pixel_w / 16)


@dataclass(frozen=True)
class Span:
    start: int
    count: int
    grid: tuple[int, int, int]


def dimensions(width: int, height: int, area: int = 1024 * 1024) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    ratio = width / height
    w = round(math.sqrt(area * ratio) / 32) * 32
    h = round(w / ratio / 32) * 32
    if min(w, h) < 32:
        raise ValueError("reference aspect ratio produces an empty grid")
    return w, h


def patchify(rgb: Image.Image) -> torch.Tensor:
    """Qwen3-VL still-image processor: fp32 rescale, normalize, duplicate temporal slot, merge-block order."""
    w, h = rgb.size
    if rgb.mode != "RGB" or w % 32 or h % 32:
        raise ValueError("vision image must be RGB with sides divisible by 32")
    x = np.asarray(rgb).astype(np.float32) * np.float32(1.0 / 255.0)
    x = (x - np.float32(0.5)) / np.float32(0.5)
    frames = np.stack([x.transpose(2, 0, 1)] * 2)
    # [time, temporal, C, block_h, merge_h, patch_h, block_w, merge_w, patch_w]
    p = frames.reshape(1, 2, 3, h // 32, 2, 16, w // 32, 2, 16)
    p = p.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(h * w // 256, 1536)
    return torch.from_numpy(p.copy())


def prepare(images) -> list[Reference]:
    images = list(images)
    if not 1 <= len(images) <= 5:
        raise ValueError("an edit needs 1 to 5 reference images")
    out = []
    for image in images:
        if isinstance(image, (str, Path)):
            with Image.open(image) as src:
                rgba = src.convert("RGBA")
        else:
            rgba = image.convert("RGBA")
        rgba = rgba.resize(dimensions(*rgba.size), Image.Resampling.LANCZOS)
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.getchannel("A"))
        w, h = rgba.size
        out.append(Reference(rgba, rgb, patchify(rgb), (1, h // 16, w // 16)))
    return out


def prompt_inputs(tokenizer, prompt: str, refs: list[Reference], drop: int):
    """Expand pad tokens, return CPU ids, spans, mrope positions and retained-text slots after system removal."""
    text = TEMPLATE.format(" ".join(f"<image{i + 1}>{VISION}" for i in range(len(refs))) + prompt)
    raw = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
    pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if raw.count(pad) != len(refs):
        raise ValueError("prompt contains an extra image placeholder")
    ids, spans, index = [], [], 0
    for token in raw:
        if token != pad:
            ids.append(token)
            continue
        grid = refs[index].grid
        count = grid[0] * grid[1] * grid[2] // 4
        spans.append(Span(len(ids), count, grid))
        ids.extend([pad] * count)
        index += 1
    if any(s.start < drop for s in spans):
        raise ValueError("system token drop overlaps an image")
    keep = torch.ones(len(ids), dtype=torch.bool)
    keep[:drop] = False
    slots = []
    for s in spans:
        slots.append(int(keep[:s.start].sum()))
        keep[s.start:s.start + s.count] = False
    return torch.tensor(ids, dtype=torch.int64), spans, positions(len(ids), spans), keep, tuple(slots)


def positions(length: int, spans: list[Span]) -> torch.Tensor:
    """[3,L] interleaved-mrope positions: text advances one, an image advances max(merged_h, merged_w)."""
    out = torch.empty(3, length, dtype=torch.float32)
    cursor = pos = 0
    for s in spans:
        t, h, w = s.grid
        h, w = h // 2, w // 2
        if t != 1 or s.count != h * w or s.start < cursor:
            raise ValueError("invalid still-image span")
        n = s.start - cursor
        out[:, cursor:s.start] = torch.arange(pos, pos + n)
        pos += n
        out[0, s.start:s.start + s.count] = pos
        out[1, s.start:s.start + s.count] = torch.arange(h).repeat_interleave(w) + pos
        out[2, s.start:s.start + s.count] = torch.arange(w).repeat(h) + pos
        cursor = s.start + s.count
        pos += max(h, w)
    out[:, cursor:] = torch.arange(pos, pos + length - cursor)
    return out


def prefix_layout(text_rows: int, ref_hw, slots):
    """Validated prefix plan: (text_start,text_end,ref_index), attention segments and next RoPE position.
    Segment (start,end,causal) reads keys [:end]; a causal segment's first query is at absolute start.
    Missing slots append references, as the published DiT does; supplied slots must be monotone and in range.
    """
    slots = tuple(slots)
    if len(slots) > len(ref_hw):
        raise ValueError("more image slots than references")
    slots += (text_rows,) * (len(ref_hw) - len(slots))
    if any(not 0 <= s <= text_rows for s in slots) or list(slots) != sorted(slots):
        raise ValueError("image slots must be monotone retained-text offsets")
    pieces, segments = [], []
    pos = length = start = 0
    for i, end in enumerate((*slots, text_rows)):
        n = end - start
        pieces.append((start, end, i if i < len(ref_hw) else None))
        if n:
            segments.append((length, length + n, True))
            length += n
            pos += n
        if i < len(ref_hw):
            h, w = ref_hw[i]
            if min(h, w) <= 0:
                raise ValueError("empty reference latent")
            segments.append((length, length + h * w, False))
            length += h * w
            pos += max(h, w)
        start = end
    return pieces, segments, pos
