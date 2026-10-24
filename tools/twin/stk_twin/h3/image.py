"""PNG RGB decode and Pillow's uint8 Lanczos resize, ported to match image.zig. JPEG/WebP are unsupported."""
import math
import struct
import zlib
from pathlib import Path
import numpy as np
from .. import smath


def decode_png(path):
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("expected PNG; JPEG and WebP are not decoded")
    pos, payload, palette, header = 8, bytearray(), None, None
    seen_data, ended_data, done = False, False, False
    while pos < len(data):
        if pos+12 > len(data): raise ValueError("truncated PNG")
        n = struct.unpack_from(">I", data, pos)[0]
        kind, chunk = data[pos+4:pos+8], data[pos+8:pos+8+n]
        if pos+12+n > len(data) or zlib.crc32(kind+chunk)&0xffffffff != struct.unpack_from(">I", data, pos+8+n)[0]:
            raise ValueError("PNG chunk checksum")
        if header is None and kind != b"IHDR": raise ValueError("PNG header must be first")
        if kind == b"IHDR":
            if header is not None or n != 13: raise ValueError("invalid PNG header")
            header = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            if ended_data: raise ValueError("noncontiguous PNG data")
            seen_data = True; payload.extend(chunk)
        elif kind == b"PLTE":
            if palette is not None or seen_data or not n or n%3 or n>768: raise ValueError("invalid PNG palette")
            palette = np.frombuffer(chunk, np.uint8).reshape(-1, 3)
        elif kind == b"IEND":
            if n or not seen_data or pos+12 != len(data): raise ValueError("invalid PNG end")
            done = True; break
        elif kind[0]&32 == 0: raise ValueError(f"unsupported PNG chunk {kind!r}")
        if seen_data and kind != b"IDAT": ended_data = True
        pos += n+12
    if not done or header is None: raise ValueError("PNG missing header or end")
    w, h, bits, color, compression, filtering, interlace = header
    valid = bits in ((1,2,4,8,16) if color==0 else (1,2,4,8) if color==3 else (8,16))
    if not w or not h or w*h > 1<<28 or not valid or interlace or compression or filtering:
        raise ValueError("PNG requires noninterlaced supported bit depth")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color)
    if channels is None: raise ValueError("unsupported PNG color type")
    if color==3 and (palette is None or len(palette)>1<<bits): raise ValueError("invalid PNG palette")
    stride = (w*channels*bits+7)//8
    byte_width = max(1, channels*bits//8)
    dec = zlib.decompressobj()
    raw = dec.decompress(payload,h*(stride+1)+1)
    if not dec.eof or dec.unused_data or dec.unconsumed_tail or len(raw) != h*(stride+1):
        raise ValueError("PNG scanline or zlib length")
    rows = np.empty((h, stride), np.uint8)
    for y in range(h):
        filt = raw[y*(stride+1)]
        if filt > 4: raise ValueError("invalid PNG filter")
        for x in range(stride):
            a = int(rows[y,x-byte_width]) if x>=byte_width else 0
            b = int(rows[y-1,x]) if y else 0
            c = int(rows[y-1,x-byte_width]) if y and x>=byte_width else 0
            p = a+b-c
            pa,pb,pc = abs(p-a),abs(p-b),abs(p-c)
            predictor = (0,a,b,(a+b)//2,a if pa<=pb and pa<=pc else b if pb<=pc else c)[filt]
            rows[y,x] = (raw[y*(stride+1)+1+x]+predictor)&255
    if bits==16:
        if color==0: pixels=np.minimum(rows.view(">u2").reshape(h,w,1),255).astype(np.uint8)
        else: pixels = rows.reshape(h,w,channels,2)[...,0]
    elif bits==8: pixels = rows.reshape(h,w,channels)
    else:
        mask = (1<<bits)-1
        pixels = np.empty((h,w,1),np.uint8)
        for x in range(w): pixels[:,x,0]=(rows[:,x*bits//8]>>(8-bits-x*bits%8))&mask
    if color==3:
        if int(pixels.max())>=len(palette): raise ValueError("PNG palette index")
        return palette[pixels[...,0]].copy()
    if color in (0,4):
        if bits<8: pixels=(pixels.astype(np.uint16)*255//((1<<bits)-1)).astype(np.uint8)
        return np.repeat(pixels[...,:1],3,axis=2)
    return pixels[...,:3].copy()


def _coefficients(src, dst):
    scale = src/dst
    fscale = max(scale, 1.)
    support = 3*fscale
    result = []
    def sinc(x):
        return 1. if x == 0 else smath.sincos(x*math.pi)[0]/(x*math.pi)
    for i in range(dst):
        center = (i+0.5)*scale
        low, high = max(0, int(center-support+0.5)), min(src, int(center+support+0.5))
        values = []
        for j in range(low, high):
            x = (j-center+0.5)/fscale
            values.append(sinc(x)*sinc(x/3) if -3<=x<3 else 0.)
        total = sum(values)
        values = [v/total if total else v for v in values]
        result.append((low, np.array([int(v*(1<<22)+(0.5 if v>=0 else -0.5)) for v in values], np.int64)))
    return result


def resize_lanczos(rgb, width, height):
    h, w, c = rgb.shape
    if c != 3 or not width or not height: raise ValueError("invalid RGB canvas")
    cur = rgb.copy()
    if w != width:
        out = np.empty((h, width, 3), np.uint8)
        for x, (low, k) in enumerate(_coefficients(w, width)):
            out[:,x] = np.clip(((cur[:,low:low+len(k)].astype(np.int64)*k[None,:,None]).sum(axis=1)+(1<<21))>>22, 0, 255)
        cur = out
    if h != height:
        out = np.empty((height, width, 3), np.uint8)
        for y, (low, k) in enumerate(_coefficients(h, height)):
            out[y] = np.clip(((cur[low:low+len(k)].astype(np.int64)*k[:,None,None]).sum(axis=0)+(1<<21))>>22, 0, 255)
        cur = out
    return cur


def load_canvas(path, width, height):
    # ComfyUI decodes uint8/255, then truncates float32*255 before the node's Lanczos pass.
    rgb = decode_png(path)
    rgb = np.clip(rgb.astype(np.float32)/np.float32(255)*np.float32(255), 0, 255).astype(np.uint8)
    return resize_lanczos(rgb, width, height).astype(np.float32)/np.float32(255)
