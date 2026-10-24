"""PNG helpers for the Jarvis smoke's edit checks (stdlib only: 8-bit RGB or RGBA, not interlaced).

  python3 -I tools/jarvis/png_check.py size FILE            prints WIDTH HEIGHT
  python3 -I tools/jarvis/png_check.py mask W H OUT         writes an RGBA mask: left half clear (the edit), right half opaque
  python3 -I tools/jarvis/png_check.py keeps INPUT OUTPUT   exit 0 when OUTPUT has INPUT's size and its right half's RGB
                                                            equals INPUT's exactly; prints what differs
"""
import struct, sys, zlib


def read(path):
    data = open(path, "rb").read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path}: not a PNG")
    pos, idat, head = 8, b"", None
    while pos < len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            head = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        pos += 12 + n
    w, h, depth, color, _, _, interlace = head
    if depth != 8 or color not in (2, 6) or interlace:
        raise SystemExit(f"{path}: only 8-bit RGB or RGBA, not interlaced (depth {depth}, color type {color})")
    bpp = 3 if color == 2 else 4
    raw, stride = zlib.decompress(idat), w * bpp
    rows, prev, i = [], bytearray(stride), 0
    for _ in range(h):
        f, line = raw[i], bytearray(raw[i + 1:i + 1 + stride])
        i += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if f == 1:
                line[x] = (line[x] + a) & 255
            elif f == 2:
                line[x] = (line[x] + b) & 255
            elif f == 3:
                line[x] = (line[x] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    return w, h, bpp, rows


def write(path, w, h, rows):  # RGBA rows
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


cmd = sys.argv[1]
if cmd == "size":
    w, h, _, _ = read(sys.argv[2])
    print(w, h)
elif cmd == "mask":
    w, h = int(sys.argv[2]), int(sys.argv[3])
    row = bytes([0, 0, 0, 0]) * (w // 2) + bytes([255, 255, 255, 255]) * (w - w // 2)
    write(sys.argv[4], w, h, [row] * h)
elif cmd == "keeps":
    iw, ih, ib, irows = read(sys.argv[2])
    ow, oh, ob, orows = read(sys.argv[3])
    if (iw, ih) != (ow, oh):
        print(f"output {ow}x{oh}, input {iw}x{ih}")
        sys.exit(1)
    worst, differ, total = 0, 0, 0
    for y in range(ih):
        for x in range(iw // 2 + (iw % 2), iw):
            d = max(abs(irows[y][x * ib + c] - orows[y][x * ob + c]) for c in range(3))
            worst, differ, total = max(worst, d), differ + (d > 0), total + 1
    print(f"right half {differ} of {total} pixels differ, max difference {worst}")
    sys.exit(0 if differ == 0 else 1)
else:
    raise SystemExit(f"png_check.py: unknown command {cmd}")
