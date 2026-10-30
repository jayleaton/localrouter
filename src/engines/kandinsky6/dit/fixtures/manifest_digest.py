"""python3 -I manifest_digest.py <transformer-header.json>: the digest layout.zig's manifest test expects.

The header is the JSON header of the checkpoint's transformer safetensors file (its first 8 + n bytes). Prints the
tensor count, the element count and sha256 over the names sorted bytewise, one line "name\tDTYPE\td0xd1...\n" each.
"""
import hashlib
import json
import math
import sys

h = json.load(open(sys.argv[1]))
h.pop("__metadata__", None)
lines = "".join(f"{k}\t{v['dtype']}\t{'x'.join(map(str, v['shape']))}\n" for k, v in sorted(h.items()))
print(len(h), sum(math.prod(v["shape"]) for v in h.values()), hashlib.sha256(lines.encode()).hexdigest())
