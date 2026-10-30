#!/usr/bin/env python3
"""Drama 3 voices for the macOS (Metal) engine: LocalRouter's codes as the native server's voice catalog.

LocalRouter keeps a voice as `drama3-voice/1` (voices/*.json: a clip's exact transcript and `vq_codes`, codebook-major,
10 rows of `frames` codes) and a clone's references as `[{text, vq_codes}]`. The Metal server built from pins-macos.json
takes neither in a request: it speaks only voices from its `--voices` catalog, each a `drama3-http-voice/1` file of
frame-major `codes` (one row of ten a frame, the semantic code first) with an attested `source`. Both come from the
same codec (codec.pth, one sha256 in both checkpoints) in the same codebook order, so a catalog row is a column of
vq_codes; nothing is re-encoded.

    voices_macos.py convert OUT_DIR voices/*.json   # each named voice, checked, to OUT_DIR/NAME.json

The catalog's own limits (the server refuses a file outside them at startup) are checked here first: 1 to 4
references, each 1 to 646 frames, codebook 0 in 0..4095 and codebooks 1 to 9 in 0..1023, transcripts of 1 to 4096
bytes without the prompt's structural tokens, a nonblank note of at most 4096 bytes, at most 256 KiB a file.
"""

import hashlib
import json
import os
import re
import struct
import sys

SOURCE_FORMAT = "drama3-voice/1"
CATALOG_FORMAT = "drama3-http-voice/1"
CODEBOOKS = 10
MAX_FRAMES = (30 * 44100 + 2047) // 2048  # 646: the server's max_frames for one reference (30 s)
MAX_REFERENCES = 4
MAX_TEXT_BYTES = 4096
MAX_FILE_BYTES = 256 * 1024
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
STRUCTURAL = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
SOURCE_KINDS = ("synthetic", "self", "consented")


class Invalid(ValueError):
    """A voice the catalog would refuse."""


def codes_sha256(vq_codes):
    """drama3-voice/1's codes_sha256: little-endian int32, codebook-major."""
    return hashlib.sha256(b"".join(struct.pack(f"<{len(row)}i", *row) for row in vq_codes)).hexdigest()


def check_vq_codes(vq_codes, what):
    """A [10, frames] grid of codes in the codec's ranges; returns frames."""
    if not isinstance(vq_codes, list) or len(vq_codes) != CODEBOOKS:
        raise Invalid(f"{what}: vq_codes must be {CODEBOOKS} rows, one a codebook")
    frames = len(vq_codes[0]) if isinstance(vq_codes[0], list) else 0
    if not 1 <= frames <= MAX_FRAMES:
        raise Invalid(f"{what}: {frames} frames; a catalog reference holds 1 to {MAX_FRAMES} (30 s)")
    for k, row in enumerate(vq_codes):
        top = 4096 if k == 0 else 1024
        if not isinstance(row, list) or len(row) != frames or \
                not all(type(c) is int and 0 <= c < top for c in row):
            raise Invalid(f"{what}: codebook {k} must be {frames} integers in 0..{top - 1}")
    return frames


def check_text(text, what):
    if not isinstance(text, str) or not text.strip():
        raise Invalid(f"{what}: the transcript must be nonblank")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise Invalid(f"{what}: the transcript is over {MAX_TEXT_BYTES} bytes")
    if any(t in text for t in STRUCTURAL):
        raise Invalid(f"{what}: the transcript holds a structural prompt token")


def frame_major(vq_codes):
    return [list(frame) for frame in zip(*vq_codes)]


def codebook_major(codes):
    return [list(row) for row in zip(*codes)]


def catalog(references, kind, note):
    """A drama3-http-voice/1 document from [{text, vq_codes}] (at most MAX_REFERENCES)."""
    if kind not in SOURCE_KINDS:
        raise Invalid(f"source kind must be one of {', '.join(SOURCE_KINDS)}")
    if not isinstance(note, str) or not note.strip() or len(note.encode("utf-8")) > MAX_TEXT_BYTES:
        raise Invalid(f"the source note must be nonblank and at most {MAX_TEXT_BYTES} bytes")
    if not isinstance(references, list) or not 1 <= len(references) <= MAX_REFERENCES:
        raise Invalid(f"a catalog voice holds 1 to {MAX_REFERENCES} references")
    refs = []
    for i, r in enumerate(references):
        if not isinstance(r, dict):
            raise Invalid(f"reference {i}: must be {{text, vq_codes}}")
        check_text(r.get("text"), f"reference {i}")
        check_vq_codes(r.get("vq_codes"), f"reference {i}")
        refs.append({"text": r["text"], "codes": frame_major(r["vq_codes"])})
    return {"format": CATALOG_FORMAT, "source": {"kind": kind, "note": note}, "references": refs}


def dumps(doc):
    """The file's exact bytes: compact, so the same voice is the same sha256 on every machine."""
    data = (json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(data) > MAX_FILE_BYTES:
        raise Invalid(f"the catalog file is {len(data)} bytes, over the server's {MAX_FILE_BYTES}")
    return data


def pack(references, limit=MAX_REFERENCES):
    """At most `limit` references from more: the adjacent pair with the fewest frames together is joined (transcripts
    with a space, codes end to end, as one take) until they fit, each join within MAX_FRAMES. LocalRouter's clone keeps
    up to 8 references; the catalog takes 4. None when they cannot be joined that far."""
    refs = [{"text": r["text"], "vq_codes": [list(row) for row in r["vq_codes"]]} for r in references]
    while len(refs) > limit:
        sizes = [len(refs[i]["vq_codes"][0]) + len(refs[i + 1]["vq_codes"][0]) for i in range(len(refs) - 1)]
        i = min(range(len(sizes)), key=sizes.__getitem__)
        if sizes[i] > MAX_FRAMES:
            return None
        a, b = refs[i], refs[i + 1]
        refs[i:i + 2] = [{"text": a["text"].rstrip() + " " + b["text"].lstrip(),
                          "vq_codes": [x + y for x, y in zip(a["vq_codes"], b["vq_codes"])]}]
    return refs


def named_voice(path):
    """A shipped drama3-voice/1 file, checked against its own codes_sha256 and frames, as a catalog document."""
    with open(path, "rb") as f:
        raw = f.read()
    doc = json.loads(raw)
    what = os.path.basename(path)
    if doc.get("format") != SOURCE_FORMAT:
        raise Invalid(f"{what}: format is {doc.get('format')!r}, not {SOURCE_FORMAT}")
    frames = check_vq_codes(doc.get("vq_codes"), what)
    if frames != doc.get("frames"):
        raise Invalid(f"{what}: {frames} frames of codes, the file says {doc.get('frames')}")
    if codes_sha256(doc["vq_codes"]) != doc.get("codes_sha256"):
        raise Invalid(f"{what}: its codes do not match its codes_sha256")
    source = doc.get("source") or {}
    if source.get("kind") != "model-invented":
        raise Invalid(f"{what}: source kind {source.get('kind')!r}; only model-invented voices are converted as synthetic")
    note = (f"Invented by Drama 3 itself, no person's voice (LocalRouter tools/speech/drama3/voices/SOURCES.md): "
            f"{source.get('note', '').strip()}. Codes from {what} (sha256 {hashlib.sha256(raw).hexdigest()}, "
            f"codes_sha256 {doc['codes_sha256']}), transposed to frame-major; not re-encoded.")
    return catalog([{"text": doc["text"], "vq_codes": doc["vq_codes"]}], "synthetic", note)


def convert(out_dir, paths):
    """Each named voice to OUT_DIR/NAME.json, written atomically; returns {file name: sha256}."""
    os.makedirs(out_dir, exist_ok=True)
    digests = {}
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if not NAME.match(name) or name.lower() == "default":
            raise Invalid(f"{name}: not a catalog voice name")
        data = dumps(named_voice(path))
        out = os.path.join(out_dir, name + ".json")
        with open(out + ".tmp", "wb") as f:
            f.write(data)
        os.replace(out + ".tmp", out)
        digests[name + ".json"] = hashlib.sha256(data).hexdigest()
    return digests


def main(argv):
    if len(argv) < 3 or argv[0] != "convert":
        print(__doc__.strip().split("\n\n")[2], file=sys.stderr)
        return 2
    try:
        for name, digest in sorted(convert(argv[1], argv[2:]).items()):
            print(f"{digest}  {name}")
    except (OSError, ValueError) as e:
        print(f"voices_macos.py: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
