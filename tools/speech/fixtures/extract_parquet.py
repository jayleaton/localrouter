#!/usr/bin/env python3
"""Extract the audio and transcripts from a LibriSpeech parquet file (needs pyarrow).

Usage: python3 -I extract_parquet.py IN.parquet OUT_DIR

OUT_DIR must be a new, empty directory. It receives <utterance id>.flac for every row plus
texts.json, a list of {"id", "text", "speaker_id", "chapter_id"} in file order. The parquet is
untrusted data: only the bytes of each audio blob are written, to names built from a validated id,
and nothing in it is ever executed. fetch.py runs this through `uv run --with pyarrow` when pyarrow
is not importable.
"""
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ID = re.compile(r"[0-9]+-[0-9]+-[0-9]+")


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    if any(out.iterdir()):
        sys.exit("output directory must be empty")
    rows = pq.read_table(src).to_pylist()
    texts = []
    for row in rows:
        uid = row["id"]
        if not ID.fullmatch(uid):
            sys.exit(f"unexpected utterance id {uid!r}")
        (out / f"{uid}.flac").write_bytes(row["audio"]["bytes"])
        texts.append({"id": uid, "text": row["text"], "speaker_id": row["speaker_id"], "chapter_id": row["chapter_id"]})
    (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"extracted {len(texts)} utterances")


if __name__ == "__main__":
    main()
