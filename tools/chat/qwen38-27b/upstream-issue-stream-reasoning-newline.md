# Draft issue for ashhart/TensorFold (not posted)

**Title:** `serve`: streamed `reasoning_content` keeps a trailing `\n` that the non-streamed reply strips (v1.1.1)

## Summary

With reasoning on, the same greedy request returns a `reasoning_content` that differs by one trailing newline
depending on `stream`. The streamed deltas, joined, end with `\n`; `choices[0].message.reasoning_content` of the
non-streamed reply does not. `content`, `tool_calls`, `finish_reason` and `usage.completion_tokens` are identical, so
the generated tokens are the same; only the whole reply trims the end of the reasoning block. A client that compares
or caches by text sees two different replies to one request.

## Environment

- TensorFold v1.1.1, `tensorfold-1.1.1-linux-aarch64.tar.gz` (sha256 `2e0103a544e64354bfec65bd8a052ede17aa9be1f6a56d8e37c868af9139cb39`)
- NVIDIA GB10 (DGX Spark), CUDA backend
- `TensorFold/Qwen3.8-27B-MLX-4bit` @ `22d8d538154e0e5b6f8dcb1fcc74a60a0798104c`, drafter `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`
- `tensorfold-native serve TensorFold/Qwen3.8-27B-MLX-4bit --drafter z-lab/Qwen3.8-27B-DFlash2 --host 127.0.0.1 --port 8123 --name qwen --context 32768 --prompt-cache-gib 4 --parallel 1`

## Repro

```sh
BODY='{"model":"qwen","messages":[{"role":"user","content":"What is the capital of Australia? Answer in one sentence."}],"temperature":0,"seed":7,"max_tokens":1024}'
curl -s localhost:8123/v1/chat/completions -H 'content-type: application/json' -d "$BODY" \
  | jq -r -j '.choices[0].message.reasoning_content' > whole.txt
curl -sN localhost:8123/v1/chat/completions -H 'content-type: application/json' \
  -d "$(jq -c '. + {stream: true}' <<< "$BODY")" \
  | sed -n 's/^data: //p' | grep -v '^\[DONE\]' | jq -r -j '.choices[0].delta.reasoning_content // empty' > stream.txt
wc -c whole.txt stream.txt     # 598 whole.txt, 599 stream.txt
cmp whole.txt stream.txt       # cmp: EOF on whole.txt after byte 598
tail -c 1 stream.txt | od -c   # \n
```

Observed in the qualification run (the same flags, served as `--name qwen3.8-27b` on an ephemeral port; two greedy
whole runs per prompt gave identical bytes):

| Prompt | whole `reasoning_content` | stream, joined | Difference |
| --- | --- | --- | --- |
| "What is the capital of Australia? Answer in one sentence." | 598 B | 599 B | the stream ends with `\n` |
| "A train leaves at 09:40 and arrives at 13:15 the same day. How long is the trip in minutes? Give the number only." with `chat_template_kwargs.enable_thinking: true` | 351 B | 352 B | the stream ends with `\n` |

The last reasoning delta of the stream is `{"reasoning_content":"The capital of Australia is Canberra.\"\n"}`, followed
by the first content delta. Prefixes are byte-identical; `content` (`The capital of Australia is Canberra.`),
`finish_reason` and `completion_tokens` (155) match.

## Expected

The joined deltas equal the non-streamed field byte for byte: either both keep the newline before `</think>` or both
strip it (api.md's "Exact decoding" section says a drafted reply equals the `"draft": false` reply and a concurrent
stream equals its solo run; the same holds between `stream: true` and `false` for `content`, so it would be expected
for `reasoning_content` too).

## Notes

Found by LocalRouter's qualification run, which compares stream against whole for every field on the GB10
(`tools/chat/qwen38-27b/qualify.sh`). It records this as the named exception
`tensorfold-1.1.1-stream-reasoning-trailing-newline`, limited to trailing whitespace at the end of a streamed
`reasoning_content`, until a fixed release.
