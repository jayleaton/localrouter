# Gufo in rootless Podman

[Gufo](https://github.com/gufo-org/gufo) (MIT) is an inference engine for AMD Strix Halo (gfx1151): Qwen3.8 Flash-Next
and 27B, Qwen3-TTS, Qwen3-ASR, Qwen-Image, MiniMax H3 and more behind OpenAI-compatible servers.
`serve.sh NAME PORT [podman options] -- gufo serve ...` runs one of them in the foreground so a LocalRouter worker can
supervise it: the `openai_server` engine for chat, `tools/speech/openai_audio` for speech and transcription.

- The container is `lr-gufo-NAME`, published on `127.0.0.1:PORT` only; `--host 0.0.0.0 --port 8080` are added after
  `serve` inside the container. `TZ` is the host's zone (`$TZ`, else `/etc/timezone`, else the `/etc/localtime`
  link). The image has no zoneinfo, so when the host has `/usr/share/zoneinfo` it is mounted read-only there;
  otherwise the container stays on UTC.
- SIGTERM or SIGINT (the worker unloading) runs `podman stop` and waits, so the GPU memory is free when the script exits.
  A SIGKILL of the worker's process group (the daemon's, on abort or timeout) skips that trap, and conmon keeps the
  container running outside the group, so a watcher in its own session removes this run's container (by its
  `--cidfile` id, never a later run's of the same name) once the script is gone (`test_serve.sh`, stub podman; checked
  with real rootless Podman on the Strix Halo box: removed within the stop timeout).
  `--replace` removes a container left by anything else; `deploy/strix-halo/stop.sh` also stops any
  `lr-gufo-*` left over.
- `/dev/kfd` and `/dev/dri` are passed with the `video` and `render` groups by name: Gufo's docs use
  `--group-add keep-groups`, which needs the crun runtime, and many hosts run runc.
- The image is pinned by digest in `IMAGE` (`gufo-runtime:0.34.0@sha256:49b1fce4…`, Gufo engine 2845939b), the
  build these figures were measured on; podman checks the pulled manifest against the digest. `GUFO_IMAGE` overrides
  it with another `name@sha256:…` (a reference without a digest is refused unless `GUFO_ALLOW_UNPINNED=1`).
  `deploy/strix-halo/start.sh` passes the chosen image into the daemon's unit, so the workers run what bootstrap pulled.

Measured on a Radeon 8060S with 96 GiB GTT (2026-11-15): Flash-Next UD-Q4_K_XL with the MTP head peaks at 88 GiB GTT
serving 32k context with 2 sessions (pp512 1196 tok/s, decode 60-78 tok/s with MTP); Qwen3-TTS CustomVoice 6.5 GiB, about
220 ms to first audio, 2.3x real time; Qwen3-ASR 5.5 GiB, 14x real time.
