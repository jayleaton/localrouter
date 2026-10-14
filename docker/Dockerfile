# LocalRouter: `localrouter` (daemon, workers, client) on the CUDA base image.
# Build:  docker buildx build -f docker/Dockerfile --platform linux/arm64 -t localrouter:dev --load .
# Run:    docker run --gpus all -p 8190:8190 -v <models>:/models -v <data>:/data localrouter:dev
# The build stage runs on the builder's own platform and cross-compiles with Zig, so no emulation is needed there.
# Its nvcc builds the engines' fatbins (sm_121 for the Spark, sm_120 for Blackwell workstations): the NGC image the
# twin's kernels are compiled in, so the image's kernels are the twin's bit for bit (SASS checked, kernels/cuda/).
# Every input is pinned: base images by index digest, the Zig archive by sha256, the shipped apt packages by
# docker/ffmpeg-<arch>.lock (see the final stage).

ARG CUDA_BASE=nvidia/cuda:13.0.1-base-ubuntu24.04@sha256:f8ef28f579ea42a44b415d2c5d46f788e6a9b395c6c83f2929416e1fc192c143
ARG NVCC_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

FROM --platform=$BUILDPLATFORM ${NVCC_IMAGE} AS build
ARG BUILDARCH
ARG TARGETARCH
ARG ZIG_VERSION=0.17.0
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils git && rm -rf /var/lib/apt/lists/*
# the sha256s are ziglang.org/download/index.json's for 0.17.0; another ZIG_VERSION fails the check until they change
RUN arch=$([ "$BUILDARCH" = arm64 ] && echo aarch64 || echo x86_64) \
 && sum=$([ "$arch" = aarch64 ] && echo 9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8 \
                                || echo 1cbe9df9f27e6b78d14ccbca43b6703a404ef79ef1c463de901d7f088d4e2026) \
 && curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-${arch}-linux-${ZIG_VERSION}.tar.xz" \
 && echo "$sum  /tmp/zig.tar.xz" | sha256sum -c - \
 && tar -xJf /tmp/zig.tar.xz -C /opt && rm /tmp/zig.tar.xz \
 && ln -s /opt/zig-${arch}-linux-${ZIG_VERSION}/zig /usr/local/bin/zig
WORKDIR /src
COPY build.zig build.zig.zon ./
COPY build build
RUN zig build --fetch
COPY kernels kernels
COPY src src
COPY tests tests
# glibc 2.39 is Ubuntu 24.04's, the runtime base below.
RUN arch=$([ "$TARGETARCH" = arm64 ] && echo aarch64 || echo x86_64) \
 && zig build -Dtarget=${arch}-linux-gnu.2.39 -Doptimize=ReleaseSafe -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121,120 --prefix /out

FROM ${CUDA_BASE}
ARG TARGETARCH
# ffmpeg muxes MP4 (H.264 + AAC). The engines' kernels are embedded in `localrouter`; the driver comes from the host.
# The lock is ffmpeg's whole closure on this base, name=version (`apt-get install -s --no-install-recommends ffmpeg`
# in the base). Each entry installs at its exact version, and the build fails unless exactly the lock was added: a
# version the archive no longer carries, or a new dependency, is a deliberate lock update.
COPY docker/ffmpeg-${TARGETARCH}.lock /tmp/ffmpeg.lock
RUN dpkg-query -W -f '${Package}=${Version}\n' | sort > /tmp/before \
 && apt-get update && xargs -a /tmp/ffmpeg.lock apt-get install -y --no-install-recommends \
 && dpkg-query -W -f '${Package}=${Version}\n' | sort | comm -13 /tmp/before - | cmp - /tmp/ffmpeg.lock \
 && rm -rf /var/lib/apt/lists/* /tmp/before /tmp/ffmpeg.lock \
 && useradd --create-home --uid 10001 stk && mkdir -p /data /models /etc/localrouter && chown stk /data
COPY --from=build /out/bin/localrouter /usr/local/bin/localrouter
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER stk
EXPOSE 8190
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=5s CMD ["localrouter", "check", "health"]
ENTRYPOINT ["localrouter"]
# inside the container the daemon listens on every interface; what can reach it is the published port (compose.yaml)
CMD ["serve", "--host", "all", "--data", "/data"]
