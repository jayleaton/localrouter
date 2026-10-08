# LocalRouter: `localrouter` (daemon, workers, client) on the CUDA base image.
# Build:  docker buildx build -f docker/Dockerfile --platform linux/arm64 -t localrouter:dev --load .
# Run:    docker run --gpus all -p 8190:8190 -v <models>:/models -v <data>:/data localrouter:dev
# The build stage runs on the builder's own platform and cross-compiles with Zig, so no emulation is needed there.
# Its nvcc builds the engines' fatbins (sm_121 for the Spark, sm_120 for Blackwell workstations): the NGC image the
# twin's kernels are compiled in, so the image's kernels are the twin's bit for bit (SASS checked, kernels/cuda/).

ARG CUDA_BASE=nvidia/cuda:13.0.1-base-ubuntu24.04
ARG NVCC_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3

FROM --platform=$BUILDPLATFORM ${NVCC_IMAGE} AS build
ARG BUILDARCH
ARG TARGETARCH
ARG ZIG_VERSION=0.17.0
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils git && rm -rf /var/lib/apt/lists/*
RUN arch=$([ "$BUILDARCH" = arm64 ] && echo aarch64 || echo x86_64) \
 && curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-${arch}-linux-${ZIG_VERSION}.tar.xz" | tar -xJ -C /opt \
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
# ffmpeg muxes MP4 (H.264 + AAC). The engines' kernels are embedded in `localrouter`; the driver comes from the host.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/* \
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
