# Multi-arch image (linux/amd64 and linux/arm64, e.g. DGX Spark).
# Build:  docker build -t trianglebench .
# Run:    docker run --rm --gpus all -v "$PWD/out:/out" -w /out trianglebench --label "my system"
#         docker run --rm --gpus all trianglebench --selftest
# Needs the NVIDIA Container Toolkit and a host driver that supports CUDA 13.0 or newer.
ARG CUDA_VERSION=13.0.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
WORKDIR /src
COPY Makefile .
COPY src ./src
ARG GIT_COMMIT=unknown
# Turing (sm_75) to Blackwell (sm_100 data center, sm_120 RTX 50, sm_121 DGX Spark GB10) plus PTX for newer GPUs
RUN make GIT_COMMIT=${GIT_COMMIT} ARCH="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
      -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 \
      -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121 \
      -gencode arch=compute_121,code=compute_121"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
COPY --from=build /src/trianglebench /usr/local/bin/trianglebench
ENTRYPOINT ["trianglebench"]
