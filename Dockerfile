# Build:  docker build -t trianglebench .
# Run:    docker run --rm --gpus all trianglebench            (needs the NVIDIA Container Toolkit)
#         docker run --rm --gpus all trianglebench --selftest
# The runtime image needs a host driver that supports CUDA 13.0 or newer.
ARG CUDA_VERSION=13.0.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
WORKDIR /src
COPY Makefile .
COPY src ./src
# Turing (sm_75) to Hopper (sm_90) plus PTX for newer GPUs
RUN make ARCH="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
      -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
      -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90"

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
COPY --from=build /src/trianglebench /usr/local/bin/trianglebench
ENTRYPOINT ["trianglebench"]
