# Multi-arch image (linux/amd64 and linux/arm64, e.g. DGX Spark). Two variants from one file:
#
# 1) Command-line benchmark (default target)
#    docker build -t trianglebench .
#    docker run --rm --gpus all trianglebench --selftest
#    docker run --rm --gpus all --user $(id -u):$(id -g) -v "$PWD:/out" -w /out trianglebench --label "my system"
#
# 2) JupyterLab with benchmark, analysis notebook and reference data
#    docker build --target jupyter -t trianglebench:jupyter .
#    docker run --rm --gpus all --user $(id -u):$(id -g) -p 127.0.0.1:8888:8888 -v "$PWD/work:/work" trianglebench:jupyter
#
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

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS jupyter
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 python3-venv \
    && apt-get clean && rm -r /var/lib/apt/lists/*
COPY jupyter/requirements.txt /opt/trianglebench/jupyter/requirements.txt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -q -r /opt/trianglebench/jupyter/requirements.txt
# HOME in /tmp: the container also works with --user <any uid> and a read-only image
ENV PATH=/opt/venv/bin:$PATH HOME=/tmp TB_HOME=/opt/trianglebench
COPY --from=build /src/trianglebench /usr/local/bin/trianglebench
COPY analysis /opt/trianglebench/analysis
COPY results /opt/trianglebench/results
COPY notebooks /opt/trianglebench/notebooks
COPY jupyter/start.sh /usr/local/bin/start-jupyter
RUN mkdir -m 1777 /work
WORKDIR /work
EXPOSE 8888
CMD ["start-jupyter"]

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
COPY --from=build /src/trianglebench /usr/local/bin/trianglebench
ENTRYPOINT ["trianglebench"]
