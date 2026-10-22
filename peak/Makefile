# Peak-throughput tools (separate from trianglebench). See peak/README.md.
# make                     -> all three tools for the GPU in this machine
# make ARCH="-gencode arch=compute_121,code=sm_121"   -> explicit architecture (CI, Docker)
# CUSPARSELT=<prefix>      -> cuSPARSELt install with include/ and lib/ (default: apt package libcusparselt0-dev-cuda-13)
# fusedtri needs CUDA >= 13.3 with CUDA Tile (cuda_tile.h, nvcc -enable-tile and the tileiras compiler).
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
ARCH ?= -arch=native
CUSPARSELT ?=
# apt packages install to non-default paths (/usr/include/libcusparseLt/13, /usr/lib/<arch>-linux-gnu/libcusparseLt/13)
SPLT_APT = /usr/lib/$(shell uname -m)-linux-gnu/libcusparseLt/13
SPLT_FLAGS = $(if $(CUSPARSELT),-I$(CUSPARSELT)/include -L$(CUSPARSELT)/lib -Xlinker -rpath=$(CUSPARSELT)/lib,\
             -I/usr/include/libcusparseLt/13 -L$(SPLT_APT) -Xlinker -rpath=$(SPLT_APT))

all: tensorpeak sparsepeak fusedtri

tensorpeak: tensorpeak.cu Makefile
	$(NVCC) -O3 $(ARCH) -o $@ $< -lcublasLt

sparsepeak: sparsepeak.cu Makefile
	$(NVCC) -O3 $(ARCH) $(SPLT_FLAGS) -o $@ $< -lcusparseLt

fusedtri: fusedtri.cu Makefile
	PATH=$(CUDA)/bin:$$PATH $(NVCC) -O3 -std=c++20 -enable-tile $(ARCH) -o $@ $< -lcublasLt

clean:
	rm -f tensorpeak sparsepeak fusedtri

.PHONY: all clean
