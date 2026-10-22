# trianglebench - build with the CUDA toolkit (>= 12.9 for DGX Spark/GB10, tested with 13.0 and 13.4)
# make                 -> binary for the GPU in this machine (-arch=native, needs a visible GPU at build time)
# make ARCH="-gencode arch=compute_86,code=sm_86"   -> explicit architecture, e.g. for Docker or CI builds
# make cpu             -> trianglebench-cpu, CPU only (g++ with OpenMP + OpenBLAS: apt install g++ libopenblas-dev)
# Tested: x86_64 + RTX A4000 (sm_86), aarch64 + DGX Spark GB10 (sm_121)
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
ARCH ?= -arch=native
CPU_ARCH ?= -march=native
CXX ?= g++
GIT_COMMIT ?= $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
CXXFLAGS = -O3 -std=c++17 -DGIT_COMMIT=\"$(GIT_COMMIT)\" -DBUILD_ARCH="\"$(ARCH)\""
CPUFLAGS = -O3 -std=c++17 -fopenmp $(CPU_ARCH) -DGIT_COMMIT=\"$(GIT_COMMIT)\" -DBUILD_ARCH="\"$(CPU_ARCH)\""
LIBS = -lcublasLt -L$(CUDA)/lib64/stubs -lnvidia-ml -lpthread

trianglebench: src/trianglebench.cu Makefile
	$(NVCC) $(CXXFLAGS) $(ARCH) -o $@ $< $(LIBS)

cpu: trianglebench-cpu
trianglebench-cpu: src/trianglebench_cpu.cpp Makefile
	$(CXX) $(CPUFLAGS) -I/usr/include/x86_64-linux-gnu/openblas-pthread -I/usr/include/aarch64-linux-gnu/openblas-pthread \
	  -o $@ $< -lopenblas -lpthread

test: trianglebench
	./trianglebench --selftest

test-cpu: trianglebench-cpu
	./trianglebench-cpu --selftest

clean:
	rm -f trianglebench trianglebench-cpu

.PHONY: cpu test test-cpu clean
