# trianglebench - build with the CUDA toolkit (>= 12.0, tested with 13.4)
# make                 -> binary for the GPU in this machine (needs a visible GPU at build time)
# make ARCH="-gencode arch=compute_86,code=sm_86"   -> explicit architecture, e.g. inside Docker builds
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
ARCH ?= -arch=native
CXXFLAGS = -O3 -std=c++17
LIBS = -lcublasLt -L$(CUDA)/lib64/stubs -lnvidia-ml -lpthread

trianglebench: src/trianglebench.cu
	$(NVCC) $(CXXFLAGS) $(ARCH) -o $@ $< $(LIBS)

test: trianglebench
	./trianglebench --selftest

clean:
	rm -f trianglebench

.PHONY: test clean
