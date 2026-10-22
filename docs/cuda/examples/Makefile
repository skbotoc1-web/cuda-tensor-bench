# Small experiments for docs/cuda. Each one is self-contained and prints English text.
# make              -> builds for the GPU in this machine
# make ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_121,code=sm_121"   -> explicit targets (CI)
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc
ARCH ?= -arch=native
PROGS = bank_conflicts popc_rate epilogue_fp64

all: $(PROGS)

%: %.cu Makefile
	$(NVCC) -O3 -std=c++17 $(ARCH) -o $@ $<

run: all
	./bank_conflicts
	./popc_rate
	./epilogue_fp64

clean:
	rm -f $(PROGS)

.PHONY: all run clean
