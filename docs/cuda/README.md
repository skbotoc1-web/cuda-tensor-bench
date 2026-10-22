# CUDA programming: theory meets this repository

These notes bridge the CUDA programming model and real measurements. Every chapter takes one idea from
the NVIDIA documentation, shows where the idea appears in the code of this repository, and then checks it
with a measurement on two GPUs:

| GPU | Compute capability | SMs | L2 | Registers / SM | Max threads / SM | Shared memory / SM |
|---|---|---|---|---|---|---|
| NVIDIA RTX A4000 (GA104) | 8.6 | 48 | 4 MB | 65,536 | 1,536 | 100 KB |
| NVIDIA GB10 (DGX Spark) | 12.1 | 48 | 24 MB | 65,536 | 1,536 | 100 KB |

The values in the table come from `cudaGetDeviceProperties` on both machines. Both GPUs have the same number
of SMs and the same per-SM limits. That makes the comparison a clean one: where results differ, the
cause is clock, memory, L2 size or the instruction mix, not the size of the chip.

## Chapters

1. [Execution model: threads, warps, blocks, occupancy](01-execution-model.md)
   Why `k_m1_bitset` runs at 83 % occupancy and a cuBLASLt INT8 kernel at 17 %, and why both are fast.
2. [Memory hierarchy: coalescing, shared-memory banks, L2](02-memory-hierarchy.md)
   The `[8][68]` padding in `k_m1_bitset`, measured with Nsight Compute, and why the fused CUDA Tile
   kernel only wins while the matrix fits in L2.
3. [Roofline and the real bottleneck](03-roofline-and-bottlenecks.md)
   Arithmetic intensity n/3, machine balance, and the pipe that actually limits each kernel, including
   a hidden FP64 bottleneck in the epilogue.
4. [Tensor cores and data types](04-tensor-cores-and-data-types.md)
   Why every format from FP32 down to FP8 gives the exact count, measured speed and energy per format
   on both GPUs, block-scaled FP4, and what 2:4 sparsity gives under a power limit.
5. [cuBLASLt in practice](05-cublaslt-in-practice.md)
   How the GEMM is set up (compute type, scale type, layout, workspace), and how far the heuristic's first
   choice is from the best of up to 400 configurations.
6. [Measuring correctly: time, energy, clocks, tools](06-measurement-methodology.md)
   Timing, warm-up, ordering, confidence intervals, the NVML energy counter, and the commands for
   Nsight Systems, Nsight Compute, `cuobjdump` and `compute-sanitizer`.

## Experiments

[`examples/`](examples) holds three small, self-contained programs. Each one isolates one effect from the
chapters and checks its own result:

| Program | Shows | Self-check |
|---|---|---|
| `bank_conflicts.cu` | 8-way shared-memory bank conflict vs. padded layout | exit code 1 if the bank analysis is wrong |
| `popc_rate.cu` | POPC instructions per clock per SM, the ceiling of method M1 | prints the rate, the SASS check is in the source |
| `epilogue_fp64.cu` | FP64 conversion as a hidden bottleneck in a memory-bound kernel | exit code 1 if the four variants disagree |

```bash
cd docs/cuda/examples
make          # builds for the GPU in this machine (-arch=native)
make run
```

The CI builds all three for `sm_86` and `sm_121` (job `compile`, step *build docs/cuda examples*).

## How claims are labelled

- **Guide** - stated in the NVIDIA CUDA Programming Guide, Release 13.4 (section number given), or in
  another NVIDIA document that is linked.
- **Measured** - measured on the GPUs above; the command or the result file is named.
- **Derived** - our own calculation from measured or documented numbers. The calculation is shown so it
  can be checked.

All measurements were taken on 22 October 2026 with CUDA 13.0 (GB10, driver 580.178.04) and CUDA 13.4
(A4000, driver 615.71.10), Nsight Compute 13.4 on the A4000. Raw benchmark data are in
[`results/`](../../results) and [`peak/results/`](../../peak/results).

## Sources

- NVIDIA, *CUDA Programming Guide*, Release 13.4: <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- NVIDIA, *CUDA C++ Best Practices Guide*: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- NVIDIA, *cuBLAS Library, cuBLASLt API*: <https://docs.nvidia.com/cuda/cublas/>
- NVIDIA, *Nsight Compute Profiling Guide*: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
- NVIDIA, *NVML API Reference*: <https://docs.nvidia.com/deploy/nvml-api/>
- S. Williams, A. Waterman, D. Patterson, *Roofline: an insightful visual performance model for multicore
  architectures*, Communications of the ACM 52(4), 2009. <https://doi.org/10.1145/1498765.1498785>
