# 1. Execution model: threads, warps, blocks, occupancy

## Theory

A kernel launch creates a **grid** of **thread blocks**. The hardware places each block on one
**streaming multiprocessor (SM)**. The block stays there until it finishes, and the order in which
blocks go to SMs is not guaranteed (Guide §1.2.2.1, §1.2.2.2). Inside a block, threads form **warps** of 32
consecutive threads. A warp executes in the SIMT style: one instruction stream, and each thread may
take its own branch (Guide §1.2.2.2).

Two points from the guide explain most of the performance behaviour:

1. *"Switching between warps incurs no cost"* (Guide §3.2.2.2). The registers of every resident warp
   stay on chip. Each cycle, a warp scheduler picks a warp that is ready and issues its next instruction.
   Latency is hidden by having **other** ready work, either other warps or independent instructions
   in the same warp.
2. Registers are allocated per thread, shared memory per block. A block only fits on an SM if
   `registers per thread x threads per block` fit into the register file (Guide §1.2.3.3).

**Occupancy** is the number of resident warps divided by the maximum. On both GPUs here the maximum is
1,536 threads = 48 warps per SM, with 65,536 32-bit registers per SM (measured with `cudaGetDeviceProperties`).

## In this repository

`k_m1_bitset` (method M1, [`src/trianglebench.cu`](../../src/trianglebench.cu)) counts common neighbours
with AND + population count:

```cuda
__global__ void __launch_bounds__(256) k_m1_bitset(const uint32_t* bits, int n, int W, u64* rowsum) {
  __shared__ uint32_t As[8][68], Bs[8][68];
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16, bi = blockIdx.y * 64, bj = blockIdx.x * 64;
  uint32_t acc[4][4] = {};
  ...
        for (int c = 0; c < 4; c++) acc[r][c] += __popc(a[r] & b[c]);
  ...
    for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);  // reduce the 16 threads sharing row i
    if (tx == 0 && s) atomicAdd(&rowsum[i], (u64)s);
```

The design decisions, each one tied to the theory:

| Decision | Reason |
|---|---|
| 256 threads per block, `__launch_bounds__(256)` | 8 warps per block. The bound tells the compiler the maximum block size, so it can budget registers for it. |
| Each thread computes a 4 x 4 output tile in `acc[4][4]` | 16 independent accumulators give **instruction-level parallelism**: the scheduler can issue the next POPC without waiting for the previous one. |
| Grid `dim3(n/64, n/64)`, one 64 x 64 output tile per block | At n = 8192 that is 16,384 blocks, far more than 48 SMs, so the tail effect is small. |
| `__shfl_xor_sync` over 16 lanes, then one `atomicAdd` per row | Warp shuffles exchange registers without shared memory. The 16 threads that share row `i` sit in the same half-warp, and the butterfly with offsets 8, 4, 2, 1 stays inside that half. One atomic per row and block instead of one per thread. |

`k_masked_rowsum` (the epilogue of M2-M4) uses the other common pattern: one block per row, a strided
loop over the columns, a warp-shuffle reduction, then a reduction over the 8 warps in shared memory.

## Measurement: occupancy is not the goal

Nsight Compute on the RTX A4000, n = 8192 (sections `Occupancy`, `LaunchStats`, `ComputeWorkloadAnalysis`;
command in [chapter 6](06-measurement-methodology.md)):

| Kernel | Registers / thread | Shared memory / block | Limiting factor | Theoretical occupancy | Achieved | Busiest pipe |
|---|---|---|---|---|---|---|
| `k_m1_bitset` (M1) | 48 | 4,352 B | registers: 5 blocks | 83.3 % | 82.8 % | XU 95.8 % |
| `k_masked_rowsum<int>` | 36 | 256 B | warps: 6 blocks | 100 % | 98.3 % | FP64 91.4 % |
| `ampere_sgemm_128x128_tn` (M2, cuBLASLt) | 118 | 16,896 B | registers: 2 blocks | 33.3 % | 33.2 % | (not recorded) |
| `cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn` (M4, cuBLASLt) | 222 | 73,728 B | registers and shared memory: 1 block | 16.7 % | 16.6 % | tensor (IMMA) 92.2 % |

**Derived** - the register limit of M1: 48 registers x 256 threads = 12,288 registers per block, and
65,536 / 12,288 = 5.3, so 5 blocks = 40 warps = 83.3 % of 48 warps. That is exactly what Nsight reports.

The INT8 tensor-core GEMM runs with **one** block of 8 warps per SM and still keeps the tensor pipe 92 %
busy. It hides latency inside each warp instead of across many warps: every thread holds a large
output tile in its 222 registers, and the kernel streams A through a multi-stage shared-memory
pipeline (in CUTLASS naming, `64x3` means a K-tile of 64 and 3 pipeline stages). M1 runs at 83 % occupancy and keeps its pipe 96 % busy.
Both kernels are close to the limit of the unit they use. Occupancy only needs to be high enough to
hide the latency; beyond that it does not help.

Rule of thumb from these numbers: look at **pipe utilisation** first (Nsight "Compute Workload
Analysis"), occupancy second. Low occupancy is only a problem if the pipes are idle and the warps
are stalled.

## Try it

```bash
nvcc -arch=native -std=c++17 --resource-usage -c src/trianglebench.cu -o /dev/null   # registers and shared memory per kernel
ncu --kernel-name k_m1_bitset --launch-count 1 --section Occupancy --section LaunchStats ./trianglebench --selftest
```

Profiling needs access to the GPU performance counters (root, or the driver option
`NVreg_RestrictProfilingToAdminUsers=0`).

## References

- CUDA Programming Guide 13.4, §1.2.2.1 Thread Blocks and Grids, §1.2.2.2 Warps and SIMT, §1.2.3.3 On-Chip Memory
  in GPUs, §3.2.2.2 Hardware Multithreading. <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- Nsight Compute Profiling Guide, sections Occupancy and Launch Statistics.
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
