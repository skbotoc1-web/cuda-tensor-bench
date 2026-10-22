# 3. Roofline and the real bottleneck

## Theory: the roofline model

A kernel needs `ops` arithmetic operations and moves `bytes` between DRAM and the chip. Its
**arithmetic intensity** is `AI = ops / bytes`. With a peak compute rate `P` and a memory bandwidth
`B`, the attainable rate is

```
attainable = min(P, AI x B)
```

(Williams, Waterman, Patterson 2009). The **machine balance** `P / B` is the intensity at which both
limits meet. Below it a kernel is memory-bound, above it compute-bound.

**Derived** for the two GPUs, with the measured dense INT8 tensor-core peak and the datasheet bandwidth:

| | INT8 peak (measured) | DRAM bandwidth (datasheet) | balance |
|---|---|---|---|
| RTX A4000 | 127 TOPS | 448 GB/s | ~283 op/B |
| GB10 | 195 TOPS | 273 GB/s | ~714 op/B |

## Arithmetic intensity of each kernel in trianglebench

**Derived**, minimum DRAM traffic, n = 8192:

| Kernel | ops | minimum bytes | AI | vs. balance |
|---|---|---|---|---|
| M4 INT8 GEMM `C = A*A` | 2n³ | A, B as INT8 + C as INT32 = 6n² | n/3 = 2,731 | far above: compute-bound |
| M3 FP16 GEMM | 2n³ | 2n² + 2n² + 4n² = 8n² | n/4 = 2,048 | compute-bound |
| M2 FP32 GEMM | 2n³ | 12n² | n/6 = 1,365 | compute-bound |
| M1 bitset AND + POPC | n³/32 POPC = 2n³ equivalent ops | 2 x n²/8 (bitset rows and columns) | 8n equivalent ops/B | extremely compute-bound |
| `k_masked_rowsum` epilogue | about n²/2 additions | 4n² (read C) + n²/8 (bitset) | ~0.12 op/B | far below: memory-bound |

So the roofline predicts: the GEMMs and M1 are limited by arithmetic units, the epilogue by DRAM
bandwidth. The next step is to check **which** unit, and whether the prediction holds.

## Which pipe is actually busy

Nsight Compute reports, for every execution pipe of the SM, how busy it was. RTX A4000, n = 8192:

| Kernel | Busiest pipe | Utilisation | DRAM throughput | Duration |
|---|---|---|---|---|
| `k_m1_bitset` (M1) | XU (the pipe Nsight attributes POPC to on sm_86) | 95.8 % | 15.8 % | 15.05 ms |
| `cutlass_80_tensorop_i16832gemm_s8...` (M4) | tensor (IMMA) | 92.2 % | 51.7 % | 7.89 ms |
| `k_masked_rowsum<int>` (epilogue) | **FP64** | 91.4 % | 62.2 % | 1.00 ms |

For XU and FP64 the table shows `sm__inst_executed_pipe_<pipe>` (issued instructions relative to the
peak issue rate). For the tensor pipe it shows `sm__pipe_tensor_cycles_active`: one IMMA instruction keeps
the pipe busy for several cycles, so the instruction metric reads only 23.0 % although the pipe is busy
92.2 % of the time.

The first two confirm the roofline. The third does not: the epilogue should be limited by DRAM, but
DRAM is only 62 % busy while the FP64 pipe is at 91 %.

### The hidden FP64 bottleneck

The epilogue converts each element of C with `llrint((double)C[...])`. That is simple and exact, but it
compiles to two FP64 instructions per element (`cuobjdump -sass trianglebench`, sm_121 build):

```
k_masked_rowsum<int>:    I2F.F64       (int32 -> double)
                         F2I.S64.F64   (double -> int64)
k_masked_rowsum<float>:  F2F.F64.F32   (float -> double)
                         F2I.S64.F64
```

On GPUs with a low FP64 rate, such as GA104, those two conversions per element cost more than loading
the 4 bytes. [`examples/epilogue_fp64.cu`](examples/epilogue_fp64.cu) times the same kernel with and
without the FP64 detour. All variants give identical sums, because the values are small integers:

| Variant | A4000 | GB10 |
|---|---|---|
| INT32 C, `llrint((double)c)` (as in trianglebench 1.3.0) | 0.852-0.853 ms, 325 GB/s | 1.11-1.17 ms |
| INT32 C, integer add | 0.754-0.755 ms, **367 GB/s** | 1.10-1.14 ms |
| FP32 C, `llrint((double)c)` | 0.868-0.881 ms, 314-319 GB/s | 1.10-1.14 ms |
| FP32 C, `__float2int_rn` (exact below 2²⁴) | 0.766-0.768 ms, 361 GB/s | 1.09-1.15 ms |

Each value is the mean of 20 launches; the table shows the range over 5 runs of the program. On the
A4000 the runs agree within 0.3 % (FP32 `llrint`: 1.5 %), on the GB10 they spread by 4-6 %.

**Derived:** on the A4000 the integer path needs 11 % less time and reaches 82 % of the 448 GB/s datasheet
bandwidth, so the kernel becomes memory-bound as the roofline predicted. On the GB10 the four variants
are within the run-to-run spread of each other: there the FP64 conversions are not the limit. For the
whole solve the gain is small, since the epilogue takes about 1 ms of the 9.4 ms INT8 solve on the
A4000. The lesson is general, though: a kernel can only be called memory-bound once the profiler shows
the memory system as the busiest unit.

## Measuring the ceiling of M1 directly

M1 does `n² x n/32` POPC instructions per solve, and each POPC replaces 32 multiply-adds (64 equivalent
operations). Its ceiling is therefore the POPC rate. The CUDA 13.4 Programming Guide no longer contains a
table of instruction throughputs, so [`examples/popc_rate.cu`](examples/popc_rate.cu) measures it with
`clock64()`:

| | A4000 (sm_86) | GB10 (sm_121) |
|---|---|---|
| POPC per clock per SM (measured) | **16.0** | **16.0** |
| XOR + IADD3 reference (measured, steps per clock per SM) | 58.7 | 32.1 |
| M1 at n = 8192, median SM clock during the run | 1,470 MHz | 2,489 MHz |
| M1 achieved POPC per clock per SM (derived from eqTOPS and clock) | 15.3 | 15.6 |
| share of the POPC ceiling | **95 %** | **98 %** |

The derivation for the A4000: `n² x n/32 = 1.72e10` POPC per solve, 68.9 eqTOPS means `2n³ / 68.9e12 =
16.0 ms` per solve, so `1.72e10 / 0.0160 s / (48 SMs x 1.470e9 Hz) = 15.3`.

M1 cannot get faster on these GPUs without doing fewer POPC. This also explains a result from the
benchmark: 16 POPC x 64 = **1,024 equivalent operations per clock per SM**, the same rate as dense FP16
on the tensor cores, and half of dense INT8 (2,048 per clock per SM). Both tensor rates are derived:
for the A4000 from the datasheet (76.7 TFLOPS dense FP16, 153.4 TOPS dense INT8, boost clock 1,560 MHz,
48 SMs), for the GB10 in [`peak/README.md`](../../peak/README.md). That is why M1 matches or beats FP16 tensor cores
(A4000 68.9 vs. 61.9 eqTOPS, GB10 119.3 vs. 86.3 eqTOPS) and loses to INT8 tensor cores.

## Where the roofline stops being enough: power

All four methods are compute-bound, so their speed depends on the clock, and the clock depends on the
power limit. On the A4000 NVML reported the reason `0x4` (software power cap) in every telemetry sample
of every method. The median SM clock varied by method: 1,470 MHz for M1 and M4, 1,410 for M3, 1,320
for M2 (the FP32 GEMM draws the most power, 140 W). A roofline drawn at the boost clock overestimates
every method. [Chapter 6](06-measurement-methodology.md) shows how this is measured.

## Try it

```bash
cd docs/cuda/examples && make && ./popc_rate && ./epilogue_fp64
cuobjdump -sass ../../../trianglebench | grep -E "I2F.F64|F2I.S64.F64"
ncu --kernel-name regex:"k_m1_bitset|k_masked_rowsum" --launch-count 2 \
    --metrics regex:sm__inst_executed_pipe_.*\.avg\.pct_of_peak_sustained_active$ \
    ./trianglebench --n 8192 --repeats 1 --seconds 0.2 --warmup 0
```

## References

- S. Williams, A. Waterman, D. Patterson, *Roofline*, CACM 52(4), 2009. <https://doi.org/10.1145/1498765.1498785>
- CUDA C++ Best Practices Guide, arithmetic instruction throughput. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- Nsight Compute Profiling Guide, Compute Workload Analysis and pipe metrics. <https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
- NVIDIA RTX A4000 datasheet (448 GB/s, 153.4 TOPS). <https://www.nvidia.com/en-us/design-visualization/rtx-a4000/>
