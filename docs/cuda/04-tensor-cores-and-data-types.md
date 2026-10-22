# 4. Tensor cores and data types

## Theory

A **tensor core** is a matrix unit inside each SM. One tensor-core instruction is issued by a whole
warp and computes a small matrix multiply-accumulate `D = A x B + C` on fragments that are spread over
the registers of the 32 threads. A CUDA core, in comparison, does one fused multiply-add per thread and
instruction. That is where the factor of 10 and more between M2 (FP32 on CUDA cores) and M4 (INT8 on
tensor cores) comes from.

Three statements from the CUDA Programming Guide 13.4 set the frame:

1. Which input types a GPU accelerates depends on its compute capability (Guide §5.1.3, Table 33,
   *Input Data Types Supported by Tensor Core Acceleration per Compute Capability*). Newer generations
   add narrower types: FP8 and FP6/FP4 do not exist on compute capability 8.6.
2. The guide *"strongly recommends"* using tensor cores through CUDA-X libraries such as cuBLAS, or
   through CUTLASS, rather than through inline PTX (Guide §5.1.3, text below Table 33). trianglebench
   follows that advice: every tensor-core path is a cuBLASLt call ([chapter 5](05-cublaslt-in-practice.md)).
3. From compute capability 9.0 on, some tensor-core instructions are **architecture-specific**. They
   need a compiler target with the `a` suffix (for example `compute_120a`), and code built for it runs
   only on exactly that compute capability (Guide §5.1.2.1, §5.1.2.3). Libraries ship such kernels for
   you; your own code built with `-arch=sm_86` or `-arch=sm_121` uses the baseline feature set.

**Derived** - the rate per clock doubles each time the input width halves. For the A4000 from the
datasheet (76.7 TFLOPS dense FP16, 153.4 TOPS dense INT8, 48 SMs, boost 1,560 MHz): 76.7e12 /
(48 x 1.56e9) = 1,024 FP16 operations and 2,048 INT8 operations per clock per SM. For the GB10 the
[`peak/README.md`](../../peak/README.md) uses 1,024 / 2,048 / 4,096 for FP16 / FP8 and INT8 / FP4,
which reproduces NVIDIA's 1 PFLOP sparse FP4 figure.

## Why every format gives the exact count here

Triangle counting needs `C = A x A` for a 0/1 adjacency matrix `A`. Two properties make every format
exact, and they are worth checking whenever you lower precision:

| Question | Answer for this problem |
|---|---|
| Are the **inputs** representable? | 0 and 1 are exact in every format, including FP4 E2M1. |
| Are the **products** exact? | 0 x 0, 0 x 1, 1 x 1: yes. |
| Is the **accumulator** wide enough? | Each entry of `C` is a sum of at most n - 1 = 8,191 ones. FP32 holds every integer up to 2²⁴ = 16,777,216 exactly, INT32 up to 2³¹. |
| Is the **output type** wide enough? | Only if it is FP32 or INT32. FP16 output would hold integers exactly only up to 2¹¹ = 2,048, BF16 only up to 2⁸ = 256. |

So trianglebench always accumulates and stores in FP32 or INT32, whatever the input type. The input
conversion is a small kernel, `k_quant` in [`src/trianglebench.cu`](../../src/trianglebench.cu). The
bit patterns of 1.0 are written out by hand for the 8-bit and 4-bit float formats:

```cuda
__device__ inline uint8_t q_e2m1(bool one) { return one ? 0x2 : 0x0; }   // E2M1: 0.0 = 0b000, 1.0 = 0b010
__device__ inline uint8_t q_e4m3(bool one) { return one ? 0x38 : 0x00; } // E4M3: 1.0 = 0b0'0111'000 = 0x38
__device__ inline uint8_t q_e5m2(bool one) { return one ? 0x3C : 0x00; } // E5M2: 1.0 = 0b0'01111'00
...
    if (fmt == CUDA_R_4I) { ((uint8_t*)out)[o] = (uint8_t)((e1 ? 1 : 0) | (e0 ? 0x10 : 0)); }  // two 4-bit values per byte
```

Each thread handles a pair of elements, because the 4-bit formats pack two values into one byte.

With weighted graphs or neural-network weights this argument does not hold, and narrow formats do
lose accuracy. That is why FP8 and FP4 GEMMs in practice come with scale factors (next section).

## Measurement: one problem, every data type

`./trianglebench --n 8192 --quant all` runs the same `C = A x A` with every input format that
cuBLASLt supports on the GPU, and compares the per-row triangle counts with the INT8 result, which
`--selftest` checks against a CPU count. Values from
[`results/rtx-a4000_quant.json`](../../results/rtx-a4000_quant.json) and
[`results/dgx-spark-gb10_quant.json`](../../results/dgx-spark-gb10_quant.json), n = 8192, mean of 5
repetitions of 5 s each (TOPS/W = mean of the per-repetition ratios, `eq_tops_per_watt`):

RTX A4000 (sm_86, 140 W power limit):

| Input -> accumulator | eqTOPS | W | max °C | median MHz | TOPS/W | J per solve | exact |
|---|---:|---:|---:|---:|---:|---:|---|
| FP32 -> FP32 (CUDA cores) | 10.21 | 137.0 | 84 | 1320 | 0.075 | 14.898 | yes |
| TF32 -> FP32 | 32.73 | 131.6 | 85 | 1455 | 0.249 | 4.473 | yes |
| FP16 -> FP32 | 61.08 | 121.0 | 84 | 1410 | 0.505 | 2.197 | yes |
| BF16 -> FP32 | 63.16 | 137.7 | 85 | 1470 | 0.459 | 2.419 | yes |
| INT8 -> INT32 | 114.00 | 124.9 | 85 | 1485 | 0.913 | 1.212 | yes |
| FP8 E4M3 / E5M2, INT4, FP4 | no cuBLASLt algorithm | | | | | | |

GB10 (sm_121):

| Input -> accumulator | eqTOPS | W | max °C | median MHz | TOPS/W | J per solve | exact |
|---|---:|---:|---:|---:|---:|---:|---|
| FP32 -> FP32 (CUDA cores) | 13.72 | 81.5 | 81 | 2496 | 0.168 | 6.634 | yes |
| TF32 -> FP32 | 39.44 | 77.6 | 81 | 2502 | 0.508 | 2.193 | yes |
| FP16 -> FP32 | 79.83 | 73.6 | 81 | 2502 | 1.085 | 1.017 | yes |
| BF16 -> FP32 | 79.63 | 80.4 | 82 | 2502 | 0.991 | 1.114 | yes |
| FP8 E4M3 -> FP32 | 150.42 | 79.9 | 82 | 2483 | 1.882 | 0.587 | yes |
| INT8 -> INT32 | 129.03 | 62.1 | 82 | 2502 | **2.077** | **0.532** | yes |
| FP8 E5M2, INT4, FP4 E2M1 (unscaled) | no cuBLASLt algorithm | | | | | | |

*eqTOPS* = 2n³ / time per solve, the same yardstick for every method. Watts are the power reported by
NVML, J per solve comes from the NVML energy counter ([chapter 6](06-measurement-methodology.md)).
On the GB10 NVML reports no clock-event reasons, so a missing throttle flag there means nothing.

What the tables teach:

- **Tensor cores against CUDA cores** on the same product: INT8 is 11.2x faster than FP32 on the A4000
  and 9.4x on the GB10, and needs about 12x less energy per solve on both (derived from the tables).
- **TF32 is the cheap upgrade for FP32 code**: same FP32 data in memory, 3.2x (A4000) and 2.9x (GB10)
  faster. It is enabled only if you ask for it (`CUBLAS_COMPUTE_32F_FAST_TF32`, [chapter 5](05-cublaslt-in-practice.md)).
- **Faster is not always more efficient.** On the GB10 FP8 is 17 % faster than INT8, but INT8 is
  10 % more efficient (2.08 vs. 1.88 TOPS/W), because it draws 62 W instead of 80 W. That the INT8 GEMM
  is slower than FP8 at all is a library effect, explained in [chapter 5](05-cublaslt-in-practice.md).
- **Hardware support is not library support.** The A4000 tensor cores accelerate INT4: the CUTLASS
  kernel in `peak/` reaches 233.8 TOPS dense. cuBLASLt simply offers no INT4 algorithm. And FP4 on the
  GB10 only works with **block scaling**: NVFP4 needs one UE4M3 scale factor per 16 elements. Without
  scales (as in `--quant fp4e2m1`) there is no algorithm; with scales (`peak/tensorpeak.cu`) cuBLASLt
  reaches 374.4 TFLOPS dense. For E5M2 x E5M2 cuBLASLt returned no algorithm on the GB10.

The scale setup in [`peak/tensorpeak.cu`](../../peak/tensorpeak.cu) is a scale tensor per operand
plus four attributes on the matmul descriptor. A scale of 1.0 (UE4M3 byte `0x38`) in every block keeps
the 0/1 values unchanged:

```cuda
CK(cudaMemset(SA, 0x38, sfbytes)); CK(cudaMemset(SB, 0x38, sfbytes));   // roundup(n,128) x roundup(n/16,4) bytes
...
if (p.bsm >= 0) { int32_t m = p.bsm;   // NVFP4: CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3, one scale per 16 elements
LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &m, sizeof m));
LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &m, sizeof m));
LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &SA, sizeof SA));
LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &SB, sizeof SB));
```

## A note on 2:4 structured sparsity

Tensor cores since compute capability 8.0 can skip zeros if exactly two of every four values along K
are zero, which doubles the nominal rate. Measured with cuSPARSELt at n = 8192
([`peak/README.md`](../../peak/README.md)): the gain over dense is 1.51x for FP16 and 1.28x for INT8 on
the A4000, and 1.16x to 1.29x for INT8, FP8 and FP4 on the GB10, never the nominal 2x. In these cases
the GPU ran into its power limit and the clock dropped (A4000 INT8 2:4 at 140 W and 1,005 MHz, GB10 FP4
2:4 at 91 W and 2,236 MHz). FP16 2:4 on the GB10 was even slower than dense (43.8 vs. 98.0 TFLOPS with
cuSPARSELt 0.10.0).
A sparse adjacency matrix does not have this regular 2:4 pattern, so trianglebench cannot use it.

## Try it

```bash
./trianglebench --n 8192 --quant all --out my_quant      # every supported input format, with exactness check
./trianglebench --n 4096 --quant fp16,bf16,int8 --repeats 3
```

## References

- CUDA Programming Guide 13.4, §5.1.3 Features and Technical Specifications (Table 33), §5.1.2.1 Architecture-Specific
  Features, §5.1.2.3 Feature Set Compiler Targets. <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- cuBLAS Library, cuBLASLt API (data types, block scaling). <https://docs.nvidia.com/cuda/cublas/>
- cuSPARSELt documentation. <https://docs.nvidia.com/cuda/cusparselt/>
- NVIDIA RTX A4000 datasheet. <https://www.nvidia.com/en-us/design-visualization/rtx-a4000/>
