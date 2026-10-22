# 5. cuBLASLt in practice

## Theory: what cuBLASLt adds to cuBLAS

`cublasGemmEx` takes a matrix product and decides everything else for you. **cuBLASLt** ("lightweight")
splits that decision into objects you can inspect and change:

| Object | Holds |
|---|---|
| `cublasLtMatmulDesc_t` | compute type (how the arithmetic is done), scale type (type of alpha and beta), transposes, epilogue, scale factors |
| `cublasLtMatrixLayout_t` (one per matrix) | data type, rows, columns, leading dimension, order |
| `cublasLtMatmulPreference_t` | limits for the search, mainly the workspace size |
| `cublasLtMatmulAlgo_t` | one concrete kernel configuration: algorithm id, tile, pipeline stages, split-K, swizzle |

`cublasLtMatmulAlgoGetHeuristic` returns a ranked list of algorithms for a problem. You can take the
first one, or time several and keep the fastest. Both are done in this repository, and the difference is
measurable.

## In this repository: the `Gemm` struct

All three tensor-core methods and the FP32 reference go through one small wrapper,
[`src/trianglebench.cu`](../../src/trianglebench.cu), `struct Gemm`:

```cuda
bool init(int n, cudaDataType ab, cudaDataType c, cublasComputeType_t ct, const void* a, void* cbuf) {
  integer = (ct == CUBLAS_COMPUTE_32I); A = a; C = cbuf;
  LT(cublasLtCreate(&h)); CK(cudaMalloc(&ws, wsz));                              // wsz = 32 MiB workspace
  LT(cublasLtMatmulDescCreate(&op, ct, integer ? CUDA_R_32I : CUDA_R_32F));       // compute type, scale type
  cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
  LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
  LT(cublasLtMatrixLayoutCreate(&la, ab, n, n, n)); LT(cublasLtMatrixLayoutCreate(&lc, c, n, n, n));
  cublasLtMatmulPreference_t pref; LT(cublasLtMatmulPreferenceCreate(&pref));
  LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
  cublasLtMatmulHeuristicResult_t r; int got = 0;
  if (cublasLtMatmulAlgoGetHeuristic(h, op, la, la, lc, lc, pref, 1, &r, &got) != CUBLAS_STATUS_SUCCESS) got = 0;
  ...
  if (!got) return false;            // no algorithm for this type on this GPU -> reported as unsupported
  algo = r.algo;                     // then algo id, tile id and stages id are read back and written to the JSON
```

and the call itself:

```cuda
void run(cudaStream_t s) {
  const void* al = integer ? (const void*)&ione : (const void*)&fone;   // alpha/beta must have the scale type
  const void* be = integer ? (const void*)&izero : (const void*)&fzero;
  LT(cublasLtMatmul(h, op, al, A, la, A, la, be, C, lc, C, lc, &algo, ws, wsz, s));
}
```

Five details, each with the reason:

1. **Compute type decides the hardware.** The four methods differ only in `(input type, output type,
   compute type)`:

   | Method | input -> output | compute type | Kernel chosen on the A4000 (Nsight Compute) |
   |---|---|---|---|
   | M2 | FP32 -> FP32 | `CUBLAS_COMPUTE_32F_PEDANTIC` | `ampere_sgemm_128x128_tn` (CUDA cores) |
   | M3 | FP16 -> FP32 | `CUBLAS_COMPUTE_32F` | FP16 tensor-core kernel |
   | M4 | INT8 -> INT32 | `CUBLAS_COMPUTE_32I` | `cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn` |
   | Q_tf32 | FP32 -> FP32 | `CUBLAS_COMPUTE_32F_FAST_TF32` | TF32 tensor-core kernel |

   `_PEDANTIC` forbids reduced-precision shortcuts, so M2 is a true FP32 reference on CUDA cores
   (Nsight Compute shows the CUDA-core kernel `ampere_sgemm_128x128_tn`). TF32 tensor cores are
   requested explicitly with `_FAST_TF32`.
2. **The scale type follows the compute type.** For `CUBLAS_COMPUTE_32I` alpha and beta are
   `int32_t`, for the others `float`. Passing a `float` 1.0 to an integer GEMM gives a wrong result
   without any error. That is why `run()` picks the pointer from the same `integer` flag that `init()`
   used.
3. **TN layout.** cuBLASLt is column-major. With `TRANSA = T, TRANSB = N` the reduction dimension K
   is the contiguous one in both operands, the layout the INT8 tensor-core kernels use (`_tn` at the end
   of every kernel name above). Because `A` is symmetric, `Aᵀ x A = A x A`, so the transpose costs
   nothing here. For a general `A x B` you would store A row-major or transpose it once.
4. **Workspace.** Some algorithms, for example split-K, need scratch memory. The 32 MiB buffer is
   allocated once in `init()` and passed to every call, never allocated per call.
5. **"No algorithm" is a result.** If the heuristic returns nothing, the method is reported as
   unsupported instead of failing. That is how the tables in [chapter 4](04-tensor-cores-and-data-types.md)
   know that the A4000 has no FP8 path.

## Measurement: what the heuristic leaves on the table

trianglebench takes the **first** heuristic result. [`peak/tensorpeak.cu`](../../peak/tensorpeak.cu)
goes further: it collects 16 heuristic results, then every algorithm id x tile x stages x swizzle
combination that `cublasLtMatmulAlgoCheck` accepts (up to 400 candidates), times each one briefly
after a 3 s warm-up, and re-times the best 8 plus the heuristic choice (median of 3). It uses a 64 MiB
workspace. Results at n = 8192 from
[`peak/results/`](../../peak/results) (TFLOPS or TOPS):

| GPU, format | candidates | best | heuristic #1 | worst | heuristic / best |
|---|---:|---:|---:|---:|---:|
| GB10, INT8 | 58 | 189.61 | 144.75 | 13.26 | **76 %** |
| GB10, FP8 E4M3 | 400 | 220.98 | 211.42 | 13.13 | 96 % |
| GB10, NVFP4 | 8 | 377.62 | 343.99 | 100.89 | 91 % |
| GB10, FP16 | 347 | 95.78 | 94.40 | 6.41 | 99 % |
| A4000, INT8 | 55 | 128.28 | 125.25 | 18.02 | 98 % |
| A4000, FP16 | 262 | 65.42 | 64.68 | 6.28 | 99 % |
| A4000, TF32 | 112 | 35.23 | 35.23 | 2.46 | 100 % |

For the best configuration of each format the sustained run on the GB10 drew 82.7 W (FP16) to
92.6 W (FP8) at 78-83 °C and 2,463-2,509 MHz, and on the A4000 124-129 W at 84-87 °C and
1,365-1,410 MHz (full telemetry in [`peak/README.md`](../../peak/README.md)).

Lessons from the table:

- **The spread between algorithms is huge** (7x to 17x between best and worst). Choosing *some*
  valid algorithm is not enough; at least use the heuristic.
- **The heuristic is usually within 1-4 %, but not always.** For INT8 on the GB10 at n = 8192 it
  picks a configuration that reaches only 76 % of the best. At n = 16384 the same heuristic gets 97 %.
  This is why the INT8 path in trianglebench (heuristic #1) is *slower* than FP8 on the GB10
  (129.0 vs. 150.4 eqTOPS), although the hardware rate for INT8 and FP8 is the same.
- **When the GEMM runs many times with the same shape, autotune once.** The search above takes
  seconds; a benchmark or a training run with millions of identical calls pays that back quickly. Save
  the chosen `cublasLtMatmulAlgo_t` (it is a plain struct) and reuse it.

## Try it

```bash
cd peak && make tensorpeak
./tensorpeak fp16,int8 4096,8192     # usage: tensorpeak [formats] [sizes]; GPUBENCH_MIN_S sets the sustained run (default 5 s)
cd .. && ./trianglebench --n 8192 --repeats 1 --out t
grep -o '"cublaslt_algo":{[^}]*}' t.json    # algorithm id, tile id and stages id chosen by the heuristic
```

## References

- cuBLAS Library, *Using the cuBLASLt API*: `cublasLtMatmul`, `cublasLtMatmulAlgoGetHeuristic`,
  `cublasLtMatmulAlgoCheck`, compute types. <https://docs.nvidia.com/cuda/cublas/>
- CUDA Programming Guide 13.4, §5.1.3, text below Table 33 (use tensor cores through CUDA-X libraries or
  CUTLASS). <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- CUTLASS. <https://github.com/NVIDIA/cutlass>
