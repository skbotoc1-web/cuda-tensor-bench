# Peak tensor-core throughput vs. datasheet

trianglebench measures one realistic workload. The tools in this folder answer a different question:
**how close does a GPU get to the TOPS/TFLOPS figure on its datasheet?** They are kept separate from
trianglebench and do not change its results.

| Tool | Library | What it measures |
|---|---|---|
| `tensorpeak` | cuBLASLt | dense GEMM: tf32, fp16, bf16, int8, fp8 (e4m3), nvfp4 (e2m1 + UE4M3 scale per 16 values); autotunes over all algorithm IDs x tiles x stages x swizzle |
| `sparsepeak` | cuSPARSELt | 2:4 structured-sparse GEMM: fp16, bf16, int8, fp8, fp4; reported as **effective (dense-equivalent)** throughput 2n³/t, the way NVIDIA quotes "with sparsity" |
| `cutlasspeak.py` | CUTLASS profiler | INT4 dense and INT4/INT8/FP16 2:4 on Ampere (cuBLASLt has no INT4 path) |
| `fusedtri.cu` | CUDA Tile + cuBLASLt | prototype: the trianglebench kernel (A·A, mask, row sum) fused into one CUDA Tile kernel vs. cuBLASLt GEMM + separate epilogue |
| `benchrun.py`, `gb10run.py` | nvidia-smi | telemetry wrappers: median power, max. temperature, SM clock per `RESULT` line (`gb10run.py` also works on the GB10, where `benchrun.py` gets no samples) |

Every tool checks its own results (host reference in double, or exact comparison) and has a negative test
that must fail: `TENSORPEAK_NEGTEST=1`, `SPARSEPEAK_NEGTEST=1`, `FUSEDTRI_NEGTEST=1` (exit code 2).

## Build

```bash
cd peak
make                                   # GPU in this machine; needs cuBLASLt, cuSPARSELt, CUDA >= 13.3 for fusedtri
make CUSPARSELT=$HOME/.local/opt/cusparselt   # cuSPARSELt from a tarball or the PyPI wheel nvidia-cusparselt-cu13
make tensorpeak sparsepeak             # without CUDA Tile (CUDA < 13.3)
```

`fusedtri` needs `nvcc -enable-tile` and the `tileiras` compiler (package `cuda-tileiras-13-x`).
Without it nvcc fails with `tileiras: not found`. On a host with an older toolkit (the DGX Spark ships CUDA 13.0),
build and run inside `nvidia/cuda:13.4.3-devel-ubuntu24.04`; it ran with driver 580.

## Run

```bash
GPUBENCH_MIN_S=12 python3 gb10run.py ./tensorpeak fp16,int8,fp8,nvfp4 8192
GPUBENCH_MIN_S=12 python3 gb10run.py ./sparsepeak fp16,int8,fp8,fp4 8192,16384
GPUBENCH_MIN_S=10 python3 benchrun.py ./fusedtri 4096,8192
```

Stop other GPU workloads first; the results below were taken on an otherwise idle GPU
(on the DGX Spark the vLLM server was stopped for the run).

## Results (22 Oct 2026)

### DGX Spark (GB10, sm_121) vs. "up to 1 PFLOP FP4 with sparsity"

n = 8192, sustained >= 12 s per value. Power is GPU power as reported by the driver (CPU share not included).

| Format | Library | Measured | Theory* | Share | vs. 1000 | W | °C | MHz | TFLOPS/W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 dense | cuBLASLt | 98.0 | 123.3 | 79 % | 9.8 % | 82.7 | 78 | 2509 | 1.18 |
| INT8 dense | cuBLASLt | 195.0 | 244.1 | 80 % | 19.5 % | 91.1 | 82 | 2483 | 2.14 |
| FP8 dense | cuBLASLt | 220.8 | 244.7 | 90 % | 22.1 % | 92.6 | 81 | 2489 | 2.38 |
| NVFP4 dense | cuBLASLt | 374.4 | 484.2 | 77 % | 37.4 % | 92.5 | 83 | 2463 | 4.05 |
| FP16 2:4 | cuSPARSELt | 43.8 | 238.0 | 18 % | 4.4 % | 38.8 | 72 | 2421 | 1.13 |
| INT8 2:4 | cuSPARSELt | 250.9 | 461.2 | 54 % | 25.1 % | 91.7 | 77 | 2346 | 2.74 |
| FP8 2:4 | cuSPARSELt | 255.5 | 457.5 | 56 % | 25.6 % | 92.8 | 81 | 2327 | 2.75 |
| FP4 2:4 | cuSPARSELt | 460.4 | 879.2 | 52 % | 46.0 % | 91.3 | 83 | 2236 | 5.04 |
| FP4 2:4, n = 16384 | cuSPARSELt | 494.2 | 850.9 | 58 % | 49.4 % | 91.3 | 82 | 2164 | 5.41 |

\*Theory at the measured clock, **derived, not published by NVIDIA**: 48 SMs x {1024 FP16, 2048 FP8/INT8,
4096 FP4} FLOP per clock per SM (same per-SM rates as other compute capability 12.x GPUs), x2 for 2:4.
This reproduces NVIDIA's 1 PFLOP FP4-sparse figure at about 2.54 GHz.

* **NVFP4 dense is the practical peak:** 374 TFLOPS at 4.05 TFLOPS/W, 1.7x FP8.
* **2:4 sparsity gains only 1.16-1.31x** instead of 2x: the sparse kernels hit the ~92 W GPU power cap
  and the clock drops from ~2490 to ~2200 MHz.
* **FP16 2:4 is slower than dense** with cuSPARSELt 0.10.0 on sm_121; avoid it.
* The best value is 494 TFLOPS, i.e. **about half of the advertised 1 PFLOP**.

### RTX A4000 (GA104, sm_86, 140 W) vs. datasheet

| Format | Library | Measured | Datasheet | Share | Clock-adjusted** | W | °C | MHz | TOPS/W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INT8 dense | cuBLASLt | 127.0 | 153.4 | 83 % | 92 % | 129 | 87 | 1410 | 0.99 |
| INT8 2:4 | cuSPARSELt | 163.1 | 306.8 | 53 % | 83 % | 140 | 85 | 1005 | 1.17 |
| INT8 2:4 | CUTLASS | 144.3 | 306.8 | 47 % | 78 % | 140 | 85 | 945 | 1.03 |
| INT4 dense | CUTLASS | 233.8 | 306.8 | 76 % | 93 % | 140 | 87 | 1275 | 1.68 |
| INT4 2:4 | CUTLASS | 295.7 | 613.4 | 48 % | 74 % | 140 | 84 | 1020 | 2.12 |
| FP16 dense | cuBLASLt | 65.4 | 76.7 | 85 % | 98 % | 124 | 84 | 1365 | 0.53 |
| FP16 2:4 | cuSPARSELt | 98.7 | 153.4 | 64 % | 79 % | 140 | 85 | 1275 | 0.71 |
| FP16 2:4 | CUTLASS | 94.1 | 153.4 | 61 % | 83 % | 140 | 85 | 1155 | 0.67 |

\*\*Relative to the 1560 MHz boost clock of the datasheet. The 140 W board limit, not the tensor cores, caps
the sparse and INT4 results. Note: the datasheet's 153.4 is FP16 *with* sparsity (INT8 dense).

### Fused CUDA Tile kernel vs. cuBLASLt + epilogue (trianglebench workload, INT8, density 0.05)

| GPU | n | unfused | fused | fused/unfused | W unfused/fused | °C | MHz |
|---|---:|---:|---:|---:|---:|---:|---:|
| GB10 | 4096 | 128.2 | 178.0 | 1.39x | 56.2 / 67.0 | 70 | 2535 |
| GB10 | 8192 | 167.2 | 158.5 | 0.95x | 64.2 / 62.0 | 71 | 2535 |
| GB10 | 16384 | 191.0 | 104.3 | 0.55x | 58.6 / 48.5 | 72 | 2535 |
| RTX A4000 | 4096 | 111.1 | 58.1 | 0.52x | 109.7 / 113.6 | 57 | 1620 / 1800 |
| RTX A4000 | 8192 | 124.8 | 58.2 | 0.47x | 108.7 / 116.9 | 67 | 1545 / 1770 |

Values in effective TOPS (2n³/t). Fusion pays off only while the matrix fits in L2 (GB10, n <= 4096);
for large n, cuBLASLt's GEMM wins. trianglebench therefore keeps the unfused path. Lessons for CUDA Tile
GEMMs: load the B tile row-wise and `ct::transpose` it (column access compiles to byte loads),
`assume_divisible` the size, and use a grouped block raster (GROUP = 8) for L2 reuse; 128x128x128 tiles were best.

Raw logs: [`results/`](results). Logs taken before the tools' output was translated to English contain
German column labels (Wert = value, Leerlauf = idle, Stichproben = samples, Drossel = throttle reasons).

## References

* NVIDIA CUDA Programming Guide, Release 13.4: Table 33 (tensor-core data types per compute capability),
  section 2.4 (CUDA Tile), section 5.1.2 (architecture-specific targets).
* NVIDIA DGX Spark product page: "up to 1 PFLOP at FP4 precision with sparsity".
* NVIDIA RTX A4000 datasheet.
