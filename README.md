# trianglebench – CUDA cores vs. Tensor Cores on a real math problem

How many triangles does a random graph with 16,384 nodes and 67 million edges contain?
`trianglebench` answers this classic graph-theory question **four times, with four independent methods**,
two on the CUDA cores and two on the Tensor Cores. All answers must agree **for every single node**,
so every speed number in the output comes with a proof of correctness.

```
triangles(G) = trace(A³) / 6 = Σ_ij A_ij · (A·A)_ij / 6        A = 0/1 adjacency matrix
```

| Method | Unit | Arithmetic | How |
|---|---|---|---|
| M1 | CUDA cores | INT32 | own kernel: rows as bitsets, `(A·A)_ij = popcount(row_i AND row_j)` |
| M2 | CUDA cores | FP32 | cuBLASLt GEMM, FP32 *pedantic* (Tensor Cores disabled) |
| M3 | Tensor Cores | FP16 → FP32 | cuBLASLt GEMM, matrix multiply-accumulate `D = A·B + C` |
| M4 | Tensor Cores | INT8 → INT32 | cuBLASLt GEMM, integer matrix multiply-accumulate |

After counting, the floating-point side computes the global clustering coefficient (FP64) and the
mean local clustering coefficient (FP32) and compares both with random-graph theory
(`E[triangles] = C(n,3)·p³`, `E[clustering] = p`).

Phase 2 then runs **all four methods at the same time** in separate CUDA streams and reports whether the
units add up or just compete for the same power budget.

## Example: NVIDIA RTX A4000 (140 W, driver 615.71.10, CUDA 13.4)

Full output: [`results/rtx-a4000_n16384.txt`](results/rtx-a4000_n16384.txt)

```
graph G(n=16384, p=0.5000, seed=1): 67111410 edges, 91636293818 triangles (theory 91609192448, ratio 1.00030)
clustering: global (FP64) 0.500049, mean local (FP32) 0.500023, theory p = 0.500000

method                               s/solve  eq.TOPS   W avg  T max   MHz   TOPS/W  J/solve  check
M1 CUDA cores  INT32 bitset+popc      0.1177    74.74   139.5     58  1590    0.536     16.4  OK
M2 CUDA cores  FP32 GEMM              0.8267    10.64   131.7     66  1350    0.081    108.9  OK
M3 Tensor Cores FP16->FP32 GEMM       0.1345    65.41   123.9     70  1410    0.528     16.7  OK
M4 Tensor Cores INT8->INT32 GEMM      0.0685   128.35   122.3     73  1455    1.049      8.4  OK

all four together: 29.02 eq.TOPS, 136.5 W avg, sum of shares 0.91
```

What this shows:

* **Tensor Cores vs. CUDA cores on the same matrix product:** INT8 Tensor Cores solve the problem
  12× faster than the FP32 CUDA cores and need 13× less energy per solution (8.4 J vs. 108.9 J).
* **Algorithms still matter:** the bitset method on plain CUDA cores is faster than the FP16 Tensor Cores,
  because one 32-bit `popcount` replaces 32 multiply-adds of 0/1 values. It reaches 1.17 T popc/s,
  about 96 % of the A4000's popcount peak (48 SMs × 16 popc/clock × 1590 MHz, see the
  [CUDA Best Practices Guide, native arithmetic throughput](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)).
  The Tensor Cores still win with INT8.
* **Running everything at once does not add up:** the four methods share one 140 W power limit and
  the same SMs; together they deliver 91 % of what they achieve one after another.

`eq.TOPS` = `2n³ / time`: the work of one dense `A·A`, used as a common yardstick for all methods.
For M1 this is an *effective* rate, the native integer rate is printed separately.

## Build and run

Requirements: NVIDIA GPU with compute capability 7.5 or newer (Turing and later, INT8 Tensor Cores),
CUDA toolkit 12.x or 13.x with cuBLASLt and NVML headers (`cuda-nvml-dev`), Linux.

```bash
make                    # builds for the GPU in this machine (-arch=native)
./trianglebench --selftest
./trianglebench                         # n=16384, p=0.5, 8 s per phase
./trianglebench --n 8192 --p 0.1 --seconds 5
```

With Docker and the NVIDIA Container Toolkit:

```bash
docker build -t trianglebench .
docker run --rm --gpus all trianglebench --selftest
docker run --rm --gpus all trianglebench
```

Options: `--n` nodes (multiple of 256, needs about `19·n²` bytes of GPU memory: 5.1 GB for 16384),
`--p` edge probability, `--seed`, `--seconds` per phase, `--selftest`.
The exit code is `0` only if all methods agree.

## How correctness is guaranteed

* The graph comes from a 64-bit integer hash (splitmix64) of the node pair, identical on CPU and GPU.
  Same `--n`, `--p`, `--seed` → same graph on every machine.
* All intermediate values are integers: `(A·A)_ij ≤ n`, exactly representable in FP16 inputs (0/1),
  FP32 accumulators (up to 2²⁴) and INT32. The node limit is enforced.
* `--selftest` compares the per-node counts of all four methods with a CPU brute-force count over all node
  triples for five graphs, including the empty graph and the complete graph (`C(n,3)` triangles).
* Every benchmark run checks all four methods node by node against each other.

## Telemetry

Power (NVML energy counter, averaged over the measuring window), maximum GPU temperature and median SM clock
are sampled every 100 ms. Results depend on cooling, power limit and clocks; please include them when you
share numbers.

## Contributing results

Pull requests with `results/<gpu>_n<nodes>.txt` are welcome. Please run the default settings and add the full
output unchanged.

## License

MIT, see [LICENSE](LICENSE). Built by [BALTX.COM UAB](https://baltx.com).
