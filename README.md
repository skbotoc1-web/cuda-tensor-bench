# trianglebench – CUDA cores vs. Tensor Cores on a real math problem

How many triangles does a random graph with 16,384 nodes and 67 million edges contain?
`trianglebench` answers this classic graph-theory question **four times, with four independent methods**,
two on the CUDA cores and two on the Tensor Cores. All answers must agree **for every single node**,
so every speed number comes with a proof of correctness. Results are written as JSON plus a raw
telemetry CSV, ready for statistical analysis.

```
triangles(G) = trace(A³) / 6 = Σ_ij A_ij · (A·A)_ij / 6        A = 0/1 adjacency matrix
```

| Method | Unit | Arithmetic | How |
|---|---|---|---|
| M1 | CUDA cores | INT32 | own kernel: rows as bitsets, `(A·A)_ij = popcount(row_i AND row_j)` |
| M2 | CUDA cores | FP32 | cuBLASLt GEMM, FP32 *pedantic* (Tensor Cores disabled) |
| M3 | Tensor Cores | FP16 → FP32 | cuBLASLt GEMM, matrix multiply-accumulate `D = A·B + C` |
| M4 | Tensor Cores | INT8 → INT32 | cuBLASLt GEMM, integer matrix multiply-accumulate |

After counting, the floating-point side computes the global clustering coefficient (FP64) and the mean
local clustering coefficient (FP32) and compares both with random-graph theory
(`E[triangles] = C(n,3)·p³`, `E[clustering] = p`). A second phase runs **all four methods at the same
time** in separate CUDA streams and measures whether the units add up or compete for power and SMs.

Tested platforms: **x86_64 + RTX A4000** (Ampere, sm_86, PCIe, 140 W) and
**aarch64 + NVIDIA DGX Spark** (GB10 Grace Blackwell, sm_121, unified memory).

## Results: RTX A4000 vs. DGX Spark (GB10)

10 repetitions × 5 s per method, mean ± 95 % confidence interval (Student t), sustained load after a
10 s warm-up. Raw data: [`results/`](results/).

```
                              RTX A4000 (140 W)           DGX Spark GB10            GB10 / A4000
n = 8192                      eq.TOPS        W   TOPS/W   eq.TOPS        W   TOPS/W   TOPS   TOPS/W
M1 CUDA INT32 bitset+popc    68.90±0.09  138.8   0.497  119.30±0.09   88.1   1.355   1.73    2.73
M2 CUDA FP32 GEMM            10.15±0.01  140.4   0.072   14.81±0.03   88.0   0.168   1.46    2.33
M3 Tensor FP16->FP32         61.93±0.05  124.2   0.499   86.30±0.18   78.7   1.097   1.39    2.20
M4 Tensor INT8->INT32       117.35±0.07  123.5   0.950  132.54±0.13   63.8   2.077   1.13    2.19
all four in parallel         29.46±0.17  137.7   0.214   42.98±0.10   80.0   0.537   1.46    2.51
median SM clock / T max      1320-1470 MHz / 85 °C       2489-2502 MHz / 83 °C
```

Both systems found the identical 11,456,628,772 triangles (n = 8192) and 1,431,534,897 (n = 4096).
Every difference in the table is significant (Welch t-test, p < 1e-12); the exact values are produced by
`analysis/summarize.py --compare`.

What the data shows:

* **Tensor Cores vs. CUDA cores on the same matrix product:** INT8 Tensor Cores are 12.2× faster than FP32
  CUDA cores on the A4000 (n = 16384) and 8.9× faster on the GB10 (n = 8192). Energy per solution drops by
  12.4× on the GB10 (6.61 J → 0.53 J).
* **Algorithms still matter:** the bitset method on plain CUDA cores beats the FP16 Tensor Cores on both
  GPUs, because one 32-bit `popcount` replaces 32 multiply-adds of 0/1 values. On the A4000 it runs at
  1.08 T popc/s, 95 % of the popcount peak (48 SMs × 16 popc/clock × 1470 MHz, see
  [CUDA Best Practices Guide, native arithmetic throughput](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)).
* **Running everything at once does not add up:** the sum of throughput shares stays between 0.92 and
  1.01 (0.99 on both GPUs at n = 8192) – the methods share one power budget and the same SMs, so
  parallel execution gives no measurable gain.
* **The GB10 is about 2.2–2.7× more energy-efficient** than the A4000 and faster on every method, but its
  lead on INT8 Tensor Cores is small (1.13×).
* The A4000 was limited by its power cap (`SW power cap` in 100 % of the telemetry samples), not by
  temperature.

`eq.TOPS` = `2n³ / time`: the work of one dense `A·A`, used as a common yardstick for all methods.
For M1 this is an *effective* rate; the native popcount rate is printed separately (`eq.TOPS / 64`).

### Limitations of these reference runs

* **DGX Spark:** a vLLM inference server was loaded but idle on the same GPU during the run
  (95 GB of unified memory in use, no requests). NVML on the GB10 reports no power limit, no memory clock
  and no clock-event reasons; whether its power figure covers the GPU only or a larger part of the
  superchip is not documented here and should be confirmed before comparing watts across platforms.
* **RTX A4000:** sustained runs reach 85 °C; short runs on a cold card are up to about 5 % faster.
* cuBLASLt picks its GEMM kernel by heuristic (logged in the JSON as `cublaslt_algo`), not by exhaustive
  search; a tuned kernel can be a few percent faster. On the GB10 the INT8 path used the same algorithm ID
  as on Ampere.
* The GB10 was not run at n = 16384 because of the vLLM memory reservation.

## Build and run

Requirements: Linux (x86_64 or aarch64), NVIDIA GPU with compute capability 7.5 or newer,
CUDA toolkit 12.9+ (13.x recommended) with cuBLASLt and NVML headers (`cuda-nvml-dev-*` package).

```bash
make                         # builds for the GPU in this machine (-arch=native)
./trianglebench --selftest   # all four methods vs. CPU brute force, exit code 0 = passed
./trianglebench --n 4096,8192 --repeats 10 --seconds 5 --warmup 10 --label "my system, stock"
```

Reference protocol used above (about 9 min for two sizes, add `16384` if the GPU has ≥ 8 GB free):

```bash
./trianglebench --n 4096,8192 --repeats 10 --seconds 5 --warmup 10 --label "<system / cooling / settings>"
```

With Docker (multi-arch image, amd64 and arm64) and the NVIDIA Container Toolkit:

```bash
docker build --build-arg GIT_COMMIT=$(git rev-parse --short=12 HEAD) -t trianglebench .
docker run --rm --gpus all trianglebench --selftest
docker run --rm --gpus all --user $(id -u):$(id -g) -v "$PWD:/out" -w /out trianglebench --label "my system"
```

| Option | Default | Meaning |
|---|---|---|
| `--n LIST` | 8192 | node counts, multiples of 256, comma separated; memory ≈ `19·n²` bytes (1.3 GB for 8192) |
| `--p` | 0.5 | edge probability |
| `--seed` | 1 | graph seed |
| `--repeats` | 5 | repetitions per size; the method order rotates every repetition |
| `--seconds` | 5 | measuring window per method and repetition |
| `--warmup` | 5 | warm-up seconds per size |
| `--cooldown` | 0 | idle seconds before every window (to study thermal effects) |
| `--label` | – | free text stored in the result file |
| `--out` | `tb_<gpu>_<utc>` | output prefix for `.json` and `.telemetry.csv` |

The exit code is `0` only if all methods agree on every node.

## Output for scientific analysis

Every run writes two files:

* **`<prefix>.json`** (schema `trianglebench-result/1`)
  * provenance: tool version, git commit, build architecture, nvcc and host compiler, command line,
    kernel, CPU architecture, GPU name/UUID, compute capability, driver, CUDA runtime, cuBLASLt version,
    power limit, clock offsets, ECC, persistence mode, PCIe link (discrete GPUs), UTC start and end time
  * graph: edges, triangles, theoretical expectation, global (FP64) and mean local (FP32) clustering
  * per method: summary statistics (k, mean, sd, 95 % CI half-width, median, min, max, coefficient of
    variation) for eq.TOPS, seconds per solve, watts, joule per solve and TOPS/W
  * per repetition: position in the rotated order, number of solves, window length, energy and its source
    (NVML energy counter or integrated power samples), max temperature, min/median/max SM clock,
    OR of all NVML clock-event reasons, and **every single solve time**
  * cuBLASLt algorithm, tile and stage IDs chosen for M2–M4
* **`<prefix>.telemetry.csv`**: raw NVML samples every 100 ms
  (`n, phase, method, repeat, t_s, power_w, temp_c, sm_mhz, mem_mhz, clock_event_reasons`)

Design choices for valid statistics:

* **Rotating method order** (repetition r runs methods r, r+1, r+2, r+3 mod 4), so warm-up and heat
  build-up do not systematically favour one method. `position_in_order` allows testing for order effects.
* **Repetitions as independent samples**, not individual solves (solves within a window are autocorrelated).
* **Energy from the hardware energy counter** where available, otherwise from the integral of sampled power.
* **Deterministic input**: the graph is a hash of `(seed, i, j)`, identical on every machine and
  architecture, so the triangle count itself is a reproducibility check.

Combine and compare result files (Python standard library only):

```bash
python3 analysis/summarize.py results/*.json                          # summary table
python3 analysis/summarize.py results/*.json --csv all_runs_tidy.csv  # one row per window, for R/pandas
python3 analysis/summarize.py a.json b.json --compare                 # ratios + Welch t-test
```

```python
import pandas as pd
df = pd.read_csv("all_runs_tidy.csv")
df.groupby(["gpu", "n", "method"])["eq_tops"].describe()
```

## JupyterLab variant: measure, analyse and share in one container

The `jupyter` target of the same Dockerfile adds JupyterLab with pinned Python packages (pandas, SciPy,
statsmodels, matplotlib), the analysis notebook and the reference results. It runs on amd64 and arm64.

```bash
docker build --target jupyter -t trianglebench:jupyter .
mkdir -p work
docker run --rm --gpus all --user $(id -u):$(id -g) -p 127.0.0.1:8888:8888 -v "$PWD/work:/work" trianglebench:jupyter
```

Open the `http://127.0.0.1:8888/lab?token=…` link printed in the log. JupyterLab is protected by a random
token and the port is bound to localhost only; from another machine use an SSH tunnel
(`ssh -L 8888:127.0.0.1:8888 user@gpu-host`) instead of exposing the port.

`work/trianglebench-analysis.ipynb` (copied once, never overwritten) does the following:

1. optionally runs the self-test and a new benchmark on the GPU of the container (`TB_RUN=1`)
2. loads the reference results and every file in `work/results/`, checks that all methods agreed and that
   the triangle counts are identical across systems
3. descriptive statistics per system, size and method (mean, 95 % CI, CV, watts, TOPS/W, J/solve, °C, MHz)
4. figures: throughput, efficiency, energy per solution, throughput vs. power/temperature/clock,
   raw telemetry over time, share of clock-event reasons
5. Welch t-tests between systems with Holm correction and Hedges' g
6. design checks: order effect and drift over repetitions (log-linear OLS, % per step with 95 % CI)
7. exports tables (CSV), figures (PNG/SVG) and `environment.json` (package versions, result provenance)
   to `work/export/`

Run a measurement and produce a shareable HTML report without opening the browser:

```bash
docker run --rm --gpus all --user $(id -u):$(id -g) -v "$PWD/work:/work" \
  -e TB_RUN=1 -e TB_LABEL="my system, stock" -e TB_ARGS="--n 4096,8192 --repeats 10 --seconds 5 --warmup 10" \
  trianglebench:jupyter sh -c "cp -n /opt/trianglebench/notebooks/*.ipynb . && \
  jupyter nbconvert --to html --execute trianglebench-analysis.ipynb --output-dir export"
```

Share `work/export/trianglebench-analysis.html` (self-contained, figures embedded) together with
`work/results/*.json` and `*.telemetry.csv`, so others can re-run the analysis on the raw data.
CI executes the notebook on the reference data for every commit.

## How correctness is guaranteed

* The graph comes from a 64-bit integer hash (splitmix64) of the node pair, identical on CPU and GPU.
* All intermediate values are integers: `(A·A)_ij ≤ n`, exactly representable in FP16 inputs (0/1),
  FP32 accumulators (up to 2²⁴) and INT32. The node limit is enforced.
* `--selftest` compares the per-node counts of all four methods with a CPU brute-force count over all node
  triples for five graphs, including the empty and the complete graph (`C(n,3)` triangles).
* Every benchmark repetition checks all four methods node by node.

## Contributing results

Pull requests with `results/<gpu>_<label>.json` and `.telemetry.csv` from the reference protocol are
welcome. Please describe cooling, power limit and any other load on the GPU in `--label`.

## License

MIT, see [LICENSE](LICENSE). Built by [BALTX.COM UAB](https://baltx.com).
