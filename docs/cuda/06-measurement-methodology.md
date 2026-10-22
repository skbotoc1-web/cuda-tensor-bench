# 6. Measuring correctly: time, energy, clocks, tools

A GPU number is only useful if someone else can reproduce it and knows how uncertain it is. This
chapter shows the measurement rules trianglebench follows, why each one exists, and the tools used for
the profiles in chapters 1 to 5.

## Theory: what can go wrong

1. **Kernel launches are asynchronous.** *"Kernel launches are asynchronous with respect to the host
   thread"* (Guide §2.1.2.2.1): the host does not wait for the kernel to complete, or even start. A host
   timer around a launch measures the launch, not the work. You have to synchronize, or time with CUDA
   events recorded in the stream (Guide §2.5; Best Practices Guide, *Timing*).
2. **The first runs are not representative.** The first calls pay for module loading, cuBLASLt
   heuristics and memory allocation, and the GPU starts at a cold, high boost clock.
3. **The clock is not constant.** Under load, a GPU lowers its clock when it reaches the power or
   temperature limit. Two methods measured one after the other do not see the same GPU: the second
   one starts warmer.
4. **One number is not a measurement.** Without the spread you cannot tell if a 2 % difference is real.

## In this repository

The measurement loop in [`src/trianglebench.cu`](../../src/trianglebench.cu), simplified (`sync(m)` stands
for `CK(cudaStreamSynchronize(x.st[m]))`):

```cuda
// warm-up: alternate M4 and M1 for --warmup seconds (default 5)
{ double t0 = now_s(); while (now_s() - t0 < warm) { x.solve(3); sync(3); x.solve(0); sync(0); } }

for (int r = 0; r < reps; r++)                   // --repeats, default 5
  for (int q = 0; q < 4; q++) {
    const int m = (r + q) % 4;                     // rotating order: every method is first once
    x.solve(m); sync(m);                           // one untimed solve
    tel.start();                                   // NVML: energy counter + 100 ms samples
    double t0 = now_s(), t = t0;
    while (t - t0 < secs || n_solves < 2) {        // --seconds, default 5: whole solves until the window is full
      x.solve(m); cudaStreamSynchronize(x.st[m]); t = now_s(); ...
    }
    Window w = tel.stop(...);                      // joules, mean W, max °C, median MHz, throttle reasons
    ok = x.result(m) == ref;                       // per-row triangle counts must equal the reference
  }
```

How each rule is implemented:

| Problem | Rule in trianglebench |
|---|---|
| asynchronous launches | `cudaStreamSynchronize` after every solve, host clock `std::chrono::steady_clock` (monotonic) |
| cold start | warm-up per graph size, plus one untimed solve before every window |
| short kernels and timer resolution | a window is at least 5 s of complete solves, not one solve |
| order and heat bias | the method order rotates every repetition: `(r + q) mod 4` |
| uncertainty | mean, standard deviation, CV and the 95 % confidence half-width with the Student t quantile (`t975`) over the repetitions |
| wrong but fast results | after every window the per-row counts are compared with the INT8 result (M4), which `--selftest` checks against a CPU count; `all_verified` in the JSON |
| reproducibility | the JSON stores the version, git commit, nvcc and driver version, the full command line and all parameters |

The confidence interval is `t(0.975, k-1) x sd / sqrt(k)`. For k = 10 repetitions the t quantile is
2.262, so the 95 % half-width is 0.72 standard deviations. Example from
[`results/rtx-a4000_std.json`](../../results/rtx-a4000_std.json) (run with `--repeats 10 --warmup 10`),
n = 8192:
M1 = 68.90 ± 0.09 eqTOPS (CV 0.18 %), M4 = 117.35 ± 0.07 eqTOPS (CV 0.08 %). Differences of a few
percent between methods are therefore real, and the run-to-run noise is far below 1 %.

## Energy: counter, not samples

NVML offers two ways to get energy:

- `nvmlDeviceGetTotalEnergyConsumption` - a counter in millijoules since the driver loaded. The
  difference between start and end of a window is the exact energy of the window. trianglebench uses
  it whenever it is available (`"energy_source": "nvml_energy_counter"` in the JSON; available on both
  GPUs here).
- `nvmlDeviceGetPowerUsage` sampled every 100 ms and integrated with the trapezoidal rule. This is the
  fallback, marked `"energy_source": "sampled_power_integral"`. It misses peaks between the samples.

From the energy follow the two efficiency figures used in every table of these notes:

```
W           = joules / window seconds
J per solve = joules / number of solves
TOPS/W      = eqTOPS / W        (= tera-operations per joule)
```

**Derived** check with the A4000 INT8 quantisation result: one solve takes 2 x 8192³ / 114.0e12 =
9.65 ms; at 124.9 W that is 1.205 J. The JSON reports 1.212 J from the counter; the 0.6 % difference
comes from the time between solves (synchronization, host loop) that is inside the window but not in
the solve time.

On the GB10 the CPU and the GPU share one chip and one memory. NVML reports a GPU power figure, but
whether it covers only the GPU is not documented. Compare GB10 watts with each other, not with the
board power of a discrete card.

## Clocks and throttle reasons

Each telemetry sample also stores the SM clock and `nvmlDeviceGetCurrentClocksEventReasons`. The JSON
keeps the OR of all reasons over a window (`clock_event_reasons_or`). The bits are defined in
`nvml.h`, for example:

| Bit | Name | Meaning |
|---|---|---|
| `0x1` | `GpuIdle` | nothing to do |
| `0x4` | `SwPowerCap` | the driver lowers the clock to stay under the power limit |
| `0x20` | `SwThermalSlowdown` | the driver lowers the clock because of temperature |

What the data show, n = 8192, standard runs:

| | A4000 | GB10 |
|---|---|---|
| reasons in every window | `0x4` (power cap) | `0x0` (NVML reports no reasons on the GB10) |
| board power, range over M1-M4 | 124-140 W | 64-88 W |
| max temperature | 85 °C | 81-83 °C |
| median SM clock | 1,320-1,470 MHz | 2,489-2,502 MHz |

**Derived** - the A4000 runs every method at its power limit, so the method that draws the most power
per operation (M2, FP32 on CUDA cores, 140 W) gets the lowest clock (1,320 MHz). On the GB10, NVML
reports neither a power limit nor clock-event reasons (`nvidia-smi` shows the limit as `[N/A]` and the
reasons as `0x0`), yet in the peak tests the clock dropped from about 2,490 to about 2,200 MHz at
91-93 W ([`peak/README.md`](../../peak/README.md)). On such a platform the clock itself is the only
reliable signal. Always record the clock next to the result.

## Tools

All commands were run for these notes; the outputs in the chapters come from them.

| Question | Tool | Command used |
|---|---|---|
| How many registers and how much shared memory does a kernel use? | `nvcc` | `nvcc -O3 -arch=sm_86 --resource-usage -c src/trianglebench.cu` |
| Which instructions did the compiler generate? | `cuobjdump` | `cuobjdump -sass trianglebench`, then search for `POPC`, `I2F.F64`, `F2I.S64.F64` |
| Where does the time go over the whole run? | Nsight Systems | `nsys profile --trace=cuda -o tb ./trianglebench --n 8192 --repeats 1 --seconds 0.3 --warmup 0` then `nsys stats --report cuda_gpu_kern_sum tb.nsys-rep` |
| Why is one kernel slow? | Nsight Compute | `ncu --kernel-name k_m1_bitset --launch-count 1 --metrics <list> ./trianglebench --n 8192 --repeats 1 --seconds 0.2 --warmup 0` (metric lists in chapters 2 and 3) |
| Are there out-of-bounds accesses or shared-memory races? | Compute Sanitizer | `compute-sanitizer --tool memcheck ./trianglebench --selftest` and `--tool racecheck` |

Results worth knowing:

- **Nsight Systems** on the GB10 shows which kernel cuBLASLt picked for each method: the FP16 GEMM runs
  `nvjet_sm121_hss_mma_128x208x64_..._tmaAB`, a kernel built for sm_121 that loads with TMA; the INT8
  GEMM runs `cutlass_80_tensorop_i16832gemm_s8_256x128_64x3_tn`, a kernel built for sm_80. This is
  consistent with the INT8 heuristic result on the GB10 being 24 % below the best configuration
  ([chapter 5](05-cublaslt-in-practice.md)). The same profile shows the share of the epilogue: 1.2 ms
  of about 10 ms per INT8 solve.
  The profile ran without warm-up and with 0.3 s windows. Summed kernel time per solve was 2 % below
  the benchmark solve time for M1, but 12 % (M3) and 20 % (M4) above it. Use such a profile for
  proportions and kernel names, not for the headline number.
- **Nsight Compute** controls the clocks while it profiles (option `--clock-control`, default `base`);
  `ncu` reported about 1,553 MHz on the A4000, while the loaded benchmark ran at 1,470 MHz. Durations from `ncu` are
  therefore not directly comparable to benchmark times. Each kernel is replayed many times to collect
  all counters, so profiling takes minutes. Counter access needs administrator rights or the driver
  option `NVreg_RestrictProfilingToAdminUsers=0`. On the GB10 the user account got `ERR_NVGPUCTRPERM`,
  which is why the profiles in these notes come from the A4000.
- **Compute Sanitizer** reported `0 errors` (memcheck) and `0 hazards` (racecheck) for
  `trianglebench --selftest` on the GB10. Racecheck covers the shared-memory tiles of `k_m1_bitset`,
  where a missing `__syncthreads()` would be a classic bug. Run it after every kernel change; it is slow
  but catches what a correct result can hide.

## Checklist for your own benchmark

1. Synchronize, or use CUDA events, before reading a timer.
2. Warm up, then measure windows of seconds, not single calls.
3. Rotate the order of the variants, and repeat at least 5 times.
4. Report mean ± 95 % CI, and the clock, power, temperature and throttle reasons of the window.
5. Use the NVML energy counter for joules; sampled power only as a fallback.
6. Check the result of every measured run, not just once.
7. Store version, commit, driver, command line and parameters with the numbers.

## References

- CUDA Programming Guide 13.4, §2.1.2.2.1 Triple Chevron Notation, §2.5 Asynchronous Execution.
  <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- CUDA C++ Best Practices Guide, *Performance Metrics: Timing*.
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>
- NVML API Reference: `nvmlDeviceGetTotalEnergyConsumption`, `nvmlDeviceGetCurrentClocksEventReasons`.
  <https://docs.nvidia.com/deploy/nvml-api/>
- Nsight Systems User Guide. <https://docs.nvidia.com/nsight-systems/UserGuide/>
- Nsight Compute Profiling Guide (clock control, replay). <https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
- Compute Sanitizer documentation. <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/>
