// trianglebench - count the triangles of a random graph G(n,p) in four independent ways
// and benchmark CUDA cores (integer + floating point) against Tensor Cores (matrix multiply-accumulate).
//
//   triangles(G) = trace(A^3) / 6 = sum_ij A_ij * (A*A)_ij / 6        (A = 0/1 adjacency matrix)
//
//   M1  CUDA cores, INT32 : own kernel, rows as bitsets, (A*A)_ij = popcount(row_i AND row_j)
//   M2  CUDA cores, FP32  : cuBLASLt GEMM, FP32 pedantic (Tensor Cores disabled)
//   M3  Tensor Cores, FP16: cuBLASLt GEMM, FP16 inputs, FP32 accumulate
//   M4  Tensor Cores, INT8: cuBLASLt GEMM, INT8 inputs, INT32 accumulate
//
// All four results are exact integers and must agree for every single node. Afterwards the clustering
// coefficients are computed in FP64/FP32 and compared with random-graph theory (E[C] = p).
// Phase 2 runs all four methods at the same time in separate CUDA streams.
//
// SPDX-License-Identifier: MIT  -  https://github.com/skbotoc1-web/cuda-tensor-bench
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <nvml.h>

#define VERSION "1.0.0"
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define LT(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLASLt error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(2); } } while (0)
typedef unsigned long long u64;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ------------------------------------------------------------------ graph (integer hash, identical on host and GPU)
__host__ __device__ inline u64 splitmix64(u64 x) {
  x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
// Undirected edge {i,j} exists with probability p; thr = p * 2^32 (so p = 1 always gives an edge).
__host__ __device__ inline bool has_edge(u64 seed, u64 thr, uint32_t n, uint32_t i, uint32_t j) {
  if (i == j) return false;
  uint32_t a = i < j ? i : j, b = i < j ? j : i;
  return (splitmix64(seed ^ ((u64)a * n + b)) >> 32) < thr;
}
__global__ void k_gen_bits(uint32_t* bits, int n, int W, u64 seed, u64 thr) {
  for (size_t w = blockIdx.x * (size_t)blockDim.x + threadIdx.x; w < (size_t)n * W; w += (size_t)gridDim.x * blockDim.x) {
    uint32_t i = w / W, j0 = (w % W) * 32, v = 0;
    for (int b = 0; b < 32; b++) v |= (uint32_t)has_edge(seed, thr, n, i, j0 + b) << b;
    bits[w] = v;
  }
}
__device__ inline uint32_t bit(const uint32_t* bits, int W, int i, int j) { return (bits[(size_t)i * W + (j >> 5)] >> (j & 31)) & 1u; }
__global__ void k_expand(const uint32_t* bits, int n, int W, int8_t* a8, __half* a16, float* a32) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < (size_t)n * n; x += (size_t)gridDim.x * blockDim.x) {
    uint32_t b = bit(bits, W, x / n, x % n);
    a8[x] = (int8_t)b; a16[x] = __uint2half_rn(b); a32[x] = (float)b;
  }
}

// ------------------------------------------------------------------ M1: bitset AND + popcount on CUDA cores
// 64x64 output tile per block, 256 threads, 4x4 outputs per thread, 8 words (256 columns) per shared-memory step.
// Fused mask: rowsum[i] += sum_j A_ij * popc(row_i & row_j)   (= 2 * triangles at node i)
__global__ void __launch_bounds__(256) k_m1_bitset(const uint32_t* bits, int n, int W, u64* rowsum) {
  __shared__ uint32_t As[8][68], Bs[8][68];  // +4 padding avoids shared-memory bank conflicts on the stores
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16, bi = blockIdx.y * 64, bj = blockIdx.x * 64;
  uint32_t acc[4][4] = {};
  for (int k0 = 0; k0 < W; k0 += 8) {
    for (int t = threadIdx.x; t < 512; t += 256) {
      int r = t / 8, w = t % 8;
      As[w][r] = bits[(size_t)(bi + r) * W + k0 + w];
      Bs[w][r] = bits[(size_t)(bj + r) * W + k0 + w];
    }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < 8; w++) {
      uint32_t a[4], b[4];
#pragma unroll
      for (int r = 0; r < 4; r++) a[r] = As[w][ty + 16 * r];
#pragma unroll
      for (int c = 0; c < 4; c++) b[c] = Bs[w][tx + 16 * c];
#pragma unroll
      for (int r = 0; r < 4; r++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[r][c] += __popc(a[r] & b[c]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < 4; r++) {
    const int i = bi + ty + 16 * r;
    uint32_t s = 0;
#pragma unroll
    for (int c = 0; c < 4; c++) s += acc[r][c] * bit(bits, W, i, bj + tx + 16 * c);
    for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);  // reduce the 16 threads sharing row i
    if (tx == 0 && s) atomicAdd(&rowsum[i], (u64)s);
  }
}

// ------------------------------------------------------------------ M2-M4: masked row sum of C = A*A (CUDA cores)
template <typename T> __global__ void k_masked_rowsum(const T* C, const uint32_t* bits, int n, int W, u64* rowsum) {
  const int i = blockIdx.x;
  u64 s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    if (bit(bits, W, i, j)) s += (u64)llrint((double)C[(size_t)i * n + j]);
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  __shared__ u64 sh[32];
  if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x == 0) { u64 t = 0; for (int k = 0; k < blockDim.x / 32; k++) t += sh[k]; rowsum[i] = t; }
}

// ------------------------------------------------------------------ floating-point statistics (clustering)
// per node: degree d (popcount), triangles t = rowsum/2, local clustering c = t / (d(d-1)/2)   [FP32]
// global:   transitivity = 3T / sum d(d-1)/2                                                   [FP64]
__global__ void k_stats(const uint32_t* bits, const u64* rowsum, int n, int W, double* triples, float* csum, u64* degsum) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    uint32_t d = 0;
    for (int w = 0; w < W; w++) d += __popc(bits[(size_t)i * W + w]);
    double tr = 0.5 * (double)d * (double)(d - 1);
    float c = d > 1 ? (float)(rowsum[i] / 2) / (float)tr : 0.f;
    atomicAdd(triples, tr); atomicAdd(csum, c); atomicAdd(degsum, (u64)d);
  }
}

// ------------------------------------------------------------------ cuBLASLt GEMM C = A*A (A symmetric, so layout is free)
struct Gemm {
  cublasLtHandle_t h; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lc; cublasLtMatmulAlgo_t algo;
  void* ws; size_t wsz = 32ull << 20; const void* A; void* C; int32_t ione = 1, izero = 0; float fone = 1, fzero = 0; bool integer;
  void init(int n, cudaDataType ab, cudaDataType c, cublasComputeType_t ct, const void* a, void* cbuf) {
    integer = (ct == CUBLAS_COMPUTE_32I); A = a; C = cbuf;
    LT(cublasLtCreate(&h)); CK(cudaMalloc(&ws, wsz));
    LT(cublasLtMatmulDescCreate(&op, ct, integer ? CUDA_R_32I : CUDA_R_32F));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
    LT(cublasLtMatrixLayoutCreate(&la, ab, n, n, n)); LT(cublasLtMatrixLayoutCreate(&lc, c, n, n, n));
    cublasLtMatmulPreference_t pref; LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof wsz));
    cublasLtMatmulHeuristicResult_t r; int got = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(h, op, la, la, lc, lc, pref, 1, &r, &got));
    if (!got) { fprintf(stderr, "cuBLASLt: no algorithm for this precision on this GPU\n"); exit(2); }
    algo = r.algo; cublasLtMatmulPreferenceDestroy(pref);
  }
  void run(cudaStream_t s) {
    const void* al = integer ? (const void*)&ione : (const void*)&fone;
    const void* be = integer ? (const void*)&izero : (const void*)&fzero;
    LT(cublasLtMatmul(h, op, al, A, la, A, la, be, C, lc, C, lc, &algo, ws, wsz, s));
  }
};

// ------------------------------------------------------------------ NVML telemetry (energy counter + 100 ms samples)
struct Telemetry {
  nvmlDevice_t dev; bool ok = false, energy_ok = false;
  std::atomic<bool> running{false}; std::thread th; unsigned tmax = 0; std::vector<unsigned> clocks; double t0 = 0, wsum = 0; int wn = 0; u64 e0 = 0;
  void init(int cuda_dev) {
    char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, cuda_dev));
    ok = nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByPciBusId_v2(bus, &dev) == NVML_SUCCESS;
    u64 e; energy_ok = ok && nvmlDeviceGetTotalEnergyConsumption(dev, &e) == NVML_SUCCESS;
  }
  void start() {
    if (!ok) return;
    tmax = 0; clocks.clear(); wsum = 0; wn = 0; t0 = now_s(); if (energy_ok) nvmlDeviceGetTotalEnergyConsumption(dev, &e0);
    running = true;
    th = std::thread([this] {
      while (running) {
        unsigned c, p;
        nvmlTemperature_t t = {}; t.version = nvmlTemperature_v1; t.sensorType = NVML_TEMPERATURE_GPU;
        if (nvmlDeviceGetTemperatureV(dev, &t) == NVML_SUCCESS && t.temperature > 0) tmax = std::max(tmax, (unsigned)t.temperature);
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &c) == NVML_SUCCESS) clocks.push_back(c);
        if (nvmlDeviceGetPowerUsage(dev, &p) == NVML_SUCCESS) { wsum += p / 1000.0; wn++; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    });
  }
  // returns average watts over the window; fills max temperature and median SM clock
  double stop(unsigned& tm, unsigned& mhz) {
    tm = 0; mhz = 0;
    if (!ok) return 0;
    running = false; th.join();
    double dt = now_s() - t0, w = wn ? wsum / wn : 0;
    u64 e1; if (energy_ok && nvmlDeviceGetTotalEnergyConsumption(dev, &e1) == NVML_SUCCESS && e1 > e0) w = (e1 - e0) / 1000.0 / dt;
    tm = tmax; if (!clocks.empty()) { std::sort(clocks.begin(), clocks.end()); mhz = clocks[clocks.size() / 2]; }
    return w;
  }
};

// ------------------------------------------------------------------ benchmark context
static const char* NAMES[4] = {"M1 CUDA cores  INT32 bitset+popc", "M2 CUDA cores  FP32 GEMM        ",
                               "M3 Tensor Cores FP16->FP32 GEMM  ", "M4 Tensor Cores INT8->INT32 GEMM "};
struct Ctx {
  int n, W; u64 seed, thr; uint32_t* bits; int8_t* a8; __half* a16; float* a32; int32_t* c8; float *c16, *c32;
  u64* rowsum[4]; Gemm g[3]; cudaStream_t st[4];
  void init(int n_, double p, u64 seed_) {
    n = n_; W = n / 32; seed = seed_; thr = (u64)llround(p * 4294967296.0);
    size_t nn = (size_t)n * n;
    CK(cudaMalloc(&bits, (size_t)n * W * 4)); CK(cudaMalloc(&a8, nn)); CK(cudaMalloc(&a16, nn * 2)); CK(cudaMalloc(&a32, nn * 4));
    CK(cudaMalloc(&c8, nn * 4)); CK(cudaMalloc(&c16, nn * 4)); CK(cudaMalloc(&c32, nn * 4));
    for (int m = 0; m < 4; m++) { CK(cudaMalloc(&rowsum[m], n * 8)); CK(cudaStreamCreateWithFlags(&st[m], cudaStreamNonBlocking)); }
    k_gen_bits<<<1024, 256>>>(bits, n, W, seed, thr); CK(cudaGetLastError());
    k_expand<<<1024, 256>>>(bits, n, W, a8, a16, a32); CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    g[0].init(n, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, a32, c32);  // M2
    g[1].init(n, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, a16, c16);           // M3
    g[2].init(n, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, a8, c8);              // M4
  }
  void release() {
    cudaFree(bits); cudaFree(a8); cudaFree(a16); cudaFree(a32); cudaFree(c8); cudaFree(c16); cudaFree(c32);
    for (int m = 0; m < 4; m++) { cudaFree(rowsum[m]); cudaStreamDestroy(st[m]); }
    for (auto& x : g) { cublasLtMatmulDescDestroy(x.op); cublasLtMatrixLayoutDestroy(x.la); cublasLtMatrixLayoutDestroy(x.lc); cudaFree(x.ws); cublasLtDestroy(x.h); }
  }
  void solve(int m) {  // one complete triangle count with method m, asynchronous on stream st[m]
    cudaStream_t s = st[m];
    if (m == 0) {
      CK(cudaMemsetAsync(rowsum[0], 0, n * 8, s));
      k_m1_bitset<<<dim3(n / 64, n / 64), 256, 0, s>>>(bits, n, W, rowsum[0]);
    } else {
      g[m - 1].run(s);
      if (m == 1) k_masked_rowsum<float><<<n, 256, 0, s>>>(c32, bits, n, W, rowsum[1]);
      if (m == 2) k_masked_rowsum<float><<<n, 256, 0, s>>>(c16, bits, n, W, rowsum[2]);
      if (m == 3) k_masked_rowsum<int32_t><<<n, 256, 0, s>>>(c8, bits, n, W, rowsum[3]);
    }
    CK(cudaGetLastError());
  }
  std::vector<u64> result(int m) { std::vector<u64> v(n); CK(cudaMemcpy(v.data(), rowsum[m], n * 8, cudaMemcpyDeviceToHost)); return v; }
};

static u64 total(const std::vector<u64>& rs) { u64 s = 0; for (u64 x : rs) s += x; return s / 6; }

// CPU reference: brute force over all node triples i<j<k, per-node counts *2 (same scale as rowsum)
static std::vector<u64> cpu_reference(int n, u64 seed, u64 thr) {
  std::vector<uint8_t> A((size_t)n * n);
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[(size_t)i * n + j] = has_edge(seed, thr, n, i, j);
  std::vector<u64> t(n, 0);
  for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (A[(size_t)i * n + j])
    for (int k = j + 1; k < n; k++) if (A[(size_t)i * n + k] && A[(size_t)j * n + k]) { t[i] += 2; t[j] += 2; t[k] += 2; }
  return t;
}

static int selftest() {
  struct Case { int n; double p; u64 seed; } cases[] = {{512, 0.5, 1}, {1024, 0.1, 2}, {768, 1.0, 3}, {256, 0.0, 4}, {1280, 0.3, 5}};
  int fails = 0;
  for (auto& c : cases) {
    Ctx x; x.init(c.n, c.p, c.seed);
    std::vector<u64> ref = cpu_reference(c.n, c.seed, x.thr);
    printf("selftest n=%5d p=%.2f : CPU %10llu triangles |", c.n, c.p, total(ref));
    bool ok = true;
    for (int m = 0; m < 4; m++) { x.solve(m); CK(cudaDeviceSynchronize()); bool eq = x.result(m) == ref; ok &= eq; printf(" M%d %s", m + 1, eq ? "OK" : "FAIL"); }
    if (c.p == 1.0) { u64 exp = (u64)c.n * (c.n - 1) * (c.n - 2) / 6; bool eq = total(ref) == exp; ok &= eq; printf(" | C(n,3) %s", eq ? "OK" : "FAIL"); }
    printf("\n"); fails += !ok; x.release();
  }
  printf("selftest: %s\n", fails ? "FAILED" : "PASSED");
  return fails ? 1 : 0;
}

struct Rate { double solves_per_s, watts; unsigned tmax, mhz; };

static void usage() {
  printf("trianglebench %s - triangle counting on CUDA cores vs Tensor Cores\n"
         "usage: trianglebench [--n N] [--p P] [--seed S] [--seconds T] [--selftest]\n"
         "  --n        nodes, multiple of 256 (default 16384)\n  --p        edge probability (default 0.5)\n"
         "  --seed     graph seed (default 1)\n  --seconds  measuring time per phase (default 8)\n"
         "  --selftest compare all methods with a CPU brute-force count and exit\n", VERSION);
}

int main(int argc, char** argv) {
  int n = 16384; double p = 0.5, secs = 8; u64 seed = 1; bool st = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i]; auto val = [&] { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
    if (a == "--n") n = atoi(val()); else if (a == "--p") p = atof(val()); else if (a == "--seed") seed = strtoull(val(), 0, 10);
    else if (a == "--seconds") secs = atof(val()); else if (a == "--selftest") st = true; else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
  }
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  int drv = 0, rt = 0; CK(cudaDriverGetVersion(&drv)); CK(cudaRuntimeGetVersion(&rt));
  Telemetry tel; tel.init(0);
  char drvs[80] = "n/a"; if (tel.ok) nvmlSystemGetDriverVersion(drvs, sizeof drvs);
  printf("trianglebench %s | %s (sm_%d%d, %d SMs, %.1f GB) | driver %s | CUDA runtime %d.%d\n", VERSION, pr.name, pr.major, pr.minor,
         pr.multiProcessorCount, pr.totalGlobalMem / 1e9, drvs, rt / 1000, (rt % 1000) / 10);
  if (st) return selftest();
  if (n < 256 || n % 256 || n >= (1 << 24) || p < 0 || p > 1) { fprintf(stderr, "invalid --n (multiple of 256) or --p (0..1)\n"); return 2; }
  size_t need = (size_t)n * n * 19 + (100ull << 20), freeb, totb; CK(cudaMemGetInfo(&freeb, &totb));
  if (need > freeb) { fprintf(stderr, "need %.1f GB GPU memory, %.1f GB free - use a smaller --n\n", need / 1e9, freeb / 1e9); return 2; }

  Ctx x; x.init(n, p, seed);
  const double ops = 2.0 * n * (double)n * n;  // operations of one dense A*A, used as common yardstick for all methods
  // reference = M4, then every method must reproduce it node by node
  x.solve(3); CK(cudaDeviceSynchronize());
  std::vector<u64> ref = x.result(3);
  double T = (double)total(ref), nc3 = (double)n * (n - 1) * (n - 2) / 6;
  double* d_tr; float* d_cs; u64* d_deg; CK(cudaMalloc(&d_tr, 8)); CK(cudaMalloc(&d_cs, 4)); CK(cudaMalloc(&d_deg, 8));
  CK(cudaMemset(d_tr, 0, 8)); CK(cudaMemset(d_cs, 0, 4)); CK(cudaMemset(d_deg, 0, 8));
  k_stats<<<(n + 255) / 256, 256>>>(x.bits, x.rowsum[3], n, x.W, d_tr, d_cs, d_deg); CK(cudaDeviceSynchronize());
  double triples; float csum; u64 degsum;
  CK(cudaMemcpy(&triples, d_tr, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&csum, d_cs, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&degsum, d_deg, 8, cudaMemcpyDeviceToHost));
  printf("graph G(n=%d, p=%.4f, seed=%llu): %llu edges, %.0f triangles (theory C(n,3)p^3 = %.0f, ratio %.5f)\n",
         n, p, seed, degsum / 2, T, nc3 * p * p * p, T / (nc3 * p * p * p));
  printf("clustering: global (FP64) %.6f, mean local (FP32) %.6f, theory p = %.6f\n\n", triples > 0 ? 3 * T / triples : 0.0, csum / n, p);

  // warm-up so that every method is measured on a warm card (power limit and boost are temperature dependent)
  { double t0 = now_s(); while (now_s() - t0 < 3) { x.solve(3); CK(cudaStreamSynchronize(x.st[3])); } }

  printf("Phase 1: each method alone (%.0f s each). 'eq.TOPS' = 2n^3 / time, the work of a dense A*A.\n", secs);
  printf("%-34s %9s %8s %7s %6s %5s %8s %8s  %s\n", "method", "s/solve", "eq.TOPS", "W avg", "T max", "MHz", "TOPS/W", "J/solve", "check");
  Rate alone[4]; int bad = 0;
  for (int m = 0; m < 4; m++) {
    x.solve(m); CK(cudaStreamSynchronize(x.st[m]));
    tel.start(); double t0 = now_s(); int k = 0;
    while (now_s() - t0 < secs || k < 2) { x.solve(m); CK(cudaStreamSynchronize(x.st[m])); k++; }
    double dt = now_s() - t0; unsigned tm, mhz; double w = tel.stop(tm, mhz);
    bool eq = x.result(m) == ref; bad += !eq;
    double s = dt / k, tops = ops / s / 1e12;
    alone[m] = {k / dt, w, tm, mhz};
    printf("%-34s %9.4f %8.2f %7.1f %6u %5u %8.3f %8.1f  %s\n", NAMES[m], s, tops, w, tm, mhz, w > 0 ? tops / w : 0, w * s, eq ? "OK" : "MISMATCH");
    if (m == 0)  // native rate: one AND + one POPC + one ADD per 32-bit word pair, n*n*(n/32) word pairs per solve
      printf("%-34s native: %.2f T popc/s, %.2f T int32 ops/s (AND+POPC+ADD); 1 popc = 32 multiply-adds of 0/1 values\n", "",
             (double)n * n * x.W / s / 1e12, 3.0 * n * n * x.W / s / 1e12);
  }

  printf("\nPhase 2: all four methods at the same time, one CUDA stream and host thread each (%.0f s)\n", secs);
  std::atomic<bool> go{false}; int cnt[4] = {}; bool eqc[4] = {};
  std::vector<std::thread> th;
  tel.start(); double t0 = now_s();
  for (int m = 0; m < 4; m++) th.emplace_back([&, m] {
    CK(cudaSetDevice(0));
    while (!go) std::this_thread::yield();
    while (now_s() - t0 < secs || cnt[m] < 1) { x.solve(m); CK(cudaStreamSynchronize(x.st[m])); cnt[m]++; }
    eqc[m] = x.result(m) == ref;
  });
  t0 = now_s(); go = true;
  for (auto& t : th) t.join();
  double dt = now_s() - t0; unsigned tm, mhz; double w = tel.stop(tm, mhz);
  printf("%-34s %9s %9s %7s  %s\n", "method", "alone/s", "parallel/s", "share", "check");
  double sum = 0, tops_all = 0;
  for (int m = 0; m < 4; m++) {
    double r = cnt[m] / dt, sh = r / alone[m].solves_per_s; sum += sh; tops_all += r * ops / 1e12; bad += !eqc[m];
    printf("%-34s %9.3f %9.3f %7.2f  %s\n", NAMES[m], alone[m].solves_per_s, r, sh, eqc[m] ? "OK" : "MISMATCH");
  }
  printf("all four together: %.2f eq.TOPS, %.1f W avg, T max %u C, %u MHz, %.3f TOPS/W\n", tops_all, w, tm, mhz, w > 0 ? tops_all / w : 0);
  printf("sum of shares %.2f  (1.00 = no gain from running in parallel; >1 = units overlap; <1 = they get in each other's way)\n", sum);
  printf("\nresult: %s\n", bad ? "MISMATCH - results differ between methods" : "all methods agree on every node");
  x.release();
  return bad ? 1 : 0;
}
