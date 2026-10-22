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
#include <ctime>
#include <sys/utsname.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <nvml.h>

#define VERSION "1.3.0"
#define SCHEMA "trianglebench-result/1"
#ifndef GIT_COMMIT
#define GIT_COMMIT "unknown"
#endif
#ifndef BUILD_ARCH
#define BUILD_ARCH "unknown"
#endif
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)
#define LT(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLASLt error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(2); } } while (0)
typedef unsigned long long u64;
static u64 total(const std::vector<u64>& rs);
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
  bool init(int n, cudaDataType ab, cudaDataType c, cublasComputeType_t ct, const void* a, void* cbuf) {  // false = no algorithm on this GPU
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
    if (cublasLtMatmulAlgoGetHeuristic(h, op, la, la, lc, lc, pref, 1, &r, &got) != CUBLAS_STATUS_SUCCESS) got = 0;
    cublasLtMatmulPreferenceDestroy(pref);
    if (!got) return false;
    algo = r.algo;
    size_t w; cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof algo_id, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof tile_id, &w);
    cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages_id, sizeof stages_id, &w);
    return true;
  }
  int32_t algo_id = -1; uint32_t tile_id = 0, stages_id = 0;
  void rebind(const void* a, void* cbuf) { A = a; C = cbuf; }  // fone/fzero live in this object: call after any copy
  void run(cudaStream_t s) {
    const void* al = integer ? (const void*)&ione : (const void*)&fone;
    const void* be = integer ? (const void*)&izero : (const void*)&fzero;
    LT(cublasLtMatmul(h, op, al, A, la, A, la, be, C, lc, C, lc, &algo, ws, wsz, s));
  }
};

// ------------------------------------------------------------------ quantisation study: A in an arbitrary input format, C = A*A
// 0/1 entries are exactly representable in every format below, so the only error source is the
// representation of the *sums* in C (and the input format itself for int4/int8, where 1 stays exact).
struct Quant { const char* name; const char* id; cudaDataType ab; cudaDataType c; cublasComputeType_t ct; int bits; bool base; };
// base=true duplicates a fixed method (M2 fp32, M3 fp16, M4 int8) and is therefore not in the default list
static const Quant QUANTS[] = {
  {"fp32",     "Q_fp32",            CUDA_R_32F,   CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, 32, true},
  {"fp16",     "Q_fp16_fp32",       CUDA_R_16F,   CUDA_R_32F, CUBLAS_COMPUTE_32F,      16, true},
  {"bf16",     "Q_bf16_fp32",       CUDA_R_16BF,  CUDA_R_32F, CUBLAS_COMPUTE_32F,      16, false},
  {"tf32",     "Q_tf32_fp32",       CUDA_R_32F,   CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 32, false},
  {"fp8e4m3",  "Q_fp8e4m3_fp32",    CUDA_R_8F_E4M3, CUDA_R_32F, CUBLAS_COMPUTE_32F,     8, false},
  {"fp8e5m2",  "Q_fp8e5m2_fp32",    CUDA_R_8F_E5M2, CUDA_R_32F, CUBLAS_COMPUTE_32F,     8, false},
  {"int8",     "Q_int8_int32",      CUDA_R_8I,    CUDA_R_32I, CUBLAS_COMPUTE_32I,       8, true},
  {"int4",     "Q_int4_int32",      CUDA_R_4I,    CUDA_R_32I, CUBLAS_COMPUTE_32I,       4, false},
  {"fp4e2m1",  "Q_fp4e2m1_fp32",    CUDA_R_4F_E2M1, CUDA_R_32F, CUBLAS_COMPUTE_32F,     4, false},
};
static const int NQUANT = (int)(sizeof QUANTS / sizeof QUANTS[0]);

__device__ inline uint8_t q_e2m1(bool one) { return one ? 0x2 : 0x0; }  // E2M1: 0.0 = 0b000, 1.0 = 0b010
__device__ inline uint8_t q_e4m3(bool one) { return one ? 0x38 : 0x00; }  // E4M3: 1.0 = 0b0'0111'000 = 0x38
__device__ inline uint8_t q_e5m2(bool one) { return one ? 0x3C : 0x00; }  // E5M2: 1.0 = 0b0'01111'00

__global__ void k_quant(const uint32_t* bits, int n, int W, int fmt, void* out) {  // one thread per element pair
  int pairs = n / 2;   // one thread per element pair: 4-bit formats pack two values per byte, the others use two elements per thread
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n * pairs; idx += gridDim.x * blockDim.x) {
    int i = idx / pairs, j0 = (idx % pairs) * 2;
    bool e0 = bit(bits, W, i, j0), e1 = bit(bits, W, i, j0 + 1);
    size_t o = (size_t)i * pairs + (idx % pairs);   // packed: byte index; non-packed: pair index
    if (fmt == CUDA_R_4I) { ((uint8_t*)out)[o] = (uint8_t)((e1 ? 1 : 0) | (e0 ? 0x10 : 0)); }
    else if (fmt == CUDA_R_4F_E2M1) { ((uint8_t*)out)[o] = (uint8_t)(q_e2m1(e1) | (q_e2m1(e0) << 4)); }
    else if (fmt == CUDA_R_8I) { ((int8_t*)out)[o * 2] = e0 ? 1 : 0; ((int8_t*)out)[o * 2 + 1] = e1 ? 1 : 0; }
    else if (fmt == CUDA_R_8F_E4M3) { ((uint8_t*)out)[o * 2] = q_e4m3(e0); ((uint8_t*)out)[o * 2 + 1] = q_e4m3(e1); }
    else if (fmt == CUDA_R_8F_E5M2) { ((uint8_t*)out)[o * 2] = q_e5m2(e0); ((uint8_t*)out)[o * 2 + 1] = q_e5m2(e1); }
    else if (fmt == CUDA_R_16F) { ((__half*)out)[o * 2] = __float2half(e0 ? 1.f : 0.f); ((__half*)out)[o * 2 + 1] = __float2half(e1 ? 1.f : 0.f); }
    else if (fmt == CUDA_R_16BF) { ((__nv_bfloat16*)out)[o * 2] = __float2bfloat16(e0 ? 1.f : 0.f); ((__nv_bfloat16*)out)[o * 2 + 1] = __float2bfloat16(e1 ? 1.f : 0.f); }
    else { ((float*)out)[o * 2] = e0 ? 1.f : 0.f; ((float*)out)[o * 2 + 1] = e1 ? 1.f : 0.f; }
  }
}

static u64 total(const std::vector<u64>& rs) { u64 s = 0; for (u64 x : rs) s += x; return s / 6; }

// error of a quantised run against the exact per-node reference
struct Err { double mean_abs = 0, max_abs = 0, exact_frac = 0, total_rel = 0; u64 total = 0; };
static Err measure_err(const std::vector<u64>& got, const std::vector<u64>& ref) {
  Err e; double sum = 0; int exact = 0;
  for (size_t i = 0; i < ref.size(); i++) {
    double d = std::fabs((double)got[i] - (double)ref[i]);
    sum += d; e.max_abs = std::max(e.max_abs, d); exact += (d == 0);
  }
  e.mean_abs = sum / ref.size(); e.exact_frac = (double)exact / ref.size();
  e.total = total(got); double tref = (double)total(ref);
  e.total_rel = tref ? (double)e.total / tref - 1.0 : 0.0;
  return e;
}

// ------------------------------------------------------------------ small helpers: JSON, statistics
static std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o + "\"";
}
static std::string jnum(double v) { if (!std::isfinite(v)) return "null"; char b[40]; snprintf(b, sizeof b, "%.10g", v); return b; }
static std::string jint(long long v, bool valid = true) { return valid ? std::to_string(v) : "null"; }
static std::string utc_now() { time_t t = time(nullptr); char b[32]; strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t)); return b; }

struct Stats { int k = 0; double mean = NAN, sd = NAN, ci95 = NAN, median = NAN, min = NAN, max = NAN, cv = NAN; };
static double t975(int df) {  // two-sided 95 % Student t quantile
  static const double t[] = {12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
                             2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
  return df <= 0 ? NAN : df <= 30 ? t[df - 1] : df <= 60 ? 2.000 : df <= 120 ? 1.980 : 1.960;
}
static Stats stats(std::vector<double> v) {
  Stats s; s.k = (int)v.size(); if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  double sum = 0; for (double x : v) sum += x; s.mean = sum / s.k;
  s.median = s.k % 2 ? v[s.k / 2] : 0.5 * (v[s.k / 2 - 1] + v[s.k / 2]); s.min = v.front(); s.max = v.back();
  if (s.k > 1) { double q = 0; for (double x : v) q += (x - s.mean) * (x - s.mean); s.sd = sqrt(q / (s.k - 1)); s.ci95 = t975(s.k - 1) * s.sd / sqrt((double)s.k); s.cv = s.sd / s.mean; }
  return s;
}
static std::string jstats(const Stats& s) {
  return "{\"k\":" + std::to_string(s.k) + ",\"mean\":" + jnum(s.mean) + ",\"sd\":" + jnum(s.sd) + ",\"ci95_halfwidth\":" + jnum(s.ci95) +
         ",\"median\":" + jnum(s.median) + ",\"min\":" + jnum(s.min) + ",\"max\":" + jnum(s.max) + ",\"cv\":" + jnum(s.cv) + "}";
}

// ------------------------------------------------------------------ NVML telemetry: samples every 100 ms, raw rows go to CSV
struct Window { double dt = 0, energy_j = NAN, watts = NAN; int tmax = -1; unsigned sm_med = 0, sm_min = 0, sm_max = 0, mem_med = 0;
                u64 reasons = 0; std::string energy_src = "none"; size_t nsamples = 0; };
struct Telemetry {
  struct S { double t, w; int temp; unsigned sm, mem; u64 reasons; };
  nvmlDevice_t dev; bool ok = false, energy_ok = false; const int period_ms = 100;
  std::atomic<bool> running{false}; std::thread th; std::vector<S> buf; double t0 = 0; u64 e0 = 0; FILE* csv = nullptr;
  void init(int cuda_dev, FILE* f) {
    csv = f;
    char bus[32]; CK(cudaDeviceGetPCIBusId(bus, sizeof bus, cuda_dev));
    ok = nvmlInit_v2() == NVML_SUCCESS && nvmlDeviceGetHandleByPciBusId_v2(bus, &dev) == NVML_SUCCESS;
    u64 e; energy_ok = ok && nvmlDeviceGetTotalEnergyConsumption(dev, &e) == NVML_SUCCESS;
    if (csv) fprintf(csv, "n,phase,method,repeat,t_s,power_w,temp_c,sm_mhz,mem_mhz,clock_event_reasons\n");
  }
  S sample() {
    S x{now_s() - t0, NAN, -1, 0, 0, 0}; unsigned p;
    nvmlTemperature_t t = {}; t.version = nvmlTemperature_v1; t.sensorType = NVML_TEMPERATURE_GPU;
    if (nvmlDeviceGetTemperatureV(dev, &t) == NVML_SUCCESS) x.temp = t.temperature;
    if (nvmlDeviceGetPowerUsage(dev, &p) == NVML_SUCCESS) x.w = p / 1000.0;
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &x.sm); nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &x.mem);
    unsigned long long r; if (nvmlDeviceGetCurrentClocksEventReasons(dev, &r) == NVML_SUCCESS) x.reasons = r;
    return x;
  }
  void start() {
    if (!ok) return;
    buf.clear(); t0 = now_s(); if (energy_ok) nvmlDeviceGetTotalEnergyConsumption(dev, &e0);
    running = true;
    th = std::thread([this] { while (running) { buf.push_back(sample()); std::this_thread::sleep_for(std::chrono::milliseconds(period_ms)); } });
  }
  Window stop(int n, const char* phase, const char* method, int rep) {
    Window w; if (!ok) return w;
    running = false; th.join(); w.dt = now_s() - t0; w.nsamples = buf.size();
    u64 e1;
    if (energy_ok && nvmlDeviceGetTotalEnergyConsumption(dev, &e1) == NVML_SUCCESS && e1 > e0) { w.energy_j = (e1 - e0) / 1000.0; w.energy_src = "nvml_energy_counter"; }
    else {  // trapezoidal integral of sampled power
      double e = 0; int used = 0;
      for (size_t i = 1; i < buf.size(); i++) if (std::isfinite(buf[i].w) && std::isfinite(buf[i - 1].w)) { e += 0.5 * (buf[i].w + buf[i - 1].w) * (buf[i].t - buf[i - 1].t); used++; }
      if (used) { w.energy_j = e * w.dt / (buf.back().t - buf.front().t); w.energy_src = "sampled_power_integral"; }
    }
    w.watts = w.energy_j / w.dt;
    std::vector<unsigned> sm, mem;
    for (auto& x : buf) {
      w.tmax = std::max(w.tmax, x.temp); w.reasons |= x.reasons; sm.push_back(x.sm); mem.push_back(x.mem);
      if (csv) fprintf(csv, "%d,%s,%s,%d,%.3f,%s,%d,%u,%u,0x%llx\n", n, phase, method, rep, x.t, jnum(x.w).c_str(), x.temp, x.sm, x.mem, x.reasons);
    }
    if (!sm.empty()) { std::sort(sm.begin(), sm.end()); std::sort(mem.begin(), mem.end());
      w.sm_med = sm[sm.size() / 2]; w.sm_min = sm.front(); w.sm_max = sm.back(); w.mem_med = mem[mem.size() / 2]; }
    return w;
  }
};

// ------------------------------------------------------------------ benchmark context (one graph size)
static const char* NAMES[4] = {"M1 CUDA cores  INT32 bitset+popc", "M2 CUDA cores  FP32 GEMM        ",
                               "M3 Tensor Cores FP16->FP32 GEMM  ", "M4 Tensor Cores INT8->INT32 GEMM "};
static const char* IDS[4] = {"M1_cuda_int32_bitset_popc", "M2_cuda_fp32_gemm", "M3_tensor_fp16_fp32acc_gemm", "M4_tensor_int8_int32acc_gemm"};
struct Ctx {
  int n, W; u64 seed, thr; uint32_t* bits; int8_t* a8; __half* a16; float* a32; int32_t* c8; float *c16, *c32;
  u64* rowsum[4]; std::vector<Gemm> g; cudaStream_t st[4];
  std::vector<int> qidx; std::vector<void*> qin, qout; std::vector<u64*> qrow; cudaStream_t qst;
  void init(int n_, double p, u64 seed_) {
    n = n_; W = n / 32; seed = seed_; thr = (u64)llround(p * 4294967296.0);
    size_t nn = (size_t)n * n;
    CK(cudaMalloc(&bits, (size_t)n * W * 4)); CK(cudaMalloc(&a8, nn)); CK(cudaMalloc(&a16, nn * 2)); CK(cudaMalloc(&a32, nn * 4));
    CK(cudaMalloc(&c8, nn * 4)); CK(cudaMalloc(&c16, nn * 4)); CK(cudaMalloc(&c32, nn * 4));
    for (int m = 0; m < 4; m++) { CK(cudaMalloc(&rowsum[m], n * 8)); CK(cudaStreamCreateWithFlags(&st[m], cudaStreamNonBlocking)); }
    k_gen_bits<<<1024, 256>>>(bits, n, W, seed, thr); CK(cudaGetLastError());
    k_expand<<<1024, 256>>>(bits, n, W, a8, a16, a32); CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    g.resize(3);
    if (!g[0].init(n, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, a32, c32)) { fprintf(stderr, "cuBLASLt: no algorithm for M2\n"); exit(2); }
    if (!g[1].init(n, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, a16, c16)) { fprintf(stderr, "cuBLASLt: no algorithm for M3\n"); exit(2); }
    if (!g[2].init(n, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, a8, c8)) { fprintf(stderr, "cuBLASLt: no algorithm for M4\n"); exit(2); }
  }
  void release() {
    for (auto& x : g) { cublasLtMatmulDescDestroy(x.op); cublasLtMatrixLayoutDestroy(x.la); cublasLtMatrixLayoutDestroy(x.lc); cudaFree(x.ws); cublasLtDestroy(x.h); }
    g.clear();
    for (auto& p : qin) cudaFree(p); for (auto& p : qout) cudaFree(p); for (auto& p : qrow) cudaFree(p);
    qin.clear(); qout.clear(); qrow.clear();
    if (qst) { cudaStreamDestroy(qst); qst = nullptr; }
    cudaFree(bits); cudaFree(a8); cudaFree(a16); cudaFree(a32); cudaFree(c8); cudaFree(c16); cudaFree(c32);
    for (int m = 0; m < 4; m++) { cudaFree(rowsum[m]); cudaStreamDestroy(st[m]); }
  }
  void init_quant(const std::vector<int>& want, std::vector<std::string>& unsupported) {  // quantisation methods follow M1-M4
    CK(cudaStreamCreateWithFlags(&qst, cudaStreamNonBlocking));
    for (int qi : want) {
      const Quant& q = QUANTS[qi];
      void* in; void* out; u64* rs;
      CK(cudaMalloc(&in, (size_t)n * n * q.bits / 8)); CK(cudaMalloc(&out, (size_t)n * n * 4)); CK(cudaMalloc(&rs, n * 8));
      Gemm gm;
      if (!gm.init(n, q.ab, q.c, q.ct, in, out)) { cudaFree(in); cudaFree(out); cudaFree(rs); unsupported.push_back(q.name); continue; }
      qidx.push_back(qi); qin.push_back(in); qout.push_back(out); qrow.push_back(rs);
      g.push_back(std::move(gm)); g.back().rebind(in, out);
    }
  }
  int nmethod() const { return 4 + (int)qidx.size(); }
  void solve(int m) {  // one complete triangle count with method m, asynchronous on one stream
    if (m >= 4) {  // quantisation method: expand A into the input format, GEMM, masked row sum
      const Quant& q = QUANTS[qidx[m - 4]];
      cudaStream_t s = qst;
      k_quant<<<1024, 256, 0, s>>>(bits, n, W, (int)q.ab, qin[m - 4]);
      g[m - 1].run(s);
      if (q.c == CUDA_R_32I) k_masked_rowsum<int32_t><<<n, 256, 0, s>>>((const int32_t*)qout[m - 4], bits, n, W, qrow[m - 4]);
      else k_masked_rowsum<float><<<n, 256, 0, s>>>((const float*)qout[m - 4], bits, n, W, qrow[m - 4]);
      CK(cudaGetLastError());
      return;
    }
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
  std::vector<u64> result(int m) { std::vector<u64> v(n); CK(cudaMemcpy(v.data(), m >= 4 ? qrow[m - 4] : rowsum[m], n * 8, cudaMemcpyDeviceToHost)); return v; }
};

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
    if (c.n == 512) {  // quantisation study on a small graph: int4/int8 must stay exact, others report their error
      std::vector<std::string> unsup; std::vector<int> all; for (int i = 0; i < NQUANT; i++) all.push_back(i);
      x.init_quant(all, unsup);
      printf(" quant:");
      for (int qi = 0; qi < (int)x.qidx.size(); qi++) {
        x.solve(4 + qi); CK(cudaStreamSynchronize(x.qst));
        Err e = measure_err(x.result(4 + qi), ref);
        printf(" %s err=%.3g", QUANTS[x.qidx[qi]].name, e.mean_abs);
        bool must_be_exact = (QUANTS[x.qidx[qi]].id == std::string("int8") || QUANTS[x.qidx[qi]].id == std::string("int4"));
        fails += (must_be_exact && e.exact_frac < 1.0);   // 0/1 inputs and INT32 accumulation are exact by construction
      }
      for (auto& u : unsup) printf(" %s=n/a", u.c_str());
      printf("\n");
    }
    printf("\n"); fails += !ok; x.release();
  }
  printf("selftest: %s\n", fails ? "FAILED" : "PASSED");
  return fails ? 1 : 0;
}

static void usage() {
  printf("trianglebench %s - triangle counting on CUDA cores vs Tensor Cores\n"
         "usage: trianglebench [options]\n"
         "  --n LIST      node counts, multiples of 256, comma separated (default 8192)\n"
         "  --p P         edge probability (default 0.5)\n"
         "  --seed S      graph seed (default 1)\n"
         "  --seconds T   measuring window per method and repetition (default 5)\n"
         "  --repeats R   repetitions per graph size (default 5); method order rotates every repetition\n"
         "  --warmup T    warm-up seconds per graph size (default 5)\n"
         "  --cooldown T  idle seconds before every measuring window (default 0)\n"
         "  --label TEXT  free text stored in the result file, e.g. system name or settings\n"
         "  --out PREFIX  writes PREFIX.json and PREFIX.telemetry.csv (default tb_<gpu>_<utc time>)\n"
         "  --quant LIST  quantisation study, comma separated from: fp32, fp16, bf16, tf32, fp8e4m3,\n"
         "                fp8e5m2, int8, int4, fp4e2m1, all, none (default: none; opt-in, each format gets\n"
         "                its own measuring windows; formats unsupported by cuBLASLt on this GPU are skipped)\n"
         "  --selftest    compare all methods with a CPU brute-force count and exit\n", VERSION);
}

int main(int argc, char** argv) {
  std::vector<int> sizes{8192}; double p = 0.5, secs = 5, warm = 5, cool = 0; int reps = 5; u64 seed = 1; bool st = false;
  std::string label, out, argline; std::vector<int> quant;
  for (int i = 0; i < argc; i++) { std::string a = argv[i]; if (a.find(' ') != std::string::npos) a = "'" + a + "'"; argline += (i ? " " : "") + a; }
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i]; auto val = [&] { if (i + 1 >= argc) { usage(); exit(2); } return std::string(argv[++i]); };
    if (a == "--n") { sizes.clear(); std::string v = val(); for (size_t q = 0, e; q < v.size(); q = e + 1) { e = v.find(',', q); if (e == std::string::npos) e = v.size(); sizes.push_back(atoi(v.substr(q, e - q).c_str())); } }
    else if (a == "--p") p = atof(val().c_str()); else if (a == "--seed") seed = strtoull(val().c_str(), 0, 10);
    else if (a == "--seconds") secs = atof(val().c_str()); else if (a == "--repeats") reps = atoi(val().c_str());
    else if (a == "--warmup") warm = atof(val().c_str()); else if (a == "--cooldown") cool = atof(val().c_str());
    else if (a == "--quant") { std::string v = val();
      for (size_t q = 0, e; q < v.size(); q = e + 1) { e = v.find(',', q); if (e == std::string::npos) e = v.size();
        std::string name = v.substr(q, e - q); if (name == "none" || name.empty()) continue;
        if (name == "all") { for (int k = 0; k < NQUANT; k++) if (std::find(quant.begin(), quant.end(), k) == quant.end()) quant.push_back(k); continue; }
        int i = 0; while (i < NQUANT && name != QUANTS[i].name) i++;
        if (i == NQUANT) { fprintf(stderr, "unknown --quant '%s' (known: fp32, fp16, bf16, tf32, fp8e4m3, fp8e5m2, int8, int4, fp4e2m1, all, none)\n", name.c_str()); return 2; }
        if (std::find(quant.begin(), quant.end(), i) == quant.end()) quant.push_back(i); } }
    else if (a == "--label") label = val(); else if (a == "--out") out = val(); else if (a == "--selftest") st = true;
    else { usage(); return a == "--help" || a == "-h" ? 0 : 2; }
  }
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  int drv = 0, rt = 0; CK(cudaDriverGetVersion(&drv)); CK(cudaRuntimeGetVersion(&rt));
  if (st) {
    printf("trianglebench %s | %s (sm_%d%d, %d SMs) | CUDA runtime %d.%d\n", VERSION, pr.name, pr.major, pr.minor, pr.multiProcessorCount, rt / 1000, (rt % 1000) / 10);
    return selftest();
  }
  if (p < 0 || p > 1 || reps < 1 || secs <= 0) { fprintf(stderr, "invalid --p, --repeats or --seconds\n"); return 2; }
  for (int n : sizes) {
    if (n < 256 || n % 256 || n >= (1 << 24)) { fprintf(stderr, "invalid --n %d (multiple of 256 required)\n", n); return 2; }
    size_t need = (size_t)n * n * 19, freeb, totb;
    for (int qi : quant) need += (size_t)n * n * (QUANTS[qi].bits / 8 + 4);
    need += 200ull << 20;
    CK(cudaMemGetInfo(&freeb, &totb));
    if (need > freeb) { fprintf(stderr, "n=%d needs %.1f GB GPU memory, %.1f GB free - use a smaller --n\n", n, need / 1e9, freeb / 1e9); return 2; }
  }
  std::string gpu = pr.name, tag;
  for (char c : gpu) tag += isalnum((unsigned char)c) ? (char)tolower(c) : '-';
  std::string t_start = utc_now();
  if (out.empty()) { std::string ts = t_start; ts.erase(std::remove(ts.begin(), ts.end(), ':'), ts.end()); ts.erase(std::remove(ts.begin(), ts.end(), '-'), ts.end()); out = "tb_" + tag + "_" + ts; }
  FILE* csv = fopen((out + ".telemetry.csv").c_str(), "w");
  if (!csv) { fprintf(stderr, "cannot write %s.telemetry.csv\n", out.c_str()); return 2; }
  Telemetry tel; tel.init(0, csv);

  // ---------------- provenance (everything that can change the numbers)
  char drvs[80] = "", uuid[96] = ""; unsigned pl = 0, smax = 0, mmax = 0, pcig = 0, pciw = 0; nvmlEnableState_t pers, eccc, eccp; int gpcoff = 0, memoff = 0;
  bool v_pl = false, v_smax = false, v_mmax = false, v_pci = false, v_pers = false, v_ecc = false, v_gpc = false, v_mem = false;
  if (tel.ok) {
    nvmlSystemGetDriverVersion(drvs, sizeof drvs); nvmlDeviceGetUUID(tel.dev, uuid, sizeof uuid);
    v_pl = nvmlDeviceGetEnforcedPowerLimit(tel.dev, &pl) == NVML_SUCCESS;
    v_smax = nvmlDeviceGetMaxClockInfo(tel.dev, NVML_CLOCK_SM, &smax) == NVML_SUCCESS;
    v_mmax = nvmlDeviceGetMaxClockInfo(tel.dev, NVML_CLOCK_MEM, &mmax) == NVML_SUCCESS;
    v_pci = !pr.integrated && nvmlDeviceGetCurrPcieLinkGeneration(tel.dev, &pcig) == NVML_SUCCESS && nvmlDeviceGetCurrPcieLinkWidth(tel.dev, &pciw) == NVML_SUCCESS;
    v_pers = nvmlDeviceGetPersistenceMode(tel.dev, &pers) == NVML_SUCCESS;
    v_ecc = nvmlDeviceGetEccMode(tel.dev, &eccc, &eccp) == NVML_SUCCESS;
    v_gpc = nvmlDeviceGetGpcClkVfOffset(tel.dev, &gpcoff) == NVML_SUCCESS;
    v_mem = nvmlDeviceGetMemClkVfOffset(tel.dev, &memoff) == NVML_SUCCESS;
  }
  struct utsname un; uname(&un);
  printf("trianglebench %s (%s) | %s (sm_%d%d, %d SMs, %.1f GB) | driver %s | CUDA runtime %d.%d | cuBLASLt %zu | %s %s\n", VERSION, GIT_COMMIT,
         pr.name, pr.major, pr.minor, pr.multiProcessorCount, pr.totalGlobalMem / 1e9, drvs, rt / 1000, (rt % 1000) / 10, cublasLtGetVersion(), un.sysname, un.machine);
  printf("power limit %s W | ECC %s | GPC offset %s MHz | mem offset %s MHz | telemetry: %s\n", v_pl ? jnum(pl / 1000.0).c_str() : "n/a",
         v_ecc ? (eccc == NVML_FEATURE_ENABLED ? "on" : "off") : "n/a", v_gpc ? std::to_string(gpcoff).c_str() : "n/a", v_mem ? std::to_string(memoff).c_str() : "n/a",
         !tel.ok ? "unavailable" : tel.energy_ok ? "NVML energy counter" : "sampled power (no energy counter)");

  std::string J = "{\"schema\":" + jstr(SCHEMA) + ",\"tool\":{\"name\":\"trianglebench\",\"version\":" + jstr(VERSION) + ",\"git_commit\":" + jstr(GIT_COMMIT) +
    ",\"build_arch\":" + jstr(BUILD_ARCH) + ",\"nvcc\":\"" + std::to_string(__CUDACC_VER_MAJOR__) + "." + std::to_string(__CUDACC_VER_MINOR__) + "." +
    std::to_string(__CUDACC_VER_BUILD__) + "\",\"host_compiler\":" + jstr(__VERSION__) + ",\"command_line\":" + jstr(argline) + "}";
  J += ",\"label\":" + jstr(label) + ",\"start_utc\":" + jstr(t_start);
  J += ",\"system\":{\"os\":" + jstr(un.sysname) + ",\"kernel\":" + jstr(un.release) + ",\"machine\":" + jstr(un.machine) + "}";
  J += ",\"gpu\":{\"name\":" + jstr(pr.name) + ",\"uuid\":" + jstr(uuid) + ",\"compute_capability\":\"" + std::to_string(pr.major) + "." + std::to_string(pr.minor) +
    "\",\"sm_count\":" + std::to_string(pr.multiProcessorCount) + ",\"memory_bytes\":" + std::to_string(pr.totalGlobalMem) +
    ",\"integrated\":" + (pr.integrated ? "true" : "false") + ",\"driver\":" + jstr(drvs) + ",\"cuda_driver_api\":" + std::to_string(drv) +
    ",\"cuda_runtime\":" + std::to_string(rt) + ",\"cublaslt\":" + std::to_string(cublasLtGetVersion()) +
    ",\"power_limit_w\":" + (v_pl ? jnum(pl / 1000.0) : "null") + ",\"max_sm_clock_mhz\":" + jint(smax, v_smax) + ",\"max_mem_clock_mhz\":" + jint(mmax, v_mmax) +
    ",\"pcie_gen\":" + jint(pcig, v_pci) + ",\"pcie_width\":" + jint(pciw, v_pci) + ",\"persistence_mode\":" + (v_pers ? (pers == NVML_FEATURE_ENABLED ? "true" : "false") : "null") +
    ",\"ecc_enabled\":" + (v_ecc ? (eccc == NVML_FEATURE_ENABLED ? "true" : "false") : "null") + ",\"gpc_clock_offset_mhz\":" + jint(gpcoff, v_gpc) +
    ",\"mem_clock_offset_mhz\":" + jint(memoff, v_mem) + "}";
  J += ",\"telemetry\":{\"available\":" + std::string(tel.ok ? "true" : "false") + ",\"energy_counter\":" + (tel.energy_ok ? "true" : "false") +
    ",\"sample_period_ms\":" + std::to_string(tel.period_ms) + ",\"file\":" + jstr(out + ".telemetry.csv") + "}";
  J += ",\"parameters\":{\"p\":" + jnum(p) + ",\"seed\":" + std::to_string(seed) + ",\"seconds\":" + jnum(secs) + ",\"repeats\":" + std::to_string(reps) +
    ",\"warmup_s\":" + jnum(warm) + ",\"cooldown_s\":" + jnum(cool) + ",\"order\":\"rotating: repetition r runs methods (r, r+1, r+2, r+3) mod 4\"}";
  J += ",\"runs\":[";

  int bad = 0;
  for (size_t si = 0; si < sizes.size(); si++) {
    const int n = sizes[si];
    std::vector<std::string> unsupported;
    Ctx x; x.init(n, p, seed); x.init_quant(quant, unsupported);
    const int nmethod = x.nmethod();
    if (!unsupported.empty()) { printf("quantisation not supported by cuBLASLt on this GPU:"); for (auto& u : unsupported) printf(" %s", u.c_str()); printf("\n"); }
    const double ops = 2.0 * n * (double)n * n;  // dense A*A, the common yardstick
    x.solve(3); CK(cudaDeviceSynchronize());
    std::vector<u64> ref = x.result(3);
    double T = (double)total(ref), nc3 = (double)n * (n - 1) * (n - 2) / 6;
    double* d_tr; float* d_cs; u64* d_deg; CK(cudaMalloc(&d_tr, 8)); CK(cudaMalloc(&d_cs, 4)); CK(cudaMalloc(&d_deg, 8));
    CK(cudaMemset(d_tr, 0, 8)); CK(cudaMemset(d_cs, 0, 4)); CK(cudaMemset(d_deg, 0, 8));
    k_stats<<<(n + 255) / 256, 256>>>(x.bits, x.rowsum[3], n, x.W, d_tr, d_cs, d_deg); CK(cudaDeviceSynchronize());
    double triples; float csum; u64 degsum;
    CK(cudaMemcpy(&triples, d_tr, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&csum, d_cs, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&degsum, d_deg, 8, cudaMemcpyDeviceToHost));
    cudaFree(d_tr); cudaFree(d_cs); cudaFree(d_deg);
    double gcc = triples > 0 ? 3 * T / triples : 0.0;
    printf("\ngraph G(n=%d, p=%.4f, seed=%llu): %llu edges, %.0f triangles (theory C(n,3)p^3 = %.0f, ratio %.5f)\n",
           n, p, seed, degsum / 2, T, nc3 * p * p * p, T / (nc3 * p * p * p));
    printf("clustering: global (FP64) %.6f, mean local (FP32) %.6f, theory p = %.6f\n", gcc, csum / n, p);

    { double t0 = now_s(); while (now_s() - t0 < warm) { x.solve(3); CK(cudaStreamSynchronize(x.st[3])); x.solve(0); CK(cudaStreamSynchronize(x.st[0])); } }

    struct Meas { std::vector<double> solve_s; Window w; bool ok; double tops, s_per_solve, j_per_solve; };
    std::vector<Meas> alone[4]; std::vector<std::vector<Meas>> qalone(x.qidx.size()); std::vector<Err> qerr(x.qidx.size());
    std::vector<std::vector<int>> par_cnt; std::vector<Window> par_w; std::vector<double> par_tops, par_sum;
    for (int r = 0; r < reps; r++) {
      printf("repetition %d/%d ...", r + 1, reps); fflush(stdout);
      double rate[4];
      for (int q = 0; q < 4; q++) {  // phase 1: each method alone, rotating order
        const int m = (r + q) % 4;
        if (cool > 0) std::this_thread::sleep_for(std::chrono::milliseconds((int)(cool * 1000)));
        x.solve(m); CK(cudaStreamSynchronize(x.st[m]));
        Meas M; tel.start(); double t0 = now_s(), t = t0;
        while (t - t0 < secs || M.solve_s.size() < 2) { x.solve(m); CK(cudaStreamSynchronize(x.st[m])); double t1 = now_s(); M.solve_s.push_back(t1 - t); t = t1; }
        double dt = t - t0; M.w = tel.stop(n, "alone", IDS[m], r + 1);
        M.ok = x.result(m) == ref; bad += !M.ok;
        M.s_per_solve = dt / M.solve_s.size(); M.tops = ops / M.s_per_solve / 1e12; M.j_per_solve = M.w.energy_j / M.solve_s.size();
        rate[m] = M.solve_s.size() / dt; alone[m].push_back(M);
      }
      for (int qi = 0; qi < (int)x.qidx.size(); qi++) {  // quantisation methods: same rotating solo phase, own stream
        const int m = 4 + qi;
        x.solve(m); CK(cudaStreamSynchronize(x.qst));
        Meas M; tel.start(); double q0 = now_s(), q1 = q0;
        while (q1 - q0 < secs || M.solve_s.size() < 2) { x.solve(m); CK(cudaStreamSynchronize(x.qst)); double tn = now_s(); M.solve_s.push_back(tn - q1); q1 = tn; }
        double qdt = q1 - q0; M.w = tel.stop(n, "alone", QUANTS[x.qidx[qi]].id, r + 1);
        M.ok = x.result(m) == ref; bad += !M.ok;
        M.s_per_solve = qdt / M.solve_s.size(); M.tops = ops / M.s_per_solve / 1e12; M.j_per_solve = M.w.energy_j / M.solve_s.size();
        qalone[qi].push_back(M); qerr[qi] = measure_err(x.result(m), ref);
      }
      // phase 2: all four at the same time
      if (cool > 0) std::this_thread::sleep_for(std::chrono::milliseconds((int)(cool * 1000)));
      std::atomic<bool> go{false}; std::vector<int> cnt(4, 0); bool eqc[4] = {};
      std::vector<std::thread> th; double t0 = 0;
      tel.start();
      for (int m = 0; m < 4; m++) th.emplace_back([&, m] {
        CK(cudaSetDevice(0));
        while (!go) std::this_thread::yield();
        while (now_s() - t0 < secs || cnt[m] < 1) { x.solve(m); CK(cudaStreamSynchronize(x.st[m])); cnt[m]++; }
        eqc[m] = x.result(m) == ref;
      });
      t0 = now_s(); go = true;
      for (auto& t : th) t.join();
      double dt = now_s() - t0; Window w = tel.stop(n, "parallel", "all", r + 1);
      double tops = 0, sum = 0;
      for (int m = 0; m < 4; m++) { tops += cnt[m] / dt * ops / 1e12; sum += cnt[m] / dt / rate[m]; bad += !eqc[m]; }
      par_cnt.push_back(cnt); par_w.push_back(w); par_tops.push_back(tops); par_sum.push_back(sum);
      printf(" done\n");
    }
    // ---------------- summary for this size
    printf("\nn=%d: %d repetitions x %.0f s, mean +- 95%% CI (Student t). eq.TOPS = 2n^3/time (work of a dense A*A)\n", n, reps, secs);
    printf("%-34s %16s %8s %6s %6s %5s %8s %10s %5s\n", "method", "eq.TOPS", "CV", "W", "T max", "MHz", "TOPS/W", "J/solve", "check");
    J += std::string(si ? "," : "") + "{\"n\":" + std::to_string(n) + ",\"graph\":{\"edges\":" + std::to_string(degsum / 2) + ",\"triangles\":" + std::to_string(total(ref)) +
      ",\"triangles_expected\":" + jnum(nc3 * p * p * p) + ",\"global_clustering_fp64\":" + jnum(gcc) + ",\"mean_local_clustering_fp32\":" + jnum(csum / n) + "}";
    J += ",\"methods\":[";
    for (int m = 0; m < 4; m++) {
      std::vector<double> v_tops, v_s, v_w, v_j, v_tpw; int tmax = -1; bool ok = true;
      for (auto& M : alone[m]) { v_tops.push_back(M.tops); v_s.push_back(M.s_per_solve); v_w.push_back(M.w.watts); v_j.push_back(M.j_per_solve);
        v_tpw.push_back(M.tops / M.w.watts); tmax = std::max(tmax, M.w.tmax); ok &= M.ok; }
      Stats s_t = stats(v_tops), s_w = stats(v_w), s_j = stats(v_j), s_e = stats(v_tpw);
      std::vector<double> mhz; for (auto& M : alone[m]) mhz.push_back(M.w.sm_med);
      printf("%-34s %8.2f +-%5.2f %7.2f%% %6.1f %6d %5.0f %8.3f %10.4g %5s\n", NAMES[m], s_t.mean, s_t.ci95, 100 * s_t.cv, s_w.mean, tmax,
             stats(mhz).median, s_e.mean, s_j.mean, ok ? "OK" : "FAIL");
      if (m == 0) { double s = stats(v_s).mean; printf("%-34s native: %.3f T popc/s (AND+POPC+ADD per 32-bit word pair)\n", "", (double)n * n * x.W / s / 1e12); }
      J += std::string(m ? "," : "") + "{\"id\":" + jstr(IDS[m]) + ",\"verified\":" + (ok ? "true" : "false");
      if (m > 0) J += ",\"cublaslt_algo\":{\"id\":" + std::to_string(x.g[m - 1].algo_id) + ",\"tile_id\":" + std::to_string(x.g[m - 1].tile_id) +
                      ",\"stages_id\":" + std::to_string(x.g[m - 1].stages_id) + "}";
      J += ",\"summary\":{\"eq_tops\":" + jstats(s_t) + ",\"seconds_per_solve\":" + jstats(stats(v_s)) + ",\"watts\":" + jstats(s_w) +
           ",\"joule_per_solve\":" + jstats(s_j) + ",\"eq_tops_per_watt\":" + jstats(s_e) + "},\"repetitions\":[";
      for (size_t r = 0; r < alone[m].size(); r++) {
        auto& M = alone[m][r];
        J += std::string(r ? "," : "") + "{\"repeat\":" + std::to_string(r + 1) + ",\"position_in_order\":" + std::to_string((m - (int)r % 4 + 4) % 4 + 1) +
          ",\"solves\":" + std::to_string(M.solve_s.size()) + ",\"window_s\":" + jnum(M.w.dt) + ",\"seconds_per_solve\":" + jnum(M.s_per_solve) +
          ",\"eq_tops\":" + jnum(M.tops) + ",\"energy_j\":" + jnum(M.w.energy_j) + ",\"energy_source\":" + jstr(M.w.energy_src) + ",\"watts\":" + jnum(M.w.watts) +
          ",\"joule_per_solve\":" + jnum(M.j_per_solve) + ",\"temp_max_c\":" + jint(M.w.tmax, M.w.tmax >= 0) + ",\"sm_mhz_median\":" + std::to_string(M.w.sm_med) +
          ",\"sm_mhz_min\":" + std::to_string(M.w.sm_min) + ",\"sm_mhz_max\":" + std::to_string(M.w.sm_max) + ",\"mem_mhz_median\":" + std::to_string(M.w.mem_med) +
          ",\"clock_event_reasons_or\":\"0x" + [&] { char b[24]; snprintf(b, sizeof b, "%llx", M.w.reasons); return std::string(b); }() + "\",\"telemetry_samples\":" +
          std::to_string(M.w.nsamples) + ",\"solve_s\":[";
        for (size_t k = 0; k < M.solve_s.size(); k++) J += (k ? "," : "") + jnum(M.solve_s[k]);
        J += "]}";
      }
      J += "]}";
    }
    Stats s_pt = stats(par_tops), s_ps = stats(par_sum); std::vector<double> pw; int ptmax = -1;
    for (auto& w : par_w) { pw.push_back(w.watts); ptmax = std::max(ptmax, w.tmax); }
    printf("%-34s %8.2f +-%5.2f %7.2f%% %6.1f %6d   sum of shares %.3f +- %.3f (1 = no gain from parallel execution)\n", "all four in parallel", s_pt.mean, s_pt.ci95,
           100 * s_pt.cv, stats(pw).mean, ptmax, s_ps.mean, s_ps.ci95);
    if (!qalone.empty()) {
      printf("\nquantisation study (same rotating solo windows as M1-M4, error against the exact INT32 result of this run):\n");
      printf("%-34s %16s %8s %6s %6s %5s %8s %10s %s\n", "input format", "eq.TOPS", "CV", "W", "T max", "MHz", "TOPS/W", "J/solve", "error vs exact");
      printf("%-34s %16s %8s %6s %6s %5s %8s %10s %s\n", "", "", "", "", "", "", "", "", "mean/max per node, exact %, total %");
    }
    J += "],\"quantization\":\"quantization_result/1\",\"quantization_methods\":[";
    for (size_t qi = 0; qi < qalone.size(); qi++) {
      const Quant& q = QUANTS[x.qidx[qi]]; const Err& e = qerr[qi];
      std::vector<double> v_tops, v_s, v_w, v_j, v_tpw, v_mhz; int tmax = -1; bool ok = true;
      for (auto& M : qalone[qi]) { v_tops.push_back(M.tops); v_s.push_back(M.s_per_solve); v_w.push_back(M.w.watts); v_j.push_back(M.j_per_solve);
        v_tpw.push_back(M.tops / M.w.watts); v_mhz.push_back(M.w.sm_med); tmax = std::max(tmax, M.w.tmax); ok &= M.ok; }
      Stats s_t = stats(v_tops), s_w = stats(v_w), s_j = stats(v_j), s_e = stats(v_tpw);
      { char b[256];
        snprintf(b, sizeof b, "%-34s", q.name); std::string line(b);
        auto add = [&](const char* fmt, auto... a) { int k = snprintf(b, sizeof b, fmt, a...); if (k > 0) line += b; };
        add("%8.2f +-%5.2f %7.2f%% %6.1f %6d %5.0f %8.3f %10.4g  err mean %8.3f max %6.0f exact %5.1f%%  dTotal %+d%%  %s\n",
            s_t.mean, s_t.ci95, 100 * s_t.cv, s_w.mean, tmax, stats(v_mhz).median, s_e.mean, s_j.mean,
            e.mean_abs, e.max_abs, 100 * e.exact_frac, (int)llround(100 * e.total_rel), ok ? "OK" : "FAIL");
        printf("%s", line.c_str());
      }
      J += std::string(qi ? "," : "") + "{\"id\":" + jstr(q.id) + ",\"input_type\":" + jstr(q.name) + ",\"input_bits\":" + std::to_string(q.bits) +
        ",\"accumulator\":" + jstr(q.ct == CUBLAS_COMPUTE_32I ? "INT32" : "FP32") + ",\"output_type\":" + jstr(q.c == CUDA_R_32I ? "INT32" : "FP32") +
        ",\"verified\":" + (ok ? "true" : "false") +
        ",\"summary\":{\"eq_tops\":" + jstats(s_t) + ",\"seconds_per_solve\":" + jstats(stats(v_s)) + ",\"watts\":" + jstats(s_w) +
          ",\"joule_per_solve\":" + jstats(s_j) + ",\"eq_tops_per_watt\":" + jstats(s_e) + "}" +
        ",\"error_vs_exact\":{\"mean_abs_per_node\":" + jnum(e.mean_abs) + ",\"max_abs_per_node\":" + jnum(e.max_abs) +
          ",\"exact_node_fraction\":" + jnum(e.exact_frac) + ",\"total_triangles\":" + std::to_string(e.total) +
          ",\"total_rel_error\":" + jnum(e.total_rel) + "}" +
        ",\"cublaslt_algo\":{\"id\":" + std::to_string(x.g[3 + (int)qi].algo_id) + ",\"tile_id\":" + std::to_string(x.g[3 + (int)qi].tile_id) +
          ",\"stages_id\":" + std::to_string(x.g[3 + (int)qi].stages_id) + "}" +
        ",\"repetitions\":[";
      for (size_t r = 0; r < qalone[qi].size(); r++) {
        auto& M = qalone[qi][r];
        J += std::string(r ? "," : "") + "{\"repeat\":" + std::to_string(r + 1) + ",\"solves\":" + std::to_string(M.solve_s.size()) +
          ",\"window_s\":" + jnum(M.w.dt) + ",\"seconds_per_solve\":" + jnum(M.s_per_solve) + ",\"eq_tops\":" + jnum(M.tops) +
          ",\"energy_j\":" + jnum(M.w.energy_j) + ",\"energy_source\":" + jstr(M.w.energy_src) + ",\"watts\":" + jnum(M.w.watts) +
          ",\"joule_per_solve\":" + jnum(M.j_per_solve) + ",\"temp_max_c\":" + jint(M.w.tmax, M.w.tmax >= 0) +
          ",\"sm_mhz_median\":" + std::to_string(M.w.sm_med) + ",\"sm_mhz_min\":" + std::to_string(M.w.sm_min) +
          ",\"sm_mhz_max\":" + std::to_string(M.w.sm_max) + ",\"mem_mhz_median\":" + std::to_string(M.w.mem_med) +
          ",\"clock_event_reasons_or\":\"0x" + [&] { char b[24]; snprintf(b, sizeof b, "%llx", M.w.reasons); return std::string(b); }() +
          "\",\"telemetry_samples\":" + std::to_string(M.w.nsamples) + ",\"solve_s\":[";
        for (size_t k = 0; k < M.solve_s.size(); k++) J += (k ? "," : "") + jnum(M.solve_s[k]);
        J += "]}";
      }
      J += "]}";
    }
    J += "]";
    if (!unsupported.empty()) {
      J += ",\"quantization_unsupported\":[";
      for (size_t ui = 0; ui < unsupported.size(); ui++) J += std::string(ui ? "," : "") + jstr(unsupported[ui]);
      J += "]";
    }
    J += ",\"parallel\":{\"summary\":{\"eq_tops\":" + jstats(s_pt) + ",\"sum_of_shares\":" + jstats(s_ps) + ",\"watts\":" + jstats(stats(pw)) + "},\"repetitions\":[";
    for (size_t r = 0; r < par_w.size(); r++) {
      J += std::string(r ? "," : "") + "{\"repeat\":" + std::to_string(r + 1) + ",\"solves\":[" + std::to_string(par_cnt[r][0]) + "," + std::to_string(par_cnt[r][1]) + "," +
        std::to_string(par_cnt[r][2]) + "," + std::to_string(par_cnt[r][3]) + "],\"window_s\":" + jnum(par_w[r].dt) + ",\"eq_tops\":" + jnum(par_tops[r]) +
        ",\"sum_of_shares\":" + jnum(par_sum[r]) + ",\"energy_j\":" + jnum(par_w[r].energy_j) + ",\"watts\":" + jnum(par_w[r].watts) +
        ",\"temp_max_c\":" + jint(par_w[r].tmax, par_w[r].tmax >= 0) + ",\"sm_mhz_median\":" + std::to_string(par_w[r].sm_med) + "}";
    }
    J += "]}}";
    x.release();
  }
  J += "],\"end_utc\":" + jstr(utc_now()) + ",\"all_verified\":" + (bad ? "false" : "true") + "}\n";
  fclose(csv);
  FILE* f = fopen((out + ".json").c_str(), "w");
  if (!f || fputs(J.c_str(), f) < 0) { fprintf(stderr, "cannot write %s.json\n", out.c_str()); return 2; }
  fclose(f);
  printf("\nresult: %s\nwritten: %s.json, %s.telemetry.csv\n", bad ? "MISMATCH - results differ between methods" : "all methods agree on every node", out.c_str(), out.c_str());
  return bad ? 1 : 0;
}
