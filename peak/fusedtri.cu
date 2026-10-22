// fusedtri.cu - prototype: triangles per vertex t2[i] = sum_j (A*A)[i][j] * A[i][j]  (= 2 x triangles at i)
// Compares two paths on the same 0/1 adjacency matrix (INT8, symmetric, zero diagonal):
//   unfused: cuBLASLt INT8 GEMM C = A*A (INT32, n x n written to memory) + SIMT kernel for mask/row sum
//   fused  : CUDA Tile kernel (CUDA 13.3+, cuda_tile.h): mma over K, mask + row sum in the epilogue,
//            atomic add per row - C is never written to memory.
// Both results must match exactly per vertex (and match an independent SIMT reference on samples).
// Output: "RESULT <path>_<n> <eqTOPS> t=<start>-<end>" for the telemetry wrappers; eqTOPS = 2 n^3 / time.
// Build: nvcc -O3 -std=c++20 -enable-tile -arch=sm_86 fusedtri.cu -lcublasLt -o fusedtri   (needs tileiras, CUDA 13.3+)
// Usage: fusedtri [n,...] [density]     FUSEDTRI_NEGTEST=1 corrupts the fused result -> exit 2
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_tile.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
#define LT(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLASLt %d @%d\n", (int)s, __LINE__); exit(1);} } while (0)
static double now_s() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

__device__ __host__ inline uint32_t mix(uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
// symmetric random graph: edge (i,j), i<j, with probability p
__global__ void gen(int8_t* A, int n, uint32_t thr) {
  size_t nn = (size_t)n * n;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < nn; idx += (size_t)gridDim.x * blockDim.x) {
    uint32_t i = idx / n, j = idx % n, a = i < j ? i : j, b = i < j ? j : i;
    A[idx] = (i != j) && mix(a * 2654435761u ^ mix(b + 0x9e3779b9u)) < thr;
  }
}
// unfused epilogue: t2[i] = sum_j C[i][j] * A[i][j]   (one block per row)
__global__ void mask_rowsum(const int32_t* C, const int8_t* A, int n, long long* t2) {
  size_t row = blockIdx.x; long long s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) s += A[row * n + j] ? C[row * n + j] : 0;
  for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ long long w[32]; if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = s; __syncthreads();
  if (threadIdx.x == 0) { long long t = 0; for (int k = 0; k < (int)(blockDim.x >> 5); k++) t += w[k]; t2[row] = t; }
}
// independent sample reference (no tensor cores): t2[i] = sum_{j,k} A[i][j] A[i][k] A[k][j]
__global__ void ref_rows(const int8_t* A, int n, const int* rows, int nr, long long* out) {
  int r = blockIdx.x; if (r >= nr) return; size_t i = rows[r]; long long s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) if (A[i * n + j])
    for (int k = 0; k < n; k++) s += A[i * n + k] & A[(size_t)k * n + j];
  atomicAdd((unsigned long long*)&out[r], (unsigned long long)s);
}

#ifndef FT_G
#define FT_G 8
#endif
namespace ct = cuda::tiles;
// V: 0 = naive (B tile (TK,TN) from rows k), 1 = n divisible + load B as (TN,TK) from rows j and transpose (A is symmetric),
//    2 = like 1 plus an L2-friendly grouped block raster (default)
template <int TM, int TN, int TK, int V>
__tile_global__ void fused(const int8_t* __restrict__ A, int32_t* __restrict__ t2, std::size_t n) {
  using namespace ct::literals;
  A = ct::assume_aligned(A, 16_ic); t2 = ct::assume_aligned(t2, 16_ic);
  if constexpr (V >= 1) n = ct::assume_divisible(n, 128_ic);
  constexpr auto tm = ct::integral_constant<TM>{}; constexpr auto tn = ct::integral_constant<TN>{};
  constexpr auto tk = ct::integral_constant<TK>{};
  auto span = ct::tensor_span{A, ct::extents{n, n}};
  auto aView = ct::partition_view{span, ct::shape{tm, tk}};
  auto mView = ct::partition_view{span, ct::shape{tm, tn}};
  auto oView = ct::partition_view{ct::tensor_span{t2, ct::extents{n, std::size_t{1}}}, ct::shape{tm, 1_ic}};
  auto [bid0, bid1, bid2] = ct::bid();
  std::size_t bx = bid0, by = bid1;
  if constexpr (V == 2) {   // 1-D grid with grouped raster (like Triton GROUP_M=8): neighbouring blocks share row/column panels in L2
    std::size_t pid = bid0, npm = n / TM, npn = n / TN, G = FT_G, gid = pid / (G * npn), first = gid * G;
    std::size_t gsz = npm - first < G ? npm - first : G;
    bx = first + pid % gsz; by = (pid % (G * npn)) / gsz;
  }
  auto acc = ct::full<ct::tile<int32_t, ct::shape<TM, TN>>>(0);
  std::size_t num_k = n / TK;
  if constexpr (V == 0) {
    auto bView = ct::partition_view{span, ct::shape{tk, tn}};
    for (auto k : ct::irange(std::size_t{0}, num_k)) acc = ct::mma(aView.load(bx, k), bView.load(k, by), acc);
  } else {
    auto bView = ct::partition_view{span, ct::shape{tn, tk}};
    for (auto k : ct::irange(std::size_t{0}, num_k)) acc = ct::mma(aView.load(bx, k), ct::transpose(bView.load(by, k)), acc);
  }
  auto prod = acc * mView.load(bx, by);          // int8 mask -> int32
  auto rs = ct::sum(prod, 1_ic);                 // (TM, 1)
  oView.atomic_add(rs, ct::memory_order_relaxed_t{}, ct::thread_scope_device_t{}, bx, 0);
}

int main(int argc, char** argv) {
  double MIN_S = getenv("GPUBENCH_MIN_S") ? atof(getenv("GPUBENCH_MIN_S")) : 6;
  std::string nlist = argc > 1 ? argv[1] : "4096,8192";
  double p = argc > 2 ? atof(argv[2]) : 0.05;
  std::vector<int> sizes; for (char* t = strtok(&nlist[0], ","); t; t = strtok(nullptr, ",")) sizes.push_back(atoi(t));
#ifndef FT_TM
#define FT_TM 128
#define FT_TN 128
#define FT_TK 128   // best on A4000 and GB10 (128x256 / 256x128 collapse)
#endif
  constexpr int TM = FT_TM, TN = FT_TN, TK = FT_TK;
  cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
  const size_t WS = 64ull << 20; void* ws; CK(cudaMalloc(&ws, WS));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  int rc = 0;
  printf("fusedtri | density %.3f | tile %dx%dx%d | eqTOPS = 2 n^3 / time\n", p, TM, TN, TK);
  for (int n : sizes) {
    if (n % 128) { printf("n=%d must be a multiple of 128\n", n); rc = 2; continue; }
    size_t nn = (size_t)n * n;
    int8_t* A; int32_t *C, *t2f; long long* t2u;
    CK(cudaMalloc(&A, nn)); CK(cudaMalloc(&C, nn * 4)); CK(cudaMalloc(&t2f, n * 4)); CK(cudaMalloc(&t2u, n * 8));
    gen<<<1024, 256>>>(A, n, (uint32_t)(p * 4294967295.0)); CK(cudaDeviceSynchronize());
    // cuBLASLt INT8 TN (A symmetric: A^T = A), best of up to 8 heuristic algorithms
    cublasLtMatmulDesc_t op; LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
    cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
    LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
    cublasLtMatrixLayout_t la, lc; LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, n, n, n)); LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, n, n, n));
    cublasLtMatmulPreference_t pref; LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
    cublasLtMatmulHeuristicResult_t hr[8]; int got = 0; LT(cublasLtMatmulAlgoGetHeuristic(lt, op, la, la, lc, lc, pref, 8, hr, &got));
    int32_t one = 1, zero = 0; int best = 0;
    auto gemm = [&](int a) { LT(cublasLtMatmul(lt, op, &one, A, la, A, la, &zero, C, lc, C, lc, &hr[a].algo, ws, WS, 0)); };
    auto unfused = [&](int a) { gemm(a); mask_rowsum<<<n, 256>>>(C, A, n, t2u); };
    int V = getenv("FUSEDTRI_V") ? atoi(getenv("FUSEDTRI_V")) : 2;
    auto fused_run = [&]() { CK(cudaMemsetAsync(t2f, 0, n * 4));
      if (V == 0) fused<TM, TN, TK, 0><<<dim3(n / TM, n / TN), 1>>>(A, t2f, (std::size_t)n);
      else if (V == 1) fused<TM, TN, TK, 1><<<dim3(n / TM, n / TN), 1>>>(A, t2f, (std::size_t)n);
      else        fused<TM, TN, TK, 2><<<dim3((n / TM) * (n / TN)), 1>>>(A, t2f, (std::size_t)n); };
    auto timeit = [&](auto&& f, int reps) { CK(cudaEventRecord(e0)); for (int r = 0; r < reps; r++) f(); CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms / 1e3 / reps; };
    double flop = 2.0 * n * (double)n * n; int reps = std::max(3, (int)(2e12 / flop));
    { double bt = 1e9; for (int a = 0; a < got; a++) { unfused(a); CK(cudaDeviceSynchronize()); double t = timeit([&] { unfused(a); }, reps); if (t < bt) { bt = t; best = a; } } }
    // Correctness: fused == unfused per vertex, plus 32 samples against the SIMT reference
    unfused(best); fused_run(); CK(cudaDeviceSynchronize());
    std::vector<long long> hu(n); std::vector<int32_t> hf(n);
    CK(cudaMemcpy(hu.data(), t2u, n * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hf.data(), t2f, n * 4, cudaMemcpyDeviceToHost));
    if (getenv("FUSEDTRI_NEGTEST")) hf[n / 3] += 2;
    std::vector<int> rows(32); for (int r = 0; r < 32; r++) rows[r] = (int)(mix(r + 7) % n);
    int* drows; long long* dref; CK(cudaMalloc(&drows, 32 * 4)); CK(cudaMalloc(&dref, 32 * 8)); CK(cudaMemset(dref, 0, 32 * 8));
    CK(cudaMemcpy(drows, rows.data(), 32 * 4, cudaMemcpyHostToDevice)); ref_rows<<<32, 256>>>(A, n, drows, 32, dref);
    std::vector<long long> hr2(32); CK(cudaMemcpy(hr2.data(), dref, 32 * 8, cudaMemcpyDeviceToHost));
    long long mism = 0, refbad = 0, tri = 0;
    for (int i = 0; i < n; i++) { mism += hu[i] != hf[i]; tri += hu[i]; }
    for (int r = 0; r < 32; r++) refbad += hr2[r] != hu[rows[r]];
    bool ok = !mism && !refbad && tri % 6 == 0;
    printf("CHECK n=%d triangles=%lld fused!=unfused:%lld samples!=reference:%lld/32 -> %s\n", n, tri / 6, mism, refbad, ok ? "OK" : "MISMATCH");
    if (!ok) rc = 2;
    // Sustained runs (each >= MIN_S, median of blocks)
    for (int w = 0; w < 2; w++) {
      std::vector<double> v; double T0 = now_s();
      while (now_s() - T0 < MIN_S || v.size() < 3) v.push_back(flop / 1e12 / (w ? timeit(fused_run, reps) : timeit([&] { unfused(best); }, reps)));
      std::sort(v.begin(), v.end());
      printf("RESULT %s_%d %.2f t=%.1f-%.1f\n", w ? (V == 2 ? "fused" : V == 1 ? "fusedv1" : "fusedv0") : "unfused", n, v[v.size() / 2], T0, now_s()); fflush(stdout);
    }
    { double tg = timeit([&] { gemm(best); }, reps); printf("INFO n=%d GEMM only %.2f eqTOPS\n", n, flop / 1e12 / tg); }
    cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    CK(cudaFree(A)); CK(cudaFree(C)); CK(cudaFree(t2f)); CK(cudaFree(t2u)); CK(cudaFree(drows)); CK(cudaFree(dref));
  }
  printf(rc ? "FAIL fusedtri\n" : "OK fusedtri\n");
  return rc;
}
