// sparsepeak.cu - 2:4 structured-sparsity Tensor-Core peak with cuSPARSELt
// A (M x K) is pruned to 2:4 and compressed, B dense, D = A * B^T  (row-major, opA=N, opB=T: K fastest for both).
// Throughput is reported as EFFECTIVE (dense-equivalent) TFLOPS/TOPS = 2*n^3 / time, the way NVIDIA's
// "with sparsity" datasheet numbers are defined; the tensor cores really execute half of it.
// cusparseLtMatmulSearch autotunes algorithm id and split-k; then 3 s warm-up and a >= GPUBENCH_MIN_S
// sustained run (median of blocks). Output "RESULT sp_<fmt>_<n> <value> t=<start>-<end>" for benchrun.py.
// Correctness: 2048 sampled output elements are recomputed from the pruned A and B in double precision.
// Build: nvcc -O3 -arch=native -I/usr/include/libcusparseLt/13 sparsepeak.cu
//        -L/usr/lib/x86_64-linux-gnu/libcusparseLt/13 -lcusparseLt -o sparsepeak
// Usage: sparsepeak [fmt,...] [n,...]     fmt: int8, fp16, bf16, tf32, fp8, fp4 (fp8/fp4 need sm_89+/sm_100+)
// SPARSEPEAK_NEGTEST=1 refills A unpruned before the check: every format must then report MISMATCH (exit 2).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cusparseLt.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
#define SP(x) do { cusparseStatus_t s = (x); if (s != CUSPARSE_STATUS_SUCCESS) { fprintf(stderr, "cuSPARSELt %d (%s) @%d\n", (int)s, cusparseLtGetErrorString(s), __LINE__); exit(1);} } while (0)
static double now_s() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

struct Fmt { const char* name; cudaDataType ab, cd; cusparseComputeType ct; int eb, cb; const char* unit; };
static const Fmt FMTS[] = {
  {"int8", CUDA_R_8I,   CUDA_R_32I,  CUSPARSE_COMPUTE_32I, 1, 4, "TOPS"},
  {"fp16", CUDA_R_16F,  CUDA_R_16F,  CUSPARSE_COMPUTE_32F, 2, 2, "TFLOPS"},
  {"bf16", CUDA_R_16BF, CUDA_R_16BF, CUSPARSE_COMPUTE_32F, 2, 2, "TFLOPS"},
  {"tf32", CUDA_R_32F,  CUDA_R_32F,  CUSPARSE_COMPUTE_32F, 4, 4, "TFLOPS"},
  // GB10 sm_121 and later: FP8 e4m3 and NVFP4-style e2m1 with UE4M3 block scales (all 1.0), output BF16
  {"fp8",  CUDA_R_8F_E4M3, CUDA_R_16BF, CUSPARSE_COMPUTE_32F, 1, 2, "TFLOPS"},
  {"fp4",  CUDA_R_4F_E2M1, CUDA_R_16BF, CUSPARSE_COMPUTE_32F, 1, 2, "TFLOPS"},   // eb=1: over-allocated, 2 per byte
};

// small integer values -3..3: exact in every input format
__device__ __host__ inline int val(size_t i, unsigned salt) {
  unsigned x = (unsigned)i * 2654435761u ^ salt; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (int)(x % 7) - 3;
}
template <typename T> __device__ inline T cvt(int v);
template <> __device__ inline int8_t cvt<int8_t>(int v) { return (int8_t)v; }
template <> __device__ inline __half cvt<__half>(int v) { return __int2half_rn(v); }
template <> __device__ inline __nv_bfloat16 cvt<__nv_bfloat16>(int v) { return __int2bfloat16_rn(v); }
template <> __device__ inline float cvt<float>(int v) { return (float)v; }
template <> __device__ inline __nv_fp8_e4m3 cvt<__nv_fp8_e4m3>(int v) { return __nv_fp8_e4m3((float)v); }
template <typename T> __device__ inline double tod(T v);
template <> __device__ inline double tod<int8_t>(int8_t v) { return v; }
template <> __device__ inline double tod<int32_t>(int32_t v) { return v; }
template <> __device__ inline double tod<__half>(__half v) { return __half2float(v); }
template <> __device__ inline double tod<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ inline double tod<float>(float v) { return v; }
template <> __device__ inline double tod<__nv_fp8_e4m3>(__nv_fp8_e4m3 v) { return (float)v; }
// e2m1 nibble <-> -3..3 (exact: 0, 1, 2, 3 = codes 0, 2, 4, 5; bit 3 = sign); element 2i = low nibble
__device__ inline unsigned enc4(int v) { const unsigned c[4] = {0, 2, 4, 5}; return (v < 0 ? 8u : 0u) | c[v < 0 ? -v : v]; }
__device__ inline double dec4(unsigned q) { const double t[8] = {0, .5, 1, 1.5, 2, 3, 4, 6}; return (q & 8) ? -t[q & 7] : t[q & 7]; }
__device__ inline double el4(const uint8_t* p, size_t idx) { return dec4((idx & 1) ? p[idx / 2] >> 4 : p[idx / 2] & 15); }
__global__ void fill4(uint8_t* p, size_t nbytes, unsigned salt) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nbytes; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (uint8_t)(enc4(val(2 * i, salt)) | enc4(val(2 * i + 1, salt)) << 4);
}

template <typename T> __global__ void fill(T* p, size_t n, unsigned salt) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = cvt<T>(val(i, salt));
}
// reference for sampled (i,j): D[i][j] = sum_k A[i][k] * B[j][k]   (A pruned, both row-major, K fastest)
template <typename TA, typename TD> __global__ void check(const TA* A, const TA* B, const TD* D, int n, int samples,
                                                          double* maxrel, unsigned long long* bad, unsigned long long* nz) {
  int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= samples) return;
  size_t i = (size_t)val(s * 7 + 1, 11u) + 3, j = (size_t)val(s * 13 + 5, 17u) + 3;   // spread over the matrix
  i = (i * 2654435761u + s * 40503u) % n; j = (j * 2246822519u + s * 69069u) % n;
  double ref = 0; unsigned long long z = 0;
  for (int k = 0; k < n; k++) { double a = tod(A[i * n + k]); z += (a == 0); ref += a * tod(B[j * n + k]); }
  double got = tod(D[i * n + j]);
  double rel = fabs(got - ref) / fmax(fabs(ref), 64.0);
  atomicAdd(nz, z);
  if (rel > 2e-2) atomicAdd(bad, 1ull);
  unsigned long long* m = (unsigned long long*)maxrel;   // positive doubles compare like their bit patterns
  atomicMax(m, (unsigned long long)__double_as_longlong(rel));
}

__global__ void check4(const uint8_t* A, const uint8_t* B, const __nv_bfloat16* D, int n, int samples,
                       double* maxrel, unsigned long long* bad, unsigned long long* nz) {
  int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= samples) return;
  size_t i = (size_t)val(s * 7 + 1, 11u) + 3, j = (size_t)val(s * 13 + 5, 17u) + 3;
  i = (i * 2654435761u + s * 40503u) % n; j = (j * 2246822519u + s * 69069u) % n;
  double ref = 0; unsigned long long z = 0;
  for (int k = 0; k < n; k++) { double a = el4(A, i * n + k); z += (a == 0); ref += a * el4(B, j * n + k); }
  double rel = fabs((double)__bfloat162float(D[i * n + j]) - ref) / fmax(fabs(ref), 64.0);
  atomicAdd(nz, z); if (rel > 2e-2) atomicAdd(bad, 1ull);
  atomicMax((unsigned long long*)maxrel, (unsigned long long)__double_as_longlong(rel));
}

template <typename TA, typename TD> static void fill_run_check(void* A, void* B, void* D, int n, bool doFill, bool doCheck,
                                                              double* dmax, unsigned long long* dbad, unsigned long long* dnz) {
  size_t nn = (size_t)n * n;
  if (doFill) { fill<TA><<<480, 256>>>((TA*)A, nn, 0x1234u); fill<TA><<<480, 256>>>((TA*)B, nn, 0xabcdu); }
  if (doCheck) check<TA, TD><<<(2048 + 127) / 128, 128>>>((TA*)A, (TA*)B, (TD*)D, n, 2048, dmax, dbad, dnz);
}
static void dispatch(const Fmt& f, void* A, void* B, void* D, int n, bool doFill, bool doCheck, double* m, unsigned long long* b, unsigned long long* z) {
  if (!strcmp(f.name, "int8")) fill_run_check<int8_t, int32_t>(A, B, D, n, doFill, doCheck, m, b, z);
  else if (!strcmp(f.name, "fp16")) fill_run_check<__half, __half>(A, B, D, n, doFill, doCheck, m, b, z);
  else if (!strcmp(f.name, "bf16")) fill_run_check<__nv_bfloat16, __nv_bfloat16>(A, B, D, n, doFill, doCheck, m, b, z);
  else if (!strcmp(f.name, "fp8")) fill_run_check<__nv_fp8_e4m3, __nv_bfloat16>(A, B, D, n, doFill, doCheck, m, b, z);
  else if (!strcmp(f.name, "fp4")) { size_t nb = (size_t)n * n / 2;
    if (doFill) { fill4<<<480, 256>>>((uint8_t*)A, nb, 0x1234u); fill4<<<480, 256>>>((uint8_t*)B, nb, 0xabcdu); }
    if (doCheck) check4<<<(2048 + 127) / 128, 128>>>((uint8_t*)A, (uint8_t*)B, (__nv_bfloat16*)D, n, 2048, m, b, z); }
  else fill_run_check<float, float>(A, B, D, n, doFill, doCheck, m, b, z);
}

int main(int argc, char** argv) {
  double MIN_S = getenv("GPUBENCH_MIN_S") ? atof(getenv("GPUBENCH_MIN_S")) : 8;
  std::string flist = argc > 1 ? argv[1] : "int8,fp16,bf16,tf32";
  std::string nlist = argc > 2 ? argv[2] : "8192,12288";
  std::vector<int> sizes; for (char* t = strtok(&nlist[0], ","); t; t = strtok(nullptr, ",")) sizes.push_back(atoi(t));
  cusparseLtHandle_t h; SP(cusparseLtInit(&h));
  int ver = 0; cusparseLtGetVersion(&h, &ver);
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  printf("sparsepeak | %s sm_%d%d | cuSPARSELt %d | 2:4 sparse A, values -3..3, effective = dense-equivalent 2n^3\n",
         pr.name, pr.major, pr.minor, ver);
  printf("%-6s %6s %6s %10s %10s %8s %8s %s\n", "Format", "n", "algId", "effektiv", "real-exec", "zeros_A", "max_rel", "check");
  double *dmax; unsigned long long *dbad, *dnz; CK(cudaMalloc(&dmax, 8)); CK(cudaMalloc(&dbad, 8)); CK(cudaMalloc(&dnz, 8));
  int* dvalid; CK(cudaMalloc(&dvalid, sizeof(int)));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  int rc = 0;

  for (const Fmt& f : FMTS) {
    if (("," + flist + ",").find("," + std::string(f.name) + ",") == std::string::npos) continue;
    for (int n : sizes) {
      size_t nn = (size_t)n * n;
      void *A, *B, *D; CK(cudaMalloc(&A, nn * f.eb)); CK(cudaMalloc(&B, nn * f.eb)); CK(cudaMalloc(&D, nn * f.cb));
      dispatch(f, A, B, D, n, true, false, nullptr, nullptr, nullptr); CK(cudaDeviceSynchronize());
      const cusparseOrder_t ord = CUSPARSE_ORDER_ROW; const unsigned al = 16;
      cusparseLtMatDescriptor_t mA, mB, mC;
      SP(cusparseLtStructuredDescriptorInit(&h, &mA, n, n, n, al, f.ab, ord, CUSPARSELT_SPARSITY_50_PERCENT));
      SP(cusparseLtDenseDescriptorInit(&h, &mB, n, n, n, al, f.ab, ord));   // B stored N x K, used transposed
      SP(cusparseLtDenseDescriptorInit(&h, &mC, n, n, n, al, f.cd, ord));
      cusparseLtMatmulDescriptor_t mm;
      cusparseStatus_t s = cusparseLtMatmulDescriptorInit(&h, &mm, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                                          &mA, &mB, &mC, &mC, f.ct);
      if (s != CUSPARSE_STATUS_SUCCESS) { printf("%-6s %6d  not supported (%s)\n", f.name, n, cusparseLtGetErrorString(s));
        CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(D)); continue; }
      void *SA = nullptr, *SB = nullptr;   // block scales for fp4: generous size, all UE4M3 1.0 -> layout-independent
      if (f.ab == CUDA_R_4F_E2M1) {
        size_t sfb = (size_t)n * n / 8; CK(cudaMalloc(&SA, sfb)); CK(cudaMalloc(&SB, sfb));
        CK(cudaMemset(SA, 0x38, sfb)); CK(cudaMemset(SB, 0x38, sfb));
        int sm = CUSPARSELT_MATMUL_MATRIX_SCALE_VEC32_UE4M3;
        s = cusparseLtMatmulDescSetAttribute(&h, &mm, CUSPARSELT_MATMUL_A_SCALE_MODE, &sm, sizeof sm);
        if (!s) s = cusparseLtMatmulDescSetAttribute(&h, &mm, CUSPARSELT_MATMUL_B_SCALE_MODE, &sm, sizeof sm);
        if (!s) s = cusparseLtMatmulDescSetAttribute(&h, &mm, CUSPARSELT_MATMUL_A_SCALE_POINTER, &SA, sizeof SA);
        if (!s) s = cusparseLtMatmulDescSetAttribute(&h, &mm, CUSPARSELT_MATMUL_B_SCALE_POINTER, &SB, sizeof SB);
        if (s) { printf("%-6s %6d  scale setup not supported (%s)\n", f.name, n, cusparseLtGetErrorString(s)); continue; }
      }
      cusparseLtMatmulAlgSelection_t sel; SP(cusparseLtMatmulAlgSelectionInit(&h, &sel, &mm, CUSPARSELT_MATMUL_ALG_DEFAULT));
      int iters = 5; SP(cusparseLtMatmulAlgSetAttribute(&h, &sel, CUSPARSELT_MATMUL_SEARCH_ITERATIONS, &iters, sizeof iters));
      cusparseLtMatmulPlan_t plan; SP(cusparseLtMatmulPlanInit(&h, &plan, &mm, &sel));
      // prune A in place to 2:4 and verify the pattern
      SP(cusparseLtSpMMAPrune(&h, &mm, A, A, CUSPARSELT_PRUNE_SPMMA_TILE, st));
      SP(cusparseLtSpMMAPruneCheck(&h, &mm, A, dvalid, st));
      int valid = 1; CK(cudaMemcpyAsync(&valid, dvalid, sizeof valid, cudaMemcpyDeviceToHost, st)); CK(cudaStreamSynchronize(st));
      size_t csz, cbsz; SP(cusparseLtSpMMACompressedSize(&h, &plan, &csz, &cbsz));
      void *Ac, *Acb; CK(cudaMalloc(&Ac, csz)); CK(cudaMalloc(&Acb, cbsz ? cbsz : 16));
      SP(cusparseLtSpMMACompress(&h, &plan, A, Ac, Acb, st));
      float f1 = 1.f, f0 = 0.f;
      size_t wsz = 0; SP(cusparseLtMatmulGetWorkspace(&h, &plan, &wsz)); void* ws = nullptr; if (wsz) CK(cudaMalloc(&ws, wsz));
      // autotuning (algorithm id, split-k)
      SP(cusparseLtMatmulSearch(&h, &plan, &f1, Ac, B, &f0, D, D, ws, &st, 1));
      size_t wsz2 = 0; SP(cusparseLtMatmulGetWorkspace(&h, &plan, &wsz2));
      if (wsz2 > wsz) { if (ws) CK(cudaFree(ws)); CK(cudaMalloc(&ws, wsz2)); }
      int algId = -1; cusparseLtMatmulAlgGetAttribute(&h, &sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &algId, sizeof algId);
      double flop = 2.0 * n * (double)n * n;
      auto run = [&]() { SP(cusparseLtMatmul(&h, &plan, &f1, Ac, B, &f0, D, D, ws, &st, 1)); };
      int reps = std::max(3, (int)(4e11 / flop));
      auto block = [&]() { CK(cudaEventRecord(e0, st)); for (int r = 0; r < reps; r++) run(); CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return flop * reps / (ms * 1e9); };
      { double w0 = now_s(); while (now_s() - w0 < 3) block(); }
      // correctness on the final result
      CK(cudaMemset(dmax, 0, 8)); CK(cudaMemset(dbad, 0, 8)); CK(cudaMemset(dnz, 0, 8)); CK(cudaStreamSynchronize(st));
      if (getenv("SPARSEPEAK_NEGTEST")) dispatch(f, A, B, D, n, true, false, nullptr, nullptr, nullptr);  // un-prune A: check must fail
      dispatch(f, A, B, D, n, false, true, dmax, dbad, dnz); CK(cudaDeviceSynchronize());
      double mrel; unsigned long long bad, nz; CK(cudaMemcpy(&mrel, dmax, 8, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&bad, dbad, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&nz, dnz, 8, cudaMemcpyDeviceToHost));
      double zfrac = nz / (2048.0 * n);   // >= 0.5 after 2:4 pruning (+ natural zeros of the -3..3 values)
      bool ok = valid == 0 && bad == 0 && zfrac >= 0.5;
      if (!ok) rc = 2;
      std::vector<double> v; double T0 = now_s();
      while (now_s() - T0 < MIN_S || v.size() < 3) v.push_back(block());
      std::sort(v.begin(), v.end()); double med = v[v.size() / 2];
      printf("%-6s %6d %6d %10.2f %10.2f %8.3f %8.1e %s\n", f.name, n, algId, med, med / 2, zfrac, mrel,
             ok ? "OK" : (valid ? "PRUNE-PATTERN-INVALID" : "MISMATCH"));
      printf("RESULT sp_%s_%d %.2f t=%.1f-%.1f\n", f.name, n, med, T0, now_s()); fflush(stdout);
      if (ws) CK(cudaFree(ws)); CK(cudaFree(Ac)); CK(cudaFree(Acb));
      cusparseLtMatmulPlanDestroy(&plan); cusparseLtMatmulAlgSelectionDestroy(&sel);
      cusparseLtMatDescriptorDestroy(&mA); cusparseLtMatDescriptorDestroy(&mB); cusparseLtMatDescriptorDestroy(&mC);
      CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(D)); if (SA) { CK(cudaFree(SA)); CK(cudaFree(SB)); }
    }
  }
  printf(rc ? "FAIL sparsepeak\n" : "OK sparsepeak done\n");
  cusparseLtDestroy(&h);
  return rc;
}
