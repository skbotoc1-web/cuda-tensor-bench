// tensorpeak.cu - tensor-core peak throughput with cuBLASLt (RTX A4000 sm_86, DGX Spark GB10 sm_121)
// Je Praezision x Matrixgroesse: Layout TN, 64 MiB Workspace, cuBLASLt-Heuristik liefert bis zu 16 Algorithmen,
// plus a search over all algo IDs x tiles x stages x swizzle (max 400); each is timed briefly (autotuning), the best then runs >= GPUBENCH_MIN_S seconds as a sustained run.
// Output: "RESULT <prec>_<n> <TFLOPS|TOPS> t=<start>-<end>" (sustained-run window) for the telemetry wrappers.
// Usage: tensorpeak [prec,...] [n,...]   e.g. tensorpeak fp16,int8 4096,8192
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
#define LT(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLASLt %d @%d\n", (int)s, __LINE__); exit(1);} } while (0)
static double now_s() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }

struct Prec { const char* name; cudaDataType ab, c, scale; cublasComputeType_t ct; int ebytes, cbytes; int bsm = -1; };  // bsm: Block-Skalierungsmodus
static const Prec PRECS[] = {
  {"fp32",     CUDA_R_32F,  CUDA_R_32F,  CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC,  4, 4},  // CUDA cores, reference
  {"tf32",     CUDA_R_32F,  CUDA_R_32F,  CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 4, 4},
  {"fp16",     CUDA_R_16F,  CUDA_R_16F,  CUDA_R_32F, CUBLAS_COMPUTE_32F,           2, 2},  // FP32-Akkumulator
  {"fp16acc16",CUDA_R_16F,  CUDA_R_16F,  CUDA_R_16F, CUBLAS_COMPUTE_16F,           2, 2},
  {"bf16",     CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F,           2, 2},
  {"int8",     CUDA_R_8I,   CUDA_R_32I,  CUDA_R_32I, CUBLAS_COMPUTE_32I,           1, 4},
  // FP8 e4m3 (tensor scale 1) and NVFP4 = e2m1 with one UE4M3 scale per 16 elements (block-scaled); need sm_89+/sm_100+
  {"fp8",      CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F,       1, 2},
  {"nvfp4",    CUDA_R_4F_E2M1, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F,       0, 2, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
};

// Host reference for fp8/nvfp4: decode operands, 64 samples in double against the BF16 result
static double dec_e4m3(uint8_t b) { int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  double v = e ? (1 + m / 8.0) * ldexp(1.0, e - 7) : m / 8.0 * ldexp(1.0, -6); return s ? -v : v; }
static double dec_e2m1(uint8_t q) { static const double t[8] = {0, .5, 1, 1.5, 2, 3, 4, 6}; return (q & 8) ? -t[q & 7] : t[q & 7]; }
static double dec_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static int verify(const Prec& p, int n, const void* dA, const void* dB, const void* dD) {
  size_t eb = p.bsm >= 0 ? (size_t)n * n / 2 : (size_t)n * n;
  std::vector<uint8_t> a(eb), b(eb); std::vector<uint16_t> d((size_t)n * n);
  CK(cudaMemcpy(a.data(), dA, eb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dB, eb, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(d.data(), dD, d.size() * 2, cudaMemcpyDeviceToHost));
  auto el = [&](const std::vector<uint8_t>& v, size_t idx) { return p.bsm >= 0
      ? dec_e2m1((idx & 1) ? v[idx / 2] >> 4 : v[idx / 2] & 15) : dec_e4m3(v[idx]); };
  int bad = 0; double worst = 0;
  for (int s = 0; s < 64; s++) {
    size_t i = (s * 2654435761u) % n, j = (s * 40503u + 17) % n;   // D(i,j) = sum_k A(k,i) B(k,j), TN, ld = n
    double ref = 0, mag = 0;
    for (int k = 0; k < n; k++) { double t = el(a, (size_t)i * n + k) * el(b, (size_t)j * n + k); ref += t; mag += fabs(t); }
    double got = dec_bf16(d[i + (size_t)j * n]), err = fabs(got - ref) / (mag + 1e-30);
    worst = std::max(worst, err); if (err > 1e-2) bad++;
  }
  printf("VERIFY %s_%d %s (64 samples, max. rel. error %.2e)\n", p.name, n, bad ? "MISMATCH" : "OK", worst);
  return bad;
}

__global__ void fill_bytes(uint8_t* p, size_t n, uint8_t mod) {  // small values: no Inf/NaN in any format
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    p[i] = (i & 1) ? 0x3c : (uint8_t)(i % mod);
}

int main(int argc, char** argv) {
  double MIN_S = getenv("GPUBENCH_MIN_S") ? atof(getenv("GPUBENCH_MIN_S")) : 5;
  std::string plist = argc > 1 ? argv[1] : "fp32,tf32,fp16,fp16acc16,bf16,int8";
  std::string nlist = argc > 2 ? argv[2] : "2048,4096,8192,12288";
  std::vector<int> sizes; for (char* t = strtok(&nlist[0], ","); t; t = strtok(nullptr, ",")) sizes.push_back(atoi(t));

  cublasLtHandle_t lt; LT(cublasLtCreate(&lt));
  const size_t WS = 64ull << 20; void* ws; CK(cudaMalloc(&ws, WS));
  int nmax = *std::max_element(sizes.begin(), sizes.end());
  size_t maxbytes = (size_t)nmax * nmax * 4;
  void *A, *B, *C; CK(cudaMalloc(&A, maxbytes)); CK(cudaMalloc(&B, maxbytes)); CK(cudaMalloc(&C, maxbytes));
  fill_bytes<<<480, 256>>>((uint8_t*)A, maxbytes, 7); fill_bytes<<<480, 256>>>((uint8_t*)B, maxbytes, 5);
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  printf("%-10s %6s %5s %9s %9s %9s  %s\n", "format", "n", "cand.", "best", "heurist.", "worst", "(autotuning, TFLOPS/TOPS)");

  // Scale tensors for block-scaled formats: roundup(n,128) x roundup(n/16,4) bytes, all 1.0 (UE4M3 0x38).
  // A uniform scale makes the result independent of the swizzled scale layout; the check covers arithmetic + decoding, not the layout.
  size_t sfbytes = (size_t)((nmax + 127) / 128 * 128) * ((nmax / 16 + 3) / 4 * 4);
  void *SA, *SB; CK(cudaMalloc(&SA, sfbytes)); CK(cudaMalloc(&SB, sfbytes));
  CK(cudaMemset(SA, 0x38, sfbytes)); CK(cudaMemset(SB, 0x38, sfbytes));
  int fails = 0;

  for (const Prec& p : PRECS) {
    if (("," + plist + ",").find("," + std::string(p.name) + ",") == std::string::npos) continue;
    for (int n : sizes) {
      cublasLtMatmulDesc_t op; LT(cublasLtMatmulDescCreate(&op, p.ct, p.scale));
      cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;  // TN: K is the fast dimension of A and B
      LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof tA));
      LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof tB));
      if (p.bsm >= 0) { int32_t m = p.bsm;
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &m, sizeof m));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &m, sizeof m));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &SA, sizeof SA));
        LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &SB, sizeof SB)); }
      cublasLtMatrixLayout_t la, lb, lc;
      LT(cublasLtMatrixLayoutCreate(&la, p.ab, n, n, n)); LT(cublasLtMatrixLayoutCreate(&lb, p.ab, n, n, n));
      LT(cublasLtMatrixLayoutCreate(&lc, p.c, n, n, n));
      cublasLtMatmulPreference_t pref; LT(cublasLtMatmulPreferenceCreate(&pref));
      LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
      // Candidates: heuristic + search over all algo IDs x tiles x stages x swizzle (validated by AlgoCheck)
      std::vector<cublasLtMatmulAlgo_t> cand;
      cublasLtMatmulHeuristicResult_t res[16]; int got = 0;
      cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 16, res, &got);
      for (int i = 0; i < got; i++) cand.push_back(res[i].algo);
      int ids[64], nid = 0;
      cublasLtMatmulAlgoGetIds(lt, p.ct, p.scale, p.ab, p.ab, p.c, p.c, 64, ids, &nid);
      for (int k = 0; k < nid && cand.size() < 400; k++) {
        cublasLtMatmulAlgo_t al0; if (cublasLtMatmulAlgoInit(lt, p.ct, p.scale, p.ab, p.ab, p.c, p.c, ids[k], &al0)) continue;
        size_t sz = 0; std::vector<uint32_t> tiles{0}, stages{0};
        if (!cublasLtMatmulAlgoCapGetAttribute(&al0, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz) && sz) {
          tiles.resize(sz / 4); cublasLtMatmulAlgoCapGetAttribute(&al0, CUBLASLT_ALGO_CAP_TILE_IDS, tiles.data(), sz, &sz); }
        if (!cublasLtMatmulAlgoCapGetAttribute(&al0, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz) && sz) {
          stages.resize(sz / 4); cublasLtMatmulAlgoCapGetAttribute(&al0, CUBLASLT_ALGO_CAP_STAGES_IDS, stages.data(), sz, &sz); }
        for (uint32_t t : tiles) for (uint32_t st : stages) for (uint32_t sw = 0; sw < 2; sw++) {
          cublasLtMatmulAlgo_t al = al0; int32_t sk = 1;
          cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_TILE_ID, &t, sizeof t);
          cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_STAGES_ID, &st, sizeof st);
          cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof sk);
          cublasLtMatmulAlgoConfigSetAttribute(&al, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &sw, sizeof sw);
          cublasLtMatmulHeuristicResult_t chk;
          if (cublasLtMatmulAlgoCheck(lt, op, la, lb, lc, lc, &al, &chk) == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= WS
              && cand.size() < 400) cand.push_back(al);
        }
      }
      got = (int)cand.size();
      if (got == 0) { printf("%-10s %6d  no algorithm\n", p.name, n); continue; }

      float f1 = 1, f0 = 0; half h1 = __float2half(1.f), h0 = __float2half(0.f); int32_t i1 = 1, i0 = 0;
      const void *al = p.scale == CUDA_R_16F ? (void*)&h1 : p.scale == CUDA_R_32I ? (void*)&i1 : (void*)&f1;
      const void *be = p.scale == CUDA_R_16F ? (void*)&h0 : p.scale == CUDA_R_32I ? (void*)&i0 : (void*)&f0;
      double flop = 2.0 * n * n * (double)n;
      auto run = [&](int a) { return cublasLtMatmul(lt, op, al, A, la, B, lb, be, C, lc, C, lc, &cand[a], ws, WS, 0); };
      int reps = std::max(3, (int)(2e11 / flop));   // Messblock ~0,2 TFLOP
      auto measure = [&](int a, int r) { CK(cudaEventRecord(e0)); for (int i = 0; i < r; i++) run(a);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return flop * r / (ms * 1e9); };

      // Warm-up (~3 s with the heuristic algo) so the screening is not biased by the cold boost clock
      { double w0 = now_s(); while (now_s() - w0 < 3) measure(0, reps); }
      // Stage 1: one short block per candidate; stage 2: top 8 + heuristic, 3 blocks each, median
      std::vector<std::pair<double, int>> sc;
      for (int a = 0; a < got; a++) { if (run(a) != CUBLAS_STATUS_SUCCESS) { cudaGetLastError(); continue; }
        CK(cudaDeviceSynchronize()); sc.push_back({measure(a, std::max(1, reps / 3)), a}); }
      if (sc.empty()) { printf("%-10s %6d  no runnable algorithm\n", p.name, n); continue; }
      std::sort(sc.rbegin(), sc.rend());
      double worstT = sc.back().first;
      std::vector<int> fin; for (size_t i = 0; i < std::min<size_t>(8, sc.size()); i++) fin.push_back(sc[i].second);
      if (std::find(fin.begin(), fin.end(), 0) == fin.end()) fin.push_back(0);
      int best = fin[0]; double bestT = 0, heurT = 0;
      for (int a : fin) { double m[3] = {measure(a, reps), measure(a, reps), measure(a, reps)}; std::sort(m, m + 3);
        if (a == 0) heurT = m[1];
        if (m[1] > bestT) { bestT = m[1]; best = a; } }
      got = (int)sc.size();
      int tid = 0, sid = 0; size_t w;
      cublasLtMatmulAlgoConfigGetAttribute(&cand[best], CUBLASLT_ALGO_CONFIG_TILE_ID, &tid, 4, &w);
      cublasLtMatmulAlgoConfigGetAttribute(&cand[best], CUBLASLT_ALGO_CONFIG_STAGES_ID, &sid, 4, &w);
      printf("%-10s %6d %5d %9.2f %9.2f %9.2f  tile=%d stages=%d\n", p.name, n, got, bestT, heurT, worstT, tid, sid); fflush(stdout);

      // Sustained run with the best algorithm, median over blocks
      std::vector<double> v; double T0 = now_s();
      while (now_s() - T0 < MIN_S || v.size() < 3) {
        CK(cudaEventRecord(e0)); for (int r = 0; r < reps; r++) run(best); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); v.push_back(flop * reps / (ms * 1e9));
      }
      std::sort(v.begin(), v.end());
      printf("RESULT %s_%d %.2f t=%.1f-%.1f\n", p.name, n, v[v.size() / 2], T0, now_s()); fflush(stdout);
      if (p.ab == CUDA_R_8F_E4M3 || p.bsm >= 0) {
        if (getenv("TENSORPEAK_NEGTEST")) CK(cudaMemset(A, 0x30, maxbytes));  // overwrite A after the GEMM -> must report MISMATCH
        fails += verify(p, n, A, B, C);
        if (getenv("TENSORPEAK_NEGTEST")) { fill_bytes<<<480, 256>>>((uint8_t*)A, maxbytes, 7); CK(cudaDeviceSynchronize()); }
      }
      cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
      cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    }
  }
  if (fails) { printf("FAIL tensorpeak: %d samples wrong\n", fails); return 2; }
  printf("OK tensorpeak done\n");
  return 0;
}
