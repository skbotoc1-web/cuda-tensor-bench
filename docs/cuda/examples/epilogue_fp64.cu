// epilogue_fp64.cu - a hidden FP64 bottleneck in a "memory-bound" kernel.
//
// k_masked_rowsum (src/trianglebench.cu) reads the n x n result C = A*A once and adds C_ij for every edge (i,j).
// It converts each element with llrint((double)C_ij). That is exact and simple, but it executes FP64
// instructions, and many GPUs run FP64 at a small fraction of the FP32 rate. Nsight Compute on an RTX A4000
// shows the FP64 pipe 91 % busy while DRAM is only 62 % busy: the kernel is compute-bound on FP64.
//
// This program times three variants on the same data and checks that they give identical sums:
//   v0  llrint((double)C)    - as in trianglebench 1.3.0 (INT32 and FP32 C)
//   v1  (u64)C               - pure integer add, for INT32 C (M4)
//   v2  (u64)__float2int_rn  - FP32 -> INT32 conversion, exact for integer values below 2^24 (M2, M3)
//
// Build: nvcc -O3 -arch=native -o epilogue_fp64 epilogue_fp64.cu
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
typedef unsigned long long u64;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

__device__ inline uint32_t bit(const uint32_t* bits, int W, int i, int j) { return (bits[(size_t)i * W + (j >> 5)] >> (j & 31)) & 1u; }

template <typename T, int V>
__global__ void masked_rowsum(const T* C, const uint32_t* bits, int n, int W, u64* rowsum) {
  const int i = blockIdx.x;
  u64 s = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x)
    if (bit(bits, W, i, j)) {
      T c = C[(size_t)i * n + j];
      if (V == 0) s += (u64)llrint((double)c);
      else if (V == 1) s += (u64)c;
      else s += (u64)__float2int_rn((float)c);
    }
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  __shared__ u64 sh[32];
  if (threadIdx.x % 32 == 0) sh[threadIdx.x / 32] = s;
  __syncthreads();
  if (threadIdx.x == 0) { u64 t = 0; for (int k = 0; k < blockDim.x / 32; k++) t += sh[k]; rowsum[i] = t; }
}

__global__ void fill(int32_t* ci, float* cf, uint32_t* bits, int n, int W) {
  for (size_t x = blockIdx.x * (size_t)blockDim.x + threadIdx.x; x < (size_t)n * n; x += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)(x * 2654435761u) ^ (uint32_t)(x >> 7);
    ci[x] = (int32_t)(h % (uint32_t)n);  // values like (A*A)_ij: non-negative integers below n
    cf[x] = (float)ci[x];
  }
  for (size_t w = blockIdx.x * (size_t)blockDim.x + threadIdx.x; w < (size_t)n * W; w += (size_t)gridDim.x * blockDim.x)
    bits[w] = (uint32_t)(w * 0x9E3779B97F4A7C15ull >> 32);
}

template <typename T, int V>
float run(const T* C, const uint32_t* bits, int n, int W, u64* rs, u64* host, int reps) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  masked_rowsum<T, V><<<n, 256>>>(C, bits, n, W, rs);
  cudaEventRecord(a);
  for (int r = 0; r < reps; r++) masked_rowsum<T, V><<<n, 256>>>(C, bits, n, W, rs);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms, a, b);
  cudaMemcpy(host, rs, n * sizeof(u64), cudaMemcpyDeviceToHost);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / reps;
}

int main() {
  const int n = 8192, W = n / 32, reps = 20;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  int32_t* ci; float* cf; uint32_t* bits; u64* rs;
  CK(cudaMalloc(&ci, (size_t)n * n * 4)); CK(cudaMalloc(&cf, (size_t)n * n * 4));
  CK(cudaMalloc(&bits, (size_t)n * W * 4)); CK(cudaMalloc(&rs, n * sizeof(u64)));
  fill<<<1024, 256>>>(ci, cf, bits, n, W); CK(cudaDeviceSynchronize());
  static u64 h[4][n];
  float t[4];
  t[0] = run<int32_t, 0>(ci, bits, n, W, rs, h[0], reps);
  t[1] = run<int32_t, 1>(ci, bits, n, W, rs, h[1], reps);
  t[2] = run<float, 0>(cf, bits, n, W, rs, h[2], reps);
  t[3] = run<float, 2>(cf, bits, n, W, rs, h[3], reps);
  CK(cudaGetLastError());
  bool same = true;
  for (int i = 0; i < n; i++) same &= h[0][i] == h[1][i] && h[0][i] == h[2][i] && h[0][i] == h[3][i];
  const double bytes = (double)n * n * 4 + (double)n * W * 4;  // C once + adjacency bits once
  const char* name[4] = {"INT32 C, llrint((double)c)  [1.3.0]", "INT32 C, integer add       ", "FP32 C,  llrint((double)c)  [1.3.0]", "FP32 C,  __float2int_rn     "};
  printf("%s (sm_%d%d), n = %d, mean of %d launches\n", p.name, p.major, p.minor, n, reps);
  for (int k = 0; k < 4; k++) printf("  %s %7.3f ms  %6.1f GB/s\n", name[k], t[k], bytes / (t[k] * 1e-3) / 1e9);
  printf("all four variants give identical row sums: %s\n", same ? "yes" : "NO");
  return same ? 0 : 1;
}
