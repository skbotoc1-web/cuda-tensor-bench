// popc_rate.cu - measures how many 32-bit population counts (__popc, SASS POPC) one SM retires per clock.
//
// Why it matters: the CUDA-core method M1 of trianglebench replaces 32 multiply-adds by one AND + one POPC.
// Its speed is therefore set by the POPC rate, not by the INT32 add rate. Nsight Compute on an RTX A4000
// shows the XU pipe (which executes POPC on that GPU) 96 % busy in k_m1_bitset.
//
// Method: one block of 1024 threads per SM (at most 1536 threads fit on sm_86 / sm_121, so two blocks can
// never share an SM). Each thread runs 8 independent POPC chains; clock64() measures the cycles of the block.
// The second kernel replaces POPC by XOR as a reference (SASS: one LOP3 + one IADD3 per step; in the POPC
// kernel one POPC + one IADD3 per step - check with: cuobjdump -sass popc_rate).
//
// Build: nvcc -O3 -arch=native -o popc_rate popc_rate.cu
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
typedef unsigned long long u64;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

template <bool POPC>
__global__ void __launch_bounds__(1024) chains(uint32_t seed, int iters, uint32_t* sink, u64* cycles) {
  uint32_t x[8];
#pragma unroll
  for (int k = 0; k < 8; k++) x[k] = seed * (threadIdx.x + 1) + k;
  __syncthreads();
  const u64 t0 = clock64();
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int k = 0; k < 8; k++) x[k] = POPC ? (__popc(x[k]) + x[k]) : ((x[k] ^ 0x5bd1e995u) + x[k]);
  }
  __syncthreads();
  const u64 t1 = clock64();
  uint32_t s = 0;
#pragma unroll
  for (int k = 0; k < 8; k++) s += x[k];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
  if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
}

template <bool POPC>
double ops_per_clock_per_sm(int sms, int iters, uint32_t* sink, u64* d_cyc, u64* h_cyc) {
  chains<POPC><<<sms, 1024>>>(12345u, iters, sink, d_cyc);  // warm-up
  chains<POPC><<<sms, 1024>>>(12345u, iters, sink, d_cyc);
  cudaMemcpy(h_cyc, d_cyc, sms * sizeof(u64), cudaMemcpyDeviceToHost);
  double mean = 0;
  for (int b = 0; b < sms; b++) mean += (double)h_cyc[b] / sms;
  return 1024.0 * 8.0 * iters / mean;  // counted ops: one POPC (or one XOR) per chain step
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  const int sms = p.multiProcessorCount, iters = 1 << 14;
  uint32_t* sink; u64* d_cyc; static u64 h_cyc[1024];
  CK(cudaMalloc(&sink, (size_t)sms * 1024 * 4)); CK(cudaMalloc(&d_cyc, sms * sizeof(u64)));
  double popc = ops_per_clock_per_sm<true>(sms, iters, sink, d_cyc, h_cyc);
  double lop = ops_per_clock_per_sm<false>(sms, iters, sink, d_cyc, h_cyc);
  CK(cudaGetLastError());
  printf("%s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, sms);
  printf("POPC: %6.1f per clock per SM\n", popc);
  printf("XOR : %6.1f per clock per SM (reference: each step is LOP3 + IADD3, i.e. %.0f integer instructions per clock)\n", lop, 2 * lop);
  printf("M1 ceiling from POPC: %.0f equivalent ops per clock per SM (each POPC covers 32 multiply-adds = 64 ops)\n", popc * 64);
  return 0;
}
