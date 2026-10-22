// bank_conflicts.cu - why k_m1_bitset declares its shared tiles as [8][68] instead of [8][64].
//
// The store pattern is copied from k_m1_bitset (src/trianglebench.cu): thread t writes word w = t % 8 of
// tile row r = t / 8 into tile[w][r]. With a row length (leading dimension) of 64 words, all 8 words that
// one warp writes for the same r land in the same shared-memory bank -> 8-way bank conflict. With 68 the
// rows are shifted by 4 banks and the 32 stores of a warp hit 32 different banks.
//
// Build: nvcc -O3 -arch=native -o bank_conflicts bank_conflicts.cu
// Verify the conflicts with Nsight Compute (needs GPU performance-counter permission):
//   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./bank_conflicts
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

template <int LD>
__global__ void __launch_bounds__(256) tile_stores(uint32_t* out, int iters) {
  __shared__ uint32_t tile[8][LD];
  const int r = threadIdx.x / 8, w = threadIdx.x % 8;  // same index mapping as k_m1_bitset
  uint32_t acc = 0;
  for (int it = 0; it < iters; it++) {
    tile[w][r] = it ^ threadIdx.x;  // the store pattern under test
    __syncthreads();
    acc += tile[w][r ^ 1];          // read a neighbour's word so the stores cannot be optimised away
    __syncthreads();
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template <int LD>
float time_ms(uint32_t* out, int blocks, int iters) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  tile_stores<LD><<<blocks, 256>>>(out, iters);  // warm-up
  cudaEventRecord(a);
  tile_stores<LD><<<blocks, 256>>>(out, iters);
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms = 0; cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms;
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  const int blocks = p.multiProcessorCount * 16, iters = 4096;
  uint32_t* out; CK(cudaMalloc(&out, (size_t)blocks * 256 * sizeof(uint32_t)));
  // bank of tile[w][r] = (w * LD + r) mod 32. For one warp, w = 0..7 and r takes 4 values.
  int worst64 = 0, worst68 = 0;
  for (int bank = 0; bank < 32; bank++) {
    int c64 = 0, c68 = 0;
    for (int lane = 0; lane < 32; lane++) {
      int r = lane / 8, w = lane % 8;
      c64 += ((w * 64 + r) % 32) == bank;
      c68 += ((w * 68 + r) % 32) == bank;
    }
    if (c64 > worst64) worst64 = c64;
    if (c68 > worst68) worst68 = c68;
  }
  float t64 = time_ms<64>(out, blocks, iters), t68 = time_ms<68>(out, blocks, iters);
  CK(cudaGetLastError());
  printf("%s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  printf("leading dimension 64: %8.3f ms   addresses per bank in one warp store: %d (%d-way conflict)\n", t64, worst64, worst64);
  printf("leading dimension 68: %8.3f ms   addresses per bank in one warp store: %d (conflict-free)\n", t68, worst68);
  printf("padding speed-up for this store/load loop: %.2fx\n", t64 / t68);
  CK(cudaFree(out));
  return (worst64 == 8 && worst68 == 1) ? 0 : 1;
}
