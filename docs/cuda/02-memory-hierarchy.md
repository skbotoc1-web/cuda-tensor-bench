# 2. Memory hierarchy: coalescing, shared-memory banks, L2

## Theory

From slow to fast: device memory (DRAM), the L2 cache shared by all SMs, and on each SM the unified
L1/shared-memory array and the register file (Guide §1.2.3.3). Three rules from the guide matter for
the kernels in this repository:

1. **Coalescing.** *"Global memory is accessed via 32-byte memory transactions."* If the 32 threads of
   a warp read 32 consecutive 4-byte words, the warp needs 128 bytes, fetched as four 32-byte
   transactions, and every byte that is fetched is used (Guide §2.3.4.1).
2. **Shared-memory banks.** *"Shared memory has 32 banks that are organized such that successive 32-bit
   words map to successive banks."* If threads of one warp access different words in the same bank,
   the accesses are serialized. Reading the same word is a broadcast and costs nothing extra
   (Guide §2.3.4.2, §2.3.4.2.2).
3. **L2 is the shared working set.** Every SM reads through L2. A kernel that re-reads data many times
   only stays off DRAM if the re-used data fits into L2 (4 MiB on the A4000, 24 MiB on the GB10).

## Coalescing in practice: the epilogue reads exactly what it needs

`k_masked_rowsum` runs one block per row `i`. Thread `j` reads `C[i*n + j]` for `j = threadIdx.x,
threadIdx.x + 256, ...`, so neighbouring threads read neighbouring words: coalesced by construction.

Nsight Compute on the A4000 at n = 8192:

| Metric | Measured | Expected (derived) |
|---|---|---|
| global load requests | 4,194,304 | 8,192 blocks x 8 warps x 32 iterations x 2 loads (C and the adjacency word) = 4,194,304 |
| global load sectors | 10,452,958 | per iteration 4 sectors for C + 1 for the adjacency word: 2,097,152 x 5 = 10,485,760 |
| DRAM bytes read | 275.9 MB | C (8192² x 4 B = 268.4 MB) + bitset (8192 x 256 x 4 B = 8.4 MB) = 276.8 MB |
| L2 hit rate | 0.14 % | every byte is used once, so there is nothing to re-use |

The kernel reads each byte from DRAM exactly once. There is nothing left to gain on the memory side.
Where its time goes instead is the subject of [chapter 3](03-roofline-and-bottlenecks.md).

## Shared-memory banks: why `As[8][68]` and not `As[8][64]`

`k_m1_bitset` stages 8 words of 64 adjacency rows in shared memory. Thread `t` loads word `w = t % 8`
of row `r = t / 8` and stores it transposed, so that the compute loop can read a column of words:

```cuda
__shared__ uint32_t As[8][68], Bs[8][68];  // +4 padding avoids shared-memory bank conflicts on the stores
...
      int r = t / 8, w = t % 8;
      As[w][r] = bits[(size_t)(bi + r) * W + k0 + w];
```

The bank of `As[w][r]` is `(w * LD + r) mod 32`, with `LD` the row length. One warp covers `w = 0..7`
and four values of `r`:

- `LD = 64`: `w * 64` is a multiple of 32, so the bank is just `r`. 8 threads with the same `r` hit the
  same bank with 8 different addresses: an **8-way conflict**.
- `LD = 68`: the bank is `(4w + r) mod 32`, and for `w = 0..7`, `r = 0..3` those are 32 different banks:
  **no conflict**. Four extra words per row cost 8 x 4 x 4 B = 128 B per array.

The reads in the compute loop, `As[w][ty + 16*r]` and `Bs[w][tx + 16*c]`, are conflict-free either way:
a warp reads at most 2 distinct words of `As` (broadcast) and 16 consecutive words of `Bs`.

**Measured** with [`examples/bank_conflicts.cu`](examples/bank_conflicts.cu), which copies exactly this
store pattern into a loop:

| | A4000 | GB10 |
|---|---|---|
| bank conflicts, LD = 64 (Nsight, stores + loads) | 176,160,768 + 176,160,768 | not measured (no counter access) |
| bank conflicts, LD = 68 | 0 + 0 | not measured |
| time LD = 64 / LD = 68 | 4.943 ms / 1.359 ms = **3.64x** | 3.580 ms / 0.979 ms = **3.66x** |

Nsight Compute also reports **0** shared-memory bank conflicts for `k_m1_bitset` itself (stores and
loads, A4000). In the real kernel the padding gains less than 3.6x, because the stores are a small part
of the work: per k-step each thread issues 4 shared-memory stores and 64 shared-memory loads in the
source, against 128 POPC.

## L2 decides whether fusion pays off

`peak/fusedtri.cu` fuses the whole trianglebench pipeline into one CUDA Tile kernel: GEMM `A*A`, mask with
`A`, row sum, so the n x n result `C` is never written to memory. The unfused path writes `C` (INT32) and
reads it back in `k_masked_rowsum`. Measured result (INT8, edge density 0.05, from
[`peak/README.md`](../../peak/README.md)):

| GPU | n | INT8 matrix A | fits in L2? | fused / unfused |
|---|---|---|---|---|
| GB10 (24 MiB L2) | 4096 | 16 MiB | yes | **1.39x** |
| GB10 | 8192 | 64 MiB | no | 0.95x |
| GB10 | 16384 | 256 MiB | no | 0.55x |
| A4000 (4 MiB L2) | 4096 | 16 MiB | no | 0.52x |
| A4000 | 8192 | 64 MiB | no | 0.47x |

**Derived** - the fused kernel wins only in the one case where A fits completely into L2. Once it does
not fit, the tile loads have to come from DRAM, and cuBLASLt's GEMM, with its tuned pipelining and
rasterisation, is faster than our simpler tile kernel by more than the cost of writing and re-reading C.
Saving the C round trip (2 x 64 MiB at n = 4096 with INT32 C) is not enough on its own. The peak README
lists this as the likely cause, not as a profiled fact; the L2 pattern in the table is consistent with it.

The same effect is visible in M1 on the A4000: its 8 MiB bitset is larger than the 4 MiB L2. Nsight
Compute measures 1.06 GB of DRAM reads for one M1 solve at n = 8192 (L2 hit rate 51 %), about 127 times
the size of the bitset. M1 still is not memory-bound (DRAM 16 % busy, [chapter 3](03-roofline-and-bottlenecks.md)),
because every word that is loaded feeds 64 POPC.

Two techniques made the CUDA Tile prototype usable at all. Loading the B tile as rows and transposing it
(`ct::transpose`) instead of reading columns: column access compiled to byte loads, and the change
raised the A4000 result from 5 to 61 eqTOPS. And a grouped 1-D block raster (groups of 8 tiles, like
`GROUP_M` in Triton), so that neighbouring blocks share row and column panels in L2: on the GB10 at
n = 8192 this raised the fused kernel from 54 to 158 eqTOPS (development log of the prototype).

## Try it

```bash
cd docs/cuda/examples && make bank_conflicts && ./bank_conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum ./bank_conflicts
ncu --kernel-name k_m1_bitset --launch-count 1 \
    --metrics dram__bytes_read.sum,lts__t_sector_hit_rate.pct ./trianglebench --n 8192 --repeats 1 --seconds 0.2 --warmup 0
```

## References

- CUDA Programming Guide 13.4, §1.2.3.3 On-Chip Memory in GPUs, §2.3.4.1 Coalesced Global Memory Access,
  §2.3.4.2 Shared Memory Access Patterns, §2.3.4.2.2 Shared Memory Bank Conflicts, §2.4 CUDA Tile.
  <https://docs.nvidia.com/cuda/cuda-programming-guide/>
- Nsight Compute Profiling Guide, Memory Workload Analysis. <https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
