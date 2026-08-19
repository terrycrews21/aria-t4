// ariaminer — Turing sm_75 variant M: 1024 threads, 32 warps, 4 tiles/warp.
//
// ncu on wideK (589ms): L1/TEX 85%, compute 37.7%, scheduler issues 0.33
// warps/cycle (67% no-eligible). Root cause: 512 thr = 16 warps = 4/scheduler,
// insufficient to hide smem+mma pipeline latency. Occupancy 50% capped by
// 114 regs/thread (acc[8][8]=64 regs for 8 tiles).
//
// Variant M: 1024 threads, 4x8 warp grid, each warp owns 2x2 = 4 tiles
// (32x32 region). acc[4][8] = 32 regs -> ~64 total -> fits 65K reg file.
// 32 warps / 4 schedulers = 8 warps/scheduler -> 2x latency hiding.
// Per substep: 1 ldmatrix.x4 (A) + 1 ldmatrix.x4 (B) = 2 instructions
// feeding 16 mma -> 0.125 LSU/mma ratio (vs 0.25 wideK, 0.375 wide3).
// Transcript fold, blake3 PoW, hit layout unchanged -> bit-identical proofs.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

namespace aria_sm75_wideM {
using namespace cute;

static constexpr int kBlockM = 128;
static constexpr int kBlockN = 256;
static constexpr int kChunkK = 32;
static constexpr int kMmaK = 16;
static constexpr int kWarpRows = 4;
static constexpr int kWarpCols = 8;
static constexpr int kTileRowsPerWarp = 2;   // 2*16 = 32 rows per warp
static constexpr int kTileColsPerWarp = 2;   // 2*16 = 32 cols per warp
static constexpr int kTilesPerWarp = kTileRowsPerWarp * kTileColsPerWarp;  // 4
static constexpr int kWarps = kWarpRows * kWarpCols;   // 32
static constexpr int kThreads = kWarps * 32;           // 1024
static constexpr int kTranscript = 16;
static constexpr int kRotate = 13;
// Padded smem row stride (words). 12 words = 48B: ldmatrix rows start at
// banks 12r mod 32 -> 8 distinct 4-bank groups (conflict-free).
static constexpr int kRowStride = 12;
static constexpr int kWordsA = kBlockM * kRowStride;
static constexpr int kWordsB = kBlockN * kRowStride;
static constexpr int kWordsStage = kWordsA + kWordsB;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
__device__ __forceinline__ void mma_8x8x16(
    int32_t& d0, int32_t& d1, uint32_t a, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0,%1}, {%2}, {%3}, {%0,%1};"
      : "+r"(d0), "+r"(d1) : "r"(a), "r"(b));
}

__device__ __forceinline__ uint32_t rotl13_xor(uint32_t previous, uint32_t value) {
  return ((previous << kRotate) | (previous >> (32 - kRotate))) ^ value;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
#endif

template <bool BigEndian = false>
__global__ __launch_bounds__(kThreads, 1)
void grind(const int8_t* __restrict__ a,
           const int8_t* __restrict__ bt,
           int m, int n, int k, int rank,
           const uint32_t* __restrict__ pow_key,
           const uint32_t* __restrict__ pow_bound,
           int* __restrict__ found_count,
           int* __restrict__ hit_rows,
           int* __restrict__ hit_cols,
           int max_hits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  (void)m;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp / kWarpCols;   // 0..3
  const int warp_n = warp % kWarpCols;   // 0..7
  // Same L2-friendly paired rasterization as wide3.
  int tile_x, tile_y;
  if ((gridDim.y & 1) == 0) {
    const int linear = blockIdx.y * gridDim.x + blockIdx.x;
    tile_x = (linear >> 1) % gridDim.x;
    tile_y = (linear / (gridDim.x << 1)) * 2 + (linear & 1);
  } else {
    tile_x = blockIdx.x;
    tile_y = blockIdx.y;
  }
  const int row_base = tile_x * kBlockM;
  const int col_base = tile_y * kBlockN;
  const int warp_row = row_base + warp_m * (kTileRowsPerWarp * 16);
  const int warp_col = col_base + warp_n * (kTileColsPerWarp * 16);

  __shared__ __align__(16) uint32_t stages[2][kWordsStage];
  __shared__ __align__(16) uint32_t transcripts[kWarps][kTilesPerWarp][kTranscript];

  if (lane < kTranscript) {
    #pragma unroll
    for (int tile = 0; tile < kTilesPerWarp; ++tile)
      transcripts[warp][tile][lane] = 0;
  }

  int32_t acc[kTilesPerWarp][8];
  #pragma unroll
  for (int tile = 0; tile < kTilesPerWarp; ++tile)
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[tile][i] = 0;

  uint4 prefetched_a;
  uint4 prefetched_b;
  // A: 128 rows x 2 uint4 = 256 loads (threads 0..255);
  // B: 256 rows x 2 uint4 = 512 loads (threads 0..511).
  const bool has_a = tid < (kBlockM * 2);

  auto prefetch = [&](int k_offset) {
    if (has_a) {
      int row = tid / 2;
      int col_vec = tid % 2;
      const uint4* ptr_a = reinterpret_cast<const uint4*>(
          a + static_cast<size_t>(row_base + row) * k + k_offset + col_vec * 16);
      prefetched_a = __ldcg(ptr_a);
    }
    if (tid < kBlockN * 2) {
      int cidx = tid / 2;
      int col_vec = tid % 2;
      const uint4* ptr_bt = reinterpret_cast<const uint4*>(
          bt + static_cast<size_t>(col_base + cidx) * k + k_offset + col_vec * 16);
      prefetched_b = __ldcg(ptr_bt);
    }
  };

  auto publish = [&](int stage) {
    if (has_a) {
      reinterpret_cast<uint4*>(stages[stage])[(tid / 2) * 3 + (tid % 2)] = prefetched_a;
    }
    if (tid < kBlockN * 2) {
      reinterpret_cast<uint4*>(stages[stage] + kWordsA)[(tid / 2) * 3 + (tid % 2)] = prefetched_b;
    }
  };

  const int chunks = k / kChunkK;
  const int chunks_per_rank = rank / kChunkK;
  prefetch(0);
  publish(0);
  __syncthreads();

  // ldmatrix address source lane: matrix (lane>>3), row (lane&7).
  const int ldm_mat = lane >> 3;
  const int ldm_row = lane & 7;

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int current = chunk & 1;
    const bool have_next = chunk + 1 < chunks;
    if (have_next) prefetch((chunk + 1) * kChunkK);

    const uint32_t* sa = stages[current];
    const uint32_t* sb = stages[current] + kWordsA;
    const int a_row_lo = warp_m * (kTileRowsPerWarp * 16);
    const int b_col_lo = warp_n * (kTileColsPerWarp * 16);

    #pragma unroll
    for (int sub = 0; sub < kChunkK / kMmaK; ++sub) {
      const int k_word = sub * (kMmaK / 4);
      uint32_t av[kTileRowsPerWarp * 2];  // 4 A fragments
      {
        const uint32_t addr_a = __cvta_generic_to_shared(
            sa + (a_row_lo + ldm_mat * 8 + ldm_row) * kRowStride + k_word);
        ldmatrix_x4(av[0], av[1], av[2], av[3], addr_a);
      }
      uint32_t bv[kTileColsPerWarp * 2];  // 4 B fragments
      {
        const uint32_t addr_b = __cvta_generic_to_shared(
            sb + (b_col_lo + ldm_mat * 8 + ldm_row) * kRowStride + k_word);
        ldmatrix_x4(bv[0], bv[1], bv[2], bv[3], addr_b);
      }
      #pragma unroll
      for (int tm = 0; tm < kTileRowsPerWarp; ++tm) {
        #pragma unroll
        for (int tn = 0; tn < kTileColsPerWarp; ++tn) {
          const int t = tm * kTileColsPerWarp + tn;
          mma_8x8x16(acc[t][0], acc[t][1], av[2 * tm],     bv[2 * tn]);
          mma_8x8x16(acc[t][2], acc[t][3], av[2 * tm + 1], bv[2 * tn]);
          mma_8x8x16(acc[t][4], acc[t][5], av[2 * tm],     bv[2 * tn + 1]);
          mma_8x8x16(acc[t][6], acc[t][7], av[2 * tm + 1], bv[2 * tn + 1]);
        }
      }
    }

    if (((chunk + 1) % chunks_per_rank) == 0) {
      #pragma unroll
      for (int tile = 0; tile < kTilesPerWarp; ++tile) {
        uint32_t folded = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) folded ^= static_cast<uint32_t>(acc[tile][i]);
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
          folded ^= __shfl_xor_sync(0xffffffffu, folded, mask);
        if (lane == 0) {
          int transcript_index = ((chunk + 1) / chunks_per_rank - 1) % kTranscript;
          transcripts[warp][tile][transcript_index] =
              rotl13_xor(transcripts[warp][tile][transcript_index], folded);
        }
      }
    }

    if (have_next) {
      publish(1 - current);
      __syncthreads();
    }
  }

  if (lane == 0) {
    #pragma unroll
    for (int tm = 0; tm < kTileRowsPerWarp; ++tm) {
      #pragma unroll
      for (int tn = 0; tn < kTileColsPerWarp; ++tn) {
        const int tile = tm * kTileColsPerWarp + tn;
        auto message = make_tensor<uint32_t>(Int<16>{});
        auto cv = make_tensor<uint32_t>(Int<8>{});
        #pragma unroll
        for (int i = 0; i < 16; ++i) message(i) = transcripts[warp][tile][i];
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv(i) = pow_key[i];
        blake3::compress_msg_block_u32(message, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
        bool found = true;
        if constexpr (BigEndian) {
          #pragma unroll
          for (int i = 0; i < 8; ++i) {
            uint32_t hash_word = __byte_perm(cv(i), 0, 0x0123);
            uint32_t bound_word = __byte_perm(pow_bound[i], 0, 0x0123);
            if (hash_word > bound_word) { found = false; break; }
            if (hash_word < bound_word) break;
          }
        } else {
          #pragma unroll
          for (int i = 7; i >= 0; --i) {
            if (cv(i) > pow_bound[i]) { found = false; break; }
            if (cv(i) < pow_bound[i]) break;
          }
        }
        if (found) {
          int slot = atomicAdd(found_count, 1);
          if (slot < max_hits) {
            const int tile_row = warp_row + tm * 16;
            const int tile_col = warp_col + tn * 16;
            #pragma unroll
            for (int i = 0; i < 128; ++i) {
              hit_rows[slot * 128 + i] = tile_row + (i & 15);
              hit_cols[slot * 128 + i] = tile_col + (i & 15);
            }
          }
        }
      }
    }
  }
#else
  (void)a; (void)bt; (void)m; (void)n; (void)k; (void)rank;
  (void)pow_key; (void)pow_bound; (void)found_count;
  (void)hit_rows; (void)hit_cols; (void)max_hits;
#endif
}

}  // namespace aria_sm75_wideM
