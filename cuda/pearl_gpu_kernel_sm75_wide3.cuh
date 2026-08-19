// ariaminer — Turing sm_75 variant C: 512-thread wide-M kernel.
//
// Derived from sm75_wide (128x256 CTA, 1024 threads, warp owns 4 strided
// 16x16 tiles). ncu on wide showed L1/TEX ~83% utilization: the limiter is
// shared-memory load traffic feeding mma fragments, not tensor pipes.
//
// Variant C changes the warp tiling:
//   - 512 threads / 16 warps in a 4x4 warp grid
//   - each warp owns a CONTIGUOUS 32x64 output region = 8 16x16 proof tiles
//     (2 rows x 4 cols), so one A fragment register feeds 8 mmas and one B
//     fragment feeds 4 mmas: 12 LDS.32 per 32 mma (0.375) vs 10/16 (0.625)
//     in wide -> ~40% fewer smem loads for the same tensor work.
//   - launch_bounds(512,1): 128 regs/thread budget, acc[8][8]=64 regs fit
//     without spills.
// Transcript fold, blake3 PoW tail, hit layout and padded smem stride are
// unchanged per 16x16 tile, so share/proof output is bit-identical to wide.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

namespace aria_sm75_wide3 {
using namespace cute;

static constexpr int kBlockM = 128;
static constexpr int kBlockN = 256;
static constexpr int kChunkK = 32;
static constexpr int kMmaK = 16;
static constexpr int kWarpRows = 4;
static constexpr int kWarpCols = 4;
static constexpr int kTileRowsPerWarp = 2;   // 2*16 = 32 rows per warp
static constexpr int kTileColsPerWarp = 4;   // 4*16 = 64 cols per warp
static constexpr int kTilesPerWarp = kTileRowsPerWarp * kTileColsPerWarp;  // 8
static constexpr int kWarps = kWarpRows * kWarpCols;   // 16
static constexpr int kThreads = kWarps * 32;           // 512
static constexpr int kTranscript = 16;
static constexpr int kRotate = 13;
// Padded smem row stride (words), same bank-conflict-free layout as wide.
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
  const int lane_group = lane >> 2;   // 0..7
  const int lane_quad = lane & 3;     // 0..3
  const int warp_m = warp / kWarpCols;
  const int warp_n = warp % kWarpCols;
  // Same L2-friendly paired rasterization as the wide kernel.
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
  // B: 256 rows x 2 uint4 = 512 loads (all threads).
  const bool has_a = tid < (kBlockM * 2);

  auto prefetch = [&](int k_offset) {
    if (has_a) {
      int row = tid / 2;       // 0..127
      int col_vec = tid % 2;   // 0 or 1
      const uint4* ptr_a = reinterpret_cast<const uint4*>(
          a + static_cast<size_t>(row_base + row) * k + k_offset + col_vec * 16);
      prefetched_a = __ldcg(ptr_a);
    }
    {
      int cidx = tid / 2;      // 0..255
      int col_vec = tid % 2;   // 0 or 1
      const uint4* ptr_bt = reinterpret_cast<const uint4*>(
          bt + static_cast<size_t>(col_base + cidx) * k + k_offset + col_vec * 16);
      prefetched_b = __ldcg(ptr_bt);
    }
  };

  auto publish = [&](int stage) {
    if (has_a) {
      reinterpret_cast<uint4*>(stages[stage])[(tid / 2) * 3 + (tid % 2)] = prefetched_a;
    }
    {
      reinterpret_cast<uint4*>(stages[stage] + kWordsA)[(tid / 2) * 3 + (tid % 2)] = prefetched_b;
    }
  };

  const int chunks = k / kChunkK;
  const int chunks_per_rank = rank / kChunkK;
  prefetch(0);
  publish(0);
  __syncthreads();

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int current = chunk & 1;
    const bool have_next = chunk + 1 < chunks;
    if (have_next) prefetch((chunk + 1) * kChunkK);

    const uint32_t* sa = stages[current];
    const uint32_t* sb = stages[current] + kWordsA;
    const int a_row_lo = warp_m * (kTileRowsPerWarp * 16) + lane_group;  // rows 0..7 of warp region
    const int b_col_lo = warp_n * (kTileColsPerWarp * 16) + lane_group;  // cols 0..7 of warp region

    #pragma unroll
    for (int sub = 0; sub < kChunkK / kMmaK; ++sub) {
      const int word_offset = sub * (kMmaK / 4) + lane_quad;
      uint32_t av[kTileRowsPerWarp * 2];
      #pragma unroll
      for (int mi = 0; mi < kTileRowsPerWarp * 2; ++mi)
        av[mi] = sa[(a_row_lo + mi * 8) * kRowStride + word_offset];
      uint32_t bv[kTileColsPerWarp * 2];
      #pragma unroll
      for (int nj = 0; nj < kTileColsPerWarp * 2; ++nj)
        bv[nj] = sb[(b_col_lo + nj * 8) * kRowStride + word_offset];
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

}  // namespace aria_sm75_wide3
