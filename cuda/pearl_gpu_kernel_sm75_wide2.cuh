// ariaminer — Turing sm_75 warp-owned Pearl grind, variant B (128x128 CTA).
//
// Derived from the wide kernel: same warp-owned contiguous 16x16 proof tiles,
// same padded smem layout, same swizzled rasterization. Differences:
//   - CTA tile 128x128 instead of 128x256, 512 threads (16 warps) instead of 1024
//   - warp grid 4x4, each warp owns 2x2 = four 16x16 tiles (32x32 region)
//   - launch_bounds allows 2 CTAs/SM -> full 32-warp occupancy, smaller barriers
//   - B replication drops 8x -> 4x: smem read traffic per SM-chunk -20% at
//     equal occupancy (ncu showed L1/TEX at 83% = bottleneck on the wide kernel)
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

namespace aria_sm75_wide2 {
using namespace cute;

static constexpr int kBlockM = 128;
static constexpr int kBlockN = 128;
static constexpr int kChunkK = 32;
static constexpr int kMmaK = 16;
static constexpr int kWarpRows = 4;
static constexpr int kWarpCols = 4;
static constexpr int kTileRowsPerWarp = 2;
static constexpr int kTileColsPerWarp = 2;
static constexpr int kTilesPerWarp = kTileRowsPerWarp * kTileColsPerWarp;  // 4
static constexpr int kWarps = kWarpRows * kWarpCols;                       // 16
static constexpr int kThreads = kWarps * 32;                               // 512
static constexpr int kTranscript = 16;
static constexpr int kRotate = 13;
// Padded smem row stride (words): rows land on banks {0,12,24,4,16,28,8,20}
// mod 32 -> zero LDS bank conflicts for the mma fragment reads.
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
__global__ __launch_bounds__(kThreads, 2)
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
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int warp_m = warp / kWarpCols;
  const int warp_n = warp % kWarpCols;
  // L2-friendly rasterization: pair consecutive launches on the same A slice
  // (bijection for even gridDim.y).
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
  // A: 128 rows x 1 uint4 (32B row chunk = 2 uint4? no: kChunkK=32 -> 32 bytes
  // per row = 2 uint4) -> 256 loads across threads 0..255.
  // B: 128 rows x 2 uint4 = 256 loads across threads 256..511.
  const bool has_a = tid < (kBlockM * 2);            // threads 0..255
  const bool has_b = tid >= (kBlockM * 2);           // threads 256..511

  auto prefetch = [&](int k_offset) {
    if (has_a) {
      int idx = tid;                    // 0..255
      int row = idx / 2;                // 0..127
      int col_vec = idx % 2;            // 0 or 1
      const uint4* ptr_a = reinterpret_cast<const uint4*>(
          a + static_cast<size_t>(row_base + row) * k + k_offset + col_vec * 16);
      prefetched_a = __ldcg(ptr_a);
    }
    if (has_b) {
      int idx = tid - kBlockM * 2;      // 0..255
      int row = idx / 2;                // 0..127
      int col_vec = idx % 2;
      const uint4* ptr_bt = reinterpret_cast<const uint4*>(
          bt + static_cast<size_t>(col_base + row) * k + k_offset + col_vec * 16);
      prefetched_b = __ldcg(ptr_bt);
    }
  };

  auto publish = [&](int stage) {
    // Padded layout: row r's two uint4 vectors at word offsets r*kRowStride and
    // r*kRowStride+4 -> uint4 slot r*3 + v (padding slots 2,5,8,... unused).
    if (has_a) {
      int idx = tid;
      reinterpret_cast<uint4*>(stages[stage])[(idx / 2) * 3 + (idx % 2)] = prefetched_a;
    }
    if (has_b) {
      int idx = tid - kBlockM * 2;
      reinterpret_cast<uint4*>(stages[stage] + kWordsA)[(idx / 2) * 3 + (idx % 2)] = prefetched_b;
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
    // This warp's two row-bands within its 32-row span.
    const int a_row0 = warp_m * 32 + lane_group;
    const int a_row1 = a_row0 + 8;
    const int a_row2 = a_row0 + 16;
    const int a_row3 = a_row2 + 8;

    #pragma unroll
    for (int sub = 0; sub < kChunkK / kMmaK; ++sub) {
      const int word_offset = sub * (kMmaK / 4) + lane_quad;
      const uint32_t a0 = sa[a_row0 * kRowStride + word_offset];
      const uint32_t a1 = sa[a_row1 * kRowStride + word_offset];
      const uint32_t a2 = sa[a_row2 * kRowStride + word_offset];
      const uint32_t a3 = sa[a_row3 * kRowStride + word_offset];
      #pragma unroll
      for (int tc = 0; tc < kTileColsPerWarp; ++tc) {
        const int b0 = warp_n * 32 + tc * 16 + lane_group;
        const int b1 = b0 + 8;
        const uint32_t bv0 = sb[b0 * kRowStride + word_offset];
        const uint32_t bv1 = sb[b1 * kRowStride + word_offset];
        // tile t = tr*2 + tc ; tr=0 uses (a0,a1), tr=1 uses (a2,a3)
        mma_8x8x16(acc[tc][0],     acc[tc][1],     a0, bv0);
        mma_8x8x16(acc[tc][2],     acc[tc][3],     a1, bv0);
        mma_8x8x16(acc[tc][4],     acc[tc][5],     a0, bv1);
        mma_8x8x16(acc[tc][6],     acc[tc][7],     a1, bv1);
        mma_8x8x16(acc[2 + tc][0], acc[2 + tc][1], a2, bv0);
        mma_8x8x16(acc[2 + tc][2], acc[2 + tc][3], a3, bv0);
        mma_8x8x16(acc[2 + tc][4], acc[2 + tc][5], a2, bv1);
        mma_8x8x16(acc[2 + tc][6], acc[2 + tc][7], a3, bv1);
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
    for (int tile = 0; tile < kTilesPerWarp; ++tile) {
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
          const int tr = tile / kTileColsPerWarp;
          const int tc = tile % kTileColsPerWarp;
          const int tile_row = warp_row + tr * 16;
          const int tile_col = warp_col + tc * 16;
          #pragma unroll
          for (int i = 0; i < 128; ++i) {
            hit_rows[slot * 128 + i] = tile_row + (i & 15);
            hit_cols[slot * 128 + i] = tile_col + (i & 15);
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

}  // namespace aria_sm75_wide2
