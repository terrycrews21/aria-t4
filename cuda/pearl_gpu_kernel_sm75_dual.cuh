// ariaminer — Turing sm_75 warp-owned four-tile Pearl grind.
//
// Each warp owns four complete contiguous 16x16 proof tiles within a 64x512
// CTA. A is reused across all four tiles. A two-stage 32-K register-prefetched
// pipeline halves synchronization frequency versus 16-K staging while staying
// below T4's 64 KiB shared-memory limit. Accumulators and transcript folds stay
// in registers; there are no shared-memory fold atomics.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

namespace aria_sm75_dual {
using namespace cute;

static constexpr int kBlockM = 128;
static constexpr int kBlockN = 512;
static constexpr int kChunkK = 64;
static constexpr int kMmaK = 16;
static constexpr int kWarpRows = 8;
static constexpr int kWarpCols = 4;
static constexpr int kTilesPerWarp = 8;
static constexpr int kWarps = kWarpRows * kWarpCols;
static constexpr int kThreads = kWarps * 32;
static constexpr int kTranscript = 4;
static constexpr int kRotate = 13;
static constexpr int kWordsA = (kBlockM * kChunkK) / 4;
static constexpr int kWordsB = (kBlockN * kChunkK) / 4;
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
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int warp_m = warp / kWarpCols;
  const int warp_n = warp % kWarpCols;
  const int row_base = blockIdx.x * kBlockM;
  const int col_base = blockIdx.y * kBlockN;
  const int warp_row = row_base + warp_m * 16;
  const int warp_col = col_base + warp_n * 128;

  __shared__ __align__(16) uint32_t stages[kWordsStage];
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

  const int chunks = k / kChunkK;
  const int chunks_per_rank = rank / kChunkK;

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int k_offset = chunk * kChunkK;

    if (tid < 512) {
      int row = tid / 4;
      int col_vec = tid % 4;
      const uint4* ptr_a = reinterpret_cast<const uint4*>(
          a + static_cast<size_t>(row_base + row) * k + k_offset + col_vec * 16);
      reinterpret_cast<uint4*>(stages)[tid] = __ldg(ptr_a);
    }

    #pragma unroll
    for (int q = 0; q < 2; ++q) {
      int vec_idx = tid + q * 1024;
      int cidx = vec_idx / 4;
      int col_vec = vec_idx % 4;
      const uint4* ptr_bt = reinterpret_cast<const uint4*>(
          bt + static_cast<size_t>(col_base + cidx) * k + k_offset + col_vec * 16);
      reinterpret_cast<uint4*>(stages + kWordsA)[vec_idx] = __ldg(ptr_bt);
    }

    __syncthreads();

    const uint32_t* sa = stages;
    const uint32_t* sb = stages + kWordsA;
    const int a_row0 = warp_m * 16 + lane_group;
    const int a_row1 = a_row0 + 8;

    #pragma unroll
    for (int sub = 0; sub < kChunkK / kMmaK; ++sub) {
      const int word_offset = sub * (kMmaK / 4) + lane_quad;
      const uint32_t a0 = sa[a_row0 * (kChunkK / 4) + word_offset];
      const uint32_t a1 = sa[a_row1 * (kChunkK / 4) + word_offset];
      #pragma unroll
      for (int tile = 0; tile < kTilesPerWarp; ++tile) {
        const int b0 = warp_n * 128 + tile * 16 + lane_group;
        const int b1 = b0 + 8;
        const uint32_t bv0 = sb[b0 * (kChunkK / 4) + word_offset];
        const uint32_t bv1 = sb[b1 * (kChunkK / 4) + word_offset];
        mma_8x8x16(acc[tile][0], acc[tile][1], a0, bv0);
        mma_8x8x16(acc[tile][2], acc[tile][3], a1, bv0);
        mma_8x8x16(acc[tile][4], acc[tile][5], a0, bv1);
        mma_8x8x16(acc[tile][6], acc[tile][7], a1, bv1);
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

    __syncthreads();
  }

  if (lane == 0) {
    #pragma unroll
    for (int tile = 0; tile < kTilesPerWarp; ++tile) {
      auto message = make_tensor<uint32_t>(Int<16>{});
      auto cv = make_tensor<uint32_t>(Int<8>{});
      #pragma unroll
      for (int i = 0; i < 16; ++i) {
        message(i) = (i < kTranscript) ? transcripts[warp][tile][i] : 0;
      }
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
          const int tile_col = warp_col + tile * 16;
          #pragma unroll
          for (int i = 0; i < 128; ++i) {
            hit_rows[slot * 128 + i] = warp_row + (i & 15);
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

}  // namespace aria_sm75_dual
