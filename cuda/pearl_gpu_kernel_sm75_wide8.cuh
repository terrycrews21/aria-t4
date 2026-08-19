// ariaminer — Turing sm_75 variant H: chunkK=64, stride 16, 2 stages, dynamic smem.
// Halves loop iterations vs wide3. Accepts 4-way LDS bank conflicts (tensor-bound).
// 512 threads, 128x256 CTA, warp owns contiguous 32x64 (2x4 tiles of 16x16).
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

namespace aria_sm75_wide8 {
using namespace cute;

static constexpr int kBlockM = 128;
static constexpr int kBlockN = 256;
static constexpr int kChunkK = 64;
static constexpr int kMmaK = 16;
static constexpr int kSubSteps = kChunkK / kMmaK;  // 4
static constexpr int kWarpRows = 4;
static constexpr int kWarpCols = 4;
static constexpr int kTileRowsPerWarp = 2;
static constexpr int kTileColsPerWarp = 4;
static constexpr int kTilesPerWarp = kTileRowsPerWarp * kTileColsPerWarp;  // 8
static constexpr int kWarps = kWarpRows * kWarpCols;   // 16
static constexpr int kThreads = kWarps * 32;           // 512
static constexpr int kTranscript = 16;
static constexpr int kRotate = 13;
static constexpr int kRowStride = 16;  // no padding, 64 bytes per row for chunkK=64
static constexpr int kWordsA = kBlockM * kRowStride;   // 2048 words = 8KB
static constexpr int kWordsB = kBlockN * kRowStride;   // 4096 words = 16KB
static constexpr int kWordsStage = kWordsA + kWordsB;  // 6144 words = 24KB
static constexpr int kStages = 2;
static constexpr int kTranscriptWords = kWarps * kTilesPerWarp * kTranscript; // 2048 words = 8KB
static constexpr int kDynamicSmemWords = kStages * kWordsStage + kTranscriptWords; // 14336 words = 57344 bytes
static constexpr int kDynamicSmemBytes = kDynamicSmemWords * 4;  // 57344 bytes

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

  extern __shared__ __align__(16) uint32_t smem[];
  uint32_t (*stages)[kWordsStage] = reinterpret_cast<uint32_t (*)[kWordsStage]>(smem);
  // transcripts start after stages
  uint32_t* transcripts_base = smem + kStages * kWordsStage;
  // transcripts layout: [warp][tile][16]
  uint32_t* my_trans = transcripts_base + (warp * kTilesPerWarp) * kTranscript;

  const int row_base = blockIdx.x * kBlockM;
  const int col_base = blockIdx.y * kBlockN;
  const int warp_row = row_base + warp_m * (kTileRowsPerWarp * 16);
  const int warp_col = col_base + warp_n * (kTileColsPerWarp * 16);

  if (lane < kTranscript) {
    #pragma unroll
    for (int tile = 0; tile < kTilesPerWarp; ++tile)
      my_trans[tile * kTranscript + lane] = 0;
  }

  int32_t acc[kTilesPerWarp][8];
  #pragma unroll
  for (int tile = 0; tile < kTilesPerWarp; ++tile)
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[tile][i] = 0;

  // chunkK=64: each row has 64 bytes = 4 uint4 vectors
  // A: 128 rows x 4 vec = 512 loads; B: 256 rows x 4 vec = 1024 loads
  // Total: 1536 loads for 512 threads = 3 loads per thread
  uint4 pf_a, pf_b0, pf_b1;

  auto prefetch = [&](int k_offset) {
    // A: threads 0-511, each loads one uint4 from A (512 total for 128 rows x 4 vecs)
    {
      int row = tid / 4;       // 0..127
      int col_vec = tid % 4;   // 0..3
      pf_a = __ldcg(reinterpret_cast<const uint4*>(
          a + static_cast<size_t>(row_base + row) * k + k_offset + col_vec * 16));
    }
    // B: need 1024 loads, 512 threads -> 2 loads each
    {
      int idx0 = tid;           // 0..511
      int idx1 = tid + kThreads; // 512..1023
      int row0 = idx0 / 4, cv0 = idx0 % 4;
      int row1 = idx1 / 4, cv1 = idx1 % 4;
      pf_b0 = __ldcg(reinterpret_cast<const uint4*>(
          bt + static_cast<size_t>(col_base + row0) * k + k_offset + cv0 * 16));
      pf_b1 = __ldcg(reinterpret_cast<const uint4*>(
          bt + static_cast<size_t>(col_base + row1) * k + k_offset + cv1 * 16));
    }
  };

  auto publish = [&](int stage) {
    uint32_t* s = stages[stage];
    // Store A: row = tid/4, vec = tid%4, stride=16 words
    {
      int row = tid / 4;
      int col_vec = tid % 4;
      reinterpret_cast<uint4*>(s + row * kRowStride)[col_vec] = pf_a;
    }
    // Store B: two stores per thread
    {
      int idx0 = tid;
      int idx1 = tid + kThreads;
      int row0 = idx0 / 4, cv0 = idx0 % 4;
      int row1 = idx1 / 4, cv1 = idx1 % 4;
      reinterpret_cast<uint4*>(s + kWordsA + row0 * kRowStride)[cv0] = pf_b0;
      reinterpret_cast<uint4*>(s + kWordsA + row1 * kRowStride)[cv1] = pf_b1;
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
    const uint32_t* sb = sa + kWordsA;
    const int a_row_lo = warp_m * (kTileRowsPerWarp * 16) + lane_group;
    const int b_col_lo = warp_n * (kTileColsPerWarp * 16) + lane_group;

    #pragma unroll
    for (int sub = 0; sub < kSubSteps; ++sub) {
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
          my_trans[tile * kTranscript + transcript_index] =
              rotl13_xor(my_trans[tile * kTranscript + transcript_index], folded);
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
        for (int i = 0; i < 16; ++i) message(i) = my_trans[tile * kTranscript + i];
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

}  // namespace aria_sm75_wide8
