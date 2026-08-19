// ariaminer — Turing sm_75 variant J: 128x128 CTA, 256 threads, 2 blocks/SM
// Warp grid 4x2, each warp owns 32x64 (2x4 tiles of 16x16).
// Lower smem allows 2 CTAs/SM = 16 warps total, better latency hiding.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

namespace aria_sm75_wideJ {
using namespace cute;

static constexpr int kBlockM = 128;
static constexpr int kBlockN = 128;
static constexpr int kChunkK = 32;
static constexpr int kMmaK = 16;
static constexpr int kWarpRows = 4;
static constexpr int kWarpCols = 2;
static constexpr int kTileRowsPerWarp = 2;   // 2*16 = 32 rows per warp
static constexpr int kTileColsPerWarp = 4;   // 4*16 = 64 cols per warp
static constexpr int kTilesPerWarp = kTileRowsPerWarp * kTileColsPerWarp;  // 8
static constexpr int kWarps = kWarpRows * kWarpCols;   // 8
static constexpr int kThreads = kWarps * 32;           // 256
static constexpr int kTranscript = 16;
static constexpr int kRotate = 13;
static constexpr int kRowStride = 12;
static constexpr int kWordsA = kBlockM * kRowStride;   // 1536
static constexpr int kWordsB = kBlockN * kRowStride;   // 1536
static constexpr int kWordsStage = kWordsA + kWordsB;  // 3072 words = 12KB

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
__device__ __forceinline__ void mma_8x8x16(
    int32_t& d0, int32_t& d1, uint32_t a, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0,%1}, {%2}, {%3}, {%0,%1};"
      : "+r"(d0), "+r"(d1) : "r"(a), "r"(b));
}
__device__ __forceinline__ uint32_t rotl13_xor(uint32_t p, uint32_t v) {
  return ((p << kRotate) | (p >> (32 - kRotate))) ^ v;
}
#endif

template <bool BigEndian = false>
__global__ __launch_bounds__(kThreads, 2)
void grind(const int8_t* __restrict__ a, const int8_t* __restrict__ bt,
           int m, int n, int k, int rank,
           const uint32_t* __restrict__ pow_key, const uint32_t* __restrict__ pow_bound,
           int* __restrict__ found_count, int* __restrict__ hit_rows,
           int* __restrict__ hit_cols, int max_hits) {
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
  const int warp_row = row_base + warp_m * (kTileRowsPerWarp * 16);
  const int warp_col = col_base + warp_n * (kTileColsPerWarp * 16);

  __shared__ __align__(16) uint32_t stages[2][kWordsStage];
  __shared__ __align__(16) uint32_t transcripts[kWarps][kTilesPerWarp][kTranscript];

  if (lane < kTranscript) {
    #pragma unroll
    for (int t = 0; t < kTilesPerWarp; ++t) transcripts[warp][t][lane] = 0;
  }

  int32_t acc[kTilesPerWarp][8];
  #pragma unroll
  for (int t = 0; t < kTilesPerWarp; ++t)
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[t][i] = 0;

  uint4 pf_a, pf_b;
  // A: 128 rows × 2 uint4 = 256 loads; B: 128 rows × 2 uint4 = 256 loads
  // Total 512 loads for 256 threads = 2 loads per thread
  const bool has_a = tid < (kBlockM * 2);  // all 256 threads load A

  auto prefetch = [&](int k_off) {
    if (has_a) {
      int row = tid / 2, cv = tid % 2;
      pf_a = __ldcg(reinterpret_cast<const uint4*>(a + (size_t)(row_base + row) * k + k_off + cv * 16));
    }
    {
      int ci = tid / 2, cv = tid % 2;
      pf_b = __ldcg(reinterpret_cast<const uint4*>(bt + (size_t)(col_base + ci) * k + k_off + cv * 16));
    }
  };
  auto publish = [&](int s) {
    uint32_t* sp = stages[s];
    if (has_a) reinterpret_cast<uint4*>(sp)[(tid/2)*3 + (tid%2)] = pf_a;
    reinterpret_cast<uint4*>(sp + kWordsA)[(tid/2)*3 + (tid%2)] = pf_b;
  };

  const int chunks = k / kChunkK;
  const int chunks_per_rank = rank / kChunkK;
  prefetch(0); publish(0); __syncthreads();

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int cur = chunk & 1;
    const bool have_next = chunk + 1 < chunks;
    if (have_next) prefetch((chunk + 1) * kChunkK);

    const uint32_t* sa = stages[cur];
    const uint32_t* sb = sa + kWordsA;
    const int arl = warp_m * (kTileRowsPerWarp * 16) + lane_group;
    const int bcl = warp_n * (kTileColsPerWarp * 16) + lane_group;

    #pragma unroll
    for (int sub = 0; sub < kChunkK / kMmaK; ++sub) {
      const int wo = sub * (kMmaK / 4) + lane_quad;
      uint32_t av[kTileRowsPerWarp * 2];
      #pragma unroll
      for (int i = 0; i < kTileRowsPerWarp * 2; ++i) av[i] = sa[(arl + i*8) * kRowStride + wo];
      uint32_t bv[kTileColsPerWarp * 2];
      #pragma unroll
      for (int j = 0; j < kTileColsPerWarp * 2; ++j) bv[j] = sb[(bcl + j*8) * kRowStride + wo];
      #pragma unroll
      for (int tm = 0; tm < kTileRowsPerWarp; ++tm)
        #pragma unroll
        for (int tn = 0; tn < kTileColsPerWarp; ++tn) {
          const int t = tm * kTileColsPerWarp + tn;
          mma_8x8x16(acc[t][0], acc[t][1], av[2*tm],   bv[2*tn]);
          mma_8x8x16(acc[t][2], acc[t][3], av[2*tm+1], bv[2*tn]);
          mma_8x8x16(acc[t][4], acc[t][5], av[2*tm],   bv[2*tn+1]);
          mma_8x8x16(acc[t][6], acc[t][7], av[2*tm+1], bv[2*tn+1]);
        }
    }

    if (((chunk + 1) % chunks_per_rank) == 0) {
      #pragma unroll
      for (int t = 0; t < kTilesPerWarp; ++t) {
        uint32_t folded = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) folded ^= (uint32_t)acc[t][i];
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
          folded ^= __shfl_xor_sync(0xffffffffu, folded, mask);
        if (lane == 0) {
          int ti = ((chunk+1)/chunks_per_rank - 1) % kTranscript;
          transcripts[warp][t][ti] = rotl13_xor(transcripts[warp][t][ti], folded);
        }
      }
    }
    if (have_next) { publish(1 - cur); __syncthreads(); }
  }

  if (lane == 0) {
    #pragma unroll
    for (int tm = 0; tm < kTileRowsPerWarp; ++tm)
      #pragma unroll
      for (int tn = 0; tn < kTileColsPerWarp; ++tn) {
        const int t = tm * kTileColsPerWarp + tn;
        auto message = make_tensor<uint32_t>(Int<16>{});
        auto cv = make_tensor<uint32_t>(Int<8>{});
        #pragma unroll
        for (int i = 0; i < 16; ++i) message(i) = transcripts[warp][t][i];
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv(i) = pow_key[i];
        blake3::compress_msg_block_u32(message, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
        bool found = true;
        #pragma unroll
        for (int i = 7; i >= 0; --i) {
          if (cv(i) > pow_bound[i]) { found = false; break; }
          if (cv(i) < pow_bound[i]) break;
        }
        if (found) {
          int slot = atomicAdd(found_count, 1);
          if (slot < max_hits) {
            int tr = warp_row + tm * 16, tc = warp_col + tn * 16;
            #pragma unroll
            for (int i = 0; i < 128; ++i) {
              hit_rows[slot*128+i] = tr + (i & 15);
              hit_cols[slot*128+i] = tc + (i & 15);
            }
          }
        }
      }
  }
#else
  (void)a;(void)bt;(void)m;(void)n;(void)k;(void)rank;
  (void)pow_key;(void)pow_bound;(void)found_count;
  (void)hit_rows;(void)hit_cols;(void)max_hits;
#endif
}
}  // namespace aria_sm75_wideJ
