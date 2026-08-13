// ariaminer — Turing sm_75 warp-owned dual-tile Pearl grind.
//
// Motivation: the older 128x128 CuTe kernel maps each thread across 64 proof
// tiles and reconstructs every fold with thousands of shared-memory atomics.
// Real-T4 profiling showed the fold dominated ~79% of setup time. This kernel
// assigns two complete 16x16 proof tiles to each warp, keeps their accumulators
// in registers, and reduces each tile with warp shuffles — no shared atomics.
// A 64x256 CTA also reuses each staged A window across twice as many columns.
//
// The implementation is independent, using NVIDIA's documented
// mma.sync.aligned.m8n8k16 row.col s8 fragment mapping. It preserves the
// consensus recurrence exactly: running int32 accumulators, XOR at each rank
// boundary, then rotl13/XOR into transcript[t % 16], keyed BLAKE3, LE bound.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

namespace aria_sm75_dual {
using namespace cute;

static constexpr int kBlockM = 64;
static constexpr int kBlockN = 256;
static constexpr int kKWindow = 16;
static constexpr int kWarpRows = 4;
static constexpr int kWarpCols = 8;
static constexpr int kWarps = kWarpRows * kWarpCols;
static constexpr int kThreads = kWarps * 32;
static constexpr int kTranscript = 16;
static constexpr int kRotate = 13;
static constexpr int kWordsA = (kBlockM * kKWindow) / 4;
static constexpr int kWordsB = (kBlockN * kKWindow) / 4;
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
  const int warp_col0 = col_base + warp_n * 16;
  const int warp_col1 = warp_col0 + 128;

  __shared__ __align__(16) uint32_t stage_words[2][kWordsStage];
  __shared__ __align__(16) uint32_t transcripts[kWarps][2][kTranscript];

  if (lane < kTranscript) {
    transcripts[warp][0][lane] = 0;
    transcripts[warp][1][lane] = 0;
  }

  // Each of 1024 threads owns one flat word, while threads 0..255 also own a
  // second word. Global loads land in registers one window early; the barrier
  // publishes them only after the previous stage has finished reading.
  uint32_t next0 = 0, next1 = 0;
  bool valid0 = false, valid1 = false;
  auto prefetch = [&](int k_offset) {
    int flat = tid;
    valid0 = flat < kWordsStage;
    if (valid0) {
      if (flat < kWordsA) {
        int byte = flat * 4;
        int r = byte / kKWindow;
        int c = byte % kKWindow;
        next0 = __ldg(reinterpret_cast<const uint32_t*>(
            a + static_cast<size_t>(row_base + r) * k + k_offset + c));
      } else {
        int byte = (flat - kWordsA) * 4;
        int cidx = byte / kKWindow;
        int c = byte % kKWindow;
        next0 = __ldg(reinterpret_cast<const uint32_t*>(
            bt + static_cast<size_t>(col_base + cidx) * k + k_offset + c));
      }
    }
    flat += kThreads;
    valid1 = flat < kWordsStage;
    if (valid1) {
      int byte = (flat - kWordsA) * 4;  // second word is always in B region
      int cidx = byte / kKWindow;
      int c = byte % kKWindow;
      next1 = __ldg(reinterpret_cast<const uint32_t*>(
          bt + static_cast<size_t>(col_base + cidx) * k + k_offset + c));
    }
  };
  auto publish = [&](int stage) {
    int flat = tid;
    if (valid0) stage_words[stage][flat] = next0;
    flat += kThreads;
    if (valid1) stage_words[stage][flat] = next1;
  };

  int32_t acc0[8] = {0,0,0,0,0,0,0,0};
  int32_t acc1[8] = {0,0,0,0,0,0,0,0};
  const int windows = k / kKWindow;
  const int windows_per_rank = rank / kKWindow;

  prefetch(0);
  publish(0);
  __syncthreads();

  for (int window = 0; window < windows; ++window) {
    const int current = window & 1;
    const bool have_next = window + 1 < windows;
    if (have_next) prefetch((window + 1) * kKWindow);

    const uint32_t* sa = stage_words[current];
    const uint32_t* sb = stage_words[current] + kWordsA;
    const int a_row0 = warp_m * 16 + lane_group;
    const int a_row1 = a_row0 + 8;
    const uint32_t a0 = sa[a_row0 * 4 + lane_quad];
    const uint32_t a1 = sa[a_row1 * 4 + lane_quad];

    const int b00 = warp_n * 16 + lane_group;
    const int b01 = b00 + 8;
    const int b10 = b00 + 128;
    const int b11 = b10 + 8;
    const uint32_t b00v = sb[b00 * 4 + lane_quad];
    const uint32_t b01v = sb[b01 * 4 + lane_quad];
    const uint32_t b10v = sb[b10 * 4 + lane_quad];
    const uint32_t b11v = sb[b11 * 4 + lane_quad];

    mma_8x8x16(acc0[0], acc0[1], a0, b00v);
    mma_8x8x16(acc0[2], acc0[3], a1, b00v);
    mma_8x8x16(acc0[4], acc0[5], a0, b01v);
    mma_8x8x16(acc0[6], acc0[7], a1, b01v);
    mma_8x8x16(acc1[0], acc1[1], a0, b10v);
    mma_8x8x16(acc1[2], acc1[3], a1, b10v);
    mma_8x8x16(acc1[4], acc1[5], a0, b11v);
    mma_8x8x16(acc1[6], acc1[7], a1, b11v);

    if (have_next) {
      publish(1 - current);
      __syncthreads();
    }

    if (((window + 1) % windows_per_rank) == 0) {
      uint32_t x0 = 0, x1 = 0;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        x0 ^= static_cast<uint32_t>(acc0[i]);
        x1 ^= static_cast<uint32_t>(acc1[i]);
      }
      #pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        x0 ^= __shfl_xor_sync(0xffffffffu, x0, mask);
        x1 ^= __shfl_xor_sync(0xffffffffu, x1, mask);
      }
      if (lane == 0) {
        int transcript_index = ((window + 1) / windows_per_rank - 1) % kTranscript;
        transcripts[warp][0][transcript_index] =
            rotl13_xor(transcripts[warp][0][transcript_index], x0);
        transcripts[warp][1][transcript_index] =
            rotl13_xor(transcripts[warp][1][transcript_index], x1);
      }
    }
  }

  if (lane == 0) {
    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
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
          uint32_t h = __byte_perm(cv(i), 0, 0x0123);
          uint32_t b = __byte_perm(pow_bound[i], 0, 0x0123);
          if (h > b) { found = false; break; }
          if (h < b) break;
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
          const int tile_col = tile == 0 ? warp_col0 : warp_col1;
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
