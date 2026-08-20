// ariaminer — Turing sm_75 variant P: wideK + chunkK=64 (half the pipeline
// iterations, half the __syncthreads), stride 20, dynamic smem.
//
// wideK (chunkK=32, 2 stages, ldmatrix) = 616ms cold / 798ms hot. ncu:
// compute 37.7%, 67% scheduler no-eligible cycles. chunkK=64 halves the
// barrier count (128 vs 256) and halves prefetch/publish phases while
// doubling work per stage. ldmatrix fragment loads preserved (stride 20
// words = 80B: row r starts bank 20r mod 32 -> 8 distinct 4-bank groups).
// Stage size (128+256)*20*4B = 30KB x 2 = 60KB -> dynamic smem opt-in.
// Transcript fold / blake3 PoW / hit layout unchanged -> bit-identical.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

namespace aria_sm75_wideP {
using namespace cute;

static constexpr int kBlockM = 128;
static constexpr int kBlockN = 256;
static constexpr int kChunkK = 64;
static constexpr int kMmaK = 16;
static constexpr int kSubSteps = kChunkK / kMmaK;   // 4
static constexpr int kWarpRows = 4;
static constexpr int kWarpCols = 4;
static constexpr int kTileRowsPerWarp = 2;
static constexpr int kTileColsPerWarp = 4;
static constexpr int kTilesPerWarp = kTileRowsPerWarp * kTileColsPerWarp;  // 8
static constexpr int kWarps = kWarpRows * kWarpCols;   // 16
static constexpr int kThreads = kWarps * 32;           // 512
static constexpr int kTranscript = 16;
static constexpr int kRotate = 13;
static constexpr int kRowStride = 20;                  // words, conflict-free
static constexpr int kWordsA = kBlockM * kRowStride;   // 2560
static constexpr int kWordsB = kBlockN * kRowStride;   // 5120
static constexpr int kWordsStage = kWordsA + kWordsB;  // 7680 words = 30KB
static constexpr int kStages = 2;
static constexpr int kDynamicSmemBytes = kStages * kWordsStage * 4;  // 61440

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
           int max_hits,
           uint32_t* __restrict__ transcript_buf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  (void)m;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int warp_m = warp / kWarpCols;
  const int warp_n = warp % kWarpCols;
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

  extern __shared__ __align__(16) uint32_t smem[];
  uint32_t (*stages)[kWordsStage] = reinterpret_cast<uint32_t (*)[kWordsStage]>(smem);
  // Transcripts in GLOBAL memory (2 CTA smem budget: 60KB dynamic + 8KB
  // static > 64KB sm_75 limit). Slot pool of 256 CTAs >= any resident count
  // on T4 (40 SM); each CTA zeroes its slot then uses it.
  constexpr int kTrWords = kWarps * kTilesPerWarp * kTranscript;  // 2048
  constexpr int kTranscriptSlots = 256;
  const int cta_linear = blockIdx.y * gridDim.x + blockIdx.x;
  uint32_t* my_tr = transcript_buf + (size_t)(cta_linear & (kTranscriptSlots - 1)) * kTrWords;
  if (tid < kTrWords / 4) {
    my_tr[tid * 4 + 0] = 0; my_tr[tid * 4 + 1] = 0;
    my_tr[tid * 4 + 2] = 0; my_tr[tid * 4 + 3] = 0;
  }
  __syncthreads();

  int32_t acc[kTilesPerWarp][8];
  #pragma unroll
  for (int tile = 0; tile < kTilesPerWarp; ++tile)
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[tile][i] = 0;

  // chunkK=64: each row holds 64B = 4 uint4. A: 128*4=512 loads (all threads),
  // B: 256*4=1024 loads = 2 per thread.
  uint4 pf_a;
  uint4 pf_b0, pf_b1;

  auto prefetch = [&](int k_offset) {
    {
      int row = tid / 4;        // 0..127
      int col_vec = tid % 4;    // 0..3
      pf_a = __ldcg(reinterpret_cast<const uint4*>(
          a + static_cast<size_t>(row_base + row) * k + k_offset + col_vec * 16));
    }
    {
      int idx0 = tid, idx1 = tid + kThreads;
      int row0 = idx0 / 4, cv0 = idx0 % 4;
      int row1 = idx1 / 4, cv1 = idx1 % 4;
      pf_b0 = __ldcg(reinterpret_cast<const uint4*>(
          bt + static_cast<size_t>(col_base + row0) * k + k_offset + cv0 * 16));
      pf_b1 = __ldcg(reinterpret_cast<const uint4*>(
          bt + static_cast<size_t>(col_base + row1) * k + k_offset + cv1 * 16));
    }
  };

  auto publish = [&](int stage) {
    {
      int row = tid / 4;
      int col_vec = tid % 4;
      reinterpret_cast<uint4*>(stages[stage] + row * kRowStride)[col_vec] = pf_a;
    }
    {
      int idx0 = tid, idx1 = tid + kThreads;
      int row0 = idx0 / 4, cv0 = idx0 % 4;
      int row1 = idx1 / 4, cv1 = idx1 % 4;
      reinterpret_cast<uint4*>(stages[stage] + kWordsA + row0 * kRowStride)[cv0] = pf_b0;
      reinterpret_cast<uint4*>(stages[stage] + kWordsA + row1 * kRowStride)[cv1] = pf_b1;
    }
  };

  const int chunks = k / kChunkK;
  const int chunks_per_rank = rank / kChunkK;   // 128/64 = 2
  prefetch(0);
  publish(0);
  __syncthreads();

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
    for (int sub = 0; sub < kSubSteps; ++sub) {
      const int k_word = sub * (kMmaK / 4);
      uint32_t av[kTileRowsPerWarp * 2];
      {
        const uint32_t addr_a = __cvta_generic_to_shared(
            sa + (a_row_lo + ldm_mat * 8 + ldm_row) * kRowStride + k_word);
        ldmatrix_x4(av[0], av[1], av[2], av[3], addr_a);
      }
      uint32_t bv[kTileColsPerWarp * 2];
      {
        const uint32_t addr_b0 = __cvta_generic_to_shared(
            sb + (b_col_lo + ldm_mat * 8 + ldm_row) * kRowStride + k_word);
        ldmatrix_x4(bv[0], bv[1], bv[2], bv[3], addr_b0);
        const uint32_t addr_b1 = __cvta_generic_to_shared(
            sb + (b_col_lo + 32 + ldm_mat * 8 + ldm_row) * kRowStride + k_word);
        ldmatrix_x4(bv[4], bv[5], bv[6], bv[7], addr_b1);
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
          uint32_t* slot = my_tr + (warp * kTilesPerWarp + tile) * kTranscript + transcript_index;
          uint32_t prev = *slot;
          *slot = rotl13_xor(prev, folded);
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
        __threadfence_block();
        #pragma unroll
        for (int i = 0; i < 16; ++i) message(i) = my_tr[(warp * kTilesPerWarp + tile) * kTranscript + i];
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
  (void)hit_rows; (void)hit_cols; (void)max_hits; (void)transcript_buf;
#endif
}

}  // namespace aria_sm75_wideP
