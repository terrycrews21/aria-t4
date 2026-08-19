// ariaminer — Turing sm_75 variant L: wideK + 2 CTAs/SM occupancy.
//
// Derived from wide3 (variant C, 512 threads, 128x256 CTA, warp owns
// contiguous 32x64 = 8 16x16 proof tiles). wide3's measured limiter is
// L1/TEX ~83% utilization: 12 LDS.32 transactions per 32-mma substep feed
// the tensor pipes. Variant K replaces them with ldmatrix:
//   - per k-substep (16 cols): one ldmatrix.x2 loads the 2 A fragments
//     one ldmatrix.x4 loads the 4 A fragments (av[0..3]); two ldmatrix.x4
//     load the 8 B fragments (bv[0..7]) -> 3 instructions instead of 12
//     LDS.32 (4x fewer LSU issue slots), same 512B/substep data volume.
//   - mma.m8n8k16 s8 A/B fragment (lane l = row l>>2, bytes 4(l&3)+i) is
//     bit-identical to ldmatrix.m8n8.b16 output (lane l = row l>>2, b16
//     cols 2(l&3), 2(l&3)+1), verified against PTX ISA 9.7.15.5.3 and
//     imma_microtest-style scalar cross-check (see ldmatrix_microtest.cu).
//   - padded smem stride 12 words (=48B) keeps ldmatrix conflict-free:
//     row r starts at bank 12r mod 32 -> 8 distinct 4-bank groups.
// Everything else (raster, transcript fold, blake3 tail, hit layout) is
// unchanged -> share/proof output is bit-identical to wide3.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

namespace aria_sm75_wideL {
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

// ldmatrix collective smem->register fragment loads. addr = smem byte
// address of the 16B row for this lane (lanes beyond 8/16 are unused by the
// hardware for x1/x2; for x4 all 32 lanes supply rows of the 4 matrices:
// lane l -> matrix l>>3, row l&7).
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
      : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
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
           int max_hits,
           uint32_t* __restrict__ transcript_buf) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  (void)m;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  (void)(lane & 31);  // lane_group/lane_quad unused in the ldmatrix variant
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
  // transcripts live in GLOBAL memory (2 CTAs/SM need static smem <= 48KB).
  // Slot pool of kTranscriptSlots CTAs (>= max resident CTAs on the device)
  // so a live CTA's slot is never reused by a concurrent CTA; each CTA
  // zeroes its slot up front (block-wide, then syncthreads) and reads it
  // back at PoW time.
  constexpr int kTrWords = kWarps * kTilesPerWarp * kTranscript;  // 2048
  constexpr int kTranscriptSlots = 256;
  const int cta_linear = blockIdx.y * gridDim.x + blockIdx.x;
  uint32_t* my_tr = transcript_buf + (size_t)(cta_linear & (kTranscriptSlots - 1)) * kTrWords;
  {
    // 2048 words / 512 threads = 4 words each
    const int w0 = tid * 4;
    my_tr[w0 + 0] = 0; my_tr[w0 + 1] = 0; my_tr[w0 + 2] = 0; my_tr[w0 + 3] = 0;
    __syncthreads();
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
    const int a_row_lo = warp_m * (kTileRowsPerWarp * 16);  // first row of warp region
    const int b_col_lo = warp_n * (kTileColsPerWarp * 16);  // first col of warp region
    // ldmatrix address source lane: matrix (lane>>3), row (lane&7).
    const int ldm_mat = lane >> 3;
    const int ldm_row = lane & 7;

    #pragma unroll
    for (int sub = 0; sub < kChunkK / kMmaK; ++sub) {
      const int k_word = sub * (kMmaK / 4);  // first word of this 16-col slice
      uint32_t av[kTileRowsPerWarp * 2];     // 4 8x16 A fragments (32 rows)
      // One x4 covers all 4 A matrices: lane l -> matrix l>>3 row l&7.
      {
        const uint32_t addr_a = __cvta_generic_to_shared(
            sa + (a_row_lo + ldm_mat * 8 + ldm_row) * kRowStride + k_word);
        ldmatrix_x4(av[0], av[1], av[2], av[3], addr_a);
      }
      uint32_t bv[kTileColsPerWarp * 2];     // 8 8x16 B fragments (64 rows)
      // Two x4 for the 8 B matrices: first covers rows 0..31, second 32..63.
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
          // rotl13 needs the previous value: read-modify-write (L2-cached, one
          // writer per slot per boundary -> no atomic contention).
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

  __threadfence_block();
  if (lane == 0) {
    #pragma unroll
    for (int tm = 0; tm < kTileRowsPerWarp; ++tm) {
      #pragma unroll
      for (int tn = 0; tn < kTileColsPerWarp; ++tn) {
        const int tile = tm * kTileColsPerWarp + tn;
        auto message = make_tensor<uint32_t>(Int<16>{});
        auto cv = make_tensor<uint32_t>(Int<8>{});
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

}  // namespace aria_sm75_wideL
