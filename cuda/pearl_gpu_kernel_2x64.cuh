// ariaminer v1.1 — kernel GPU 2×64 (AlphaPool). GEMM IMMA identique à v1.0, MAIS
// le fold jackpot regroupe les cellules selon le PATTERN DICTÉ PAR ALPHAPOOL :
//   rows_pattern=[0,32] (h=2)  ·  cols_pattern=[0..63] (w=64)  → tuile 2×64 = 128 cellules.
// Le tensor-core distribue les colonnes entre threads (fragment éparpillé), donc on ne
// peut PAS obtenir un thread = 2×64 par permutation : on REGROUPE explicitement via
// un accumulateur partagé `tileacc[128]` (atomicXor), tous les R=128 k.
//   tuile(row,col) = rowtile*2 + col/64,  rowtile = (row/64)*32 + (row%64 < 32 ? row%64 : row%64-32)
//   thread T possède la tuile T : rowtile=T/2, colgroup=T%2 ; 2 lignes {b*64+p, +32} × 64 cols.
#pragma once
#include <cute/tensor.hpp>
#include "pearl_fold.cuh"
#include "blake3/blake3.cuh"
using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage2x64 {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  uint32_t tileacc[128];   // 1 accumulateur XOR par tuile 2×64 du CTA (512 o)
};

// row local (0..127) → index de row-tile (0..63) : paires {p, p+32} par bloc de 64.
__device__ __forceinline__ int rowtile_of(int row) {
  int blk = row >> 6;           // row / 64
  int w   = row & 63;           // row % 64
  int p   = (w < 32) ? w : w - 32;
  return blk * 32 + p;          // 0..63
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_2x64(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    int reduce_every_k,
    const uint32_t* pow_key, const uint32_t* pow_bound, int* found_count,
    int* hit_rows, int* hit_cols, int max_hits) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  extern __shared__ char smem_[];
  using SS = SharedStorage2x64<TA,TB,ASmemLayout,BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(smem_);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);

  auto K_PIPE_MAX = size<3>(tAsA);
  int k_tile_count = size<3>(tAgA);
  int k_tile_next = 0;
  CUTE_UNROLL
  for (int kp = 0; kp < K_PIPE_MAX-1; ++kp) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,kp));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,kp));
    cp_async_fence();
    --k_tile_count; if (k_tile_count > 0) ++k_tile_next;
  }

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // Coords locales (0..127) de chaque cellule du fragment → index de tuile 2×64 (précalc).
  auto cD = make_identity_tensor(make_shape(get<0>(cta_tiler), get<1>(cta_tiler)));
  auto tCcD = thr_mma.partition_C(cD);
  constexpr int NCELL = decltype(size(tCcD))::value;   // = 128
  int cell_tile[NCELL];
  CUTE_UNROLL
  for (int i = 0; i < NCELL; ++i) {
    int r = get<0>(tCcD(i));
    int c = get<1>(tCcD(i));
    cell_tile[i] = rowtile_of(r) * 2 + (c >> 6);        // 0..127
  }

  uint32_t transcript[pearl_fold::JACKPOT_SIZE];
  for (int ti=0; ti<pearl_fold::JACKPOT_SIZE; ++ti) transcript[ti]=0;
  int gk = 0;

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

  int smem_pipe_read = 0, smem_pipe_write = K_PIPE_MAX-1;
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);
  auto K_BLOCK_MAX = size<2>(tCrA);

  if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE_MAX-2>(); __syncthreads();
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1)) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);
        cp_async_wait<K_PIPE_MAX-2>(); __syncthreads();
      }
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
      if (k_block == 0) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();
        --k_tile_count; if (k_tile_count > 0) ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
      }
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
      if ((++gk % reduce_every_k) == 0) {
        // --- fold 2×64 : XOR des running-sums par tuile via smem ---
        __syncthreads();
        smem.tileacc[threadIdx.x] = 0;          // 128 threads ↔ 128 tuiles
        __syncthreads();
        CUTE_UNROLL
        for (int i = 0; i < NCELL; ++i)
          atomicXor(&smem.tileacc[cell_tile[i]], (uint32_t)tCrC(i));
        __syncthreads();
        int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
        transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(
            transcript[idx], smem.tileacc[threadIdx.x]);
      }
    }
  }
  // POW-CHECK : blake3 keyed sur le transcript de MA tuile (= threadIdx.x).
  auto msg = make_tensor<uint32_t>(Int<16>{});
  for (int i=0;i<16;i++) msg(i)=transcript[i];
  auto cv = make_tensor<uint32_t>(Int<8>{});
  for (int i=0;i<8;i++) cv(i)=pow_key[i];
  blake3::compress_msg_block_u32(msg, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
  bool found = true;
  for (int i=7;i>=0;--i){ if(cv(i)>pow_bound[i]){found=false;break;} if(cv(i)<pow_bound[i])break; }
  if (found) {
    int slot = atomicAdd(found_count, 1);
    if (slot < max_hits) {
      // Tuile T = threadIdx.x : 2 lignes {b*64+p, +32} × 64 colonnes {colgroup*64 ..}.
      int T = threadIdx.x;
      int rowtile = T >> 1, colgroup = T & 1;
      int blk = rowtile / 32, p = rowtile % 32;
      int row0 = blockIdx.x * (int)get<0>(cta_tiler);
      int col0 = blockIdx.y * (int)get<1>(cta_tiler);
      int gr0 = row0 + blk * 64 + p;
      int gr1 = gr0 + 32;
      int gc0 = col0 + colgroup * 64;
      for (int j = 0; j < 64; ++j) {
        hit_rows[slot*128 + j]      = gr0; hit_cols[slot*128 + j]      = gc0 + j;
        hit_rows[slot*128 + 64 + j] = gr1; hit_cols[slot*128 + 64 + j] = gc0 + j;
      }
    }
  }
}
