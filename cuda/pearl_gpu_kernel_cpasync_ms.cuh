// ============================================================================
// v0.6.3-beta — gemm_device_cpasync_ms : JUMEAU PORTABLE du grind multistage.
//
// = `gemm_device_tma_ms` (cuda/pearl_gpu_kernel_tma.cuh) À L'IDENTIQUE, sauf que
// B est chargé en **cp.async** (comme A) au lieu du **TMA** (SM90, Hopper/Blackwell).
// → tourne sur Ampere (sm_80/86, RTX 30xx) et Ada (sm_89, RTX 40xx), pas seulement
//   Blackwell. La lib choisit ce kernel quand le GPU n'a pas le TMA (major < 9 hors
//   sm_120). Le cp.async (SM80) est le SEUL prérequis HW, présent dès Ampere.
//
// CE QUI EST BYTE-IDENTIQUE au path TMA (donc fold/jackpot inchangé) :
//   - géométrie période-256 : bM=128, bN=256, bK=128, MMA 2×4 warps, tuile 16×8×32
//   - swizzle de grille (v0.6.1), rank/reduce_every_k, ordre des tuiles k
//   - smem B = MÊME layout swizzlé sB_layout → contenu smem identique au remplissage
//     TMA → fragments s2r identiques → MMA identiques → transcript/cv identiques
//   - fold xor_reduction, blake3, pow-check LE/BE, bound, coords des hits
// SEULE DIFFÉRENCE : le MÉCANISME de transfert gmem→smem de B (cp.async vs TMA).
// Le smem B est rempli selon sB_layout dans les deux cas → même octets. Prouvé
// byte-exact sur 5080 via jackpot_diff/overlap_check en forçant ce path (ARIA_FORCE_CPASYNC).
//
// Pas de mbarrier (le TMA en avait besoin ; cp.async se synchronise via
// cp_async_wait + __syncthreads, comme le path A). SharedStorage allégé.
// ============================================================================
#pragma once
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"
#include "pearl_fold.cuh"
using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorageCPA_MS {
  alignas(1024) ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;   // (bM,bK,PIPE)
  alignas(1024) ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;   // (bN,bK,PIPE)
  // pas de mbar : cp.async se synchronise par cp_async_wait + __syncthreads
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class TiledCopyB, class BSmemLayout, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma, bool DumpC = true,
          bool BigEndian = false>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value, 1)
void gemm_device_cpasync_ms(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, TiledCopyB copy_b, BSmemLayout sB_layout, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    int reduce_every_k, int swz_g,
    const uint32_t* pow_key, const uint32_t* pow_bound, int* found_count,
    int* hit_rows, int* hit_cols, int max_hits) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);   // (N,K) plain (pas TMA)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

  // SWIZZLE de grille (v0.6.1) — IDENTIQUE au path TMA, pur calcul d'index, portable.
  int bx = blockIdx.x, by = blockIdx.y;
  if (swz_g > 1) {
    int gm = gridDim.x, gn = gridDim.y;
    int bid  = bx + by * gm;
    int band = bid / (swz_g * gn);
    int r    = bid % (swz_g * gn);
    int g    = min(swz_g, gm - band * swz_g);
    bx = band * swz_g + (r % g);
    by = r / g;
  }
  auto cta_coord = make_coord(bx, by, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});   // (bM,bK,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});   // (bN,bK,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});   // (bM,bN)

  extern __shared__ char smem_[];
  using SS = SharedStorageCPA_MS<TA,TB,ASmemLayout,BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(smem_);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (bM,bK,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (bN,bK,PIPE)
  constexpr int K_PIPE = size<2>(ASmemLayout{});

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);   // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);   // (CPY,CPY_M,CPY_K,PIPE)
  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);   // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);   // (CPY,CPY_N,CPY_K,PIPE)

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

  uint32_t transcript[pearl_fold::JACKPOT_SIZE];
  for (int ti=0; ti<pearl_fold::JACKPOT_SIZE; ++ti) transcript[ti]=0;
  int gk = 0;
  auto K_BLOCK_MAX = size<2>(tCrA);
  int k_tile_count = size<3>(tAgA);
  int k_tile_next = 0;

  // ---- prologue : remplir les stages 0..K_PIPE-2 (A ET B en cp.async, même fence) ----
  CUTE_UNROLL
  for (int kp = 0; kp < K_PIPE-1; ++kp) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,kp));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,kp));
    cp_async_fence();
    --k_tile_count; if (k_tile_count > 0) ++k_tile_next;
  }

  int smem_pipe_read = 0, smem_pipe_write = K_PIPE-1;
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  // amorce du 1er k_block (attend A+B cp.async de la stage read)
  if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE-2>();
    __syncthreads();
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE-1)) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);
        cp_async_wait<K_PIPE-2>();
        __syncthreads();
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
        smem_pipe_read = (smem_pipe_read == K_PIPE-1) ? 0 : smem_pipe_read+1;
      }
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
      if ((++gk % reduce_every_k) == 0) {
        uint32_t h = pearl_fold::xor_reduction(tCrC);
        int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
        transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(transcript[idx], h);
      }
    }
  }

  if constexpr (DumpC) copy(tCrC, tCgC);

  // ---- fold final + pow-check : STRICTEMENT IDENTIQUE au path TMA ----
  auto msg = make_tensor<uint32_t>(Int<16>{});
  for (int i=0;i<16;i++) msg(i)=transcript[i];
  auto cv = make_tensor<uint32_t>(Int<8>{});
  for (int i=0;i<8;i++) cv(i)=pow_key[i];
  blake3::compress_msg_block_u32(msg, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
  bool found = true;
  if constexpr (BigEndian) {
    for (int i=0;i<8;++i){
      uint32_t h = __byte_perm(cv(i), 0, 0x0123);
      uint32_t b = __byte_perm(pow_bound[i], 0, 0x0123);
      if(h>b){found=false;break;} if(h<b)break;
    }
  } else {
    for (int i=7;i>=0;--i){ if(cv(i)>pow_bound[i]){found=false;break;} if(cv(i)<pow_bound[i])break; }
  }
  if (found) {
    int slot = atomicAdd(found_count, 1);
    if (slot < max_hits) {
      auto cD = make_identity_tensor(make_shape(get<0>(cta_tiler), get<1>(cta_tiler)));
      auto tCcD = thr_mma.partition_C(cD);
      int cnt = size(tCcD);
      int row0 = bx * (int)get<0>(cta_tiler);
      int col0 = by * (int)get<1>(cta_tiler);
      for (int i=0;i<cnt;++i){
        hit_rows[slot*128 + i] = row0 + get<0>(tCcD(i));
        hit_cols[slot*128 + i] = col0 + get<1>(tCcD(i));
      }
    }
  }
}
