// ariaminer v0.6.0-ws — kernel GPU GEMM IMMA avec chargement MIXTE A=cp.async + B=TMA (greffe #1).
// Single-buffered (1 tuile K à la fois) : CORRECTNESS d'abord (proof_check bit-exact vs gemm_device 151),
// PUIS pipeline multi-stage pour la perf. Garde fold jackpot + blake3 pow-check + coords hits intacts.
// Briques validées isolées : tma_gemm_test.cu (TMA bit-exact) + tma_gemm_mixed_test.cu (A cp.async + B TMA).
#pragma once
// ⚠️ RÉORIENTATION CHANTIER (10/06, recette dev jetski) : le grind multistage ci-dessous
// touche le mur en GRIND SEUL (218 TH/s) mais NE GAGNE RIEN end-to-end (~163 vs 151) car le
// goulot = le PROLOGUE re-payé par-setup. La vraie voie = PERSISTANT (prologue/fill payé 1×)
// + STREAM du traffic (bandwidth, pas recompute) + 1 feeder CPU/carte + OFFLOAD commit→pool.
// L'OVERLAP est mort (toutes variantes testées). Détail : memory ETAT_ACTUEL + project-pearl-kernel-optimization.
#include <cute/tensor.hpp>
#include "pearl_fold.cuh"
#include "blake3/blake3.cuh"
using namespace cute;

// SharedStorage dédié TMA : A et B mono-stage + 1 mbarrier (TMA de B).
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorageTMA {
  alignas(1024) ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(1024) ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  alignas(16) uint64_t mbar[1];
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class TmaB, class BSmemLayout, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_tma(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, CUTE_GRID_CONSTANT TmaB const tma_b, BSmemLayout sB_layout, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    int reduce_every_k,
    const uint32_t* pow_key, const uint32_t* pow_bound, int* found_count,
    int* hit_rows, int* hit_cols, int max_hits) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);
  Tensor mB = tma_b.get_tma_tensor(select<1,2>(shape_MNK));   // (N,K) via TMA

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});   // (bM,bK,k_tiles)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});   // (bN,bK,k_tiles)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});   // (bM,bN)

  extern __shared__ char smem_[];
  using SS = SharedStorageTMA<TA,TB,ASmemLayout,BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(smem_);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (bM,bK)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (bN,bK)

  // ---- partitions A (cp.async) ----
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);   // (CPY,CPY_M,CPY_K,k_tiles)
  Tensor tAsA = thr_copy_a.partition_D(sA);   // (CPY,CPY_M,CPY_K)

  // ---- partitions B (TMA) ----
  auto cta_b = tma_b.get_slice(Int<0>{});
  Tensor tBgB = cta_b.partition_S(gB);        // (TMA,...,k_tiles)
  Tensor tBsB = cta_b.partition_D(sB);        // (TMA,...)
  constexpr int kTmaBytes = int(size(select<1,2>(CtaTiler{})) * sizeof(TB));

  // ---- fragments MMA + s2r ----
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA);
  Tensor tCrB = thr_mma.partition_fragment_B(sB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);    // (CPY,CPY_M,k_block)
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

  // ================== boucle k-tiles (single-buffered) ==================
  for (int kt = 0; kt < k_tile_count; ++kt) {
    // charge A (cp.async) + B (TMA) de la tuile kt
    copy(copy_a, tAgA(_,_,_,kt), tAsA);
    cp_async_fence();
    if (threadIdx.x == 0) {
      smem.mbar[0] = 0;
      cute::initialize_barrier(smem.mbar[0], 1);
      cute::set_barrier_transaction_bytes(smem.mbar[0], kTmaBytes);
      copy(tma_b.with(smem.mbar[0]), tBgB(_,_,_,kt), tBsB);
    }
    cp_async_wait<0>();
    __syncthreads();
    cute::wait_barrier(smem.mbar[0], 0);
    __syncthreads();

    // s2r tous les k_blocks puis MMA + fold
    CUTE_UNROLL
    for (int kb = 0; kb < K_BLOCK_MAX; ++kb) {
      copy(s2r_atom_a, tXsA(_,_,kb), tXrA(_,_,kb));
      copy(s2r_atom_b, tXsB(_,_,kb), tXrB(_,_,kb));
    }
    CUTE_UNROLL
    for (int kb = 0; kb < K_BLOCK_MAX; ++kb) {
      gemm(mma, tCrA(_,_,kb), tCrB(_,_,kb), tCrC);
      if ((++gk % reduce_every_k) == 0) {
        uint32_t h = pearl_fold::xor_reduction(tCrC);
        int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
        transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(transcript[idx], h);
      }
    }
    __syncthreads();   // smem réutilisé à la tuile suivante
  }

  // dump C accumulé (proof_check correctness GEMM ; inoffensif en prod, retiré pour la perf finale)
  copy(tCrC, tCgC);

  // ================== POW-CHECK (identique gemm_device) ==================
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
      auto cD = make_identity_tensor(make_shape(get<0>(cta_tiler), get<1>(cta_tiler)));
      auto tCcD = thr_mma.partition_C(cD);
      int cnt = size(tCcD);
      int row0 = blockIdx.x * (int)get<0>(cta_tiler);
      int col0 = blockIdx.y * (int)get<1>(cta_tiler);
      for (int i=0;i<cnt;++i){
        hit_rows[slot*128 + i] = row0 + get<0>(tCcD(i));
        hit_cols[slot*128 + i] = col0 + get<1>(tCcD(i));
      }
    }
  }
}

// ============================================================================
// ÉTAPE 1 (10/06) — gemm_device_tma_ms : MULTI-STAGE.
// = la structure pipeline du 151 (gemm_device : A cp.async multi-stage, prologue
//   + steady-state + ring de buffers) MAIS B chargé en TMA multi-stage (1 mbarrier
//   par stage). Tous les 8 warps font du MMA (saveur alpha-native, PAS de prod/conso).
// smem A,B = 3-mode (bM/bN, bK, PIPE) ; PIPE = size<2>(ASmemLayout). PIPE=2 pour 128×256.
// Ring mbarrier : ré-init du barrier de la stage écrite juste avant son TMA → le
// consommateur attend toujours phase 0 ; les __syncthreads garantissent la sûreté
// (aucune ré-init concurrente d'un wait : on n'écrit une stage que K_PIPE-1 tuiles
//  après l'avoir lue, avec __syncthreads entre).
// Fold/pow-check/coords INCHANGÉS → transcript bit-identique à gemm_device (ordre
// des tuiles préservé). proof_check bit-exact AVANT toute mesure de perf.
// ============================================================================
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorageTMA_MS {
  alignas(1024) ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;   // (bM,bK,PIPE)
  alignas(1024) ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;   // (bN,bK,PIPE)
  alignas(16) uint64_t mbar[4];   // 1 par stage (≤4 stages)
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class TmaB, class BSmemLayout, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma, bool DumpC = true>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_tma_ms(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, CUTE_GRID_CONSTANT TmaB const tma_b, BSmemLayout sB_layout, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    int reduce_every_k,
    const uint32_t* pow_key, const uint32_t* pow_bound, int* found_count,
    int* hit_rows, int* hit_cols, int max_hits) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);
  Tensor mB = tma_b.get_tma_tensor(select<1,2>(shape_MNK));

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});   // (bM,bK,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});   // (bN,bK,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});   // (bM,bN)

  extern __shared__ char smem_[];
  using SS = SharedStorageTMA_MS<TA,TB,ASmemLayout,BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(smem_);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (bM,bK,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (bN,bK,PIPE)
  constexpr int K_PIPE = size<2>(ASmemLayout{});

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);   // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);   // (CPY,CPY_M,CPY_K,PIPE)
  auto cta_b = tma_b.get_slice(Int<0>{});
  Tensor tBgB = cta_b.partition_S(gB);        // (TMA,...,k)
  Tensor tBsB = cta_b.partition_D(sB);        // (TMA,...,PIPE)
  constexpr int kTmaBytes = int(size(select<1,2>(CtaTiler{})) * sizeof(TB));

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);    // (CPY,CPY_M,k_block,PIPE)
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

  // ---- prologue : remplir les stages 0..K_PIPE-2 ----
  CUTE_UNROLL
  for (int kp = 0; kp < K_PIPE-1; ++kp) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,kp));
    cp_async_fence();
    if (threadIdx.x == 0) {
      cute::initialize_barrier(smem.mbar[kp], 1);
      cute::set_barrier_transaction_bytes(smem.mbar[kp], kTmaBytes);
      copy(tma_b.with(smem.mbar[kp]), tBgB(_,_,_,k_tile_next), tBsB(_,_,_,kp));
    }
    --k_tile_count; if (k_tile_count > 0) ++k_tile_next;
  }

  int smem_pipe_read = 0, smem_pipe_write = K_PIPE-1;
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  // amorce du 1er k_block (attend A cp.async + B TMA de la stage read)
  if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE-2>();
    cute::wait_barrier(smem.mbar[smem_pipe_read], 0);
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
        cute::wait_barrier(smem.mbar[smem_pipe_read], 0);
        __syncthreads();
      }
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
      if (k_block == 0) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        cp_async_fence();
        if (threadIdx.x == 0) {
          cute::initialize_barrier(smem.mbar[smem_pipe_write], 1);   // ré-init → wait phase 0
          cute::set_barrier_transaction_bytes(smem.mbar[smem_pipe_write], kTmaBytes);
          copy(tma_b.with(smem.mbar[smem_pipe_write]), tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        }
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

  if constexpr (DumpC) copy(tCrC, tCgC);   // dump C pour proof_check (retiré à la perf finale)

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
      auto cD = make_identity_tensor(make_shape(get<0>(cta_tiler), get<1>(cta_tiler)));
      auto tCcD = thr_mma.partition_C(cD);
      int cnt = size(tCcD);
      int row0 = blockIdx.x * (int)get<0>(cta_tiler);
      int col0 = blockIdx.y * (int)get<1>(cta_tiler);
      for (int i=0;i<cnt;++i){
        hit_rows[slot*128 + i] = row0 + get<0>(tCcD(i));
        hit_cols[slot*128 + i] = col0 + get<1>(tCcD(i));
      }
    }
  }
}
