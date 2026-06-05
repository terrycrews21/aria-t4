// ariaminer v0.4.0 — kernel GPU complet (GEMM IMMA + fold jackpot + pow-check),
// extrait validé de jackpot_cute.cu. Inclus par pearl_gpu_lib.cu (FFI) et le harnais de test.
#pragma once
#include <cute/tensor.hpp>
#include "pearl_fold.cuh"
#include "blake3/blake3.cuh"
using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
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
  using SS = SharedStorage<TA,TB,ASmemLayout,BSmemLayout>;
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
        uint32_t h = pearl_fold::xor_reduction(tCrC);
        int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
        transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(transcript[idx], h);
      }
    }
  }
  // POW-CHECK : blake3 keyed officiel + compare bound. Sur hit → enregistre (block,lane).
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
      // coords GLOBALES des 128 cellules du fragment (= 8 rows × 16 cols) ; Rust dédoublonne.
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
// v0.5.0 — GRIND PERSISTANT À OCCUPATION BRIDÉE.
// Identique à gemm_device, MAIS lancé avec une grille 1D de G blocs < nb_tuiles :
// chaque bloc BOUCLE en grid-stride sur les tuiles (tile = blockIdx.x ; += gridDim.x).
// En choisissant G < (SM × occupancy), on laisse des SM LIBRES → le prologue du
// setup N+1 (stream séparé, memory-bound) tourne EN CONCURRENCE pendant le grind
// du setup N (tensor-core-bound). C'est le headroom que Ctx2 (grille pleine) ratait.
//   nbx = ceil(m/128), nby = ceil(n/128), total = nbx*nby.
//   coords de tuile : bx = tile % nbx, by = tile / nbx  (col-major sur x = lignes).
// ============================================================================
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_persist(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    int reduce_every_k,
    const uint32_t* pow_key, const uint32_t* pow_bound, int* found_count,
    int* hit_rows, int* hit_cols, int max_hits, int nbx, int nby) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

  extern __shared__ char smem_[];
  using SS = SharedStorage<TA,TB,ASmemLayout,BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(smem_);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);

  const int total = nbx * nby;
  for (int tile = blockIdx.x; tile < total; tile += gridDim.x) {
    int bx = tile % nbx, by = tile / nbx;
    auto cta_coord = make_coord(bx, by, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

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
          uint32_t h = pearl_fold::xor_reduction(tCrC);
          int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
          transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(transcript[idx], h);
        }
      }
    }
    // POW-CHECK : blake3 keyed officiel + compare bound. Sur hit → enregistre (block,lane).
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
        int row0 = bx * (int)get<0>(cta_tiler);
        int col0 = by * (int)get<1>(cta_tiler);
        for (int i=0;i<cnt;++i){
          hit_rows[slot*128 + i] = row0 + get<0>(tCcD(i));
          hit_cols[slot*128 + i] = col0 + get<1>(tCcD(i));
        }
      }
    }
    __syncthreads();   // libère le smem avant la tuile suivante (WAR)
  }
}
