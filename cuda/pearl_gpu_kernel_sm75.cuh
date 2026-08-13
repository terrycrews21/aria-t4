// ariaminer — kernel GPU 8×16 TURING (sm_75, Tesla T4). GEMM IMMA + fold jackpot + pow-check.
// Clone de gemm_device_2x64 (v1.1, regroup accumulateur partagé) porté sur Turing :
//   * MMA  : SM75_8x8x16_S32S8S8S32_TN (atom K=16, moitié du pas SM80 K=32).
//   * G2S  : Copy_Atom<UniversalCopy<uint128_t>,int8_t> — cp.async N'EXISTE PAS sur
//            Turing ; cp_async_fence()/cp_async_wait() restent en place (no-ops quand
//            CUTE_ARCH_CP_ASYNC_SM80_ENABLED est undefined) : la structure pipelinée
//            se dégrade proprement en copies synchrones + __syncthreads, bit-exact.
//   * Smem : pipeline 2 STAGES (Shape<_1,_2>) → (128×64 + 128×64)×2 o + tileacc 512 o
//            ≈ 33 Ko < 48 Ko/CTA (limite Turing) — cudaFuncSetAttribute réflexe gardé.
//   * S2R  : ldmatrix (SM75_U32x4_LDSM_N) — valide dès sm_75.
//
// Tuile-canonical 8×16 CONTIGUË (pool-legal : le node reconstruit les PeriodicPattern
// depuis la preuve soumise ; Rust canonical_gpu_config gate ARIA_SM75 → rows=[0..7],
// cols=[0..15]). Découpage du CTA 128×128 :
//   16 row-groupes (r>>3) × 8 col-groupes (c>>4) = 128 tuiles de 8 rows × 16 cols.
//   cell_tile(r,c) = (r>>3)*8 + (c>>4)          (coords LOCALES au CTA, 0..127)
//   thread T (0..127) possède la tuile T :
//     tr = T>>3 (0..15)  →  rows = row0 + tr*8 + [0..7]
//     tc = T&7  (0..7)   →  cols = col0 + tc*16 + [0..15]
// Le fragment MMA d'un thread est éparpillé (mapping tensor-core) : impossible
// d'obtenir "thread = tuile contiguë" par permutation. On REGROUPE explicitement
// comme dans gemm_device_2x64 : accumulateur partagé tileacc[128] + atomicXor à
// chaque fold, puis thread T relit tileacc[T] (= XOR des cellules de SA tuile) et
// folde dans son transcript. Le pow-check est identique : blake3 keyed officiel
// sur le transcript de la tuile, compare bound, émission des 128 coords (row-major
// 8 outer × 16 inner).
#pragma once
#include <cute/tensor.hpp>
#include "pearl_fold.cuh"
#include "blake3/blake3.cuh"
using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorageSm75 {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  uint32_t tileacc[128];   // 1 accumulateur XOR par tuile-canonical 8×16 du CTA (512 o)
  uint32_t wsum[2][4 * 64];  // v2 fold: sommes par warp (2 buffers × 4 warps × 64 tuiles)
};

// Coords locales (0..127) d'une cellule du CTA → index de sa tuile-canonical (0..127).
__device__ __forceinline__ int sm75_cell_tile(int row, int col) {
  return (row >> 3) * 8 + (col >> 4);
}

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_sm75(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    int reduce_every_k,
    const uint32_t* pow_key, const uint32_t* pow_bound, int* found_count,
    int* hit_rows, int* hit_cols, int max_hits, int nbx, int nby,
    // --- dump DEBUG validation (optionnel, passé nullptr en prod → coût nul) : si
    //     dump_tr non-null, le CTA (0,0) écrit le transcript[16] de chaque thread
    //     (= sa tuile), les 128 coords (r,c) locales de son fragment MMA et les 128
    //     sommes int32 FINALES de son fragment. Paramètres EXPLICITES : nvcc
    //     n'accepte pas de valeur par défaut sur les arguments d'un kernel template.
    uint32_t* dump_tr, int* dump_rows, int* dump_cols, int32_t* dump_c) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  extern __shared__ char smem_[];
  using SS = SharedStorageSm75<TA,TB,ASmemLayout,BSmemLayout>;
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
    cp_async_fence();          // no-op sm_75 (guardé CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    --k_tile_count; if (k_tile_count > 0) ++k_tile_next;
  }

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // Coords locales (0..127) de chaque cellule du fragment → tuile-canonical (précalc).
  auto cD = make_identity_tensor(make_shape(get<0>(cta_tiler), get<1>(cta_tiler)));
  auto tCcD = thr_mma.partition_C(cD);
  constexpr int NCELL = decltype(size(tCcD))::value;   // = 128 (128×128 / 128 thr)
  int cell_tile[NCELL];
  CUTE_UNROLL
  for (int i = 0; i < NCELL; ++i) {
    int r = get<0>(tCcD(i));
    int c = get<1>(tCcD(i));
    cell_tile[i] = sm75_cell_tile(r, c);
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
  // ── Turing double-buffering REGISTRE ────────────────────────────────────────
  // Pas de cp.async sur sm_75 : une copie globale→smem est SYNCHRONE, donc le CTA
  // bloque sur la latence DRAM juste avant la barrière de fin de tuile (mesuré :
  // ~2.8 % du peak IMMA T4). On étage donc par REGISTRES — global→regs émis un
  // tuile-k EN AVANCE (au k_block 0), regs→smem publié en fin de tuile — pour que
  // la latence se recouvre avec les MMA de la tuile courante.
  Tensor tArA = make_fragment_like(tAsA(_,_,_,0));
  Tensor tBrB = make_fragment_like(tBsB(_,_,_,0));
  int stage_fill = -1;   // stage smem que le prefetch registre doit publier

  if (K_BLOCK_MAX > 1) {
    if constexpr (K_PIPE_MAX > 1) {
      cp_async_wait<K_PIPE_MAX-2>();
    }
    __syncthreads();   // wait no-op + syncthreads réel
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1)) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
        // Publication du prefetch registre dans le stage préparé au k_block 0 :
        // AVANT la barrière, donc visible pour les lectures de la tuile suivante.
        if (stage_fill >= 0) {
          copy(tArA, tAsA(_,_,_,stage_fill));
          copy(tBrB, tBsB(_,_,_,stage_fill));
        }
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);
        if constexpr (K_PIPE_MAX > 1) {
          cp_async_wait<K_PIPE_MAX-2>();
        }
        __syncthreads();
      }
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
      if (k_block == 0) {
        // Émission ANTICIPÉE global→registres (latence recouverte par les MMA
        // de cette tuile). La publication regs→smem a lieu en fin de tuile.
        stage_fill = smem_pipe_write;
        copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
        cp_async_fence();
        --k_tile_count; if (k_tile_count > 0) ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
      }
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
      if ((++gk % reduce_every_k) == 0) {
        // --- fold 8×16 canonical SANS atomiques (v2) --------------------------
        // Chaque tuile canonical reçoit 64 contributions = 32 dans chacune de DEUX
        // warps de même parité (mapping vérifié : tile = 16*(g&7) + 8*(w&1) + (g>>3)).
        // 1) XOR-pair en registres (2 cellules adjacentes partagent la tuile).
        // 2) réduction XOR complète du warp par 5 shuffles butterflies.
        // 3) écriture bankée par warp (wsum[4][64], double-buffer) puis UNE barrière.
        // XOR est associatif/commutatif → BIT-EXACT avec l'ancienne voie atomics.
        const int lane = threadIdx.x & 31;
        const int w    = threadIdx.x >> 5;
        const int par  = w & 1;
        const int buf  = (gk / reduce_every_k - 1) & 1;
        CUTE_UNROLL
        for (int g = 0; g < NCELL / 2; ++g) {
          uint32_t x = (uint32_t)tCrC(2 * g) ^ (uint32_t)tCrC(2 * g + 1);
          CUTE_UNROLL
          for (int m = 16; m > 0; m >>= 1)
            x ^= __shfl_xor_sync(0xffffffffu, x, m);
          // store groupé : lane (g&31) écrit la tuile g de SA warp (aucune course,
          // chaque warp écrit dans SON slot).
          if (lane == (g & 31)) smem.wsum[buf][w * 64 + g] = x;
        }
        __syncthreads();
        // tuile t = threadIdx.x ; son index g-local : g = (t&7)*8 + (t>>4) (vérifié).
        const int p  = (threadIdx.x >> 3) & 1;
        const int go = (threadIdx.x & 7) * 8 + (threadIdx.x >> 4);
        const int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
        transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(
            transcript[idx], smem.wsum[buf][p * 64 + go] ^ smem.wsum[buf][(p + 2) * 64 + go]);
        // PAS de barrière finale : le prochain fold écrit l'AUTRE buffer.
        // (le __syncthreads() du pipeline G2S garde la cohérence entre folds)
      }
    }
  }

  // ---- dump DEBUG validation (optionnel) : CTA (0,0) seulement, co-indexé fragment ----
  if (dump_tr != nullptr && blockIdx.x == 0 && blockIdx.y == 0) {
    for (int i = 0; i < 16; ++i) dump_tr[threadIdx.x * 16 + i] = transcript[i];
    if (dump_rows != nullptr && dump_cols != nullptr) {
      for (int i = 0; i < NCELL; ++i) {
        dump_rows[threadIdx.x * 128 + i] = get<0>(tCcD(i));
        dump_cols[threadIdx.x * 128 + i] = get<1>(tCcD(i));
      }
    }
    if (dump_c != nullptr) {
      for (int i = 0; i < NCELL; ++i) dump_c[threadIdx.x * 128 + i] = tCrC(i);
    }
  }

  // POW-CHECK : blake3 keyed officiel + compare bound. Sur hit → émission des
  // 128 coords de la tuile contiguë 8×16 de threadIdx.x (row-major : 8 rows outer).
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
      int T = threadIdx.x;
      int tr = T >> 3, tc = T & 7;            // row-groupe 0..15, col-groupe 0..7
      int row0 = blockIdx.x * (int)get<0>(cta_tiler);
      int col0 = blockIdx.y * (int)get<1>(cta_tiler);
      for (int i = 0; i < 8; ++i) {           // 8 rows de la tuile contiguë
        int gr = row0 + tr * 8 + i;
        for (int j = 0; j < 16; ++j) {        // 16 cols de la tuile contiguë
          hit_rows[slot*128 + i*16 + j] = gr;
          hit_cols[slot*128 + i*16 + j] = col0 + tc * 16 + j;
        }
      }
      (void)nbx; (void)nby;   // dims de grille — signature commune (variante persistante)
    }
  }
}
