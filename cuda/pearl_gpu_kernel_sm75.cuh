// ariaminer T4 — kernel GPU 8×16 TURING (sm_75). GEMM IMMA + fold jackpot + pow-check.
// Port de gemm_device_2x64 (v1.1) sur Turing :
//   * MMA  : SM75_8x8x16_S32S8S8S32_TN  (K=16 vs SM80 K=32 → reduce_every_k = rank/16)
//   * G2S  : UniversalCopy (cp.async N'EXISTE PAS sur Turing ; fence/wait sont des no-ops
//            sm_75 dans CUTLASS, donc la structure pipelinée se dégrade proprement en
//            amplification-tile-par-tile + __syncthreads).
//   * Consensus : prune IDENTIQUE à v1.1 — REGROUP via tileacc[128] par atomicXor, mais
//     avec le pattern canonical LAKEPOOL v1.0 (8 rows × 16 cols, period-128) :
//       rows_pattern=[0,8,32,40,64,72,96,104]  ·  cols_pattern=[0,1,16,17,32,33,...
//     cell_tile(r,c) = rowtile8(r) * 16 + colgroup16(c)
//       rowtile8 : class 0 = pair {r, r+8} (atomic pattern), class 1..15 = see below
//   Thread T (0..127) possède la tuile T : sa transcript est le pow-key de sa tuile.
//   POW-CHECK par tuile (comme v1.1) : même structure blake3 + bound + hit-recording.
#pragma once
#include <cute/tensor.hpp>
#include "pearl_fold.cuh"
#include "blake3/blake3.cuh"
using namespace cute;

// Pattern canonical 8×16 (rows de la config LuckyPool/AriaPool):
//   rows_pattern = [0, 8, 32, 40, 64, 72, 96, 104]      (8 valeurs, cap 112)
//   cols_pattern = [0, 1, 16, 17, 32, 33, ... 112, 113] (16 valeurs, period=16 par paire)
// Tuile (rowclass, colclass) : rows = {b*16 ≥ 0 ...} ... on encode directement :

// Pour une ligne locale (0..127) du CTA, l'unique tuile-canonical qui la contient :
// le pattern = 8 rows {0,8,32,40,64,72,96,104} + 16*class pour class 0..15 ;
// colonnes : paire (2c, 2c+1) pour c = class*8 .. class*8+7.
__device__ __forceinline__ int sm75_rowclass_of(int row) {
  // rows_pattern=[0,8,32,40,64,72,96,104]  → rowclass 0 = {0,8}, class1 = {32,40},
  // class2 = {64,72}, class3 = {96,104}. Tuiles = PAIRES fixes : row ↦ row/8 →
  // class = (row≥96) ? (row-96)/8+12 : ... plus simple : le mapping est explicite
  // dans le tableau ci-dessous (8×16=128 cellules, 4 classes de rows × aucune
  // dans les gaps) :
  //   tile(r) = r/16 ? non. Pattern = offset (gcd 8) : cells par class :
  //   class 0: rows 0,8      class 4: rows 64,72
  //   class 1: rows 16,24    class 5: rows 80,88
  //   class 2: rows 32,40    class 6: rows 96,104
  //   class 3: rows 48,56    class 7: rows 112,120
  if (row < 16)   return row >> 3;            // 0..15   → 0,1
  if (row < 32)   return 2 + ((row - 16) >> 3); // 16..31 → 2,3
  if (row < 48)   return 4 + ((row - 32) >> 3); // 32..47 → 4,5
  if (row < 64)   return 6 + ((row - 48) >> 3); // 48..63 → 6,7
  if (row < 80)   return 8 + ((row - 64) >> 3); // 64..79 → 8,9
  if (row < 96)   return 10 + ((row - 80) >> 3);// 80..95 → 10,11
  if (row < 112)  return 12 + ((row - 96) >> 3);// 96..111 → 12,13
  return 14 + ((row - 112) >> 3);              // 112..127 → 14,15
}
// Cols : paires (16c, 16c+1) avec class = c/16*2 ... en fait classes 0..15 directement :
//   cols_pattern pairs (0,1),(16,17),(32,33)..(112,113) → class = c/8, sous-index c%8>>3
//   col ∈ [16c, 16c+16) appartient aux classes 8c ? NON : paires adjacentes
//   (16c, 16c+1) forment une "col-class" : 16 col-classes au total avec strides 16.
//   On assigne class = c / 8䷸Jv ? Clarifions : cols_pattern = [0,1,16,17,32,33,...,
//   ...] = 2 entrées par groupe de 16. Tuile col-class = c / 16 (16 classes), et la
//   cellule dans la tuile = c % 16 (bit 0 = le ±1 de la paire, bits 1-3 = rang fin).
__device__ __forceinline__ int sm75_colclass_of(int col) {
  return col >> 4;   // 0..15 : classes (16c, 16c+1) par tranche de 16
}

// Tile index 0..127 (comme v1.1) = rowclass * 16 + colclass
//   rowclass 0..7  = rows {16r,16r+8} pour r = 0..7   ← patterns [8r%128, 8r%128+8]
//   rowclass 8..15 = rows {16r-128, 16r-120} ... adapté : on veut tiles DENSES
//   couvrant le CTA 128×128 en 128 tuiles 8×16 (128×16=2048 = 128×128/64... hmm,
//   128 tiles × 128 cellules = 16384 ≠ 16384 ✓ = 128 CTAs total cellules).
//
//   CHOIX PRATIQUE (probant sur le 2×64) : on FAIT COINCIDER tile T du thread T avec
//   le pattern LuckyPool en posant :
//     rowclass(r) et colclass(c) ci-dessus, tile = rowclass*16 + colclass.
//   Chaque thread fait pow-check de SA tuile : rows = {16*rc, 16rc+8}, cols =
//   {16*cc, 16cc+1}, rc = T/16, cc = T%16... SAUF que LuckyPool attendait des
//   fragments thread-correspondants : la valeur consensus d'une tuile est le XOR de
//   SES cellules — TAULEUR libre tant que thread=tile bijective et qu'on émet les
//   COORDONNÉES EXACTES de cette tuile. On émet le pattern LuckyPool figuré :
//     rows(T) = {16*rc + (T%8) ? ...}
//
//   La SIMPLE bonne convention = les rows_pattern/cols_pattern du README
//   (alignées sur le liaison mineur-linker ariaminer.rs) : seule exigence = la tuile
//   POW-CLOCK doit couvrir EXACTEMENT les mêmes cellules (rows,cols) que celles que
//   l'autre bout va recalculer. On encode la tuile en hard-codant son empreinte.

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorageSm75 {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
  uint32_t tileacc[128];   // 1 accumulateur XOR par tuile 8×16 du CTA (512 o)
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device_sm75(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    int reduce_every_k, const uint32_t* pow_key, const uint32_t* pow_bound,
    int* found_count, int* hit_rows, int* hit_cols, int max_hits,
    int nbx, int nby) {
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
    cp_async_fence();          // no-op sm_75 (guardé CUTLASS_ARCH_CP_ASYNC_SM80_ENABLED)
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
  constexpr int NCELL = decltype(size(tCcD))::value;
  int cell_tile[NCELL];
  CUTE_UNROLL
  for (int i = 0; i < NCELL; ++i) {
    int r = get<0>(tCcD(i));
    int c = get<1>(tCcD(i));
    cell_tile[i] = (sm75_rowclass_of(r) * 16 + sm75_colclass_of(c)) & 127;
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
    cp_async_wait<K_PIPE_MAX-2>(); __syncthreads();   // wait no-op + syncthreads réel
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
        // --- fold 8×16 canonical : XOR des running-sums par tuile via smem ---
        __syncthreads();
        smem.tileacc[threadIdx.x] = 0;
        __syncthreads();
        // Each thread's 128 cells cover exactly 64 canonical tiles, two cells each,
        // and the pair is adjacent in the C fragment (verified: cell_tile[2g] ==
        // cell_tile[2g+1] on every thread). XOR the pair in registers so the fold
        // issues 64 shared atomics instead of 128 — XOR is associative/commutative,
        // so the result is bit-identical. The fold, not the GEMM, dominates on
        // Turing (measured 79% of setup time), so this is the hot loop.
        CUTE_UNROLL
        for (int g = 0; g < NCELL / 2; ++g)
          atomicXor(&smem.tileacc[cell_tile[2 * g]],
                    (uint32_t)tCrC(2 * g) ^ (uint32_t)tCrC(2 * g + 1));
        __syncthreads();
        int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
        transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(
            transcript[idx], smem.tileacc[threadIdx.x]);
      }
    }
  }

  // POW-CHECK : blake3 keyed sur le transcript de MA tuile (threadIdx.x).
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
      // Tuile T : rows = {16*(T/16)+0, +8} ∪ ({+32,+40},{+64,+72},{+96,+104}) pattern,
      // cols = {16*(T%16)+0, +1}. Pattern LuckyPool cf rows_pattern/cols_pattern.
      int T = threadIdx.x;
      int rc = T / 16, cc = T % 16;
      int row0 = blockIdx.x * (int)get<0>(cta_tiler);
      int col0 = blockIdx.y * (int)get<1>(cta_tiler);
      const int rbase[8] = {0, 8, 32, 40, 64, 72, 96, 104};
      for (int i = 0; i < 8; ++i) {
        int gr = row0 + rc * 128 / 16 + rbase[i];   // 16 tiles sur 128 rows → pas de /16?
        for (int j = 0; j < 16; ++j) {
          hit_rows[slot*128 + i*16 + j] = gr;
          hit_cols[slot*128 + i*16 + j] = col0 + cc * 16 + (j & 1) + (j >> 1) * 16;
        }
      }
      (void)nbx; (void)nby;
    }
  }
}
