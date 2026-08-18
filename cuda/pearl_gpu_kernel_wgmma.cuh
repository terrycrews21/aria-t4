// pearl_gpu_kernel_wgmma.cuh — SM90 WGMMA grind kernel for H100/H200.
// ============================================================================
// WHY THIS EXISTS:
//   gemm_device_tma_ms uses SM80_16x8x32_S32S8S8S32_TN (mma.sync). On H100
//   (sm_90a) mma.sync must stage operands through registers via ldmatrix and
//   caps at roughly half the tensor-core throughput of wgmma.mma_async, which
//   reads A/B DIRECTLY from shared memory descriptors.
//
//   Measured on H100 (same 700W power cap):
//     mma.sync tworker : ~380 TH/s displayed, 1815 MHz, 29% mem-util
//     PeakMiner ref    : ~552 TH/s,           1350 MHz, 67% mem-util
//   More work at lower clock => wgmma-class kernel. This file ports the grind
//   to wgmma to close that gap.
//
// ARCHITECTURE:
//   - Tile: 128x256x128 (same grid geometry as the mma.sync path)
//   - Threads: 256 = 2 warpgroups; WG0 -> rows 0-63, WG1 -> rows 64-127
//   - Each wgmma: m64n256k32 s8.s8.s32, both operands from SMEM (SS variant)
//   - Pipeline: 4 stages (4 x 48KB = 192KB < 228KB H100 opt-in smem, 1 CTA/SM)
//   - Loads: TMA for BOTH A and B (one mbarrier per stage, thread 0 issues)
//   - Early prefetch: the TMA for the NEXT tile is issued right after waiting
//     on the CURRENT stage, so loads overlap the entire wgmma compute window.
//
// FOLD / TRANSCRIPT (semantics identical to mma.sync path):
//   Every reduce_every_k (= rank/32) k-blocks, each thread XOR-reduces its
//   accumulator fragment (128 x s32) to one u32 and folds it into a 16-slot
//   rotating transcript via rotl<13>^= . After the full K loop, blake3 keyed
//   hash of the transcript is compared LE/BE against the bound; <= bound=HIT.
//   Because XOR is commutative, the fold value per thread only depends on the
//   SET of C elements that thread owns, not their register order.
//
// PATTERN (job_key / verifier) — BAN-AVOIDANCE CRITICAL:
//   wgmma CLayout_64x256 gives each thread rows {0,8} (relative to its 64-row
//   warpgroup slice) x cols {0,1,8,9,...,248,249} = 128 elements. This is a
//   DIFFERENT footprint from mma.sync (8 rows x 16 cols). The rows_pattern /
//   cols_pattern declared to the pool MUST be re-derived for this kernel
//   (wgmma_pattern_dump) and shipped together with it; a mismatch means every
//   share fails verifier reconstruction (Hash A mismatch) and risks bans.
//   Do NOT enable on a real pool before pattern verification + a low-stakes
//   live test showing zero rejects.
//
// T4 / SM75 NOTE: this file is sm_90a-only (wgmma + TMA do not exist on
// Turing). The sm75 paths remain separate; the general lessons (deeper
// pipeline, early prefetch, avoiding register staging) still apply there but
// with cp.async + mma.sync instead.
// ============================================================================
#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include "pearl_fold.cuh"
#include "blake3/blake3.cuh"

using namespace cute;

// Shared memory: A tile + B tile per stage + one mbarrier per stage.
// alignas(1024): TMA requires 128B-aligned smem boxes; 1KB keeps it safe.
template <class SmemLayoutA, class SmemLayoutB, int kStages>
struct SharedStorageWGMMA {
  alignas(1024) ArrayEngine<int8_t, cosize_v<SmemLayoutA>> A;  // (bM,bK,PIPE)
  alignas(1024) ArrayEngine<int8_t, cosize_v<SmemLayoutB>> B;  // (bN,bK,PIPE)
  alignas(16) uint64_t mbar[kStages];
};

template <class ProblemShape, class CtaTiler,
          class ASmemLayout, class TmaA,
          class BSmemLayout, class TmaB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          bool DumpC = false, bool BigEndian = false>
__global__ static __launch_bounds__(256, 1)  // 2 warpgroups, 1 CTA/SM (max regs)
void gemm_device_wgmma_ms(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    ASmemLayout sA_layout, CUTE_GRID_CONSTANT TmaA const tma_a,
    BSmemLayout sB_layout, CUTE_GRID_CONSTANT TmaB const tma_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    int reduce_every_k, int swz_g,
    const uint32_t* pow_key, const uint32_t* pow_bound,
    int* found_count, int* hit_rows, int* hit_cols, int max_hits) {

  constexpr int K_PIPE = size<2>(ASmemLayout{});  // pipeline depth (4 on H100)

  // ---- GMEM tensors via TMA descriptors ----
  // IMPORTANT: get_tma_tensor() builds the coordinate-space tensor that TMA
  // partition_S expects. A raw make_tensor(make_gmem_ptr(...)) makes the TMA
  // copy_unpack see scalar int8 values instead of coordinates and fails to
  // compile ("tuple_size<int8_t> incomplete"). The mma.sync path used raw
  // pointers for A because cp.async needs them — TMA does not.
  Tensor mA = tma_a.get_tma_tensor(select<0,2>(shape_MNK));  // (M,K)
  Tensor mB = tma_b.get_tma_tensor(select<1,2>(shape_MNK));  // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

  // ---- Grid swizzle for L2 locality (same bijection as mma.sync path) ----
  // Bands of swz_g M-tiles swept across all N before the next band, so A rows
  // stay resident in L2. Only CTA execution ORDER changes -> per-tile compute
  // and transcript are unaffected.
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
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (bM,bK,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (bN,bK,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (bM,bN)

  extern __shared__ char smem_[];
  using SS = SharedStorageWGMMA<ASmemLayout, BSmemLayout, K_PIPE>;
  SS& smem = *reinterpret_cast<SS*>(smem_);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);  // (bM,bK,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);  // (bN,bK,PIPE)

  // ---- TMA partitions (single-thread issue; hardware DMA does the move) ----
  auto cta_tma_a = tma_a.get_slice(Int<0>{});
  Tensor tAgA = cta_tma_a.partition_S(gA);   // (TMA,1,1,k_tiles)
  Tensor tAsA = cta_tma_a.partition_D(sA);   // (TMA,1,1,PIPE)
  auto cta_tma_b = tma_b.get_slice(Int<0>{});
  Tensor tBgB = cta_tma_b.partition_S(gB);
  Tensor tBsB = cta_tma_b.partition_D(sB);

  // One mbarrier per stage tracks BOTH A+B bytes (complete_tx::bytes).
  constexpr int kTmaBytes = int(size<0>(ASmemLayout{}) * size<1>(ASmemLayout{}))
                          + int(size<0>(BSmemLayout{}) * size<1>(BSmemLayout{}));

  // ---- GMMA partitioning ----
  // No ldmatrix / S2R copies: make_fragment_A/B produce SMEM DESCRIPTORS that
  // wgmma consumes directly. Accumulator stays in registers.
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);    // only used when DumpC
  Tensor tCsA = thr_mma.partition_A(sA);    // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);    // (MMA,MMA_N,MMA_K,PIPE)

  // Accumulator fragment: (64x256)/128 threads = 128 x s32 per thread.
  // MUST be sized from the static cta_tiler, not the dynamic problem shape.
  Tensor tCrC = thr_mma.make_fragment_C(
      thr_mma.partition_C(make_identity_tensor(
          make_shape(get<0>(cta_tiler), get<1>(cta_tiler)))));
  clear(tCrC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // smem descriptors
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);

  // ---- Transcript / fold state ----
  uint32_t transcript[pearl_fold::JACKPOT_SIZE];
  #pragma unroll
  for (int i = 0; i < pearl_fold::JACKPOT_SIZE; ++i) transcript[i] = 0;
  int gk = 0;  // global k-block counter (k-block = 32 K elements)

  constexpr int K_BLOCK_MAX = size<2>(tCrA);  // bK/32 = 4
  int k_tile_count = size<3>(tAgA);           // K/bK
  int k_tile_next  = 0;

  // ---- Prologue: fill stages 0..K_PIPE-2 via TMA ----
  CUTE_UNROLL
  for (int stage = 0; stage < K_PIPE - 1; ++stage) {
    if (threadIdx.x == 0) {
      cute::initialize_barrier(smem.mbar[stage], /*thread_count=*/1);
      cute::set_barrier_transaction_bytes(smem.mbar[stage], kTmaBytes);
      copy(tma_a.with(smem.mbar[stage]), tAgA(_,_,_,k_tile_next), tAsA(_,_,_,stage));
      copy(tma_b.with(smem.mbar[stage]), tBgB(_,_,_,k_tile_next), tBsB(_,_,_,stage));
    }
    --k_tile_count;
    if (k_tile_count > 0) ++k_tile_next;
  }
  // CRITICAL: every other thread goes straight to wait_barrier(mbar[0]) in the
  // main loop; without this sync they could wait on an UNINITIALIZED mbarrier
  // (thread 0 has not executed initialize_barrier yet) and hang forever.
  __syncthreads();

  int smem_pipe_read  = 0;
  int smem_pipe_write = K_PIPE - 1;

  // ---- Main loop: wait stage -> issue NEXT tile -> wgmma current + fold ----
  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE - 1)) {
    // Wait for TMA to deliver the read stage (mbarrier flip = all bytes in).
    cute::wait_barrier(smem.mbar[smem_pipe_read], /*phase=*/0);
    __syncthreads();  // smem writes visible to every thread before wgmma reads

    // Issue NEXT tile early into the write stage (the stage consumed in the
    // PREVIOUS iteration; the __syncthreads above guarantees all threads are
    // done with it). Early issue maximizes TMA/compute overlap.
    //
    // UNCONDITIONAL issue+decrement (mirrors gemm_device_tma_ms): the loop
    // only terminates because k_tile_count keeps going negative during the
    // final K_PIPE-1 drain iterations. Those drain issues re-load the last
    // k-tile (k_tile_next stops advancing once count <= 0) into stages that
    // are never consumed again — harmless, and keeps every wait_barrier
    // paired with a fresh initialize+complete_tx. Guarding this block with
    // (k_tile_count > 0) freezes the counter at 0 and hangs the kernel.
    if (threadIdx.x == 0) {
      cute::initialize_barrier(smem.mbar[smem_pipe_write], 1);
      cute::set_barrier_transaction_bytes(smem.mbar[smem_pipe_write], kTmaBytes);
      copy(tma_a.with(smem.mbar[smem_pipe_write]),
           tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
      copy(tma_b.with(smem.mbar[smem_pipe_write]),
           tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
    }
    --k_tile_count;
    if (k_tile_count > 0) ++k_tile_next;

    // Prefetch TMA descriptors for the next load (cheap latency hide).
    cute::prefetch_tma_descriptor(tma_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_b.get_tma_descriptor());

    // ---- wgmma over the K blocks of the current tile ----
    CUTE_UNROLL
    for (int kb = 0; kb < K_BLOCK_MAX; ++kb) {
      // Compiler fence: accumulator regs must not be reordered across wgmma.
      warpgroup_fence_operand(tCrC);
      // PTX wgmma.fence.sync.aligned: smem operands ready, registers fenced.
      warpgroup_arrive();
      // THE wgmma: 64x256x32 s8.s8.s32 straight from smem descriptors.
      gemm(mma, tCrA(_,_,kb,smem_pipe_read), tCrB(_,_,kb,smem_pipe_read), tCrC);
      // PTX wgmma.commit_group.sync.aligned: group issued wgmmas for waits.
      warpgroup_commit_batch();

      // ---- FOLD every reduce_every_k k-blocks (rank boundary snapshot) ----
      // Must drain pending wgmma groups first so the accumulator is final.
      // This is the only serialization point; it lands on tile boundaries
      // (reduce_every_k == K_BLOCK_MAX for rank 128 / bK 128), so wgmma
      // pipelining within a tile is untouched and TMA for the next tile is
      // already in flight.
      if ((++gk % reduce_every_k) == 0) {
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrC);
        uint32_t h = pearl_fold::xor_reduction(tCrC);
        int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
        transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(
            transcript[idx], h);
      }
    }

    // Drain this tile's wgmma before the stage can be overwritten later.
    warpgroup_wait<0>();
    __syncthreads();

    // Advance ring pointers: the stage we just consumed becomes the next
    // write target; read moves to the following stage.
    smem_pipe_write = smem_pipe_read;
    smem_pipe_read  = (smem_pipe_read == K_PIPE - 1) ? 0 : smem_pipe_read + 1;
  }

  // Optional C dump for bit-exactness validation (tests only; prod = false).
  if constexpr (DumpC) copy(tCrC, tCgC);

  // ---- POW CHECK (identical semantics to the mma.sync path) ----
  auto msg = make_tensor<uint32_t>(Int<16>{});
  #pragma unroll
  for (int i = 0; i < 16; ++i) msg(i) = transcript[i];
  auto cv = make_tensor<uint32_t>(Int<8>{});
  #pragma unroll
  for (int i = 0; i < 8; ++i) cv(i) = pow_key[i];
  blake3::compress_msg_block_u32(msg, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);

  bool found = true;
  if constexpr (BigEndian) {
    // LuckyPool dialect: compare big-endian (byte-swap each u32 word).
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t h = __byte_perm(cv(i), 0, 0x0123);
      uint32_t b = __byte_perm(pow_bound[i], 0, 0x0123);
      if (h > b) { found = false; break; }
      if (h < b) break;
    }
  } else {
    // HeroMiners dialect: little-endian word order, compare MSW first.
    for (int i = 7; i >= 0; --i) {
      if (cv(i) > pow_bound[i]) { found = false; break; }
      if (cv(i) < pow_bound[i]) break;
    }
  }

  // ---- HIT REPORTING ----
  // Coordinates come from the GMMA CLayout (DIFFERENT from mma.sync!).
  // The pool-side pattern declaration must match — see header comment.
  if (found) {
    int slot = atomicAdd(found_count, 1);
    if (slot < max_hits) {
      auto cD = make_identity_tensor(make_shape(get<0>(cta_tiler), get<1>(cta_tiler)));
      auto tCcD = thr_mma.partition_C(cD);
      int cnt = size(tCcD);   // 128 for 64x256 per warpgroup, 2 WGs per CTA
      int row0 = bx * (int)get<0>(cta_tiler);
      int col0 = by * (int)get<1>(cta_tiler);
      for (int i = 0; i < cnt; ++i) {
        hit_rows[slot * 128 + i] = row0 + (int)get<0>(tCcD(i));
        hit_cols[slot * 128 + i] = col0 + (int)get<1>(tCcD(i));
      }
    }
  }
}
