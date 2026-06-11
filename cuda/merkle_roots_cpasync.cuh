// Jumeau PORTABLE (SM80+) du MerkleTreeRootsKernel officiel (TMA/SM90-only).
// Même découpage chunks→threads, mêmes layouts smem swizzlés, même cœur Blake3,
// même réduction merkle_tree_utils → roots byte-identiques au kernel officiel.
// Différence unique : le staging gmem→smem se fait en cp.async coopératif par les
// 256 threads consommateurs eux-mêmes (pas de warpgroup producteur, pas de
// descripteur TMA — c'est sa création hôte qui crashe (801) sur Ampere/Ada).
#pragma once

#include "blake3/blake3.cuh"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "tensor_hash/merkle_tree_utils.hpp"
#include "tensor_hash/tensor_hash_constants.cuh"

#include <cutlass/arch/memory_sm80.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/layout.hpp>
#include <cutlass/gemm/collective/builders/sm90_common.inl>  // GMMA layout atoms (pur layout, OK SM80)

namespace pearl {

using namespace cute;

template <int kNumConsumerThreads, int kNumStages, int kThreadLoadSize>
class MerkleTreeRootsKernelCpAsync {
 public:
  using Element = uint8_t;

  static constexpr int kNumThreads = kNumConsumerThreads;  // pas de producteur dédié
  static constexpr uint32_t MaxThreadsPerBlock = kNumThreads;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Mode single-pipeline uniquement (l'appelant utilise 256 consommateurs).
  static_assert(kNumConsumerThreads <= 256, "jumeau cp.async : single pipeline only");
  static_assert(kNumConsumerThreads % 128 == 0, "multiple de warpgroup");

  static constexpr int kLoadSize = 16;
  static constexpr int kChunkSize = 1024;
  static constexpr int kWordSize = 4;
  static constexpr int kPipelineStages = kNumStages;

  static_assert(kThreadLoadSize == 64 || kThreadLoadSize == 128 ||
                kThreadLoadSize == 256 || kThreadLoadSize == 512,
                "kThreadLoadSize must be 64, 128, 256, or 512");
  static_assert(kChunkSize % kThreadLoadSize == 0);

  static constexpr int kNumBlocksPerChunk = kChunkSize / blake3::MSG_BLOCK_SIZE;
  static constexpr int kNumWordsPerBlock = blake3::MSG_BLOCK_SIZE / sizeof(uint32_t);
  static constexpr int kNumWordsPerLoad = kThreadLoadSize / sizeof(uint32_t);
  static constexpr int kNumBlocksPerLoad = kThreadLoadSize / blake3::MSG_BLOCK_SIZE;
  static constexpr int kNumLoads = kChunkSize / kThreadLoadSize;

  // cp.async : vecteurs de 16o ; une rangée (= 1 tranche de chunk) = kVecsPerRow vecteurs.
  static constexpr int kVecsPerRow = kThreadLoadSize / 16;
  static constexpr int kTotalVecs = kNumConsumerThreads * kVecsPerRow;
  static constexpr int kOpsPerThread = kTotalVecs / kNumThreads;  // = kVecsPerRow

  // Layouts smem IDENTIQUES à l'officiel (le swizzle préserve les blocs de 16o,
  // donc les uint4 du hash et les dst cp.async restent alignés 16o).
  using SmemLayoutAtomA =
      std::conditional_t<kThreadLoadSize == 64,
                         GMMA::Layout_K_SW64_Atom<uint32_t>,
                         GMMA::Layout_K_SW128_Atom<uint32_t>>;
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(Int<kNumConsumerThreads>{}, Int<kNumWordsPerLoad>{},
                 Int<kPipelineStages>{})));

  using SmemLayoutAtomLeaves = GMMA::Layout_K_SW128_Atom<uint32_t>;
  using SmemLayoutLeaves = decltype(tile_to_shape(
      SmemLayoutAtomLeaves{},
      Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>, Int<kNumConsumerThreads>>{}));

  static constexpr size_t AlignmentLeaves =
      cutlass::detail::alignment_for_swizzle(SmemLayoutLeaves{});
  static constexpr size_t AlignmentA =
      cutlass::detail::alignment_for_swizzle(SmemLayoutA{});
  static constexpr size_t Alignment = cute::max(AlignmentLeaves, AlignmentA);

  using RmemLayoutChainingValue = Layout<Shape<Int<blake3::CHAINING_VALUE_SIZE_U32>>>;
  using RmemLayoutBlock = Layout<Shape<Int<kNumWordsPerBlock>>>;

  struct SharedStorage : cute::aligned_struct<Alignment> {
    cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutLeaves>, AlignmentLeaves>
        smem_leaves;
    cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutA>, AlignmentA> smem_a;
  };
  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments {
    const Element* ptr_data;
    const u32 data_len;
    Element* ptr_roots;
  };
  struct alignas(128) Params {
    const Element* ptr_data;
    u32 data_len;
    Element* ptr_roots;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return Params{args.ptr_data, args.data_len, args.ptr_roots};
  }

  static dim3 get_grid_shape(Params const& params) {
    const size_t num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
    return dim3((num_chunks + kNumConsumerThreads - 1) / kNumConsumerThreads);
  }
  static dim3 get_block_shape() { return dim3(kNumThreads); }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    const int tid = threadIdx.x;

    Tensor sA = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{}));
    Tensor sLeaves = as_position_independent_swizzle_tensor(make_tensor(
        make_smem_ptr(shared_storage.smem_leaves.data()), SmemLayoutLeaves{}));

    const size_t num_chunks =
        (params.data_len + blake3::CHUNK_SIZE - 1) / blake3::CHUNK_SIZE;
    const size_t num_grid_blocks =
        (num_chunks + kNumConsumerThreads - 1) / kNumConsumerThreads;
    const size_t bid = blockIdx.x;

    Tensor mRoots = make_tensor(
        reinterpret_cast<uint32_t*>(params.ptr_roots),
        make_layout(
            make_shape(Int<blake3::CHAINING_VALUE_SIZE_U32>{}, num_grid_blocks),
            make_stride(Int<1>{}, Int<blake3::CHAINING_VALUE_SIZE_U32>{})));

    // ---- hash des chunks (1 chunk de 1024o par thread, kNumLoads tranches) ----
    Tensor rChainingValue = make_tensor<uint32_t>(RmemLayoutChainingValue{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      rChainingValue(i) = c_key[i];
    }

    const u32 remainder = params.data_len % blake3::CHUNK_SIZE;
    const u32 last_chunk_size = (remainder == 0) ? blake3::CHUNK_SIZE : remainder;
    const u32 global_chunk_idx = bid * kNumConsumerThreads + tid;
    const bool is_last_chunk = (global_chunk_idx == num_chunks - 1) &&
                               (last_chunk_size < blake3::CHUNK_SIZE);

    // Prologue : remplir les étages du ring.
    constexpr int kPrologue =
        (kPipelineStages < kNumLoads) ? kPipelineStages : kNumLoads;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kPrologue; ++s) {
      issue_load(params, sA, tid, bid, num_chunks, s, /*stage=*/s % kPipelineStages);
      cutlass::arch::cp_async_fence();
    }

    CUTLASS_PRAGMA_NO_UNROLL
    for (int load_idx = 0; load_idx < kNumLoads; ++load_idx) {
      const int stage = load_idx % kPipelineStages;
      // Attendre la complétion du groupe du load courant (les groupes finissent
      // dans l'ordre ; au dernier load il ne reste que lui en vol → wait<0>).
      if (load_idx + kPipelineStages <= kNumLoads - 1) {
        cutlass::arch::cp_async_wait<kPipelineStages - 1>();
      } else {
        cutlass::arch::cp_async_wait<0>();
      }
      __syncthreads();

      if (is_last_chunk) {
        zero_pad_partial_chunk_load(sA, tid, stage, load_idx, last_chunk_size);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int block_in_load = 0; block_in_load < kNumBlocksPerLoad; ++block_in_load) {
        const int block_idx = load_idx * kNumBlocksPerLoad + block_in_load;
        compress_block(sA, rChainingValue, tid, stage, block_in_load, block_idx);
      }

      // Tous les threads ont consommé l'étage avant qu'un cp.async le réécrive.
      __syncthreads();
      const int next = load_idx + kPipelineStages;
      if (next < kNumLoads) {
        issue_load(params, sA, tid, bid, num_chunks, next, next % kPipelineStages);
        cutlass::arch::cp_async_fence();
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) {
      sLeaves(i, tid) = rChainingValue(i);
    }

    __syncthreads();

    // ---- réduction Merkle (identique à l'officiel) ----
    const bool is_last_block = (bid == num_grid_blocks - 1);
    const u32 num_leaves = [is_last_block, num_chunks, &params]() -> u32 {
      if (!is_last_block) return static_cast<u32>(kNumConsumerThreads);
      const u32 chunks_in_this_block = num_chunks % kNumConsumerThreads;
      const u32 actual_chunks_in_block =
          (chunks_in_this_block == 0) ? static_cast<u32>(kNumConsumerThreads)
                                      : chunks_in_this_block;
      const u32 remainder_bytes = params.data_len % blake3::CHUNK_SIZE;
      const bool last_chunk_too_small =
          (remainder_bytes > 0) && (remainder_bytes < blake3::MSG_BLOCK_SIZE);
      return last_chunk_too_small
                 ? (actual_chunks_in_block > 0 ? actual_chunks_in_block - 1 : 0)
                 : actual_chunks_in_block;
    }();

    if (!is_last_block) {
      merkle_tree_utils::compute_perfect_mt<false>(sLeaves, kNumConsumerThreads);
    } else if ((num_leaves & (num_leaves - 1)) == 0) {
      merkle_tree_utils::compute_perfect_mt<false>(sLeaves, num_leaves);
    } else {
      merkle_tree_utils::compute_blake_mt<false>(sLeaves, num_leaves);
    }

    if (tid < blake3::CHAINING_VALUE_SIZE_U32) {
      mRoots(tid, blockIdx.x) = sLeaves(tid, 0);
    }
  }

 private:
  // Charge la tranche load_idx (kThreadLoadSize octets de chacun des
  // kNumConsumerThreads chunks du bloc) dans l'étage `stage`, en cp.async 16o.
  // Threads d'un warp → vecteurs consécutifs d'une même rangée (coalescé par
  // segments de kThreadLoadSize octets). Rangées hors num_chunks → zfill
  // (équivalent du oobFill=0 du descripteur TMA).
  template <class SmemTensorA>
  CUTLASS_DEVICE void issue_load(Params const& params, SmemTensorA& sA, int tid,
                                 size_t bid, size_t num_chunks, int load_idx,
                                 int stage) {
    const uint32_t* gbase = reinterpret_cast<const uint32_t*>(params.ptr_data);
    constexpr int kWordsPerChunk = kChunkSize / kWordSize;
    constexpr int kLog2VecsPerRow = __builtin_ctz(kVecsPerRow);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kOpsPerThread; ++j) {
      const int o = j * kNumThreads + tid;
      const int row = o >> kLog2VecsPerRow;
      const int vec = o & (kVecsPerRow - 1);
      const size_t grow = bid * kNumConsumerThreads + row;
      const uint32_t* src =
          gbase + grow * kWordsPerChunk + size_t(load_idx) * kNumWordsPerLoad + vec * 4;
      void* dst = &sA(row, vec * 4, stage);
      cutlass::arch::cp_async_zfill<16, cutlass::arch::CacheOperation::Always>(
          dst, src, grow < num_chunks);
    }
  }

  // Copies conformes de l'officiel (zero-pad + compress) — mêmes octets, mêmes flags.
  template <class SmemTensorA>
  CUTLASS_DEVICE void zero_pad_partial_chunk_load(SmemTensorA& sA, int consumer_tid,
                                                  int stage, int load_idx,
                                                  u32 last_chunk_len) {
    const u32 load_start_byte = load_idx * kThreadLoadSize;
    if (load_start_byte >= last_chunk_len) {
      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < kNumWordsPerLoad; ++w) sA(consumer_tid, w, stage) = 0;
      return;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int w = 0; w < kNumWordsPerLoad; ++w) {
      const u32 word_start_byte = load_start_byte + w * sizeof(uint32_t);
      const u32 word_end_byte = word_start_byte + sizeof(uint32_t);
      if (word_start_byte >= last_chunk_len) {
        sA(consumer_tid, w, stage) = 0;
      } else if (word_end_byte > last_chunk_len) {
        const u32 valid_bytes = last_chunk_len - word_start_byte;
        const u32 mask = (1u << (valid_bytes * 8)) - 1;
        uint32_t val = sA(consumer_tid, w, stage);
        sA(consumer_tid, w, stage) = val & mask;
      }
    }
  }

  template <class SmemTensorA, class RmemTensorChainingValue>
  CUTLASS_DEVICE void compress_block(SmemTensorA const& sA,
                                     RmemTensorChainingValue& rChainingValue,
                                     int consumer_tid, int stage, int block_in_load,
                                     int block_idx) {
    Tensor rBlock = make_tensor<uint32_t>(RmemLayoutBlock{});
    int word_offset = block_in_load * kNumWordsPerBlock;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kNumWordsPerBlock / 4; ++i) {
      uint4 tmp = *reinterpret_cast<const uint4*>(
          &sA(consumer_tid, word_offset + i * 4, stage));
      rBlock(i * 4 + 0) = tmp.x;
      rBlock(i * 4 + 1) = tmp.y;
      rBlock(i * 4 + 2) = tmp.z;
      rBlock(i * 4 + 3) = tmp.w;
    }
    blake3::CompressParams cparams{
        .counter = blockIdx.x * kNumConsumerThreads + consumer_tid,
        .block_len = blake3::MSG_BLOCK_SIZE,
        .flags = blake3::KEYED_HASH};
    if (block_idx == 0) cparams.flags |= blake3::CHUNK_START;
    if (block_idx == kNumBlocksPerChunk - 1) cparams.flags |= blake3::CHUNK_END;
    blake3::compress_msg_block_u32(rBlock, rChainingValue, cparams);
  }
};

}  // namespace pearl
