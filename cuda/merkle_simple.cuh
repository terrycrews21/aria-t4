#pragma once
// Simple, portable keyed-blake3 Merkle root — one thread per leaf, then one kernel
// per parent level. This is the SAME construction the proof context uses
// (pearl_gpu_proof_lib.cu), which is why their roots agree bit-for-bit.
//
// Why this exists: the cp.async/TMA staged commit (`MerkleTreeRootsKernelCpAsync`)
// returns a DIFFERENT root on some Ampere parts (measured: NVIDIA A10 vs RTX 3080 Ti,
// same sm_86 cubin, same launch geometry, no launch error, deterministic on each
// machine). A wrong commitment root poisons stir -> noise seeds -> the permutation ->
// the noised matrices, so the GEMM folds data the verifier cannot reproduce and the
// pool rejects every share with "Jackpot condition not satisfied" (code 23).
//
// This path has no cross-thread staging and no pipeline, so it cannot depend on
// scheduling or on SM count. It is the commitment of record for the grind.
#include "blake3/blake3.cuh"

namespace pearl_simple {

using namespace cute;

// Blake3Hasher::chunk_cv(data, chunk_idx): CV init = key, 16 chained blocks,
// counter = chunk_idx, CHUNK_START on block 0, CHUNK_END on block 15, KEYED_HASH.
static __device__ __forceinline__ void leaf_chunk_cv_s(const uint32_t* cw, uint64_t chunk_idx,
                                                       const uint32_t key[8], uint32_t out[8]) {
  auto cv = make_tensor<uint32_t>(Int<8>{});
  for (int i = 0; i < 8; ++i) cv(i) = key[i];
  auto blk = make_tensor<uint32_t>(Int<16>{});
  for (int b = 0; b < 16; ++b) {
    for (int w = 0; w < 16; ++w) blk(w) = cw[b * 16 + w];
    blake3::CompressParams p{};
    p.counter = chunk_idx;
    p.block_len = blake3::MSG_BLOCK_SIZE;
    p.flags = blake3::KEYED_HASH;
    if (b == 0) p.flags |= blake3::CHUNK_START;
    if (b == 15) p.flags |= blake3::CHUNK_END;
    blake3::compress_msg_block_u32(blk, cv, p);
  }
  for (int i = 0; i < 8; ++i) out[i] = cv(i);
}

static __device__ __forceinline__ void merge_cv_s(const uint32_t l[8], const uint32_t r[8],
                                                  const uint32_t key[8], bool is_root, uint32_t out[8]) {
  auto cv = make_tensor<uint32_t>(Int<8>{});
  for (int i = 0; i < 8; ++i) cv(i) = key[i];
  auto blk = make_tensor<uint32_t>(Int<16>{});
  for (int i = 0; i < 8; ++i) { blk(i) = l[i]; blk(8 + i) = r[i]; }
  blake3::CompressParams p{};
  p.counter = 0;
  p.block_len = blake3::MSG_BLOCK_SIZE;
  p.flags = blake3::KEYED_HASH | blake3::PARENT | (is_root ? blake3::ROOT : 0u);
  blake3::compress_msg_block_u32(blk, cv, p);
  for (int i = 0; i < 8; ++i) out[i] = cv(i);
}

__global__ void msimple_hash_leaves(const uint8_t* data, const uint32_t* key,
                                    uint32_t* out_leaves, uint64_t num_leaves) {
  uint64_t c = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_leaves) return;
  const uint32_t* cw = (const uint32_t*)(data + c * 1024ull);
  uint32_t cv[8];
  leaf_chunk_cv_s(cw, c, key, cv);
  for (int i = 0; i < 8; ++i) out_leaves[c * 8 + i] = cv[i];
}

// Pair (2p, 2p+1) -> merge; a lone odd node is carried up (= MerkleTree::combine_layer).
__global__ void msimple_parent_level(const uint32_t* key, uint32_t* out, const uint32_t* in,
                                     uint64_t in_count, bool is_root) {
  uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t n_out = (in_count + 1) / 2;
  if (p >= n_out) return;
  uint32_t l[8], o[8];
  for (int i = 0; i < 8; ++i) l[i] = in[(2 * p) * 8 + i];
  if (2 * p + 1 < in_count) {
    uint32_t r[8];
    for (int i = 0; i < 8; ++i) r[i] = in[(2 * p + 1) * 8 + i];
    merge_cv_s(l, r, key, is_root, o);
    for (int i = 0; i < 8; ++i) out[p * 8 + i] = o[i];
  } else {
    for (int i = 0; i < 8; ++i) out[p * 8 + i] = l[i];
  }
}

/// Number of u32 words the `layers` scratch must hold for `data_size` bytes.
static inline size_t msimple_layers_words(size_t data_size) {
  size_t leaves = data_size / 1024;
  size_t total = 0, cur = leaves;
  while (cur > 1) { total += cur; cur = (cur + 1) / 2; }
  total += 1;
  return total * 8;
}

/// Keyed Merkle root of `data` into `out` (32 bytes, device->device).
static inline void commit_simple(const uint8_t* data, size_t data_size, uint8_t* out,
                                 const uint32_t* key, uint32_t* layers, cudaStream_t stream) {
  const uint64_t num_leaves = (uint64_t)(data_size / 1024);
  const int tpb = 256;
  msimple_hash_leaves<<<(int)((num_leaves + tpb - 1) / tpb), tpb, 0, stream>>>(
      data, key, layers, num_leaves);
  int64_t cur_off = 0, cur_s = (int64_t)num_leaves;
  while (cur_s > 1) {
    const int64_t next_off = cur_off + cur_s;
    const int64_t next_s = (cur_s + 1) / 2;
    const bool is_root = (cur_s == 2);
    msimple_parent_level<<<(int)((next_s + tpb - 1) / tpb), tpb, 0, stream>>>(
        key, layers + next_off * 8, layers + cur_off * 8, (uint64_t)cur_s, is_root);
    cur_off = next_off;
    cur_s = next_s;
  }
  cudaMemcpyAsync(out, layers + cur_off * 8, 32, cudaMemcpyDeviceToDevice, stream);
}

}  // namespace pearl_simple
