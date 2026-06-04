// ariaminer v0.4.0 — brique 4 : noise structuré Pearl sur GPU.
// Port 1:1 du spec Rust (zk_pow::circuit::pearl_noise) — validé bit-exact vs
// compute_noise_for_indices. Le PRF = get_random_hash = blake3 keyed d'un bloc 64o
// (= compress_msg_block_u32 SINGLE_BLOCK_KEYED, déjà validé au pow-check).
//
// Constantes (LUES dans pearl_noise.rs) : NOISE_RANGE=128, IDXS_PER_COL=2,
//   UNIFORM_NOISE_RANGE=64, ZERO_POINT_TRANSLATION=32, RANGE_MASK=63, rank=128.
// e_al[i][c] = (hash_byte & 63) - 32 ;  perm[j]=(first,second) ;
// noise[i][j] = e_al[i][first[j]] - e_al[i][second[j]] (wrapping i8).
// Build : nvcc -arch=sm_120a -O3 -std=c++17 -I<csrc> -I<cutlass>/include --expt-relaxed-constexpr
#include <cstdint>
#include <cstdio>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"
using namespace cute;

#define RANK 128
#define RANGE_MASK 63
#define ZERO_POINT 32

// get_random_hash(index, seed_label[8 u32], key[8 u32], prepend_index) -> out[8 u32]
__device__ __forceinline__ void get_random_hash(
    uint32_t index, const uint32_t* seed, const uint32_t* key,
    int prepend_index, uint32_t out[8]) {
  auto msg = make_tensor<uint32_t>(Int<16>{});
  for (int i = 0; i < 16; ++i) msg(i) = 0;
  msg(prepend_index) = 1u + index;          // (1+index) i32 LE au byte prepend_index*4
  for (int i = 0; i < 8; ++i) msg(8 + i) = seed[i];   // seed aux bytes 32..64
  auto cv = make_tensor<uint32_t>(Int<8>{});
  for (int i = 0; i < 8; ++i) cv(i) = key[i];
  blake3::compress_msg_block_u32(msg, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
  for (int i = 0; i < 8; ++i) out[i] = cv(i);
}

__device__ __forceinline__ uint32_t mul_hi_u32(uint32_t a, uint32_t b) {
  return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}
__device__ __forceinline__ uint8_t byte_of(const uint32_t* h, int k) {
  return (uint8_t)((h[k >> 2] >> (8 * (k & 3))) & 0xff);
}

// Permutation : 1 thread = 1 colonne j ; first[j], second[j] (0..127).
__global__ void perm_kernel(const uint32_t* seed, const uint32_t* key, int k,
                            uint8_t* first, uint8_t* second) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= k) return;
  uint32_t h[8];
  get_random_hash(j / 8, seed, key, 1, h);   // chunk i=j/8, prepend_index=1
  uint32_t ru = h[j & 7];                     // slot j%8 = u32 LE
  uint32_t f = ru & (RANK - 1);
  uint32_t s = f ^ (1u + mul_hi_u32((uint32_t)(RANK - 1), ru));
  first[j] = (uint8_t)f; second[j] = (uint8_t)s;
}

// Noise : 1 thread = 1 ligne i ; calcule e_al[128] (4 hashes) puis applique la perm.
__global__ void noise_kernel(const uint32_t* seed, const uint32_t* key,
                             int m, int k, const uint8_t* first,
                             const uint8_t* second, int8_t* noise) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  int8_t e_al[RANK];
  for (int b = 0; b < 4; ++b) {              // 4 blocs de 32 octets = 128
    uint32_t h[8];
    get_random_hash((uint32_t)(i * 4 + b), seed, key, 0, h);
    for (int kk = 0; kk < 32; ++kk)
      e_al[b * 32 + kk] = (int8_t)((byte_of(h, kk) & RANGE_MASK) - ZERO_POINT);
  }
  int8_t* row = noise + (size_t)i * k;
  for (int j = 0; j < k; ++j) {
    int p = e_al[first[j]];
    int n = e_al[second[j]];
    row[j] = (int8_t)(p - n);
  }
}

#define CKR(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"noise CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return -1;}}while(0)

static void bytes_to_u32(const uint8_t* b, uint32_t* w) {  // 32 bytes -> 8 u32 LE
  for (int i = 0; i < 8; ++i)
    w[i] = (uint32_t)b[i*4] | ((uint32_t)b[i*4+1]<<8) | ((uint32_t)b[i*4+2]<<16) | ((uint32_t)b[i*4+3]<<24);
}

extern "C" {
// noise[m*k] i8 = noise structuré pour les lignes 0..m, clé=key (a_noise_seed),
// seed_label = "A_tensor"/"B_tensor" paddé. Doit == compute_noise_for_indices.{a|b}.
int pearl_gpu_noise(const uint8_t seed_label[32], const uint8_t key[32],
                    int m, int k, int8_t* noise_out) {
  uint32_t sw[8], kw[8]; bytes_to_u32(seed_label, sw); bytes_to_u32(key, kw);
  uint32_t *d_seed, *d_key; uint8_t *d_first, *d_second; int8_t* d_noise;
  CKR(cudaMalloc(&d_seed, 32)); CKR(cudaMalloc(&d_key, 32));
  CKR(cudaMalloc(&d_first, k)); CKR(cudaMalloc(&d_second, k));
  CKR(cudaMalloc(&d_noise, (size_t)m * k));
  CKR(cudaMemcpy(d_seed, sw, 32, cudaMemcpyHostToDevice));
  CKR(cudaMemcpy(d_key, kw, 32, cudaMemcpyHostToDevice));

  perm_kernel<<<(k + 255) / 256, 256>>>(d_seed, d_key, k, d_first, d_second);
  CKR(cudaGetLastError());
  noise_kernel<<<(m + 127) / 128, 128>>>(d_seed, d_key, m, k, d_first, d_second, d_noise);
  CKR(cudaDeviceSynchronize()); CKR(cudaGetLastError());
  CKR(cudaMemcpy(noise_out, d_noise, (size_t)m * k, cudaMemcpyDeviceToHost));

  cudaFree(d_seed); cudaFree(d_key); cudaFree(d_first); cudaFree(d_second); cudaFree(d_noise);
  return 0;
}
}  // extern "C"
