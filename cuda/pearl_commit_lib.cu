// ariaminer v0.4.0 — brique 2 : FFI commitment GPU (tensor_hash officiel).
// Calcule le blake3-tree keyed d'un buffer sur GPU via le `tensor_hash` officiel
// (csrc/tensor_hash). Validé bit-exact en Rust vs pearl_blake3::blake3_digest
// (= blake3::keyed_hash). Réplique la dérivation num_blocks de run_tensor_hash.
// Build : nvcc -arch=sm_120a -O3 -std=c++17 -I cuda/shim -I<csrc> -I<cutlass>/include
//         --expt-relaxed-constexpr -Xcompiler -fPIC -c
#include <cstdint>
#include <cstdio>
#include "tensor_hash/tensor_hash_host.hpp"

#define CKR(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"commit CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return -1;}}while(0)

extern "C" {

// out[32] = blake3 keyed tree de data[len] avec key[32]. len doit être > 131072.
// Doit == pearl_blake3::blake3_digest(data, Some(key)).
int pearl_gpu_tensor_hash(const uint8_t* data, uint32_t len,
                          const uint8_t key[32], uint8_t out[32]) {
  cudaDeviceProp prop; CKR(cudaGetDeviceProperties(&prop, 0));

  const uint32_t threads_per_block = 256, num_stages = 2, leaves_per_mt_block = 256;
  const uint32_t chunk = 1024;
  uint32_t num_chunks = (len + chunk - 1) / chunk;
  uint32_t num_blocks = (num_chunks + threads_per_block - 1) / threads_per_block;

  uint8_t *d_data, *d_out, *d_roots;
  CKR(cudaMalloc(&d_data, len));
  CKR(cudaMalloc(&d_out, 32));
  CKR(cudaMalloc(&d_roots, (size_t)(num_blocks + 64) * 32));   // scratchpad + marge
  CKR(cudaMemcpy(d_data, data, len, cudaMemcpyHostToDevice));

  tensor_hash(d_data, len, d_out, key, num_blocks,
              threads_per_block, num_stages, leaves_per_mt_block,
              d_roots, prop, 0);
  CKR(cudaDeviceSynchronize());
  CKR(cudaMemcpy(out, d_out, 32, cudaMemcpyDeviceToHost));

  cudaFree(d_data); cudaFree(d_out); cudaFree(d_roots);
  return 0;
}

}  // extern "C"
