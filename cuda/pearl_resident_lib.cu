// ariaminer v0.4.0 — brique 5 : pipeline résident GPU.
// Pièces nouvelles : (3) signal-gen reproductible, (2b) seed-stir unkeyed.
// Assemblage : gen A,B (int7) -> commit (tensor_hash) -> stir seeds -> [noise+grind].
// Ce TU expose d'abord `pearl_gpu_commit_seeds` (gen+commit+stir) pour valider
// bit-exact vs compute_commitment_hash CPU AVANT le grind complet.
// Build : nvcc -arch=sm_120a -O3 -std=c++17 -I cuda/shim -I<csrc> -I<cutlass>/include
#include <cstdint>
#include <cstdio>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"
#include "tensor_hash/tensor_hash_host.hpp"
#include "pearl_gpu_kernel.cuh"   // gemm_device (grind GEMM IMMA + jackpot + pow)
using namespace cute;

// --- helpers noise (miroir de pearl_noise_lib.cu, inline) ---
__device__ __forceinline__ void rh_(uint32_t index, const uint32_t* seed,
                                     const uint32_t* key, int prep, uint32_t out[8]) {
  auto msg = make_tensor<uint32_t>(Int<16>{});
  for (int i = 0; i < 16; ++i) msg(i) = 0;
  msg(prep) = 1u + index;
  for (int i = 0; i < 8; ++i) msg(8 + i) = seed[i];
  auto cv = make_tensor<uint32_t>(Int<8>{});
  for (int i = 0; i < 8; ++i) cv(i) = key[i];
  blake3::compress_msg_block_u32(msg, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
  for (int i = 0; i < 8; ++i) out[i] = cv(i);
}
__device__ __forceinline__ uint32_t mulhi_(uint32_t a, uint32_t b) {
  return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}
__device__ __forceinline__ uint8_t bo_(const uint32_t* h, int k) {
  return (uint8_t)((h[k >> 2] >> (8 * (k & 3))) & 0xff);
}
// perm (1 thr = 1 col)
__global__ void perm_k_(const uint32_t* seed, const uint32_t* key, int k,
                        uint8_t* first, uint8_t* second) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= k) return;
  uint32_t h[8]; rh_((uint32_t)(j / 8), seed, key, 1, h);
  uint32_t ru = h[j & 7]; uint32_t f = ru & 127u;
  first[j] = (uint8_t)f; second[j] = (uint8_t)(f ^ (1u + mulhi_(127u, ru)));
}
// noise-add FUSÉ : mat (signal in-place) += noise. mat devient a_eff/b_eff.
// 1 BLOC = 1 ligne ; les threads coopèrent sur les k colonnes (e_al en shared).
// 4 threads calculent les 4 blake3 d'e_al, puis tous parcourent k en grid-stride.
__global__ void noise_add_k_(const uint32_t* seed, const uint32_t* key, int m, int k,
                             const uint8_t* first, const uint8_t* second, int8_t* mat) {
  int i = blockIdx.x; if (i >= m) return;
  __shared__ int8_t e_al[128];
  if (threadIdx.x < 4) {
    uint32_t h[8]; rh_((uint32_t)(i * 4 + threadIdx.x), seed, key, 0, h);
    for (int kk = 0; kk < 32; ++kk) e_al[threadIdx.x * 32 + kk] = (int8_t)((bo_(h, kk) & 63) - 32);
  }
  __syncthreads();
  // Vectorisé : 4 colonnes/itér via uint32 (lectures/écritures 4 octets = plein débit mémoire).
  uint32_t* row4 = (uint32_t*)(mat + (size_t)i * k);
  const uint32_t* f4 = (const uint32_t*)first;
  const uint32_t* s4 = (const uint32_t*)second;
  int k4 = k >> 2;
  for (int q = threadIdx.x; q < k4; q += blockDim.x) {
    uint32_t v = row4[q], fa = f4[q], sa = s4[q];
    int8_t r0 = (int8_t)((int)((int8_t)(v      & 0xff)) + (e_al[ fa      & 0xff] - e_al[ sa      & 0xff]));
    int8_t r1 = (int8_t)((int)((int8_t)((v>> 8)& 0xff)) + (e_al[(fa>> 8) & 0xff] - e_al[(sa>> 8) & 0xff]));
    int8_t r2 = (int8_t)((int)((int8_t)((v>>16)& 0xff)) + (e_al[(fa>>16) & 0xff] - e_al[(sa>>16) & 0xff]));
    int8_t r3 = (int8_t)((int)((int8_t)((v>>24)& 0xff)) + (e_al[(fa>>24) & 0xff] - e_al[(sa>>24) & 0xff]));
    row4[q] = (uint32_t)(uint8_t)r0 | ((uint32_t)(uint8_t)r1<<8) | ((uint32_t)(uint8_t)r2<<16) | ((uint32_t)(uint8_t)r3<<24);
  }
}

// ---- (3) signal int7 reproductible : a_sig[i][j] = int7(seed, sel, i, j) ----
// MÊME formule côté Rust (sur win : regen des strips gagnants pour make_proof).
__host__ __device__ __forceinline__ int8_t int7_at(uint64_t seed, uint32_t sel,
                                                    uint32_t i, uint32_t j) {
  uint64_t x = seed ^ ((uint64_t)sel << 62) ^ ((uint64_t)i << 32) ^ (uint64_t)j;
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x ^= (x >> 31);
  return (int8_t)((int)(x & 0x7F) - 64);   // [-64,63]
}

// Grille 2D (blockIdx.y=ligne, zéro division) + écritures 4 octets (uint32) = plein débit.
__global__ void gen_signal_kernel(int8_t* mat, uint64_t seed, uint32_t sel,
                                   uint32_t rows, uint32_t k) {
  uint32_t i = blockIdx.y;
  if (i >= rows) return;
  uint32_t* row4 = (uint32_t*)(mat + (size_t)i * k);
  uint32_t k4 = k >> 2;
  for (uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < k4; q += gridDim.x * blockDim.x) {
    uint32_t j = q << 2;
    uint8_t b0 = (uint8_t)int7_at(seed, sel, i, j);
    uint8_t b1 = (uint8_t)int7_at(seed, sel, i, j + 1);
    uint8_t b2 = (uint8_t)int7_at(seed, sel, i, j + 2);
    uint8_t b3 = (uint8_t)int7_at(seed, sel, i, j + 3);
    row4[q] = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
  }
}

// ---- (2b) blake3 UNKEYED d'un bloc 64o (= blake3_digest(msg, None)) ----
__device__ __forceinline__ void blake3_unkeyed_64(const uint32_t msg16[16], uint32_t out8[8]) {
  blake3::CompressParams p{};
  p.counter = 0; p.block_len = blake3::MSG_BLOCK_SIZE;
  p.flags = blake3::CHUNK_START | blake3::CHUNK_END | blake3::ROOT;   // pas KEYED_HASH
  auto m = make_tensor<uint32_t>(Int<16>{});
  for (int i = 0; i < 16; ++i) m(i) = msg16[i];
  auto cv = make_tensor<uint32_t>(Int<8>{});
  for (int i = 0; i < 8; ++i) cv(i) = blake3::IV[i];   // CV init = IV (unkeyed)
  blake3::compress_msg_block_u32(m, cv, p);
  for (int i = 0; i < 8; ++i) out8[i] = cv(i);
}

// seeds = stir(job_key, hash_a, hash_b) : b_seed=blake3(job_key‖hash_b),
//          a_seed=blake3(b_seed‖hash_a). (hash_* en u32[8] LE).
__global__ void stir_kernel(const uint32_t* hash_a, const uint32_t* hash_b,
                            const uint32_t* job_key, uint32_t* a_seed, uint32_t* b_seed) {
  if (threadIdx.x || blockIdx.x) return;
  uint32_t msg[16];
  for (int i = 0; i < 8; ++i) { msg[i] = job_key[i]; msg[8 + i] = hash_b[i]; }
  uint32_t bs[8]; blake3_unkeyed_64(msg, bs);
  for (int i = 0; i < 8; ++i) { msg[i] = bs[i]; msg[8 + i] = hash_a[i]; }
  uint32_t as_[8]; blake3_unkeyed_64(msg, as_);
  for (int i = 0; i < 8; ++i) { b_seed[i] = bs[i]; a_seed[i] = as_[i]; }
}

#define CKR(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"resident CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return -1;}}while(0)
static void b2w(const uint8_t* b, uint32_t* w, int n) { for (int i=0;i<n;i++) w[i]=(uint32_t)b[i*4]|((uint32_t)b[i*4+1]<<8)|((uint32_t)b[i*4+2]<<16)|((uint32_t)b[i*4+3]<<24); }
static void w2b(const uint32_t* w, uint8_t* b, int n) { for (int i=0;i<n;i++){ b[i*4]=w[i]&0xff; b[i*4+1]=(w[i]>>8)&0xff; b[i*4+2]=(w[i]>>16)&0xff; b[i*4+3]=(w[i]>>24)&0xff; } }

extern "C" {
// Génère A(m×k),B(n×k) int7 depuis setup_seed sur GPU, commit (tensor_hash, keyé job_key),
// stir → a_seed/b_seed (32o chacun). Doit == compute_commitment_hash CPU sur les mêmes A,B.
int pearl_gpu_commit_seeds(uint64_t setup_seed, int m, int n, int k,
                           const uint8_t job_key[32], uint8_t a_seed_out[32], uint8_t b_seed_out[32]) {
  cudaDeviceProp prop; CKR(cudaGetDeviceProperties(&prop, 0));
  uint32_t jk[8]; b2w(job_key, jk, 8);
  uint32_t *d_jk, *d_ha, *d_hb, *d_as, *d_bs;
  CKR(cudaMalloc(&d_jk,32)); CKR(cudaMalloc(&d_ha,32)); CKR(cudaMalloc(&d_hb,32));
  CKR(cudaMalloc(&d_as,32)); CKR(cudaMalloc(&d_bs,32));
  CKR(cudaMemcpy(d_jk, jk, 32, cudaMemcpyHostToDevice));

  int8_t *d_A, *d_B; CKR(cudaMalloc(&d_A,(size_t)m*k)); CKR(cudaMalloc(&d_B,(size_t)n*k));
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,m),256>>>(d_A, setup_seed, 0, m, k);
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,n),256>>>(d_B, setup_seed, 1, n, k);
  CKR(cudaGetLastError());

  // commit : tensor_hash(A, job_key) -> hash_a (idem B). num_blocks comme run_tensor_hash.
  const uint32_t tpb=256, ns=2, lpm=256, chunk=1024;
  auto roots_alloc = [&](uint32_t len, uint8_t** r)->int{ uint32_t nc=(len+chunk-1)/chunk, nb=(nc+tpb-1)/tpb; CKR(cudaMalloc(r,(size_t)(nb+64)*32)); return (int)nb; };
  uint8_t *d_ra, *d_rb; int nbA=roots_alloc((uint32_t)((size_t)m*k), &d_ra); int nbB=roots_alloc((uint32_t)((size_t)n*k), &d_rb);
  tensor_hash((const uint8_t*)d_A, (uint32_t)((size_t)m*k), (uint8_t*)d_ha, job_key, nbA, tpb, ns, lpm, d_ra, prop, 0);
  tensor_hash((const uint8_t*)d_B, (uint32_t)((size_t)n*k), (uint8_t*)d_hb, job_key, nbB, tpb, ns, lpm, d_rb, prop, 0);
  CKR(cudaDeviceSynchronize());

  stir_kernel<<<1,1>>>(d_ha, d_hb, d_jk, d_as, d_bs);
  CKR(cudaDeviceSynchronize()); CKR(cudaGetLastError());

  uint32_t aw[8], bw[8];
  CKR(cudaMemcpy(aw, d_as, 32, cudaMemcpyDeviceToHost));
  CKR(cudaMemcpy(bw, d_bs, 32, cudaMemcpyDeviceToHost));
  w2b(aw, a_seed_out, 8); w2b(bw, b_seed_out, 8);

  cudaFree(d_jk);cudaFree(d_ha);cudaFree(d_hb);cudaFree(d_as);cudaFree(d_bs);
  cudaFree(d_A);cudaFree(d_B);cudaFree(d_ra);cudaFree(d_rb);
  return 0;
}
}  // extern "C" (commit_seeds)

// Commit (tensor_hash) SANS set_key — copie de tensor_hash_impl ; c_key posée 1× (job_key
// change rarement) → plus de cudaMemcpyToSymbol sync par setup → streams overlapent.
template <int kNumConsumerThreads, int kNumStages, int kLeavesPerMTBlock, int kThreadLoadSize = 128>
static void commit_nokey(const uint8_t* data, uint32_t data_size, uint8_t* out,
                         uint32_t num_blocks, uint8_t* roots, cudaStream_t stream) {
  using MerkleTreeRootsKernel = pearl::MerkleTreeRootsKernel<kNumConsumerThreads, kNumStages, kThreadLoadSize>;
  constexpr static int merkle_roots_smem_size = MerkleTreeRootsKernel::SharedStorageSize;
  typename MerkleTreeRootsKernel::Arguments args{ data, data_size, reinterpret_cast<uint8_t*>(roots) };
  typename MerkleTreeRootsKernel::Params kp = MerkleTreeRootsKernel::to_underlying_arguments(args);
  auto rk = cutlass::device_kernel<MerkleTreeRootsKernel>;
  dim3 grid = MerkleTreeRootsKernel::get_grid_shape(kp), block = MerkleTreeRootsKernel::get_block_shape();
  if (merkle_roots_smem_size >= 48*1024) cudaFuncSetAttribute(reinterpret_cast<const void*>(rk), cudaFuncAttributeMaxDynamicSharedMemorySize, merkle_roots_smem_size);
  rk<<<grid, block, merkle_roots_smem_size, stream>>>(kp);
  const int nbmt = (num_blocks + kLeavesPerMTBlock - 1) / kLeavesPerMTBlock;
  if (nbmt == 1) {
    using K = pearl::ComputeBlakeMTKernel<kLeavesPerMTBlock, true>;
    typename K::Arguments a2{ reinterpret_cast<uint32_t*>(roots), num_blocks };
    typename K::Params p2 = K::to_underlying_arguments(a2);
    auto bk = cutlass::device_kernel<K>; constexpr static int sm = K::SharedStorageSize;
    if (sm >= 48*1024) cudaFuncSetAttribute(reinterpret_cast<const void*>(bk), cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    bk<<<K::get_grid_shape(p2), K::get_block_shape(), sm, stream>>>(p2);
  } else {
    using K = pearl::ComputeBlakeMTKernel<kLeavesPerMTBlock, false>;
    typename K::Arguments a2{ reinterpret_cast<uint32_t*>(roots), num_blocks };
    typename K::Params p2 = K::to_underlying_arguments(a2);
    auto bk = cutlass::device_kernel<K>; constexpr static int sm = K::SharedStorageSize;
    if (sm >= 48*1024) cudaFuncSetAttribute(reinterpret_cast<const void*>(bk), cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    bk<<<K::get_grid_shape(p2), K::get_block_shape(), sm, stream>>>(p2);
  }
  if (nbmt > 1) {
    using R = pearl::ReduceRootsKernel<kNumConsumerThreads>;
    typename R::Arguments a3{ reinterpret_cast<uint32_t*>(roots), static_cast<uint32_t>(nbmt) };
    typename R::Params p3 = R::to_underlying_arguments(a3);
    auto rr = cutlass::device_kernel<R>; constexpr static int sm = R::SharedStorageSize;
    if (sm >= 48*1024) cudaFuncSetAttribute(reinterpret_cast<const void*>(rr), cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    rr<<<R::get_grid_shape(p3), R::get_block_shape(), sm, stream>>>(p3);
  }
  cudaMemcpyAsync(out, roots, blake3::CHAINING_VALUE_SIZE, cudaMemcpyDeviceToDevice, stream);
}

// Contexte persistant : buffers alloués 1×. 2 streams (sA/sB) pour overlaper les
// côtés A et B indépendants (gen‖, commit‖, noise‖) jusqu'au stir.
struct Ctx {
  int m, n, k, max_hits;
  uint32_t nbA, nbB;
  void* prop;                       // cudaDeviceProp*
  int8_t *d_A, *d_B; int32_t* d_C;
  uint32_t *d_jk,*d_bnd,*d_ha,*d_hb,*d_as,*d_bs,*d_sla,*d_slb;
  uint8_t *d_ra,*d_rb,*d_faA,*d_saA,*d_faB,*d_saB;
  int *d_found,*d_hr,*d_hc;
  cudaStream_t sA, sB; cudaEvent_t eA, eB, eStir;
  uint8_t last_jk[32]; bool jk_set;
};

// Lance la chaîne résidente — côtés A‖B sur 2 streams (overlap prologue). Grind sur sA.
static int resident_run(Ctx* c, uint64_t setup_seed,
                        const uint8_t job_key[32], int* hit_rows, int* hit_cols) {
  int m=c->m, n=c->n, k=c->k, max_hits=c->max_hits;
  // set_key 1× (job_key change rarement → pas de cudaMemcpyToSymbol sync par setup)
  if (!c->jk_set || memcmp(c->last_jk, job_key, 32) != 0) {
    cudaDeviceSynchronize(); set_key(job_key);
    for (int i=0;i<32;i++) c->last_jk[i]=job_key[i]; c->jk_set=true;
  }
  cudaMemsetAsync(c->d_found,0,4,c->sA);
  // Phase 1 : A‖B (gen + commit), indépendants
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,m),256,0,c->sA>>>(c->d_A, setup_seed, 0, m, k);
  commit_nokey<256,2,256>((const uint8_t*)c->d_A,(uint32_t)((size_t)m*k),(uint8_t*)c->d_ha,c->nbA,c->d_ra,c->sA);
  cudaEventRecord(c->eA, c->sA);
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,n),256,0,c->sB>>>(c->d_B, setup_seed, 1, n, k);
  commit_nokey<256,2,256>((const uint8_t*)c->d_B,(uint32_t)((size_t)n*k),(uint8_t*)c->d_hb,c->nbB,c->d_rb,c->sB);
  cudaEventRecord(c->eB, c->sB);
  // Phase 2 : stir sur sA (attend hash_b côté sB)
  cudaStreamWaitEvent(c->sA, c->eB, 0);
  stir_kernel<<<1,1,0,c->sA>>>(c->d_ha,c->d_hb,c->d_jk,c->d_as,c->d_bs);
  cudaEventRecord(c->eStir, c->sA);
  // Phase 3 : noise A‖B (sB attend les seeds via eStir)
  perm_k_<<<(k+255)/256,256,0,c->sA>>>(c->d_sla, c->d_as, k, c->d_faA, c->d_saA);
  noise_add_k_<<<m,256,0,c->sA>>>(c->d_sla, c->d_as, m, k, c->d_faA, c->d_saA, c->d_A);
  cudaStreamWaitEvent(c->sB, c->eStir, 0);
  perm_k_<<<(k+255)/256,256,0,c->sB>>>(c->d_slb, c->d_bs, k, c->d_faB, c->d_saB);
  noise_add_k_<<<n,256,0,c->sB>>>(c->d_slb, c->d_bs, n, k, c->d_faB, c->d_saB, c->d_B);
  cudaEventRecord(c->eB, c->sB);
  // Phase 4 : GEMM sur sA (attend noise B via eB) ; pow_key=a_seed, bound=d_bnd
  cudaStreamWaitEvent(c->sA, c->eB, 0);
  auto prob = make_shape(m, n, k);
  auto dA = make_stride(k, Int<1>{}); auto dB = make_stride(k, Int<1>{}); auto dC = make_stride(n, Int<1>{});
  auto bM=Int<128>{}; auto bN=Int<128>{}; auto bK=Int<64>{};
  auto cta = make_shape(bM,bN,bK);
  using SmemBase = Layout<Shape <Shape <_16,_8>,        Shape <_64,_1>, Shape <_1,_3>>,
                          Stride<Stride<_64,Int<1024>>, Stride<_1,_0>,  Stride<_0,Int<8192>>>>;
  auto sA = composition(Swizzle<2,4,3>{}, SmemBase{});
  auto sB = composition(Swizzle<2,4,3>{}, SmemBase{});
  auto sC = make_layout(make_shape(bM,bN));
  using AtomG  = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>;
  using TVcopy = Layout<Shape <Shape <_4,_32>,       _16>, Stride<Stride<Int<512>,_1>, _32>>;
  using TilerC = Shape<_32,_64>;
  TiledCopy<AtomG, TVcopy, TilerC> copyA, copyB;
  TiledMMA mma = make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},
      Layout<Shape<_2,_2,_1>, Stride<_1,_2,_0>>{}, Tile<_32,_32,_32>{});
  Copy_Atom<SM75_U32x4_LDSM_N, int8_t> s2rA, s2rB;
  int reduce_every_k = 128/32;
  int smem = int(sizeof(SharedStorage<int8_t,int8_t,decltype(sA),decltype(sB)>));
  dim3 grd(size(ceil_div(m,bM)), size(ceil_div(n,bN))), blk(size(mma));
  auto kfn = gemm_device<decltype(prob),decltype(cta),
      int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),
      int8_t,decltype(dB),decltype(sB),decltype(copyB),decltype(s2rB),
      int32_t,decltype(dC),decltype(sC),decltype(mma)>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kfn<<<grd,blk,smem,c->sA>>>(prob,cta, c->d_A,dA,sA,copyA,s2rA, c->d_B,dB,sB,copyB,s2rB,
      c->d_C,dC,sC,mma, reduce_every_k, c->d_as, c->d_bnd, c->d_found, c->d_hr,c->d_hc,max_hits);
  CKR(cudaStreamSynchronize(c->sA)); CKR(cudaGetLastError());

  int found=0; CKR(cudaMemcpy(&found,c->d_found,4,cudaMemcpyDeviceToHost));
  int nret = found<max_hits?found:max_hits;
  if(nret>0 && hit_rows) CKR(cudaMemcpy(hit_rows,c->d_hr,(size_t)nret*128*4,cudaMemcpyDeviceToHost));
  if(nret>0 && hit_cols) CKR(cudaMemcpy(hit_cols,c->d_hc,(size_t)nret*128*4,cudaMemcpyDeviceToHost));
  return found;
}

extern "C" {
// ---- contexte persistant : alloc 1× (create), réutilise (grind), free (destroy) ----
void* pearl_resident_create(int m, int n, int k, int max_hits) {
  Ctx* c = new Ctx(); c->m=m; c->n=n; c->k=k; c->max_hits=max_hits;
  c->prop = new cudaDeviceProp(); if(cudaGetDeviceProperties((cudaDeviceProp*)c->prop,0)){delete c; return nullptr;}
  const uint32_t tpb=256, chunk=1024;
  c->nbA=(((uint32_t)((size_t)m*k)+chunk-1)/chunk + tpb-1)/tpb;
  c->nbB=(((uint32_t)((size_t)n*k)+chunk-1)/chunk + tpb-1)/tpb;
  auto ok=[&](cudaError_t e){ return e==cudaSuccess; };
  bool good = ok(cudaMalloc(&c->d_A,(size_t)m*k)) && ok(cudaMalloc(&c->d_B,(size_t)n*k))
    && ok(cudaMalloc(&c->d_C,(size_t)m*n*4))
    && ok(cudaMalloc(&c->d_jk,32)) && ok(cudaMalloc(&c->d_bnd,32))
    && ok(cudaMalloc(&c->d_ha,32)) && ok(cudaMalloc(&c->d_hb,32))
    && ok(cudaMalloc(&c->d_as,32)) && ok(cudaMalloc(&c->d_bs,32))
    && ok(cudaMalloc(&c->d_sla,32)) && ok(cudaMalloc(&c->d_slb,32))
    && ok(cudaMalloc(&c->d_faA,k)) && ok(cudaMalloc(&c->d_saA,k))
    && ok(cudaMalloc(&c->d_faB,k)) && ok(cudaMalloc(&c->d_saB,k))
    && ok(cudaMalloc(&c->d_found,4)) && ok(cudaMalloc(&c->d_hr,(size_t)max_hits*128*4))
    && ok(cudaMalloc(&c->d_hc,(size_t)max_hits*128*4))
    && ok(cudaMalloc(&c->d_ra,(size_t)(c->nbA+64)*32)) && ok(cudaMalloc(&c->d_rb,(size_t)(c->nbB+64)*32));
  if(!good){ delete (cudaDeviceProp*)c->prop; delete c; return nullptr; }
  // constantes : seed labels A/B
  uint8_t sla[32]={0}, slb[32]={0}; const char* la="A_tensor"; const char* lb="B_tensor";
  for(int i=0;i<8;i++){ sla[i]=la[i]; slb[i]=lb[i]; }
  uint32_t slaw[8], slbw[8]; b2w(sla,slaw,8); b2w(slb,slbw,8);
  cudaMemcpy(c->d_sla,slaw,32,cudaMemcpyHostToDevice);
  cudaMemcpy(c->d_slb,slbw,32,cudaMemcpyHostToDevice);
  cudaStreamCreate(&c->sA); cudaStreamCreate(&c->sB);
  cudaEventCreateWithFlags(&c->eA, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&c->eB, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&c->eStir, cudaEventDisableTiming);
  c->jk_set = false;
  return c;
}

int pearl_resident_grind_ctx(void* ctx, uint64_t setup_seed,
                             const uint8_t job_key[32], const uint8_t pow_bound_le[32],
                             int* hit_rows, int* hit_cols) {
  Ctx* c = (Ctx*)ctx;
  uint32_t jk[8], bnd[8]; b2w(job_key, jk, 8); b2w(pow_bound_le, bnd, 8);
  cudaMemcpy(c->d_jk, jk, 32, cudaMemcpyHostToDevice);
  cudaMemcpy(c->d_bnd, bnd, 32, cudaMemcpyHostToDevice);
  return resident_run(c, setup_seed, job_key, hit_rows, hit_cols);
}

void pearl_resident_destroy(void* ctx) {
  Ctx* c = (Ctx*)ctx; if(!c) return;
  cudaFree(c->d_A);cudaFree(c->d_B);cudaFree(c->d_C);cudaFree(c->d_jk);cudaFree(c->d_bnd);
  cudaFree(c->d_ha);cudaFree(c->d_hb);cudaFree(c->d_as);cudaFree(c->d_bs);cudaFree(c->d_sla);cudaFree(c->d_slb);
  cudaFree(c->d_faA);cudaFree(c->d_saA);cudaFree(c->d_faB);cudaFree(c->d_saB);
  cudaFree(c->d_found);cudaFree(c->d_hr);cudaFree(c->d_hc);cudaFree(c->d_ra);cudaFree(c->d_rb);
  cudaStreamDestroy(c->sA); cudaStreamDestroy(c->sB);
  cudaEventDestroy(c->eA); cudaEventDestroy(c->eB); cudaEventDestroy(c->eStir);
  delete (cudaDeviceProp*)c->prop; delete c;
}

// one-shot (utilisé par le test correctness) = create+grind+destroy.
int pearl_gpu_resident_grind(uint64_t setup_seed, int m, int n, int k,
                             const uint8_t job_key[32], const uint8_t pow_bound_le[32],
                             int* hit_rows, int* hit_cols, int max_hits) {
  void* c = pearl_resident_create(m,n,k,max_hits);
  if(!c) return -1;
  int r = pearl_resident_grind_ctx(c, setup_seed, job_key, pow_bound_le, hit_rows, hit_cols);
  pearl_resident_destroy(c);
  return r;
}
}  // extern "C"

// ============ DOUBLE-BUFFER BATCHÉ (inspiré du persistant d'alpha) ============
// 2 jeux de buffers + 2 streams : prologue(setup N+1) overlap grind(setup N).
// 1 appel = un BATCH de setups (job_key/bound constants) → amortit le host overhead
// et laisse les 2 streams se recouvrir. Le commit (set_key sync) est le seul point
// potentiellement sérialisant — mesuré empiriquement.
struct Ctx2 {
  int m, n, k;
  uint32_t nbA, nbB;
  void* prop;
  cudaStream_t st[2];
  int32_t* d_C;                       // partagé (jamais écrit)
  uint32_t *d_sla, *d_slb, *d_jk;     // partagés (constants dans le batch)
  int8_t *d_A[2], *d_B[2];
  uint32_t *d_ha[2], *d_hb[2], *d_as[2], *d_bs[2], *d_bnd[2];
  uint8_t *d_ra[2], *d_rb[2], *d_faA[2], *d_saA[2], *d_faB[2], *d_saB[2];
  int *d_found[2], *d_hr[2], *d_hc[2];
};


// Lance toute la chaîne pour 1 setup sur (slot) — ASYNC sur st[slot]. Clé déjà posée.
static void launch_chain2(Ctx2* c, int slot, uint64_t seed, const uint8_t job_key[32]) {
  cudaStream_t s = c->st[slot];
  int m=c->m,n=c->n,k=c->k;
  cudaMemsetAsync(c->d_found[slot], 0, 4, s);
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,m),256,0,s>>>(c->d_A[slot], seed, 0, m, k);
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,n),256,0,s>>>(c->d_B[slot], seed, 1, n, k);
  // commit SANS set_key (clé posée 1× par batch) → overlap réel des streams
  commit_nokey<256,2,256>((const uint8_t*)c->d_A[slot],(uint32_t)((size_t)m*k),(uint8_t*)c->d_ha[slot],c->nbA,c->d_ra[slot],s);
  commit_nokey<256,2,256>((const uint8_t*)c->d_B[slot],(uint32_t)((size_t)n*k),(uint8_t*)c->d_hb[slot],c->nbB,c->d_rb[slot],s);
  stir_kernel<<<1,1,0,s>>>(c->d_ha[slot],c->d_hb[slot],c->d_jk,c->d_as[slot],c->d_bs[slot]);
  perm_k_<<<(k+255)/256,256,0,s>>>(c->d_sla, c->d_as[slot], k, c->d_faA[slot], c->d_saA[slot]);
  perm_k_<<<(k+255)/256,256,0,s>>>(c->d_slb, c->d_bs[slot], k, c->d_faB[slot], c->d_saB[slot]);
  noise_add_k_<<<m,256,0,s>>>(c->d_sla, c->d_as[slot], m, k, c->d_faA[slot], c->d_saA[slot], c->d_A[slot]);
  noise_add_k_<<<n,256,0,s>>>(c->d_slb, c->d_bs[slot], n, k, c->d_faB[slot], c->d_saB[slot], c->d_B[slot]);
  // grind (recette alpha)
  using namespace cute;
  auto prob = make_shape(m,n,k);
  auto dA=make_stride(k,Int<1>{}); auto dB=make_stride(k,Int<1>{}); auto dC=make_stride(n,Int<1>{});
  auto bM=Int<128>{}; auto bN=Int<128>{}; auto bK=Int<64>{};
  auto cta=make_shape(bM,bN,bK);
  using SmemBase=Layout<Shape<Shape<_16,_8>,Shape<_64,_1>,Shape<_1,_3>>,Stride<Stride<_64,Int<1024>>,Stride<_1,_0>,Stride<_0,Int<8192>>>>;
  auto sA=composition(Swizzle<2,4,3>{},SmemBase{}); auto sB=composition(Swizzle<2,4,3>{},SmemBase{});
  auto sC=make_layout(make_shape(bM,bN));
  using AtomG=Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>,int8_t>;
  using TVcopy=Layout<Shape<Shape<_4,_32>,_16>,Stride<Stride<Int<512>,_1>,_32>>;
  TiledCopy<AtomG,TVcopy,Shape<_32,_64>> copyA, copyB;
  TiledMMA mma=make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},Layout<Shape<_2,_2,_1>,Stride<_1,_2,_0>>{},Tile<_32,_32,_32>{});
  Copy_Atom<SM75_U32x4_LDSM_N,int8_t> s2rA,s2rB;
  int rek=128/32, smem=int(sizeof(SharedStorage<int8_t,int8_t,decltype(sA),decltype(sB)>));
  dim3 grd(size(ceil_div(m,bM)),size(ceil_div(n,bN))), blk(size(mma));
  auto kfn=gemm_device<decltype(prob),decltype(cta),int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),int8_t,decltype(dB),decltype(sB),decltype(copyB),decltype(s2rB),int32_t,decltype(dC),decltype(sC),decltype(mma)>;
  cudaFuncSetAttribute(kfn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  kfn<<<grd,blk,smem,s>>>(prob,cta,c->d_A[slot],dA,sA,copyA,s2rA,c->d_B[slot],dB,sB,copyB,s2rB,c->d_C,dC,sC,mma,rek,c->d_as[slot],c->d_bnd[slot],c->d_found[slot],c->d_hr[slot],c->d_hc[slot],64);
}

#define MK(p,sz) do{ if(cudaMalloc((void**)&(p),(sz))!=cudaSuccess){return nullptr;} }while(0)
extern "C" {
void* pearl_resident2_create(int m,int n,int k){
  Ctx2* c=new Ctx2(); c->m=m;c->n=n;c->k=k;
  c->prop=new cudaDeviceProp(); if(cudaGetDeviceProperties((cudaDeviceProp*)c->prop,0)){delete c;return nullptr;}
  const uint32_t tpb=256,chunk=1024;
  c->nbA=(((uint32_t)((size_t)m*k)+chunk-1)/chunk+tpb-1)/tpb;
  c->nbB=(((uint32_t)((size_t)n*k)+chunk-1)/chunk+tpb-1)/tpb;
  MK(c->d_C,(size_t)m*n*4); MK(c->d_sla,32); MK(c->d_slb,32); MK(c->d_jk,32);
  for(int s=0;s<2;s++){ cudaStreamCreate(&c->st[s]);
    MK(c->d_A[s],(size_t)m*k); MK(c->d_B[s],(size_t)n*k);
    MK(c->d_ha[s],32);MK(c->d_hb[s],32);MK(c->d_as[s],32);MK(c->d_bs[s],32);MK(c->d_bnd[s],32);
    MK(c->d_faA[s],k);MK(c->d_saA[s],k);MK(c->d_faB[s],k);MK(c->d_saB[s],k);
    MK(c->d_found[s],4);MK(c->d_hr[s],(size_t)64*128*4);MK(c->d_hc[s],(size_t)64*128*4);
    MK(c->d_ra[s],(size_t)(c->nbA+64)*32);MK(c->d_rb[s],(size_t)(c->nbB+64)*32); }
  uint8_t sla[32]={0},slb[32]={0}; const char* la="A_tensor";const char* lb="B_tensor";
  for(int i=0;i<8;i++){sla[i]=la[i];slb[i]=lb[i];}
  uint32_t sw[8],sw2[8]; b2w(sla,sw,8); b2w(slb,sw2,8);
  cudaMemcpy(c->d_sla,sw,32,cudaMemcpyHostToDevice); cudaMemcpy(c->d_slb,sw2,32,cudaMemcpyHostToDevice);
  return c;
}
// Pipeline un BATCH de num setups. Remplit out_idx/out_rows/out_cols (≤max_out hits, 1er hit/setup).
// Retour = nb de hits écrits. setup i utilise seed=base_seed+i.
int pearl_resident2_grind_batch(void* ctx, unsigned long long base_seed, const uint8_t job_key[32],
                                const uint8_t bound_le[32], int num,
                                int* out_idx, int* out_rows, int* out_cols, int max_out){
  Ctx2* c=(Ctx2*)ctx;
  uint32_t jk[8],bnd[8]; b2w(job_key,jk,8); b2w(bound_le,bnd,8);
  cudaMemcpy(c->d_jk,jk,32,cudaMemcpyHostToDevice);
  for(int s=0;s<2;s++) cudaMemcpy(c->d_bnd[s],bnd,32,cudaMemcpyHostToDevice);
  set_key(job_key);                  // clé c_key posée 1× pour TOUT le batch (job_key constant)
  cudaDeviceSynchronize();           // garantit c_key prête avant les commits async
  int out=0;
  auto collect=[&](int setup_i){ int slot=setup_i&1;
    cudaStreamSynchronize(c->st[slot]);
    int found=0; cudaMemcpy(&found,c->d_found[slot],4,cudaMemcpyDeviceToHost);
    if(found>0 && out<max_out){
      cudaMemcpy(out_rows+(size_t)out*128,c->d_hr[slot],128*4,cudaMemcpyDeviceToHost);
      cudaMemcpy(out_cols+(size_t)out*128,c->d_hc[slot],128*4,cudaMemcpyDeviceToHost);
      out_idx[out]=setup_i; out++; }
  };
  for(int i=0;i<num;i++){
    if(i>=2) collect(i-2);
    launch_chain2(c,i&1,base_seed+(unsigned long long)i,job_key);
  }
  for(int i=(num>=2?num-2:0); i<num; i++) collect(i);
  return out;
}
void pearl_resident2_destroy(void* ctx){
  Ctx2* c=(Ctx2*)ctx; if(!c)return;
  cudaFree(c->d_C);cudaFree(c->d_sla);cudaFree(c->d_slb);cudaFree(c->d_jk);
  for(int s=0;s<2;s++){ cudaStreamDestroy(c->st[s]);
    cudaFree(c->d_A[s]);cudaFree(c->d_B[s]);cudaFree(c->d_ha[s]);cudaFree(c->d_hb[s]);
    cudaFree(c->d_as[s]);cudaFree(c->d_bs[s]);cudaFree(c->d_bnd[s]);
    cudaFree(c->d_faA[s]);cudaFree(c->d_saA[s]);cudaFree(c->d_faB[s]);cudaFree(c->d_saB[s]);
    cudaFree(c->d_found[s]);cudaFree(c->d_hr[s]);cudaFree(c->d_hc[s]);cudaFree(c->d_ra[s]);cudaFree(c->d_rb[s]); }
  delete (cudaDeviceProp*)c->prop; delete c;
}
}  // extern "C"
