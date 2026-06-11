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
#include "pearl_gpu_kernel.cuh"        // gemm_device (grind 8×16, v1.0)
#include "pearl_gpu_kernel_2x64.cuh"   // gemm_device_2x64 (grind 2×64, v1.1 AlphaPool)
#include "pearl_gpu_kernel_tma.cuh"    // gemm_device_tma_ms (grind multistage TMA 128×256, v0.6.0-ws)
#include "pearl_gpu_kernel_cpasync_ms.cuh" // gemm_device_cpasync_ms (jumeau PORTABLE Ampere/Ada, v0.6.3-beta)
#include "merkle_roots_cpasync.cuh"        // MerkleTreeRootsKernelCpAsync (commit PORTABLE — l'officiel est TMA/SM90-only)
using namespace cute;

// v0.6.3 : faut-il le path PORTABLE cp.async (au lieu du TMA Blackwell) ?
// Vrai si le GPU n'a pas le TMA (major < 9 = Ampere sm_80/86, Ada sm_89) OU si forcé
// par ARIA_FORCE_CPASYNC=1 (validation byte-exact du path cp.async sur 5080).
static bool aria_use_cpasync(int major) {
  if (getenv("ARIA_FORCE_CPASYNC")) return true;
  return major < 9;   // TMA dispo seulement sm_90 (Hopper) + sm_100/120 (Blackwell)
}

// Major du device courant, caché (pour les helpers sans accès au Ctx, ex commit_nokey).
static int aria_device_major() {
  static int m = -1;
  if (m < 0) { int d = 0; cudaGetDevice(&d); cudaDeviceProp p; cudaGetDeviceProperties(&p, d); m = p.major; }
  return m;
}

// v0.6.1 : largeur de bande (en tuiles-M) du swizzle de grille du grind multistage.
// ARIA_SWZ_G (défaut 64, 0 = off). Optimum mesuré RTX 5080 forme 131072² : plateau
// 64-96 (+12%) ; 128 déborde du L2. Changement d'ORDRE des CTAs uniquement.
static int aria_swz_g() {
  static int v = -1;
  if (v < 0) {
    const char* e = getenv("ARIA_SWZ_G");
    if (e) { v = atoi(e); if (v < 0) v = 0; }
    else {
      // Défaut AUTO par taille de L2 (la bande de tuiles-M doit tenir dans le L2).
      // Mesuré : 5080 L2 64Mo → 64 (plateau 64-96) ; 4060 L2 24Mo → 32 (plateau 16-32).
      int dev = 0, l2 = 0; cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
      const int l2mb = l2 >> 20;
      v = (l2mb >= 48) ? 64 : (l2mb >= 16) ? 32 : (l2mb >= 8) ? 16 : 8;
    }
  }
  return v;
}

// v0.6.2 : overlap prologue (prefetch N+1 pendant le grind N). ARIA_OVERLAP=1 pour activer.
// DÉFAUT 0 (OFF) : mesuré WASH à toutes les puissances/priorités — le kernel est SM 100%
// compute-bound, aucun trou à remplir, cacher le prologue déplace le même ALU (11/06).
// Code gardé (A/B validé byte-exact via overlap_check) : resservira si le prologue grossit.
static int aria_overlap() {
  static int v = -1;
  if (v < 0) { const char* e = getenv("ARIA_OVERLAP"); v = e ? atoi(e) : 0; }
  return v;
}

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
// perm (1 thr = 1 col). `rank` = noise_rank (puissance de 2, ≤256 : les index
// first/second sont stockés en uint8). Formules = zk-pow pearl_noise.rs
// generate_permutation_matrix, paramétriques en rank.
__global__ void perm_k_(const uint32_t* seed, const uint32_t* key, int k,
                        uint8_t* first, uint8_t* second, int rank) {
  int j = blockIdx.x * blockDim.x + threadIdx.x; if (j >= k) return;
  uint32_t h[8]; rh_((uint32_t)(j / 8), seed, key, 1, h);
  uint32_t ru = h[j & 7]; uint32_t f = ru & (uint32_t)(rank - 1);
  first[j] = (uint8_t)f; second[j] = (uint8_t)(f ^ (1u + mulhi_((uint32_t)(rank - 1), ru)));
}
// noise-add FUSÉ : mat (signal in-place) += noise. mat devient a_eff/b_eff.
// 1 BLOC = 1 ligne ; les threads coopèrent sur les k colonnes (e_al en shared).
// rank/32 threads calculent les blake3 d'e_al, puis tous parcourent k en grid-stride.
__global__ void noise_add_k_(const uint32_t* seed, const uint32_t* key, int m, int k,
                             const uint8_t* first, const uint8_t* second, int8_t* mat,
                             int rank) {
  int i = blockIdx.x; if (i >= m) return;
  __shared__ int8_t e_al[256];          // dimensionné au rank max supporté (uint8 idx)
  int nblk = rank >> 5;                  // rank/32 octets-blocs blake3
  if (threadIdx.x < (unsigned)nblk) {
    uint32_t h[8]; rh_((uint32_t)(i * nblk + threadIdx.x), seed, key, 0, h);
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
  uint32_t k4 = k >> 2;
  // grille.y plafonnée (max 65535) → boucle sur les lignes (transparent si rows ≤ gridDim.y,
  // ex. 8192). Indispensable à 131072 (sinon grid.y > 65535 = invalid configuration).
  for (uint32_t i = blockIdx.y; i < rows; i += gridDim.y) {
    uint32_t* row4 = (uint32_t*)(mat + (size_t)i * k);
    for (uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < k4; q += gridDim.x * blockDim.x) {
      uint32_t j = q << 2;
      uint8_t b0 = (uint8_t)int7_at(seed, sel, i, j);
      uint8_t b1 = (uint8_t)int7_at(seed, sel, i, j + 1);
      uint8_t b2 = (uint8_t)int7_at(seed, sel, i, j + 2);
      uint8_t b3 = (uint8_t)int7_at(seed, sel, i, j + 3);
      row4[q] = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
    }
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
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,(unsigned)(m<32768?m:32768)),256>>>(d_A, setup_seed, 0, m, k);
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,(unsigned)(n<32768?n:32768)),256>>>(d_B, setup_seed, 1, n, k);
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
  if (aria_use_cpasync(aria_device_major())) {
    // Jumeau portable : même mapping/réduction, staging cp.async (l'officiel crée un
    // descripteur TMA côté hôte → erreur 801 sur Ampere/Ada avant même le launch).
    using KCpa = pearl::MerkleTreeRootsKernelCpAsync<kNumConsumerThreads, kNumStages, kThreadLoadSize>;
    constexpr static int smem_cpa = KCpa::SharedStorageSize;
    typename KCpa::Arguments args{ data, data_size, reinterpret_cast<uint8_t*>(roots) };
    typename KCpa::Params kp = KCpa::to_underlying_arguments(args);
    auto rk = cutlass::device_kernel<KCpa>;
    dim3 grid = KCpa::get_grid_shape(kp), block = KCpa::get_block_shape();
    if (smem_cpa >= 48*1024) cudaFuncSetAttribute(reinterpret_cast<const void*>(rk), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_cpa);
    rk<<<grid, block, smem_cpa, stream>>>(kp);
  } else {
  using MerkleTreeRootsKernel = pearl::MerkleTreeRootsKernel<kNumConsumerThreads, kNumStages, kThreadLoadSize>;
  constexpr static int merkle_roots_smem_size = MerkleTreeRootsKernel::SharedStorageSize;
  typename MerkleTreeRootsKernel::Arguments args{ data, data_size, reinterpret_cast<uint8_t*>(roots) };
  typename MerkleTreeRootsKernel::Params kp = MerkleTreeRootsKernel::to_underlying_arguments(args);
  auto rk = cutlass::device_kernel<MerkleTreeRootsKernel>;
  dim3 grid = MerkleTreeRootsKernel::get_grid_shape(kp), block = MerkleTreeRootsKernel::get_block_shape();
  if (merkle_roots_smem_size >= 48*1024) cudaFuncSetAttribute(reinterpret_cast<const void*>(rk), cudaFuncAttributeMaxDynamicSharedMemorySize, merkle_roots_smem_size);
  rk<<<grid, block, merkle_roots_smem_size, stream>>>(kp);
  }
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
  int rank;                         // noise_rank (env ARIA_RANK, défaut 128, ≤256)
  bool big_endian;                  // ARIA_BE : pow-check big-endian (LuckyPool) sur le multistage
  uint32_t nbA, nbB;
  void* prop;                       // cudaDeviceProp*
  int8_t *d_A, *d_B; int32_t* d_C;
  uint32_t *d_jk,*d_bnd,*d_ha,*d_hb,*d_as,*d_bs,*d_sla,*d_slb;
  uint8_t *d_ra,*d_rb,*d_faA,*d_saA,*d_faB,*d_saB;
  int *d_found,*d_hr,*d_hc;
  cudaStream_t sA, sB; cudaEvent_t eA, eB, eStir;
  uint8_t last_jk[32]; bool jk_set;
  bool tile2x64;   // false = grind 8×16 (v1.0) ; true = grind 2×64 (v1.1 AlphaPool)
  // --- instrumentation (v0.5.0) : split prologue vs grind par cudaEvents timés ---
  cudaEvent_t tStart, tMid, tPro, tEnd; float last_pro_ms, last_grind_ms, last_genc_ms, last_noise_ms; bool do_timing;
  // --- v0.6.2 : OVERLAP PROLOGUE (pipeline 2 slots A, prefetch N+1 pendant le grind N) ---
  // Slot 1 = d_A2/d_ha2/d_as2. Le grind N lit slot p pendant que le prologue N+1
  // (gen+commit+stir+perm+noise A, fix-B : B résident intouché) écrit le slot 1-p sur
  // sP (stream HAUTE PRIORITÉ non-bloquant). Scratch prologue (d_sla/d_faA/d_saA/d_ra)
  // partagé : UN seul prologue en vol à la fois. nullptr = pipeline désactivé.
  int8_t* d_A2; uint32_t *d_ha2, *d_as2;
  cudaStream_t sP; cudaEvent_t ePro, p0, pMid, p1;
  bool bfilled;                       // fix-B : B résident construit pour ce job (membre Ctx, pas static)
  bool pend_valid; uint64_t pend_seed; int pend_slot;   // prologue préfetché en attente
};

// Lance la chaîne résidente — côtés A‖B sur 2 streams (overlap prologue). Grind sur sA.
static int resident_run(Ctx* c, uint64_t setup_seed,
                        const uint8_t job_key[32], int* hit_rows, int* hit_cols) {
  int m=c->m, n=c->n, k=c->k, max_hits=c->max_hits;
  // VALIDATION étape A (10/06) : mode fix-B (ARIA_FIXB) — b_noise_seed ne dépend QUE de B
  // (official_grind compute_commitment_hash) → on fixe B (gen+commit+noise B faits 1×),
  // on ne fait varier qu'A. Mesure le gain prologue. s_bfilled réinit si job_key change.
  static bool s_bfilled = false;
  bool fixb = (getenv("ARIA_FIXB") != nullptr);
  // set_key 1× (job_key change rarement → pas de cudaMemcpyToSymbol sync par setup)
  if (!c->jk_set || memcmp(c->last_jk, job_key, 32) != 0) {
    cudaDeviceSynchronize(); set_key(job_key);
    for (int i=0;i<32;i++) c->last_jk[i]=job_key[i]; c->jk_set=true;
    s_bfilled = false;
  }
  cudaMemsetAsync(c->d_found,0,4,c->sA);
  if (c->do_timing) cudaEventRecord(c->tStart, c->sA);
  // Phase 1 : A‖B (gen + commit), indépendants
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,(unsigned)(m<32768?m:32768)),256,0,c->sA>>>(c->d_A, setup_seed, 0, m, k);
  commit_nokey<256,2,256>((const uint8_t*)c->d_A,(uint32_t)((size_t)m*k),(uint8_t*)c->d_ha,c->nbA,c->d_ra,c->sA);
  cudaEventRecord(c->eA, c->sA);
  if (!(fixb && s_bfilled)) {   // fix-B : gen+commit B faits 1× seulement
    gen_signal_kernel<<<dim3(((k>>2)+255)/256,(unsigned)(n<32768?n:32768)),256,0,c->sB>>>(c->d_B, setup_seed, 1, n, k);
    commit_nokey<256,2,256>((const uint8_t*)c->d_B,(uint32_t)((size_t)n*k),(uint8_t*)c->d_hb,c->nbB,c->d_rb,c->sB);
  }
  cudaEventRecord(c->eB, c->sB);
  // Phase 2 : stir sur sA (attend hash_b côté sB)
  cudaStreamWaitEvent(c->sA, c->eB, 0);
  stir_kernel<<<1,1,0,c->sA>>>(c->d_ha,c->d_hb,c->d_jk,c->d_as,c->d_bs);
  cudaEventRecord(c->eStir, c->sA);
  if (c->do_timing) cudaEventRecord(c->tMid, c->sA); // fin gen+commit+stir, avant noise
  // Phase 3 : noise A‖B (sB attend les seeds via eStir)
  perm_k_<<<(k+255)/256,256,0,c->sA>>>(c->d_sla, c->d_as, k, c->d_faA, c->d_saA, c->rank);
  noise_add_k_<<<m,256,0,c->sA>>>(c->d_sla, c->d_as, m, k, c->d_faA, c->d_saA, c->d_A, c->rank);
  cudaStreamWaitEvent(c->sB, c->eStir, 0);
  if (!(fixb && s_bfilled)) {   // fix-B : noise B fait 1× → d_B garde b_eff résident
    perm_k_<<<(k+255)/256,256,0,c->sB>>>(c->d_slb, c->d_bs, k, c->d_faB, c->d_saB, c->rank);
    noise_add_k_<<<n,256,0,c->sB>>>(c->d_slb, c->d_bs, n, k, c->d_faB, c->d_saB, c->d_B, c->rank);
  }
  cudaEventRecord(c->eB, c->sB);
  if (fixb) s_bfilled = true;
  // Phase 4 : GEMM sur sA (attend noise B via eB) ; pow_key=a_seed, bound=d_bnd
  cudaStreamWaitEvent(c->sA, c->eB, 0);
  if (c->do_timing) cudaEventRecord(c->tPro, c->sA); // fin prologue = juste avant le GEMM
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
  // Le fold officiel (zk-pow compute_jackpot) réduit l'accumulateur partiel tous
  // les `rank` colonnes de K : `for ll in (rank..=k).step_by(rank)`. Le kernel
  // réduit tous les `reduce_every_k` MMA-steps (32 cols chacun) → reduce_every_k =
  // rank/32. ⚠️ ÉTAIT hardcodé 128/32=4 (rank 128) → FAUX à rank 256 (doit être 8).
  int reduce_every_k = c->rank / 32;
  dim3 grd(size(ceil_div(m,bM)), size(ceil_div(n,bN))), blk(size(mma));
  if (getenv("ARIA_TMA_MS")) {
    // étape 1 (v0.6.0-ws) : grind MULTISTAGE TMA 128×256 (8 warps, A cp.async + B TMA ring).
    // DumpC=false. ⚠️ PERF ONLY : coords 2×4 PAS encore consensus-validées (bound=0 → 0 hit).
    auto bM2=Int<128>{}; auto bN2=Int<256>{}; auto bK2=Int<128>{};
    auto cta2 = make_shape(bM2,bN2,bK2);
    // K_PIPE = dernière dim du Shape (étages du ring TMA). =2 : smem ~96KB déjà au
    // plafond du 5080 (~100KB/SM) → impossible d'ajouter un stage (tâtonnement : _3 =
    // cudaErrorInvalidValue). Le vrai goulot = occupation 1 CTA/SM (smem trop gros).
    auto sAm = composition(Swizzle<3,4,3>{},
        Layout<Shape<Shape<_16,_8 >,Shape<_128,_1>,_2>, Stride<Stride<_128,Int<2048>>,Stride<_1,_0>,Int<16384>>>{});
    auto sBm = composition(Swizzle<3,4,3>{},
        Layout<Shape<Shape<_16,_16>,Shape<_128,_1>,_2>, Stride<Stride<_128,Int<2048>>,Stride<_1,_0>,Int<32768>>>{});
    auto sB1 = composition(Swizzle<3,4,3>{},
        Layout<Shape<Shape<_16,_16>,Shape<_128,_1>>, Stride<Stride<_128,Int<2048>>,Stride<_1,_0>>>{});
    auto sCm = make_layout(make_shape(bM2,bN2));
    using AtomG2 = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>;
    using TVcopy2 = Layout<Shape<Shape<_8,_32>,_16>, Stride<Stride<_512,_1>,_32>>;
    using TilerC2 = Shape<_32,_128>;
    TiledCopy<AtomG2,TVcopy2,TilerC2> copyA2;
    // copyB2 : MÊME type de TiledCopy que A — le tiler 32×128 partitionne aussi la
    // tuile B (256×128) en 8 répétitions N → B chargeable en cp.async (path portable).
    TiledCopy<AtomG2,TVcopy2,TilerC2> copyB2;
    TiledMMA mma2 = make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{}, Layout<Shape<_2,_4,_1>,Stride<_1,_2,_0>>{}, Tile<_32,_32,_32>{});
    Copy_Atom<SM75_U32x4_LDSM_N,int8_t> s2rA2; Copy_Atom<SM75_U32x2_LDSM_N,int8_t> s2rB2;
    bool use_cpa = aria_use_cpasync(((cudaDeviceProp*)c->prop)->major);
    // ---- v0.6.3-beta : path PORTABLE cp.async (Ampere/Ada, ou forcé pour validation) ----
    if (use_cpa) {
      int smem_c = int(sizeof(SharedStorageCPA_MS<int8_t,int8_t,decltype(sAm),decltype(sBm)>));
      dim3 grd2(size(ceil_div(m,bM2)), size(ceil_div(n,bN2))), blk2(size(mma2));
      auto setattr = [&](auto kfn){ cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_c); };
      if (c->big_endian) {
        auto kfn = gemm_device_cpasync_ms<decltype(prob),decltype(cta2),
            int8_t,decltype(dA),decltype(sAm),decltype(copyA2),decltype(s2rA2),
            int8_t,decltype(dB),decltype(copyB2),decltype(sBm),decltype(s2rB2),
            int32_t,decltype(dC),decltype(sCm),decltype(mma2), /*DumpC=*/false, /*BE=*/true>;
        setattr(kfn);
        kfn<<<grd2,blk2,smem_c,c->sA>>>(prob,cta2, c->d_A,dA,sAm,copyA2,s2rA2, c->d_B,dB,copyB2,sBm,s2rB2,
            c->d_C,dC,sCm,mma2, reduce_every_k, aria_swz_g(), c->d_as, c->d_bnd, c->d_found, c->d_hr,c->d_hc,max_hits);
      } else {
        auto kfn = gemm_device_cpasync_ms<decltype(prob),decltype(cta2),
            int8_t,decltype(dA),decltype(sAm),decltype(copyA2),decltype(s2rA2),
            int8_t,decltype(dB),decltype(copyB2),decltype(sBm),decltype(s2rB2),
            int32_t,decltype(dC),decltype(sCm),decltype(mma2), /*DumpC=*/false, /*BE=*/false>;
        setattr(kfn);
        kfn<<<grd2,blk2,smem_c,c->sA>>>(prob,cta2, c->d_A,dA,sAm,copyA2,s2rA2, c->d_B,dB,copyB2,sBm,s2rB2,
            c->d_C,dC,sCm,mma2, reduce_every_k, aria_swz_g(), c->d_as, c->d_bnd, c->d_found, c->d_hr,c->d_hc,max_hits);
      }
      if (c->do_timing) cudaEventRecord(c->tEnd, c->sA);
      CKR(cudaStreamSynchronize(c->sA)); CKR(cudaGetLastError());
      if (c->do_timing) {
        cudaEventElapsedTime(&c->last_pro_ms,   c->tStart, c->tPro);
        cudaEventElapsedTime(&c->last_grind_ms, c->tPro,   c->tEnd);
        cudaEventElapsedTime(&c->last_genc_ms,  c->tStart, c->tMid);
        cudaEventElapsedTime(&c->last_noise_ms, c->tMid,   c->tPro);
      }
      int found=0; CKR(cudaMemcpy(&found,c->d_found,4,cudaMemcpyDeviceToHost));
      int nret = found<max_hits?found:max_hits;
      if(nret>0 && hit_rows) CKR(cudaMemcpy(hit_rows,c->d_hr,(size_t)nret*128*4,cudaMemcpyDeviceToHost));
      if(nret>0 && hit_cols) CKR(cudaMemcpy(hit_cols,c->d_hc,(size_t)nret*128*4,cudaMemcpyDeviceToHost));
      return found;
    }
    // Descripteur TMA créé APRÈS le dispatch : cuTensorMapEncode échoue (801,
    // cudaErrorNotSupported) sur pré-SM90, et le path cp.async ne l'utilise pas.
    Tensor gB_t = make_tensor(make_gmem_ptr<int8_t>(c->d_B), make_layout(make_shape(n,k), make_stride(k,Int<1>{})));
    auto tma_b = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gB_t, sB1, make_shape(bN2,bK2), Int<1>{});
    int smem = int(sizeof(SharedStorageTMA_MS<int8_t,int8_t,decltype(sAm),decltype(sBm)>));
    cudaFuncSetAttribute(
        gemm_device_tma_ms<decltype(prob),decltype(cta2),
          int8_t,decltype(dA),decltype(sAm),decltype(copyA2),decltype(s2rA2),
          int8_t,decltype(dB),decltype(tma_b),decltype(sBm),decltype(s2rB2),
          int32_t,decltype(dC),decltype(sCm),decltype(mma2), /*DumpC=*/false, /*BE=*/false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(
        gemm_device_tma_ms<decltype(prob),decltype(cta2),
          int8_t,decltype(dA),decltype(sAm),decltype(copyA2),decltype(s2rA2),
          int8_t,decltype(dB),decltype(tma_b),decltype(sBm),decltype(s2rB2),
          int32_t,decltype(dC),decltype(sCm),decltype(mma2), /*DumpC=*/false, /*BE=*/true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grd2(size(ceil_div(m,bM2)), size(ceil_div(n,bN2))), blk2(size(mma2));
    // ARIA_BE=1 (LuckyPool) → pow-check big-endian. d_bnd doit alors contenir les
    // OCTETS du target big-endian tels quels (le grind FFI les reçoit ainsi).
    if (c->big_endian) {
      auto kfn = gemm_device_tma_ms<decltype(prob),decltype(cta2),
          int8_t,decltype(dA),decltype(sAm),decltype(copyA2),decltype(s2rA2),
          int8_t,decltype(dB),decltype(tma_b),decltype(sBm),decltype(s2rB2),
          int32_t,decltype(dC),decltype(sCm),decltype(mma2), /*DumpC=*/false, /*BE=*/true>;
      kfn<<<grd2,blk2,smem,c->sA>>>(prob,cta2, c->d_A,dA,sAm,copyA2,s2rA2, c->d_B,dB,tma_b,sBm,s2rB2,
          c->d_C,dC,sCm,mma2, reduce_every_k, aria_swz_g(), c->d_as, c->d_bnd, c->d_found, c->d_hr,c->d_hc,max_hits);
    } else {
      auto kfn = gemm_device_tma_ms<decltype(prob),decltype(cta2),
          int8_t,decltype(dA),decltype(sAm),decltype(copyA2),decltype(s2rA2),
          int8_t,decltype(dB),decltype(tma_b),decltype(sBm),decltype(s2rB2),
          int32_t,decltype(dC),decltype(sCm),decltype(mma2), /*DumpC=*/false, /*BE=*/false>;
      kfn<<<grd2,blk2,smem,c->sA>>>(prob,cta2, c->d_A,dA,sAm,copyA2,s2rA2, c->d_B,dB,tma_b,sBm,s2rB2,
          c->d_C,dC,sCm,mma2, reduce_every_k, aria_swz_g(), c->d_as, c->d_bnd, c->d_found, c->d_hr,c->d_hc,max_hits);
    }
  } else if (c->tile2x64) {
    // v1.1 AlphaPool : grind 2×64 (fold tuile 2 lignes {r,r+32} × 64 cols).
    int smem = int(sizeof(SharedStorage2x64<int8_t,int8_t,decltype(sA),decltype(sB)>));
    auto kfn = gemm_device_2x64<decltype(prob),decltype(cta),
        int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),
        int8_t,decltype(dB),decltype(sB),decltype(copyB),decltype(s2rB),
        int32_t,decltype(dC),decltype(sC),decltype(mma)>;
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kfn<<<grd,blk,smem,c->sA>>>(prob,cta, c->d_A,dA,sA,copyA,s2rA, c->d_B,dB,sB,copyB,s2rB,
        c->d_C,dC,sC,mma, reduce_every_k, c->d_as, c->d_bnd, c->d_found, c->d_hr,c->d_hc,max_hits);
  } else {
    // v1.0 AriaPool : grind 8×16.
    int smem = int(sizeof(SharedStorage<int8_t,int8_t,decltype(sA),decltype(sB)>));
    auto kfn = gemm_device<decltype(prob),decltype(cta),
        int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),
        int8_t,decltype(dB),decltype(sB),decltype(copyB),decltype(s2rB),
        int32_t,decltype(dC),decltype(sC),decltype(mma)>;
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kfn<<<grd,blk,smem,c->sA>>>(prob,cta, c->d_A,dA,sA,copyA,s2rA, c->d_B,dB,sB,copyB,s2rB,
        c->d_C,dC,sC,mma, reduce_every_k, c->d_as, c->d_bnd, c->d_found, c->d_hr,c->d_hc,max_hits);
  }
  if (c->do_timing) cudaEventRecord(c->tEnd, c->sA);
  CKR(cudaStreamSynchronize(c->sA)); CKR(cudaGetLastError());
  if (c->do_timing) {
    cudaEventElapsedTime(&c->last_pro_ms,   c->tStart, c->tPro);
    cudaEventElapsedTime(&c->last_grind_ms, c->tPro,   c->tEnd);
    cudaEventElapsedTime(&c->last_genc_ms,  c->tStart, c->tMid);  // gen+commit+stir
    cudaEventElapsedTime(&c->last_noise_ms, c->tMid,   c->tPro);  // perm+noise
  }

  int found=0; CKR(cudaMemcpy(&found,c->d_found,4,cudaMemcpyDeviceToHost));
  int nret = found<max_hits?found:max_hits;
  if(nret>0 && hit_rows) CKR(cudaMemcpy(hit_rows,c->d_hr,(size_t)nret*128*4,cudaMemcpyDeviceToHost));
  if(nret>0 && hit_cols) CKR(cudaMemcpy(hit_cols,c->d_hc,(size_t)nret*128*4,cudaMemcpyDeviceToHost));
  return found;
}

// ============================================================================
// v0.6.2 — OVERLAP PROLOGUE.
// Pendant que le grind(N) tourne sur sA (~342 ms), le prologue du setup N+1
// (gen A → commit → stir → perm → noise, ~2.7 ms) tourne sur sP (haute priorité)
// dans le slot opposé (d_A2/d_ha2/d_as2). Au setup suivant le grind démarre
// immédiatement (waitEvent ePro). Réservé au chemin LuckyPool (ARIA_TMA_MS +
// ARIA_FIXB : B résident → le prologue prefetch ne touche QUE le côté A) ;
// sinon fallback resident_run. Le kernel grind est INCHANGÉ : mêmes entrées
// par setup ⇒ mêmes sorties (validé A/B par overlap_check + live).
// Races éliminées par construction : grind lit slot p / prefetch écrit 1-p ;
// d_hb,d_jk,d_sla stables ; d_bs écrit sans lecteur (fix-B) ; scratch commit
// d_ra protégé par eStir sur le chemin froid ; prefetch-vs-prefetch sérialisés
// par sP lui-même.
// ============================================================================
static void prologue_A_slot(Ctx* c, uint64_t seed, int8_t* dA, uint32_t* dha, uint32_t* das,
                            cudaStream_t s) {
  int m=c->m, k=c->k;
  cudaEventRecord(c->p0, s);
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,(unsigned)(m<32768?m:32768)),256,0,s>>>(dA, seed, 0, m, k);
  commit_nokey<256,2,256>((const uint8_t*)dA,(uint32_t)((size_t)m*k),(uint8_t*)dha,c->nbA,c->d_ra,s);
  stir_kernel<<<1,1,0,s>>>(dha,c->d_hb,c->d_jk,das,c->d_bs);
  cudaEventRecord(c->pMid, s);
  perm_k_<<<(k+255)/256,256,0,s>>>(c->d_sla, das, k, c->d_faA, c->d_saA, c->rank);
  noise_add_k_<<<m,256,0,s>>>(c->d_sla, das, m, k, c->d_faA, c->d_saA, dA, c->rank);
  cudaEventRecord(c->p1, s);
}

static int resident_run2(Ctx* c, uint64_t seed, uint64_t next_seed, int has_next,
                         const uint8_t job_key[32], int* hit_rows, int* hit_cols) {
  bool fixb = (getenv("ARIA_FIXB") != nullptr);
  // L'overlap n'a pas de jumeau cp.async (TMA only) → fallback classique sur Ampere/Ada.
  if (!(fixb && getenv("ARIA_TMA_MS") && c->d_A2 && aria_overlap())
      || aria_use_cpasync(((cudaDeviceProp*)c->prop)->major))
    return resident_run(c, seed, job_key, hit_rows, hit_cols);
  int m=c->m, n=c->n, k=c->k, max_hits=c->max_hits;
  // job change → set_key + reset pipeline (pending de l'ancien job jeté)
  if (!c->jk_set || memcmp(c->last_jk, job_key, 32) != 0) {
    cudaDeviceSynchronize(); set_key(job_key);
    for (int i=0;i<32;i++) c->last_jk[i]=job_key[i]; c->jk_set=true;
    c->bfilled=false; c->pend_valid=false;
  }
  int8_t*   dA_[2]  = { c->d_A,  c->d_A2 };
  uint32_t* dha_[2] = { c->d_ha, c->d_ha2 };
  uint32_t* das_[2] = { c->d_as, c->d_as2 };
  int slot; bool cold;
  cudaMemsetAsync(c->d_found,0,4,c->sA);
  if (c->pend_valid && c->pend_seed == seed && c->bfilled) {
    // ✓ prologue déjà préfetché pendant le grind précédent → grind direct
    slot = c->pend_slot; cold = false;
    cudaStreamWaitEvent(c->sA, c->ePro, 0);
    if (c->do_timing) {  // events du prefetch déjà complétés (fini pendant le grind N-1)
      cudaEventElapsedTime(&c->last_genc_ms, c->p0, c->pMid);
      cudaEventElapsedTime(&c->last_noise_ms, c->pMid, c->p1);
      c->last_pro_ms = c->last_genc_ms + c->last_noise_ms;
    }
  } else {
    // pipeline froid (1er setup du job) : prologue complet inline, B compris si besoin
    slot = 0; cold = true;
    if (c->do_timing) cudaEventRecord(c->tStart, c->sA);
    gen_signal_kernel<<<dim3(((k>>2)+255)/256,(unsigned)(m<32768?m:32768)),256,0,c->sA>>>(dA_[0], seed, 0, m, k);
    commit_nokey<256,2,256>((const uint8_t*)dA_[0],(uint32_t)((size_t)m*k),(uint8_t*)dha_[0],c->nbA,c->d_ra,c->sA);
    if (!c->bfilled) {
      gen_signal_kernel<<<dim3(((k>>2)+255)/256,(unsigned)(n<32768?n:32768)),256,0,c->sB>>>(c->d_B, seed, 1, n, k);
      commit_nokey<256,2,256>((const uint8_t*)c->d_B,(uint32_t)((size_t)n*k),(uint8_t*)c->d_hb,c->nbB,c->d_rb,c->sB);
    }
    cudaEventRecord(c->eB, c->sB);
    cudaStreamWaitEvent(c->sA, c->eB, 0);
    stir_kernel<<<1,1,0,c->sA>>>(dha_[0],c->d_hb,c->d_jk,das_[0],c->d_bs);
    cudaEventRecord(c->eStir, c->sA);
    if (c->do_timing) cudaEventRecord(c->tMid, c->sA);
    perm_k_<<<(k+255)/256,256,0,c->sA>>>(c->d_sla, das_[0], k, c->d_faA, c->d_saA, c->rank);
    noise_add_k_<<<m,256,0,c->sA>>>(c->d_sla, das_[0], m, k, c->d_faA, c->d_saA, dA_[0], c->rank);
    if (!c->bfilled) {
      cudaStreamWaitEvent(c->sB, c->eStir, 0);
      perm_k_<<<(k+255)/256,256,0,c->sB>>>(c->d_slb, c->d_bs, k, c->d_faB, c->d_saB, c->rank);
      noise_add_k_<<<n,256,0,c->sB>>>(c->d_slb, c->d_bs, n, k, c->d_faB, c->d_saB, c->d_B, c->rank);
    }
    cudaEventRecord(c->eB, c->sB);
    c->bfilled = true;
    cudaStreamWaitEvent(c->sA, c->eB, 0);
  }
  c->pend_valid = false;
  if (c->do_timing) cudaEventRecord(c->tPro, c->sA);
  // ---- grind multistage TMA sur sA, slot courant (mêmes types que resident_run) ----
  {
    int reduce_every_k = c->rank / 32;
    auto prob = make_shape(m, n, k);
    auto dA = make_stride(k, Int<1>{}); auto dB = make_stride(k, Int<1>{}); auto dC = make_stride(n, Int<1>{});
    auto bM2=Int<128>{}; auto bN2=Int<256>{}; auto bK2=Int<128>{};
    auto cta2 = make_shape(bM2,bN2,bK2);
    auto sAm = composition(Swizzle<3,4,3>{},
        Layout<Shape<Shape<_16,_8 >,Shape<_128,_1>,_2>, Stride<Stride<_128,Int<2048>>,Stride<_1,_0>,Int<16384>>>{});
    auto sBm = composition(Swizzle<3,4,3>{},
        Layout<Shape<Shape<_16,_16>,Shape<_128,_1>,_2>, Stride<Stride<_128,Int<2048>>,Stride<_1,_0>,Int<32768>>>{});
    auto sB1 = composition(Swizzle<3,4,3>{},
        Layout<Shape<Shape<_16,_16>,Shape<_128,_1>>, Stride<Stride<_128,Int<2048>>,Stride<_1,_0>>>{});
    auto sCm = make_layout(make_shape(bM2,bN2));
    using AtomG2 = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>;
    using TVcopy2 = Layout<Shape<Shape<_8,_32>,_16>, Stride<Stride<_512,_1>,_32>>;
    using TilerC2 = Shape<_32,_128>;
    TiledCopy<AtomG2,TVcopy2,TilerC2> copyA2;
    TiledMMA mma2 = make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{}, Layout<Shape<_2,_4,_1>,Stride<_1,_2,_0>>{}, Tile<_32,_32,_32>{});
    Copy_Atom<SM75_U32x4_LDSM_N,int8_t> s2rA2; Copy_Atom<SM75_U32x2_LDSM_N,int8_t> s2rB2;
    Tensor gB_t = make_tensor(make_gmem_ptr<int8_t>(c->d_B), make_layout(make_shape(n,k), make_stride(k,Int<1>{})));
    auto tma_b = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gB_t, sB1, make_shape(bN2,bK2), Int<1>{});
    int smem = int(sizeof(SharedStorageTMA_MS<int8_t,int8_t,decltype(sAm),decltype(sBm)>));
    dim3 grd2(size(ceil_div(m,bM2)), size(ceil_div(n,bN2))), blk2(size(mma2));
    if (c->big_endian) {
      auto kfn = gemm_device_tma_ms<decltype(prob),decltype(cta2),
          int8_t,decltype(dA),decltype(sAm),decltype(copyA2),decltype(s2rA2),
          int8_t,decltype(dB),decltype(tma_b),decltype(sBm),decltype(s2rB2),
          int32_t,decltype(dC),decltype(sCm),decltype(mma2), /*DumpC=*/false, /*BE=*/true>;
      cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      kfn<<<grd2,blk2,smem,c->sA>>>(prob,cta2, dA_[slot],dA,sAm,copyA2,s2rA2, c->d_B,dB,tma_b,sBm,s2rB2,
          c->d_C,dC,sCm,mma2, reduce_every_k, aria_swz_g(), das_[slot], c->d_bnd, c->d_found, c->d_hr,c->d_hc,max_hits);
    } else {
      auto kfn = gemm_device_tma_ms<decltype(prob),decltype(cta2),
          int8_t,decltype(dA),decltype(sAm),decltype(copyA2),decltype(s2rA2),
          int8_t,decltype(dB),decltype(tma_b),decltype(sBm),decltype(s2rB2),
          int32_t,decltype(dC),decltype(sCm),decltype(mma2), /*DumpC=*/false, /*BE=*/false>;
      cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      kfn<<<grd2,blk2,smem,c->sA>>>(prob,cta2, dA_[slot],dA,sAm,copyA2,s2rA2, c->d_B,dB,tma_b,sBm,s2rB2,
          c->d_C,dC,sCm,mma2, reduce_every_k, aria_swz_g(), das_[slot], c->d_bnd, c->d_found, c->d_hr,c->d_hc,max_hits);
    }
  }
  if (c->do_timing) cudaEventRecord(c->tEnd, c->sA);
  // ---- prefetch du prologue N+1 sur sP (slot opposé), recouvert par le grind ----
  if (has_next) {
    int nxt = 1 - slot;
    if (cold) cudaStreamWaitEvent(c->sP, c->eStir, 0);  // scratch commit d_ra libre
    prologue_A_slot(c, next_seed, dA_[nxt], dha_[nxt], das_[nxt], c->sP);
    cudaEventRecord(c->ePro, c->sP);
    c->pend_valid = true; c->pend_seed = next_seed; c->pend_slot = nxt;
  }
  CKR(cudaStreamSynchronize(c->sA)); CKR(cudaGetLastError());
  if (c->do_timing) {
    cudaEventElapsedTime(&c->last_grind_ms, c->tPro, c->tEnd);
    if (cold) {
      cudaEventElapsedTime(&c->last_pro_ms,   c->tStart, c->tPro);
      cudaEventElapsedTime(&c->last_genc_ms,  c->tStart, c->tMid);
      cudaEventElapsedTime(&c->last_noise_ms, c->tMid,   c->tPro);
    }
  }
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
  // noise_rank : env ARIA_RANK (défaut mainnet 128). ≤256 car first/second en uint8.
  c->rank = 128;
  if (const char* r = getenv("ARIA_RANK")) {
    int v = atoi(r);
    if (v >= 32 && v <= 256 && (v & (v-1)) == 0) c->rank = v;
    else { fprintf(stderr, "ARIA_RANK=%s invalide (puissance de 2, 32..256)\n", r); delete c; return nullptr; }
  }
  c->big_endian = (getenv("ARIA_BE") != nullptr);   // LuckyPool : pow-check big-endian
  c->prop = new cudaDeviceProp(); if(cudaGetDeviceProperties((cudaDeviceProp*)c->prop,0)){delete c; return nullptr;}
  const uint32_t tpb=256, chunk=1024;
  c->nbA=(((uint32_t)((size_t)m*k)+chunk-1)/chunk + tpb-1)/tpb;
  c->nbB=(((uint32_t)((size_t)n*k)+chunk-1)/chunk + tpb-1)/tpb;
  auto ok=[&](cudaError_t e){ return e==cudaSuccess; };
  // d_C est VESTIGIAL : le kernel 2×64 (et 8×16) accumule en registres (tCrC), folde dans
  // `transcript` et fait le pow-check — il n'écrit JAMAIS C en global. On alloue donc un
  // buffer minuscule (le pointeur par-CTA est calculé mais jamais déréférencé). Sans ça,
  // (size_t)m*n*4 explose à 131072² (68 Go) ; ici 1 Mo suffit pour TOUTE forme.
  bool good = ok(cudaMalloc(&c->d_A,(size_t)m*k)) && ok(cudaMalloc(&c->d_B,(size_t)n*k))
    && ok(cudaMalloc(&c->d_C,(size_t)1<<20))
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
  cudaEventCreate(&c->tStart); cudaEventCreate(&c->tMid); cudaEventCreate(&c->tPro); cudaEventCreate(&c->tEnd); // timés
  c->do_timing = false; c->last_pro_ms = 0.f; c->last_grind_ms = 0.f; c->last_genc_ms = 0.f; c->last_noise_ms = 0.f;
  c->jk_set = false;
  c->tile2x64 = false;   // défaut = grind 8×16 (v1.0 AriaPool)
  // v0.6.2 : pipeline overlap prologue — slot 2 (+m·k octets VRAM) + stream haute priorité.
  // Alloué seulement pour le chemin multistage (ARIA_TMA_MS) avec overlap actif ;
  // échec d'alloc = dégradé silencieux (d_A2=nullptr → fallback resident_run).
  c->d_A2=nullptr; c->d_ha2=nullptr; c->d_as2=nullptr; c->sP=nullptr;
  c->bfilled=false; c->pend_valid=false; c->pend_seed=0; c->pend_slot=0;
  if (getenv("ARIA_TMA_MS") && aria_overlap()) {
    if (ok(cudaMalloc(&c->d_A2,(size_t)m*k)) && ok(cudaMalloc(&c->d_ha2,32)) && ok(cudaMalloc(&c->d_as2,32))) {
      int lo=0, hi=0; cudaDeviceGetStreamPriorityRange(&lo,&hi);
      // ARIA_OVERLAP_PRIO=low → le prefetch REMPLIT les trous d'occupation du grind
      // (pas de préemption) ; défaut high → le prefetch passe devant (peut voler du SM).
      const char* pp = getenv("ARIA_OVERLAP_PRIO");
      int prio = (pp && pp[0]=='l') ? lo : hi;
      cudaStreamCreateWithPriority(&c->sP, cudaStreamNonBlocking, prio);
      cudaEventCreateWithFlags(&c->ePro, cudaEventDisableTiming);
      cudaEventCreate(&c->p0); cudaEventCreate(&c->pMid); cudaEventCreate(&c->p1);
    } else {
      if(c->d_A2) cudaFree(c->d_A2); if(c->d_ha2) cudaFree(c->d_ha2); if(c->d_as2) cudaFree(c->d_as2);
      c->d_A2=nullptr; c->d_ha2=nullptr; c->d_as2=nullptr;
      fprintf(stderr, "ariaminer: VRAM insuffisante pour l'overlap prologue — désactivé\n");
    }
  }
  return c;
}

// Variante AlphaPool (v1.1) : contexte résident qui grind en tuile 2×64.
void* pearl_resident_create_2x64(int m, int n, int k, int max_hits) {
  Ctx* c = (Ctx*)pearl_resident_create(m, n, k, max_hits);
  if (c) c->tile2x64 = true;
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

// v0.6.2 : grind avec OVERLAP PROLOGUE — `next_seed` (si has_next) = seed du setup
// suivant, dont le prologue est préfetché pendant ce grind. Fallback automatique
// resident_run hors chemin LuckyPool (ARIA_TMA_MS+ARIA_FIXB) ou si ARIA_OVERLAP=0.
int pearl_resident_grind2_ctx(void* ctx, uint64_t setup_seed, uint64_t next_seed, int has_next,
                              const uint8_t job_key[32], const uint8_t pow_bound_le[32],
                              int* hit_rows, int* hit_cols) {
  Ctx* c = (Ctx*)ctx;
  uint32_t jk[8], bnd[8]; b2w(job_key, jk, 8); b2w(pow_bound_le, bnd, 8);
  cudaMemcpy(c->d_jk, jk, 32, cudaMemcpyHostToDevice);
  cudaMemcpy(c->d_bnd, bnd, 32, cudaMemcpyHostToDevice);
  return resident_run2(c, setup_seed, next_seed, has_next, job_key, hit_rows, hit_cols);
}

// --- instrumentation (v0.5.0) ---
void pearl_resident_set_timing(void* ctx, int on) { ((Ctx*)ctx)->do_timing = (on != 0); }
void pearl_resident_last_times(void* ctx, float* prologue_ms, float* grind_ms) {
  Ctx* c = (Ctx*)ctx;
  if (prologue_ms) *prologue_ms = c->last_pro_ms;
  if (grind_ms)    *grind_ms    = c->last_grind_ms;
}
// Split fin : gen+commit+stir / noise / grind (ms).
void pearl_resident_last_times4(void* ctx, float* genc_ms, float* noise_ms, float* grind_ms) {
  Ctx* c = (Ctx*)ctx;
  if (genc_ms)  *genc_ms  = c->last_genc_ms;
  if (noise_ms) *noise_ms = c->last_noise_ms;
  if (grind_ms) *grind_ms = c->last_grind_ms;
}

void pearl_resident_destroy(void* ctx) {
  Ctx* c = (Ctx*)ctx; if(!c) return;
  cudaEventDestroy(c->tStart); cudaEventDestroy(c->tMid); cudaEventDestroy(c->tPro); cudaEventDestroy(c->tEnd);
  cudaFree(c->d_A);cudaFree(c->d_B);cudaFree(c->d_C);cudaFree(c->d_jk);cudaFree(c->d_bnd);
  cudaFree(c->d_ha);cudaFree(c->d_hb);cudaFree(c->d_as);cudaFree(c->d_bs);cudaFree(c->d_sla);cudaFree(c->d_slb);
  cudaFree(c->d_faA);cudaFree(c->d_saA);cudaFree(c->d_faB);cudaFree(c->d_saB);
  cudaFree(c->d_found);cudaFree(c->d_hr);cudaFree(c->d_hc);cudaFree(c->d_ra);cudaFree(c->d_rb);
  cudaStreamDestroy(c->sA); cudaStreamDestroy(c->sB);
  cudaEventDestroy(c->eA); cudaEventDestroy(c->eB); cudaEventDestroy(c->eStir);
  if (c->d_A2) {   // v0.6.2 : pipeline overlap
    cudaFree(c->d_A2); cudaFree(c->d_ha2); cudaFree(c->d_as2);
    cudaStreamDestroy(c->sP);
    cudaEventDestroy(c->ePro); cudaEventDestroy(c->p0); cudaEventDestroy(c->pMid); cudaEventDestroy(c->p1);
  }
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
  int grind_blocks;   // v0.5.0 : 0 = grille pleine (gemm_device) ; >0 = persistant G blocs (gemm_device_persist, occupation bridée → overlap)
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
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,(unsigned)(m<32768?m:32768)),256,0,s>>>(c->d_A[slot], seed, 0, m, k);
  gen_signal_kernel<<<dim3(((k>>2)+255)/256,(unsigned)(n<32768?n:32768)),256,0,s>>>(c->d_B[slot], seed, 1, n, k);
  // commit SANS set_key (clé posée 1× par batch) → overlap réel des streams
  commit_nokey<256,2,256>((const uint8_t*)c->d_A[slot],(uint32_t)((size_t)m*k),(uint8_t*)c->d_ha[slot],c->nbA,c->d_ra[slot],s);
  commit_nokey<256,2,256>((const uint8_t*)c->d_B[slot],(uint32_t)((size_t)n*k),(uint8_t*)c->d_hb[slot],c->nbB,c->d_rb[slot],s);
  stir_kernel<<<1,1,0,s>>>(c->d_ha[slot],c->d_hb[slot],c->d_jk,c->d_as[slot],c->d_bs[slot]);
  // noise_rank : même env ARIA_RANK que le chemin résident (lu 1×, défaut 128).
  static const int rank = [](){ const char* r = getenv("ARIA_RANK"); int v = r ? atoi(r) : 128;
    return (v >= 32 && v <= 256 && (v & (v-1)) == 0) ? v : 128; }();
  perm_k_<<<(k+255)/256,256,0,s>>>(c->d_sla, c->d_as[slot], k, c->d_faA[slot], c->d_saA[slot], rank);
  perm_k_<<<(k+255)/256,256,0,s>>>(c->d_slb, c->d_bs[slot], k, c->d_faB[slot], c->d_saB[slot], rank);
  noise_add_k_<<<m,256,0,s>>>(c->d_sla, c->d_as[slot], m, k, c->d_faA[slot], c->d_saA[slot], c->d_A[slot], rank);
  noise_add_k_<<<n,256,0,s>>>(c->d_slb, c->d_bs[slot], n, k, c->d_faB[slot], c->d_saB[slot], c->d_B[slot], rank);
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
  int nbx=size(ceil_div(m,bM)), nby=size(ceil_div(n,bN));
  dim3 blk(size(mma));
  if (getenv("ARIA_TMA_MS")) {
    // étape 1 dans le pipeline overlap : grind multistage TMA 128×256 (basse occupation).
    // teste si le prologue du slot concurrent se recouvre enfin (l'angle neuf de l'idée 1).
    auto bM2=Int<128>{}; auto bN2=Int<256>{}; auto bK2=Int<128>{};
    auto cta2=make_shape(bM2,bN2,bK2);
    auto sAm=composition(Swizzle<3,4,3>{},Layout<Shape<Shape<_16,_8 >,Shape<_128,_1>,_2>,Stride<Stride<_128,Int<2048>>,Stride<_1,_0>,Int<16384>>>{});
    auto sBm=composition(Swizzle<3,4,3>{},Layout<Shape<Shape<_16,_16>,Shape<_128,_1>,_2>,Stride<Stride<_128,Int<2048>>,Stride<_1,_0>,Int<32768>>>{});
    auto sB1=composition(Swizzle<3,4,3>{},Layout<Shape<Shape<_16,_16>,Shape<_128,_1>>,Stride<Stride<_128,Int<2048>>,Stride<_1,_0>>>{});
    auto sCm=make_layout(make_shape(bM2,bN2));
    using AtomG2=Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>,int8_t>;
    TiledCopy<AtomG2,Layout<Shape<Shape<_8,_32>,_16>,Stride<Stride<_512,_1>,_32>>,Shape<_32,_128>> copyA2;
    TiledMMA mma2=make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},Layout<Shape<_2,_4,_1>,Stride<_1,_2,_0>>{},Tile<_32,_32,_32>{});
    Copy_Atom<SM75_U32x4_LDSM_N,int8_t> s2rA2; Copy_Atom<SM75_U32x2_LDSM_N,int8_t> s2rB2;
    Tensor gB_t=make_tensor(make_gmem_ptr<int8_t>(c->d_B[slot]),make_layout(make_shape(n,k),make_stride(k,Int<1>{})));
    auto tma_b=make_tma_copy<int8_t>(SM90_TMA_LOAD{},gB_t,sB1,make_shape(bN2,bK2),Int<1>{});
    int smem2=int(sizeof(SharedStorageTMA_MS<int8_t,int8_t,decltype(sAm),decltype(sBm)>));
    auto kfn=gemm_device_tma_ms<decltype(prob),decltype(cta2),int8_t,decltype(dA),decltype(sAm),decltype(copyA2),decltype(s2rA2),int8_t,decltype(dB),decltype(tma_b),decltype(sBm),decltype(s2rB2),int32_t,decltype(dC),decltype(sCm),decltype(mma2),false>;
    cudaFuncSetAttribute(kfn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem2);
    dim3 grd2(size(ceil_div(m,bM2)),size(ceil_div(n,bN2))),blk2(size(mma2));
    kfn<<<grd2,blk2,smem2,s>>>(prob,cta2,c->d_A[slot],dA,sAm,copyA2,s2rA2,c->d_B[slot],dB,tma_b,sBm,s2rB2,c->d_C,dC,sCm,mma2,rek,aria_swz_g(),c->d_as[slot],c->d_bnd[slot],c->d_found[slot],c->d_hr[slot],c->d_hc[slot],64);
  } else if (c->grind_blocks > 0) {
    // v0.5.0 : grind PERSISTANT à occupation bridée (G blocs grid-stride) → laisse
    // des SM libres pour que le prologue du slot suivant tourne en concurrence.
    auto kfn=gemm_device_persist<decltype(prob),decltype(cta),int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),int8_t,decltype(dB),decltype(sB),decltype(copyB),decltype(s2rB),int32_t,decltype(dC),decltype(sC),decltype(mma)>;
    cudaFuncSetAttribute(kfn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
    kfn<<<dim3(c->grind_blocks),blk,smem,s>>>(prob,cta,c->d_A[slot],dA,sA,copyA,s2rA,c->d_B[slot],dB,sB,copyB,s2rB,c->d_C,dC,sC,mma,rek,c->d_as[slot],c->d_bnd[slot],c->d_found[slot],c->d_hr[slot],c->d_hc[slot],64,nbx,nby);
  } else {
    dim3 grd(nbx,nby);
    auto kfn=gemm_device<decltype(prob),decltype(cta),int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),int8_t,decltype(dB),decltype(sB),decltype(copyB),decltype(s2rB),int32_t,decltype(dC),decltype(sC),decltype(mma)>;
    cudaFuncSetAttribute(kfn,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
    kfn<<<grd,blk,smem,s>>>(prob,cta,c->d_A[slot],dA,sA,copyA,s2rA,c->d_B[slot],dB,sB,copyB,s2rB,c->d_C,dC,sC,mma,rek,c->d_as[slot],c->d_bnd[slot],c->d_found[slot],c->d_hr[slot],c->d_hc[slot],64);
  }
}

#define MK(p,sz) do{ if(cudaMalloc((void**)&(p),(sz))!=cudaSuccess){return nullptr;} }while(0)
extern "C" {
void* pearl_resident2_create(int m,int n,int k){
  Ctx2* c=new Ctx2(); c->m=m;c->n=n;c->k=k; c->grind_blocks=0;
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
// v0.5.0 : règle la grille du grind persistant. 0 = grille pleine (défaut).
void pearl_resident2_set_grind_blocks(void* ctx, int g){ ((Ctx2*)ctx)->grind_blocks = g; }
// Nombre de SM du GPU 0 (pour calibrer G = headroom).
int pearl_gpu_sm_count(){ cudaDeviceProp p; if(cudaGetDeviceProperties(&p,0)) return 0; return p.multiProcessorCount; }

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
