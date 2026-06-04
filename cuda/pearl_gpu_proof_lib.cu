// Aria GPU miner v2.0 — preuve Merkle FULL GPU (réplique de gpu_plain_proof_cu d'alpha).
//
// Objectif : sortir le chemin d'emballage de preuve du CPU. Avant, sur chaque hit le
// CPU régénérait 67M int7 (signal) + reconstruisait 2 arbres blake3 de 33 Mo (~300ms/preuve
// → 12 builders plafonnés à ~25 preuves/s). Ici TOUT le hachage est sur GPU :
//   - hash_leaves_kernel  : 1 thread/feuille → chunk_cv keyé (CV=key, 16 blocs, CHUNK_START/END,
//                           counter=chunk_idx) — copie EXACTE de compress_block officiel.
//   - parent_level_kernel : combine une couche → la suivante (parent_cv = INNER_NODE, ou
//                           root_cv = ROOT au dernier niveau) — miroir de MerkleTree::new.
//   - on GARDE toutes les couches en VRAM (alpha fait pareil), puis on copie côté hôte les
//     couches + les chunks de feuilles demandés. Le CPU ne fait QUE le walk d'indices (µs).
//
// Bit-exact garanti : mêmes primitives que le commit déjà validé (BRIQUE 2) — racine GPU
// == MerkleTree::new CPU. Build : nvcc -arch=sm_120a -I cuda/shim -I<csrc> -I<cutlass>/include.
#include <cstdint>
#include <cstdio>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"
using namespace cute;

// Noms de kernels uniques (proof_*, hash_leaves_kernel, parent_level_kernel) → pas de collision
// inter-TU ; les helpers sont `static` (linkage interne). Pas de namespace anonyme (incompatible
// avec la génération de stub cudafe quand cute ouvre ses propres `namespace {}`).

// signal int7 reproductible — MÊME formule que pearl_resident_lib.cu::int7_at et Rust::int7_at.
static __host__ __device__ __forceinline__ int8_t proof_int7_at(uint64_t seed, uint32_t sel,
                                                          uint32_t i, uint32_t j) {
  uint64_t x = seed ^ ((uint64_t)sel << 62) ^ ((uint64_t)i << 32) ^ (uint64_t)j;
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x ^= (x >> 31);
  return (int8_t)((int)(x & 0x7F) - 64);
}

// Régénère la matrice signal `sel` (rows×k) dans `mat` (octets). Grille 2D, écriture 4 octets.
__global__ void proof_gen_signal(int8_t* mat, uint64_t seed, uint32_t sel,
                                 uint32_t rows, uint32_t k) {
  uint32_t i = blockIdx.y;
  if (i >= rows) return;
  uint32_t* row4 = (uint32_t*)(mat + (size_t)i * k);
  uint32_t k4 = k >> 2;
  for (uint32_t q = blockIdx.x * blockDim.x + threadIdx.x; q < k4; q += gridDim.x * blockDim.x) {
    uint32_t j = q << 2;
    uint8_t b0 = (uint8_t)proof_int7_at(seed, sel, i, j);
    uint8_t b1 = (uint8_t)proof_int7_at(seed, sel, i, j + 1);
    uint8_t b2 = (uint8_t)proof_int7_at(seed, sel, i, j + 2);
    uint8_t b3 = (uint8_t)proof_int7_at(seed, sel, i, j + 3);
    row4[q] = (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
  }
}

// chunk_cv keyé d'un chunk de 1024 octets (256 mots) — non-root.
// = Blake3Hasher::chunk_cv(data, chunk_idx) : CV init = key, 16 blocs chaînés,
//   counter=chunk_idx, CHUNK_START sur bloc 0, CHUNK_END sur bloc 15, flags KEYED_HASH.
static __device__ __forceinline__ void leaf_chunk_cv(const uint32_t* cw, uint64_t chunk_idx,
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

// 1 thread = 1 feuille. data = octets signal (rows×k), num_leaves = rows*k/1024.
__global__ void hash_leaves_kernel(const uint8_t* data, const uint32_t* key,
                                   uint32_t* out_leaves, uint64_t num_leaves) {
  uint64_t c = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_leaves) return;
  const uint32_t* cw = (const uint32_t*)(data + c * 1024ull);
  uint32_t cv[8];
  leaf_chunk_cv(cw, c, key, cv);
  for (int i = 0; i < 8; ++i) out_leaves[c * 8 + i] = cv[i];
}

// merge deux CV → parent. is_root=false → parent_cv (INNER_NODE), true → root_cv (ROOT).
// = merge_subtrees_non_root / merge_subtrees_root keyé : CV init = key, bloc = left‖right,
//   counter=0, block_len=64, flags = KEYED_HASH|PARENT (+ROOT si root).
static __device__ __forceinline__ void merge_cv(const uint32_t l[8], const uint32_t r[8],
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

// Combine la couche `in` (in_count nœuds) → `out` (ceil(in_count/2) nœuds).
// Paire (2p,2p+1) → merge ; nœud impair seul → recopié (= MerkleTree combine_layer).
__global__ void parent_level_kernel(const uint32_t* key, uint32_t* out, const uint32_t* in,
                                    uint64_t in_count, bool is_root) {
  uint64_t p = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t n_out = (in_count + 1) / 2;
  if (p >= n_out) return;
  uint32_t l[8], o[8];
  for (int i = 0; i < 8; ++i) l[i] = in[(2 * p) * 8 + i];
  if (2 * p + 1 < in_count) {
    uint32_t r[8];
    for (int i = 0; i < 8; ++i) r[i] = in[(2 * p + 1) * 8 + i];
    merge_cv(l, r, key, is_root, o);
    for (int i = 0; i < 8; ++i) out[p * 8 + i] = o[i];
  } else {
    for (int i = 0; i < 8; ++i) out[p * 8 + i] = l[i];  // report du nœud impair
  }
}

// Ramasse les chunks bruts (1024 o) des feuilles demandées : out[li] = data[leaf_idx[li]*1024..].
__global__ void gather_leaves_kernel(const uint8_t* data, const long* leaf_idx, int n,
                                     uint8_t* out) {
  int li = blockIdx.x;
  if (li >= n) return;
  const uint8_t* src = data + (size_t)leaf_idx[li] * 1024ull;
  uint8_t* dst = out + (size_t)li * 1024ull;
  for (int t = threadIdx.x; t < 1024; t += blockDim.x) dst[t] = src[t];
}

struct ProofCtx {
  int max_leaves;        // capacité (feuilles)
  int max_leafdata;      // capacité (chunks de feuilles demandés)
  int8_t* d_signal;      // max_leaves*1024 octets
  uint32_t* d_layers;    // tous les nœuds (≤ 2*max_leaves) × 8 mots
  uint32_t* d_key;       // 8 mots (job_key LE)
  long* d_leafidx;       // indices de feuilles demandés
  uint8_t* d_leafdata;   // chunks ramassés
  cudaStream_t s;
};

static void b2w(const uint8_t* b, uint32_t* w, int n) {
  for (int i = 0; i < n; i++)
    w[i] = (uint32_t)b[i*4] | ((uint32_t)b[i*4+1]<<8) | ((uint32_t)b[i*4+2]<<16) | ((uint32_t)b[i*4+3]<<24);
}

#define PKR(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"proof CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return -1;}}while(0)

extern "C" {

// Alloue un contexte de preuve réutilisable. max_rows = max(m,n), k = common_dim.
// max_leafdata = nombre max de chunks de feuilles révélés (A:2 lignes=8, B:64 cols=256 → 256 large).
void* pearl_proof_create(int max_rows, int k, int max_leafdata) {
  ProofCtx* c = new ProofCtx();
  long max_leaves = ((long)max_rows * k) / 1024;
  c->max_leaves = (int)max_leaves;
  c->max_leafdata = max_leafdata;
  auto ok = [&](cudaError_t e){ return e == cudaSuccess; };
  bool good =
      ok(cudaMalloc(&c->d_signal, (size_t)max_leaves * 1024)) &&
      ok(cudaMalloc(&c->d_layers, (size_t)(2 * max_leaves + 8) * 8 * sizeof(uint32_t))) &&
      ok(cudaMalloc(&c->d_key, 8 * sizeof(uint32_t))) &&
      ok(cudaMalloc(&c->d_leafidx, (size_t)max_leafdata * sizeof(long))) &&
      ok(cudaMalloc(&c->d_leafdata, (size_t)max_leafdata * 1024));
  if (!good) { delete c; return nullptr; }
  cudaStreamCreate(&c->s);
  return c;
}

// Construit l'arbre Merkle keyé de la matrice signal `sel` (rows×k, regénérée depuis setup_seed,
// clé job_key) en GARDANT toutes les couches, puis copie côté hôte :
//   - out_leaf_data  : les n_leaves chunks bruts (1024 o) des feuilles `leaf_indices`
//   - out_layers     : TOUS les nœuds, octets, couche par couche (32 o/nœud) — pour le walk Rust
//   - out_layer_lens : nb de nœuds par couche  ;  *out_num_layers
//   - out_root       : la racine (= dernier nœud)
// Renvoie 0 si OK. Toute la cryptographie (chunk_cv + arbre) est sur GPU.
int pearl_proof_build(void* ctx, uint64_t setup_seed, int sel, int rows, int k,
                      const uint8_t job_key[32],
                      const long* leaf_indices, int n_leaves,
                      uint8_t* out_leaf_data,
                      uint8_t* out_layers, long* out_layer_lens, int* out_num_layers,
                      uint8_t out_root[32]) {
  ProofCtx* c = (ProofCtx*)ctx;
  if (((long)rows * k) % 1024 != 0) { fprintf(stderr, "proof: rows*k non multiple de 1024\n"); return -1; }
  long num_leaves = ((long)rows * k) / 1024;
  if (num_leaves < 2 || num_leaves > c->max_leaves) { fprintf(stderr, "proof: num_leaves hors capacité\n"); return -1; }
  if (n_leaves < 1 || n_leaves > c->max_leafdata) { fprintf(stderr, "proof: n_leaves hors capacité\n"); return -1; }
  cudaStream_t s = c->s;

  uint32_t kw[8]; b2w(job_key, kw, 8);
  PKR(cudaMemcpyAsync(c->d_key, kw, 32, cudaMemcpyHostToDevice, s));

  // 1) régénère le signal (sel=0 → A row-major, sel=1 → Bᵀ)
  proof_gen_signal<<<dim3(((k>>2)+255)/256, rows), 256, 0, s>>>(c->d_signal, setup_seed, (uint32_t)sel, rows, k);

  // 2) feuilles : 1 thread/feuille
  int tpb = 256;
  hash_leaves_kernel<<<(int)((num_leaves + tpb - 1) / tpb), tpb, 0, s>>>(
      (const uint8_t*)c->d_signal, c->d_key, c->d_layers, (uint64_t)num_leaves);

  // 3) couches parentes — miroir EXACT de MerkleTree::new :
  //    combine tant que la couche courante > 2 (parent_cv) ; la dernière paire (==2) → root_cv.
  long cur_off = 0, cur_s = num_leaves;
  out_layer_lens[0] = num_leaves;
  int nl = 1;
  while (cur_s > 1) {
    long next_off = cur_off + cur_s;
    long next_s = (cur_s + 1) / 2;
    bool is_root = (cur_s == 2);
    parent_level_kernel<<<(int)((next_s + tpb - 1) / tpb), tpb, 0, s>>>(
        c->d_key, c->d_layers + next_off * 8, c->d_layers + cur_off * 8, (uint64_t)cur_s, is_root);
    out_layer_lens[nl++] = next_s;
    cur_off = next_off;
    cur_s = next_s;
  }
  *out_num_layers = nl;
  long total_nodes = cur_off + cur_s;  // + dernière couche (taille 1)

  // 4) ramasse les chunks de feuilles demandés
  PKR(cudaMemcpyAsync(c->d_leafidx, leaf_indices, (size_t)n_leaves * sizeof(long), cudaMemcpyHostToDevice, s));
  gather_leaves_kernel<<<n_leaves, 256, 0, s>>>((const uint8_t*)c->d_signal, c->d_leafidx, n_leaves, c->d_leafdata);

  // 5) copies retour : leaf_data + tous les nœuds + racine
  PKR(cudaMemcpyAsync(out_leaf_data, c->d_leafdata, (size_t)n_leaves * 1024, cudaMemcpyDeviceToHost, s));
  PKR(cudaMemcpyAsync(out_layers, c->d_layers, (size_t)total_nodes * 32, cudaMemcpyDeviceToHost, s));
  PKR(cudaMemcpyAsync(out_root, (const uint8_t*)(c->d_layers + cur_off * 8), 32, cudaMemcpyDeviceToHost, s));
  PKR(cudaStreamSynchronize(s));
  PKR(cudaGetLastError());
  return 0;
}

void pearl_proof_destroy(void* ctx) {
  ProofCtx* c = (ProofCtx*)ctx;
  if (!c) return;
  cudaFree(c->d_signal); cudaFree(c->d_layers); cudaFree(c->d_key);
  cudaFree(c->d_leafidx); cudaFree(c->d_leafdata);
  cudaStreamDestroy(c->s);
  delete c;
}

}  // extern "C"
