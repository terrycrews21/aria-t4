// ariaminer v0.4.0 — Phase 1a : GEMM int8 IMMA + cp.async en CuTe (recette alpha).
// Base = gemm_device générique du tutoriel CuTe sm80 (cp.async pipeliné), atome MMA
// remplacé par SM80_16x8x32_S32S8S8S32_TN (= notre IMMA Blackwell sm_120). But : battre
// les 22 TOPS du kernel hand-roll (bloqué par L1/loads scalaires). Épilogue jackpot = Phase 2.
// Build : nvcc -arch=sm_120a -O3 -std=c++17 -I<cutlass>/include --expt-relaxed-constexpr
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include "pearl_fold.cuh"        // primitives jackpot portées de l'officiel
#include "blake3/blake3.cuh"     // blake3 keyed officiel (consensus) — header-only portable

using namespace cute;

template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value)
void gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    uint32_t* transcript_out, int reduce_every_k, int* coords_out,
    const uint32_t* pow_key, const uint32_t* pow_bound, int* found_count) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  extern __shared__ char smem_[];
  using SS = SharedStorage<TA,TB,ASmemLayout,BSmemLayout>;
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
    cp_async_fence();
    --k_tile_count; if (k_tile_count > 0) ++k_tile_next;
  }

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  // === ÉPILOGUE JACKPOT (Phase 2) : transcript par thread, fold périodique ===
  uint32_t transcript[pearl_fold::JACKPOT_SIZE];
  for (int ti=0; ti<pearl_fold::JACKPOT_SIZE; ++ti) transcript[ti]=0;
  int gk = 0;  // compteur global de k-blocks (pour la fréquence de réduction R/32)

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
    cp_async_wait<K_PIPE_MAX-2>(); __syncthreads();
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
      // fold jackpot tous les reduce_every_k k-blocks (= R=128 K) — recette officielle
      if ((++gk % reduce_every_k) == 0) {
        uint32_t h = pearl_fold::xor_reduction(tCrC);
        int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
        transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(transcript[idx], h);
      }
    }
  }
  // POW-CHECK : blake3 keyed du transcript → comparaison au bound. CHAQUE thread = 1 candidat (tuile 8×16).
  {
    auto msg = make_tensor<uint32_t>(Int<16>{});
    for (int i=0;i<16;i++) msg(i)=transcript[i];
    auto cv = make_tensor<uint32_t>(Int<8>{});
    for (int i=0;i<8;i++) cv(i)=pow_key[i];
    blake3::compress_msg_block_u32(msg, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
    bool found = true;                         // hash <= bound ? (uint256, MSW=7 -> LSW=0)
    for (int i=7;i>=0;--i){ if(cv(i)>pow_bound[i]){found=false;break;} if(cv(i)<pow_bound[i])break; }
    if (found) atomicAdd(found_count, 1);
  }

  // smoke test : thread 0 de chaque CTA écrit son transcript[16]
  if (threadIdx.x == 0) {
    int blk = blockIdx.x * gridDim.y + blockIdx.y;
    for (int ti=0; ti<pearl_fold::JACKPOT_SIZE; ++ti)
      transcript_out[(size_t)blk*pearl_fold::JACKPOT_SIZE + ti] = transcript[ti];
  }
  // VALIDATION : thread 0 du CTA (0,0) dumpe les coords (m,n) de son fragment accu
  if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) {
    auto cD = make_identity_tensor(make_shape(get<0>(cta_tiler), get<1>(cta_tiler)));
    auto tCcD = thr_mma.partition_C(cD);
    int cnt = size(tCcD);
    coords_out[0] = cnt;
    for (int i=0;i<cnt;++i){ coords_out[1+2*i]=get<0>(tCcD(i)); coords_out[2+2*i]=get<1>(tCcD(i)); }
  }
}

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

int main(int argc, char** argv){
  int M = (argc>1)?atoi(argv[1]):4096;
  int N = (argc>2)?atoi(argv[2]):4096;
  int K = 4096;
  auto prob = make_shape(M, N, K);
  auto dA = make_stride(K, Int<1>{});   // A (M,K) row-major
  auto dB = make_stride(K, Int<1>{});   // B (N,K) row-major
  auto dC = make_stride(N, Int<1>{});   // C (M,N) row-major

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<64>{};
  auto bP = Int<3>{};
  auto cta = make_shape(bM, bN, bK);
  // ===== RECETTE EXACTE D'ALPHA (démanglée de son binaire sm_120) — transcription 1:1 =====
  // smem : ComposedLayout<Swizzle<2,4,3>, _0, Layout<((16,8),(64,1),(1,3)),((64,1024),(1,0),(0,8192))>>
  using SmemBase = Layout<Shape <Shape <_16,_8>,        Shape <_64,_1>, Shape <_1,_3>>,
                          Stride<Stride<_64,Int<1024>>, Stride<_1,_0>,  Stride<_0,Int<8192>>>>;
  auto sA = composition(Swizzle<2,4,3>{}, SmemBase{});
  auto sB = composition(Swizzle<2,4,3>{}, SmemBase{});
  auto sC = make_layout(make_shape(bM, bN));

  // cp.async : TiledCopy<atom, TV-layout, tiler> EXACT d'alpha
  using AtomG  = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>;
  using TVcopy = Layout<Shape <Shape <_4,_32>,       _16>,
                        Stride<Stride<Int<512>,_1>,  _32>>;
  using TilerC = Shape<_32,_64>;
  TiledCopy<AtomG, TVcopy, TilerC> copyA;
  TiledCopy<AtomG, TVcopy, TilerC> copyB;

  // TiledMMA : atome IMMA + warp-layout 2×2 + permutation Tile 32×32×32 (= alpha)
  TiledMMA mma = make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},
      Layout<Shape<_2,_2,_1>, Stride<_1,_2,_0>>{}, Tile<_32,_32,_32>{});
  // smem->reg : ldmatrix (= alpha)
  Copy_Atom<SM75_U32x4_LDSM_N, int8_t> s2rA;
  Copy_Atom<SM75_U32x4_LDSM_N, int8_t> s2rB;

  int smem = int(sizeof(SharedStorage<int8_t,int8_t,decltype(sA),decltype(sB)>));
  dim3 blk(size(mma)), grd(size(ceil_div(M,bM)), size(ceil_div(N,bN)));

  std::vector<int8_t> a((size_t)M*K), b((size_t)N*K);
  auto sm=[](uint64_t x){x+=0x9E3779B97F4A7C15ULL;x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;
                         x=(x^(x>>27))*0x94D049BB133111EBULL;return x^(x>>31);};
  for(size_t i=0;i<a.size();i++) a[i]=(int8_t)((int)(sm((i<<1)|1ULL)&0x7F)-64);
  for(size_t i=0;i<b.size();i++) b[i]=(int8_t)((int)(sm((i<<1)^0xABCDEFULL)&0x7F)-64);

  int n_blocks = (int)(size(ceil_div(M,bM)) * size(ceil_div(N,bN)));
  int reduce_every_k = 128 / 32;   // R / MMAAtom_K = 4 (fold tous les R=128 K)
  int8_t *da,*db; int32_t *dc; uint32_t *dtr; int *dco;
  CK(cudaMalloc(&da,a.size())); CK(cudaMalloc(&db,b.size())); CK(cudaMalloc(&dc,(size_t)M*N*4));
  CK(cudaMalloc(&dtr,(size_t)n_blocks*16*4));
  CK(cudaMalloc(&dco, (1+2*512)*4));   // coords du fragment thread0 (count + paires)
  uint32_t *dkey,*dbnd; int *dfound;
  CK(cudaMalloc(&dkey,32)); CK(cudaMalloc(&dbnd,32)); CK(cudaMalloc(&dfound,4));
  uint32_t key[8]; for(int i=0;i<8;i++) key[i]=0x6a09e667u+i;   // clé bidon (= a_noise_seed plus tard)
  CK(cudaMemcpy(dkey,key,32,cudaMemcpyHostToDevice));
  uint32_t bnd_easy[8]; for(int i=0;i<8;i++) bnd_easy[i]=0xFFFFFFFFu;  // tout passe
  CK(cudaMemcpy(dbnd,bnd_easy,32,cudaMemcpyHostToDevice));
  CK(cudaMemset(dfound,0,4));
  CK(cudaMemcpy(da,a.data(),a.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db,b.data(),b.size(),cudaMemcpyHostToDevice));

  auto kfn = gemm_device<decltype(prob),decltype(cta),
      int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),
      int8_t,decltype(dB),decltype(sB),decltype(copyB),decltype(s2rB),
      int32_t,decltype(dC),decltype(sC),decltype(mma)>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  kfn<<<grd,blk,smem>>>(prob,cta, da,dA,sA,copyA,s2rA,
      db,dB,sB,copyB,s2rB, dc,dC,sC,mma, dtr, reduce_every_k, dco, dkey, dbnd, dfound);
  CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
  float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
  int found_easy=0; CK(cudaMemcpy(&found_easy,dfound,4,cudaMemcpyDeviceToHost));
  int total_threads = n_blocks * (int)blk.x;
  // 2e run : bound dur (tout à 0) → aucun candidat (valide le comparateur)
  uint32_t bnd_hard[8]={0,0,0,0,0,0,0,0};
  CK(cudaMemcpy(dbnd,bnd_hard,32,cudaMemcpyHostToDevice)); CK(cudaMemset(dfound,0,4));
  kfn<<<grd,blk,smem>>>(prob,cta, da,dA,sA,copyA,s2rA,
      db,dB,sB,copyB,s2rB, dc,dC,sC,mma, dtr, reduce_every_k, dco, dkey, dbnd, dfound);
  CK(cudaDeviceSynchronize()); int found_hard=0; CK(cudaMemcpy(&found_hard,dfound,4,cudaMemcpyDeviceToHost));

  std::vector<uint32_t> tr((size_t)n_blocks*16); CK(cudaMemcpy(tr.data(),dtr,(size_t)n_blocks*16*4,cudaMemcpyDeviceToHost));
  std::vector<int> co(1+2*512); CK(cudaMemcpy(co.data(),dco,(1+2*512)*4,cudaMemcpyDeviceToHost));
  double tops = 2.0*(double)M*N*K/1e9/ms;
  printf("=== CuTe int8 GEMM + ÉPILOGUE JACKPOT (Phase 2) M=%d N=%d K=%d ===\n", M,N,K);
  printf("smem=%d o  grid=(%d,%d) blk=%d  reduce_every_k=%d  %.3f ms  %.1f TOPS\n",
         smem, grd.x,grd.y, blk.x, reduce_every_k, ms, tops);

  // ===== VALIDATION CONSENSUS : recalcul CPU du transcript attendu pour le fragment du thread 0 =====
  int cnt = co[0];
  printf("\n[VALIDATION] thread0(CTA0) possède %d cellules d'accu.\n", cnt);
  // structure des cellules : lignes & cols distinctes
  std::vector<int> rows, cols;
  for(int i=0;i<cnt;i++){ int m=co[1+2*i], n=co[2+2*i];
    if(std::find(rows.begin(),rows.end(),m)==rows.end()) rows.push_back(m);
    if(std::find(cols.begin(),cols.end(),n)==cols.end()) cols.push_back(n); }
  printf("[VALIDATION] cellules couvrent %zu lignes distinctes × %zu colonnes distinctes\n", rows.size(), cols.size());

  // transcript attendu : fold prefix-XOR sur CES cellules, tous les R=128, rotl13
  auto rotl=[](uint32_t x,int s){return (x<<s)|(x>>(32-s));};
  uint32_t exp[16]={0};
  int R=128, nb=K/R;          // 32 frontières
  for(int bnd=0;bnd<nb;bnd++){ int ll=(bnd+1)*R;
    uint32_t h=0;
    for(int c=0;c<cnt;c++){ int m=co[1+2*c], n=co[2+2*c];
      int32_t acc=0; for(int l=0;l<ll;l++) acc+=(int32_t)a[(size_t)m*K+l]*(int32_t)b[(size_t)n*K+l];
      h ^= (uint32_t)acc; }
    int idx=bnd%16; exp[idx]=rotl(exp[idx],13)^h;
  }
  // tr du bloc 0 = tr[0..15]
  int bad=0; for(int i=0;i<16;i++) if(exp[i]!=tr[i]) bad++;
  printf("[VALIDATION] GPU  :"); for(int i=0;i<16;i++) printf(" %08x",tr[i]); printf("\n");
  printf("[VALIDATION] CPU  :"); for(int i=0;i<16;i++) printf(" %08x",exp[i]); printf("\n");
  printf(bad==0 ? "✅ BIT-EXACT : le fold GPU reproduit l'algo officiel (prefix-XOR + rotl13) au bit près.\n"
                : "❌ %d/16 mots divergent — port à corriger.\n", bad);

  // ===== POW-CHECK (blake3 keyed officiel + compare bound) =====
  printf("\n[POW] %d threads-candidats. bound facile(0xFF..)→%d trouvés  bound dur(0)→%d trouvés\n",
         total_threads, found_easy, found_hard);
  bool pow_ok = (found_easy==total_threads) && (found_hard==0);
  printf(pow_ok ? "✅ POW-CHECK câblé : blake3 keyed officiel + comparateur uint256 OK (tout/rien selon bound).\n"
                : "❌ POW-CHECK incohérent (easy=%d attendu %d, hard=%d attendu 0).\n", found_easy,total_threads,found_hard);
  return (bad||!pow_ok)?1:0;
}
