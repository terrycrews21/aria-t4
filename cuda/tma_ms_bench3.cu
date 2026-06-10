// Bench v0.6.1 étape 2 — SWIZZLE de grille (localité L2 sur forme 131072²).
// Copie de gemm_device_tma_ms avec remap (bx,by) en bandes de swz_g tuiles-M :
// l'ordre de lancement CUDA (x fastest) balaie alors un patch swz_g×bM lignes de A
// (qui tient en L2) sur TOUTES les colonnes N avant de passer à la bande suivante.
// → trafic DRAM de A divisé par ~(gridDim.x/swz_g). Fold/transcript par CTA INCHANGÉ
// (même tuile, même ordre k) — seul l'ORDRE des CTAs change.
// Usage : ./tma_ms_bench3 [M] [N] [K] [iters] [reduce_every_k] [swz_g]   (swz_g=0/1 = off)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"
#include "pearl_gpu_kernel_tma.cuh"
using namespace cute;

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class TmaB, class BSmemLayout, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value, 1)
void gemm_device_tma_ms_swz(ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
    TB const* B, BStride dB, CUTE_GRID_CONSTANT TmaB const tma_b, BSmemLayout sB_layout, S2RAtomB s2r_atom_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma,
    int reduce_every_k, int swz_g,
    const uint32_t* pow_key, const uint32_t* pow_bound, int* found_count,
    int* hit_rows, int* hit_cols, int max_hits) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC);
  Tensor mB = tma_b.get_tma_tensor(select<1,2>(shape_MNK));

  // ---- SWIZZLE : remap (bx,by) en bandes de swz_g tuiles-M ----
  int bx = blockIdx.x, by = blockIdx.y;
  if (swz_g > 1) {
    int gm = gridDim.x, gn = gridDim.y;
    int bid  = bx + by * gm;              // ordre de lancement réel (x fastest)
    int band = bid / (swz_g * gn);
    int r    = bid % (swz_g * gn);
    int g    = min(swz_g, gm - band * swz_g);   // bande partielle en fin de grille
    bx = band * swz_g + (r % g);
    by = r / g;
  }
  auto cta_coord = make_coord(bx, by, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});   // (bM,bK,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});   // (bN,bK,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});   // (bM,bN)

  extern __shared__ char smem_[];
  using SS = SharedStorageTMA_MS<TA,TB,ASmemLayout,BSmemLayout>;
  SS& smem = *reinterpret_cast<SS*>(smem_);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (bM,bK,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (bN,bK,PIPE)
  constexpr int K_PIPE = size<2>(ASmemLayout{});

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  auto cta_b = tma_b.get_slice(Int<0>{});
  Tensor tBgB = cta_b.partition_S(gB);
  Tensor tBsB = cta_b.partition_D(sB);
  constexpr int kTmaBytes = int(size(select<1,2>(CtaTiler{})) * sizeof(TB));

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);
  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

  uint32_t transcript[pearl_fold::JACKPOT_SIZE];
  for (int ti=0; ti<pearl_fold::JACKPOT_SIZE; ++ti) transcript[ti]=0;
  int gk = 0;
  auto K_BLOCK_MAX = size<2>(tCrA);
  int k_tile_count = size<3>(tAgA);
  int k_tile_next = 0;

  CUTE_UNROLL
  for (int kp = 0; kp < K_PIPE-1; ++kp) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,kp));
    cp_async_fence();
    if (threadIdx.x == 0) {
      cute::initialize_barrier(smem.mbar[kp], 1);
      cute::set_barrier_transaction_bytes(smem.mbar[kp], kTmaBytes);
      copy(tma_b.with(smem.mbar[kp]), tBgB(_,_,_,k_tile_next), tBsB(_,_,_,kp));
    }
    --k_tile_count; if (k_tile_count > 0) ++k_tile_next;
  }

  int smem_pipe_read = 0, smem_pipe_write = K_PIPE-1;
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  if (K_BLOCK_MAX > 1) {
    cp_async_wait<K_PIPE-2>();
    cute::wait_barrier(smem.mbar[smem_pipe_read], 0);
    __syncthreads();
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE-1)) {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
      if (k_block == K_BLOCK_MAX - 1) {
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);
        cp_async_wait<K_PIPE-2>();
        cute::wait_barrier(smem.mbar[smem_pipe_read], 0);
        __syncthreads();
      }
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
      if (k_block == 0) {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        cp_async_fence();
        if (threadIdx.x == 0) {
          cute::initialize_barrier(smem.mbar[smem_pipe_write], 1);
          cute::set_barrier_transaction_bytes(smem.mbar[smem_pipe_write], kTmaBytes);
          copy(tma_b.with(smem.mbar[smem_pipe_write]), tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        }
        --k_tile_count; if (k_tile_count > 0) ++k_tile_next;
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE-1) ? 0 : smem_pipe_read+1;
      }
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
      if ((++gk % reduce_every_k) == 0) {
        uint32_t h = pearl_fold::xor_reduction(tCrC);
        int idx = (gk / reduce_every_k - 1) % pearl_fold::JACKPOT_SIZE;
        transcript[idx] = pearl_fold::rotl_xor<pearl_fold::HASH_ACCUMULATE_ROTATION>(transcript[idx], h);
      }
    }
  }

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
      auto cD = make_identity_tensor(make_shape(get<0>(cta_tiler), get<1>(cta_tiler)));
      auto tCcD = thr_mma.partition_C(cD);
      int cnt = size(tCcD);
      int row0 = bx * (int)get<0>(cta_tiler);     // bx/by SWIZZLÉS (cohérent avec la tuile)
      int col0 = by * (int)get<1>(cta_tiler);
      for (int i=0;i<cnt;++i){
        hit_rows[slot*128 + i] = row0 + get<0>(tCcD(i));
        hit_cols[slot*128 + i] = col0 + get<1>(tCcD(i));
      }
    }
  }
}

int main(int argc, char** argv) {
  int M = argc>1?atoi(argv[1]):8192;
  int N = argc>2?atoi(argv[2]):8192;
  int K = argc>3?atoi(argv[3]):4096;
  int iters = argc>4?atoi(argv[4]):300;
  int reduce_every_k = argc>5?atoi(argv[5]):4;
  int swz_g = argc>6?atoi(argv[6]):0;
  auto prob = make_shape(M, N, K);
  auto dA = make_stride(K, Int<1>{});
  auto dB = make_stride(K, Int<1>{});
  auto dC = make_stride(N, Int<1>{});
  auto bM=Int<128>{}; auto bN=Int<256>{}; auto bK=Int<128>{};
  auto cta = make_shape(bM,bN,bK);

  auto sA = composition(Swizzle<3,4,3>{},
      Layout<Shape<Shape<_16,_8 >, Shape<_128,_1>, _2>,
             Stride<Stride<_128,Int<2048>>, Stride<_1,_0>, Int<16384>>>{});
  auto sB = composition(Swizzle<3,4,3>{},
      Layout<Shape<Shape<_16,_16>, Shape<_128,_1>, _2>,
             Stride<Stride<_128,Int<2048>>, Stride<_1,_0>, Int<32768>>>{});
  auto sB1 = composition(Swizzle<3,4,3>{},
      Layout<Shape<Shape<_16,_16>, Shape<_128,_1>>, Stride<Stride<_128,Int<2048>>, Stride<_1,_0>>>{});
  auto sC = make_layout(make_shape(bM,bN));

  using AtomG  = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>;
  using TVcopy = Layout<Shape<Shape<_8,_32>,_16>, Stride<Stride<_512,_1>,_32>>;
  using TilerC = Shape<_32,_128>;
  TiledCopy<AtomG, TVcopy, TilerC> copyA;
  TiledMMA mma = make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},
      Layout<Shape<_2,_4,_1>, Stride<_1,_2,_0>>{}, Tile<_32,_32,_32>{});
  Copy_Atom<SM75_U32x4_LDSM_N, int8_t> s2rA;
  Copy_Atom<SM75_U32x2_LDSM_N, int8_t> s2rB;

  int8_t *dAp,*dBp; int32_t *dCp; uint32_t *dkey,*dbnd; int *dfound,*dhr,*dhc;
  cudaMalloc(&dAp,(size_t)M*K); cudaMalloc(&dBp,(size_t)N*K);
  cudaMalloc(&dCp,4); // factice (jamais écrit, pas de DumpC dans cette copie)
  cudaMalloc(&dkey,32); cudaMalloc(&dbnd,32); cudaMalloc(&dfound,4);
  cudaMalloc(&dhr,128*4); cudaMalloc(&dhc,128*4);
  {
    std::vector<int8_t> rA((size_t)M*K), rB((size_t)N*K);
    for (size_t i=0;i<rA.size();++i) rA[i]=(int8_t)(((i*1103515245u+12345u)>>16)&0x7F)-64;
    for (size_t i=0;i<rB.size();++i) rB[i]=(int8_t)(((i*1103515245u+54321u)>>16)&0x7F)-64;
    cudaMemcpy(dAp,rA.data(),(size_t)M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(dBp,rB.data(),(size_t)N*K,cudaMemcpyHostToDevice);
  }
  cudaMemset(dkey,0,32); cudaMemset(dbnd,0,32); cudaMemset(dfound,0,4);

  Tensor gB_t = make_tensor(make_gmem_ptr<int8_t>(dBp), make_layout(make_shape(N,K), make_stride(K, Int<1>{})));
  auto tma_b = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gB_t, sB1, make_shape(bN,bK), Int<1>{});

  int smem = int(sizeof(SharedStorageTMA_MS<int8_t,int8_t,decltype(sA),decltype(sB)>));
  auto kfn = gemm_device_tma_ms_swz<decltype(prob),decltype(cta),
      int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),
      int8_t,decltype(dB),decltype(tma_b),decltype(sB),decltype(s2rB),
      int32_t,decltype(dC),decltype(sC),decltype(mma)>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grd(size(ceil_div(M,bM)), size(ceil_div(N,bN))), blk(size(mma));

  auto launch = [&](){ kfn<<<grd,blk,smem>>>(prob,cta, dAp,dA,sA,copyA,s2rA, dBp,dB,tma_b,sB,s2rB,
      dCp,dC,sC,mma, reduce_every_k, swz_g, dkey,dbnd,dfound, dhr,dhc, 0); };

  for (int i=0;i<5;i++) launch();
  cudaError_t e = cudaDeviceSynchronize();
  if (e!=cudaSuccess){ printf("RUNTIME ERROR (warmup): %s\n", cudaGetErrorString(e)); return 2; }

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i=0;i<iters;i++) launch();
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms=0; cudaEventElapsedTime(&ms,t0,t1);
  e = cudaGetLastError();
  if (e!=cudaSuccess){ printf("RUNTIME ERROR: %s\n", cudaGetErrorString(e)); return 2; }

  double sec = ms/1e3;
  double setups_s = iters/sec;
  double ths = (double)M*N*K*iters/sec/1e12;
  printf("multi-stage TMA SWZ : grid %dx%dx%d, rek=%d, swz_g=%d, %d iters, %.1f ms\n", M,N,K,reduce_every_k,swz_g,iters,ms);
  printf("  setups/s = %.1f   TH/s = %.2f   (smem %d o)\n", setups_s, ths, smem);
  return 0;
}
