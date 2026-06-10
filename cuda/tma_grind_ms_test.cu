// proof_check du gemm_device_tma_ms (MULTI-STAGE, PIPE=2) : A=cp.async ms + B=TMA ms.
// Config alpha 128×256×128, smem 3-mode (PIPE=2), 8 warps. M=128 N=256 K=512 (4 k-tiles).
// pow_bound=0 (powcheck off) → valide C accumulé vs réf CPU bit-exact (= même C que gemm_device).
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"
#include "pearl_gpu_kernel_tma.cuh"
using namespace cute;

int main() {
  const int M=128, N=256, K=512;
  auto prob = make_shape(M, N, K);
  auto dA = make_stride(K, Int<1>{});
  auto dB = make_stride(K, Int<1>{});
  auto dC = make_stride(N, Int<1>{});
  auto bM=Int<128>{}; auto bN=Int<256>{}; auto bK=Int<128>{};
  auto cta = make_shape(bM,bN,bK);

  // smem 3-mode PIPE=2 : base alpha (Swizzle<3,4,3>) + stage stride = cosize 1 tuile.
  auto sA = composition(Swizzle<3,4,3>{},
      Layout<Shape<Shape<_16,_8 >, Shape<_128,_1>, _2>,
             Stride<Stride<_128,Int<2048>>, Stride<_1,_0>, Int<16384>>>{});
  auto sB = composition(Swizzle<3,4,3>{},
      Layout<Shape<Shape<_16,_16>, Shape<_128,_1>, _2>,
             Stride<Stride<_128,Int<2048>>, Stride<_1,_0>, Int<32768>>>{});
  // layout 1 tuile (box) pour le descripteur TMA
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
  int reduce_every_k = 128/32;

  std::vector<int8_t> hA(M*K), hB(N*K); std::vector<int32_t> hC(M*N,0), ref(M*N,0);
  for (int i=0;i<M*K;i++) hA[i]=(int8_t)((i*7+3)%13 - 6);
  for (int i=0;i<N*K;i++) hB[i]=(int8_t)((i*5+1)%11 - 5);
  for (int m=0;m<M;m++) for (int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(int)hA[m*K+k]*(int)hB[n*K+k]; ref[m*N+n]=(int32_t)s; }

  int8_t *dAp,*dBp; int32_t *dCp; uint32_t *dkey,*dbnd; int *dfound,*dhr,*dhc;
  cudaMalloc(&dAp,M*K); cudaMalloc(&dBp,N*K); cudaMalloc(&dCp,M*N*4);
  cudaMalloc(&dkey,32); cudaMalloc(&dbnd,32); cudaMalloc(&dfound,4);
  cudaMalloc(&dhr,128*4); cudaMalloc(&dhc,128*4);
  cudaMemcpy(dAp,hA.data(),M*K,cudaMemcpyHostToDevice);
  cudaMemcpy(dBp,hB.data(),N*K,cudaMemcpyHostToDevice);
  cudaMemset(dkey,0,32); cudaMemset(dbnd,0,32); cudaMemset(dfound,0,4); cudaMemset(dCp,0,M*N*4);

  Tensor gB_t = make_tensor(make_gmem_ptr<int8_t>(dBp), make_layout(make_shape(N,K), make_stride(K, Int<1>{})));
  auto tma_b = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gB_t, sB1, make_shape(bN,bK), Int<1>{});

  int smem = int(sizeof(SharedStorageTMA_MS<int8_t,int8_t,decltype(sA),decltype(sB)>));
  printf("smem = %d octets (limite Blackwell 101376)\n", smem);
  auto kfn = gemm_device_tma_ms<decltype(prob),decltype(cta),
      int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),
      int8_t,decltype(dB),decltype(tma_b),decltype(sB),decltype(s2rB),
      int32_t,decltype(dC),decltype(sC),decltype(mma)>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grd(size(ceil_div(M,bM)), size(ceil_div(N,bN))), blk(size(mma));
  kfn<<<grd,blk,smem>>>(prob,cta, dAp,dA,sA,copyA,s2rA, dBp,dB,tma_b,sB,s2rB,
      dCp,dC,sC,mma, reduce_every_k, dkey,dbnd,dfound, dhr,dhc, 1);
  cudaError_t e = cudaDeviceSynchronize();
  if (e!=cudaSuccess){ printf("RUNTIME ERROR: %s\n", cudaGetErrorString(e)); return 2; }

  cudaMemcpy(hC.data(),dCp,M*N*4,cudaMemcpyDeviceToHost);
  int mism=0,first=-1; for(int i=0;i<M*N;i++) if(hC[i]!=ref[i]){ if(first<0)first=i; ++mism; }
  if(mism==0) printf("GRIND-TMA-MS OK — C bit-exact vs ref (128x256x512, A=cp.async ms + B=TMA ms, PIPE=2, 8 warps)\n");
  else printf("GRIND-TMA-MS MISMATCH — %d/%d, first@%d (gpu=%d ref=%d)\n", mism,M*N,first,hC[first],ref[first]);
  return mism==0?0:1;
}
