// Bench v0.6.1 étape 2 — décomposition du surcoût trame LuckyPool.
// Copie de tma_ms_bench.cu avec : reduce_every_k en argv[5] (4 = rank128, 8 = rank256)
// et C factice (DumpC=false ne l'écrit jamais) pour pouvoir bencher M=N=131072 sans 68 Go.
// Usage : ./tma_ms_bench2 [M] [N] [K] [iters] [reduce_every_k]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"
#include "pearl_gpu_kernel_tma.cuh"
using namespace cute;

int main(int argc, char** argv) {
  int M = argc>1?atoi(argv[1]):8192;
  int N = argc>2?atoi(argv[2]):8192;
  int K = argc>3?atoi(argv[3]):4096;
  int iters = argc>4?atoi(argv[4]):300;
  int reduce_every_k = argc>5?atoi(argv[5]):4;   // 4 = rank128, 8 = rank256
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
  cudaMalloc(&dCp,4); // factice : DumpC=false ne l'écrit jamais
  cudaMalloc(&dkey,32); cudaMalloc(&dbnd,32); cudaMalloc(&dfound,4);
  cudaMalloc(&dhr,128*4); cudaMalloc(&dhc,128*4);
  { // données ALÉATOIRES (pas memset constant → évite l'artefact de compression DRAM)
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
  auto kfn = gemm_device_tma_ms<decltype(prob),decltype(cta),
      int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),
      int8_t,decltype(dB),decltype(tma_b),decltype(sB),decltype(s2rB),
      int32_t,decltype(dC),decltype(sC),decltype(mma), /*DumpC=*/false>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grd(size(ceil_div(M,bM)), size(ceil_div(N,bN))), blk(size(mma));

  auto launch = [&](){ kfn<<<grd,blk,smem>>>(prob,cta, dAp,dA,sA,copyA,s2rA, dBp,dB,tma_b,sB,s2rB,
      dCp,dC,sC,mma, reduce_every_k, dkey,dbnd,dfound, dhr,dhc, 0); };

  // warmup
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
  printf("multi-stage TMA : grid %dx%dx%d, rek=%d, %d iters, %.1f ms\n", M,N,K,reduce_every_k,iters,ms);
  printf("  setups/s = %.1f   TH/s = %.2f   (smem %d o)\n", setups_s, ths, smem);
  return 0;
}
