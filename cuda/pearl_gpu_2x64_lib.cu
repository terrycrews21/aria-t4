// ariaminer v1.1 — lib FFI 2×64 (AlphaPool). Identique à pearl_gpu_lib.cu mais
// instancie gemm_device_2x64 (fold 2×64) + SharedStorage2x64. La v1.0 (8×16) reste intacte.
#include <cstdint>
#include <cstdio>
#include "pearl_gpu_kernel_2x64.cuh"

#define CKR(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"pearl_gpu_2x64 CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));return -1;}}while(0)

extern "C" {

// Idem pearl_gpu_grind mais tuile 2×64. hit_rows/hit_cols : max_hits*128 coords (2 lignes × 64 cols).
int pearl_gpu_grind_2x64(const int8_t* a_eff, const int8_t* b_eff, int M, int N, int K,
                         const uint32_t* pow_key, const uint32_t* pow_bound,
                         int* hit_rows, int* hit_cols, int max_hits) {
  using namespace cute;
  auto prob = make_shape(M, N, K);
  auto dA = make_stride(K, Int<1>{});
  auto dB = make_stride(K, Int<1>{});
  auto dC = make_stride(N, Int<1>{});
  auto bM = Int<128>{}; auto bN = Int<128>{}; auto bK = Int<64>{};
  auto cta = make_shape(bM, bN, bK);

  using SmemBase = Layout<Shape <Shape <_16,_8>,        Shape <_64,_1>, Shape <_1,_3>>,
                          Stride<Stride<_64,Int<1024>>, Stride<_1,_0>,  Stride<_0,Int<8192>>>>;
  auto sA = composition(Swizzle<2,4,3>{}, SmemBase{});
  auto sB = composition(Swizzle<2,4,3>{}, SmemBase{});
  auto sC = make_layout(make_shape(bM, bN));
  using AtomG  = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>;
  using TVcopy = Layout<Shape <Shape <_4,_32>,       _16>, Stride<Stride<Int<512>,_1>, _32>>;
  using TilerC = Shape<_32,_64>;
  TiledCopy<AtomG, TVcopy, TilerC> copyA, copyB;
  TiledMMA mma = make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},
      Layout<Shape<_2,_2,_1>, Stride<_1,_2,_0>>{}, Tile<_32,_32,_32>{});
  Copy_Atom<SM75_U32x4_LDSM_N, int8_t> s2rA, s2rB;

  int reduce_every_k = 128 / 32;
  int smem = int(sizeof(SharedStorage2x64<int8_t,int8_t,decltype(sA),decltype(sB)>));
  dim3 grd(size(ceil_div(M,bM)), size(ceil_div(N,bN))), blk(size(mma));

  int8_t *da,*db; int32_t *dc; uint32_t *dkey,*dbnd; int *dfound,*dhr,*dhc;
  CKR(cudaMalloc(&da,(size_t)M*K)); CKR(cudaMalloc(&db,(size_t)N*K)); CKR(cudaMalloc(&dc,(size_t)M*N*4));
  CKR(cudaMalloc(&dkey,32)); CKR(cudaMalloc(&dbnd,32)); CKR(cudaMalloc(&dfound,4));
  CKR(cudaMalloc(&dhr,(size_t)max_hits*128*4)); CKR(cudaMalloc(&dhc,(size_t)max_hits*128*4));
  CKR(cudaMemcpy(da,a_eff,(size_t)M*K,cudaMemcpyHostToDevice));
  CKR(cudaMemcpy(db,b_eff,(size_t)N*K,cudaMemcpyHostToDevice));
  CKR(cudaMemcpy(dkey,pow_key,32,cudaMemcpyHostToDevice));
  CKR(cudaMemcpy(dbnd,pow_bound,32,cudaMemcpyHostToDevice));
  CKR(cudaMemset(dfound,0,4));

  auto kfn = gemm_device_2x64<decltype(prob),decltype(cta),
      int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2rA),
      int8_t,decltype(dB),decltype(sB),decltype(copyB),decltype(s2rB),
      int32_t,decltype(dC),decltype(sC),decltype(mma)>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kfn<<<grd,blk,smem>>>(prob,cta, da,dA,sA,copyA,s2rA, db,dB,sB,copyB,s2rB,
      dc,dC,sC,mma, reduce_every_k, dkey,dbnd,dfound, dhr,dhc,max_hits);
  CKR(cudaDeviceSynchronize());

  int found=0; CKR(cudaMemcpy(&found,dfound,4,cudaMemcpyDeviceToHost));
  int nret = found<max_hits?found:max_hits;
  if (nret>0 && hit_rows) CKR(cudaMemcpy(hit_rows,dhr,(size_t)nret*128*4,cudaMemcpyDeviceToHost));
  if (nret>0 && hit_cols) CKR(cudaMemcpy(hit_cols,dhc,(size_t)nret*128*4,cudaMemcpyDeviceToHost));

  cudaFree(da);cudaFree(db);cudaFree(dc);cudaFree(dkey);cudaFree(dbnd);
  cudaFree(dfound);cudaFree(dhr);cudaFree(dhc);
  return found;
}

}  // extern "C"
