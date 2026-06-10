// Mini-GEMM TMA (correctness) — soude les 2 maillons prouvés : TMA load swizzlé + ldmatrix→IMMA.
// Single-tile : 1 CTA calcule C(128×256) = A(128×128)·B(256×128)^T (TN, int8→int32).
// A,B chargés en TMA dans les smem swizzlés EXACTS d'alpha (Swizzle<3,4,3>), MMA 8 warps (atom 2×4).
// But : C == réf GEMM int8 bit-exact. Recette = ALPHA_NATIVE_RECIPE.md.
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
using namespace cute;

template <class TA, class TB, class SA, class SB>
struct SharedStorage {
  alignas(1024) ArrayEngine<TA, cosize_v<SA>> A;
  alignas(1024) ArrayEngine<TB, cosize_v<SB>> B;
  alignas(16) uint64_t mbar[2];
};

template <class TmaA, class TmaB, class GA, class GB, class SA, class SB, class Mma,
          class S2RA, class S2RB>
__global__ static __launch_bounds__(256)
void tma_gemm(int8_t const* A, int8_t const* B, int32_t* C, int M, int N, int K,
              CUTE_GRID_CONSTANT TmaA const tma_a, CUTE_GRID_CONSTANT TmaB const tma_b,
              GA gA_layout, GB gB_layout, SA sA_layout, SB sB_layout, Mma mma,
              S2RA s2r_atom_a, S2RB s2r_atom_b) {
  extern __shared__ char smem_[];
  using SS = SharedStorage<int8_t,int8_t,SA,SB>;
  SS& smem = *reinterpret_cast<SS*>(smem_);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);  // (128,128)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);  // (256,128)

  // ---- TMA load A et B (single tile), pattern prouvé (tma_swizzle_test, variables nommées _x) ----
  Tensor mA = tma_a.get_tma_tensor(shape(gA_layout));
  Tensor mB = tma_b.get_tma_tensor(shape(gB_layout));
  Tensor gA = flat_divide(mA, make_shape(Int<128>{}, Int<128>{}));   // (128,128,1,1)
  Tensor gB = flat_divide(mB, make_shape(Int<256>{}, Int<128>{}));   // (256,128,1,1)
  auto cta_a = tma_a.get_slice(Int<0>{});
  auto cta_b = tma_b.get_slice(Int<0>{});
  Tensor tAgA_x = cta_a.partition_S(gA);  Tensor tAsA_x = cta_a.partition_D(sA);
  Tensor tBgB_x = cta_b.partition_S(gB);  Tensor tBsB_x = cta_b.partition_D(sB);
  Tensor tAgA = group_modes<1,rank(tAgA_x)>(tAgA_x);
  Tensor tAsA = group_modes<1,rank(tAsA_x)>(tAsA_x);
  Tensor tBgB = group_modes<1,rank(tBgB_x)>(tBgB_x);
  Tensor tBsB = group_modes<1,rank(tBsB_x)>(tBsB_x);

  if (threadIdx.x == 0) {
    smem.mbar[0] = 0; smem.mbar[1] = 0;
    cute::initialize_barrier(smem.mbar[0], 1);
    cute::initialize_barrier(smem.mbar[1], 1);
    cute::set_barrier_transaction_bytes(smem.mbar[0], int(128*128*sizeof(int8_t)));
    cute::set_barrier_transaction_bytes(smem.mbar[1], int(256*128*sizeof(int8_t)));
    copy(tma_a.with(smem.mbar[0]), tAgA(_,0), tAsA(_,0));
    copy(tma_b.with(smem.mbar[1]), tBgB(_,0), tBsB(_,0));
  }
  __syncthreads();
  cute::wait_barrier(smem.mbar[0], 0);
  cute::wait_barrier(smem.mbar[1], 0);
  __syncthreads();

  // ---- MMA : ldmatrix → IMMA sur les k_blocks ----
  // C single-tile 128×256 : layout STATIQUE (sinon fragment registre = "dynamic owning tensor" interdit)
  Tensor mC = make_tensor(make_gmem_ptr(C),
      make_layout(make_shape(Int<128>{}, Int<256>{}), make_stride(Int<256>{}, Int<1>{})));
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCrA = thr_mma.partition_fragment_A(sA);   // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB);   // (MMA,MMA_N,MMA_K)
  Tensor tCrC = thr_mma.partition_fragment_C(mC);   // (MMA,MMA_M,MMA_N) statique
  clear(tCrC);

  TiledCopy s2r_a = make_tiled_copy_A(s2r_atom_a, mma);
  TiledCopy s2r_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy s2r_ta = s2r_a.get_slice(threadIdx.x);
  ThrCopy s2r_tb = s2r_b.get_slice(threadIdx.x);
  Tensor tXsA = s2r_ta.partition_S(sA);  Tensor tXrA = s2r_ta.retile_D(tCrA);
  Tensor tXsB = s2r_tb.partition_S(sB);  Tensor tXrB = s2r_tb.retile_D(tCrB);
  copy(s2r_atom_a, tXsA, tXrA);
  copy(s2r_atom_b, tXsB, tXrB);
  __syncthreads();

  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTE_UNROLL
  for (int kb = 0; kb < K_BLOCK_MAX; ++kb)
    gemm(mma, tCrA(_,_,kb), tCrB(_,_,kb), tCrC);

  // ---- épilogue : accumulateurs → C (mC statique défini plus haut) ----
  Tensor tCgC = thr_mma.partition_C(mC);
  copy(tCrC, tCgC);
}

int main() {
  using namespace cute;
  const int M=128, N=256, K=128;
  // gmem K-contigu (TN) : A(M,K) stride(K,1), B(N,K) stride(K,1)
  auto gA_layout = make_layout(make_shape(M,K), make_stride(K,1));
  auto gB_layout = make_layout(make_shape(N,K), make_stride(K,1));
  // smem swizzlés EXACTS d'alpha (1 stage)
  auto sA_layout = composition(Swizzle<3,4,3>{},
      Layout<Shape<Shape<_16,_8 >, Shape<_128,_1>>, Stride<Stride<_128,Int<2048>>, Stride<_1,_0>>>{});
  auto sB_layout = composition(Swizzle<3,4,3>{},
      Layout<Shape<Shape<_16,_16>, Shape<_128,_1>>, Stride<Stride<_128,Int<2048>>, Stride<_1,_0>>>{});
  auto mma = make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},
                            Layout<Shape<_2,_4,_1>>{}, Tile<_32,_32,_32>{});
  auto s2r_a = Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{};
  auto s2r_b = Copy_Atom<SM75_U32x2_LDSM_N, int8_t>{};

  // host data (petites valeurs signées → pas d'overflow int32)
  std::vector<int8_t> hA(M*K), hB(N*K); std::vector<int32_t> hC(M*N,0), ref(M*N,0);
  for (int i=0;i<M*K;i++) hA[i]=(int8_t)((i%7)-3);
  for (int i=0;i<N*K;i++) hB[i]=(int8_t)((i%5)-2);
  for (int m=0;m<M;m++) for (int n=0;n<N;n++){ long s=0; for(int k=0;k<K;k++) s+=(int)hA[m*K+k]*(int)hB[n*K+k]; ref[m*N+n]=(int32_t)s; }

  int8_t *dA,*dB; int32_t *dC;
  cudaMalloc(&dA,M*K); cudaMalloc(&dB,N*K); cudaMalloc(&dC,M*N*4);
  cudaMemcpy(dA,hA.data(),M*K,cudaMemcpyHostToDevice);
  cudaMemcpy(dB,hB.data(),N*K,cudaMemcpyHostToDevice);
  cudaMemset(dC,0,M*N*4);

  Tensor gA = make_tensor(make_gmem_ptr<int8_t>(dA), gA_layout);
  Tensor gB = make_tensor(make_gmem_ptr<int8_t>(dB), gB_layout);
  auto tma_a = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gA, sA_layout, make_shape(Int<128>{},Int<128>{}), Int<1>{});
  auto tma_b = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gB, sB_layout, make_shape(Int<256>{},Int<128>{}), Int<1>{});

  int smem_size = int(sizeof(SharedStorage<int8_t,int8_t,decltype(sA_layout),decltype(sB_layout)>));
  auto* k = &tma_gemm<decltype(tma_a),decltype(tma_b),decltype(gA_layout),decltype(gB_layout),
                      decltype(sA_layout),decltype(sB_layout),decltype(mma),decltype(s2r_a),decltype(s2r_b)>;
  cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  k<<<1,256,smem_size>>>(dA,dB,dC,M,N,K,tma_a,tma_b,gA_layout,gB_layout,sA_layout,sB_layout,mma,s2r_a,s2r_b);
  cudaError_t e = cudaDeviceSynchronize();
  if (e!=cudaSuccess){ printf("RUNTIME ERROR: %s\n", cudaGetErrorString(e)); return 2; }

  cudaMemcpy(hC.data(),dC,M*N*4,cudaMemcpyDeviceToHost);
  int mism=0,first=-1; for(int i=0;i<M*N;i++) if(hC[i]!=ref[i]){ if(first<0)first=i; ++mism; }
  if(mism==0) printf("MINI-GEMM TMA OK — C bit-exact vs ref (128x256x128 int8, TMA A+B, 8 warps)\n");
  else printf("GEMM MISMATCH — %d/%d, first@%d (gpu=%d ref=%d)\n", mism,M*N,first,hC[first],ref[first]);
  cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return mism==0?0:1;
}
