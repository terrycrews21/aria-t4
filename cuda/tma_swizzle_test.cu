// Test TMA + le SmemLayoutB EXACT d'alpha (ComposedLayout swizzlé nesté, pas un swizzle plat).
// But : prouver que la TMA remplit bit-exact le smem swizzlé que notre ldmatrix→IMMA consomme.
// Layout démanglé de pearl_blackwell_native (1 stage) : Swizzle<3,4,3> ∘ ((16,16),(128,1)).
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
using namespace cute;

template <class ElementType, class SmemLayout>
struct SharedStorage {
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
  alignas(16) cute::uint64_t tma_load_mbar[1];
};

template <class T, class TiledCopy, class CTA_Tiler, class GmemLayout, class SmemLayout>
__global__ void
tma_test_device_cute(T const* g_in, T* g_out,
                     CUTE_GRID_CONSTANT TiledCopy const tma, CTA_Tiler cta_tiler,
                     GmemLayout gmem_layout, SmemLayout smem_layout)
{
  CUTE_STATIC_ASSERT_V(product_each(shape(cta_tiler)) == product_each(shape(smem_layout)));
  extern __shared__ char shared_memory[];
  using SharedStorageT = SharedStorage<T, SmemLayout>;
  SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem.begin()), smem_layout);
  uint64_t* tma_load_mbar = shared_storage.tma_load_mbar;

  Tensor mA = tma.get_tma_tensor(shape(gmem_layout));
  Tensor mB = make_tensor(make_gmem_ptr<T>(g_out), gmem_layout);
  constexpr int R = rank_v<CTA_Tiler>;
  Tensor gA = flat_divide(mA, cta_tiler);
  Tensor gB = flat_divide(mB, cta_tiler);

  auto cta_tma = tma.get_slice(Int<0>{});
  Tensor tAgA_x = cta_tma.partition_S(gA);
  Tensor tAsA_x = cta_tma.partition_D(sA);
  Tensor tAgA = group_modes<1,rank(tAgA_x)>(tAgA_x);
  Tensor tAsA = group_modes<1,rank(tAsA_x)>(tAsA_x);
  static_assert(size<1>(tAsA) == 1);
  Tensor tBgB = group_modes<0,R>(group_modes<R,rank(gB)>(gB));

  if (threadIdx.x == 0) { prefetch(tma, tAgA); }
  for (int stage = 0; stage < size<1>(tAgA); ++stage) {
    int kTmaTransactionBytes = (int)(size(cta_tiler) * sizeof(T)); // swizzle-safe
    if (threadIdx.x == 0) {
      tma_load_mbar[0] = 0;
      cute::initialize_barrier(tma_load_mbar[0], 1);
      cute::set_barrier_transaction_bytes(tma_load_mbar[0], kTmaTransactionBytes);
      copy(tma.with(tma_load_mbar[0]), tAgA(_,stage), tAsA(_,0));
    }
    __syncthreads();
    cute::wait_barrier(tma_load_mbar[0], 0);
    if (thread0()) { copy(sA, tBgB(_,stage)); }
    __syncthreads();
  }
}

int main() {
  using T = int8_t;
  // SmemLayoutB d'alpha (1 stage) : Swizzle<3,4,3> ∘ ((16,16),(128,1)) strides ((128,2048),(1,0))
  auto smem_layout = composition(Swizzle<3,4,3>{},
      Layout<Shape <Shape <_16, _16>,        Shape <_128, _1>>,
             Stride<Stride<_128, Int<2048>>,  Stride<_1,  _0>>>{});
  auto cta_tile    = make_shape(Int<256>{}, Int<128>{});
  // smem B d'alpha = K-contigu (dim K stride 1) → gmem aussi K-contigu : stride (128,1)
  auto gmem_layout = make_layout(make_shape(256, 128), make_stride(128, 1));

  const int N = 256 * 128;
  std::vector<int8_t> h_in(N), h_out(N, 0);
  for (int i = 0; i < N; ++i) h_in[i] = (int8_t)(i % 13);
  int8_t *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_in, N); cudaMalloc(&d_out, N);
  cudaMemcpy(d_in, h_in.data(), N, cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0xFF, N);

  Tensor gA = make_tensor(make_gmem_ptr<T>(d_in), gmem_layout);
  auto tma = make_tma_copy<T>(SM90_TMA_LOAD{}, gA, smem_layout, cta_tile, Int<1>{});

  int smem_size = int(sizeof(SharedStorage<T, decltype(smem_layout)>));
  tma_test_device_cute<<<1, 128, smem_size>>>(d_in, d_out, tma, cta_tile, gmem_layout, smem_layout);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("RUNTIME ERROR: %s\n", cudaGetErrorString(e)); return 2; }

  cudaMemcpy(h_out.data(), d_out, N, cudaMemcpyDeviceToHost);
  int mism = 0, first = -1;
  for (int i = 0; i < N; ++i) if (h_out[i] != h_in[i]) { if (first < 0) first = i; ++mism; }
  if (mism == 0) printf("TMA+Swizzle<3,4,3> (smem B alpha) OK — bit-exact %d octets\n", N);
  else printf("MISMATCH — %d/%d diff, first@%d (in=%d out=%d)\n", mism, N, first, h_in[first], h_out[first]);
  cudaFree(d_in); cudaFree(d_out);
  return mism == 0 ? 0 : 1;
}
