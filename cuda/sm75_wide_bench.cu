// Standalone perf + coords bench for the aria_sm75_wide::grind kernel.
// Lets us iterate on kernel variants in seconds (no cargo rebuild).
// Usage: sm75_wide_bench [m] [n] [k] [iters] [rank]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <set>
#include <cuda_runtime.h>
#include "pearl_gpu_kernel_sm75_wide.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 16384;
  int N = argc > 2 ? atoi(argv[2]) : 65536;
  int K = argc > 3 ? atoi(argv[3]) : 8192;
  int iters = argc > 4 ? atoi(argv[4]) : 5;
  int rank = argc > 5 ? atoi(argv[5]) : 128;
  int bound_byte = argc > 6 ? (int)strtol(argv[6], NULL, 0) : 0xFF;

  printf("sm75_wide_bench m=%d n=%d k=%d iters=%d rank=%d bound=0x%02X\n", M, N, K, iters, rank, bound_byte);
  if (M % aria_sm75_wide::kBlockM != 0 || N % aria_sm75_wide::kBlockN != 0) {
    printf("bad dims: need M%%%d==0 N%%%d==0\n", aria_sm75_wide::kBlockM, aria_sm75_wide::kBlockN);
    return 2;
  }

  int8_t *dA, *dB; uint32_t *dkey, *dbnd; int *dfound, *dhr, *dhc;
  CK(cudaMalloc(&dA, (size_t)M * K));
  CK(cudaMalloc(&dB, (size_t)N * K));
  CK(cudaMalloc(&dkey, 32));
  CK(cudaMalloc(&dbnd, 32));
  CK(cudaMalloc(&dfound, 4));
  const int MAXH = 64;
  CK(cudaMalloc(&dhr, (size_t)MAXH * 128 * 4));
  CK(cudaMalloc(&dhc, (size_t)MAXH * 128 * 4));

  // Deterministic pseudo-signal data in [-64, 63] (same range as the miner's).
  std::vector<int8_t> hA((size_t)M * K), hB((size_t)N * K);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = (int8_t)(((i * 1103515245u + 12345u) >> 16) & 0x7F) - 64;
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = (int8_t)(((i * 1103515245u + 54321u) >> 16) & 0x7F) - 64;
  CK(cudaMemcpy(dA, hA.data(), (size_t)M * K, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), (size_t)N * K, cudaMemcpyHostToDevice));
  CK(cudaMemset(dkey, 0x5a, 32));
  CK(cudaMemset(dbnd, bound_byte, 32));  // 0xFF=all hit, 0x00=no hits (pure GEMM throughput)

  dim3 grd((unsigned)(M / aria_sm75_wide::kBlockM), (unsigned)(N / aria_sm75_wide::kBlockN));
  dim3 blk(aria_sm75_wide::kThreads);
  printf("grid=(%u,%u) block=%u\n", grd.x, grd.y, blk.x);

  // Warmup (clock ramp + caches)
  for (int i = 0; i < 2; ++i) {
    CK(cudaMemset(dfound, 0, 4));
    aria_sm75_wide::grind<false><<<grd, blk, 0>>>(dA, dB, M, N, K, rank, dkey, dbnd, dfound, dhr, dhc, MAXH);
  }
  CK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  for (int i = 0; i < iters; ++i) {
    CK(cudaMemset(dfound, 0, 4));
    aria_sm75_wide::grind<false><<<grd, blk, 0>>>(dA, dB, M, N, K, rank, dkey, dbnd, dfound, dhr, dhc, MAXH);
  }
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, t0, t1));
  double per = ms / iters;

  int found = 0;
  CK(cudaMemcpy(&found, dfound, 4, cudaMemcpyDeviceToHost));
  double work = (double)M * (double)N * (double)K;
  double ths = work / (per / 1000.0) / 1e12;
  printf("total=%.1fms per-setup=%.2fms found=%d (expect>0)\n", ms, per, found);
  printf("RESULT internal TH/s = %.2f | display(x1.1) = %.2f\n", ths, ths * 1.1);

  // Coords sanity: slot 0 normalized rows/cols should be 16x16 contiguous.
  if (found > 0) {
    std::vector<int> hr(128), hc(128);
    CK(cudaMemcpy(hr.data(), dhr, 128 * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hc.data(), dhc, 128 * 4, cudaMemcpyDeviceToHost));
    std::set<int> rs(hr.begin(), hr.end()), cs(hc.begin(), hc.end());
    int rmin = *rs.begin(), cmin = *cs.begin();
    printf("slot0 distinct rows=%zu cols=%zu (expect 16x16)\nnormalized rows:", rs.size(), cs.size());
    for (int r : rs) printf(" %d", r - rmin);
    printf("\nnormalized cols:");
    for (int c : cs) printf(" %d", c - cmin);
    printf("\n");
  }
  return 0;
}
