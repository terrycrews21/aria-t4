// A/B bench: sm75_wide (baseline), sm75_wide3 (variant C), sm75_wide4 (variant D).
// Usage: sm75_cmp_bench2 [m] [n] [k] [iters] [rank] [bound_byte]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <set>
#include <cuda_runtime.h>
#include "pearl_gpu_kernel_sm75_wide.cuh"
#include "pearl_gpu_kernel_sm75_wide3.cuh"
#include "pearl_gpu_kernel_sm75_wide4.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

static double g_work = 0;

template <typename F>
static void timeit(const char* name, int iters, F launch) {
  launch();  // extra warmup
  CK(cudaDeviceSynchronize());
  cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  CK(cudaEventRecord(t0));
  for (int i = 0; i < iters; ++i) launch();
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
  double per = ms / iters;
  double ths = g_work / (per / 1000.0) / 1e12;
  printf("%-28s per-setup=%8.2fms internal=%6.2f TH/s display=%6.2f TH/s\n", name, per, ths, ths * 1.1);
  CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 16384;
  int N = argc > 2 ? atoi(argv[2]) : 65536;
  int K = argc > 3 ? atoi(argv[3]) : 8192;
  int iters = argc > 4 ? atoi(argv[4]) : 5;
  int rank = argc > 5 ? atoi(argv[5]) : 128;
  int bound_byte = argc > 6 ? (int)strtol(argv[6], NULL, 0) : 0x00;

  printf("sm75_cmp_bench2 m=%d n=%d k=%d iters=%d rank=%d bound=0x%02X\n", M, N, K, iters, rank, bound_byte);
  if (M % 128 != 0 || N % 256 != 0) { printf("bad dims\n"); return 2; }
  g_work = (double)M * (double)N * (double)K;

  int8_t *dA, *dB; uint32_t *dkey, *dbnd; int *dfound, *dhr, *dhc;
  CK(cudaMalloc(&dA, (size_t)M * K));
  CK(cudaMalloc(&dB, (size_t)N * K));
  CK(cudaMalloc(&dkey, 32));
  CK(cudaMalloc(&dbnd, 32));
  CK(cudaMalloc(&dfound, 4));
  const int MAXH = 4096;
  CK(cudaMalloc(&dhr, (size_t)MAXH * 128 * 4));
  CK(cudaMalloc(&dhc, (size_t)MAXH * 128 * 4));

  std::vector<int8_t> hA((size_t)M * K), hB((size_t)N * K);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = (int8_t)(((i * 1103515245u + 12345u) >> 16) & 0x7F) - 64;
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = (int8_t)(((i * 1103515245u + 54321u) >> 16) & 0x7F) - 64;
  CK(cudaMemcpy(dA, hA.data(), (size_t)M * K, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), (size_t)N * K, cudaMemcpyHostToDevice));
  CK(cudaMemset(dkey, 0x5a, 32));
  CK(cudaMemset(dbnd, bound_byte, 32));

  dim3 grd((unsigned)(M / 128), (unsigned)(N / 256));

  // ---------------- baseline: wide (1024 thr, 2-stage, paired) ----------------
  {
    dim3 blk(aria_sm75_wide::kThreads);
    timeit("WIDE  (1024thr 2st pair)", iters, [&]() {
      CK(cudaMemset(dfound, 0, 4));
      aria_sm75_wide::grind<false><<<grd, blk, 0>>>(dA, dB, M, N, K, rank, dkey, dbnd, dfound, dhr, dhc, MAXH);
    });
  }

  // ---------------- variant C: wide3 (512 thr, 2-stage, paired) ----------------
  {
    dim3 blk(aria_sm75_wide3::kThreads);
    timeit("WIDE3 (512thr 2st pair)", iters, [&]() {
      CK(cudaMemset(dfound, 0, 4));
      aria_sm75_wide3::grind<false><<<grd, blk, 0>>>(dA, dB, M, N, K, rank, dkey, dbnd, dfound, dhr, dhc, MAXH);
    });
  }

  // ---------------- variant D: wide4 (512 thr, 3-stage, grouped) ----------------
  {
    CK(cudaFuncSetAttribute((const void*)aria_sm75_wide4::grind<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, aria_sm75_wide4::kDynamicSmemBytes));
    CK(cudaFuncSetAttribute((const void*)aria_sm75_wide4::grind<false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    dim3 blk(aria_sm75_wide4::kThreads);
    const int groups[][2] = {{1,1},{2,2},{4,4},{8,4},{4,8},{8,8},{16,8},{8,16},{16,16},{32,8}};
    for (auto& g : groups) {
      int gx = g[0], gy = g[1];
      if (gx > grd.x || gy > grd.y) continue;
      char name[64];
      snprintf(name, sizeof(name), "WIDE4 (512thr 3st g%dx%d)", gx, gy);
      timeit(name, iters, [&]() {
        CK(cudaMemset(dfound, 0, 4));
        aria_sm75_wide4::grind<false><<<grd, blk, aria_sm75_wide4::kDynamicSmemBytes>>>(
            dA, dB, M, N, K, rank, dkey, dbnd, dfound, dhr, dhc, MAXH, gx, gy);
      });
    }
  }

  CK(cudaDeviceSynchronize());
  printf("DONE\n");
  return 0;
}
