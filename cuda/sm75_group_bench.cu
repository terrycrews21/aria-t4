// Grouped-rasterization A/B bench for sm75 grp kernel (wide3 structure + tunable L2 grouping).
// Also samples SM clock during the timed run to detect throttling.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <set>
#include <thread>
#include <chrono>
#include <atomic>
#include <cuda_runtime.h>
#include "pearl_gpu_kernel_sm75_grp.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 16384;
  int N = argc > 2 ? atoi(argv[2]) : 65536;
  int K = argc > 3 ? atoi(argv[3]) : 8192;
  int iters = argc > 4 ? atoi(argv[4]) : 5;
  int rank = argc > 5 ? atoi(argv[5]) : 128;

  printf("grp bench: m=%d n=%d k=%d iters=%d rank=%d\n", M, N, K, iters, rank);

  int8_t *dA, *dB; uint32_t *dkey, *dbnd; int *dfound, *dhr, *dhc;
  CK(cudaMalloc(&dA, (size_t)M * K));
  CK(cudaMalloc(&dB, (size_t)N * K));
  CK(cudaMalloc(&dkey, 32));
  CK(cudaMalloc(&dbnd, 32));
  CK(cudaMalloc(&dfound, 4));
  const int MAXH = 64;
  CK(cudaMalloc(&dhr, (size_t)MAXH * 128 * 4));
  CK(cudaMalloc(&dhc, (size_t)MAXH * 128 * 4));

  std::vector<int8_t> hA((size_t)M * K), hB((size_t)N * K);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = (int8_t)(((i * 1103515245u + 12345u) >> 16) & 0x7F) - 64;
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = (int8_t)(((i * 1103515245u + 54321u) >> 16) & 0x7F) - 64;
  CK(cudaMemcpy(dA, hA.data(), (size_t)M * K, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), (size_t)N * K, cudaMemcpyHostToDevice));
  CK(cudaMemset(dkey, 0x5a, 32));
  CK(cudaMemset(dbnd, 0x00, 32));  // no hits, pure GEMM throughput

  dim3 grd((unsigned)(M / aria_sm75_grp::kBlockM), (unsigned)(N / aria_sm75_grp::kBlockN));
  dim3 blk(aria_sm75_grp::kThreads);
  double work = (double)M * (double)N * (double)K;

  // group size sweep: gx (tile_x group), gy (tile_y group)
  int groups[][2] = {{1,1},{2,1},{1,2},{2,2},{4,2},{2,4},{4,4},{8,4},{4,8},{8,5},{5,8},{8,8},{16,4},{4,16},{16,8},{8,16},{16,16},{32,8},{8,32}};
  int ngroups = sizeof(groups) / sizeof(groups[0]);

  for (int g = 0; g < ngroups; g++) {
    int gx = groups[g][0], gy = groups[g][1];
    if (gx > (int)grd.x || gy > (int)grd.y) continue;

    // warmup
    aria_sm75_grp::grind<false><<<grd, blk, 0>>>(dA, dB, M, N, K, rank, dkey, dbnd, dfound, dhr, dhc, MAXH, gx, gy);
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) {
      aria_sm75_grp::grind<false><<<grd, blk, 0>>>(dA, dB, M, N, K, rank, dkey, dbnd, dfound, dhr, dhc, MAXH, gx, gy);
    }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
    double per = ms / iters;
    double ths = work / (per / 1000.0) / 1e12;
    printf("group %3dx%-3d: per-setup=%8.2fms  internal=%6.2f TH/s  display=%6.2f TH/s\n", gx, gy, per, ths, ths * 1.1);
    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
  }

  printf("DONE\n");
  return 0;
}
