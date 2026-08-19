// Bench: wide3 (2-stage) vs wide6 (3-stage dynamic smem + parallel blake3)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "pearl_gpu_kernel_sm75_wide3.cuh"
#include "pearl_gpu_kernel_sm75_wide6.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 16384;
  int N = argc > 2 ? atoi(argv[2]) : 65536;
  int K = argc > 3 ? atoi(argv[3]) : 8192;
  int iters = argc > 4 ? atoi(argv[4]) : 5;
  int rank = argc > 5 ? atoi(argv[5]) : 128;

  printf("bench_v3: m=%d n=%d k=%d iters=%d rank=%d\n", M, N, K, iters, rank);

  int8_t *dA, *dB; uint32_t *dkey, *dbnd; int *dfound, *dhr, *dhc;
  CK(cudaMalloc(&dA, (size_t)M * K));
  CK(cudaMalloc(&dB, (size_t)N * K));
  CK(cudaMalloc(&dkey, 32)); CK(cudaMalloc(&dbnd, 32));
  CK(cudaMalloc(&dfound, 4));
  CK(cudaMalloc(&dhr, 64*128*4)); CK(cudaMalloc(&dhc, 64*128*4));

  std::vector<int8_t> hA((size_t)M*K), hB((size_t)N*K);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = (int8_t)(((i*1103515245u+12345u)>>16)&0x7F)-64;
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = (int8_t)(((i*1103515245u+54321u)>>16)&0x7F)-64;
  CK(cudaMemcpy(dA, hA.data(), (size_t)M*K, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), (size_t)N*K, cudaMemcpyHostToDevice));
  CK(cudaMemset(dkey, 0x5a, 32));
  CK(cudaMemset(dbnd, 0x00, 32));

  dim3 grd(M/128, N/256);
  double work = (double)M * (double)N * (double)K;

  // wide3 baseline
  {
    dim3 blk(aria_sm75_wide3::kThreads);
    aria_sm75_wide3::grind<false><<<grd, blk>>>(dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i=0;i<iters;i++) {
      CK(cudaMemset(dfound,0,4));
      aria_sm75_wide3::grind<false><<<grd,blk>>>(dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
    }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms,t0,t1));
    double per=ms/iters, ths=work/(per/1000.0)/1e12;
    printf("wide3  2-stage static : %7.2f ms  %6.2f TH/s int  %6.2f disp\n", per, ths, ths*1.1);
  }

  // wide6: 3-stage dynamic smem
  {
    dim3 blk(aria_sm75_wide6::kThreads);
    size_t smem = aria_sm75_wide6::kDynamicSmemBytes;
    CK(cudaFuncSetAttribute((const void*)aria_sm75_wide6::grind<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CK(cudaFuncSetAttribute((const void*)aria_sm75_wide6::grind<false>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    aria_sm75_wide6::grind<false><<<grd, blk, smem>>>(dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("wide6 FAILED: %s\n", cudaGetErrorString(err));
    } else {
      cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
      CK(cudaEventRecord(t0));
      for (int i=0;i<iters;i++) {
        CK(cudaMemset(dfound,0,4));
        aria_sm75_wide6::grind<false><<<grd,blk,smem>>>(dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
      }
      CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
      float ms; CK(cudaEventElapsedTime(&ms,t0,t1));
      double per=ms/iters, ths=work/(per/1000.0)/1e12;
      printf("wide6  3-stage dynamic: %7.2f ms  %6.2f TH/s int  %6.2f disp\n", per, ths, ths*1.1);
    }
  }

  printf("DONE\n");
  return 0;
}
