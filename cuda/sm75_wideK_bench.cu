// Bench: wide3 (LDS.32 fragments) vs wideK (ldmatrix fragments)
// Same geometry/proof output; compares per-setup time and found-count parity.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <time.h>
#include <cuda_runtime.h>
#include "pearl_gpu_kernel_sm75_wide3.cuh"
#include "pearl_gpu_kernel_sm75_wideK.cuh"
#include "pearl_gpu_kernel_sm75_wideL.cuh"
#include "pearl_gpu_kernel_sm75_wideM.cuh"
#include "pearl_gpu_kernel_sm75_wideP.cuh"
#include "pearl_gpu_kernel_sm75_wideQ.cuh"
#include "pearl_gpu_kernel_sm75_wideS.cuh"
#include "pearl_gpu_kernel_sm75_wideT.cuh"
#include "pearl_gpu_kernel_sm75_wideV.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 16384;
  int N = argc > 2 ? atoi(argv[2]) : 65536;
  int K = argc > 3 ? atoi(argv[3]) : 8192;
  int iters = argc > 4 ? atoi(argv[4]) : 5;
  int rank = argc > 5 ? atoi(argv[5]) : 128;

  printf("bench: m=%d n=%d k=%d iters=%d rank=%d\n", M, N, K, iters, rank);

  int8_t *dA, *dB; uint32_t *dkey, *dbnd; int *dfound, *dhr, *dhc;
  CK(cudaMalloc(&dA, (size_t)M * K));
  CK(cudaMalloc(&dB, (size_t)N * K));
  CK(cudaMalloc(&dkey, 32)); CK(cudaMalloc(&dbnd, 32));
  CK(cudaMalloc(&dfound, 4));
  CK(cudaMalloc(&dhr, 64*128*4)); CK(cudaMalloc(&dhc, 64*128*4));
  // transcript scratch for wideL: 256-slot pool * 16 warps * 8 tiles * 16 words
  uint32_t* dtrL = nullptr;
  CK(cudaMalloc(&dtrL, 256ull * 16 * 8 * 16 * 4));

  std::vector<int8_t> hA((size_t)M*K), hB((size_t)N*K);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = (int8_t)(((i*1103515245u+12345u)>>16)&0x7F)-64;
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = (int8_t)(((i*1103515245u+54321u)>>16)&0x7F)-64;
  CK(cudaMemcpy(dA, hA.data(), (size_t)M*K, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), (size_t)N*K, cudaMemcpyHostToDevice));
  CK(cudaMemset(dkey, 0x5a, 32));
  // ~25% pass bound (LE compare, top word < 0x40): found counts are a
  // statistical fingerprint of transcript bit-exactness.
  CK(cudaMemset(dbnd, 0x00, 32));
  { unsigned char hbound[32] = {0}; hbound[31] = 0x40;
    CK(cudaMemcpy(dbnd, hbound, 32, cudaMemcpyHostToDevice)); }
  uint64_t hr_ref_sig = 0;  // reference hit-set signature from wide3

  double work = (double)M * (double)N * (double)K;
  auto bench = [&](const char* name, auto launch, bool capture) {
    // 25s idle cooldown so every kernel starts from the same thermal state;
    // the 15-iter average then reflects sustained (hot) performance, which is
    // what continuous mining actually gets.
    CK(cudaDeviceSynchronize());
    { struct timespec ts; ts.tv_sec = 25; ts.tv_nsec = 0; nanosleep(&ts, nullptr); }
    CK(cudaMemset(dfound,0,4));
    launch();  // warmup
    CK(cudaDeviceSynchronize());
    int found0 = 0; CK(cudaMemcpy(&found0, dfound, 4, cudaMemcpyDeviceToHost));
    // Order-independent fingerprint: XOR-fold all 128 (row,col) pairs per slot
    // (atomicAdd slot order legitimately differs between kernels).
    uint64_t sig = 0;
    {
      std::vector<int> hr(64*128), hc(64*128);
      CK(cudaMemcpy(hr.data(), dhr, hr.size()*4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hc.data(), dhc, hc.size()*4, cudaMemcpyDeviceToHost));
      for (int s = 0; s < found0 && s < 64; ++s)
        for (int i = 0; i < 128; ++i)
          sig ^= (uint64_t)(uint32_t)(hr[s*128+i] * 1315423911u + hc[s*128+i]) << (i & 15);
      if (capture) { hr_ref_sig = sig; }
      else {
        printf("  [BIT-EXACT] hit-set signature: %s (ref=%016llx this=%016llx)\n",
               sig == hr_ref_sig ? "IDENTICAL" : "MISMATCH!!!",
               (unsigned long long)hr_ref_sig, (unsigned long long)sig);
      }
    }
    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i=0;i<iters;i++) { CK(cudaMemset(dfound,0,4)); launch(); }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms,t0,t1));
    int found1 = 0; CK(cudaMemcpy(&found1, dfound, 4, cudaMemcpyDeviceToHost));
    double per=ms/iters, ths=work/(per/1000.0)/1e12;
    printf("%-28s: %8.2f ms  %6.2f TH/s int  %6.2f disp  found=%d/%d\n",
           name, per, ths, ths*1.1, found0, found1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
  };

  dim3 grd(M/128, N/256);
  bench("wide3  (LDS.32)", [&](){
    aria_sm75_wide3::grind<false><<<grd, dim3(aria_sm75_wide3::kThreads)>>>(
        dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
  }, true);
  bench("wideK  (ldmatrix)", [&](){
    aria_sm75_wideK::grind<false><<<grd, dim3(aria_sm75_wideK::kThreads)>>>(
        dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
  }, false);
  bench("wideL  (ldmatrix 2CTA/SM)", [&](){
    aria_sm75_wideL::grind<false><<<grd, dim3(aria_sm75_wideL::kThreads)>>>(
        dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64,dtrL);
  }, false);
  bench("wideM  (1024thr 4tile/warp)", [&](){
    aria_sm75_wideM::grind<false><<<grd, dim3(aria_sm75_wideM::kThreads)>>>(
        dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
  }, false);
  {
    size_t smemP = aria_sm75_wideP::kDynamicSmemBytes;
    CK(cudaFuncSetAttribute((const void*)aria_sm75_wideP::grind<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemP));
    bench("wideP  (chunkK=64 ldmatrix)", [&](){
      aria_sm75_wideP::grind<false><<<grd, dim3(aria_sm75_wideP::kThreads), smemP>>>(
          dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64,dtrL);
    }, false);
  }
  bench("wideQ  (frag dblbuf)", [&](){
    aria_sm75_wideQ::grind<false><<<grd, dim3(aria_sm75_wideQ::kThreads)>>>(
        dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
  }, false);
  {
    // wideS: 64x128 CTA -> grid (M/64, N/128)
    dim3 grdS(M/aria_sm75_wideS::kBlockM, N/aria_sm75_wideS::kBlockN);
    bench("wideS  (64x128 2CTA/SM)", [&](){
      aria_sm75_wideS::grind<false><<<grdS, dim3(aria_sm75_wideS::kThreads)>>>(
          dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
    }, false);
  }
  {
    dim3 grdS(M/aria_sm75_wideT::kBlockM, N/aria_sm75_wideT::kBlockN);
    bench("wideT  (64x128 3-stage)", [&](){
      aria_sm75_wideT::grind<false><<<grdS, dim3(aria_sm75_wideT::kThreads)>>>(
          dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
    }, false);
  }
  {
    size_t smemV = aria_sm75_wideV::kDynamicSmemBytes;
    CK(cudaFuncSetAttribute((const void*)aria_sm75_wideV::grind<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smemV));
    bench("wideV  (3-stage ldmatrix)", [&](){
      aria_sm75_wideV::grind<false><<<grd, dim3(aria_sm75_wideV::kThreads), smemV>>>(
          dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
    }, false);
  }
  // second pass: wide3 again to detect clock drift
  bench("wide3  (LDS.32 rerun)", [&](){
    aria_sm75_wide3::grind<false><<<grd, dim3(aria_sm75_wide3::kThreads)>>>(
        dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
  }, false);
  return 0;
}
