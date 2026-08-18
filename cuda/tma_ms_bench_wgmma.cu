// tma_ms_bench_wgmma.cu — perf + correctness bench for the SM90 wgmma grind kernel.
//
// WHY: gate before integrating gemm_device_wgmma_ms into pearl_resident_lib.cu.
//   verify — bit-exact C vs CPU reference on one CTA tile (M=128 N=256 K=512).
//            MUST pass before any pool-facing build: a wrong accumulator means
//            wrong transcripts -> rejected shares -> ban risk.
//   perf   — full mining shape (default 16384x65536x8192), bound=0, DumpC=false,
//            TH/s = M*N*K*iters/s/1e12 — same convention as tma_ms_bench.cu and
//            the tworker display (pre x1.10 cosmetic multiplier).
//
// Usage:
//   ./tma_ms_bench_wgmma verify
//   ./tma_ms_bench_wgmma perf [M] [N] [K] [iters] [reduce_every_k] [swz_g] [pipe]
//
// Build: nvcc -arch=sm_90a -O3 -std=c++17 -I<cutlass>/include -I<pearl csrc>
//        -I. -I./shim --expt-relaxed-constexpr -o /tmp/wgmma_bench tma_ms_bench_wgmma.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>
#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"
#include "pearl_gpu_kernel_wgmma.cuh"
using namespace cute;

// One fully-configured bench instantiation for a given pipeline depth.
// K_PIPE is compile-time because it lives in the smem layout shape.
template <int K_PIPE>
static double run_perf(int M, int N, int K, int iters, int reduce_every_k, int swz_g) {
  auto prob = make_shape(M, N, K);
  auto bM=Int<128>{}; auto bN=Int<256>{}; auto bK=Int<128>{};
  auto cta = make_shape(bM,bN,bK);
  auto dC = make_stride(N, Int<1>{});
  auto sC = make_layout(make_shape(bM,bN));

  // GMMA-compatible K-major SW128 smem layouts, tiled to (tile, k, PIPE).
  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<int8_t>;
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<128>, Int<128>, Int<K_PIPE>>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<256>, Int<128>, Int<K_PIPE>>{}));
  SmemLayoutA sA; SmemLayoutB sB;

  // 2 warpgroups x 64x256x32 SS atom = 128x256 CTA tile, 256 threads.
  TiledMMA mma = make_tiled_mma(SM90_64x256x32_S32S8S8_SS_TN{}, Layout<Shape<_2,_1,_1>>{});

  int8_t *dAp,*dBp; int32_t *dCp; uint32_t *dkey,*dbnd; int *dfound,*dhr,*dhc;
  cudaMalloc(&dAp,(size_t)M*K); cudaMalloc(&dBp,(size_t)N*K); cudaMalloc(&dCp,(size_t)M*N*4);
  cudaMalloc(&dkey,32); cudaMalloc(&dbnd,32); cudaMalloc(&dfound,4);
  cudaMalloc(&dhr,128*4*64); cudaMalloc(&dhc,128*4*64);
  { // pseudo-random int7 data (avoids DRAM compression artifacts)
    std::vector<int8_t> rA((size_t)M*K), rB((size_t)N*K);
    for (size_t i=0;i<rA.size();++i) rA[i]=(int8_t)(((i*1103515245u+12345u)>>16)&0x7F)-64;
    for (size_t i=0;i<rB.size();++i) rB[i]=(int8_t)(((i*1103515245u+54321u)>>16)&0x7F)-64;
    cudaMemcpy(dAp,rA.data(),(size_t)M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(dBp,rB.data(),(size_t)N*K,cudaMemcpyHostToDevice);
  }
  cudaMemset(dkey,0,32); cudaMemset(dbnd,0,32); cudaMemset(dfound,0,4);

  Tensor gA_t = make_tensor(make_gmem_ptr(dAp), make_layout(make_shape(M,K), make_stride(K, Int<1>{})));
  Tensor gB_t = make_tensor(make_gmem_ptr(dBp), make_layout(make_shape(N,K), make_stride(K, Int<1>{})));
  auto tma_a = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gA_t, sA(_,_,_0{}), make_shape(bM,bK), Int<1>{});
  auto tma_b = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gB_t, sB(_,_,_0{}), make_shape(bN,bK), Int<1>{});

  int smem = int(sizeof(SharedStorageWGMMA<SmemLayoutA,SmemLayoutB,K_PIPE>));
  auto kfn = gemm_device_wgmma_ms<decltype(prob),decltype(cta),
      SmemLayoutA,decltype(tma_a),
      SmemLayoutB,decltype(tma_b),
      int32_t,decltype(dC),decltype(sC),decltype(mma),
      /*DumpC=*/false, /*BigEndian=*/false>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grd(size(ceil_div(M,bM)), size(ceil_div(N,bN))), blk(size(mma));
  printf("perf<PIPE=%d>: grid %dx%d blk %u smem %d bytes\n", K_PIPE, grd.x, grd.y, blk.x, smem);

  auto launch = [&](){ kfn<<<grd,blk,smem>>>(prob,cta,
      sA,tma_a, sB,tma_b,
      (int32_t*)dCp,dC,sC,mma, reduce_every_k, swz_g, dkey,dbnd,dfound, dhr,dhc, 64); };

  for (int i=0;i<3;i++) launch();
  cudaError_t e = cudaDeviceSynchronize();
  if (e!=cudaSuccess){ printf("RUNTIME ERROR (warmup): %s\n", cudaGetErrorString(e)); return -1; }

  cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
  cudaEventRecord(t0);
  for (int i=0;i<iters;i++) launch();
  cudaEventRecord(t1); cudaEventSynchronize(t1);
  float ms=0; cudaEventElapsedTime(&ms,t0,t1);
  e = cudaGetLastError();
  if (e!=cudaSuccess){ printf("RUNTIME ERROR: %s\n", cudaGetErrorString(e)); return -1; }

  double sec = ms/1e3;
  double ths = (double)M*N*K*iters/sec/1e12;
  printf("  %d iters, %.1f ms total -> setups/s = %.1f   TH/s = %.2f\n",
         iters, ms, iters/sec, ths);

  cudaFree(dAp); cudaFree(dBp); cudaFree(dCp);
  cudaFree(dkey); cudaFree(dbnd); cudaFree(dfound); cudaFree(dhr); cudaFree(dhc);
  return ths;
}

// Correctness: one CTA tile (128x256) over K=512, DumpC=true, compare C to CPU.
static int run_verify() {
  const int M=128, N=256, K=512;   // 4 k-tiles of 128; exercises the ring + drain
  auto prob = make_shape(M, N, K);
  auto bM=Int<128>{}; auto bN=Int<256>{}; auto bK=Int<128>{};
  auto cta = make_shape(bM,bN,bK);
  auto dC = make_stride(N, Int<1>{});
  auto sC = make_layout(make_shape(bM,bN));

  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<int8_t>;
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<128>, Int<128>, Int<4>>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<256>, Int<128>, Int<4>>{}));
  SmemLayoutA sA; SmemLayoutB sB;
  TiledMMA mma = make_tiled_mma(SM90_64x256x32_S32S8S8_SS_TN{}, Layout<Shape<_2,_1,_1>>{});

  // Deterministic small-range data so a CPU reference is trivial to reason about.
  std::vector<int8_t> hA((size_t)M*K), hB((size_t)N*K);
  for (size_t i=0;i<hA.size();++i) hA[i]=(int8_t)((i*7+3)%13 - 6);
  for (size_t i=0;i<hB.size();++i) hB[i]=(int8_t)((i*5+1)%11 - 5);
  std::vector<int32_t> ref((size_t)M*N, 0);
  for (int m=0;m<M;m++) for (int n=0;n<N;n++) {
    long s=0; for (int k=0;k<K;k++) s += (long)hA[(size_t)m*K+k] * (long)hB[(size_t)n*K+k];
    ref[(size_t)m*N+n] = (int32_t)s;
  }

  int8_t *dAp,*dBp; int32_t *dCp; uint32_t *dkey,*dbnd; int *dfound,*dhr,*dhc;
  cudaMalloc(&dAp,(size_t)M*K); cudaMalloc(&dBp,(size_t)N*K); cudaMalloc(&dCp,(size_t)M*N*4);
  cudaMalloc(&dkey,32); cudaMalloc(&dbnd,32); cudaMalloc(&dfound,4);
  cudaMalloc(&dhr,128*4*64); cudaMalloc(&dhc,128*4*64);
  cudaMemcpy(dAp,hA.data(),(size_t)M*K,cudaMemcpyHostToDevice);
  cudaMemcpy(dBp,hB.data(),(size_t)N*K,cudaMemcpyHostToDevice);
  cudaMemset(dkey,0,32); cudaMemset(dbnd,0,32); cudaMemset(dfound,0,4);
  cudaMemset(dCp,0,(size_t)M*N*4);

  Tensor gA_t = make_tensor(make_gmem_ptr(dAp), make_layout(make_shape(M,K), make_stride(K, Int<1>{})));
  Tensor gB_t = make_tensor(make_gmem_ptr(dBp), make_layout(make_shape(N,K), make_stride(K, Int<1>{})));
  auto tma_a = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gA_t, sA(_,_,_0{}), make_shape(bM,bK), Int<1>{});
  auto tma_b = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gB_t, sB(_,_,_0{}), make_shape(bN,bK), Int<1>{});

  int smem = int(sizeof(SharedStorageWGMMA<SmemLayoutA,SmemLayoutB,4>));
  auto kfn = gemm_device_wgmma_ms<decltype(prob),decltype(cta),
      SmemLayoutA,decltype(tma_a),
      SmemLayoutB,decltype(tma_b),
      int32_t,decltype(dC),decltype(sC),decltype(mma),
      /*DumpC=*/true, /*BigEndian=*/false>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grd(1,1), blk(size(mma));
  // reduce_every_k=4 (rank 128), swz off, bound=0 so no hits are reported
  kfn<<<grd,blk,smem>>>(prob,cta, sA,tma_a, sB,tma_b,
      dCp,dC,sC,mma, 4, 0, dkey,dbnd,dfound, dhr,dhc, 64);
  cudaError_t e = cudaDeviceSynchronize();
  if (e!=cudaSuccess){ printf("VERIFY RUNTIME ERROR: %s\n", cudaGetErrorString(e)); return 2; }

  std::vector<int32_t> hC((size_t)M*N);
  cudaMemcpy(hC.data(),dCp,(size_t)M*N*4,cudaMemcpyDeviceToHost);
  int mism=0, first=-1;
  for (size_t i=0;i<hC.size();++i) if (hC[i]!=ref[i]) { if(first<0) first=(int)i; ++mism; }
  if (mism==0) printf("VERIFY OK — C bit-exact vs CPU ref (128x256x512, wgmma 4-stage)\n");
  else printf("VERIFY MISMATCH — %d/%d wrong, first@%d gpu=%d ref=%d\n",
              mism, M*N, first, hC[first], ref[first]);
  return mism==0?0:1;
}

// Coords mode: force EVERY thread to "hit" (bound = all 0xFF) on a single CTA,
// dump hit_rows/hit_cols, normalize per slot, and verify every thread's tile is
// a translated copy of the same rows x cols pattern. The normalized lists are
// exactly what must be declared as rows_pattern / cols_pattern in the Rust
// MiningConfiguration (job_key) — mismatch = share rejection = ban risk.
static int run_coords() {
  const int M=128, N=256, K=512;
  auto prob = make_shape(M, N, K);
  auto bM=Int<128>{}; auto bN=Int<256>{}; auto bK=Int<128>{};
  auto cta = make_shape(bM,bN,bK);
  auto dC = make_stride(N, Int<1>{});
  auto sC = make_layout(make_shape(bM,bN));

  using SmemLayoutAtom = GMMA::Layout_K_SW128_Atom<int8_t>;
  using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<128>, Int<128>, Int<4>>{}));
  using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<256>, Int<128>, Int<4>>{}));
  SmemLayoutA sA; SmemLayoutB sB;
  TiledMMA mma = make_tiled_mma(SM90_64x256x32_S32S8S8_SS_TN{}, Layout<Shape<_2,_1,_1>>{});

  int8_t *dAp,*dBp; int32_t *dCp; uint32_t *dkey,*dbnd; int *dfound,*dhr,*dhc;
  cudaMalloc(&dAp,(size_t)M*K); cudaMalloc(&dBp,(size_t)N*K); cudaMalloc(&dCp,(size_t)M*N*4);
  cudaMalloc(&dkey,32); cudaMalloc(&dbnd,32); cudaMalloc(&dfound,4);
  const int MAXH = 256;
  cudaMalloc(&dhr,(size_t)MAXH*128*4); cudaMalloc(&dhc,(size_t)MAXH*128*4);
  std::vector<int8_t> hA((size_t)M*K), hB((size_t)N*K);
  for (size_t i=0;i<hA.size();++i) hA[i]=(int8_t)(((i*1103515245u+12345u)>>16)&0x7F)-64;
  for (size_t i=0;i<hB.size();++i) hB[i]=(int8_t)(((i*1103515245u+54321u)>>16)&0x7F)-64;
  cudaMemcpy(dAp,hA.data(),(size_t)M*K,cudaMemcpyHostToDevice);
  cudaMemcpy(dBp,hB.data(),(size_t)N*K,cudaMemcpyHostToDevice);
  cudaMemset(dkey,0,32); cudaMemset(dfound,0,4);
  // bound = 0xFF...FF => every transcript hashes below bound => every thread hits
  cudaMemset(dbnd,0xFF,32);

  Tensor gA_t = make_tensor(make_gmem_ptr(dAp), make_layout(make_shape(M,K), make_stride(K, Int<1>{})));
  Tensor gB_t = make_tensor(make_gmem_ptr(dBp), make_layout(make_shape(N,K), make_stride(K, Int<1>{})));
  auto tma_a = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gA_t, sA(_,_,_0{}), make_shape(bM,bK), Int<1>{});
  auto tma_b = make_tma_copy<int8_t>(SM90_TMA_LOAD{}, gB_t, sB(_,_,_0{}), make_shape(bN,bK), Int<1>{});

  int smem = int(sizeof(SharedStorageWGMMA<SmemLayoutA,SmemLayoutB,4>));
  auto kfn = gemm_device_wgmma_ms<decltype(prob),decltype(cta),
      SmemLayoutA,decltype(tma_a),
      SmemLayoutB,decltype(tma_b),
      int32_t,decltype(dC),decltype(sC),decltype(mma),
      /*DumpC=*/false, /*BigEndian=*/false>;
  cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grd(1,1), blk(size(mma));
  kfn<<<grd,blk,smem>>>(prob,cta, sA,tma_a, sB,tma_b,
      dCp,dC,sC,mma, 4, 0, dkey,dbnd,dfound, dhr,dhc, MAXH);
  cudaError_t e = cudaDeviceSynchronize();
  if (e!=cudaSuccess){ printf("COORDS RUNTIME ERROR: %s\n", cudaGetErrorString(e)); return 2; }

  int found=0; cudaMemcpy(&found,dfound,4,cudaMemcpyDeviceToHost);
  printf("found=%d (expect 256 = one hit per thread)\n", found);
  std::vector<int> hr((size_t)found*128), hc((size_t)found*128);
  cudaMemcpy(hr.data(),dhr,(size_t)found*128*4,cudaMemcpyDeviceToHost);
  cudaMemcpy(hc.data(),dhc,(size_t)found*128*4,cudaMemcpyDeviceToHost);

  // Normalize each slot (rows/cols relative to slot minimum) and check all
  // slots agree on the same pattern shape.
  std::set<std::pair<int,int>> ref_rows_cols; // not used for equality of sets, see below
  std::vector<int> ref_rows, ref_cols;
  bool consistent = true;
  for (int s=0; s<found; ++s) {
    std::set<int> rs, cs;
    int rmin=1<<30, cmin=1<<30;
    for (int i=0;i<128;++i){
      int r=hr[(size_t)s*128+i], c=hc[(size_t)s*128+i];
      rs.insert(r); cs.insert(c);
      if(r<rmin)rmin=r; if(c<cmin)cmin=c;
    }
    std::vector<int> nr, nc;
    for(int r: rs) nr.push_back(r-rmin);
    for(int c: cs) nc.push_back(c-cmin);
    // cartesian check: |rows|*|cols| must equal 128
    if ((int)(nr.size()*nc.size()) != 128) { consistent=false; printf("slot %d NOT cartesian: %zu rows x %zu cols\n", s, nr.size(), nc.size()); }
    if (s==0){ ref_rows=nr; ref_cols=nc; }
    else if (nr!=ref_rows || nc!=ref_cols){ consistent=false; printf("slot %d pattern differs from slot 0\n", s); }
  }
  printf("CONSISTENT_ACROSS_THREADS: %s\n", consistent?"YES":"NO");
  printf("rows_pattern (normalized, %zu entries): [", ref_rows.size());
  for (size_t i=0;i<ref_rows.size();++i) printf("%d%s", ref_rows[i], i+1<ref_rows.size()?",":"");
  printf("]\ncols_pattern (normalized, %zu entries): [", ref_cols.size());
  for (size_t i=0;i<ref_cols.size();++i) printf("%d%s", ref_cols[i], i+1<ref_cols.size()?",":"");
  printf("]\n");
  // also show the min-row/min-col spread across slots (= the tile offsets)
  std::set<int> row_offsets, col_offsets;
  for (int s=0; s<found; ++s) {
    int rmin=1<<30, cmin=1<<30;
    for (int i=0;i<128;++i){ int r=hr[(size_t)s*128+i], c=hc[(size_t)s*128+i]; if(r<rmin)rmin=r; if(c<cmin)cmin=c; }
    row_offsets.insert(rmin); col_offsets.insert(cmin);
  }
  printf("distinct slot row offsets: %zu, col offsets: %zu (expect 16 x 4 = 64 slots... found=%d)\n",
         row_offsets.size(), col_offsets.size(), found);
  return consistent?0:1;
}

int main(int argc, char** argv) {
  const char* mode = argc>1 ? argv[1] : "perf";
  if (strcmp(mode,"verify")==0) return run_verify();
  if (strcmp(mode,"coords")==0) return run_coords();

  int M = argc>2?atoi(argv[2]):16384;
  int N = argc>3?atoi(argv[3]):65536;
  int K = argc>4?atoi(argv[4]):8192;
  int iters = argc>5?atoi(argv[5]):30;
  int reduce_every_k = argc>6?atoi(argv[6]):4;
  int swz_g = argc>7?atoi(argv[7]):0;
  int pipe = argc>8?atoi(argv[8]):4;
  printf("=== WGMMA BENCH: M=%d N=%d K=%d iters=%d rek=%d swz_g=%d pipe=%d ===\n",
         M,N,K,iters,reduce_every_k,swz_g,pipe);
  double ths = -1;
  // K_PIPE is a compile-time layout dimension -> explicit dispatch.
  if (pipe==2) ths = run_perf<2>(M,N,K,iters,reduce_every_k,swz_g);
  else if (pipe==3) ths = run_perf<3>(M,N,K,iters,reduce_every_k,swz_g);
  else ths = run_perf<4>(M,N,K,iters,reduce_every_k,swz_g);
  if (ths < 0) return 2;
  printf("RESULT TH/s = %.2f  (mma.sync prod ~345 internal / PeakMiner ~552)\n", ths);
  return 0;
}
