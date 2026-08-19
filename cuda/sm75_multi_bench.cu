// Multi-variant bench for sm75 kernels
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "pearl_gpu_kernel_sm75_wide3.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

// Variant: 3-stage pipeline using dynamic shared memory
namespace aria_sm75_v3stage {
using namespace cute;

static constexpr int kBlockM = 128;
static constexpr int kBlockN = 256;
static constexpr int kChunkK = 32;
static constexpr int kMmaK = 16;
static constexpr int kWarpRows = 4;
static constexpr int kWarpCols = 4;
static constexpr int kTileRowsPerWarp = 2;
static constexpr int kTileColsPerWarp = 4;
static constexpr int kTilesPerWarp = 8;
static constexpr int kWarps = 16;
static constexpr int kThreads = 512;
static constexpr int kTranscript = 16;
static constexpr int kRotate = 13;
static constexpr int kRowStride = 12;
static constexpr int kWordsA = kBlockM * kRowStride;
static constexpr int kWordsB = kBlockN * kRowStride;
static constexpr int kWordsStage = kWordsA + kWordsB;
static constexpr int kStages = 3;
static constexpr int kDynamicSmem = kStages * kWordsStage * 4;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
__device__ __forceinline__ void mma_8x8x16(
    int32_t& d0, int32_t& d1, uint32_t a, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
      "{%0,%1}, {%2}, {%3}, {%0,%1};"
      : "+r"(d0), "+r"(d1) : "r"(a), "r"(b));
}
__device__ __forceinline__ uint32_t rotl13_xor(uint32_t p, uint32_t v) {
  return ((p << kRotate) | (p >> (32 - kRotate))) ^ v;
}
#endif

template <bool BigEndian = false>
__global__ __launch_bounds__(kThreads, 1)
void grind(const int8_t* __restrict__ a, const int8_t* __restrict__ bt,
           int m, int n, int k, int rank,
           const uint32_t* __restrict__ pow_key, const uint32_t* __restrict__ pow_bound,
           int* __restrict__ found_count, int* __restrict__ hit_rows,
           int* __restrict__ hit_cols, int max_hits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  (void)m;
  extern __shared__ __align__(16) uint32_t smem[];
  uint32_t (*stages)[kWordsStage] = reinterpret_cast<uint32_t (*)[kWordsStage]>(smem);
  uint32_t* transcripts = smem + kStages * kWordsStage;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int warp_m = warp / kWarpCols;
  const int warp_n = warp % kWarpCols;

  const int row_base = blockIdx.x * kBlockM;
  const int col_base = blockIdx.y * kBlockN;
  const int warp_row = row_base + warp_m * 32;
  const int warp_col = col_base + warp_n * 64;

  // Init transcripts
  uint32_t* my_trans = transcripts + (warp * kTilesPerWarp + 0) * kTranscript;
  if (lane < kTranscript) {
    #pragma unroll
    for (int t = 0; t < kTilesPerWarp; ++t)
      transcripts[(warp * kTilesPerWarp + t) * kTranscript + lane] = 0;
  }

  int32_t acc[kTilesPerWarp][8];
  #pragma unroll
  for (int t = 0; t < kTilesPerWarp; ++t)
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[t][i] = 0;

  uint4 pf_a, pf_b;
  const bool has_a = tid < (kBlockM * 2);

  auto do_load = [&](int k_off) {
    if (has_a) {
      int row = tid / 2, cv = tid % 2;
      pf_a = __ldcg(reinterpret_cast<const uint4*>(
          a + (size_t)(row_base + row) * k + k_off + cv * 16));
    }
    {
      int cidx = tid / 2, cv = tid % 2;
      pf_b = __ldcg(reinterpret_cast<const uint4*>(
          bt + (size_t)(col_base + cidx) * k + k_off + cv * 16));
    }
  };

  auto do_store = [&](int stage) {
    uint32_t* s = stages[stage];
    if (has_a)
      reinterpret_cast<uint4*>(s)[(tid/2)*3 + (tid%2)] = pf_a;
    reinterpret_cast<uint4*>(s + kWordsA)[(tid/2)*3 + (tid%2)] = pf_b;
  };

  const int chunks = k / kChunkK;
  const int chunks_per_rank = rank / kChunkK;

  // Prologue: fill stages 0 and 1
  do_load(0); do_store(0);
  if (chunks > 1) { do_load(kChunkK); do_store(1); }
  if (chunks > 2) { do_load(2*kChunkK); }
  __syncthreads();

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int cur = chunk % kStages;
    const uint32_t* sa = stages[cur];
    const uint32_t* sb = sa + kWordsA;
    const int a_row_lo = warp_m * 32 + lane_group;
    const int b_col_lo = warp_n * 64 + lane_group;

    #pragma unroll
    for (int sub = 0; sub < kChunkK / kMmaK; ++sub) {
      const int wo = sub * 4 + lane_quad;
      uint32_t av[4], bv[8];
      #pragma unroll
      for (int i = 0; i < 4; ++i) av[i] = sa[(a_row_lo + i*8) * kRowStride + wo];
      #pragma unroll
      for (int j = 0; j < 8; ++j) bv[j] = sb[(b_col_lo + j*8) * kRowStride + wo];
      #pragma unroll
      for (int tm = 0; tm < 2; ++tm)
        #pragma unroll
        for (int tn = 0; tn < 4; ++tn) {
          const int t = tm * 4 + tn;
          mma_8x8x16(acc[t][0], acc[t][1], av[2*tm],   bv[2*tn]);
          mma_8x8x16(acc[t][2], acc[t][3], av[2*tm+1], bv[2*tn]);
          mma_8x8x16(acc[t][4], acc[t][5], av[2*tm],   bv[2*tn+1]);
          mma_8x8x16(acc[t][6], acc[t][7], av[2*tm+1], bv[2*tn+1]);
        }
    }

    if (((chunk + 1) % chunks_per_rank) == 0) {
      #pragma unroll
      for (int t = 0; t < kTilesPerWarp; ++t) {
        uint32_t folded = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) folded ^= (uint32_t)acc[t][i];
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
          folded ^= __shfl_xor_sync(0xffffffffu, folded, mask);
        if (lane == 0) {
          int ti = ((chunk+1)/chunks_per_rank - 1) % kTranscript;
          transcripts[(warp * kTilesPerWarp + t) * kTranscript + ti] =
              rotl13_xor(transcripts[(warp * kTilesPerWarp + t) * kTranscript + ti], folded);
        }
      }
    }

    // Store prefetched data into the stage that was just consumed 2 iterations ago
    if (chunk + 2 < chunks) {
      do_store((chunk + 2) % kStages);
      __syncthreads();
      if (chunk + 3 < chunks) do_load((chunk + 3) * kChunkK);
    } else if (chunk + 1 < chunks) {
      __syncthreads();
    }
  }

  // BLAKE3 tail
  if (lane == 0) {
    #pragma unroll
    for (int tm = 0; tm < 2; ++tm)
      #pragma unroll
      for (int tn = 0; tn < 4; ++tn) {
        const int t = tm * 4 + tn;
        auto message = make_tensor<uint32_t>(Int<16>{});
        auto cv = make_tensor<uint32_t>(Int<8>{});
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          message(i) = transcripts[(warp * kTilesPerWarp + t) * kTranscript + i];
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv(i) = pow_key[i];
        blake3::compress_msg_block_u32(message, cv, blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);
        bool found = true;
        #pragma unroll
        for (int i = 7; i >= 0; --i) {
          if (cv(i) > pow_bound[i]) { found = false; break; }
          if (cv(i) < pow_bound[i]) break;
        }
        if (found) {
          int slot = atomicAdd(found_count, 1);
          if (slot < max_hits) {
            int tr = warp_row + tm * 16, tc = warp_col + tn * 16;
            #pragma unroll
            for (int i = 0; i < 128; ++i) {
              hit_rows[slot*128+i] = tr + (i & 15);
              hit_cols[slot*128+i] = tc + (i & 15);
            }
          }
        }
      }
  }
#else
  (void)a;(void)bt;(void)m;(void)n;(void)k;(void)rank;
  (void)pow_key;(void)pow_bound;(void)found_count;
  (void)hit_rows;(void)hit_cols;(void)max_hits;
#endif
}
}  // namespace

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 16384;
  int N = argc > 2 ? atoi(argv[2]) : 65536;
  int K = argc > 3 ? atoi(argv[3]) : 8192;
  int iters = argc > 4 ? atoi(argv[4]) : 5;
  int rank = argc > 5 ? atoi(argv[5]) : 128;

  printf("multi-bench: m=%d n=%d k=%d iters=%d rank=%d\n", M, N, K, iters, rank);
  if (M % 128 != 0 || N % 256 != 0) { printf("bad dims\n"); return 2; }

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

  // Test wide3 baseline
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
    printf("wide3  (2-stage):  %7.2f ms  %6.2f TH/s  %6.2f display\n", per, ths, ths*1.1);
  }

  // Test 3-stage variant
  {
    dim3 blk(aria_sm75_v3stage::kThreads);
    size_t smem = aria_sm75_v3stage::kDynamicSmem;
    CK(cudaFuncSetAttribute((const void*)aria_sm75_v3stage::grind<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    aria_sm75_v3stage::grind<false><<<grd, blk, smem>>>(dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i=0;i<iters;i++) {
      CK(cudaMemset(dfound,0,4));
      aria_sm75_v3stage::grind<false><<<grd,blk,smem>>>(dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64);
    }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms,t0,t1));
    double per=ms/iters, ths=work/(per/1000.0)/1e12;
    printf("v3stage(3-stage):  %7.2f ms  %6.2f TH/s  %6.2f display\n", per, ths, ths*1.1);
  }

  printf("DONE\n");
  return 0;
}
