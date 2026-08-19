// Comprehensive rasterization + chunk size benchmark
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "pearl_gpu_kernel_sm75_wide3.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA ERROR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

// Variant with swapped pairing (pair along x to share B slabs)
namespace aria_sm75_swap {
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
           int* __restrict__ hit_cols, int max_hits, int mode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  (void)m;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lane_group = lane >> 2;
  const int lane_quad = lane & 3;
  const int warp_m = warp / kWarpCols;
  const int warp_n = warp % kWarpCols;

  int tile_x, tile_y;
  const int linear = blockIdx.y * gridDim.x + blockIdx.x;

  if (mode == 0) {
    // Original: pair along y (share A)
    tile_x = (linear >> 1) % gridDim.x;
    tile_y = (linear / (gridDim.x << 1)) * 2 + (linear & 1);
  } else if (mode == 1) {
    // Swapped: pair along x (share B)
    tile_y = (linear >> 1) % gridDim.y;
    tile_x = (linear / (gridDim.y << 1)) * 2 + (linear & 1);
  } else if (mode == 2) {
    // Group 4x10 (40 blocks per group)
    const int gx = 4, gy = 10;
    const int per_group = gx * gy;
    const int group_id = linear / per_group;
    const int within = linear % per_group;
    const int groups_x = (gridDim.x + gx - 1) / gx;
    tile_x = (group_id % groups_x) * gx + (within % gx);
    tile_y = (group_id / groups_x) * gy + (within / gx);
    if (tile_x >= gridDim.x || tile_y >= gridDim.y) { tile_x = blockIdx.x; tile_y = blockIdx.y; }
  } else if (mode == 3) {
    // Group 2x20 (40 blocks per group)
    const int gx = 2, gy = 20;
    const int per_group = gx * gy;
    const int group_id = linear / per_group;
    const int within = linear % per_group;
    const int groups_x = (gridDim.x + gx - 1) / gx;
    tile_x = (group_id % groups_x) * gx + (within % gx);
    tile_y = (group_id / groups_x) * gy + (within / gx);
    if (tile_x >= gridDim.x || tile_y >= gridDim.y) { tile_x = blockIdx.x; tile_y = blockIdx.y; }
  } else if (mode == 4) {
    // Group 1x40 (40 blocks, all same A, different B)
    const int gx = 1, gy = 40;
    const int per_group = gx * gy;
    const int group_id = linear / per_group;
    const int within = linear % per_group;
    const int groups_x = (gridDim.x + gx - 1) / gx;
    tile_x = (group_id % groups_x) * gx + (within % gx);
    tile_y = (group_id / groups_x) * gy + (within / gx);
    if (tile_x >= gridDim.x || tile_y >= gridDim.y) { tile_x = blockIdx.x; tile_y = blockIdx.y; }
  } else if (mode == 5) {
    // Group 40x1 (40 blocks, all same B, different A)
    const int gx = 40, gy = 1;
    const int per_group = gx * gy;
    const int group_id = linear / per_group;
    const int within = linear % per_group;
    const int groups_x = (gridDim.x + gx - 1) / gx;
    tile_x = (group_id % groups_x) * gx + (within % gx);
    tile_y = (group_id / groups_x) * gy + (within / gx);
    if (tile_x >= gridDim.x || tile_y >= gridDim.y) { tile_x = blockIdx.x; tile_y = blockIdx.y; }
  } else {
    tile_x = blockIdx.x;
    tile_y = blockIdx.y;
  }

  const int row_base = tile_x * kBlockM;
  const int col_base = tile_y * kBlockN;
  const int warp_row = row_base + warp_m * 32;
  const int warp_col = col_base + warp_n * 64;

  __shared__ __align__(16) uint32_t stages[2][kWordsStage];
  __shared__ __align__(16) uint32_t transcripts[kWarps][kTilesPerWarp][kTranscript];

  if (lane < kTranscript) {
    #pragma unroll
    for (int t = 0; t < kTilesPerWarp; ++t) transcripts[warp][t][lane] = 0;
  }

  int32_t acc[kTilesPerWarp][8];
  #pragma unroll
  for (int t = 0; t < kTilesPerWarp; ++t)
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc[t][i] = 0;

  uint4 pf_a, pf_b;
  const bool has_a = tid < (kBlockM * 2);

  auto prefetch = [&](int k_off) {
    if (has_a) {
      int row = tid / 2, cv = tid % 2;
      pf_a = __ldcg(reinterpret_cast<const uint4*>(a + (size_t)(row_base + row) * k + k_off + cv * 16));
    }
    {
      int ci = tid / 2, cv = tid % 2;
      pf_b = __ldcg(reinterpret_cast<const uint4*>(bt + (size_t)(col_base + ci) * k + k_off + cv * 16));
    }
  };
  auto publish = [&](int s) {
    uint32_t* sp = stages[s];
    if (has_a) reinterpret_cast<uint4*>(sp)[(tid/2)*3 + (tid%2)] = pf_a;
    reinterpret_cast<uint4*>(sp + kWordsA)[(tid/2)*3 + (tid%2)] = pf_b;
  };

  const int chunks = k / kChunkK;
  const int chunks_per_rank = rank / kChunkK;
  prefetch(0); publish(0); __syncthreads();

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int cur = chunk & 1;
    const bool have_next = chunk + 1 < chunks;
    if (have_next) prefetch((chunk + 1) * kChunkK);

    const uint32_t* sa = stages[cur];
    const uint32_t* sb = sa + kWordsA;
    const int arl = warp_m * 32 + lane_group;
    const int bcl = warp_n * 64 + lane_group;

    #pragma unroll
    for (int sub = 0; sub < 2; ++sub) {
      const int wo = sub * 4 + lane_quad;
      uint32_t av[4], bv[8];
      #pragma unroll
      for (int i = 0; i < 4; ++i) av[i] = sa[(arl + i*8) * kRowStride + wo];
      #pragma unroll
      for (int j = 0; j < 8; ++j) bv[j] = sb[(bcl + j*8) * kRowStride + wo];
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
          transcripts[warp][t][ti] = rotl13_xor(transcripts[warp][t][ti], folded);
        }
      }
    }
    if (have_next) { publish(1 - cur); __syncthreads(); }
  }

  if (lane == 0) {
    #pragma unroll
    for (int tm = 0; tm < 2; ++tm)
      #pragma unroll
      for (int tn = 0; tn < 4; ++tn) {
        const int t = tm * 4 + tn;
        auto message = make_tensor<uint32_t>(Int<16>{});
        auto cv = make_tensor<uint32_t>(Int<8>{});
        #pragma unroll
        for (int i = 0; i < 16; ++i) message(i) = transcripts[warp][t][i];
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
  (void)hit_rows;(void)hit_cols;(void)max_hits;(void)mode;
#endif
}
}  // namespace aria_sm75_swap

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 16384;
  int N = argc > 2 ? atoi(argv[2]) : 65536;
  int K = argc > 3 ? atoi(argv[3]) : 8192;
  int iters = argc > 4 ? atoi(argv[4]) : 5;
  int rank = argc > 5 ? atoi(argv[5]) : 128;

  printf("raster bench: m=%d n=%d k=%d iters=%d rank=%d\n", M, N, K, iters, rank);

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
  dim3 blk(512);
  double work = (double)M * (double)N * (double)K;

  const char* names[] = {"pair_y(shareA)", "pair_x(shareB)", "grp4x10", "grp2x20", "grp1x40", "grp40x1", "linear"};
  for (int mode = 0; mode <= 6; ++mode) {
    // warmup
    aria_sm75_swap::grind<false><<<grd, blk>>>(dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64,mode);
    CK(cudaDeviceSynchronize());
    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i=0;i<iters;i++) {
      CK(cudaMemset(dfound,0,4));
      aria_sm75_swap::grind<false><<<grd,blk>>>(dA,dB,M,N,K,rank,dkey,dbnd,dfound,dhr,dhc,64,mode);
    }
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms; CK(cudaEventElapsedTime(&ms,t0,t1));
    double per=ms/iters, ths=work/(per/1000.0)/1e12;
    printf("%-16s: %7.2f ms  %6.2f TH/s int  %6.2f disp\n", names[mode], per, ths, ths*1.1);
    CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
  }
  printf("DONE\n");
  return 0;
}
