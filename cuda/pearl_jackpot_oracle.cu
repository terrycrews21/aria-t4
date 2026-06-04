// ariaminer v0.4.0 — oracle bit-exact du jackpot Pearl (étape 2, route B hand-roll).
//
// Valide que le kernel GPU "fused rank-fold" produit EXACTEMENT le même jackpot[16]
// par tuile que la référence CPU (transcription fidèle de
// src/pearl_compute.rs::compute_jackpot_pearl / official_grind.rs::tile_jackpot_flat).
//
// Cette V1 utilise un MAC scalaire int8->int32 (pas encore IMMA) : on valide D'ABORD
// l'ALGORITHME (accumulateur préfixe + XOR-fold + rotate). L'IMMA donnera des entiers
// identiques (int8xint8->int32 exact) → on le greffera ensuite sans changer le résultat.
//
// Spec : h=2, w=64, k=4096, r=128, JACKPOT_SIZE=16, LROT_PER_TILE=13.
// Tuiles contiguës : row-tile rt = lignes [rt*h, rt*h+h), col-tile ct = cols [ct*w, ct*w+w).
//
// Build : nvcc -arch=sm_120a -O3 pearl_jackpot_oracle.cu -o pearl_jackpot_oracle

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define H 2
#define W 64
#define JACK 16
#define LROT 13

__host__ __device__ inline uint32_t rotl32(uint32_t x, uint32_t s) {
    return (x << s) | (x >> (32 - s));
}

// ---- Référence CPU : 1 tuile -> jackpot[16] (= tile_jackpot_flat, mapping contigu) ----
static int DBG = 0;
static void jackpot_ref_tile(const int8_t* a_eff, const int8_t* b_eff,
                             int rt, int ct, int n_cols /*unused*/, int k, int r,
                             uint32_t out[JACK]) {
    (void)n_cols;
    int32_t tile[H * W];
    for (int i = 0; i < H * W; i++) tile[i] = 0;
    uint32_t jack[JACK];
    for (int i = 0; i < JACK; i++) jack[i] = 0;

    for (int ll = r; ll <= k; ll += r) {
        for (int u = 0; u < H; u++) {
            const int8_t* arow = a_eff + (size_t)(rt * H + u) * k;
            for (int v = 0; v < W; v++) {
                const int8_t* brow = b_eff + (size_t)(ct * W + v) * k;
                int32_t acc = 0;
                for (int l = ll - r; l < ll; l++)
                    acc += (int32_t)arow[l] * (int32_t)brow[l];
                tile[u * W + v] += acc;
            }
        }
        uint32_t xored = 0;
        for (int i = 0; i < H * W; i++) xored ^= (uint32_t)tile[i];
        int tid = (ll / r - 1) % JACK;
        jack[tid] = rotl32(jack[tid], LROT) ^ xored;
        if (DBG && ll <= r*4) printf("[DBG ref] ll=%d tid=%d xored=%08x jack[tid]=%08x\n",
                                     ll, tid, xored, jack[tid]);
    }
    for (int i = 0; i < JACK; i++) out[i] = jack[i];
}

// ---- Kernel GPU : 1 thread = 1 tuile, même algorithme (MAC scalaire pour l'instant) ----
__global__ void jackpot_gpu(const int8_t* __restrict__ a_eff,
                            const int8_t* __restrict__ b_eff,
                            int n_row_tiles, int n_col_tiles, int k, int r,
                            uint32_t* __restrict__ out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n_tiles = n_row_tiles * n_col_tiles;
    if (t >= n_tiles) return;
    int rt = t / n_col_tiles;
    int ct = t % n_col_tiles;

    int32_t tile[H * W];
    #pragma unroll
    for (int i = 0; i < H * W; i++) tile[i] = 0;
    uint32_t jack[JACK];
    #pragma unroll
    for (int i = 0; i < JACK; i++) jack[i] = 0;

    for (int ll = r; ll <= k; ll += r) {
        for (int u = 0; u < H; u++) {
            const int8_t* arow = a_eff + (size_t)(rt * H + u) * k;
            for (int v = 0; v < W; v++) {
                const int8_t* brow = b_eff + (size_t)(ct * W + v) * k;
                int32_t acc = 0;
                for (int l = ll - r; l < ll; l++)
                    acc += (int32_t)arow[l] * (int32_t)brow[l];
                tile[u * W + v] += acc;
            }
        }
        uint32_t xored = 0;
        #pragma unroll
        for (int i = 0; i < H * W; i++) xored ^= (uint32_t)tile[i];
        int tid = (ll / r - 1) % JACK;
        jack[tid] = rotl32(jack[tid], LROT) ^ xored;
    }
    uint32_t* o = out + (size_t)t * JACK;
    #pragma unroll
    for (int i = 0; i < JACK; i++) o[i] = jack[i];
}

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

int main(int argc, char** argv) {
    int k = 4096, r = 128;
    int n_row_tiles = (argc > 1) ? atoi(argv[1]) : 256;  // -> m = n_row_tiles*H lignes
    int n_col_tiles = (argc > 2) ? atoi(argv[2]) : 4;    // -> n = n_col_tiles*W cols
    int n_tiles = n_row_tiles * n_col_tiles;
    int m = n_row_tiles * H, n = n_col_tiles * W;

    printf("=== Pearl jackpot oracle — bit-exact GPU vs CPU ===\n");
    printf("k=%d r=%d  tuiles=%d (%d row-tiles x %d col-tiles)  A=%dx%d B=%dx%d\n",
           k, r, n_tiles, n_row_tiles, n_col_tiles, m, k, n, k);

    // Inputs int7-range effectifs [-64,63] (peu importe la distribution pour l'oracle).
    size_t na = (size_t)m * k, nb = (size_t)n * k;
    std::vector<int8_t> a(na), b(nb);
    // splitmix64 indexé par position (avalanche complète, pas de périodicité bas-bits
    // comme un LCG → lignes indépendantes, XOR-fold non dégénéré).
    auto sm64 = [](uint64_t x){
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    };
    for (size_t i = 0; i < na; i++) a[i] = (int8_t)((int)(sm64((i<<1) | 1ULL) & 0x7F) - 64);
    for (size_t i = 0; i < nb; i++) b[i] = (int8_t)((int)(sm64((i<<1) ^ 0xABCDEF12345ULL) & 0x7F) - 64);

    // DIAGNOSTIC inputs (détecter buffers nuls / dégénérés)
    { long suma = 0, sumb = 0; int nza = 0;
      for (size_t i = 0; i < na; i++) { suma += a[i]; if (a[i]) nza++; }
      for (size_t i = 0; i < nb; i++) sumb += b[i];
      printf("[DIAG] a[0..7]= %d %d %d %d %d %d %d %d  | non-zero A=%d/%zu  Σa=%ld Σb=%ld\n",
             a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7], nza, na, suma, sumb);
      // dot brut ligne0(A)·ligne0(B) sur les r=128 premiers
      long d = 0; for (int l = 0; l < r; l++) d += (int)a[l]*(int)b[l];
      printf("[DIAG] dot(A0,B0)[0..%d) = %ld\n", r, d);
    }

    // GPU
    int8_t *da, *db; uint32_t *dout;
    CK(cudaMalloc(&da, na)); CK(cudaMalloc(&db, nb));
    CK(cudaMalloc(&dout, (size_t)n_tiles * JACK * 4));
    CK(cudaMemcpy(da, a.data(), na, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(), nb, cudaMemcpyHostToDevice));

    int threads = 128, blocks = (n_tiles + threads - 1) / threads;
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    jackpot_gpu<<<blocks, threads>>>(da, db, n_row_tiles, n_col_tiles, k, r, dout);
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    CK(cudaGetLastError());
    float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));

    std::vector<uint32_t> gout((size_t)n_tiles * JACK);
    CK(cudaMemcpy(gout.data(), dout, (size_t)n_tiles * JACK * 4, cudaMemcpyDeviceToHost));

    // CPU ref + comparaison
    printf("\n[CPU] calcul de la référence sur %d tuiles...\n", n_tiles);
    int mism = 0, first_bad = -1;
    for (int rt = 0; rt < n_row_tiles; rt++)
      for (int ct = 0; ct < n_col_tiles; ct++) {
        int t = rt * n_col_tiles + ct;
        uint32_t ref[JACK];
        jackpot_ref_tile(a.data(), b.data(), rt, ct, n, k, r, ref);
        for (int i = 0; i < JACK; i++)
            if (ref[i] != gout[(size_t)t * JACK + i]) {
                mism++; if (first_bad < 0) first_bad = t;
            }
      }

    DBG = 1;
    { uint32_t ref[JACK]; jackpot_ref_tile(a.data(), b.data(), 0, 0, n, k, r, ref); }
    DBG = 0;
    printf("\n[ÉCHANTILLON] tuile 0 jackpot[16] :\n  GPU:");
    for (int i = 0; i < JACK; i++) printf(" %08x", gout[i]);
    printf("\n  CPU:");
    { uint32_t ref[JACK]; jackpot_ref_tile(a.data(), b.data(), 0, 0, n, k, r, ref);
      for (int i = 0; i < JACK; i++) printf(" %08x", ref[i]); }
    printf("\n");

    double macs = (double)n_tiles * H * W * (double)k;
    printf("\n[PERF] kernel %.3f ms  (%.1f G MAC, %.1f GMAC/s — V1 scalaire, PAS encore IMMA)\n",
           ms, macs/1e9, macs/1e6/ms);

    if (mism == 0) {
        printf("\n✅ BIT-EXACT : GPU == CPU sur %d tuiles (%d mots). Algorithme jackpot porté correctement.\n",
               n_tiles, n_tiles * JACK);
        return 0;
    } else {
        printf("\n❌ DIVERGENCE : %d mots faux (1ère tuile fausse = %d). Algorithme à corriger.\n",
               mism, first_bad);
        return 1;
    }
}
