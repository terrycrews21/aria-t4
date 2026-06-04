// ariaminer v0.4.0 — micro-test IMMA m16n8k32.s8 : valide le layout de fragments.
// Un warp calcule D[16x8] = A[16x32] · Bᵀ[8x32] (int8->int32) via mma.sync, comparé
// au scalaire. Prérequis avant de greffer l'IMMA dans le kernel jackpot fusé.
// Build : nvcc -arch=sm_120a -O3 imma_microtest.cu -o imma_microtest

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// A : 16x32 row-major (s8).  B : 8x32 row-major (= Bᵀ, n=8 lignes de k=32).
// D[i][j] = Σ_{l<32} A[i][l]*B[j][l].
__global__ void imma_kernel(const int8_t* A, const int8_t* B, int32_t* D) {
    int lane = threadIdx.x & 31;
    int g = lane >> 2;        // groupID 0..7
    int t = lane & 3;         // thread-in-group 0..3

    // Charge fragment A (a0..a3) : 4x .b32, chacun 4 int8.
    auto ld4 = [](const int8_t* p)->uint32_t {
        uint32_t v; v  = (uint32_t)(uint8_t)p[0];
        v |= (uint32_t)(uint8_t)p[1] << 8;
        v |= (uint32_t)(uint8_t)p[2] << 16;
        v |= (uint32_t)(uint8_t)p[3] << 24; return v;
    };
    uint32_t a0 = ld4(&A[(g)   *32 + t*4 + 0 ]);
    uint32_t a1 = ld4(&A[(g+8) *32 + t*4 + 0 ]);
    uint32_t a2 = ld4(&A[(g)   *32 + t*4 + 16]);
    uint32_t a3 = ld4(&A[(g+8) *32 + t*4 + 16]);
    uint32_t b0 = ld4(&B[(g)   *32 + t*4 + 0 ]);
    uint32_t b1 = ld4(&B[(g)   *32 + t*4 + 16]);

    int32_t d0=0, d1=0, d2=0, d3=0;
    asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
      : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
        "r"(d0),"r"(d1),"r"(d2),"r"(d3));

    // Store : d0->D[g][2t], d1->D[g][2t+1], d2->D[g+8][2t], d3->D[g+8][2t+1]
    D[(g)  *8 + 2*t + 0] = d0;
    D[(g)  *8 + 2*t + 1] = d1;
    D[(g+8)*8 + 2*t + 0] = d2;
    D[(g+8)*8 + 2*t + 1] = d3;
}

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s\n",cudaGetErrorString(e));return 1;}}while(0)

int main() {
    int8_t A[16*32], B[8*32];
    auto sm = [](uint64_t x){ x+=0x9E3779B97F4A7C15ULL; x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;
                              x=(x^(x>>27))*0x94D049BB133111EBULL; return x^(x>>31); };
    for (int i=0;i<16*32;i++) A[i]=(int8_t)((int)(sm((i<<1)|1)&0x7F)-64);
    for (int i=0;i<8*32;i++)  B[i]=(int8_t)((int)(sm((i<<1)^0xBEEF)&0x7F)-64);

    int32_t ref[16*8];
    for (int i=0;i<16;i++) for (int j=0;j<8;j++){ int32_t s=0;
        for (int l=0;l<32;l++) s += (int32_t)A[i*32+l]*(int32_t)B[j*32+l];
        ref[i*8+j]=s; }

    int8_t *dA,*dB; int32_t *dD;
    CK(cudaMalloc(&dA,sizeof A)); CK(cudaMalloc(&dB,sizeof B)); CK(cudaMalloc(&dD,sizeof ref));
    CK(cudaMemcpy(dA,A,sizeof A,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,B,sizeof B,cudaMemcpyHostToDevice));
    imma_kernel<<<1,32>>>(dA,dB,dD);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    int32_t got[16*8]; CK(cudaMemcpy(got,dD,sizeof ref,cudaMemcpyDeviceToHost));

    int bad=0, first=-1;
    for (int i=0;i<16*8;i++) if (got[i]!=ref[i]){ bad++; if(first<0)first=i; }
    printf("=== IMMA m16n8k32.s8 micro-test ===\n");
    printf("D[0][0..3] ref= %d %d %d %d  imma= %d %d %d %d\n",
           ref[0],ref[1],ref[2],ref[3], got[0],got[1],got[2],got[3]);
    printf("D[15][4..7] ref= %d %d %d %d  imma= %d %d %d %d\n",
           ref[15*8+4],ref[15*8+5],ref[15*8+6],ref[15*8+7],
           got[15*8+4],got[15*8+5],got[15*8+6],got[15*8+7]);
    if(bad==0){ printf("\n✅ IMMA == scalaire sur les 128 éléments. Layout de fragments MAÎTRISÉ.\n"); return 0; }
    printf("\n❌ %d/128 faux (1er = index %d). Layout à corriger.\n", bad, first); return 1;
}
