// ariaminer v0.4.0 — kernel jackpot FUSÉ avec IMMA m16n8k32.s8 (étape 2, perf).
// Un warp = 8 row-tiles × 1 col-tile = super-tuile 16 a-rows × 64 b-cols.
//   - accumulateur IMMA en registres (8 n-tiles × 4 regs) PERSISTE sur les 32 blocs r=128 (prefix sum)
//   - à chaque frontière r=128 : écrit l'accu en smem[16][64], fold les 8 tuiles (XOR 2×64 -> rotl13)
// Validé bit-exact vs la référence scalaire (= tile_jackpot_flat).
// Build : nvcc -arch=sm_120a -O3 jackpot_imma.cu -o jackpot_imma

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define H 2
#define W 64
#define JACK 16
#define LROT 13
#define NT 8            // n-tiles par super-tuile (64/8)
#define TILES_PER_WARP 8  // row-tiles par warp (16 rows / 2)

__host__ __device__ inline uint32_t rotl32(uint32_t x, uint32_t s){ return (x<<s)|(x>>(32-s)); }

// ---------- Référence CPU scalaire (= tile_jackpot_flat, mapping contigu) ----------
static void ref_tile(const int8_t* a,const int8_t* b,int rt,int ct,int k,int r,uint32_t out[JACK]){
    int32_t tile[H*W]; for(int i=0;i<H*W;i++) tile[i]=0;
    uint32_t jk[JACK]; for(int i=0;i<JACK;i++) jk[i]=0;
    for(int ll=r; ll<=k; ll+=r){
        for(int u=0;u<H;u++){ const int8_t* ar=a+(size_t)(rt*H+u)*k;
          for(int v=0;v<W;v++){ const int8_t* br=b+(size_t)(ct*W+v)*k;
            int32_t acc=0; for(int l=ll-r;l<ll;l++) acc+=(int32_t)ar[l]*(int32_t)br[l];
            tile[u*W+v]+=acc; } }
        uint32_t x=0; for(int i=0;i<H*W;i++) x^=(uint32_t)tile[i];
        jk[(ll/r-1)%JACK]=rotl32(jk[(ll/r-1)%JACK],LROT)^x;
    }
    for(int i=0;i<JACK;i++) out[i]=jk[i];
}

// ---------- Kernel IMMA fusé ----------
__device__ inline uint32_t ld4(const int8_t* p){
    return (uint32_t)(uint8_t)p[0]|((uint32_t)(uint8_t)p[1]<<8)|
           ((uint32_t)(uint8_t)p[2]<<16)|((uint32_t)(uint8_t)p[3]<<24);
}
__global__ void jackpot_imma(const int8_t* __restrict__ A,const int8_t* __restrict__ B,
                             int n_row_tiles,int n_col_tiles,int k,int r,uint32_t* __restrict__ out){
    int warp = blockIdx.x*(blockDim.x>>5) + (threadIdx.x>>5);
    int n_super = (n_row_tiles/TILES_PER_WARP)*n_col_tiles;
    if (warp>=n_super) return;
    int rtbase = (warp / n_col_tiles) * TILES_PER_WARP;   // 1er row-tile
    int ct     =  warp % n_col_tiles;                      // col-tile
    int lane = threadIdx.x&31, g=lane>>2, t=lane&3;

    const int8_t* Abase = A + (size_t)(rtbase*H)*k;        // 16 a-rows contigües
    const int8_t* Bbase = B + (size_t)(ct*W)*k;            // 64 b-cols contigües

    int32_t acc[NT][4]; for(int n=0;n<NT;n++) acc[n][0]=acc[n][1]=acc[n][2]=acc[n][3]=0;

    __shared__ uint32_t rx[16];                            // 16 row-XOR (fold via shuffle)
    uint32_t jk[JACK];                                     // SEULES les lanes 0..7 l'utilisent (tuile p=lane)
    for(int i=0;i<JACK;i++) jk[i]=0;

    int blocks_r = k/r, ksteps = r/32;                     // r=128 -> 4 k32-steps
    for(int blk=0; blk<blocks_r; blk++){
        for(int ks=0; ks<ksteps; ks++){
            int ko = (blk*ksteps+ks)*32;                   // offset colonne k
            uint32_t a0=ld4(&Abase[(size_t)(g)  *k+ko+t*4]),
                     a1=ld4(&Abase[(size_t)(g+8)*k+ko+t*4]),
                     a2=ld4(&Abase[(size_t)(g)  *k+ko+t*4+16]),
                     a3=ld4(&Abase[(size_t)(g+8)*k+ko+t*4+16]);
            for(int n=0;n<NT;n++){
                const int8_t* Bn = Bbase + (size_t)(n*8 + g)*k + ko;
                uint32_t b0=ld4(&Bn[t*4]), b1=ld4(&Bn[t*4+16]);
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                  "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                  :"+r"(acc[n][0]),"+r"(acc[n][1]),"+r"(acc[n][2]),"+r"(acc[n][3])
                  :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
        // frontière r : fold XOR par SHUFFLE (pas de round-trip shared).
        // 1) chaque thread XOR ses propres valeurs par moitié-ligne (row g et row g+8).
        uint32_t pg=0, pg8=0;
        for(int n=0;n<NT;n++){ pg ^= (uint32_t)acc[n][0]^(uint32_t)acc[n][1];
                               pg8 ^= (uint32_t)acc[n][2]^(uint32_t)acc[n][3]; }
        // 2) réduction XOR dans le groupe de 4 threads (t=0..3) → row-XOR complet (64 cols).
        pg  ^= __shfl_xor_sync(0xffffffffu, pg, 1);  pg  ^= __shfl_xor_sync(0xffffffffu, pg, 2);
        pg8 ^= __shfl_xor_sync(0xffffffffu, pg8,1);  pg8 ^= __shfl_xor_sync(0xffffffffu, pg8,2);
        // 3) un seul thread/groupe publie les 16 row-XOR ; combine pairwise par tuile.
        if(t==0){ rx[g]=pg; rx[g+8]=pg8; }
        __syncwarp();
        if(lane<TILES_PER_WARP){
            uint32_t x = rx[2*lane]^rx[2*lane+1];
            jk[blk%JACK]=rotl32(jk[blk%JACK],LROT)^x;
        }
        __syncwarp();
    }
    if(lane<TILES_PER_WARP){
        int gt=(rtbase+lane)*n_col_tiles+ct;               // index tuile global (= rt*n_col_tiles+ct)
        for(int i=0;i<JACK;i++) out[(size_t)gt*JACK+i]=jk[i];
    }
}

#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

int main(int argc,char**argv){
    int k=4096,r=128;
    int n_row_tiles=(argc>1)?atoi(argv[1]):256;   // multiple de 8
    int n_col_tiles=(argc>2)?atoi(argv[2]):4;
    if(n_row_tiles%TILES_PER_WARP){ printf("n_row_tiles doit être multiple de 8\n"); return 1; }
    int n_tiles=n_row_tiles*n_col_tiles, m=n_row_tiles*H, n=n_col_tiles*W;
    printf("=== jackpot IMMA fusé vs scalaire ===\n");
    printf("k=%d r=%d tuiles=%d (%dx%d)  A=%dx%d B=%dx%d\n",k,r,n_tiles,n_row_tiles,n_col_tiles,m,k,n,k);

    size_t na=(size_t)m*k, nb=(size_t)n*k;
    std::vector<int8_t> a(na),b(nb);
    auto sm=[](uint64_t x){x+=0x9E3779B97F4A7C15ULL;x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;
                           x=(x^(x>>27))*0x94D049BB133111EBULL;return x^(x>>31);};
    for(size_t i=0;i<na;i++) a[i]=(int8_t)((int)(sm((i<<1)|1ULL)&0x7F)-64);
    for(size_t i=0;i<nb;i++) b[i]=(int8_t)((int)(sm((i<<1)^0xABCDEF12345ULL)&0x7F)-64);

    int8_t *da,*db; uint32_t *dout;
    CK(cudaMalloc(&da,na)); CK(cudaMalloc(&db,nb)); CK(cudaMalloc(&dout,(size_t)n_tiles*JACK*4));
    CK(cudaMemcpy(da,a.data(),na,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,b.data(),nb,cudaMemcpyHostToDevice));

    int n_super=(n_row_tiles/TILES_PER_WARP)*n_col_tiles;
    int warps_per_block=4, threads=warps_per_block*32;
    int blocks=(n_super+warps_per_block-1)/warps_per_block;
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // NB: smem dimensionné pour 1 warp → 1 warp/block pour rester correct
    warps_per_block=1; threads=32; blocks=n_super;
    CK(cudaEventRecord(e0));
    jackpot_imma<<<blocks,threads>>>(da,db,n_row_tiles,n_col_tiles,k,r,dout);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));

    std::vector<uint32_t> got((size_t)n_tiles*JACK);
    CK(cudaMemcpy(got.data(),dout,(size_t)n_tiles*JACK*4,cudaMemcpyDeviceToHost));

    int bad=0,first=-1;
    for(int rt=0;rt<n_row_tiles;rt++) for(int ct=0;ct<n_col_tiles;ct++){
        int tt=rt*n_col_tiles+ct; uint32_t ref[JACK]; ref_tile(a.data(),b.data(),rt,ct,k,r,ref);
        for(int i=0;i<JACK;i++) if(ref[i]!=got[(size_t)tt*JACK+i]){bad++; if(first<0)first=tt;}
    }
    printf("\n[tuile 0] imma:"); for(int i=0;i<JACK;i++) printf(" %08x",got[i]);
    { uint32_t ref[JACK]; ref_tile(a.data(),b.data(),0,0,k,r,ref);
      printf("\n[tuile 0]  cpu:"); for(int i=0;i<JACK;i++) printf(" %08x",ref[i]); }
    double macs=(double)n_tiles*H*W*(double)k;
    printf("\n[PERF] %.3f ms  %.1f G MAC  %.1f TOPS (IMMA, 1 warp/block non optimisé)\n",
           ms, macs/1e9, 2.0*macs/1e9/ms);
    if(bad==0){ printf("\n✅ BIT-EXACT IMMA == scalaire sur %d tuiles. Kernel fusé GPU validé.\n",n_tiles); return 0; }
    printf("\n❌ %d mots faux (1ère tuile %d).\n",bad,first); return 1;
}
