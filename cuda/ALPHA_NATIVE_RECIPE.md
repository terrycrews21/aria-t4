# Recette `pearl_blackwell_native` — blueprint des 174 TH/s (RTX 5080, sm_120)

Source : `~/alpha-miner-1.7.6-beta/alpha`, cubin `alpha.40.sm_120.cubin`
(extrait via **cuobjdump 12.8** — le 12.0 du PATH plante `fatal: sm_2`).
Désasm SASS = nvdisasm Triton 12.8.55 → `/tmp/dis_alpha/alpha40.sass`.
Kernel = `pearl_blackwell_native_impl::headless_mine_kernel<bool,bool,...>`.

## Signature template démanglée (la recette exacte)

- **ProblemShape** = `tuple<int,int,int>`
- **CtaTiler** = `tuple<C<128>, C<256>, C<128>>`  → **bM=128, bN=256, bK=128**
- **MMA** = `TiledMMA< MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
                       Layout<Shape<2,4,1>,Stride<1,2,0>>,   // 8 warps = 256 threads
                       Tile<32,32,32> >`
  - atome IMMA = **identique au nôtre** (`SM80_16x8x32_S8`). Seul change : 2×4 (8 warps) au lieu de 2×2 (4 warps).
- **Swizzle smem** = `Swizzle<3,4,3>`  (le nôtre = `<2,4,3>`)
  - SmemLayout A : ComposedLayout cosize **16384** (128×128)
  - SmemLayout B : ComposedLayout cosize **32768** (256×128)

### Chargement MIXTE (la découverte clé — pas tout-TMA)
- **A (128×128)** : g→s = `SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>`  (cp.async)
                    s→r = `SM75_U32x4_LDSM_N`                     (ldmatrix ×4)
- **B (256×128, le gros)** : g→s = `SM90_TMA_LOAD` box `C<131072>` + `AuxTmaParams`  (**TMA**)
                    s→r = `SM75_U32x2_LDSM_N`                     (ldmatrix ×2)
- TiledCopy cp.async (A) = `Layout<((8,32),16),((512,1),32)>`, tiler `Shape<32,128>`.

### Persistance / host-signal
- Params kernel : `... HostSignalHeader*, HostSignalSync*, PearlHeadlessJackpotDumpEntry*, int*, uint32_t* ...`
- Kernel **résident**, boucle interne, dump des hits via `PearlHeadlessJackpotDumpEntry*`.

## Confirmation SASS (cub40)
| marqueur | n | sens |
|---|---|---|
| `UTMALDG.2D` | 12 | TMA charge B (`@P0 ELECT P1` → un thread élu issue le load, desc[UR]) |
| `LDGSTS` | 24 | cp.async charge A |
| `ELECT` | 15 | thread leader (issue TMA) |
| `NANOSLEEP.SYNCS 0x989680` | 6 | spin-wait host-signal (=10 ms backoff, `@!P0/@!P1`) |
| `LDSM` / `IMMA` | 224 / 512 | mainloop consommateur |
| `USETMAXREG` | **0** | **PAS de setmaxnreg** (alpha s'en passe) |
| `MEMBAR` / `BAR.SYNC` | 15 / 11 | handshake mbarrier |
- REG kernel = 208-248 selon toggles bool, **STACK 0 / LOCAL 0 = 0 spill**.

## ⇒ Ce n'est PAS un producteur/consommateur warpgroup
Les 8 warps font TOUS du MMA (TiledMMA 2×4). Le "warp-spec" = TMA async (issue par
1 thread élu) + cp.async en fond pendant que les 8 warps grindent, kernel persistant
(NANOSLEEP poll). = GEMM pipeline TMA multi-stage persistant → réf CUTLASS
`collective/sm90_mma_tma_*` (adapter GMMA→IMMA `SM80_16x8x32_S8`, garder TMA).

## DELTA vs ce qu'on a (d9d81b1 = 151, bn256 = 153) → les 20 TH/s manquants
On a déjà : GEMM CuTe IMMA, fold jackpot, blake3, pow-check, prologue GPU, make_proof, stratum.
3 greffes à faire, chacune `proof_check` bit-exact AVANT pool :
1. **TMA sur B** (cp.async A inchangé) : descripteur TMA host (`cute::make_tma_copy`),
   atome `SM90_TMA_LOAD` dans la mainloop, Swizzle<3,4,3>. = morceau technique neuf.
2. **Persistance host-signal** : kernel résident + boucle interne + NANOSLEEP poll pinned mem
   → tue le relaunch/attempt (la "béquille" CPU). Dump hits via buffer.
3. **Tuile 128×256 + 8 warps** : déjà fait (bn256, commit 4f36737). Coords à re-dériver
   pour atom-layout 2×4 (validation exhaustive façon JALON 9) → nouvelle `canonical_gpu_config`.

NB : bn256 SEUL = 153 (= fallback flat cub39 d'alpha, SANS TMA/persistance). Les 174 viennent
de TMA(B) + persistance, PAS de la tuile. C'est pour ça que bn256 n'a rien donné.
Clusters (CGA) = levier marginal, en dernier.
