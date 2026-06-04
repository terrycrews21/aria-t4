# ariaminer v0.4.0 — GPU grind (Pearl), design & contrat

> But : greffer un chemin GPU dans ariaminer (le miner CPU qui mine déjà sur AriaPool),
> SANS toucher au proof/stratum/submit prouvés. v0.3.0 reste intact (tag immuable).
> Cible réaliste = ~170-175 TH/s/carte (= alpha-miner 174.78, même instruction IMMA),
> valeur = 0% fee + open-source + Blackwell-natif. RIEN à voir avec EXFER.

## 1. Ce qu'on réutilise TEL QUEL (côté CPU, déjà prouvé)

Dans `src/official_grind.rs` :
- `make_proof(a_sig, b_sig, m, n, k, rank, job_key, a_rows, b_cols) -> PlainProof`
  — reconstruit la preuve canonique sur un HIT (rare). Oracle-guarded, passe la vérif officielle.
- `mine_share()` — boucle d'orchestration.
Dans `src/official_proof.rs` / `src/stratum.rs` / `src/stratum_to_official.rs` :
- `encode_base64(&proof)` + client stratum → submit `pool.ariabrain.com:3334` (ACCEPTÉ en prod).

Le GPU ne touche À RIEN de ça. Il remplace UNIQUEMENT le calcul chaud
(`try_mine_one_bounded`), et sur un hit renvoie de quoi appeler `make_proof` inchangé.

## 2. Le calcul chaud à porter sur GPU (réf = `src/pearl_compute.rs::compute_jackpot_pearl`)

Constantes Pearl mainnet : `h=2`, `w=64`, `k=4096`, `r=rank=128`, `JACKPOT_SIZE=16`, `LROT_PER_TILE=13`.

Par ATTEMPT (= un setup d'une grille m×n, partitionné en (m/h)×(n/w) tuiles indépendantes) :
1. Tirer signal int7 : `a_sig[m*k]`, `b_sig[n*k]` dans `[-64, 63]` (= `(rng_byte & 0x7F) - 64`). A/B = choix LIBRE du mineur.
2. `commitment_hash(job_key, a_sig, b_sig)` (blake3) → `(b_noise_seed, a_noise_seed)`.
3. Noise additif in-place → `a_noised`, `b_noised_t` (wrapping i8 add).
4. Par TUILE (h=2 lignes × w=64 cols) — c'est ici le cœur :
   ```
   jackpot_tile[h*w] = 0           // accumulateur i32, PERSISTE sur les blocs
   jackpot_msg[16]   = 0
   for ll in (r..=k).step_by(r):           // 32 frontières de rang
       for u in 0..h: for v in 0..w:
           jackpot_tile[u][v] += dot_i8(a_noised[u, ll-r..ll], b_noised_t[v, ll-r..ll])  // IMMA
       xored = XOR-fold de jackpot_tile[..h*w]      // tuile -> 1 u32
       tid   = (ll/r - 1) % 16
       jackpot_msg[tid] = jackpot_msg[tid].rotate_left(13) ^ xored
   ```
   ⚠️ POINT NON-TRIVIAL : `jackpot_tile` est un accumulateur PRÉFIXE (somme cumulée
   sur tous les blocs). Le XOR-fold à la frontière ll utilise la somme partielle
   `Σ_{l<ll} a·b`, PAS le bloc seul. ⇒ un `cutlass::device::Gemm` standard (somme
   finale unique) NE SUFFIT PAS. Il faut un kernel FUSÉ qui snapshot l'accumulateur
   à chaque frontière r=128.
5. `compute_jackpot_hash(jackpot_msg, a_noise_seed)` (blake3 keyed) → candidate hash 32o.
6. `le_leq(hash, bound_le)` ? → HIT : renvoyer (attempt_seed, a_rows, b_cols) du tile gagnant.

Arithmétique : `(secret + noise)` reste en i8 range ; MAC en i32 wrapping. Tout doit
être bit-exact vs `compute_jackpot_pearl` (lui-même validé vs `zk_pow::...::compute_jackpot`).

## 3. Le moteur déjà prouvé

`pearl-gemm/bench_int8_tile.cu` = `cutlass::device::Gemm` avec `GemmShape<16,8,32>` int8
= instruction `IMMA.16832.S8.S8`, compilé `sm_120a`, mesuré **371 TOPS sur RTX 5080**.
C'est EXACTEMENT l'instruction qu'alpha-miner utilise sur sm_120 (vérifié au désassemblage,
zéro tcgen05). ⇒ moteur matmul validé ; reste à l'envelopper de l'épilogue rank-fold §2.4.

## 4. Kernel v0.4.0 = IMMA + épilogue rank-fold FUSÉ

Deux routes pour le fusé :
- **A. Custom mainloop CUTLASS** : réutiliser le tensor-op IMMA m16n8k32 mais avec un
  épilogue par bloc K (snapshot tous les r=128 → XOR-fold → rotate). Garde ~371 TOPS.
- **B. Kernel hand-roll par batch de tuiles** : h=2×w=64 est minuscule, le parallélisme
  vient des millions de tuiles. Plus simple à rendre bit-exact, perf à mesurer.
→ Décision après un premier proto + bench. Le hash final réutilise `inner_hash_kernel.cu`
  officiel (`launch_inner_hash_kernel`, PORTABLE) ou un blake3 maison.

## 5. Plan de validation (LINCHPIN — mesurer, pas supposer)

1. **Oracle bit-exact** : ABI `pearl_gpu_jackpot_oracle(a_eff, b_eff, m,n,k,rank) -> jackpot[16]/tuile`.
   Comparer GPU vs `compute_jackpot_pearl` sur des inputs random. Doit être IDENTIQUE.
2. **End-to-end** : même `(job_key, config, attempt_seed)` → GPU et path CPU sortent la MÊME PlainProof.
3. **Live** : brancher stratum, run sur AriaPool, vérifier shares acceptées.

## 6. ABI C proposée (Rust FFI `src/gpu_ffi.rs` → `extern "C"`)

```c
// Contexte GPU persistant (buffers réutilisés, pas d'alloc par attempt).
typedef struct PearlGpuCtx PearlGpuCtx;
PearlGpuCtx* pearl_gpu_create(uint32_t m, uint32_t n, uint32_t k, uint32_t rank);
void         pearl_gpu_destroy(PearlGpuCtx*);

// (1) Oracle de validation : inputs EXPLICITES (déjà noised), sort jackpot[16] par tuile.
//     Sert UNIQUEMENT au cross-check bit-exact vs compute_jackpot_pearl.
void pearl_gpu_jackpot_oracle(PearlGpuCtx*,
        const int8_t* a_eff /*m*k*/, const int8_t* b_eff /*n*k*/,
        uint32_t* out_jackpot /*(m/h)*(n/w)*16*/);

// (2) Grind complet d'un setup : gen signal int7 depuis seed -> noise -> fused -> hash -> bound.
//     Retour : 1 si hit. Sur hit, remplit *hit (coords tuile + seed pour regen a_sig/b_sig CPU).
typedef struct { uint64_t attempt_seed; uint32_t row_tile; uint32_t col_tile; } PearlHit;
int pearl_gpu_grind_setup(PearlGpuCtx*,
        const uint8_t job_key[32], const uint8_t bound_le[32],
        uint64_t base_seed, uint32_t num_attempts, PearlHit* hit /*out*/);
```

Côté Rust : `try_mine_one_bounded_gpu()` appelle (2) ; sur hit, regen `a_sig`/`b_sig`
depuis `attempt_seed` (PRNG identique CPU/GPU) et appelle `make_proof(...)` INCHANGÉ.

## 7. Infra build (à transplanter de `/mnt/aria/aria-pearl-miner/`)
- `build.rs` : nvcc `-arch=sm_120a -O3` → `libpearl_gpu.a` linkée + cudart.
- `src/gpu_ffi.rs` : déclarations extern "C" + wrappers sûrs.
- `src/official_grind.rs` : ajouter `try_mine_one_bounded_gpu` à côté du CPU (feature `gpu`).
